"""Phase-0 keystone spike — does the teacher-forced new-token forward RUN?

Run THIS on the GPU box before building any of Phase 2. It answers the one
question the whole localization-half design rests on and that cannot be checked
without the real Sa2VA checkpoint + transformers:

    Can we append `[SEG_DIR]/[SEG_IND]/[EDIT]×N` as held nn.Parameter embedding
    vectors (vocab UNTOUCHED), run a plain `inputs_embeds` forward
    (output_hidden_states=True, no generate()), recover hidden at the appended
    positions, and drive SAM2 to a (T,H,W) mask?

If yes → the overlay/teacher_forced design is sound; build Phase 2.
If the forward raises a shape / M-RoPE position error → that is the landmine; the
fallback below (arange position continuation) is the first thing to try, and the
exact `get_rope_index` / vision-merge incantation gets pinned here, once, cheaply
— NOT discovered at integration.

This script mutates nothing on disk and never touches the vocab, so the vanilla
`[SEG]` path stays byte-identical regardless of outcome.

Usage (GPU box):
    cd e2w
    python -m e2w_localization.spike_teacher_forced \
        --weights-config configs/weights.v0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2w_localization.planner import CausalPlanner, PlannerConfig


def _sa2va_path(weights_config: Path) -> str:
    cfg = json.loads(Path(weights_config).read_text())
    return cfg["models"]["sa2va_qwen2_5_vl_7b"]["path"]


def _synthetic_image(size: int = 448):
    """A throwaway RGB frame — the spike only asks 'does it run', not 'is it right'."""
    import numpy as np
    from PIL import Image

    grad = np.linspace(0, 255, size, dtype=np.uint8)
    arr = np.stack([np.tile(grad, (size, 1)), np.tile(grad[:, None], (1, size)),
                    np.full((size, size), 128, dtype=np.uint8)], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _build_mm_inputs(model, processor, image, text):
    """Mirror predict_forward's message → processor path (modeling_sa2va_qwen.py:180-207)."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": text}]}]
    processed_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    mm_inputs = processor(
        text=[processed_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
        min_pixels=model.min_pixels, max_pixels=model.max_pixels)
    return mm_inputs.to(model.device)


def _merged_inputs_embeds(qwen, mm_inputs):
    """Text embeds with the vision tower scattered into image-pad positions.

    Replicates Qwen2_5_VLForConditionalGeneration's internal merge so we can append
    extra tokens at the embedding level. Attribute names verified against the
    pinned Sa2VA wrapper (`self.model` = Qwen2_5_VLForConditionalGeneration).
    """
    import torch

    input_ids = mm_inputs["input_ids"]
    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    pixel_values = mm_inputs.get("pixel_values")
    grid_thw = mm_inputs.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds = qwen.visual(pixel_values, grid_thw=grid_thw)
        image_token_id = qwen.config.image_token_id
        mask = (input_ids == image_token_id)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)
    return inputs_embeds


def _position_ids(qwen, mm_inputs, total_len):
    """M-RoPE positions for the original sequence, extended by +1 per appended token.

    THE landmine. Try the model's own get_rope_index; on any failure fall back to a
    plain arange continuation (good enough for V0: the forward only has to run)."""
    import torch

    input_ids = mm_inputs["input_ids"]
    attention_mask = mm_inputs.get("attention_mask")
    orig_len = input_ids.shape[1]
    n_extra = total_len - orig_len
    # get_rope_index may live on the ForConditionalGeneration (`qwen`) or the inner
    # text model (`qwen.model`) depending on the transformers version — try both.
    rope_fn = getattr(qwen, "get_rope_index", None) or getattr(qwen.model, "get_rope_index", None)
    try:
        position_ids, _ = rope_fn(
            input_ids, mm_inputs.get("image_grid_thw"),
            attention_mask=attention_mask)
        # position_ids: (3, batch, seq). Extend each rope axis by +1 continuation.
        last = position_ids[:, :, -1:]
        cont = last + torch.arange(1, n_extra + 1, device=position_ids.device).view(1, 1, -1)
        position_ids = torch.cat([position_ids, cont], dim=-1)
        return position_ids, "get_rope_index+extend"
    except Exception as exc:  # noqa: BLE001 - the spike's job is to report this
        print(f"  [position_ids] get_rope_index failed ({type(exc).__name__}: {exc}); "
              f"falling back to arange")
        pos = torch.arange(total_len, device=input_ids.device).view(1, 1, -1).expand(3, 1, -1)
        return pos.contiguous(), "arange-fallback"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase-0 teacher-forced keystone spike")
    parser.add_argument("--weights-config", default="configs/weights.v0.json")
    parser.add_argument("--num-edit-slots", type=int, default=4)
    parser.add_argument("--text", default="Please segment the object.")
    args = parser.parse_args(argv)

    import torch
    from torch import nn

    weights_path = _sa2va_path(Path(args.weights_config))
    print(f"[1/5] loading Sa2VA from {weights_path} ...")
    planner = CausalPlanner(PlannerConfig(weights_path=weights_path))
    model, processor = planner._load_model()  # reuse the verified load path
    qwen = model.model  # Qwen2_5_VLForConditionalGeneration
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = int(model.config.text_config.hidden_size)

    print("[2/5] building multimodal inputs ...")
    mm_inputs = _build_mm_inputs(model, processor, _synthetic_image(), args.text)

    print("[3/5] appending held-Parameter new tokens (vocab untouched) ...")
    n_new = 2 + int(args.num_edit_slots)  # [SEG_DIR],[SEG_IND],[EDIT]xN
    new_tokens = nn.Parameter(torch.randn(1, n_new, hidden, device=device, dtype=dtype))
    inputs_embeds = _merged_inputs_embeds(qwen, mm_inputs)
    inputs_embeds = torch.cat([inputs_embeds, new_tokens], dim=1)
    total_len = inputs_embeds.shape[1]
    attn = mm_inputs.get("attention_mask")
    if attn is not None:
        attn = torch.cat([attn, torch.ones(attn.shape[0], n_new, device=attn.device, dtype=attn.dtype)], dim=1)
    position_ids, pos_mode = _position_ids(qwen, mm_inputs, total_len)
    print(f"      position_ids via {pos_mode}; inputs_embeds {tuple(inputs_embeds.shape)}")

    print("[4/5] teacher-forced forward (no generate) ...")
    with torch.no_grad():
        out = qwen.model(  # inner Qwen2_5_VLModel decoder
            inputs_embeds=inputs_embeds, attention_mask=attn,
            position_ids=position_ids, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1]  # (1, total_len, hidden)
    appended = last_hidden[:, -n_new:, :]
    assert appended.shape == (1, n_new, hidden), appended.shape
    seg_dir_hidden = appended[:, 0]      # would drive the DIRECT mask
    edit_hidden = appended[:, 2:]        # would become edit_tokens
    print(f"      OK: recovered hidden at appended positions {tuple(appended.shape)}")

    print("[5/5] drive SAM2 from the recovered hidden -> (T,H,W) mask ...")
    query = model.text_hidden_fcs(seg_dir_hidden)  # reuse shared projection (01:43)
    g_image = _synthetic_image(1024)
    import numpy as np
    g_pixel = torch.from_numpy(np.asarray(g_image)).permute(2, 0, 1).contiguous().to(dtype)
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g_pixel)]).to(device)
    sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel)
    # predict_forward feeds a (1, sam_dim) tensor per frame (modeling_sa2va_qwen.py:241-245);
    # query is already (1, sam_dim) here, so do NOT add another dim.
    masks = model.grounding_encoder.language_embd_inference(sam_states, [query] * 1)
    print(f"      OK: SAM2 returned masks {tuple(masks.shape)} from injected hidden")

    print("\nKEYSTONE PASSED — teacher-forced new-token forward runs and reaches SAM2.")
    print(f"  edit_hidden shape (pre-projection) {tuple(edit_hidden.shape)} -> edit_hidden_fcs -> (Nt,4096)")
    print("  Safe to build Phase 2 (overlay.py / teacher_forced.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
