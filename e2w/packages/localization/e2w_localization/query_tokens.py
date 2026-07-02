"""Query-token localization forward — the real (untrained) A.1 planner path.

An untrained model never *emits* ``[SEG_DIR]/[SEG_IND]/[EDIT]`` autoregressively and
``predict_forward`` only reads hidden for *generated* tokens, so the trained-path
mechanism cannot be reached via ``generate()``. Instead we inject fixed query
tokens: append the held-Parameter token embeddings (from
``overlay.attach_e2w_heads``) to the prompt's ``inputs_embeds`` and run a plain
forward with ``output_hidden_states=True``. The recovered hidden at the appended
positions drive:

- ``text_hidden_fcs`` -> SAM2 (dual pass) -> DIRECT and INDIRECT masks
- ``edit_hidden_fcs`` -> ``edit_tokens`` (Nt, 4096) for the renderer

Untrained -> the masks/tokens are shape-correct garbage by design (V0). This
current 4D-mask + tied-RoPE mechanism is validated on the real checkpoint by
``gate_query_tokens_gpu.py`` (GATE PASS 2026-07-02: forward runs; edit->seg
attention weight == 0 across all layers). (``spike_query_tokens.py`` is the older
pre-ADR-0006 artifact — plain causal + arange, not this mechanism.)

This is NOT teacher forcing (ADR-0004 amendment): there is no loss and no ground
truth being substituted for the model's own output here, only an unconditional
append. "Query tokens" borrows the DETR/Q-Former *concept* (a learnable, non-vocab
slot used to read out information) but not their mechanism — these 6 slots are
concatenated into the LM's own self-attention sequence and share its QKV
projections, not routed through an independent cross-attention decoder.

Attention/position design among the 6 appended positions (ADR-0006): default
causal masking + concatenation order would leave ``[EDIT]`` slots able to see
``[SEG_DIR]``/``[SEG_IND]`` (an order artifact) while only seeing *earlier*
``[EDIT]`` slots, not later ones — a mismatch with ``edit_tokens``' role as a
stand-in for a (bidirectional) T5 encoder's output (ADR-0005). ``[EDIT]`` slots
are therefore made mutually bidirectional and isolated from ``[SEG_DIR]``/
``[SEG_IND]`` via a custom 4D attention mask (:func:`_build_query_attention_mask`),
and tied to a single shared position id so their pairwise RoPE relative offset is
exactly 0 (:func:`_continuation_offsets`) — masking connectivity alone does not
remove the RoPE positional bias between two mutually-visible positions.
``[SEG_DIR]``/``[SEG_IND]`` mutual visibility is left unchanged (still causal):
there is no analogous "what does this impersonate" argument to resolve it either
way (open question, ADR-0006).

Boundary: emits only e2w_core-compatible payloads; never imports generation.
"""
from __future__ import annotations

from typing import Any


def _build_video_inputs(model: Any, processor: Any, video_frames: list, instruction: str,
                        max_content_frames: int = 5):
    """Mirror predict_forward's video block (modeling_sa2va_qwen.py:120-165)."""
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info

    ori_image_size = video_frames[0].size  # (W, H) for PIL
    extra_pixel_values = []
    content = []
    for frame_idx, frame_image in enumerate(video_frames):
        g_image = np.array(frame_image)
        g_image = model.extra_image_processor.apply_image(g_image)
        g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
        extra_pixel_values.append(g_image)
        if frame_idx < max_content_frames:
            content.append({"type": "image", "image": frame_image})
    content.append({"type": "text", "text": instruction})

    messages = [{"role": "user", "content": content}]
    processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    mm_inputs = processor(
        text=[processed_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", min_pixels=model.min_pixels, max_pixels=model.max_pixels,
    ).to(model.device)

    g_pixel_values = torch.stack(
        [model.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values]
    ).to(model.torch_dtype)
    num_frames = min(max_content_frames, len(video_frames))
    return mm_inputs, g_pixel_values, ori_image_size, num_frames


def _merged_inputs_embeds(qwen: Any, mm_inputs: Any):
    """Text embeds with the vision tower scattered into image-pad positions."""
    input_ids = mm_inputs["input_ids"]
    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    pixel_values = mm_inputs.get("pixel_values")
    grid_thw = mm_inputs.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds = qwen.visual(pixel_values, grid_thw=grid_thw)
        mask = (input_ids == qwen.config.image_token_id)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)
    return inputs_embeds


def _continuation_offsets(num_edit_slots: int) -> list[int]:
    """Position-id offsets for the appended block, relative to the prompt's last
    position (ADR-0006): ``[SEG_DIR]``=+1, ``[SEG_IND]``=+2, all ``[EDIT]`` slots
    tied at +3 so their pairwise RoPE relative offset is exactly 0 (order-free,
    matching :func:`_build_query_attention_mask`'s bidirectional-among-``[EDIT]``
    connectivity — a mask alone does not remove the RoPE positional bias between
    two positions that can see each other)."""
    return [1, 2] + [3] * int(num_edit_slots)


def _build_query_attention_mask(*, n_prompt: int, num_edit_slots: int, prompt_padding_mask: Any,
                                dtype: Any, device: Any):
    """Additive 4D attention mask (ADR-0006): causal over prompt/video as before;
    ``[EDIT]`` slots bidirectional among themselves; ``[EDIT]`` <-> ``[SEG_DIR]``/
    ``[SEG_IND]`` blocked both directions; ``[SEG_DIR]``/``[SEG_IND]`` mutual
    visibility unchanged (still causal, left as an open question in ADR-0006).

    ``batch_size=1`` only — the planner probes one clip per call.
    """
    import torch

    assert prompt_padding_mask is None or prompt_padding_mask.shape[0] == 1, (
        f"batch_size=1 only, got {prompt_padding_mask.shape[0]}"
    )
    n_new = 2 + int(num_edit_slots)
    total_len = n_prompt + n_new
    seg_idx = torch.arange(n_prompt, n_prompt + 2, device=device)
    edit_idx = torch.arange(n_prompt + 2, total_len, device=device)

    row = torch.arange(total_len, device=device).view(-1, 1)
    col = torch.arange(total_len, device=device).view(1, -1)
    causal_ok = col <= row
    valid_col = torch.ones(total_len, dtype=torch.bool, device=device)
    if prompt_padding_mask is not None:
        valid_col[:n_prompt] = prompt_padding_mask[0].bool()
    allowed = causal_ok & valid_col.view(1, -1)

    min_val = torch.finfo(dtype).min
    mask = torch.full((total_len, total_len), min_val, dtype=dtype, device=device)
    mask[allowed] = 0.0
    # [EDIT] cannot see [SEG_DIR]/[SEG_IND] (was allowed under plain causal — order artifact).
    mask[edit_idx[:, None], seg_idx] = min_val
    # [EDIT] <-> [EDIT] fully bidirectional (was one-directional under plain causal).
    mask[edit_idx[:, None], edit_idx] = 0.0
    return mask.view(1, 1, total_len, total_len)


def _position_ids(qwen: Any, mm_inputs: Any, total_len: int, *, num_edit_slots: int):
    """M-RoPE positions; the appended block uses :func:`_continuation_offsets`
    (ADR-0006) instead of a plain arange continuation, so all ``[EDIT]`` slots
    share one position id.

    Returns ``(position_ids, mode)``. ``mode`` is ``"get_rope_index+extend"`` on the
    real M-RoPE path or ``"arange-fallback"`` if get_rope_index raised — the fallback
    keeps the run alive but is NOT the validated path, so it is surfaced (warned +
    recorded in run_meta) rather than swallowed silently.
    """
    import warnings

    import torch

    input_ids = mm_inputs["input_ids"]
    orig_len = input_ids.shape[1]
    n_extra = total_len - orig_len
    offsets = _continuation_offsets(num_edit_slots)
    assert len(offsets) == n_extra, f"offsets length {len(offsets)} != n_extra {n_extra}"
    rope_fn = getattr(qwen, "get_rope_index", None) or getattr(qwen.model, "get_rope_index", None)
    try:
        position_ids, _ = rope_fn(input_ids, mm_inputs.get("image_grid_thw"),
                                  attention_mask=mm_inputs.get("attention_mask"))
        last = position_ids[:, :, -1:]
        offsets_t = torch.tensor(offsets, device=position_ids.device).view(1, 1, -1)
        cont = last + offsets_t
        return torch.cat([position_ids, cont], dim=-1), "get_rope_index+extend"
    except Exception as exc:  # noqa: BLE001 - keep the run alive, but make the degradation loud
        warnings.warn(f"M-RoPE get_rope_index failed ({type(exc).__name__}: {exc}); "
                      f"falling back to arange position_ids — NOT the validated path")
        base = torch.arange(orig_len, device=input_ids.device)
        offsets_t = torch.tensor(offsets, device=input_ids.device)
        cont = (orig_len - 1) + offsets_t
        pos = torch.cat([base, cont]).view(1, 1, -1).expand(3, 1, -1).contiguous()
        return pos, "arange-fallback"


def _sam2_mask(model: Any, query: Any, g_pixel_values: Any, ori_image_size, num_frames: int):
    """Drive SAM2 from one query embedding -> (T, H, W) bool mask (predict_forward:243-250)."""
    import torch.nn.functional as F

    sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel_values)  # fresh state per layer
    pred_masks = model.grounding_encoder.language_embd_inference(sam_states, [query] * num_frames)
    w, h = ori_image_size
    masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    masks = masks[:, 0].sigmoid() > 0.5
    return masks.cpu().numpy()


def localize_three_layer(model: Any, processor: Any, video_frames: list, instruction: str, *,
                         num_edit_slots: int = 4, max_content_frames: int = 5) -> dict:
    """Query-token three-layer localization. Returns direct/indirect masks + planner vectors.

    ``model`` must already have E2W heads attached (overlay.attach_e2w_heads).
    """
    import torch

    qwen = model.model
    # Untrained inference only — everything (incl. the vision tower) under no_grad so
    # activations are not retained (otherwise the vision forward OOMs on multi-frame
    # clips). There is no backward anywhere in V0.
    with torch.no_grad():
        mm_inputs, g_pixel_values, ori_image_size, num_frames = _build_video_inputs(
            model, processor, video_frames, instruction, max_content_frames)

        # Append [SEG_DIR],[SEG_IND],[EDIT]xN as embedding vectors — vocab untouched.
        new_tokens = torch.cat([
            model.seg_dir_embed.view(1, 1, -1),
            model.seg_ind_embed.view(1, 1, -1),
            model.edit_embeds.view(1, int(num_edit_slots), -1),
        ], dim=1).to(dtype=next(model.parameters()).dtype)

        inputs_embeds = _merged_inputs_embeds(qwen, mm_inputs)
        n_prompt = inputs_embeds.shape[1]
        inputs_embeds = torch.cat([inputs_embeds, new_tokens], dim=1)
        total_len = inputs_embeds.shape[1]
        n_new = total_len - n_prompt
        attn_mask_4d = _build_query_attention_mask(
            n_prompt=n_prompt, num_edit_slots=int(num_edit_slots),
            prompt_padding_mask=mm_inputs.get("attention_mask"),
            dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )
        position_ids, position_ids_mode = _position_ids(
            qwen, mm_inputs, total_len, num_edit_slots=int(num_edit_slots))

        out = qwen.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask_4d,
                         position_ids=position_ids, output_hidden_states=True, use_cache=False)
        appended = out.hidden_states[-1][:, -n_new:, :]  # (1, n_new, hidden)

        seg_dir_hidden = appended[:, 0]          # (1, hidden)
        seg_ind_hidden = appended[:, 1]          # (1, hidden)
        edit_hidden = appended[:, 2:]            # (1, Nt, hidden)

        # region_query: shared seg projection (Sa2VA-plan change A, 01:43)
        query_dir = model.text_hidden_fcs(seg_dir_hidden)   # (1, sam_dim)
        query_ind = model.text_hidden_fcs(seg_ind_hidden)
        # edit_tokens: per-slot projection to renderer width (Nt, 4096)
        edit_tokens = model.edit_hidden_fcs(edit_hidden).squeeze(0)  # (Nt, 4096)

        direct = _sam2_mask(model, query_dir, g_pixel_values, ori_image_size, num_frames)
        indirect = _sam2_mask(model, query_ind, g_pixel_values, ori_image_size, num_frames)

    return {
        "direct": direct,
        "indirect": indirect,
        "region_query": torch.stack([query_dir, query_ind], dim=1).detach(),  # (1, 2, sam_dim)
        "edit_tokens": edit_tokens.detach(),                                   # (Nt, 4096)
        "position_ids_mode": position_ids_mode,
    }
