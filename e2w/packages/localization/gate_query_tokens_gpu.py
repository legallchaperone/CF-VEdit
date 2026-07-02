"""E2W localization full-path GPU GATE — current ADR-0006 mechanism, real checkpoint.

Validates the CURRENT 4D-mask + tied-RoPE query-token forward on the real Sa2VA
checkpoint. Supersedes spike_query_tokens.py (which predates ADR-0006 and used a
plain causal + arange mechanism — historical artifact, not this mechanism). Two layers:
  (a) full-path localize_three_layer runs; direct/indirect (T,H,W); edit_tokens
      (4,4096); position_ids_mode == 'get_rope_index+extend' (not arange fallback).
  (b) 4D mask semantics really take effect: blocked edit->seg attention weight == 0
      across all layers/heads; positive control edit->edit nonzero; softmax row-sum ~1.

Result (2026-07-02, seed=0, untrained): GATE PASS — (a) ok; (b) blocked max = 0.000
over 28 layers, edit->edit ~3-6e-4, row-sum 1.0. Mechanism runs to spec on the real
checkpoint (untrained -> mask/token content is garbage by design; not checked here).

Verify-only, mutates nothing on disk, vocab untouched. Run:
  CUDA_VISIBLE_DEVICES=<idle> /data/cwx/conda/envs/void/bin/python <this>
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import torch

# repo imports
E2W = Path("/home/cwx/CF-VEdit/e2w")
sys.path.insert(0, str(E2W / "packages/localization"))
sys.path.insert(0, str(E2W / "packages/e2w_core"))

from e2w_localization.planner import CausalPlanner, PlannerConfig
from e2w_localization.overlay import attach_e2w_heads
from e2w_localization import query_tokens as QT
from e2w_localization.query_tokens import (
    localize_three_layer, _build_video_inputs, _merged_inputs_embeds,
    _build_query_attention_mask, _position_ids,
)

WEIGHTS_CFG = json.loads((E2W / "configs/weights.v0.json").read_text())
SA2VA = WEIGHTS_CFG["models"]["sa2va_qwen2_5_vl_7b"]["path"]
VIDEO = "/home/cwx/CF-VEdit/physics_iq_for_simple_eval/videos/source/piq_simple_eval_0018_remove.mp4"
INSTRUCTION = "Remove the tennis ball released above the green kinetic sand."
NUM_EDIT = 4
MAX_CONTENT_FRAMES = 5

print("[load] Sa2VA ...", flush=True)
planner = CausalPlanner(PlannerConfig(weights_path=SA2VA, device="cuda:0",
                                      edit_token_slots=NUM_EDIT,
                                      max_content_frames=MAX_CONTENT_FRAMES))
model, processor = planner._load_model()
attach_e2w_heads(model, num_edit_slots=NUM_EDIT, renderer_condition_dim=4096, seed=0)
frames = planner._load_frames(VIDEO)
print(f"[load] frames={len(frames)} size={frames[0].size}", flush=True)

# ---------------- (a) real shipping path ----------------
print("\n===== (a) full-path localize_three_layer =====", flush=True)
out = localize_three_layer(model, processor, frames, INSTRUCTION,
                           num_edit_slots=NUM_EDIT, max_content_frames=MAX_CONTENT_FRAMES)
direct, indirect = out["direct"], out["indirect"]
edit_tokens = out["edit_tokens"]
pos_mode = out["position_ids_mode"]
print(f"forward: OK (no error)")
print(f"direct  shape={direct.shape} dtype={direct.dtype}")
print(f"indirect shape={indirect.shape} dtype={indirect.dtype}")
print(f"edit_tokens shape={tuple(edit_tokens.shape)} dtype={edit_tokens.dtype}")
print(f"position_ids_mode={pos_mode!r}  (validated path == 'get_rope_index+extend')")
a_pass = (direct.ndim == 3 and indirect.ndim == 3
          and tuple(edit_tokens.shape) == (NUM_EDIT, 4096)
          and pos_mode == "get_rope_index+extend")
print(f"(a) PASS={a_pass}")

# ---------------- (b) 4D mask semantics via output_attentions ----------------
print("\n===== (b) 4D attention-mask semantics =====", flush=True)
qwen = model.model
with torch.no_grad():
    mm_inputs, g_pixel_values, ori_image_size, num_frames = _build_video_inputs(
        model, processor, frames, INSTRUCTION, MAX_CONTENT_FRAMES)
    new_tokens = torch.cat([
        model.seg_dir_embed.view(1, 1, -1),
        model.seg_ind_embed.view(1, 1, -1),
        model.edit_embeds.view(1, NUM_EDIT, -1),
    ], dim=1).to(dtype=next(model.parameters()).dtype)
    inputs_embeds = _merged_inputs_embeds(qwen, mm_inputs)
    n_prompt = inputs_embeds.shape[1]
    inputs_embeds = torch.cat([inputs_embeds, new_tokens], dim=1)
    total_len = inputs_embeds.shape[1]
    n_new = total_len - n_prompt
    attn_mask_4d = _build_query_attention_mask(
        n_prompt=n_prompt, num_edit_slots=NUM_EDIT,
        prompt_padding_mask=mm_inputs.get("attention_mask"),
        dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    position_ids, pmode2 = _position_ids(qwen, mm_inputs, total_len, num_edit_slots=NUM_EDIT)

    # indices
    seg_cols = [n_prompt, n_prompt + 1]          # seg_dir, seg_ind
    edit_idx = list(range(n_prompt + 2, total_len))  # edit_0..3
    print(f"n_prompt={n_prompt} total_len={total_len} n_new={n_new}")
    print(f"seg cols (keys)={seg_cols}  edit rows/cols={edit_idx}")

    # confirm mask design directly
    m2d = attn_mask_4d[0, 0]
    min_val = torch.finfo(inputs_embeds.dtype).min
    er, sc = edit_idx[0], seg_cols[0]
    print(f"mask[edit0,seg_dir]={m2d[er, sc].item():.3e} (expect =={min_val:.1e} blocked)")
    print(f"mask[edit0,edit1]  ={m2d[edit_idx[0], edit_idx[1]].item():.3e} (expect 0.0 open)")
    print(f"mask[edit0,edit3future]={m2d[edit_idx[0], edit_idx[3]].item():.3e} (expect 0.0 bidir)")

    out2 = qwen.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask_4d,
                      position_ids=position_ids, output_hidden_states=False,
                      output_attentions=True, use_cache=False)

atts = out2.attentions
if atts is None or len(atts) == 0:
    print("!! output_attentions returned None -- cannot read weights via out.attentions")
    sys.exit(2)
print(f"num layers with attentions={len(atts)}, per-layer shape={tuple(atts[0].shape)}")

# For each layer: blocked edit->seg weight (should ~0) vs positive edit->edit (nonzero)
et = torch.tensor(edit_idx)
sg = torch.tensor(seg_cols)
blocked_max_per_layer = []
edit_edit_mean_per_layer = []
edit_selfrow_sum_per_layer = []  # softmax normalization sanity over all keys
for li, A in enumerate(atts):
    A = A.float()  # (1, heads, q, kv)
    # blocked: edit rows attending to seg cols
    blk = A[0][:, et][:, :, sg]              # (heads, |edit|, |seg|)
    blocked_max_per_layer.append(blk.max().item())
    # positive control: edit -> edit (off-diagonal, e.g. edit0->edit1..3 + diagonal)
    ee = A[0][:, et][:, :, et]               # (heads, |edit|, |edit|)
    edit_edit_mean_per_layer.append(ee.mean().item())
    # sanity: an edit row's full attention over all keys should sum ~1
    edit_selfrow_sum_per_layer.append(A[0][:, et[0], :].sum(-1).mean().item())

bmax = max(blocked_max_per_layer)
print(f"\nBLOCKED edit->seg  attention weight: max over ALL layers/heads/pairs = {bmax:.3e}")
print(f"  per-layer max (first5)={['%.2e'%x for x in blocked_max_per_layer[:5]]} ... "
      f"(last5)={['%.2e'%x for x in blocked_max_per_layer[-5:]]}")
ee_lo = min(edit_edit_mean_per_layer); ee_hi = max(edit_edit_mean_per_layer)
print(f"POSITIVE edit->edit attention weight (mean/layer): range [{ee_lo:.3e}, {ee_hi:.3e}]")
print(f"  per-layer mean (first5)={['%.3e'%x for x in edit_edit_mean_per_layer[:5]]}")
print(f"softmax row-sum sanity (edit0 over all keys): "
      f"range [{min(edit_selfrow_sum_per_layer):.4f}, {max(edit_selfrow_sum_per_layer):.4f}] (expect ~1.0)")

b_pass = (bmax < 1e-8) and (ee_hi > 1e-4)
print(f"\n(b) mask-semantics PASS={b_pass}  "
      f"(blocked==0 AND positive-control nonzero)")

print("\n================ GATE SUMMARY ================")
print(f"(a) forward+shapes+RoPE : {'PASS' if a_pass else 'FAIL'}")
print(f"(b) 4D mask semantics    : {'PASS' if b_pass else 'FAIL'}")
print(f"OVERALL GATE: {'PASS' if (a_pass and b_pass) else 'FAIL'}")
