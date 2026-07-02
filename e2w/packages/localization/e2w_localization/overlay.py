"""B4-legal runtime composition of E2W heads onto a loaded Sa2VA instance.

The Sa2VA runtime code is the checkpoint snapshot (loaded via trust_remote_code),
not ``third_party/sa2va`` — so editing third_party would not change the model and is
forbidden by B4 (CI keeps third_party byte-clean). Instead we *compose* the new
heads onto the loaded instance at runtime:

- ``[SEG_DIR]/[SEG_IND]/[EDIT]×N`` are held ``nn.Parameter`` embedding vectors, NOT
  vocab rows. The tokenizer/vocab is never resized, so the vanilla ``[SEG]``
  ``generate()`` path stays byte-identical (this is what sidesteps the
  transformers>=4.51 embedding-resize hazard the planner comments flag).
- ``edit_hidden_fcs`` projects each ``[EDIT]`` hidden state to the v0 renderer's
  text-condition width — CogVideoX-Fun-InP's T5 (4096). Mirrors stock ``text_hidden_fcs``.
- ``text_hidden_fcs`` (already on the model) is reused for the seg-layer queries
  (Sa2VA-plan change A: the seg query projection is shared, 01:43).

All weights are random/identity init — V0 is UNTRAINED, so the masks/edit_tokens
this produces are shape-correct garbage by design. The corresponding train fork's
source-level diff lives, unapplied, under ``patches/`` for auditability.

Verified working end-to-end on the real checkpoint by ``spike_query_tokens.py``
(query-token forward + M-RoPE + dual SAM2 path).
"""
from __future__ import annotations

from typing import Any

E2W_HEADS_FLAG = "_e2w_heads_attached"


def attach_e2w_heads(model: Any, *, num_edit_slots: int = 4, renderer_condition_dim: int = 4096,
                     seed: int = 0) -> Any:
    """Compose the E2W new-token embeddings + edit projection onto ``model``.

    Idempotent: a second call is a no-op. Returns ``model`` for chaining.
    """
    if getattr(model, E2W_HEADS_FLAG, False):
        return model

    import torch
    from torch import nn

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = int(model.config.text_config.hidden_size)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def _param(*shape):
        return nn.Parameter(torch.randn(*shape, generator=gen).to(device=device, dtype=dtype))

    # New "tokens" as held embedding vectors — vocab untouched.
    model.seg_dir_embed = _param(hidden)
    model.seg_ind_embed = _param(hidden)
    model.edit_embeds = _param(int(num_edit_slots), hidden)

    # Per-slot [EDIT] hidden -> v0 renderer text-condition width: CogVideoX-Fun T5 (4096). Random init; gets no
    # gradient inside localization (01:67-71) — trained jointly with the renderer.
    edit_fcs = nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, int(renderer_condition_dim)), nn.Dropout(0.0),
    ).to(device=device, dtype=dtype)
    for m in edit_fcs:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
    model.edit_hidden_fcs = edit_fcs

    model.e2w_num_edit_slots = int(num_edit_slots)
    model.e2w_renderer_condition_dim = int(renderer_condition_dim)
    setattr(model, E2W_HEADS_FLAG, True)
    return model
