# ADR-0004 — Fixed query tokens as a B4-legal runtime overlay

- **Status:** Accepted
- **Date:** 2026-06-20
- **Anchors:** Sa2VA-Modification-Plan §A/§B; Architecture §A.2【2】【3】; Repo-Design §2 (B4); ADR-0003

## Context

Three hard facts collide:

1. **Untrained tokens can't be emitted.** An untrained model never generates
   `[SEG_DIR]/[SEG_IND]/[EDIT]` autoregressively, and the HF wrapper's
   `predict_forward` only reads hidden states for *generated* tokens. So the
   trained-path mechanism is unreachable via `generate()`.
2. **The embedding-resize hazard.** `tokenizer.add_tokens` + `resize_token_embeddings`
   on this Sa2VA-Qwen2.5-VL checkpoint under transformers≥4.51 perturbs generation
   (the reason V0 bypassed the heads, `planner.py:140-143`).
3. **B4.** `third_party/` must stay byte-clean (CI gate). And the *runtime* Sa2VA
   code is the checkpoint snapshot (`weights.v0.json` → `/data/cwx/...`, loaded via
   `trust_remote_code`), **not** `third_party/sa2va` — editing third_party would not
   even change the loaded model.

## Decision

Work entirely in **`inputs_embeds` space; never mutate tokenizer/vocab.**

- `overlay.attach_e2w_heads` composes onto the *loaded instance*: `[SEG_DIR]`,
  `[SEG_IND]`, `[EDIT]×N` as held `nn.Parameter` embedding vectors (not vocab rows),
  plus `edit_hidden_fcs` (hidden → 4096). `text_hidden_fcs` is reused for the seg
  queries.
- `query_tokens.localize_three_layer` builds the prompt's merged `inputs_embeds`
  (vision tower scattered in), **appends** the new-token vectors, runs a plain
  forward (`output_hidden_states=True`, no `generate()`), recovers hidden **by
  position**, and drives two independent SAM2 passes (direct + indirect) + the edit
  projection. M-RoPE positions come from `get_rope_index` extended by +1
  continuation per appended token.
- Deltas live in `e2w_localization/` (runtime composition); a `patches/` directory
  holds the **unapplied** source-level diff for the eventual train fork
  (auditability only). `third_party/` and the snapshot are never mutated.

## Consequences

- **+** Sidesteps the resize hazard *and* B4 at once: vocab untouched → the vanilla
  `[SEG]` `generate()` path is byte-identical; nothing in `third_party/` changes.
- **+** Validated on the real checkpoint (`spike_query_tokens.py` and a
  single-sample full run): forward runs, hidden recovered `(1, 2+N, 3584)`, dual
  SAM2 yields real `(T,H,W)` direct **and** indirect masks, `edit_tokens (Nt,4096)`.
  The spike confirmed the **`get_rope_index+extend`** M-RoPE path specifically (it
  prints the mode). Because `_position_ids` falls back to a plain `arange` on any
  `get_rope_index` failure, the chosen mode is **warned + recorded per-sample in
  `run_meta.position_ids_modes`** so a silent fallback can never be mistaken for the
  validated path.
- **−** The held-`nn.Parameter` form is a V0 device; the train fork must own real
  token registration (the `patches/` diff is its starting point).
- **−** `edit_hidden_fcs` gets no gradient inside localization (01:67-71); it is
  trained jointly with the renderer in stage ②.
- **−** Multi-frame vision forward must run under `no_grad` or it OOMs (activations);
  enforced in `query_tokens.py`.

## Amendment (2026-07-01) — renamed from "teacher forcing" to "query tokens"

The original title/body called this mechanism "teacher forcing." That was an
inaccurate borrowed term: teacher forcing requires (a) a training loss and (b)
substituting *ground truth* for the model's own autoregressive output — neither
existed here (V0 runs `no_grad`, no loss, no ground truth; the 3 appended vectors
are unconditionally injected every call, not overridden from a labeled target).
Renamed to "query tokens," borrowing the DETR/Q-Former **concept** — a learnable,
non-vocab slot used to read out information — but not their mechanism: DETR/
Q-Former queries run through an independent cross-attention module against
separately-computed encoder K/V, whereas these 6 slots are concatenated straight
into the LM's own self-attention sequence and share its QKV projections. "Query
tokens" names what role they play, not which architecture they belong to. See
ADR-0006 for the attention/position-id design these query tokens use among
themselves.
