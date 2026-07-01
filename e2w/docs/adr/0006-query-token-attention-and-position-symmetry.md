# ADR-0006 — Query-token attention grouping and RoPE position symmetry

- **Status:** Accepted
- **Date:** 2026-07-01
- **Anchors:** ADR-0004 (query-token overlay mechanism); ADR-0005 (`edit_tokens` as a T5-encoder
  substitute); Sa2VA-Modification-Plan §B (`[EDIT]` supports "N slots, e.g. 4–8" — an
  undifferentiated capacity pool, not named per-slot fields)

## Context

`query_tokens.py`'s `localize_three_layer` appends 6 positions to the prompt's `inputs_embeds`
in a fixed order: `[SEG_DIR]`, `[SEG_IND]`, `[EDIT]_0..N-1`. Nothing in ADR-0004 specified how
these 6 positions should attend to each other or to the real prompt/video context — the
resulting behavior was whatever the LM backbone's default causal masking + M-RoPE continuation
produced as a side effect of concatenation order, not a design choice. Two independent
asymmetries fell out of that:

1. **Attention connectivity.** Under plain causal masking, `[EDIT]` slots can see `[SEG_DIR]`/
   `[SEG_IND]` (an order artifact — they happen to be concatenated after them), but each `[EDIT]`
   slot only sees *earlier* `[EDIT]` slots, not later ones (`edit_0` sees none of its siblings;
   `edit_{N-1}` sees all of them).
2. **RoPE relative position.** Even if connectivity were fixed to let all `[EDIT]` slots see each
   other, M-RoPE attention scores depend on the *signed* relative position offset between query
   and key positions (Q and K are different projections, so the offset is not a symmetric
   function). Two mutually-visible positions at different sequence indices still carry a
   positional bias from which one comes "first" — masking alone does not remove it.

Both asymmetries matter because `edit_tokens` is explicitly designed to **substitute the output
of a T5 text encoder** (ADR-0005's `_get_t5_prompt_embeds` override), and `[EDIT]` slots have no
per-slot semantic assignment — Sa2VA-Modification-Plan §B describes `[EDIT]` as "N slots, e.g.
4–8" (a capacity knob, like sequence length), not 4 named, ordered fields. A real T5 encoder
computes its embedding sequence with full bidirectional self-attention and no positional
privilege among its tokens. The causal chain + RoPE order bias among `[EDIT]` slots is a
structural mismatch with the thing they impersonate — and fixing it is only free *before*
training gives these slots learned, position-dependent roles.

## Decision

1. **Attention mask** (`_build_query_attention_mask`): a custom 4D additive mask, verified
   supported by the installed `transformers==4.51.3` (`Qwen2_5_VLModel`'s
   `_prepare_4d_causal_attention_mask_with_cache_position` passes a 4D mask through verbatim;
   `planner.py` already pins `attn_implementation="eager"`, which adds it directly to raw
   attention scores). Rules: prompt/video block stays causal as before (including honoring
   padding); `[EDIT]` slots attend to each other bidirectionally (including "forward," overriding
   plain causal); `[EDIT]` and `[SEG_DIR]`/`[SEG_IND]` are mutually blocked in both directions
   (removing the order-artifact visibility `[EDIT]` had into `[SEG_DIR]`/`[SEG_IND]`);
   `[SEG_DIR]`/`[SEG_IND]` mutual visibility is **left unchanged** (still causal, `[SEG_IND]` sees
   `[SEG_DIR]`) — there is no "what does this impersonate" argument to resolve that pair either
   way, so changing it would be an unsupported guess.
2. **Position ids** (`_continuation_offsets`): all `[EDIT]` slots are tied to one shared position
   id (offset `+3` from the prompt's last position), so their pairwise RoPE relative offset is
   exactly 0 — truly order-free, not just mutually visible. `[SEG_DIR]`/`[SEG_IND]` keep distinct
   sequential offsets (`+1`, `+2`) as before. This applies to both the `get_rope_index+extend`
   path and the `arange-fallback` path, so a silent M-RoPE failure does not also silently drop
   the position-symmetry fix.

## Consequences

- **+** `[EDIT]` slots now match the T5-substitution intent on both axes that matter for a
  bidirectional, position-symmetric encoder output: mutual visibility and zero relative position
  bias.
- **+** Both fixes are pre-training tensor construction, no vendored/remote model code touched.
  Doing this now is free; doing it after stage-① training starts would invalidate whatever
  position-dependent roles the slots had already learned.
- **−** `[SEG_DIR]`↔`[SEG_IND]` mutual visibility is **explicitly left open** — flagged here so it
  is not silently assumed settled by this ADR. If a future "what does this impersonate" argument
  emerges for that pair (there is none today), it gets its own ADR.
- **−** Tying `[EDIT]` positions means the model can never use position to distinguish the 4 (or
  N) slots, even if that turns out to be useful after training starts — accepted as the direct
  cost of true order-freedom, not a bug.
- **−** `_build_query_attention_mask` assumes `batch_size=1` (asserted) — the planner probes one
  clip per call today; a future batched-inference path would need to generalize it.
- **Not validated on GPU as part of this change** — a unit test (`tests/test_query_tokens.py`)
  proves the mask/offset tensors are correct in isolation; it does not prove the real checkpoint's
  forward pass still runs end-to-end under the new 4D mask. Flagged as a follow-up smoke run
  (`e2w_full_smoke`-style) when GPU access is convenient, not required to land this ADR.
