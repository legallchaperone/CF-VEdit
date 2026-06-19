# localization — 定位半 (Causal Planner + Mask decoder)

Decides **which regions to change**. Built on **Sa2VA** (`bytedance/Sa2VA`,
Apache-2.0, variant `Sa2VA-Qwen2_5-VL-7B`). Implements architecture §A.2 【2】+【3】
and the deltas in [sa2va-plan](../../docs/proposal/README.md) (changes A–D).

## What Sa2VA gives for free vs. what we add

| we need | Sa2VA has it? |
|---|---|
| query vector (`[SEG]` hidden → `text_hidden_fcs` → `pred_embeddings`) | ✅ ready |
| mask decoder + temporal propagation (SAM2 `inject_language_embd` + memory) | ✅ ready |
| **three-layer** causal mask (split `[SEG]` → `[SEG_DIR]`/`[SEG_IND]`; unchanged = complement) | ❌ change A |
| edit-plan `[EDIT]` tokens + projection head (gradient comes from the renderer) | ❌ change B |
| CF sim paired dataset with causal-layer labels | ❌ change C |
| training config (unfreeze mask decoder, two-layer loss weights) | ❌ change D |

## Honest risk (sa2va-plan §3 — read this)

> **Mechanism free ≠ capability free.** Sa2VA hands over the tedious, error-prone
> 1/3 (instruction → mask plumbing) and saves nothing on the hard research 1/3.

The **indirect / multi-hop** layer is the 命门: Sa2VA never learned *how far*
causation propagates — that has to be taught with the `data_engine` dependency
graph. This is also the model's ceiling.

## Planned layout (created when vendoring — boundary B4)

```
localization/
  third_party/sa2va/   vendored submodule, pinned commit, NEVER edited in place
  patches/             our overlay deltas to Sa2VA (auditable, separate)
  e2w_localization/    new code: three-layer mask, [SEG_DIR]/[SEG_IND]/[EDIT], CF dataset, config
  tests/
```

Depends on `e2w_core` only (produces `ThreeLayerMask` + `EditPlan`). Must never
import `generation` (B3).
