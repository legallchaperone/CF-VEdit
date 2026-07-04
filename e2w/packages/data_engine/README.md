# data_engine — Stage 0/1 labels plus future sim data

Current v0 code builds remove-only Stage 0/1 training rows from DAVIS2017 via
VOID's VLM-MASK-REASONER (stage2 + `stage3a_generate_grey_masks_v2.py`), per
[ADR-0008](../../docs/adr/0008-davis2017-void-stage01-data.md). The future
Kubric-style sim engine remains the Stage 2 paired-video path, not this
builder's job.

## DAVIS2017 remove builder

`e2w_data_engine/davis2017_remove.py` writes one training row per named DAVIS
object, plus merged rows for integral object pairs identified by VOID stage2.
It never widens a prompt silently: the 0/remove mask is exactly the instruction
referent.

Data flow:

```
DAVIS JPEGImages/480p/<sequence>/*.jpg   -> frames_dir training input
DAVIS Annotations/480p/<sequence>/*.png  -> direct_mask_npy
VOID stage2 VLM analysis                 -> post_removal_description + integral/affected nouns
VOID stage3a_v2 grey masks               -> indirect_mask_npy
direct + indirect                        -> quadmask_npy
```

Label provenance:

- `direct`: `davis_gt`, read from palette indices 1..254; index 0 is
  background, 255 is DAVIS void/boundary and is excluded.
- `indirect`: `void_vlm_weak` only when `grey_mask.mp4` was actually read;
  `void_vlm_none` means stage2 listed no affected objects; missing/mismatched
  grey masks quarantine the row.
- `text_condition`: `void_bg`, stage2's scene description after removal.

`source_video` is a lossy preview for VOID scripts and human review. Stage 0
training must consume `frames_dir` JPEGs, not `source_video`.

For multi-object sequences, pass explicit names with `--object-names-json`.
Unnamed multi-object rows quarantine as `unresolvable_target_ref` rather than
emitting "remove the highlighted object".

## Future paired simulator

A Kubric-style simulation engine should render, from a **shared seed**, a
`factual` / `counterfactual` pair plus an **object-level causal dependency
graph** (proposal §3, architecture §A.3). The dependency graph is what teaches
the hard part — it is the supervision for the **indirect / multi-hop** mask
layer that no pretrained component knows ([novelty ②](../../docs/TRACEABILITY.md)).

### What one simulator sample yields (the "three-piece set")

- source video **V** (no intervention),
- ground-truth **V\*** (same seed + `do(X=x)`),
- label map **M** (per-frame, from the sim log): direct / indirect / unchanged.

Because factual and counterfactual share the seed, non-descendants are
bit-identical → exact invariant labels and the cleanest invariant-loss signal.

## Boundaries

- **B5 — train/eval disjoint:** this engine is **dev/val only** and never appears
  in the report. Evaluation uses real held-out video. Keep the two strictly
  separate; record provenance.
- Depends on `e2w_core` only (emits `ThreeLayerMask`-shaped labels + `Operation`
  intervention metadata).

## Layout

```
data_engine/
  e2w_data_engine/davis2017_remove.py  DAVIS2017 -> Stage 0/1 rows
  e2w_data_engine/                     future sim data code
  tests/
```

Scope note: the engine only covers the physics it models — semantic/social/
biological counterfactuals are out of coverage ([SCOPE.md](../../docs/SCOPE.md)).
