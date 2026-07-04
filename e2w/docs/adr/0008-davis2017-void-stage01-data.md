# ADR-0008 — Stage 0/1 training data uses DAVIS2017 through VOID VLM-MASK-REASONER

- **Status:** Accepted
- **Date:** 2026-07-04
- **Anchors:** `E2W-v0-Remove-Only-Spec.md` §2 Stage 0/1,
  [ADR-0007](0007-e2w-v0-remove-only-void-renderer.md),
  `packages/data_engine/e2w_data_engine/davis2017_remove.py`

## Context

ADR-0007 narrows v0 to remove-only on VOID's frozen renderer. Stage 0 needs
direct/indirect mask supervision, and Stage 1 needs a post-removal text
condition. The v0 spec allows real videos for Stage 0/1 pseudo-labeling through
VOID's VLM-MASK-REASONER.

DAVIS2017 gives frame-level instance masks for real videos. VOID's stage2 VLM
analysis can supply post-removal scene text plus affected/integral object
lists, and `stage3a_generate_grey_masks_v2.py` tracks affected objects through
all frames.

## Decision

Stage 0/1 training rows are built from DAVIS2017 train sequences with the
local builder:

- DAVIS palette-index annotations provide the direct object mask. Valid object
  ids are 1..254; 0 is background; 255 is void/boundary and is never emitted as
  an object.
- VOID stage2 (`stage2_vlm_analysis_cf.py`) provides the post-removal scene
  description, integral belongings, and affected-object nouns.
- VOID stage3a uses `stage3a_generate_grey_masks_v2.py`, not the older static
  first-frame script.
- The instruction-mask invariant is mandatory: the 0/remove mask must be
  exactly the referent of the instruction. "remove the bike" uses the bike
  mask; "remove the person and bike" uses the union of both masks.
- Multi-object rows need explicit per-object names. If a name is missing or
  ambiguous, the row quarantines as `unresolvable_target_ref`; the builder never
  emits "highlighted object" as a clean training target.
- If stage2 marks another DAVIS object as an integral belonging, the builder
  emits a merged row for the union and quarantines the individual member rows.
  If the integral noun has no matching DAVIS object, the row quarantines.
- `grey_mask.mp4` absence or frame/shape mismatch quarantines the row. A clean
  `void_vlm_weak` indirect label is only stamped when the grey mask was read;
  VLM-listed-zero affected objects are stamped `void_vlm_none`.

Stage 0 training input is `frames_dir` JPEGs. `source_video` is a lossy preview
re-encode kept for VOID scripts and human review only.

## Consequences

- The training labels stay aligned with the text query used by `[SEG_DIR]` and
  `[EDIT]`; contradictory "same video, same generic instruction, different
  masks" rows are quarantined.
- The builder remains a thin adapter around DAVIS + VOID. It records audit data
  for VLM affected nouns that disappear after VOID's proximity-filtered grey
  mask, but it does not fork VOID's stage3a implementation.
- Stage 2 paired factual/counterfactual video data remains out of scope for this
  ADR; the future simulator path still needs its own implementation and tests.
