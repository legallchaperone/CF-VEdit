"""Stage-0 seg-branch training (seg_dir first).

For a single binary mask, seg_dir is the stock Sa2VA `[SEG]` referring-seg case,
so we reuse the vendored Sa2VA XTuner trainer (loss = 2*BCE + 0.5*Dice, LoRA,
frozen SAM2-injection — all upstream) instead of reimplementing a loop. This
package is only the glue:

  - ``data``   : DAVIS/VOID manifest -> Sa2VA finetune annotations.json + images
  - ``configs``: our 7B run config (copy of the vendored one, B4-clean)
  - ``eval``   : held-out IoU/Dice vs GT (inference-only, no xtuner)
  - ``run_stage0.md`` : env + weight-conversion + launch

Spec: E2W-v0-Remove-Only-Spec.md §Stage 0. Decisions/deferred ideas: the ADR
(0009) and ../../TRAINING_NOTES.md. The fixed dual-token single-forward
architecture (seg_dir + seg_ind in one pass, spec change A) is deferred; v0
Stage-0 trains seg_dir as referring seg.
"""
