"""Stage-0 seg-branch training (seg_dir / seg_ind).

Fine-tunes the Sa2VA planner + the seg query tokens + shared projection with a
per-layer Dice+BCE mask loss (frozen SAM2 decoder), supervised by the DAVIS/VOID
quadmask pseudo-labels the data_engine produces. Spec: E2W-v0-Remove-Only-Spec.md
§Stage 0. Method decisions + deferred ideas live in ../../TRAINING_NOTES.md.
"""
