# Localization training notes (Stage 0 seg branch)

Running log of training-method decisions and deferred ideas for the seg_dir /
seg_ind branch. Design anchors: `Sa2VA-Modification-Plan.md` §1 (on main) and
`E2W-v0-Remove-Only-Spec.md` §Stage 0 (authoritative build spec, currently on the
`feat/e2w-void-renderer` branch — not yet merged to main).

## Current method — what THIS PR actually trains (ADR-0009)
**seg_dir only**, as stock Sa2VA `[SEG]` referring segmentation ("segment
<target_ref>" → its mask), on the DAVIS direct masks, via the **vendored Sa2VA
XTuner trainer**. Not the dual-token path. Concretely:
- One `[SEG]` per image → `text_hidden_fcs` projection → SAM2 mask decoder →
  one binary mask. Loss (upstream) = **2·BCE + 0.5·Dice**.
- Trainable = LoRA on the Qwen LLM (r=128) + the `text_hidden_fcs` projection +
  the **SAM2 mask decoder** (`frozen_sam2_decoder=False`, per the config); the
  SAM2 image/prompt encoder, the visual encoder, and the LLM base are frozen.
  (This matches Sa2VA's own finetune recipe — the mask *decoder* is tuned, not
  the whole of SAM2.)
- Data adapter expands clips → per-frame `{image, direct-mask polygon,
  target_ref}` (`training/data.py`).

## Eventual target — deferred (spec change A)
The spec's architecture is two fixed query tokens `seg_dir`/`seg_ind` read in
**one** forward → shared projection → SAM2 → two masks, `seg_ind` supervised by
the affected GT, seg branch trained **only** by the mask loss (the
threshold→quadmask step is non-differentiable, so renderer/video loss can't reach
it). That dual-token trainer + `seg_ind` are NOT built here — deferred until the
seg_dir floor + learning curve are read and the data source is scaled (see the
data-scaling note below).

## Deferred ideas (not doing now)

### Soft distillation instead of hard binary GT  — worth trying
Train `seg_ind` (and possibly `seg_dir`) against the teacher's **probability
maps** (SAM2 / langsam soft outputs) rather than thresholded binary masks. Soft
targets preserve the teacher's uncertainty and are markedly more robust to the
coarse/gridified/partial affected labels than hard Dice+BCE — we stop forcing the
student to be confident where the teacher wasn't. No architecture change; strictly
more information per label. Try after a hard-label baseline exists, compare
held-out IoU. (Rejected alternatives that break the design: "reason-then-segment"
= structurally VOID + splits the single differentiable planner into a pipeline.)

### End-to-end differentiable masks — only once renderer is trainable
Straight-through / soft masks so the renderer's video loss can also shape the
seg masks. Rejected for v0 (frozen renderer is OOD on continuous mask values, and
video-loss-through-a-frozen-renderer is a weak localization signal vs direct
Dice+BCE). Revisit when the renderer is unfrozen for sim-data training.

### Data scaling, not loss tricks — the real lever
93 DAVIS rows is a pilot, likely enough for `seg_dir` (in-distribution) but thin
for `seg_ind` (novel skill, ~74 noisy positives). Scale via the builder on more
video sources → ultimately Kubric sim (unlimited + perfect physics labels).
Calibrate appetite empirically: learning curve (25/50/100% subsets) vs a clean
held-out IoU; watch for `seg_ind`≈dilated-`seg_dir` collapse and empty-prediction
collapse.
