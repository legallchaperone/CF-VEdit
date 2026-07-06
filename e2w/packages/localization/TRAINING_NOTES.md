# Localization training notes (Stage 0 seg branch)

Running log of training-method decisions and deferred ideas for the seg_dir /
seg_ind branch. Design anchor: `E2W-v0-Remove-Only-Spec.md` §Stage 0.

## Current method (baseline)
Pseudo-label distillation of VOID's quadmask into two query tokens: one Sa2VA
forward → `seg_dir`/`seg_ind` hidden → shared projection (`text_hidden_fcs`) →
frozen SAM2 → two binary masks. Loss = per-layer **Dice + BCE** (seg_dir ←
direct GT, seg_ind ← affected GT). Trainable: Sa2VA LoRA (shared) + seg-token
embeddings + shared projection. SAM2 frozen. Seg branch gets gradient **only**
from this mask loss (threshold→quadmask is non-differentiable; video loss can't
reach it).

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
