# ADR-0009 — Stage-0 seg_dir trains as Sa2VA referring-seg on the vendored XTuner trainer

- **Status:** Accepted
- **Date:** 2026-07-06
- **Anchors:** `Sa2VA-Modification-Plan.md` §1 (localization / seg branch, on main);
  [ADR-0003](0003-enable-full-untrained-a1-path.md) (untrained A.1 path),
  [ADR-0008](0008-davis2017-void-stage01-data.md) (the data);
  `packages/localization/e2w_localization/training/`, `TRAINING_NOTES.md`.
  The fuller `E2W-v0-Remove-Only-Spec.md` §Stage 0 is the authoritative build
  spec but currently lives on the model-build branch (`feat/e2w-void-renderer`),
  not yet merged to main — cited for provenance, resolves post-merge.

## Context

Stage 0 must train the seg branch to predict direct/indirect masks. The spec's
target architecture (sa2va-plan change A) is two fixed query tokens
`seg_dir`/`seg_ind` read in **one** forward, projected through a shared head into
frozen SAM2. Only the untrained forward of that path exists today
(`query_tokens.py`, ADR-0003); there is no trainer — no loss, no optimizer, no
dataloader for our masks.

Two facts narrow the first step:
- For a **single** binary mask, `seg_dir` ("segment the object named X" → its
  mask) is exactly Sa2VA's pretrained `[SEG]` referring-segmentation task — no new
  architecture is required to train it.
- The vendored Sa2VA (`third_party/sa2va`) ships a **complete XTuner LoRA
  trainer** whose loss is already `2·BCE + 0.5·Dice` with SAM2 mask-decoder
  injection, and a finetune dataset (`Sa2VAFinetuneDataset`) taking
  `{image, mask:[polygons], text:[phrase]}`.

Reimplementing that forward + collate (the alternative) is high-risk and
duplicative.

## Decision

1. **v0 Stage-0 `seg_dir` is trained as stock Sa2VA referring segmentation** on
   the DAVIS direct masks, via the vendored XTuner trainer. We add only glue:
   - `training/data.py` — manifest → `annotations.json` (per-frame image + direct
     mask as COCO polygons + `target_ref` phrase). `CHAIN_APPROX_NONE` + a
     **polygon-fidelity gate** (drop frames whose polygon reconstructs the source
     mask at < 0.85 IoU — e.g. holey wheel masks RETR_EXTERNAL can't represent)
     keeps labels faithful; sequence-level train/val split (no leakage).
   - `training/configs/sa2va_qwen7b_stage0_segdir.py` — our run config, a copy of
     the vendored 3B config retargeted to the 7B checkpoint + our data (B4:
     vendored upstream is not edited in place; the run config lives in our tree).
   - `training/eval.py` — held-out IoU/Dice via the inference `CausalPlanner`
     (no xtuner needed), so the same command gives the zero-shot floor and the
     post-finetune number.
   - `training/run_stage0.md` — env + weight-conversion (`convert_to_pth`) +
     launch. The env (a conda clone with `xtuner`+`deepspeed`) is not yet stood
     up on this box; it is the one pending piece.
2. **Trainable set** = LoRA on the Qwen LLM (r=128) + the mask projection
   (`text_hidden_fcs`) + SAM2 mask decoder (`frozen_sam2_decoder=False`); visual
   encoder frozen. This matches the Stage-0 spec (LoRA + projection + decoder).
3. **`seg_ind` and the fixed dual-token single-forward path (change A) are
   deferred.** `data.py --layer indirect` already emits affected-mask data, but
   training a genuine causal-region skill from ~74 noisy clips is the known
   thin-data / label-noise problem (TRAINING_NOTES); seg_ind lands after the
   seg_dir floor + learning curve are read and the data source is scaled.

## Consequences

- A running seg_dir trainer with almost no new modeling code; the risky VLM+SAM2
  forward stays upstream and battle-tested.
- Expanding clips to frames yields ~974 clean labelled frames (not 93), so
  seg_dir is not data-starved even before scaling sources.
- Divergence from the spec's single-forward dual-token architecture is **explicit
  and temporary** — recorded here rather than silently. Moving to change A is a
  follow-up ADR when seg_ind is built.
- The polygon path caps label fidelity at the gate threshold (~0.85–0.95 IoU);
  acceptable because Sa2VA was itself pretrained on polygon GT. RLE (lossless)
  would need editing the vendored dataset (B4) — not done.
