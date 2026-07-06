"""Held-out seg_dir eval — IoU/Dice of predicted masks vs DAVIS GT.

Runs on the *inference* path (reuses `CausalPlanner`), so it needs only a
Sa2VA-runnable env (no xtuner/deepspeed). Point `--weights` at the stock Sa2VA
checkpoint for the zero-shot baseline, or at the finetuned checkpoint
(convert_to_hf) after training — same command, so it doubles as the before/after
metric the spec asks for (Stage-0 held-out IoU/Dice).

Reads `val.jsonl` written by `training.data` (held-out clips, no leakage — split
by sequence). Reports mean IoU/Dice + the two collapse diagnostics from
TRAINING_NOTES: empty-prediction rate, and (when an indirect GT is available) the
seg_ind≈seg_dir overlap — here we log empty-rate for seg_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iou_dice(pred, gt):
    import numpy as np
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    psum = p.sum() + g.sum()
    iou = float(inter / union) if union else 1.0
    dice = float(2 * inter / psum) if psum else 1.0
    return iou, dice, bool(p.sum() == 0)


def evaluate(val_jsonl: str, out_root: str, weights: str, *, device: str = "cuda:0",
             max_frames: int = 21) -> dict:
    import numpy as np
    from PIL import Image
    from e2w_localization.planner import CausalPlanner, PlannerConfig

    out = Path(out_root)
    clips = [json.loads(l) for l in Path(val_jsonl).read_text().splitlines() if l.strip()]
    planner = CausalPlanner(PlannerConfig(weights_path=weights, device=device,
                                          max_frames_for_segmentation=max_frames))

    per_clip, empties = [], 0
    for c in clips:
        frame_paths = sorted((out / c["frames_dir"]).glob("*.jpg"))[:max_frames]
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        gt = np.load(out / c["mask_npy"])[:len(frames)]
        try:
            pred = planner._predict_stock_seg_mask(frames, c["target_ref"])  # (T,H,W) bool
        except Exception as e:
            per_clip.append({"sample_id": c["sample_id"], "iou": 0.0, "dice": 0.0,
                             "error": str(e)[:120]})
            empties += 1
            continue
        T = min(len(gt), pred.shape[0])
        ious = [_iou_dice(pred[t], gt[t]) for t in range(T)]
        iou = float(np.mean([x[0] for x in ious]))
        dice = float(np.mean([x[1] for x in ious]))
        empty_frac = float(np.mean([x[2] for x in ious]))
        per_clip.append({"sample_id": c["sample_id"], "target_ref": c["target_ref"],
                         "iou": iou, "dice": dice, "empty_frac": empty_frac})
        empties += empty_frac >= 0.5

    # A planner failure means seg_dir produced no mask -> a total miss (IoU 0),
    # so failed clips already carry iou/dice = 0. The HEADLINE metric averages
    # over ALL clips (failure = 0 in the denominator) so a run that fails on many
    # clips cannot report a high mean off the few that succeeded. valid-only is
    # kept as a secondary diagnostic.
    valid = [c for c in per_clip if "error" not in c]
    failed = len(per_clip) - len(valid)
    n_all = max(1, len(per_clip))
    n_valid = max(1, len(valid))
    summary = {
        "weights": weights,
        "clips": len(clips),
        "failed_clips": int(failed),
        "empty_or_failed_clips": int(empties),
        "mean_iou": float(sum(c["iou"] for c in per_clip) / n_all),
        "mean_dice": float(sum(c["dice"] for c in per_clip) / n_all),
        "mean_iou_valid_only": float(sum(c["iou"] for c in valid) / n_valid),
        "mean_dice_valid_only": float(sum(c["dice"] for c in valid) / n_valid),
    }
    return {"summary": summary, "per_clip": per_clip}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Held-out seg_dir IoU/Dice vs DAVIS GT")
    p.add_argument("--val-jsonl", required=True, help="val.jsonl from training.data")
    p.add_argument("--out-root", required=True, help="data_engine out-root (for GT npy + frames)")
    p.add_argument("--weights", required=True, help="Sa2VA checkpoint (stock or finetuned-HF)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report", help="optional path to write full per-clip json")
    args = p.parse_args(argv)
    res = evaluate(args.val_jsonl, args.out_root, args.weights, device=args.device)
    print(json.dumps(res["summary"], indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
