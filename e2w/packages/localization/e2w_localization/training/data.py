"""Adapt the data_engine DAVIS/VOID manifest into Sa2VA's finetune format.

Stage-0 `seg_dir` is trained as stock Sa2VA referring segmentation ("segment
<target_ref>" -> direct mask) via the vendored XTuner trainer — for a single
binary mask this is exactly the pretrained `[SEG]` single-token case (scout /
ADR-0009), so we reuse `Sa2VAFinetuneDataset` rather than the untrained
dual-token path. This module is the only new data logic; the trainer, loss
(2*BCE + 0.5*Dice), LoRA and SAM2 injection are all upstream.

Input:  a data_engine out-root with `manifest.jsonl` + `frames/<seq>/*.jpg` +
        `masks/<sample>_{direct,indirect}.npy` (T,H,W bool, frame-aligned).
Output: `<out>/images/*.jpg` (sampled frames) + `<out>/annotations.json`
        (train, Sa2VA schema) + `<out>/val.jsonl` (held-out, for our IoU eval,
        which reads GT npy directly — not the trainer format).

Sa2VA finetune item schema (sa2va_data_finetune.py:68-107):
    {"image": "<rel>.jpg", "mask": [[ [x1,y1,...], ... ]], "text": ["<phrase>"]}
`mask[i]` is a list of COCO polygons for object i; `text[i]` is its phrase.
We emit one object per image (the direct — or indirect — GT).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterConfig:
    out_root: str                 # data_engine out-root (has manifest.jsonl)
    dst: str                      # where to write images/ + annotations.json
    layer: str = "direct"         # "direct" (seg_dir) or "indirect" (seg_ind)
    frame_stride: int = 5         # sample every Nth frame (adjacent frames redundant)
    min_area_px: int = 12         # drop contours smaller than this (poly noise)
    val_fraction: float = 0.15    # held-out fraction, split by SEQUENCE (no leak)
    max_val_sequences: int = 12
    require_nonempty: bool = True  # skip frames whose GT mask is empty
    min_fidelity: float = 0.85    # drop frames whose polygon reconstruction IoU
                                  # vs the source mask is below this (holes / thin
                                  # shapes RETR_EXTERNAL can't represent) — keeps
                                  # the training labels faithful, not corrupted.


def _mask_to_polygons(mask_2d, min_area_px: int) -> list[list[float]]:
    """Binary (H,W) -> list of COCO polygons [x1,y1,x2,y2,...] (>=3 points).

    Uses external contours; Sa2VA's RefCOCO GT is polygon-shaped, so this matches
    the pretrained supervision distribution. Coarse/blocky indirect masks convert
    fine (they're already low-frequency).
    """
    import cv2
    import numpy as np

    m = (np.asarray(mask_2d) > 0).astype(np.uint8)
    # CHAIN_APPROX_NONE keeps every boundary point -> the polygon rasterizes back
    # to ~the original mask (Sa2VA rasterizes GT the same way via frPyObjects), so
    # we preserve label fidelity instead of the lossy SIMPLE corner-only approx.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        polys.append([float(v) for v in pts.flatten()])
    return polys


def _polygon_fidelity(polys: list[list[float]], mask_2d) -> float:
    """IoU of the polygons rasterized back (the way Sa2VA will) vs the source mask.

    Uses the exact decode path the finetune dataset uses (frPyObjects -> decode),
    so this measures the label the trainer would actually see.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    gt = np.asarray(mask_2d) > 0
    h, w = gt.shape
    b = np.zeros((h, w), dtype=np.uint8)
    for seg in polys:
        b += mask_utils.decode(mask_utils.frPyObjects([seg], h, w)).squeeze().astype(np.uint8)
    b = b > 0
    union = (b | gt).sum()
    return float((b & gt).sum() / union) if union else 1.0


def _split_sequences(sequences: list[str], val_fraction: float, max_val: int) -> set[str]:
    """Deterministic held-out sequence set (sorted, take a stride — no RNG so the
    split is reproducible across runs and reviewable)."""
    seqs = sorted(set(sequences))
    if len(seqs) < 2:
        # Need >=2 sequences to hold one out without emptying the train split;
        # with 1 sequence, keep everything for training (no held-out eval).
        return set()
    n_val = min(max_val, max(1, round(len(seqs) * val_fraction)))
    if n_val >= len(seqs):
        n_val = max(1, len(seqs) // 5)
    stride = max(1, len(seqs) // n_val)
    return set(seqs[::stride][:n_val])


def build(cfg: AdapterConfig) -> dict:
    import numpy as np
    from PIL import Image

    out_root = Path(cfg.out_root)
    dst = Path(cfg.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (out_root / "manifest.jsonl").read_text().splitlines() if l.strip()]

    mask_field = "direct_mask_npy" if cfg.layer == "direct" else "indirect_mask_npy"
    val_seqs = _split_sequences([r["sequence"] for r in rows], cfg.val_fraction, cfg.max_val_sequences)

    train_items: list[dict] = []
    val_records: list[dict] = []
    stats = {"train_images": 0, "val_clips": 0, "skipped_empty_frames": 0,
             "skipped_no_polygon": 0, "skipped_low_fidelity": 0, "rows": len(rows)}

    for r in rows:
        sid, seq, phrase = r["sample_id"], r["sequence"], r["target_ref"]
        if not phrase:
            continue
        mask = np.load(out_root / r[mask_field])          # (T,H,W) bool
        frame_paths = sorted((out_root / r["frames_dir"]).glob("*.jpg"))
        T = min(len(frame_paths), mask.shape[0])
        is_val = seq in val_seqs

        if is_val:
            # Held-out: record clip-level pointer for our own IoU eval (reads npy).
            val_records.append({"sample_id": sid, "sequence": seq, "target_ref": phrase,
                                "frames_dir": r["frames_dir"], "mask_npy": r[mask_field],
                                "num_frames": T})
            stats["val_clips"] += 1
            continue

        for t in range(0, T, cfg.frame_stride):
            m = mask[t]
            if cfg.require_nonempty and m.sum() == 0:
                stats["skipped_empty_frames"] += 1
                continue
            polys = _mask_to_polygons(m, cfg.min_area_px)
            if not polys:
                stats["skipped_no_polygon"] += 1
                continue
            if _polygon_fidelity(polys, m) < cfg.min_fidelity:
                stats["skipped_low_fidelity"] += 1
                continue
            img_name = f"{sid}_f{t:05d}.jpg"
            # copy the exact training frame (symlink-safe: re-encode via PIL load)
            Image.open(frame_paths[t]).convert("RGB").save(dst / "images" / img_name)
            train_items.append({"image": img_name, "mask": [polys], "text": [phrase]})
            stats["train_images"] += 1

    if not train_items:
        # Fail loud instead of writing an empty annotations.json the Sa2VA
        # trainer would only choke on much later ("no samples"). Covers: <2
        # sequences that still yielded nothing, all-empty masks, or every frame
        # dropped by the fidelity gate.
        raise ValueError(
            f"adapter produced 0 training frames from {stats['rows']} rows "
            f"(val_sequences={sorted(val_seqs)}, stats={stats}). Check that "
            f"'{mask_field}' masks are non-empty and pass the fidelity gate, and "
            "that the manifest has >=2 sequences if you want a held-out split.")

    (dst / "annotations.json").write_text(json.dumps(train_items))
    with (dst / "val.jsonl").open("w") as f:
        for v in val_records:
            f.write(json.dumps(v) + "\n")
    meta = {"config": asdict(cfg), "val_sequences": sorted(val_seqs), "stats": stats}
    (dst / "adapter_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DAVIS/VOID manifest -> Sa2VA finetune data")
    p.add_argument("--out-root", required=True, help="data_engine out-root (has manifest.jsonl)")
    p.add_argument("--dst", required=True, help="destination dir for images/ + annotations.json")
    p.add_argument("--layer", default="direct", choices=["direct", "indirect"])
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    args = p.parse_args(argv)
    meta = build(AdapterConfig(out_root=args.out_root, dst=args.dst, layer=args.layer,
                               frame_stride=args.frame_stride, val_fraction=args.val_fraction))
    print(json.dumps(meta["stats"], indent=2))
    print("val sequences:", meta["val_sequences"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
