"""Unit tests for the Stage-0 seg_dir data adapter (training.data).

Run (needs numpy/cv2/PIL/pycocotools — the void or edit2world env):
    cd e2w/packages/localization
    PYTHONPATH=$(pwd) python -m unittest tests.test_training_data
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from e2w_localization.training import data as adapter  # noqa: E402


def _write_clip(root: Path, seq: str, sample: str, *, frames: int, box, empty_frames=()):
    """A synthetic clip: solid-rectangle direct mask (poly-friendly) + jpg frames."""
    fdir = root / "frames" / seq
    fdir.mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    h, w = 64, 96
    mask = np.zeros((frames, h, w), dtype=bool)
    y0, x0, y1, x1 = box
    for t in range(frames):
        Image.fromarray(np.full((h, w, 3), 100, np.uint8)).save(fdir / f"{t:05d}.jpg")
        if t not in empty_frames:
            mask[t, y0:y1, x0:x1] = True
    np.save(root / "masks" / f"{sample}_direct.npy", mask)
    return {
        "sample_id": sample, "sequence": seq, "target_ref": "the box",
        "instruction": "remove the box", "operation": "remove",
        "frames_dir": f"frames/{seq}", "direct_mask_npy": f"masks/{sample}_direct.npy",
        "indirect_mask_npy": f"masks/{sample}_direct.npy",
        "label_quality": {"direct": "davis_gt"},
    }


class TrainingDataAdapterTest(unittest.TestCase):
    def _build(self, tmp, **kw):
        root = Path(tmp) / "out"
        (root).mkdir()
        rows = [
            _write_clip(root, "seqA", "seqA_obj000", frames=10, box=(10, 10, 40, 50),
                        empty_frames=(0,)),
            _write_clip(root, "seqB", "seqB_obj000", frames=10, box=(20, 30, 55, 80)),
        ]
        (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        dst = Path(tmp) / "dst"
        cfg = adapter.AdapterConfig(out_root=str(root), dst=str(dst), layer="direct",
                                    frame_stride=1, val_fraction=0.5, max_val_sequences=1, **kw)
        meta = adapter.build(cfg)
        return root, dst, meta

    def test_emits_wellformed_sa2va_items_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, dst, meta = self._build(tmp)
            items = json.loads((dst / "annotations.json").read_text())
            self.assertGreater(len(items), 0)
            it = items[0]
            self.assertEqual(set(it), {"image", "mask", "text"})
            self.assertEqual(it["text"], ["the box"])
            self.assertEqual(len(it["mask"]), 1)                 # one object
            self.assertTrue((dst / "images" / it["image"]).exists())

    def test_val_split_is_by_sequence_no_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, dst, meta = self._build(tmp)
            val_seqs = set(meta["val_sequences"])
            self.assertEqual(len(val_seqs), 1)                    # exactly one held-out seq
            train_imgs = [i["image"] for i in json.loads((dst / "annotations.json").read_text())]
            for img in train_imgs:                                # no val-seq frame in train
                self.assertFalse(any(img.startswith(v) for v in val_seqs))
            val = [json.loads(l) for l in (dst / "val.jsonl").read_text().splitlines()]
            self.assertTrue(all(v["sequence"] in val_seqs for v in val))

    def test_empty_frame_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, dst, meta = self._build(tmp)
            # seqA frame 0 is empty; if seqA is the train seq it must be skipped
            self.assertGreaterEqual(meta["stats"]["skipped_empty_frames"], 0)
            for it in json.loads((dst / "annotations.json").read_text()):
                self.assertFalse(it["image"].endswith("_f00000.jpg")
                                 and it["image"].startswith("seqA"))

    def test_polygon_fidelity_high_for_solid_rectangle(self):
        mask = np.zeros((64, 96), bool)
        mask[10:40, 10:50] = True
        polys = adapter._mask_to_polygons(mask, min_area_px=4)
        self.assertTrue(polys)
        # ~1px contour->raster boundary offset caps a small rectangle near 0.94;
        # real (larger) DAVIS masks sit ~0.95. Well above the 0.85 gate either way.
        self.assertGreaterEqual(adapter._polygon_fidelity(polys, mask), 0.90)

    def test_fidelity_gate_drops_ring_mask(self):
        # a ring (hole) round-trips badly under RETR_EXTERNAL -> gate must catch it
        ring = np.zeros((64, 64), bool)
        ring[8:56, 8:56] = True
        ring[20:44, 20:44] = False
        polys = adapter._mask_to_polygons(ring, min_area_px=4)
        self.assertLess(adapter._polygon_fidelity(polys, ring), 0.85)


if __name__ == "__main__":
    unittest.main()
