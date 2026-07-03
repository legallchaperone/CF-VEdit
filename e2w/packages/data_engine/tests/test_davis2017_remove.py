import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from e2w_data_engine import davis2017_remove as davis


class Davis2017RemoveTest(unittest.TestCase):
    def test_object_colors_ignore_black_and_are_bgr(self):
        mask = np.zeros((3, 4, 3), dtype=np.uint8)
        mask[0, 0] = (0, 0, 128)
        mask[1, 1] = (0, 128, 0)

        self.assertEqual(davis.object_colors_bgr(mask), [(0, 0, 128), (0, 128, 0)])

    def test_quadmask_direct_wins_and_uses_canonical_values(self):
        direct = np.array([[[False, True], [False, True]]])
        indirect = np.array([[[False, False], [True, True]]])

        quadmask = davis.three_layer_to_quadmask(direct, indirect)

        self.assertEqual(quadmask.dtype, np.uint8)
        self.assertEqual(quadmask.tolist(), [[[255, 0], [127, 0]]])

    def test_validate_rejects_bad_quadmask_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "masks").mkdir()
            (out / "quadmasks").mkdir()
            direct = np.zeros((1, 2, 2), dtype=bool)
            direct[0, 0, 0] = True
            indirect = np.zeros_like(direct)
            quadmask = np.array([[[0, 99], [127, 255]]], dtype=np.uint8)
            np.save(out / "masks" / "toy_direct.npy", direct)
            np.save(out / "masks" / "toy_indirect.npy", indirect)
            np.save(out / "quadmasks" / "toy_quadmask.npy", quadmask)
            row = {
                "sample_id": "toy",
                "split": "train",
                "sequence": "toy",
                "operation": "remove",
                "instruction": "remove the toy",
                "target_ref": "toy",
                "direct_mask_npy": "masks/toy_direct.npy",
                "indirect_mask_npy": "masks/toy_indirect.npy",
                "quadmask_npy": "quadmasks/toy_quadmask.npy",
                "post_removal_description": "an empty scene",
            }
            (out / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (out / "quarantine.jsonl").write_text("", encoding="utf-8")

            errors, warnings = davis.validate_out_root(out)

        self.assertTrue(any("quadmask values" in error for error in errors), errors)
        self.assertEqual(warnings, [])

    def test_build_skip_vlm_quarantines_missing_scene_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            _write_synthetic_davis(davis_root)

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--skip-vlm",
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            quarantine = _read_jsonl(out / "quarantine.jsonl")
            config = json.loads((out / "void_reasoner" / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest, [])
            self.assertEqual(len(quarantine), 1)
            row = quarantine[0]
            self.assertEqual(row["sample_id"], "davis2017_train_toy_obj000")
            self.assertEqual(row["difficulty_class"], "")
            self.assertEqual(row["operation"], "remove")
            self.assertNotIn("edit_prompt", row)
            self.assertEqual(row["post_removal_description"], "")
            self.assertIn("missing_scene_description", row["quarantine_reasons"])
            self.assertTrue((out / row["direct_mask_npy"]).exists())
            self.assertTrue((out / row["quadmask_npy"]).exists())
            self.assertEqual(len(config), 1)
            self.assertEqual(config[0]["instruction"], "remove the toy")


def _write_synthetic_davis(root: Path) -> None:
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "train.txt").write_text("toy\n", encoding="utf-8")
    (root / "ImageSets" / "2017" / "val.txt").write_text("", encoding="utf-8")
    frame_dir = root / "JPEGImages" / "480p" / "toy"
    ann_dir = root / "Annotations" / "480p" / "toy"
    video_dir = root / "preview_videos"
    frame_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    frames = []
    for idx in range(2):
        frame = np.full((12, 16, 3), 80 + idx * 20, dtype=np.uint8)
        frame[2:6, 3:7] = (10, 20, 200)
        frames.append(frame)
        cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), frame)

        ann = np.zeros((12, 16, 3), dtype=np.uint8)
        ann[2:6, 3:7] = (0, 0, 128)
        cv2.imwrite(str(ann_dir / f"{idx:05d}.png"), ann)

    writer = cv2.VideoWriter(
        str(video_dir / "toy.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12,
        (16, 12),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
