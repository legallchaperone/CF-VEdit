import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from e2w_data_engine import davis2017_remove as davis


class Davis2017RemoveTest(unittest.TestCase):
    def test_object_colors_read_palette_indices_and_ignore_void(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.png"
            _write_palette_png(path, np.array([[0, 1, 2, 255]], dtype=np.uint8))

            self.assertEqual(davis.object_colors_bgr(path), [(0, 0, 128), (0, 128, 0)])

    def test_palette_reader_accepts_one_bit_palette_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.png"
            _write_palette_png(path, np.array([[0, 1, 0, 1]], dtype=np.uint8), bits=1)

            self.assertEqual(davis.object_colors_bgr(path), [(0, 0, 128)])

    def test_quadmask_direct_wins_and_uses_canonical_values(self):
        direct = np.array([[[False, True], [False, True]]])
        indirect = np.array([[[False, False], [True, True]]])

        quadmask = davis.three_layer_to_quadmask(direct, indirect)

        self.assertEqual(quadmask.dtype, np.uint8)
        self.assertEqual(quadmask.tolist(), [[[255, 0], [127, 0]]])

    def test_black_mask_video_preserves_exact_zero_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            direct = np.zeros((2, 12, 16), dtype=bool)
            direct[:, 2:6, 3:8] = True
            path = Path(tmp) / "black_mask.mp4"

            davis._write_mask_video(direct, path)

            cap = cv2.VideoCapture(str(path))
            zero_counts = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                zero_counts.append(int((gray == 0).sum()))
            cap.release()

        self.assertEqual(zero_counts, [20, 20])

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
            _write_synthetic_davis(davis_root, objects=1)

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

    def test_two_objects_void_excluded_and_other_object_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            names = Path(tmp) / "names.json"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="clean")
            _write_synthetic_davis(davis_root, objects=2, include_void=True)
            names.write_text(json.dumps({"toy": ["person", "bike"]}), encoding="utf-8")

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--object-names-json", str(names),
                "--python-bin", sys.executable,
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            quarantine = _read_jsonl(out / "quarantine.jsonl")
            self.assertEqual(quarantine, [])
            self.assertEqual([row["target_ref"] for row in manifest], ["person", "bike"])
            self.assertEqual([row["object_ids"] for row in manifest], [[1], [2]])

            person = manifest[0]
            quadmask = np.load(out / person["quadmask_npy"], allow_pickle=False)
            self.assertEqual(int(quadmask[0, 7, 10]), 255)

    def test_vlm_namer_makes_multi_object_sequence_clean_without_manual_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="clean")
            namer = _write_namer_script(Path(tmp), mode="names")
            _write_synthetic_davis(davis_root, objects=2)

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--python-bin", sys.executable,
                "--name-objects-script", str(namer),
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            self.assertEqual([row["target_ref"] for row in manifest], ["person", "bike"])
            self.assertTrue(all(row["label_quality"]["target_ref"] == "vlm" for row in manifest))
            self.assertTrue((out / "void_reasoner" / "object_names.vlm.json").exists())

    def test_vlm_namer_duplicate_names_quarantine_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="clean")
            namer = _write_namer_script(Path(tmp), mode="duplicates")
            _write_synthetic_davis(davis_root, objects=2)

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--python-bin", sys.executable,
                "--name-objects-script", str(namer),
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            self.assertEqual(_read_jsonl(out / "manifest.jsonl"), [])
            quarantine = _read_jsonl(out / "quarantine.jsonl")
            self.assertEqual(len(quarantine), 2)
            self.assertTrue(all("unresolvable_target_ref" in row["quarantine_reasons"] for row in quarantine))

    def test_vlm_name_cache_skips_namer_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="clean")
            namer = _write_namer_script(Path(tmp), mode="fail")
            _write_synthetic_davis(davis_root, objects=2)
            (out / "void_reasoner").mkdir(parents=True)
            (out / "void_reasoner" / "object_names.vlm.json").write_text(
                json.dumps({"toy": {"1": "person", "2": "bike"}}),
                encoding="utf-8",
            )

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--python-bin", sys.executable,
                "--name-objects-script", str(namer),
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            self.assertEqual([row["target_ref"] for row in _read_jsonl(out / "manifest.jsonl")], ["person", "bike"])

    def test_manual_name_beats_vlm_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            names = Path(tmp) / "names.json"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="clean")
            namer = _write_namer_script(Path(tmp), mode="fail")
            _write_synthetic_davis(davis_root, objects=2)
            names.write_text(json.dumps({"toy": ["rider", "cycle"]}), encoding="utf-8")
            (out / "void_reasoner").mkdir(parents=True)
            (out / "void_reasoner" / "object_names.vlm.json").write_text(
                json.dumps({"toy": {"1": "person", "2": "bike"}}),
                encoding="utf-8",
            )

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--object-names-json", str(names),
                "--python-bin", sys.executable,
                "--name-objects-script", str(namer),
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            self.assertEqual([row["target_ref"] for row in manifest], ["rider", "cycle"])
            self.assertTrue(all(row["label_quality"]["target_ref"] == "manual" for row in manifest))

    def test_grey_mask_frame_mismatch_quarantines_instead_of_empty_weak_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="mismatch")
            _write_synthetic_davis(davis_root, objects=1)

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--python-bin", sys.executable,
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            quarantine = _read_jsonl(out / "quarantine.jsonl")
            self.assertEqual(manifest, [])
            self.assertEqual(len(quarantine), 1)
            self.assertIn("indirect_frame_mismatch", quarantine[0]["quarantine_reasons"])
            self.assertNotEqual(quarantine[0]["label_quality"]["indirect"], "void_vlm_weak")

    def test_integral_pair_emits_merged_row_and_quarantines_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            davis_root = Path(tmp) / "DAVIS"
            out = Path(tmp) / "out"
            names = Path(tmp) / "names.json"
            stage2, stage3a = _write_stage_scripts(Path(tmp), mode="integral")
            _write_synthetic_davis(davis_root, objects=2)
            names.write_text(json.dumps({"toy": ["person", "bike"]}), encoding="utf-8")

            code = davis.main([
                "build",
                "--davis-root", str(davis_root),
                "--split", "train",
                "--out-root", str(out),
                "--object-names-json", str(names),
                "--python-bin", sys.executable,
                "--stage2-script", str(stage2),
                "--stage3a-script", str(stage3a),
                "--limit", "1",
            ])

            self.assertEqual(code, 0)
            manifest = _read_jsonl(out / "manifest.jsonl")
            quarantine = _read_jsonl(out / "quarantine.jsonl")
            self.assertEqual(len(manifest), 1)
            merged = manifest[0]
            self.assertEqual(merged["target_ref"], "person and bike")
            self.assertIn("person", merged["instruction"])
            self.assertIn("bike", merged["instruction"])
            self.assertEqual(merged["merged_from"], [
                "davis2017_train_toy_obj000",
                "davis2017_train_toy_obj001",
            ])

            direct = np.load(out / merged["direct_mask_npy"], allow_pickle=False)
            self.assertTrue(bool(direct[0, 2, 3]))
            self.assertTrue(bool(direct[0, 7, 10]))
            self.assertEqual(len(quarantine), 2)
            self.assertTrue(all("integral_pair_member" in row["quarantine_reasons"] for row in quarantine))


def _write_synthetic_davis(root: Path, *, objects: int, include_void: bool = False) -> None:
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
    for idx in range(3):
        frame = np.full((12, 16, 3), 80 + idx * 20, dtype=np.uint8)
        frame[2:6, 3:7] = (10, 20, 200)
        if objects > 1:
            frame[7:10, 10:14] = (10, 200, 20)
        frames.append(frame)
        cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), frame)

        ann = np.zeros((12, 16), dtype=np.uint8)
        ann[2:6, 3:7] = 1
        if objects > 1:
            ann[7:10, 10:14] = 2
        if include_void:
            ann[0, 0] = 255
        _write_palette_png(ann_dir / f"{idx:05d}.png", ann)

    writer = cv2.VideoWriter(
        str(video_dir / "toy.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12,
        (16, 12),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def _write_palette_png(path: Path, arr: np.ndarray, *, bits: int | None = None) -> None:
    image = Image.fromarray(arr.astype(np.uint8), mode="P")
    palette = [0, 0, 0] * 256
    palette[1 * 3:1 * 3 + 3] = [128, 0, 0]
    palette[2 * 3:2 * 3 + 3] = [0, 128, 0]
    palette[255 * 3:255 * 3 + 3] = [255, 255, 255]
    image.putpalette(palette)
    save_kwargs = {"bits": bits} if bits is not None else {}
    image.save(path, **save_kwargs)


def _write_stage_scripts(root: Path, *, mode: str) -> tuple[Path, Path]:
    stage2 = root / f"stage2_{mode}.py"
    stage3a = root / f"stage3a_{mode}.py"
    stage2.write_text(f"""
import json
import sys
from pathlib import Path

config = Path(sys.argv[sys.argv.index("--config") + 1])
for row in json.loads(config.read_text()):
    out = Path(row["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    instruction = row["instruction"]
    affected = []
    integral = []
    if {mode!r} == "mismatch":
        affected = [{{"noun": "shadow", "will_move": False, "first_appears_frame": 0}}]
    if {mode!r} == "integral" and "person" in instruction and "bike" not in instruction:
        integral = [{{"noun": "bike", "why": "bike belongs to rider"}}]
    (out / "vlm_analysis.json").write_text(json.dumps({{
        "scene_description": "the scene after removal",
        "integral_belongings": integral,
        "affected_objects": affected,
        "confidence": 1.0,
    }}))
""", encoding="utf-8")
    stage3a.write_text(f"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

config = Path(sys.argv[sys.argv.index("--config") + 1])
for row in json.loads(config.read_text()):
    out = Path(row["output_dir"])
    if {mode!r} != "mismatch":
        continue
    frame = np.full((12, 16, 3), 127, dtype=np.uint8)
    writer = cv2.VideoWriter(str(out / "grey_mask.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (16, 12))
    writer.write(frame)
    writer.release()
""", encoding="utf-8")
    return stage2, stage3a


def _write_namer_script(root: Path, *, mode: str) -> Path:
    path = root / f"namer_{mode}.py"
    path.write_text(f"""
import json
import sys
from pathlib import Path

if {mode!r} == "fail":
    raise SystemExit(42)

config = Path(sys.argv[sys.argv.index("--config") + 1])
out_json = Path(sys.argv[sys.argv.index("--out-json") + 1])
cache = json.loads(out_json.read_text()) if out_json.exists() else {{}}
for row in json.loads(config.read_text()):
    if {mode!r} == "duplicates":
        cache[row["sequence"]] = {{str(obj["object_id"]): "dog" for obj in row["objects"]}}
    else:
        names = ["person", "bike", "third object"]
        cache[row["sequence"]] = {{
            str(obj["object_id"]): names[idx]
            for idx, obj in enumerate(row["objects"])
        }}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(cache))
""", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
