"""Build DAVIS2017 remove-only training rows for E2W/Sa2VA.

The builder uses DAVIS instance masks as direct-mask ground truth and the VOID
VLM-MASK-REASONER only for after-removal text plus weak indirect masks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PYTHON = "/data/cwx/conda/envs/edit2world-phase1-real/bin/python"
DEFAULT_STAGE2 = "/data/cwx/void-model/VLM-MASK-REASONER/stage2_vlm_analysis_cf.py"
DEFAULT_STAGE3A = "/data/cwx/void-model/VLM-MASK-REASONER/stage3a_generate_grey_masks.py"
CANONICAL_QUADMASK_VALUES = {0, 127, 255}


@dataclass(frozen=True)
class SampleWork:
    sample_id: str
    sequence: str
    obj_idx: int
    object_color_bgr: tuple[int, int, int]
    target_ref: str
    instruction: str
    direct_mask_path: Path
    workdir: Path


def object_colors_bgr(mask: Any) -> list[tuple[int, int, int]]:
    """Return non-black DAVIS instance colors from a BGR annotation image."""
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"annotation mask must be HxWx3 or HxW, got {arr.shape}")
    colors = np.unique(arr[..., :3].reshape(-1, 3), axis=0)
    out = []
    for color in colors.tolist():
        bgr = tuple(int(x) for x in color)
        if bgr != (0, 0, 0):
            out.append(bgr)
    return sorted(out)


def three_layer_to_quadmask(direct: Any, indirect: Any):
    """Convert boolean direct/indirect stacks to E2W's canonical quadmask file."""
    import numpy as np

    direct = np.asarray(direct).astype(bool)
    indirect = np.asarray(indirect).astype(bool)
    if direct.shape != indirect.shape:
        raise ValueError(f"direct {direct.shape} != indirect {indirect.shape}")
    out = np.full(direct.shape, 255, dtype=np.uint8)
    out[np.logical_and(indirect, np.logical_not(direct))] = 127
    out[direct] = 0
    return out


def validate_out_root(out_root: str | Path) -> tuple[list[str], list[str]]:
    out_root = Path(out_root)
    errors: list[str] = []
    warnings: list[str] = []
    for manifest_name, relaxed in (("manifest.jsonl", False), ("quarantine.jsonl", True)):
        for row in _read_jsonl(out_root / manifest_name):
            _validate_row(out_root, row, relaxed=relaxed, errors=errors, warnings=warnings)
    return errors, warnings


def build(args: argparse.Namespace) -> int:
    davis_root = Path(args.davis_root).resolve()
    out_root = Path(args.out_root).resolve()
    if args.split != "train":
        raise ValueError("DAVIS builder currently only writes ImageSets/2017/train.txt")

    sequences = _read_split(davis_root, args.split)
    if args.limit is not None:
        sequences = sequences[: int(args.limit)]

    _ensure_dirs(out_root)
    samples: list[SampleWork] = []
    config_rows: list[dict[str, str]] = []

    for sequence in sequences:
        ann_paths = sorted((davis_root / "Annotations" / "480p" / sequence).glob("*.png"))
        frame_paths = sorted((davis_root / "JPEGImages" / "480p" / sequence).glob("*.jpg"))
        if not ann_paths:
            raise FileNotFoundError(f"no DAVIS annotations for {sequence}")
        colors = _sequence_colors(ann_paths)
        _link_or_copy(davis_root / "preview_videos" / f"{sequence}.mp4",
                      out_root / "videos" / f"{sequence}.mp4")
        _link_or_copy(davis_root / "JPEGImages" / "480p" / sequence,
                      out_root / "frames" / sequence)

        for obj_idx, color in enumerate(colors):
            sample = _prepare_sample(
                davis_root=davis_root,
                out_root=out_root,
                split=args.split,
                sequence=sequence,
                obj_idx=obj_idx,
                object_count=len(colors),
                color=color,
                ann_paths=ann_paths,
                frame_paths=frame_paths,
            )
            samples.append(sample)
            config_rows.append({
                "video_path": _rel(sample.workdir / "input_video.mp4", out_root),
                "output_dir": _rel(sample.workdir, out_root),
                "instruction": sample.instruction,
            })

    config_path = out_root / "void_reasoner" / "config.json"
    _write_json(config_path, config_rows)

    if not args.skip_vlm:
        if args.overwrite_vlm or not all((s.workdir / "vlm_analysis.json").exists() for s in samples):
            _run([args.python_bin, args.stage2_script, "--config", str(config_path)], cwd=out_root)
        _run([
            args.python_bin,
            args.stage3a_script,
            "--config", str(config_path),
            "--segmentation-model", args.segmentation_model,
        ], cwd=out_root)

    manifest_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = _finalize_sample(out_root, args.split, sample)
        if row["quarantine_reasons"]:
            quarantine_rows.append(row)
        else:
            row.pop("quarantine_reasons", None)
            manifest_rows.append(row)

    _write_jsonl(out_root / "manifest.jsonl", manifest_rows)
    _write_jsonl(out_root / "quarantine.jsonl", quarantine_rows)
    _write_json(out_root / "summary.json", {
        "split": args.split,
        "sequences": len(sequences),
        "samples": len(samples),
        "accepted": len(manifest_rows),
        "quarantined": len(quarantine_rows),
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/validate DAVIS2017 remove data for E2W")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--davis-root", required=True)
    b.add_argument("--split", default="train")
    b.add_argument("--out-root", required=True)
    b.add_argument("--limit", type=int)
    b.add_argument("--skip-vlm", action="store_true")
    b.add_argument("--overwrite-vlm", action="store_true")
    b.add_argument("--python-bin", default=DEFAULT_PYTHON)
    b.add_argument("--stage2-script", default=DEFAULT_STAGE2)
    b.add_argument("--stage3a-script", default=DEFAULT_STAGE3A)
    b.add_argument("--segmentation-model", default="langsam")

    v = sub.add_parser("validate")
    v.add_argument("--out-root", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "build":
        return build(args)
    if args.cmd == "validate":
        errors, warnings = validate_out_root(args.out_root)
        for warning in warnings:
            print(f"WARN: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        return 1 if errors else 0
    raise AssertionError(args.cmd)


def _prepare_sample(
    *,
    davis_root: Path,
    out_root: Path,
    split: str,
    sequence: str,
    obj_idx: int,
    object_count: int,
    color: tuple[int, int, int],
    ann_paths: list[Path],
    frame_paths: list[Path],
) -> SampleWork:
    import cv2
    import numpy as np

    sample_id = f"davis2017_{split}_{sequence}_obj{obj_idx:03d}"
    target_ref = _target_ref(sequence, obj_idx, object_count)
    instruction = f"remove the {target_ref}"
    direct = []
    for path in ann_paths:
        ann = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if ann is None:
            raise ValueError(f"failed to read annotation {path}")
        if ann.ndim == 2:
            ann = np.repeat(ann[..., None], 3, axis=2)
        direct.append(np.all(ann[..., :3] == np.asarray(color, dtype=np.uint8), axis=2))
    direct_stack = np.stack(direct).astype(bool)

    direct_path = out_root / "masks" / f"{sequence}_obj{obj_idx:03d}_direct.npy"
    np.save(direct_path, direct_stack)

    workdir = out_root / "void_reasoner" / sample_id
    workdir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(out_root / "videos" / f"{sequence}.mp4", workdir / "input_video.mp4")
    _write_mask_video(direct_stack, workdir / "black_mask.mp4")
    if frame_paths:
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    else:
        first = _read_first_video_frame(workdir / "input_video.mp4")
    if first is None:
        raise ValueError(f"failed to read first frame for {sequence}")
    cv2.imwrite(str(workdir / "first_frame.jpg"), first)
    cv2.imwrite(str(workdir / "first_frame_overlay.jpg"), _overlay_direct(first, direct_stack[0]))
    _write_grid_frames(frame_paths, workdir)

    return SampleWork(
        sample_id=sample_id,
        sequence=sequence,
        obj_idx=obj_idx,
        object_color_bgr=color,
        target_ref=target_ref,
        instruction=instruction,
        direct_mask_path=direct_path,
        workdir=workdir,
    )


def _finalize_sample(out_root: Path, split: str, sample: SampleWork) -> dict[str, Any]:
    import numpy as np

    direct = np.load(sample.direct_mask_path, allow_pickle=False).astype(bool)
    analysis = _read_vlm_analysis(sample.workdir / "vlm_analysis.json")
    post_desc = str(analysis.get("scene_description") or "").strip()
    if post_desc:
        _write_json(sample.workdir / "prompt.json", {"bg": post_desc})

    indirect = _read_grey_mask_or_empty(sample.workdir / "grey_mask.mp4", direct.shape)
    indirect_path = out_root / "masks" / f"{sample.sequence}_obj{sample.obj_idx:03d}_indirect.npy"
    quadmask_path = out_root / "quadmasks" / f"{sample.sequence}_obj{sample.obj_idx:03d}_quadmask.npy"
    np.save(indirect_path, indirect)
    np.save(quadmask_path, three_layer_to_quadmask(direct, indirect))

    reasons = []
    if not direct.any():
        reasons.append("empty_direct_mask")
    if not post_desc:
        reasons.append("missing_scene_description")

    return {
        "sample_id": sample.sample_id,
        "split": split,
        "sequence": sample.sequence,
        "operation": "remove",
        "instruction": sample.instruction,
        "target_ref": sample.target_ref,
        "object_color_bgr": list(sample.object_color_bgr),
        "source_video": f"videos/{sample.sequence}.mp4",
        "frames_dir": f"frames/{sample.sequence}",
        "direct_mask_npy": _rel(sample.direct_mask_path, out_root),
        "indirect_mask_npy": _rel(indirect_path, out_root),
        "quadmask_npy": _rel(quadmask_path, out_root),
        "post_removal_description": post_desc,
        "difficulty_class": "",
        "vlm_analysis_json": _rel(sample.workdir / "vlm_analysis.json", out_root),
        "label_quality": {
            "direct": "davis_gt",
            "indirect": "void_vlm_weak",
            "text_condition": "void_bg",
        },
        "quarantine_reasons": reasons,
    }


def _validate_row(
    out_root: Path,
    row: dict[str, Any],
    *,
    relaxed: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    import numpy as np

    sid = row.get("sample_id", "<missing>")
    try:
        direct = np.load(out_root / row["direct_mask_npy"], allow_pickle=False)
        indirect = np.load(out_root / row["indirect_mask_npy"], allow_pickle=False)
        quadmask = np.load(out_root / row["quadmask_npy"], allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - validator reports all row issues.
        errors.append(f"{sid}: failed to load masks: {exc}")
        return

    if direct.shape != indirect.shape or direct.shape != quadmask.shape:
        errors.append(f"{sid}: mask shape mismatch direct={direct.shape} indirect={indirect.shape} quadmask={quadmask.shape}")
    values = set(int(x) for x in np.unique(quadmask).tolist())
    if not values <= CANONICAL_QUADMASK_VALUES:
        errors.append(f"{sid}: quadmask values {sorted(values)} not subset of {sorted(CANONICAL_QUADMASK_VALUES)}")
    if row.get("split") != "train":
        errors.append(f"{sid}: split must be train, got {row.get('split')!r}")
    if not relaxed and not np.asarray(direct).astype(bool).any():
        errors.append(f"{sid}: empty direct mask")
    desc = str(row.get("post_removal_description") or "").strip()
    if not relaxed and not desc:
        errors.append(f"{sid}: missing post_removal_description")
    target_ref = str(row.get("target_ref") or "").strip().casefold()
    if desc and target_ref and target_ref in desc.casefold():
        warnings.append(f"{sid}: post_removal_description contains target_ref {target_ref!r}")


def _read_split(davis_root: Path, split: str) -> list[str]:
    path = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sequence_colors(ann_paths: list[Path]) -> list[tuple[int, int, int]]:
    import cv2

    colors: set[tuple[int, int, int]] = set()
    for path in ann_paths:
        ann = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if ann is None:
            raise ValueError(f"failed to read annotation {path}")
        colors.update(object_colors_bgr(ann))
    return sorted(colors)


def _target_ref(sequence: str, obj_idx: int, object_count: int) -> str:
    if object_count == 1:
        return sequence.replace("-", " ")
    return "highlighted object"


def _ensure_dirs(out_root: Path) -> None:
    for rel in ("videos", "frames", "masks", "quadmasks", "void_reasoner"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _write_mask_video(direct: Any, out_path: Path, *, fps: int = 12) -> None:
    import cv2
    import numpy as np

    direct = np.asarray(direct).astype(bool)
    h, w = direct.shape[1:]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), True)
    if not writer.isOpened():
        raise ValueError(f"failed to open mask video writer: {out_path}")
    for frame in direct:
        gray = np.where(frame, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


def _read_first_video_frame(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _overlay_direct(frame: Any, direct: Any):
    import numpy as np

    out = np.asarray(frame).copy()
    mask = np.asarray(direct).astype(bool)
    red = np.zeros_like(out)
    red[..., 2] = 255
    out[mask] = (0.55 * out[mask] + 0.45 * red[mask]).astype(np.uint8)
    return out


def _write_grid_frames(frame_paths: list[Path], workdir: Path, *, count: int = 4) -> None:
    import cv2
    import numpy as np

    if not frame_paths:
        return
    idxs = np.linspace(0, len(frame_paths) - 1, min(count, len(frame_paths))).round().astype(int)
    for idx in idxs.tolist():
        frame = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(str(workdir / f"grid_sample_frame_{idx:05d}.jpg"), frame)


def _read_grey_mask_or_empty(path: Path, shape: tuple[int, int, int]):
    import cv2
    import numpy as np

    empty = np.zeros(shape, dtype=bool)
    if not path.exists():
        return empty
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(np.abs(gray.astype(np.int16) - 127) <= 32)
    cap.release()
    if len(frames) != shape[0]:
        return empty
    arr = np.stack(frames).astype(bool)
    return arr if arr.shape == shape else empty


def _read_vlm_analysis(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    path = path if path.is_absolute() else path.absolute()
    root = root if root.is_absolute() else root.absolute()
    return path.relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
