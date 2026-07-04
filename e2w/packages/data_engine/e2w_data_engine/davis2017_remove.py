"""Build DAVIS2017 remove-only training rows for E2W/Sa2VA.

The builder uses DAVIS palette-index instance masks as direct-mask ground truth
and the VOID VLM-MASK-REASONER only for after-removal text plus weak indirect
masks.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PYTHON = "/data/cwx/conda/envs/edit2world-phase1-real/bin/python"
DEFAULT_STAGE2 = "/data/cwx/void-model/VLM-MASK-REASONER/stage2_vlm_analysis_cf.py"
DEFAULT_STAGE3A = "/data/cwx/void-model/VLM-MASK-REASONER/stage3a_generate_grey_masks_v2.py"
CANONICAL_QUADMASK_VALUES = {0, 127, 255}
BAD_TARGET_REFS = {"", "highlighted object"}


@dataclass(frozen=True)
class DavisObject:
    obj_idx: int
    object_id: int
    object_color_bgr: tuple[int, int, int]
    first_appears_frame: int


@dataclass(frozen=True)
class SampleWork:
    sample_id: str
    sequence: str
    obj_idx: int
    object_ids: tuple[int, ...]
    object_color_bgr: tuple[int, int, int]
    target_ref: str
    instruction: str
    direct_mask_path: Path
    workdir: Path
    merged_from: tuple[str, ...] = ()


def object_colors_bgr(annotation: str | Path | Any) -> list[tuple[int, int, int]]:
    """Return DAVIS object colors for palette indices 1..254.

    DAVIS uses palette index 255 as void/boundary; expanded BGR reads turn that
    into a fake white object, so production paths read the indexed PNG directly.
    """
    if isinstance(annotation, (str, Path)):
        arr, palette = _read_annotation_indices(Path(annotation))
        return [_palette_bgr(palette, idx) for idx in _object_indices(arr)]

    import numpy as np

    arr = np.asarray(annotation)
    if arr.ndim == 2:
        return [(idx, idx, idx) for idx in _object_indices(arr)]
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"annotation mask must be HxW or HxWx3, got {arr.shape}")
    colors = np.unique(arr[..., :3].reshape(-1, 3), axis=0)
    return sorted(
        tuple(int(x) for x in color)
        for color in colors.tolist()
        if tuple(int(x) for x in color) not in {(0, 0, 0), (255, 255, 255)}
    )


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

    object_names = _read_object_names(args.object_names_json)
    _ensure_dirs(out_root)

    samples: list[SampleWork] = []
    samples_by_sequence: dict[str, list[SampleWork]] = {}
    forced_quarantine: dict[str, set[str]] = {}

    for sequence in sequences:
        ann_paths = sorted((davis_root / "Annotations" / "480p" / sequence).glob("*.png"))
        frame_paths = sorted((davis_root / "JPEGImages" / "480p" / sequence).glob("*.jpg"))
        if not ann_paths:
            raise FileNotFoundError(f"no DAVIS annotations for {sequence}")
        objects = _sequence_objects(ann_paths)
        _link_or_copy(
            davis_root / "preview_videos" / f"{sequence}.mp4",
            out_root / "videos" / f"{sequence}.mp4",
        )
        _link_or_copy(
            davis_root / "JPEGImages" / "480p" / sequence,
            out_root / "frames" / sequence,
        )

        sequence_samples: list[SampleWork] = []
        for obj in objects:
            sample = _prepare_sample(
                out_root=out_root,
                split=args.split,
                sequence=sequence,
                object_count=len(objects),
                davis_object=obj,
                ann_paths=ann_paths,
                frame_paths=frame_paths,
                object_names=object_names,
            )
            samples.append(sample)
            sequence_samples.append(sample)
            if _bad_target_ref(sample.target_ref):
                forced_quarantine.setdefault(sample.sample_id, set()).add("unresolvable_target_ref")
        samples_by_sequence[sequence] = sequence_samples

    _write_json(out_root / "void_reasoner" / "config.json", [_config_row(s, out_root) for s in samples if s.target_ref])

    if not args.skip_vlm:
        _run_stage2(args, out_root, samples, forced_quarantine)

    merged_samples = _build_integral_merged_samples(
        out_root=out_root,
        split=args.split,
        frame_root=out_root / "frames",
        samples_by_sequence=samples_by_sequence,
        forced_quarantine=forced_quarantine,
    )
    samples.extend(merged_samples)

    if not args.skip_vlm:
        _run_stage2(args, out_root, merged_samples, forced_quarantine)
        _run_stage3a(args, out_root, samples, forced_quarantine)

    manifest_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = _finalize_sample(out_root, args.split, sample, forced_quarantine.get(sample.sample_id, set()))
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
    b.add_argument("--object-names-json")
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
    out_root: Path,
    split: str,
    sequence: str,
    object_count: int,
    davis_object: DavisObject,
    ann_paths: list[Path],
    frame_paths: list[Path],
    object_names: dict[str, Any],
) -> SampleWork:
    import numpy as np

    sample_id = f"davis2017_{split}_{sequence}_obj{davis_object.obj_idx:03d}"
    target_ref = _target_ref(sequence, davis_object, object_count, object_names)
    instruction = f"remove the {target_ref}" if target_ref else ""
    direct_stack = _direct_stack_for_object(ann_paths, davis_object.object_id)

    direct_path = out_root / "masks" / f"{sequence}_obj{davis_object.obj_idx:03d}_direct.npy"
    np.save(direct_path, direct_stack)

    workdir = out_root / "void_reasoner" / sample_id
    _write_void_inputs(
        out_root=out_root,
        sequence=sequence,
        workdir=workdir,
        direct_stack=direct_stack,
        first_appears_frame=davis_object.first_appears_frame,
        frame_paths=frame_paths,
        metadata={
            "sample_id": sample_id,
            "object_ids": [davis_object.object_id],
            "target_ref": target_ref,
            "instruction": instruction,
            "first_appears_frame": davis_object.first_appears_frame,
        },
    )

    return SampleWork(
        sample_id=sample_id,
        sequence=sequence,
        obj_idx=davis_object.obj_idx,
        object_ids=(davis_object.object_id,),
        object_color_bgr=davis_object.object_color_bgr,
        target_ref=target_ref,
        instruction=instruction,
        direct_mask_path=direct_path,
        workdir=workdir,
    )


def _prepare_merged_sample(
    *,
    out_root: Path,
    split: str,
    frame_root: Path,
    first: SampleWork,
    second: SampleWork,
) -> SampleWork:
    import numpy as np

    frame_paths = sorted((frame_root / first.sequence).glob("*.jpg"))
    direct_a = np.load(first.direct_mask_path, allow_pickle=False).astype(bool)
    direct_b = np.load(second.direct_mask_path, allow_pickle=False).astype(bool)
    direct = np.logical_or(direct_a, direct_b)

    obj_idx = min(first.obj_idx, second.obj_idx)
    suffix = "_".join(f"obj{i:03d}" for i in sorted((first.obj_idx, second.obj_idx)))
    sample_id = f"davis2017_{split}_{first.sequence}_{suffix}"
    target_ref = f"{first.target_ref} and {second.target_ref}"
    instruction = f"remove the {target_ref}"
    direct_path = out_root / "masks" / f"{first.sequence}_{suffix}_direct.npy"
    np.save(direct_path, direct)

    info_a = _read_json(first.workdir / "segmentation_info.json")
    info_b = _read_json(second.workdir / "segmentation_info.json")
    first_appears_frame = min(
        int(info_a.get("first_appears_frame", 0)),
        int(info_b.get("first_appears_frame", 0)),
    )
    workdir = out_root / "void_reasoner" / sample_id
    object_ids = tuple(sorted(first.object_ids + second.object_ids))
    _write_void_inputs(
        out_root=out_root,
        sequence=first.sequence,
        workdir=workdir,
        direct_stack=direct,
        first_appears_frame=first_appears_frame,
        frame_paths=frame_paths,
        metadata={
            "sample_id": sample_id,
            "object_ids": list(object_ids),
            "target_ref": target_ref,
            "instruction": instruction,
            "first_appears_frame": first_appears_frame,
            "merged_from": [first.sample_id, second.sample_id],
        },
    )
    return SampleWork(
        sample_id=sample_id,
        sequence=first.sequence,
        obj_idx=obj_idx,
        object_ids=object_ids,
        object_color_bgr=first.object_color_bgr,
        target_ref=target_ref,
        instruction=instruction,
        direct_mask_path=direct_path,
        workdir=workdir,
        merged_from=(first.sample_id, second.sample_id),
    )


def _finalize_sample(
    out_root: Path,
    split: str,
    sample: SampleWork,
    extra_reasons: set[str],
) -> dict[str, Any]:
    import numpy as np

    direct = np.load(sample.direct_mask_path, allow_pickle=False).astype(bool)
    analysis_path = sample.workdir / "vlm_analysis.json"
    analysis = _read_vlm_analysis(analysis_path)
    post_desc = str(analysis.get("scene_description") or "").strip()
    if post_desc:
        _write_json(sample.workdir / "prompt.json", {"bg": post_desc})

    indirect, indirect_quality, indirect_reason = _read_grey_mask(
        sample.workdir / "grey_mask.mp4",
        direct.shape,
        analysis_exists=analysis_path.exists(),
        analysis=analysis,
    )
    indirect_path = out_root / "masks" / f"{sample.sample_id}_indirect.npy"
    quadmask_path = out_root / "quadmasks" / f"{sample.sample_id}_quadmask.npy"
    np.save(indirect_path, indirect)
    np.save(quadmask_path, three_layer_to_quadmask(direct, indirect))

    reasons = sorted(extra_reasons)
    if not direct.any():
        reasons.append("empty_direct_mask")
    if _bad_target_ref(sample.target_ref):
        reasons.append("unresolvable_target_ref")
    if not post_desc:
        reasons.append("missing_scene_description")
    if indirect_reason:
        reasons.append(indirect_reason)

    affected_nouns = _affected_nouns(analysis)
    row = {
        "sample_id": sample.sample_id,
        "split": split,
        "sequence": sample.sequence,
        "operation": "remove",
        "instruction": sample.instruction,
        "target_ref": sample.target_ref,
        "object_ids": list(sample.object_ids),
        "object_color_bgr": list(sample.object_color_bgr),
        "source_video": f"videos/{sample.sequence}.mp4",
        "frames_dir": f"frames/{sample.sequence}",
        "direct_mask_npy": _rel(sample.direct_mask_path, out_root),
        "indirect_mask_npy": _rel(indirect_path, out_root),
        "quadmask_npy": _rel(quadmask_path, out_root),
        "post_removal_description": post_desc,
        "difficulty_class": "",
        "vlm_analysis_json": _rel(analysis_path, out_root),
        "label_quality": {
            "direct": "davis_gt",
            "indirect": indirect_quality,
            "text_condition": "void_bg",
        },
        "indirect_audit": {
            "vlm_affected_nouns": affected_nouns,
            "zero_pixel_affected_nouns": affected_nouns if affected_nouns and not indirect.any() else [],
        },
        "quarantine_reasons": sorted(set(reasons)),
    }
    if sample.merged_from:
        row["merged_from"] = list(sample.merged_from)
    return row


def _validate_row(
    out_root: Path,
    row: dict[str, Any],
    *,
    relaxed: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    import cv2
    import numpy as np

    sid = row.get("sample_id", "<missing>")
    try:
        direct = np.load(out_root / row["direct_mask_npy"], allow_pickle=False).astype(bool)
        indirect = np.load(out_root / row["indirect_mask_npy"], allow_pickle=False).astype(bool)
        quadmask = np.load(out_root / row["quadmask_npy"], allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - validator reports all row issues.
        errors.append(f"{sid}: failed to load masks: {exc}")
        return

    if direct.shape != indirect.shape or direct.shape != quadmask.shape:
        errors.append(f"{sid}: mask shape mismatch direct={direct.shape} indirect={indirect.shape} quadmask={quadmask.shape}")
    else:
        expected = three_layer_to_quadmask(direct, indirect)
        if not np.array_equal(quadmask, expected):
            errors.append(f"{sid}: quadmask does not match direct/indirect masks")

    values = set(int(x) for x in np.unique(quadmask).tolist())
    if not values <= CANONICAL_QUADMASK_VALUES:
        errors.append(f"{sid}: quadmask values {sorted(values)} not subset of {sorted(CANONICAL_QUADMASK_VALUES)}")
    if row.get("split") != "train":
        errors.append(f"{sid}: split must be train, got {row.get('split')!r}")
    if not relaxed and not direct.any():
        errors.append(f"{sid}: empty direct mask")

    frames_dir = out_root / str(row.get("frames_dir", ""))
    frame_paths = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if len(frame_paths) != direct.shape[0]:
        errors.append(f"{sid}: mask frames {direct.shape[0]} != frames_dir jpg count {len(frame_paths)}")
    elif frame_paths:
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            errors.append(f"{sid}: failed to read first frame {frame_paths[0]}")
        elif tuple(first.shape[:2]) != tuple(direct.shape[1:]):
            errors.append(f"{sid}: mask HxW {direct.shape[1:]} != frame HxW {first.shape[:2]}")

    desc = str(row.get("post_removal_description") or "").strip()
    if not relaxed and not desc:
        errors.append(f"{sid}: missing post_removal_description")

    target_ref = str(row.get("target_ref") or "").strip()
    instruction = str(row.get("instruction") or "").strip()
    if not relaxed:
        if _bad_target_ref(target_ref):
            errors.append(f"{sid}: invalid target_ref {target_ref!r}")
        if target_ref.casefold() not in instruction.casefold():
            errors.append(f"{sid}: instruction does not contain target_ref {target_ref!r}")
    if desc and target_ref and target_ref.casefold() in desc.casefold():
        warnings.append(f"{sid}: post_removal_description contains target_ref {target_ref!r}")


def _run_stage2(
    args: argparse.Namespace,
    out_root: Path,
    samples: list[SampleWork],
    forced_quarantine: dict[str, set[str]],
) -> None:
    pending = [
        sample for sample in samples
        if _stage_runnable(sample, forced_quarantine)
        and (args.overwrite_vlm or not (sample.workdir / "vlm_analysis.json").exists())
    ]
    _run_filtered_config(
        out_root / "void_reasoner" / "config.stage2.json",
        [_config_row(sample, out_root) for sample in pending],
        [args.python_bin, args.stage2_script, "--config"],
        cwd=out_root,
    )


def _run_stage3a(
    args: argparse.Namespace,
    out_root: Path,
    samples: list[SampleWork],
    forced_quarantine: dict[str, set[str]],
) -> None:
    pending = [
        sample for sample in samples
        if _stage_runnable(sample, forced_quarantine)
        and (sample.workdir / "vlm_analysis.json").exists()
        and (args.overwrite_vlm or not (sample.workdir / "grey_mask.mp4").exists())
    ]
    _run_filtered_config(
        out_root / "void_reasoner" / "config.stage3a.json",
        [_config_row(sample, out_root) for sample in pending],
        [args.python_bin, args.stage3a_script, "--config"],
        cwd=out_root,
        extra_args=["--segmentation-model", args.segmentation_model],
    )


def _run_filtered_config(
    config_path: Path,
    rows: list[dict[str, str]],
    command_prefix: list[str],
    *,
    cwd: Path,
    extra_args: list[str] | None = None,
) -> None:
    if not rows:
        return
    _write_json(config_path, rows)
    _run([*command_prefix, str(config_path), *(extra_args or [])], cwd=cwd)


def _stage_runnable(sample: SampleWork, forced_quarantine: dict[str, set[str]]) -> bool:
    reasons = forced_quarantine.get(sample.sample_id, set())
    return bool(sample.target_ref) and not reasons.intersection({
        "unresolvable_target_ref",
        "integral_pair_member",
        "integral_belonging_unresolved",
    })


def _build_integral_merged_samples(
    *,
    out_root: Path,
    split: str,
    frame_root: Path,
    samples_by_sequence: dict[str, list[SampleWork]],
    forced_quarantine: dict[str, set[str]],
) -> list[SampleWork]:
    merged: list[SampleWork] = []
    seen_pairs: set[tuple[str, str]] = set()
    for sequence_samples in samples_by_sequence.values():
        for sample in sequence_samples:
            if _bad_target_ref(sample.target_ref):
                continue
            analysis = _read_vlm_analysis(sample.workdir / "vlm_analysis.json")
            for belonging in analysis.get("integral_belongings", []):
                noun = str(belonging.get("noun") or "").strip()
                if not noun:
                    continue
                matches = [
                    other for other in sequence_samples
                    if other.sample_id != sample.sample_id and _noun_matches(noun, other.target_ref)
                ]
                if len(matches) != 1:
                    forced_quarantine.setdefault(sample.sample_id, set()).add("integral_belonging_unresolved")
                    continue

                other = matches[0]
                pair = tuple(sorted((sample.sample_id, other.sample_id)))
                forced_quarantine.setdefault(sample.sample_id, set()).add("integral_pair_member")
                forced_quarantine.setdefault(other.sample_id, set()).add("integral_pair_member")
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                merged.append(_prepare_merged_sample(
                    out_root=out_root,
                    split=split,
                    frame_root=frame_root,
                    first=sample,
                    second=other,
                ))
    return merged


def _read_split(davis_root: Path, split: str) -> list[str]:
    path = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sequence_objects(ann_paths: list[Path]) -> list[DavisObject]:
    import numpy as np

    first_seen: dict[int, int] = {}
    palette: dict[int, tuple[int, int, int]] = {}
    for frame_idx, path in enumerate(ann_paths):
        arr, frame_palette = _read_annotation_indices(path)
        for object_id in _object_indices(arr):
            first_seen.setdefault(object_id, frame_idx)
            if object_id in frame_palette:
                palette.setdefault(object_id, frame_palette[object_id])
    return [
        DavisObject(
            obj_idx=obj_idx,
            object_id=object_id,
            object_color_bgr=_palette_bgr(palette, object_id),
            first_appears_frame=first_seen[object_id],
        )
        for obj_idx, object_id in enumerate(sorted(np.int64(k).item() for k in first_seen))
    ]


def _direct_stack_for_object(ann_paths: list[Path], object_id: int):
    import numpy as np

    frames = []
    for path in ann_paths:
        arr, _ = _read_annotation_indices(path)
        frames.append(arr == object_id)
    return np.stack(frames).astype(bool)


def _read_annotation_indices(path: Path):
    import cv2
    import numpy as np

    indexed = _read_indexed_png(path)
    if indexed is not None:
        return indexed

    ann = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ann is None:
        raise ValueError(f"failed to read annotation {path}")
    if ann.ndim == 2:
        return ann.astype(np.uint8), {}
    raise ValueError(f"{path} is not an indexed or grayscale annotation; expanded RGB cannot preserve DAVIS object ids")


def _read_indexed_png(path: Path):
    import numpy as np

    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    palette: dict[int, tuple[int, int, int]] = {}
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"PLTE":
            for idx in range(0, len(chunk), 3):
                palette[idx // 3] = (chunk[idx], chunk[idx + 1], chunk[idx + 2])
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if color_type != 3:
        return None
    if bit_depth != 8 or interlace != 0:
        raise ValueError(f"{path} uses unsupported indexed PNG bit depth/interlace")

    raw = zlib.decompress(bytes(idat))
    stride = int(width)
    rows = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(int(height)):
        filter_type = raw[offset]
        offset += 1
        cur = bytearray(raw[offset:offset + stride])
        offset += stride
        _unfilter_png_row(cur, prev, filter_type)
        rows.append(bytes(cur))
        prev = cur
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(int(height), int(width)).copy(), palette


def _unfilter_png_row(cur: bytearray, prev: bytearray, filter_type: int) -> None:
    for i, value in enumerate(cur):
        left = cur[i - 1] if i else 0
        up = prev[i]
        upper_left = prev[i - 1] if i else 0
        if filter_type == 0:
            continue
        if filter_type == 1:
            cur[i] = (value + left) & 0xFF
        elif filter_type == 2:
            cur[i] = (value + up) & 0xFF
        elif filter_type == 3:
            cur[i] = (value + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            cur[i] = (value + _paeth(left, up, upper_left)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter {filter_type}")


def _paeth(left: int, up: int, upper_left: int) -> int:
    p = left + up - upper_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return upper_left


def _object_indices(arr: Any) -> list[int]:
    import numpy as np

    return [
        int(x) for x in np.unique(np.asarray(arr)).tolist()
        if 1 <= int(x) <= 254
    ]


def _palette_bgr(palette: dict[int, tuple[int, int, int]], idx: int) -> tuple[int, int, int]:
    rgb = palette.get(int(idx), (idx, idx, idx))
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _target_ref(
    sequence: str,
    davis_object: DavisObject,
    object_count: int,
    object_names: dict[str, Any],
) -> str:
    explicit = _lookup_object_name(object_names, sequence, davis_object)
    if explicit:
        return explicit
    if object_count == 1:
        return sequence.replace("-", " ")
    return ""


def _lookup_object_name(object_names: dict[str, Any], sequence: str, davis_object: DavisObject) -> str:
    entry = object_names.get(sequence)
    if isinstance(entry, list) and davis_object.obj_idx < len(entry):
        return str(entry[davis_object.obj_idx]).strip().lower()
    if isinstance(entry, dict):
        for key in (str(davis_object.object_id), str(davis_object.obj_idx), f"obj{davis_object.obj_idx:03d}"):
            if key in entry:
                return str(entry[key]).strip().lower()
    return ""


def _read_object_names(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bad_target_ref(target_ref: str) -> bool:
    return target_ref.strip().casefold() in BAD_TARGET_REFS


def _write_void_inputs(
    *,
    out_root: Path,
    sequence: str,
    workdir: Path,
    direct_stack: Any,
    first_appears_frame: int,
    frame_paths: list[Path],
    metadata: dict[str, Any],
) -> None:
    import cv2

    workdir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(out_root / "videos" / f"{sequence}.mp4", workdir / "input_video.mp4")
    _write_mask_video(direct_stack, workdir / "black_mask.mp4")

    frame_idx = min(max(int(first_appears_frame), 0), max(len(frame_paths) - 1, 0))
    first = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR) if frame_paths else _read_first_video_frame(workdir / "input_video.mp4")
    if first is None:
        raise ValueError(f"failed to read first frame for {sequence}")
    cv2.imwrite(str(workdir / "first_frame.jpg"), first)
    _write_json(workdir / "segmentation_info.json", metadata)


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(temp_avi), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), False)
    if not writer.isOpened():
        raise ValueError(f"failed to open mask video writer: {temp_avi}")
    for frame in direct:
        writer.write(np.where(frame, 0, 255).astype(np.uint8))
    writer.release()

    cmd = [
        "ffmpeg", "-y", "-i", str(temp_avi),
        "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv444p", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    temp_avi.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed writing {out_path}: {result.stderr.strip()}")


def _read_first_video_frame(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _read_grey_mask(
    path: Path,
    shape: tuple[int, int, int],
    *,
    analysis_exists: bool,
    analysis: dict[str, Any],
):
    import cv2
    import numpy as np

    empty = np.zeros(shape, dtype=bool)
    affected_nouns = _affected_nouns(analysis)
    if not analysis_exists:
        return empty, "not_run", None
    if not affected_nouns:
        return empty, "void_vlm_none", None
    if not path.exists():
        return empty, "missing", "missing_indirect_mask"

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
        return empty, "missing", "indirect_frame_mismatch"
    arr = np.stack(frames).astype(bool)
    if arr.shape != shape:
        return empty, "missing", "indirect_shape_mismatch"
    return arr, "void_vlm_weak", None


def _affected_nouns(analysis: dict[str, Any]) -> list[str]:
    return [
        str(obj.get("noun") or "").strip().lower()
        for obj in analysis.get("affected_objects", [])
        if str(obj.get("noun") or "").strip()
    ]


def _noun_matches(noun: str, target_ref: str) -> bool:
    noun_tokens = _noun_tokens(noun)
    target_tokens = _noun_tokens(target_ref)
    return bool(noun_tokens and (noun_tokens <= target_tokens or target_tokens <= noun_tokens))


def _noun_tokens(text: str) -> set[str]:
    stop = {"a", "an", "the", "his", "her", "its", "their", "and"}
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if token in stop:
            continue
        tokens.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return tokens


def _read_vlm_analysis(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _config_row(sample: SampleWork, out_root: Path) -> dict[str, str]:
    return {
        "video_path": _rel(sample.workdir / "input_video.mp4", out_root),
        "output_dir": _rel(sample.workdir, out_root),
        "instruction": sample.instruction,
    }


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
