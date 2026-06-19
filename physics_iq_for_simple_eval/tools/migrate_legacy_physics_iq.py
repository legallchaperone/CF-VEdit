#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_rate(value):
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def probe_video(path):
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    info = json.loads(completed.stdout)
    stream = info["streams"][0]
    fps = parse_rate(stream.get("avg_frame_rate"))
    nb_frames = stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        num_frames = int(nb_frames)
    else:
        num_frames = max(1, round(float(stream.get("duration", 0.0)) * fps))
    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def contract_from_legacy(row):
    return {
        "sample_id": row["sample_id"],
        "operation": row["operation"],
        "target_ref": row["target_object"],
        "counterfactual_state": {
            "visible_outcome": row["expected_visible_outcome"],
            "physical_effect": row["expected_physical_effect"],
            "temporal": "The requested edit should remain consistent across the full edited video.",
        },
        "affected_regions": [
            f"target intervention: {row['target_object']}",
            f"visible consequence: {row['expected_visible_outcome']}",
            f"physical consequence: {row['expected_physical_effect']}",
        ],
        "preserve_regions": row["must_preserve"],
        "expected_visible_outcome": row["expected_visible_outcome"],
        "expected_physical_effect": row["expected_physical_effect"],
        "source_description": row.get("physics_iq_description", ""),
    }


def vlm_prompt(row, contract):
    prompt_shape = {
        "target_success": 0,
        "preservation_success": 0,
        "effect_hits": [],
        "physical_effect_success": 0,
        "temporal_consistency": 0,
        "major_artifacts": 0,
        "overall_pass": 0,
        "short_reason": "",
    }
    return (
        "You are judging a CF-VEdit counterfactual video edit.\n\n"
        "Inputs:\n"
        "1. Original source video.\n"
        "2. Edited candidate video.\n"
        f"3. Operation: {row['operation']}\n"
        f"4. Edit request: {row['user_prompt']}\n"
        f"5. Target: {row['target_object']}\n"
        f"6. Preserve regions: {', '.join(contract['preserve_regions'])}\n"
        f"7. Counterfactual effects: {json.dumps(all_effect_labels(contract), ensure_ascii=False)}\n\n"
        "Judge only the final edited video against the instruction and contract. "
        "Do not use any model internals or reasoning trace. Preserve regions are "
        "judged at object or identity level, not by pixel equality.\n\n"
        "Answer only JSON with exactly this shape:\n"
        f"{json.dumps(prompt_shape, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- If target_success is 0, physical_effect_success must be 0 and effect_hits must be empty.\n"
        "- effect_hits must contain only strings from the listed counterfactual effects.\n"
        "- Penalize unrelated object removal, scene identity changes, one-frame edits, and major artifacts.\n"
    )


def all_effect_labels(contract):
    labels = list(contract["affected_regions"])
    labels.extend(contract["counterfactual_state"].values())
    return labels


def migrate(legacy_manifest):
    rows = read_jsonl(legacy_manifest)

    for directory in [
        ROOT / "videos" / "source",
        ROOT / "contracts",
        ROOT / "annotations" / "masks",
        ROOT / "judges",
        ROOT / "predictions",
        ROOT / "results",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    legacy_manifest_copy = ROOT / "annotations" / "legacy_manifest.jsonl"
    if legacy_manifest.resolve() != legacy_manifest_copy.resolve() and not legacy_manifest_copy.exists():
        shutil.copy2(legacy_manifest, legacy_manifest_copy)

    root_summary = ROOT / "summary.json"
    legacy_summary = ROOT / "annotations" / "legacy_summary.json"
    if root_summary.exists() and not legacy_summary.exists():
        shutil.move(root_summary, legacy_summary)

    manifest_rows = []
    provenance_rows = []
    prompt_rows = []

    for row in rows:
        sample_id = row["sample_id"]
        legacy_video_name = Path(row["converted_video"]).name
        legacy_video_path = ROOT / "converted" / legacy_video_name
        source_video_rel = f"videos/source/{sample_id}.mp4"
        source_video_path = ROOT / source_video_rel
        if not source_video_path.exists():
            if not legacy_video_path.exists():
                raise FileNotFoundError(f"missing legacy video for {sample_id}: {legacy_video_path}")
            shutil.move(str(legacy_video_path), str(source_video_path))

        contract = contract_from_legacy(row)
        contract_rel = f"contracts/{sample_id}.json"
        write_json(ROOT / contract_rel, contract)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_video": source_video_rel,
                "operation": row["operation"],
                "instruction": row["user_prompt"],
                "target_ref": row["target_object"],
                "split": "test",
                "category": row["physics_iq_category"],
                "scene_type": row["physics_iq_category"],
                "difficulty": "medium",
                "identifiability": "identifiable",
                "pair_id": None,
                "video_meta": probe_video(source_video_path),
                "contract": contract_rel,
                "annotations": {
                    "target_mask": None,
                    "affected_mask": None,
                    "preserve_mask": None,
                },
            }
        )

        leakage = row.get("leakage_exclusion_evidence", {})
        provenance_rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": row.get("source_metadata", {}).get("dataset", "Physics-IQ"),
                "source_uri": row.get("source_full_video"),
                "source_scenario": row.get("physics_iq_scenario"),
                "source_metadata": row.get("source_metadata", {}),
                "leakage_checked": bool(leakage.get("leakage_checked", False)),
                "leaked": bool(leakage.get("leaked", False)),
                "matched_paths": leakage.get("matched_paths", []),
            }
        )
        prompt_rows.append(
            {
                "sample_id": sample_id,
                "prompt_version": "cf_vedit_v0.1",
                "prompt": vlm_prompt(row, contract),
            }
        )

    write_jsonl(ROOT / "manifest.jsonl", manifest_rows)
    write_jsonl(ROOT / "annotations" / "provenance.jsonl", provenance_rows)
    write_jsonl(ROOT / "judges" / "vlm_prompts.jsonl", prompt_rows)

    old_prompt_path = ROOT / "vlm_judge_prompts.jsonl"
    if old_prompt_path.exists():
        legacy_prompt_path = ROOT / "judges" / "legacy_vlm_judge_prompts.jsonl"
        if not legacy_prompt_path.exists():
            shutil.move(old_prompt_path, legacy_prompt_path)
        else:
            old_prompt_path.unlink()

    converted_dir = ROOT / "converted"
    if converted_dir.exists() and not any(converted_dir.iterdir()):
        converted_dir.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate legacy Physics-IQ simple eval assets")
    parser.add_argument("--legacy-manifest", default=str(ROOT / "manifest.jsonl"))
    args = parser.parse_args(argv)
    migrate(Path(args.legacy_manifest))
    print("migrated legacy Physics-IQ simple eval assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
