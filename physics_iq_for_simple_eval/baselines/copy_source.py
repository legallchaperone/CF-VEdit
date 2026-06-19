#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import BENCHMARK_VERSION, load_manifest, manifest_sha256, utc_now, write_json, write_jsonl


def reusable_baseline_dir(run_dir):
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return False
    try:
        import json

        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return False
    return meta.get("baseline_type") == "copy_source"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create the copy_source CF-VEdit baseline run")
    parser.add_argument("--run-name", default="copy_source")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    rows = load_manifest(validate=True)
    run_dir = ROOT / "predictions" / args.run_name
    if run_dir.exists():
        if not args.overwrite and not reusable_baseline_dir(run_dir):
            raise SystemExit(f"refusing to overwrite non-copy_source run: {run_dir}")
        shutil.rmtree(run_dir)

    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        source = ROOT / row["source_video"]
        destination = videos_dir / f"{sample_id}.mp4"
        shutil.copy2(source, destination)
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "video": f"videos/{sample_id}.mp4",
                "status": "ok",
                "runtime_sec": 0.0,
            }
        )

    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "run_name": args.run_name,
            "model_name": "copy_source",
            "model_version": "baseline",
            "benchmark_version": BENCHMARK_VERSION,
            "manifest_sha256": manifest_sha256(),
            "command": "python baselines/copy_source.py --run-name " + args.run_name,
            "created_at": utc_now(),
            "num_samples": len(rows),
            "hardware": {},
            "notes": "Lower-bound anchor: copy the source video without editing.",
            "baseline_type": "copy_source",
        },
    )
    print(f"wrote predictions/{args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
