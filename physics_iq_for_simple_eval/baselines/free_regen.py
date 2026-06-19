#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import BENCHMARK_VERSION, load_manifest, manifest_sha256, utc_now, write_json, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Package a source-free regeneration baseline into predictions/<run_name>"
    )
    parser.add_argument(
        "--generated-dir",
        required=True,
        help="Directory containing regenerated <sample_id>.mp4 files produced without using source videos.",
    )
    parser.add_argument("--run-name", default="free_regen")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    generated_dir = Path(args.generated_dir).resolve()
    rows = load_manifest(validate=True)
    run_dir = ROOT / "predictions" / args.run_name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
        shutil.rmtree(run_dir)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        source = generated_dir / f"{sample_id}.mp4"
        if source.exists():
            destination = videos_dir / f"{sample_id}.mp4"
            shutil.copy2(source, destination)
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "video": f"videos/{sample_id}.mp4",
                    "status": "ok",
                    "runtime_sec": None,
                }
            )
        else:
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "video": None,
                    "status": "failed",
                    "error": f"missing regenerated video: {source}",
                }
            )

    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "run_name": args.run_name,
            "model_name": "free_regen",
            "model_version": "external",
            "benchmark_version": BENCHMARK_VERSION,
            "manifest_sha256": manifest_sha256(),
            "command": (
                "python baselines/free_regen.py --generated-dir "
                + str(generated_dir)
                + " --run-name "
                + args.run_name
            ),
            "created_at": utc_now(),
            "num_samples": len(rows),
            "hardware": {},
            "notes": "Upper-bound anchor: package videos regenerated without conditioning on source videos.",
            "baseline_type": "free_regen",
        },
    )
    print(f"wrote predictions/{args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
