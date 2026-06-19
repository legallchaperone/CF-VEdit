#!/usr/bin/env python3
"""Package any editing model's outputs into a CF-VEdit predictions/<run_name>/.

The benchmark never calls the model (boundary B1). This adapter runs your model
once per sample by shelling out to a command template, writes each edited clip to
``predictions/<run_name>/videos/<sample_id>.mp4``, and emits the two files that
are easy to get wrong by hand: ``predictions.jsonl`` (one row per sample, failures
included) and ``run_meta.json`` (with the manifest hash + version lock).

Example — run Bernini on every sample (replace the command with your real CLI):

    python tools/make_prediction_run.py \
        --run-name bernini_v1 --model-name Bernini --model-version v1 \
        --cmd 'bernini-edit --input {source} --instruction {instruction} --output {out}'

Placeholders available in --cmd (each is substituted as a single argv token, so
{instruction} stays one argument even though it contains spaces):
    {source}      absolute path to the source clip
    {out}         absolute path to write the edited clip to (it is what gets scored)
    {sample_id}   e.g. piq_simple_eval_0038_add
    {instruction} natural-language edit instruction
    {operation}   add | remove | attribute | force_event
    {target_ref}  the edit target, e.g. "red clamp"

A sample is recorded ``status: ok`` only if the command exits 0 AND {out} exists;
otherwise it is ``status: failed`` (counted in the failure rate, scored 0). If
your model is a Python API rather than a CLI, replace ``run_one`` below.

After this, score and report it with a real judge (a normal run has no offline
anchor):

    python bench.py validate bernini_v1
    python bench.py score  bernini_v1 --judge vlm      # OPENROUTER_API_KEY + ffmpeg
    python bench.py report bernini_v1 --judge vlm
    # or: score --judge human  (built-in UI),  then report --judge human
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import (  # noqa: E402
    BENCHMARK_VERSION,
    load_manifest,
    manifest_sha256,
    utc_now,
    write_json,
    write_jsonl,
)


def run_one(cmd_template, fields, timeout):
    """Run the model command for one sample. Return (ok, error_message)."""
    tokens = [token.format(**fields) for token in shlex.split(cmd_template)]
    try:
        proc = subprocess.run(tokens, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError as exc:
        return False, f"command not found: {exc}"
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {proc.stderr.strip()[:300]}"
    return True, ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package model outputs into predictions/<run_name>/")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--cmd",
        required=True,
        help="command template invoked once per sample; see the module docstring for placeholders",
    )
    parser.add_argument("--timeout", type=float, default=600, help="per-sample timeout in seconds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the command for each sample and exit")
    args = parser.parse_args(argv)

    rows = load_manifest(validate=True)
    run_dir = ROOT / "predictions" / args.run_name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing run (use --overwrite): {run_dir}")
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    ok = 0
    for row in rows:
        sample_id = row["sample_id"]
        out_path = videos_dir / f"{sample_id}.mp4"
        fields = {
            "source": str((ROOT / row["source_video"]).resolve()),
            "out": str(out_path.resolve()),
            "sample_id": sample_id,
            "instruction": row["instruction"],
            "operation": row["operation"],
            "target_ref": row["target_ref"],
        }
        if args.dry_run:
            print(" ".join(shlex.quote(t.format(**fields)) for t in shlex.split(args.cmd)))
            continue

        success, error = run_one(args.cmd, fields, args.timeout)
        if success and out_path.exists():
            ok += 1
            prediction_rows.append(
                {"sample_id": sample_id, "video": f"videos/{sample_id}.mp4", "status": "ok"}
            )
            print(f"[ok]     {sample_id}")
        else:
            out_path.unlink(missing_ok=True)  # never leave a half-written video behind
            reason = error or "command exited 0 but produced no output video"
            prediction_rows.append(
                {"sample_id": sample_id, "video": None, "status": "failed", "error": reason}
            )
            print(f"[failed] {sample_id}: {reason}", file=sys.stderr)

    if args.dry_run:
        return 0

    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "run_name": args.run_name,
            "model_name": args.model_name,
            "model_version": args.model_version,
            "benchmark_version": BENCHMARK_VERSION,
            "manifest_sha256": manifest_sha256(),
            "command": f"tools/make_prediction_run.py --run-name {args.run_name} --cmd {args.cmd!r}",
            "created_at": utc_now(),
            "num_samples": len(rows),
            "hardware": {},
            "notes": "Created by tools/make_prediction_run.py",
        },
    )
    print(f"\nwrote predictions/{args.run_name}  ({ok}/{len(rows)} ok)")
    print(f"next: python bench.py validate {args.run_name} && python bench.py score {args.run_name} --judge vlm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
