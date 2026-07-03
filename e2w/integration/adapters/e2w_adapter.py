#!/usr/bin/env python3
"""Run E2W V0 vanilla over the CF-VEdit benchmark and write predictions/<run>.

Boundary B1 is preserved: the benchmark consumes files; it never imports E2W.
This adapter is producer-side integration code and writes the disk contract via
`e2w_core.io_contract`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

E2W_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = E2W_ROOT.parent
for rel in ("packages/e2w_core", "packages/localization", "packages/generation", "."):
    p = E2W_ROOT / rel
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e2w_core.io_contract import (  # noqa: E402
    BENCHMARK_VERSION,
    PREDICTIONS_INDEX,
    PREDICTIONS_VIDEO_DIR,
    RUN_META,
    STATUS_OK,
    PredictionRow,
)
from integration.pipelines.e2w_pipeline import build_v0_pipeline  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_sha256(manifest: Path) -> str:
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _render_one_job(pipeline: Any, job: dict[str, Any]) -> PredictionRow:
    import numpy as np
    from e2w_core.masks import ThreeLayerMask
    from e2w_core.plan import validate_edit_tokens_shape

    arrays = np.load(job["mask_npz"], allow_pickle=False)
    mask = ThreeLayerMask(direct=arrays["direct"].astype(bool), indirect=arrays["indirect"].astype(bool))
    source = pipeline.abductor.invert(job["source_path"])
    # Full A.1 MUST condition on edit_tokens — never silently fall back to the
    # instruction string, or a broken planner would still render `status: ok` while
    # bypassing the [EDIT] -> edit_tokens -> renderer path. Vanilla carries no
    # edit_tokens and intentionally uses the native instruction text.
    if not job.get("vanilla", True):
        if "edit_tokens" not in arrays.files:
            raise RuntimeError("full mode requires edit_tokens in the mask npz, but none were saved")
        edit = arrays["edit_tokens"]
        validate_edit_tokens_shape(edit.shape, slots=int(job["edit_slots"]))
        import torch

        edit_condition = torch.from_numpy(edit)
    else:
        edit_condition = job["instruction"]
    pipeline.renderer.render(source, edit_condition, mask, out_path=job["out_path"])
    return PredictionRow(sample_id=job["sample_id"], video=job["video"], status=STATUS_OK, error=None)


def render_worker(job_path: Path) -> int:
    payload = json.loads(job_path.read_text())
    if "jobs" in payload:
        jobs = payload["jobs"]
        config = payload["config"]
        results_path = Path(payload["results_path"])
        continue_on_error = bool(payload.get("continue_on_error", False))
    else:
        jobs = [payload]
        config = payload["config"]
        results_path = Path(payload.get("results_path", job_path.with_suffix(".results.jsonl")))
        continue_on_error = bool(payload.get("continue_on_error", False))

    pipeline = build_v0_pipeline(config)
    ok = True
    for job in jobs:
        try:
            pred = _render_one_job(pipeline, job)
        except Exception as exc:  # noqa: BLE001 - persisted for benchmark accounting.
            traceback.print_exc()
            pred = PredictionRow(
                sample_id=job.get("sample_id", Path(job.get("out_path", "unknown")).stem),
                video=None,
                status="failed",
                error=f"render {type(exc).__name__}: {exc}",
            )
            ok = False
            append_jsonl(results_path, pred.to_json())
            if not continue_on_error:
                return 1
            continue
        append_jsonl(results_path, pred.to_json())
    return 0 if ok or continue_on_error else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run E2W V0 vanilla benchmark adapter")
    parser.add_argument("--run-name", default="e2w_vanilla_v0")
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark-root", default=str(REPO_ROOT / "physics_iq_for_simple_eval"))
    parser.add_argument("--sample-id", action="append", help="Run only selected sample id(s); repeatable")
    parser.add_argument("--sample-limit", type=int, help="Run only the first N manifest rows")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing predictions/<run-name> before writing")
    parser.add_argument("--continue-on-error", action="store_true", help="Write failed rows instead of aborting on first sample error")
    parser.add_argument("--vanilla", action="store_true", help="Run the V0 vanilla bypass path")
    parser.add_argument("--full", action="store_true",
                        help="Run the full (untrained) A.1 architecture: query-token three-layer mask + edit_tokens")
    parser.add_argument("--render-worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.render_worker_json:
        return render_worker(Path(args.render_worker_json).resolve())

    if args.vanilla == args.full:
        parser.error("pass exactly one of --vanilla (bypass) or --full (untrained A.1)")
    vanilla = args.vanilla
    if args.config is None:
        name = "vanilla.v0.json" if vanilla else "full.v0.json"
        args.config = str(E2W_ROOT / "configs" / name)
    config_path = Path(args.config).resolve()
    version = json.loads(config_path.read_text()).get("version", "")
    if ("vanilla" in version) != vanilla:
        mode = "vanilla" if vanilla else "full"
        parser.error(f"--{mode} requires a matching config version, got {version!r} from {config_path}")

    benchmark_root = Path(args.benchmark_root).resolve()
    manifest_path = benchmark_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    run_dir = benchmark_root / "predictions" / args.run_name
    videos_dir = run_dir / PREDICTIONS_VIDEO_DIR
    masks_dir = run_dir / "e2w_masks"
    jobs_dir = run_dir / "e2w_render_jobs"
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"prediction run exists; pass --overwrite: {run_dir}")
        shutil.rmtree(run_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(manifest_path)
    if args.sample_id:
        wanted = set(args.sample_id)
        manifest_rows = [row for row in manifest_rows if row["sample_id"] in wanted]
    if args.sample_limit is not None:
        manifest_rows = manifest_rows[: args.sample_limit]

    write_json(
        run_dir / RUN_META,
        {
            "run_name": args.run_name,
            "benchmark_version": BENCHMARK_VERSION,
            "manifest_sha256": manifest_sha256(manifest_path),
            "model_name": (
                "E2W V0 vanilla: Sa2VA [SEG] + frozen CogVideoX-Fun/VOID pass1 "
                "(quadmask channel-concat)"
                if vanilla else
                "E2W V0 full (untrained A.1): query-token [SEG_DIR]/[SEG_IND]/[EDIT] "
                "three-layer mask + edit_tokens hard-replacing T5 -> frozen CogVideoX-Fun/VOID pass1"
            ),
            "model_version": "v0-vanilla-eval" if vanilla else "v0-full-untrained-eval",
            "mode": "vanilla" if vanilla else "full-untrained",
            "created_at": utc_now(),
            "num_samples": len(manifest_rows),
            "command": " ".join(sys.argv),
            "config": str(config_path),
            "adapter": str(Path(__file__).resolve()),
            "status_policy": "continue-on-error" if args.continue_on_error else "abort-on-error",
        },
    )

    pipeline = build_v0_pipeline(config_path)
    slots = int(pipeline.planner.config.edit_token_slots)
    prediction_rows: list[dict[str, Any]] = []
    planned: dict[str, tuple[dict[str, Any], str, str]] = {}
    failures: dict[str, PredictionRow] = {}
    position_ids_modes: dict[str, Any] = {}

    # Stage 1: Sa2VA localization only. Keep the renderer unloaded, then release
    # Sa2VA before the CogVideoX-Fun/VOID backend enters GPU memory.
    for row in manifest_rows:
        sample_id = row["sample_id"]
        video_name = f"{sample_id}.mp4"
        source_path = benchmark_root / "videos" / "source" / video_name
        instruction = row["instruction"]
        target_ref = row.get("target_ref") or row.get("target") or instruction
        operation = row.get("operation", "attribute")
        try:
            print(f"[e2w-adapter] localize {sample_id}: {instruction}", flush=True)
            import numpy as np

            from e2w_core.plan import validate_edit_tokens_shape

            mask, plan = pipeline.planner.plan(
                source_path,
                instruction,
                target_ref=target_ref,
                operation=operation,
                vanilla=vanilla,
            )
            mask_path = masks_dir / f"{sample_id}.npz"
            save_kwargs = dict(direct=mask.direct.astype(bool), indirect=mask.indirect.astype(bool))
            if not vanilla:
                # Fail loudly: a full run with no edit_tokens is a broken planner, not
                # a renderable sample. Validate shape before it reaches the renderer.
                if plan.edit_tokens is None:
                    raise RuntimeError(f"full mode planner returned no edit_tokens for {sample_id}")
                edit_np = plan.edit_tokens.float().cpu().numpy()
                validate_edit_tokens_shape(edit_np.shape, slots=slots)
                save_kwargs["edit_tokens"] = edit_np
                if plan.region_query is not None:
                    save_kwargs["region_query"] = plan.region_query.float().cpu().numpy()
                mode = getattr(pipeline.planner, "last_position_ids_mode", None)
                position_ids_modes[sample_id] = mode
                if mode and mode != "get_rope_index+extend":
                    print(f"[e2w-adapter] WARNING {sample_id}: position_ids fell back to {mode!r} "
                          f"(M-RoPE path not used)", flush=True)
            np.savez_compressed(mask_path, **save_kwargs)
            planned[sample_id] = (row, str(source_path), str(mask_path))
        except Exception as exc:  # noqa: BLE001 - persisted for benchmark failure accounting.
            err = f"localization {type(exc).__name__}: {exc}"
            traceback.print_exc()
            failures[sample_id] = PredictionRow(sample_id=sample_id, video=None, status="failed", error=err)
            if not args.continue_on_error:
                prediction_rows.append(failures[sample_id].to_json())
                write_jsonl(run_dir / PREDICTIONS_INDEX, prediction_rows)
                return 1

    pipeline.planner.unload()

    # Record which position-id path each sample actually used, so a silent M-RoPE
    # fallback (arange) is visible in the run artifact and never mistaken for the
    # validated get_rope_index path (ADR-0004).
    if not vanilla and position_ids_modes:
        meta_path = run_dir / RUN_META
        meta = json.loads(meta_path.read_text())
        meta["position_ids_modes"] = position_ids_modes
        write_json(meta_path, meta)

    # Stage 2: CogVideoX-Fun/VOID rendering. Sa2VA is no longer resident.
    render_jobs: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        video_name = f"{sample_id}.mp4"
        out_rel = f"{PREDICTIONS_VIDEO_DIR}/{video_name}"
        out_path = run_dir / out_rel
        if sample_id in failures:
            continue

        instruction = row["instruction"]
        print(f"[e2w-adapter] queue render {sample_id}: {instruction}", flush=True)
        _row, source_path_str, mask_path_str = planned[sample_id]
        job = {
            "config": str(config_path),
            "sample_id": sample_id,
            "source_path": source_path_str,
            "mask_npz": mask_path_str,
            "instruction": instruction,
            "out_path": str(out_path),
            "video": out_rel,
            "vanilla": vanilla,
            "edit_slots": slots,
        }
        write_json(jobs_dir / f"{sample_id}.json", job)
        render_jobs.append(job)

    render_results_path = jobs_dir / "render_results.jsonl"
    render_worker_failed = False
    if render_jobs:
        batch_path = jobs_dir / "batch.json"
        write_json(
            batch_path,
            {
                "config": str(config_path),
                "results_path": str(render_results_path),
                "continue_on_error": args.continue_on_error,
                "jobs": render_jobs,
            },
        )
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--render-worker-json", str(batch_path)],
            cwd=str(E2W_ROOT),
            env=env,
        )
        render_worker_failed = proc.returncode != 0

    rendered = {}
    if render_results_path.exists():
        rendered = {row["sample_id"]: row for row in read_jsonl(render_results_path)}

    for row in manifest_rows:
        sample_id = row["sample_id"]
        if sample_id in failures:
            pred = failures[sample_id]
        elif sample_id in rendered:
            prediction_rows.append(rendered[sample_id])
            write_jsonl(run_dir / PREDICTIONS_INDEX, prediction_rows)
            continue
        else:
            pred = PredictionRow(
                sample_id=sample_id,
                video=None,
                status="failed",
                error="render missing result row from worker",
            )
            render_worker_failed = True
        prediction_rows.append(pred.to_json())
        write_jsonl(run_dir / PREDICTIONS_INDEX, prediction_rows)

    if render_worker_failed and not args.continue_on_error:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
