#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK_VERSION = "0.1.0"
RUN_META_REQUIRED = {
    "run_name",
    "model_name",
    "model_version",
    "benchmark_version",
    "manifest_sha256",
    "command",
    "created_at",
    "num_samples",
}
JUDGE_FIELDS = {
    "target_success": 0,
    "preservation_success": 0,
    "physical_effect_success": 0,
    "temporal_consistency": 0,
    "major_artifacts": 1,
    "overall_pass": 0,
}


class ValidationError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path):
    if not path.exists():
        raise ValidationError(f"missing JSONL file: {path.relative_to(ROOT)}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(rel_path):
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValidationError(f"path must be relative to repo root: {rel_path}")
    resolved = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValidationError(f"path escapes repo root: {rel_path}")
    return resolved


def _type_matches(value, expected):
    if isinstance(expected, list):
        return any(_type_matches(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def validate_against_schema(obj, schema, where="$"):
    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(obj, expected_type):
        raise ValidationError(f"{where}: expected type {expected_type}, got {type(obj).__name__}")

    if "enum" in schema and obj not in schema["enum"]:
        raise ValidationError(f"{where}: expected one of {schema['enum']}, got {obj!r}")

    if isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]:
            raise ValidationError(f"{where}: string shorter than {schema['minLength']}")
        if "pattern" in schema and not re.match(schema["pattern"], obj):
            raise ValidationError(f"{where}: string does not match {schema['pattern']}")

    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            raise ValidationError(f"{where}: array shorter than {schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(obj):
                validate_against_schema(item, item_schema, f"{where}[{index}]")

    if isinstance(obj, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in obj:
                raise ValidationError(f"{where}: missing required field {key!r}")
        if "minProperties" in schema and len(obj) < schema["minProperties"]:
            raise ValidationError(f"{where}: object has fewer than {schema['minProperties']} properties")
        for key, child_schema in properties.items():
            if key in obj:
                validate_against_schema(obj[key], child_schema, f"{where}.{key}")
        additional = schema.get("additionalProperties")
        if additional is not None and additional is not True:
            for key, value in obj.items():
                if key in properties:
                    continue
                if additional is False:
                    raise ValidationError(f"{where}.{key}: additional property not allowed")
                validate_against_schema(value, additional, f"{where}.{key}")


def load_contract_for_manifest_row(row, validate=False):
    contract_path = resolve_repo_path(row["contract"])
    if not contract_path.exists():
        raise ValidationError(f"missing contract for {row['sample_id']}: {row['contract']}")
    contract = read_json(contract_path)
    if validate:
        schema = read_json(ROOT / "schemas" / "contract.schema.json")
        validate_against_schema(contract, schema, f"contract[{row['sample_id']}]")
    if contract["sample_id"] != row["sample_id"]:
        raise ValidationError(f"contract sample_id mismatch for {row['sample_id']}")
    if contract["operation"] != row["operation"]:
        raise ValidationError(f"contract operation mismatch for {row['sample_id']}")
    if contract["target_ref"] != row["target_ref"]:
        raise ValidationError(f"contract target_ref mismatch for {row['sample_id']}")
    return contract


def load_manifest(validate=False):
    manifest_path = ROOT / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    if not validate:
        return rows

    schema = read_json(ROOT / "schemas" / "manifest.schema.json")
    seen = set()
    for index, row in enumerate(rows, start=1):
        validate_against_schema(row, schema, f"manifest row {index}")
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValidationError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)

        source_path = resolve_repo_path(row["source_video"])
        if not source_path.exists():
            raise ValidationError(f"missing source video for {sample_id}: {row['source_video']}")
        if source_path.name != f"{sample_id}.mp4":
            raise ValidationError(f"source video must be named {sample_id}.mp4")

        annotations = row.get("annotations", {})
        for key in ("target_mask", "affected_mask", "preserve_mask"):
            mask_path = annotations.get(key)
            if mask_path is not None and not resolve_repo_path(mask_path).exists():
                raise ValidationError(f"missing {key} for {sample_id}: {mask_path}")

        load_contract_for_manifest_row(row, validate=True)

    provenance_rows = read_jsonl(ROOT / "annotations" / "provenance.jsonl")
    provenance_ids = {row.get("sample_id") for row in provenance_rows}
    missing_provenance = seen - provenance_ids
    if missing_provenance:
        raise ValidationError(f"missing provenance rows: {sorted(missing_provenance)}")
    return rows


def manifest_sha256():
    return sha256_file(ROOT / "manifest.jsonl")


def total_effect_count(contract):
    return len(contract.get("affected_regions", [])) + len(contract.get("counterfactual_state", {}))


def all_effect_labels(contract):
    labels = list(contract.get("affected_regions", []))
    labels.extend(contract.get("counterfactual_state", {}).values())
    return labels


def cmd_validate_manifest(_args):
    rows = load_manifest(validate=True)
    print(f"valid manifest rows: {len(rows)}")
    print(f"valid contracts: {len(rows)}")
    print(f"manifest_sha256: {manifest_sha256()}")
    return 0


def cmd_list(_args):
    rows = load_manifest(validate=True)
    print(f"samples: {len(rows)}")
    for key in ("split", "operation", "category", "difficulty", "identifiability"):
        counts = Counter(row.get(key, "<missing>") for row in rows)
        formatted = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"{key}: {formatted}")
    return 0


def validate_predictions(run_name):
    rows = load_manifest(validate=True)
    sample_ids = {row["sample_id"] for row in rows}
    run_dir = ROOT / "predictions" / run_name
    if not run_dir.exists():
        raise ValidationError(f"missing predictions run: predictions/{run_name}")
    videos_dir = run_dir / "videos"
    if not videos_dir.exists():
        raise ValidationError(f"missing predictions video directory: predictions/{run_name}/videos")

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise ValidationError(f"missing run_meta.json for {run_name}")
    run_meta = read_json(meta_path)
    missing_meta = RUN_META_REQUIRED - set(run_meta)
    if missing_meta:
        raise ValidationError(f"run_meta missing required fields: {sorted(missing_meta)}")
    if run_meta["run_name"] != run_name:
        raise ValidationError(f"run_meta run_name does not match directory: {run_meta['run_name']} != {run_name}")
    if run_meta["benchmark_version"] != BENCHMARK_VERSION:
        raise ValidationError(
            f"benchmark_version mismatch: {run_meta['benchmark_version']} != {BENCHMARK_VERSION}"
        )
    if run_meta["manifest_sha256"] != manifest_sha256():
        raise ValidationError("run_meta manifest_sha256 does not match current manifest.jsonl")
    if run_meta["num_samples"] != len(rows):
        raise ValidationError(f"run_meta num_samples should be {len(rows)}")

    predictions = read_jsonl(run_dir / "predictions.jsonl")
    pred_counts = Counter(row.get("sample_id") for row in predictions)
    duplicate_ids = sorted(sid for sid, count in pred_counts.items() if count > 1)
    if duplicate_ids:
        raise ValidationError(
            f"duplicate sample_id rows in predictions.jsonl: {duplicate_ids}"
        )
    prediction_ids = set(pred_counts)
    if prediction_ids != sample_ids:
        missing = sorted(sample_ids - prediction_ids)
        extra = sorted(prediction_ids - sample_ids)
        raise ValidationError(f"prediction sample mismatch; missing={missing}, extra={extra}")

    for pred in predictions:
        sample_id = pred["sample_id"]
        status = pred.get("status")
        if not status:
            raise ValidationError(f"prediction missing status: {sample_id}")
        video = pred.get("video")
        if status == "ok":
            expected = f"videos/{sample_id}.mp4"
            if video != expected:
                raise ValidationError(f"{sample_id}: video must be {expected}, got {video!r}")
            if not (run_dir / video).exists():
                raise ValidationError(f"{sample_id}: missing predicted video {video}")
        elif video is not None:
            raise ValidationError(f"{sample_id}: failed predictions must use video=null")
    return rows, predictions, run_meta, run_dir


def cmd_validate(args):
    _rows, predictions, _run_meta, _run_dir = validate_predictions(args.run)
    ok_count = sum(1 for row in predictions if row.get("status") == "ok")
    failed_count = len(predictions) - ok_count
    print(f"valid predictions: {len(predictions)}")
    print(f"ok: {ok_count}")
    print(f"failed: {failed_count}")
    return 0


def empty_score_row(sample_id, judge, status, reason):
    row = {
        "sample_id": sample_id,
        "judge": judge,
        "status": status,
        "missing": status != "ok",
        "effect_hits": [],
        "short_reason": reason,
    }
    row.update(JUDGE_FIELDS)
    return row


def normalize_score_row(raw, sample_id, judge, status, contract):
    row = empty_score_row(sample_id, judge, status, raw.get("short_reason", ""))
    row["missing"] = bool(raw.get("missing", status != "ok"))
    for field, default in JUDGE_FIELDS.items():
        value = raw.get(field, default)
        row[field] = 1 if value else 0
    effect_hits = raw.get("effect_hits", [])
    if isinstance(effect_hits, str):
        effect_hits = [effect_hits]
    row["effect_hits"] = list(effect_hits)
    row["effect_total"] = total_effect_count(contract)
    # target_success is the precondition gate (spec §5.2): if the edit did not
    # land, the consequence/physical signals are not meaningful. Zero them here
    # so every stored row is self-consistent — the same gate the VLM parser
    # applies — instead of relying on per_sample_metrics to mask them at
    # aggregation time (which left raw rows contradictory and skewed `agree`).
    if not row["target_success"]:
        row["physical_effect_success"] = 0
        row["effect_hits"] = []
    return row


def baseline_score_row(sample_id, judge, status, contract, baseline_type):
    if status != "ok":
        return empty_score_row(sample_id, judge, status, "prediction missing or failed")

    if baseline_type == "copy_source":
        raw = {
            "target_success": 0,
            "preservation_success": 1,
            "effect_hits": [],
            "physical_effect_success": 0,
            "temporal_consistency": 1,
            "major_artifacts": 0,
            "overall_pass": 0,
            "short_reason": "copy_source anchor preserves the source but does not perform the edit",
        }
    elif baseline_type == "free_regen":
        raw = {
            "target_success": 1,
            "preservation_success": 0,
            "effect_hits": all_effect_labels(contract),
            "physical_effect_success": 1,
            "temporal_consistency": 1,
            "major_artifacts": 0,
            "overall_pass": 0,
            "short_reason": "free_regen anchor ignores source identity and targets consequences",
        }
    else:
        raise ValidationError(f"unsupported baseline anchor: {baseline_type}")
    return normalize_score_row(raw, sample_id, judge, status, contract)


def import_judge_output(path, rows, predictions, judge):
    raw_rows = {row["sample_id"]: row for row in read_jsonl(path)}
    pred_by_id = {row["sample_id"]: row for row in predictions}
    scored = []
    for manifest_row in rows:
        sample_id = manifest_row["sample_id"]
        contract = load_contract_for_manifest_row(manifest_row)
        status = pred_by_id[sample_id].get("status", "failed")
        if sample_id in raw_rows:
            scored.append(normalize_score_row(raw_rows[sample_id], sample_id, judge, status, contract))
        else:
            scored.append(empty_score_row(sample_id, judge, status, "missing judge output"))
    return scored


def write_judge_meta(run_name, judge, run_meta, baseline_type=None, judge_model=None, extra=None):
    prompt_path = ROOT / "judges" / "vlm_prompts.jsonl"
    prompt_hash = sha256_file(prompt_path) if prompt_path.exists() else None
    if judge_model is None:
        judge_model = "deterministic_baseline_anchor" if baseline_type else "external"
    meta = {
        "run_name": run_name,
        "judge": judge,
        "created_at": utc_now(),
        "baseline_type": baseline_type,
        "judge_model": judge_model,
        "judge_version": BENCHMARK_VERSION,
        "temperature": 0,
        "prompt_sha256": prompt_hash,
        "source_run_model": run_meta.get("model_name"),
        "source_run_version": run_meta.get("model_version"),
    }
    if extra:
        meta.update(extra)
    write_json(ROOT / "results" / run_name / f"{judge}_judge_meta.json", meta)


def run_openrouter_judge(rows, predictions, run_dir, raw_path, args):
    """Score a real run with the OpenRouter (google/gemini-2.5-pro) VLM judge."""
    api_key = getattr(args, "api_key", None) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValidationError(
            "VLM judge needs an OpenRouter API key: set OPENROUTER_API_KEY (or pass "
            "--api-key). The only built-in VLM backend is OpenRouter with google/gemini-2.5-pro."
        )
    sys.path.insert(0, str(ROOT))
    try:
        from judges import vlm_judge
    except ImportError as exc:
        raise ValidationError(f"could not import judges/vlm_judge.py: {exc}") from exc

    pred_by_id = {row["sample_id"]: row for row in predictions}
    try:
        results = vlm_judge.judge_predictions(
            ROOT, rows, pred_by_id, run_dir, api_key,
            model=args.judge_model, num_frames=args.frames,
            log=lambda message: print(message, file=sys.stderr),
        )
    except vlm_judge.JudgeError as exc:
        raise ValidationError(str(exc)) from exc
    vlm_judge.write_raw(raw_path, results)
    return results


def cmd_score(args):
    rows, predictions, run_meta, run_dir = validate_predictions(args.run)
    results_dir = ROOT / "results" / args.run
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.judge == "human" and not args.judge_output:
        return launch_human_ui(args.run)

    judge_model = None
    judge_extra = None
    if args.judge_output:
        scored = import_judge_output(Path(args.judge_output), rows, predictions, args.judge)
        baseline_type = None
    else:
        baseline_type = run_meta.get("baseline_type")
        if baseline_type in {"copy_source", "free_regen"}:
            pred_by_id = {row["sample_id"]: row for row in predictions}
            scored = []
            for manifest_row in rows:
                sample_id = manifest_row["sample_id"]
                contract = load_contract_for_manifest_row(manifest_row)
                status = pred_by_id[sample_id].get("status", "failed")
                scored.append(baseline_score_row(sample_id, args.judge, status, contract, baseline_type))
        elif args.judge == "vlm":
            raw_path = results_dir / "vlm_raw_judge.jsonl"
            run_openrouter_judge(rows, predictions, run_dir, raw_path, args)
            scored = import_judge_output(raw_path, rows, predictions, args.judge)
            baseline_type = None
            judge_model = f"openrouter/{args.judge_model}"
            judge_extra = {
                "backend": "openrouter",
                "frames_per_video": args.frames,
                "raw_judge_output": "vlm_raw_judge.jsonl",
            }
        else:
            raise ValidationError(
                "human judging uses the built-in UI (omit --judge-output) or pass "
                "--judge-output with reproducible judge JSONL"
            )

    output_name = "human_per_sample.jsonl" if args.judge == "human" else "per_sample.jsonl"
    write_jsonl(results_dir / output_name, scored)
    write_judge_meta(args.run, args.judge, run_meta, baseline_type, judge_model, judge_extra)
    print(f"wrote results: results/{args.run}/{output_name}")
    return 0


def per_sample_metrics(score_row, contract):
    missing = bool(score_row.get("missing")) or score_row.get("status") != "ok"
    if missing:
        return {
            "preservation_axis": 0.0,
            "consequence_axis": 0.0,
            "physical_effect": 0.0,
            "edit_success": 0.0,
            "quality": 0.0,
            "missing": True,
        }

    target_success = 1 if score_row.get("target_success") else 0
    preservation = 1 if score_row.get("preservation_success") else 0
    effect_total = score_row.get("effect_total") or total_effect_count(contract)
    effect_hits = score_row.get("effect_hits", [])
    consequence = 0.0
    if target_success and effect_total:
        consequence = min(len(effect_hits), effect_total) / effect_total
    physical = float(score_row.get("physical_effect_success", 0)) if target_success else 0.0
    temporal = float(score_row.get("temporal_consistency", 0))
    artifact_clean = 1.0 - float(score_row.get("major_artifacts", 1))
    quality = (temporal + artifact_clean) / 2.0
    return {
        "preservation_axis": float(preservation),
        "consequence_axis": float(consequence),
        "physical_effect": physical,
        "edit_success": float(target_success),
        "quality": quality,
        "missing": False,
    }


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def rounded(value):
    return round(float(value), 6)


def aggregate_metrics(metrics):
    n = len(metrics)
    missing = sum(1 for row in metrics if row["missing"])
    return {
        "n": n,
        "missing": missing,
        "failure_rate": rounded(missing / n) if n else 0.0,
        "preservation_axis": rounded(mean(row["preservation_axis"] for row in metrics)),
        "consequence_axis": rounded(mean(row["consequence_axis"] for row in metrics)),
        "physical_effect": rounded(mean(row["physical_effect"] for row in metrics)),
        "edit_success": rounded(mean(row["edit_success"] for row in metrics)),
        "quality": rounded(mean(row["quality"] for row in metrics)),
    }


def per_sample_filename(judge):
    return "human_per_sample.jsonl" if judge == "human" else "per_sample.jsonl"


def load_primary_scores(run_name, judge="vlm"):
    filename = per_sample_filename(judge)
    result_path = ROOT / "results" / run_name / filename
    if not result_path.exists():
        hint = ""
        other_judge = "vlm" if judge == "human" else "human"
        if (ROOT / "results" / run_name / per_sample_filename(other_judge)).exists():
            hint = f" (found {other_judge} results; try: report {run_name} --judge {other_judge})"
        raise ValidationError(
            f"missing {judge} per-sample results: results/{run_name}/{filename}{hint}"
        )
    return read_jsonl(result_path)


def build_summary(run_name, score_rows, judge="vlm"):
    manifest_rows = load_manifest(validate=True)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    score_by_id = {row["sample_id"]: row for row in score_rows}

    metrics_by_id = {}
    for sample_id, manifest_row in manifest_by_id.items():
        contract = load_contract_for_manifest_row(manifest_row)
        score_row = score_by_id.get(sample_id)
        if score_row is None:
            score_row = empty_score_row(sample_id, judge, "failed", "missing per-sample result")
        metrics_by_id[sample_id] = per_sample_metrics(score_row, contract)

    summary = aggregate_metrics(metrics_by_id.values())
    summary["run_name"] = run_name
    summary["judge"] = judge
    summary["benchmark_version"] = BENCHMARK_VERSION
    summary["created_at"] = utc_now()
    summary["保不变量"] = summary["preservation_axis"]
    summary["命中后果"] = summary["consequence_axis"]
    summary["物理可信"] = summary["physical_effect"]
    summary["编辑落地"] = summary["edit_success"]
    summary["质量"] = summary["quality"]

    for group_key in ("category", "operation", "difficulty"):
        grouped = defaultdict(list)
        for row in manifest_rows:
            grouped[row[group_key]].append(metrics_by_id[row["sample_id"]])
        summary[f"by_{group_key}"] = {
            group: aggregate_metrics(group_metrics)
            for group, group_metrics in sorted(grouped.items())
        }

    agreement_path = ROOT / "results" / run_name / "agreement.json"
    if agreement_path.exists():
        summary["human_vlm_agreement"] = read_json(agreement_path)

    return summary


def write_leaderboard(run_name, summary):
    agreement = summary.get("human_vlm_agreement", {})
    agreement_value = agreement.get("overall_accuracy", "")
    lines = [
        "# CF-VEdit Leaderboard",
        "",
        "| run | judge | 保不变量 | 命中后果 | physical | edit_success | quality | n | missing | failure_rate | human_vlm_accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {run_name} | {summary.get('judge', 'vlm')} | "
            f"{summary['preservation_axis']:.3f} | "
            f"{summary['consequence_axis']:.3f} | {summary['physical_effect']:.3f} | "
            f"{summary['edit_success']:.3f} | {summary['quality']:.3f} | "
            f"{summary['n']} | {summary['missing']} | {summary['failure_rate']:.3f} | "
            f"{agreement_value} |"
        ),
        "",
        "Scope notes: pair examples and broader edit types are reserved fields in v0.1.0.",
    ]
    path = ROOT / "results" / run_name / "leaderboard.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_report(args):
    score_rows = load_primary_scores(args.run, args.judge)
    summary = build_summary(args.run, score_rows, args.judge)
    write_json(ROOT / "results" / args.run / "summary.json", summary)
    write_leaderboard(args.run, summary)
    print(f"wrote summary: results/{args.run}/summary.json  (judge: {args.judge})")
    print(f"wrote leaderboard: results/{args.run}/leaderboard.md")
    return 0


def binary_kappa(pairs):
    if not pairs:
        return 0.0
    total = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / total
    left_yes = sum(1 for left, _right in pairs if left) / total
    right_yes = sum(1 for _left, right in pairs if right) / total
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def cmd_agree(args):
    vlm_path = ROOT / "results" / args.run / "per_sample.jsonl"
    human_path = ROOT / "results" / args.run / "human_per_sample.jsonl"
    vlm_rows = {row["sample_id"]: row for row in read_jsonl(vlm_path)}
    human_rows = {row["sample_id"]: row for row in read_jsonl(human_path)}
    shared_ids = sorted(set(vlm_rows) & set(human_rows))
    if not shared_ids:
        raise ValidationError("no overlapping human and VLM samples")

    fields = ["target_success", "preservation_success", "physical_effect_success", "overall_pass"]
    by_field = {}
    all_pairs = []
    for field in fields:
        pairs = [
            (1 if vlm_rows[sample_id].get(field) else 0, 1 if human_rows[sample_id].get(field) else 0)
            for sample_id in shared_ids
        ]
        all_pairs.extend(pairs)
        by_field[field] = {
            "accuracy": rounded(mean(1.0 if left == right else 0.0 for left, right in pairs)),
            "kappa": rounded(binary_kappa(pairs)),
            "n": len(pairs),
        }

    agreement = {
        "run_name": args.run,
        "n": len(shared_ids),
        "overall_accuracy": rounded(mean(1.0 if left == right else 0.0 for left, right in all_pairs)),
        "overall_kappa": rounded(binary_kappa(all_pairs)),
        "by_field": by_field,
    }
    write_json(ROOT / "results" / args.run / "agreement.json", agreement)
    print(f"wrote agreement: results/{args.run}/agreement.json")
    return 0


HUMAN_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>CF-VEdit Human Judge</title>
<style>
body{font-family:-apple-system,system-ui,Segoe UI,sans-serif;margin:0;padding:16px;background:#0f1115;color:#e6e6e6}
#nav{display:flex;gap:8px;align-items:center;margin:8px 0}
#status{margin-left:auto;color:#9ca3af}
.videos{display:flex;gap:16px;flex-wrap:wrap}
.videos figure{margin:0}
video{width:420px;max-width:46vw;background:#000;border-radius:8px}
h3{margin:4px 0}
.box{background:#1a1d24;border:1px solid #2a2f3a;border-radius:10px;padding:12px 16px;margin:12px 0}
label.chk{display:block;margin:4px 0}
.flags label{margin-right:18px;display:inline-block}
button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:8px 16px;font-size:14px;cursor:pointer}
button.secondary{background:#374151}
textarea{width:100%;background:#0f1115;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:8px;padding:8px;box-sizing:border-box}
.muted{color:#9ca3af;font-size:13px}
</style></head><body>
<div id="nav">
  <button class="secondary" onclick="go(-1)">&larr; Prev</button>
  <button class="secondary" onclick="go(1)">Next &rarr;</button>
  <span id="counter"></span><span id="status"></span>
</div>
<div id="meta"></div>
<div class="videos">
  <figure><h3>Source</h3><video id="src" controls loop muted></video></figure>
  <figure><h3>Edited</h3><video id="edt" controls loop muted></video></figure>
</div>
<div class="box"><b>Preserve regions unchanged</b><div id="preserve"></div></div>
<div class="box"><b>Counterfactual effects hit</b><div id="effects"></div></div>
<div class="box flags"><b>Flags</b><br>
  <label><input type="checkbox" id="target_success"> Target edit landed</label>
  <label><input type="checkbox" id="physical_effect_success"> Physical effect correct</label>
  <label><input type="checkbox" id="temporal_consistency"> Temporal consistency</label>
  <label><input type="checkbox" id="major_artifacts"> Major artifacts</label>
  <label><input type="checkbox" id="overall_pass"> Overall pass</label>
</div>
<div class="box"><b>Short reason</b><br><textarea id="short_reason" rows="2"></textarea></div>
<div id="nav"><button onclick="save(true)">Save &amp; next</button>
  <button class="secondary" onclick="save(false)">Save</button></div>
<script>
let samples=[], i=0;
function el(id){return document.getElementById(id);}
async function load(){samples=await (await fetch('api/samples')).json(); render();}
function checks(container, items, checked){
  const c=el(container); c.innerHTML='';
  if(!items.length){c.innerHTML='<span class="muted">none</span>'; return;}
  items.forEach(function(it){
    const l=document.createElement('label'); l.className='chk';
    const cb=document.createElement('input'); cb.type='checkbox'; cb.value=it;
    cb.checked=checked.indexOf(it)>=0;
    l.appendChild(cb); l.appendChild(document.createTextNode(' '+it)); c.appendChild(l);
  });
}
function getChecks(container){
  return Array.prototype.slice.call(el(container).querySelectorAll('input:checked')).map(function(x){return x.value;});
}
function render(){
  const s=samples[i];
  el('counter').textContent=(i+1)+' / '+samples.length+'  ['+s.sample_id+']';
  el('meta').innerHTML='<div class="box"><div><b>'+s.operation+'</b> &mdash; '+s.instruction+'</div>'
    +'<div class="muted">target: '+s.target_ref+' &middot; prediction status: '+s.status+'</div></div>';
  el('src').src='video/source/'+s.sample_id;
  if(s.has_edited){el('edt').src='video/edited/'+s.sample_id; el('edt').style.display='';}
  else {el('edt').removeAttribute('src'); el('edt').style.display='none';}
  checks('preserve', s.preserve_regions, s.existing.preserve_hits);
  checks('effects', s.effects, s.existing.effect_hits);
  ['target_success','physical_effect_success','temporal_consistency','major_artifacts','overall_pass']
    .forEach(function(k){el(k).checked=!!s.existing[k];});
  el('short_reason').value=s.existing.short_reason||'';
  setStatus(s.saved_ids);
}
function setStatus(ids){el('status').textContent=ids.length+' / '+samples.length+' labeled';}
function go(d){i=Math.max(0,Math.min(samples.length-1,i+d)); render();}
async function save(next){
  const s=samples[i];
  const payload={sample_id:s.sample_id, preserve_hits:getChecks('preserve'), effect_hits:getChecks('effects'),
    target_success:el('target_success').checked, physical_effect_success:el('physical_effect_success').checked,
    temporal_consistency:el('temporal_consistency').checked, major_artifacts:el('major_artifacts').checked,
    overall_pass:el('overall_pass').checked, short_reason:el('short_reason').value};
  const r=await (await fetch('api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)})).json();
  if(r.ok){
    samples[i].existing={preserve_hits:payload.preserve_hits, effect_hits:payload.effect_hits,
      target_success:payload.target_success, physical_effect_success:payload.physical_effect_success,
      temporal_consistency:payload.temporal_consistency, major_artifacts:payload.major_artifacts,
      overall_pass:payload.overall_pass, short_reason:payload.short_reason};
    samples.forEach(function(x){x.saved_ids=r.saved_ids;});
    setStatus(r.saved_ids); if(next) go(1);
  } else { alert('save failed: '+r.error); }
}
load();
</script></body></html>
"""


def launch_human_ui(run_name, host="127.0.0.1", port=0, open_browser=True):
    """Serve a dependency-free local web UI for human judging."""
    import http.server
    import socketserver
    import webbrowser
    from urllib.parse import urlparse

    rows, predictions, _run_meta, run_dir = validate_predictions(run_name)
    pred_by_id = {row["sample_id"]: row for row in predictions}
    manifest_by_id = {row["sample_id"]: row for row in rows}
    output_path = ROOT / "results" / run_name / "human_per_sample.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_saved():
        return {row["sample_id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}

    def sample_view(index, saved):
        manifest_row = rows[index]
        sample_id = manifest_row["sample_id"]
        contract = load_contract_for_manifest_row(manifest_row)
        prediction = pred_by_id[sample_id]
        existing = saved.get(sample_id, {})
        return {
            "index": index,
            "sample_id": sample_id,
            "instruction": manifest_row["instruction"],
            "operation": manifest_row["operation"],
            "target_ref": manifest_row["target_ref"],
            "preserve_regions": contract["preserve_regions"],
            "effects": all_effect_labels(contract),
            "status": prediction.get("status"),
            "has_edited": bool(prediction.get("video")),
            "saved_ids": sorted(saved.keys()),
            "existing": {
                "preserve_hits": existing.get("preserve_hits", []),
                "effect_hits": existing.get("effect_hits", []),
                "target_success": bool(existing.get("target_success")),
                "physical_effect_success": bool(existing.get("physical_effect_success")),
                "temporal_consistency": bool(existing.get("temporal_consistency")),
                "major_artifacts": bool(existing.get("major_artifacts")),
                "overall_pass": bool(existing.get("overall_pass")),
                "short_reason": existing.get("short_reason", ""),
            },
        }

    def save_label(payload):
        sample_id = payload["sample_id"]
        manifest_row = manifest_by_id[sample_id]
        contract = load_contract_for_manifest_row(manifest_row)
        preserve_choices = payload.get("preserve_hits", [])
        row = normalize_score_row(
            {
                "target_success": payload.get("target_success"),
                "preservation_success": set(preserve_choices) == set(contract["preserve_regions"]),
                "effect_hits": payload.get("effect_hits", []),
                "physical_effect_success": payload.get("physical_effect_success"),
                "temporal_consistency": payload.get("temporal_consistency"),
                "major_artifacts": payload.get("major_artifacts"),
                "overall_pass": payload.get("overall_pass"),
                "short_reason": payload.get("short_reason", ""),
            },
            sample_id,
            "human",
            pred_by_id[sample_id].get("status", "failed"),
            contract,
        )
        row["preserve_hits"] = list(preserve_choices)
        saved = load_saved()
        saved[sample_id] = row
        ordered = [saved[r["sample_id"]] for r in rows if r["sample_id"] in saved]
        write_jsonl(output_path, ordered)
        return {"ok": True, "saved_ids": sorted(saved.keys())}

    def serve_video(handler, video_path):
        if not video_path.exists():
            handler.send_error(404)
            return
        file_size = video_path.stat().st_size
        range_header = handler.headers.get("Range")
        start, end = 0, file_size - 1
        partial = bool(range_header and range_header.startswith("bytes="))
        if partial:
            spec = range_header[len("bytes="):].split("-")
            start = int(spec[0]) if spec[0] else 0
            if len(spec) > 1 and spec[1]:
                end = min(int(spec[1]), file_size - 1)
        length = end - start + 1
        handler.send_response(206 if partial else 200)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Accept-Ranges", "bytes")
        if partial:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        handler.send_header("Content-Length", str(length))
        handler.end_headers()
        with video_path.open("rb") as handle:
            handle.seek(start)
            handler.wfile.write(handle.read(length))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                body = HUMAN_UI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/samples":
                saved = load_saved()
                self._send_json([sample_view(i, saved) for i in range(len(rows))])
            elif route.startswith("/video/source/"):
                manifest_row = manifest_by_id.get(route[len("/video/source/"):])
                if not manifest_row:
                    self.send_error(404)
                else:
                    serve_video(self, resolve_repo_path(manifest_row["source_video"]))
            elif route.startswith("/video/edited/"):
                prediction = pred_by_id.get(route[len("/video/edited/"):])
                if not prediction or not prediction.get("video"):
                    self.send_error(404)
                else:
                    serve_video(self, run_dir / prediction["video"])
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(save_label(payload))
            except Exception as exc:  # surfaced to the browser, never crashes the server
                self._send_json({"ok": False, "error": str(exc)}, code=400)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server((host, port), Handler)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"CF-VEdit human judge for run '{run_name}'")
    print(f"open {url}")
    print(f"labels save to results/{run_name}/human_per_sample.jsonl  (Ctrl-C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping human judge")
    finally:
        server.server_close()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="CF-VEdit benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_manifest = subparsers.add_parser("validate-manifest")
    validate_manifest.set_defaults(func=cmd_validate_manifest)

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(func=cmd_list)

    validate = subparsers.add_parser("validate")
    validate.add_argument("run")
    validate.set_defaults(func=cmd_validate)

    score = subparsers.add_parser("score")
    score.add_argument("run")
    score.add_argument("--judge", choices=["vlm", "human"], required=True)
    score.add_argument(
        "--judge-output",
        help="JSONL file with judge rows using the shared per-sample schema "
        "(overrides the built-in OpenRouter VLM / human UI judge)",
    )
    score.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key for the VLM judge (defaults to OPENROUTER_API_KEY)",
    )
    score.add_argument(
        "--judge-model",
        default="google/gemini-2.5-pro",
        help="OpenRouter model slug for the VLM judge",
    )
    score.add_argument(
        "--frames",
        type=int,
        default=8,
        help="frames sampled per video for the VLM judge",
    )
    score.set_defaults(func=cmd_score)

    report = subparsers.add_parser("report")
    report.add_argument("run")
    report.add_argument(
        "--judge",
        choices=["vlm", "human"],
        default="vlm",
        help="which judge's per-sample results to aggregate (default: vlm)",
    )
    report.set_defaults(func=cmd_report)

    agree = subparsers.add_parser("agree")
    agree.add_argument("run")
    agree.set_defaults(func=cmd_agree)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
