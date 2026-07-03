# AGENTS.md

This file provides guidance to Codex (and other coding agents) when working with code in this repository.

## What this repo is

Two layers live here, and they are at very different maturity:

1. **Design docs at the root** (`*.md`, written in Chinese) — the "proposal-as-truth" for **CF-VEdit / E2W**, a counterfactual video-editing system. The monorepo they describe (`e2w/` with `e2w_core`, `localization`, `generation`, `data_engine`, `integration`) is **scaffolded** — 5 packages with code, tests, and ADRs 0001–0007. Current status: `e2w/README.md`; decisions: `e2w/docs/adr/`; spec-vs-implemented: `e2w/docs/TRACEABILITY.md`.
2. **`physics_iq_for_simple_eval/`** — mature, fully-working code. The **P0 benchmark** (`cf_vedit_bench`): 12 Physics-IQ clips packaged as a pluggable, file-based evaluation suite for *any* video-editing model.

The deliberate sequencing (`CF-VEdit-Repo-Design.md` §5): **build the ruler (benchmark) before the machine (model)**, so the model is always measured against a spec it cannot quietly drift from. The model half is now underway (v0 remove-only, ADR-0007) — check `e2w/README.md` and the ADRs before assuming what's built.

## Guardrail — don't casually change existing rules

This repo has real invariants enforced by code/tests, not just prose. Before editing near one of these, stop and confirm with the user rather than just "fixing" it — a failing check usually means your change is wrong, not the check:

- `physics_iq_for_simple_eval/tests/test_cf_vedit_benchmark.py` — spec-as-test for the benchmark.
- Benchmark read-only assets (`manifest.jsonl`, `contracts/`, `videos/source/`, `annotations/`, `judges/`, `schemas/`) — a run must never write here; see B2 below.
- `bench.py`'s offline-baseline branch in `cmd_score` (copy_source/free_regen) — keeps tests runnable without network/API key.
- e2w's import-linter graph (`e2w/pyproject.toml` `[tool.importlinter]`) — `e2w_core` has no deps; other packages must not depend on each other.
- Anything already covered by an ADR in `e2w/docs/adr/` — a deviation needs a new ADR, not a silent edit.

If unsure whether something is load-bearing, ask instead of assuming it's dead weight.

### The design docs (read these before changing the benchmark's shape)

- `E2W-v0-Remove-Only-Spec.md` — **the current authoritative build spec for the model** (remove-only, frozen CogVideoX-Fun/VOID renderer, query-token planner). Read this first if you're building or reviewing `e2w/`. Supersedes the architecture/novelty content of the four docs below for v0 — see `e2w/docs/adr/0007-e2w-v0-remove-only-void-renderer.md`.
- `Counterfactual-Video-Editing-Proposal.md` — the original research proposal (long-run open-domain thesis; superseded for v0, kept as historical record).
- `CF-VEdit-Architecture-and-Narrative (给人看的）.md` — human-facing architecture + naming (same superseded status).
- `CF-VEdit-Benchmark-Spec.md` — **the executable spec for the benchmark below.** Every structural rule in `physics_iq_for_simple_eval/` traces to a section here. Not affected by the v0 pivot.
- `Sa2VA-Modification-Plan.md` — original plan for the localization half (Sa2VA deltas); §1 changes A/B carry into v0 respecified, §2–3 (VACE/Wan) do not.
- `CF-VEdit-Repo-Design.md` — boundaries, reuse strategy, and anti-drift mechanisms for the full monorepo. Not affected by the v0 pivot.

## Working in the benchmark (`physics_iq_for_simple_eval/`)

Pure Python 3 standard library — no `pyproject.toml`, no `requirements.txt`, no pip dependencies, no build step. The two external requirements are both for the VLM judge only: **ffmpeg** on `PATH` (frame extraction) and an `OPENROUTER_API_KEY` (the network call uses `urllib`, no SDK). The human judge and everything else are stdlib-only. **All commands must be run from inside `physics_iq_for_simple_eval/`** — `bench.py` resolves every path relative to its own location.

```bash
cd physics_iq_for_simple_eval

# Tests (pytest is NOT installed; use unittest)
python3 -m unittest tests.test_cf_vedit_benchmark -v
python3 -m unittest tests.test_cf_vedit_benchmark.CfVEditBenchmarkShapeTest.test_manifest_is_lightweight_and_points_to_external_assets  # single test

# Benchmark CLI
python3 bench.py validate-manifest          # schema-check manifest.jsonl + every contract
python3 bench.py list                        # sample/operation/category/difficulty/split counts
python3 bench.py validate <run_name>         # check predictions/<run_name>/ is complete + well-named
python3 bench.py score <run_name> --judge vlm     # OpenRouter/Gemini judge -> results/<run>/per_sample.jsonl (needs OPENROUTER_API_KEY + ffmpeg)
python3 bench.py score <run_name> --judge human   # launch built-in stdlib web UI (or --judge-output <jsonl>)
python3 bench.py report <run_name>           # write summary.json + leaderboard.md
python3 bench.py agree <run_name>            # human↔VLM accuracy + Cohen's κ

# End-to-end smoke test (lower-bound baseline)
python3 baselines/copy_source.py --run-name copy_source
python3 bench.py validate copy_source && python3 bench.py score copy_source --judge vlm && python3 bench.py report copy_source
```

### Architecture: file-based contract, not function calls

The benchmark **never imports or calls editing models.** A model is integrated by writing files to disk, then `bench.py` consumes directories. The data flow:

```
manifest.jsonl ─┬─► contracts/<sample_id>.json   (the counterfactual contract: what must change / stay)
                ├─► videos/source/<sample_id>.mp4
                └─► annotations/provenance.jsonl  (source + leakage evidence)
                         │
   model runs externally, writes ► predictions/<run_name>/{videos/, predictions.jsonl, run_meta.json}
                         │
   bench.py score/report consumes ► results/<run_name>/{per_sample.jsonl, summary.json, leaderboard.md}
```

**Read-only assets vs. run outputs is a hard boundary.** `manifest.jsonl`, `contracts/`, `videos/source/`, `annotations/`, `judges/`, `schemas/` are inputs and must not be mutated by a run. Everything a model produces goes under `predictions/<run>/`; everything scoring produces goes under `results/<run>/`.

### Invariants enforced by `tests/test_cf_vedit_benchmark.py` (spec-as-test)

This test file is the **executable spec** — when it goes red, the benchmark has drifted from `CF-VEdit-Benchmark-Spec.md`. The invariants it (and `bench.py`) protect:

- **Two-axis metrics never collapse into a single total score.** Every summary carries both `preservation_axis` / `保不变量` (preserve regions untouched) and `consequence_axis` / `命中后果` (required effects hit). `summary.json` keys are bilingual (also `物理可信`, `编辑落地`, `质量`).
- **`copy_source` must anchor at preservation≈1, consequence≈0, edit_success≈0** (lower bound); `free_regen` is the inverse (upper bound). A baseline that lands off the diagonal means the metric code has a bug.
- **manifest stays lightweight** — heavy info (contract, masks, provenance) is external; a set of forbidden fat fields (`expected_physical_effect`, `must_preserve`, `vlm_judge_prompt`, …) must not appear inline.
- **Scoring gates on `target_success`:** if the edit did not land, consequence and physical scores are forced to 0. Failed predictions (`status != "ok"`, `video: null`) count toward `failure_rate` and score 0.

### Scoring backends

Two judges, selected with `--judge`, both emitting the shared per-sample schema (`bench.py` `JUDGE_FIELDS`) so `agree` can cross-validate them:

- **`--judge vlm`** → OpenRouter, model `google/gemini-2.5-pro` (`judges/vlm_judge.py`). Requires `OPENROUTER_API_KEY` (or `--api-key`) and **ffmpeg** on `PATH`. Videos are sent as `--frames` evenly sampled JPEG frames (source then edited) via the OpenAI-compatible chat API (`urllib`, no SDK). Raw verdicts are persisted to `results/<run>/vlm_raw_judge.jsonl`, then imported through the same path as `--judge-output`. **Caveat:** temporal consistency is judged from sampled frames, not native video.
- **`--judge human`** → a dependency-free local web UI (`launch_human_ui` in `bench.py`, Python stdlib `http.server` only — gradio was removed). Source/edited videos side by side (HTTP Range supported), checkboxes for preserve regions + effects, streams to `results/<run>/human_per_sample.jsonl`, resumable.

Deterministic baseline runs (`copy_source`/`free_regen`, detected via `baseline_type` in `run_meta.json`) are scored **offline** with fixed anchor rows — no API key, no network. This is what keeps the smoke test and unit tests runnable offline; preserve that branch when touching `cmd_score`. `--judge-output <jsonl>` still works as an escape hatch that bypasses both built-in judges.

### `run_meta.json` is the reproducibility lock

`validate <run>` rejects a run whose `run_meta.json` is missing any of `RUN_META_REQUIRED`, whose `benchmark_version` ≠ `BENCHMARK_VERSION` (currently `0.1.0`, in `bench.py`), or whose `manifest_sha256` ≠ the live manifest hash. Editing `manifest.jsonl` therefore invalidates every prior run — regenerate baselines after any manifest change.

## Working in `e2w/`

Scaffolded monorepo. Read `e2w/AGENTS.md` (the constitution — five boundaries + change discipline) and `e2w/README.md` first. Status drifts fast — check `e2w/docs/adr/` before assuming what's built. The five boundaries B1–B5 are import-linter/CI enforced; don't route around them.

## Boundaries & anti-drift (when extending toward the full monorepo)

The design docs encode five hard boundaries (`CF-VEdit-Repo-Design.md` §2) meant to become CI-enforced (import-linter, schema checks). The two already live in the benchmark:

- **B1 — benchmark ↔ model:** the benchmark consumes the `predictions/` directory and must never import model code.
- **B2 — data assets ↔ run outputs:** the read-only/produced split above.

Others (B3 localization↔generation, B4 vendored upstream untouched, B5 train/eval source disjoint) apply as the model packages mature. The intended discipline: **change `docs/proposal/` before changing code; any deviation gets an ADR; reserved scope** (pair examples `pair_id`, edit breadth `attribute`/`force_event`) **stays as placeholder fields, not half-built features.**
