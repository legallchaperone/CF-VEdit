# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# User Preferences

## Style Be terse. 
Drop articles and filler. Sacrifice Grammar for concise. 

## Planning in plan mode 
At the end of your plan, list uncertainties needing my confirmation before proceeding.

## What this repo is

Two layers, very different maturity:

1. **Design docs at the root** (`*.md`, Chinese) — proposal-as-truth for **CF-VEdit / E2W**, a counterfactual video-editing system. The monorepo they describe (`e2w/`) is scaffolded — see `e2w/README.md` for current status, `e2w/docs/adr/` for decisions, `e2w/docs/TRACEABILITY.md` for spec-vs-implemented.
2. **`physics_iq_for_simple_eval/`** — mature, fully-working code. The **P0 benchmark** (`cf_vedit_bench`). Details: `physics_iq_for_simple_eval/README.md` + `CF_VEDIT_BENCHMARK_SPEC.md`.

Sequencing (`CF-VEdit-Repo-Design.md` §5): **ruler (benchmark) before machine (model)**.

## Guardrail — don't casually change existing rules

This repo has real invariants enforced by code/tests, not just prose. Before editing near one of these, stop and confirm with the user rather than just "fixing" it — a failing check usually means your change is wrong, not the check:

- `physics_iq_for_simple_eval/tests/test_cf_vedit_benchmark.py` — spec-as-test for the benchmark.
- Benchmark read-only assets (`manifest.jsonl`, `contracts/`, `videos/source/`, `annotations/`, `judges/`, `schemas/`) — a run must never write here; see B2 below.
- `bench.py`'s offline-baseline branch in `cmd_score` (copy_source/free_regen) — keeps tests runnable without network/API key.
- e2w's import-linter graph (`e2w/pyproject.toml` `[tool.importlinter]`) — `e2w_core` has no deps; other packages must not depend on each other.
- Anything already covered by an ADR in `e2w/docs/adr/` — a deviation needs a new ADR, not a silent edit.

If unsure whether something is load-bearing, ask instead of assuming it's dead weight.

### Design docs (read before changing benchmark/model shape)

- `E2W-v0-Remove-Only-Spec.md` — **current authoritative model build spec** (remove-only, frozen CogVideoX-Fun/VOID `void_pass1` renderer). Supersedes proposal/architecture/sa2va-plan's architecture content for v0, see `e2w/docs/adr/0007-*.md`.
- `Counterfactual-Video-Editing-Proposal.md` — original research proposal, long-run thesis, superseded for v0
- `CF-VEdit-Architecture-and-Narrative (给人看的）.md` — human-facing architecture + naming, superseded for v0
- `CF-VEdit-Benchmark-Spec.md` — executable spec source for the benchmark, unaffected by v0 pivot
- `Sa2VA-Modification-Plan.md` — localization half plan, §1 carries into v0 respecified, §2-3 (VACE/Wan) don't
- `CF-VEdit-Repo-Design.md` — boundaries, reuse strategy, anti-drift for the monorepo, unaffected by v0 pivot

## Working in `physics_iq_for_simple_eval/`

Details (assets, model output contract, invariants, scoring backends) live in `README.md` and `CF_VEDIT_BENCHMARK_SPEC.md` — read those, not this file, for the specifics. Agent-specific gotchas not covered there:

- Pure stdlib, no deps except the VLM judge (ffmpeg + `OPENROUTER_API_KEY`).
- **All commands run from inside this dir** — `bench.py` resolves paths relative to itself.
- pytest not installed — use `python3 -m unittest tests.test_cf_vedit_benchmark`.
- `copy_source` must anchor preservation≈1/consequence≈0/edit_success≈0 (lower bound); `free_regen` is the inverse (upper bound). Off-diagonal result means a metric bug, not a real score — don't "adjust" the anchor to make it pass.

## Working in `e2w/`

Skeleton monorepo. Details: `e2w/README.md`, `e2w/AGENTS.md` (constitution), `e2w/docs/adr/`. Status drifts fast — check the ADRs before assuming what's built.

## Boundaries & anti-drift

Five hard boundaries (`CF-VEdit-Repo-Design.md` §2); B1/B2 are already enforced in the benchmark:

- **B1** — benchmark never imports model code, only consumes `predictions/`.
- **B2** — read-only assets vs. run outputs (see Guardrail above).
- B3–B5 (localization↔generation, vendored upstream untouched, train/eval source disjoint) apply once the model packages mature.

Discipline: change `docs/proposal/` before code; any deviation gets an ADR; reserved scope stays placeholder fields, not half-built features.
