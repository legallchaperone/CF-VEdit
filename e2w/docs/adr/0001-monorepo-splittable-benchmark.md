# ADR-0001 — Monorepo with a splittable benchmark

- **Status:** Accepted
- **Date:** 2026-06-19
- **Anchors:** Repo-Design §1, §2, §5; benchmark-spec §1; proposal P0

## Context

The proposal wants two things that pull in opposite directions:

1. **The benchmark must be independently releasable** (P0: "ship the ruler on its
   own"), so other groups can score Bernini / VEGGIE / VOID without our model.
2. **The model and benchmark must share contracts** — the three-layer mask, the
   edit tokens, the source latent, and the `predictions/` IO shape. Two repos
   would let those contracts drift apart silently.

We also need the five boundaries (B1–B5) to be physically enforceable, not just
documented.

## Decision

A single **monorepo** with the benchmark as a **splittable subpackage**:

- `packages/e2w_core` is the only shared contract layer (the "narrow waist"); it
  depends on nothing internal.
- `packages/cf_vedit_bench` has its own `pyproject.toml` and depends only on a
  thin schema subset of `e2w_core` (or zero internal deps), so it can be pip-
  installed / released alone.
- Dependency direction is one-way toward `e2w_core` and is machine-checked by
  import-linter (`pyproject.toml [tool.importlinter]`): B1 (benchmark imports no
  model) and B3 (the two halves are independent).

## Consequences

- **+** One place for shared contracts; anti-drift CI (import-linter + schema +
  spec-test) becomes a merge precondition.
- **+** The benchmark stays releasable: its only coupling is to `e2w_core`, which
  we keep deliberately minimal.
- **−** `e2w_core` must be guarded carefully — if it grows model-specific code,
  the benchmark stops being lightweight. Reviews are mandatory on it.
- The benchmark *can* still be extracted into its own repo later with a git
  subtree split, because its dependency surface is bounded by design.
