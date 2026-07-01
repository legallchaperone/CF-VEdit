# ADR-0003 — Enable the full (untrained) A.1 path alongside the vanilla floor

- **Status:** Accepted
- **Date:** 2026-06-20
- **Anchors:** Architecture §A.1; build/00-vanilla-eval.md; build/04; Repo-Design §2 (B4)

## Context

V0 shipped the *vanilla floor* (`00-vanilla-eval.md`): the three counterfactual
mechanisms — `[SEG_DIR]/[SEG_IND]` (three-layer mask) and `[EDIT]` (edit tokens) —
were deliberately bypassed (`region_query=None`, `edit_tokens=None`,
`indirect=zeros`), so the runtime collapsed to "stock `[SEG]` → VACE inpaint".

The request is to stand up the **entire A.1 architecture, wired end-to-end, but
untrained** — every block real, new weights random/identity. `00-vanilla-eval.md`
explicitly warns that running the new heads untrained yields near-random output;
so this is a deliberate deviation from the eval-first "measure the floor first"
sequencing, taken to make the *structure* exist and run before training.

## Decision

1. Add a `--full` mode to `e2w_adapter` and a `vanilla=False` branch to
   `CausalPlanner.plan` that runs the complete A.1 data path with untrained heads:
   teacher-forced three-layer mask + `region_query` + `edit_tokens` → renderer.
2. **Vanilla stays the default floor and byte-identical** (its branch is untouched;
   the `[SEG]` `generate()` path is not perturbed — see ADR-0004).
3. **Quality is explicitly out of scope.** Only structural completeness + clean
   end-to-end execution is claimed. The untrained `edit_tokens`/`indirect` are
   shape-correct garbage and the full path is expected to score *worse* than
   vanilla on preservation; runs are reported per-operation, never as an
   improvement over the floor.

## Consequences

- **+** The full architecture is real and runnable; `indirect` and `edit_tokens`
  flow through the seam. A **single-sample** GPU smoke (`e2w_full_smoke`, 1 sample,
  4 steps) ran end-to-end: npz carries `direct`/real `indirect`/`edit_tokens` and a
  video was written. This is **not** a benchmark-valid run (the validator requires
  all 12 samples); a full `--full` run over the manifest is the next step.
- **+** Eval-first integrity preserved: vanilla floor and `eval₁` after training
  use the identical harness; this path is a separate, labelled run
  (`model_version=v0-full-untrained-eval`).
- **−** A path that scores below the floor now exists; it must always be presented
  with the "untrained → quality regresses until training" caveat.
- **Follow-up:** training (stage ②) replaces random heads; the masks/edit_tokens
  become meaningful and `eval₁` measures the lift.
