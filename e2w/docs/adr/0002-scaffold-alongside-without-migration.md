# ADR-0002 — Scaffold e2w/ alongside the live benchmark; reference the proposal

- **Status:** Accepted
- **Date:** 2026-06-19
- **Anchors:** ADR-0001; Repo-Design §1, §4①

## Context

We are standing up the `e2w/` skeleton **while** a working P0 benchmark already
lives at `physics_iq_for_simple_eval/`, is committed, and is pushed to a public
remote. The four proposal notes are already canonical at the repository root.

Two risks if we follow Repo-Design §1 literally right now:

1. **Migrating the benchmark** into `e2w/packages/cf_vedit_bench/` immediately
   would move a self-contained, path-sensitive, already-published package — a
   large rename that breaks documented commands, CLAUDE.md/AGENTS.md paths, and
   the remote layout, for no functional gain yet.
2. **Duplicating the four notes** into `e2w/docs/proposal/` would put two copies
   of the truth source in one repo — exactly the drift "proposal-as-truth" exists
   to prevent.

## Decision

1. **Scaffold without migrating.** Create the `e2w/` structure and the real
   `e2w_core` contracts, but leave the benchmark where it is.
   `packages/cf_vedit_bench/` is a **pointer** (README) to the live benchmark
   until a dedicated migration step.
2. **Reference, don't duplicate, the proposal.** `e2w/docs/proposal/` links to
   the root canonical notes. On split-out into a standalone repo, they get
   materialized as copies (the §4① intent), but not while they'd be a second copy
   in the same tree.

## Consequences

- **+** Zero risk to the working, published benchmark; the skeleton lands purely
  additively.
- **+** No duplicate proposal copies to drift; the root notes stay the single
  truth source.
- **−** Transitional asymmetry: the benchmark sits outside `e2w/packages/` for
  now, so the B1 import-linter contract for `cf_vedit_bench` is aspirational
  until migration.
- **Follow-up (tracked):** a later ADR + PR migrates `physics_iq_for_simple_eval/`
  → `e2w/packages/cf_vedit_bench/` (git move preserving history) and materializes
  `docs/proposal/` if/when the repo is split.
