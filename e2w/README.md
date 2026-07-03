# e2w — CF-VEdit / E2W monorepo

Counterfactual video editing: given a **source video + edit instruction**,
produce the target video where the directly-touched region is re-rendered, the
*causal consequences* follow, and **everything else is pinned frame-by-frame**.
The thesis (vs. Bernini): treat editing as `abduction + do() + causal closure +
invariant constraint`, not as "write the prompt in more detail".

> **New here? Read [`../E2W-v0-Remove-Only-Spec.md`](../E2W-v0-Remove-Only-Spec.md)
> first** — the current authoritative build spec (remove-only, frozen
> CogVideoX-Fun/VOID renderer). See
> [ADR-0007](docs/adr/0007-e2w-v0-remove-only-void-renderer.md) for why. The
> four notes in [`docs/proposal/`](docs/proposal/README.md) are the long-run
> open-domain thesis — still canonical for research direction, superseded for
> v0's concrete architecture. Then read [`AGENTS.md`](AGENTS.md) (the repo
> constitution: five boundaries + change discipline).
>
> **Building the model?** [`docs/build/`](docs/build/README.md) has the
> implementation-ready plan, but its renderer target (VACE/Wan) and abduction
> step are pre-pivot — see its own banner. The v0 renderer path has since been
> ported to CogVideoX-Fun/VOID; the build docs remain historical until refreshed.

## Status: v0 scaffold plus remove-only renderer path

This keeps the scaffold the design prescribes — **ruler before machine,
contracts before halves** (Repo-Design §5). `e2w_core` carries the seam types;
`localization` and `generation` now include the v0 remove-only adapter path.
The working P0 benchmark still lives at
[`../physics_iq_for_simple_eval/`](../physics_iq_for_simple_eval/) and is *not*
migrated yet (see [ADR-0002](docs/adr/0002-scaffold-alongside-without-migration.md)).

## Layout

```
e2w/
├── docs/
│   ├── proposal/      canonical spec (proposal-as-truth; references the root notes)
│   ├── TRACEABILITY.md  proposal claim → module → test → status (drift radar)
│   ├── SCOPE.md         deferred / reserved scope (placeholders, not half-builds)
│   └── adr/             Architecture Decision Records (every deviation gets one)
├── packages/
│   ├── e2w_core/        ★ shared contracts (masks, plan, latent, io) — the seam
│   ├── cf_vedit_bench/  the benchmark (currently a pointer to the live P0)
│   ├── localization/    定位半: Causal Planner + Mask decoder (from Sa2VA)
│   ├── generation/      生成半: CogVideoX-Fun/VOID source payload + renderer
│   └── data_engine/     Kubric-style sim pairs + dependency-graph labels
├── integration/         端到端 pipelines + adapters → write predictions/
└── configs/  scripts/  tools/   guards (import-linter, schema lint, proposal-link)
```

CI lives at the **repo-root** `.github/workflows/ci.yml` (GitHub only runs
workflows at the repository root) during the transition; it moves into `e2w/` on
split-out — see [ADR-0002](docs/adr/0002-scaffold-alongside-without-migration.md).

## Dependency direction (the constitution, CI-enforced)

```
                         e2w_core  (contracts; no internal deps)
                        ▲   ▲   ▲   ▲
       ┌────────────────┘   │   │   └────────────────┐
 cf_vedit_bench       localization  generation     data_engine
 (consumes predictions/)   └────── integration ──────┘ ──writes──► predictions/
        ▲                                                              │
        └──────────────── benchmark only CONSUMES the directory ◄──────┘
```

See [ADR-0001](docs/adr/0001-monorepo-splittable-benchmark.md) for why the
benchmark is a splittable subpackage, and `pyproject.toml` (`[tool.importlinter]`)
for the machine-checked form of this graph.
