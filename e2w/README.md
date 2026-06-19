# e2w — CF-VEdit / E2W monorepo

Counterfactual video editing: given a **source video + edit instruction**,
produce the target video where the directly-touched region is re-rendered, the
*causal consequences* follow, and **everything else is pinned frame-by-frame**.
The thesis (vs. Bernini): treat editing as `abduction + do() + causal closure +
invariant constraint`, not as "write the prompt in more detail".

> **New here? Read [`docs/proposal/`](docs/proposal/README.md) first** — the four
> design notes are the canonical source of truth. Then read [`AGENTS.md`](AGENTS.md)
> (the repo constitution: five boundaries + change discipline).
>
> **Building the model?** The implementation-ready plan — edit Sa2VA, adapt
> VACE/Wan, sim data engine, three-stage training, acceptance gates — is in
> [`docs/build/`](docs/build/README.md).

## Status: skeleton

This is the scaffold the design prescribes — **ruler before machine, contracts
before halves** (Repo-Design §5). Only `e2w_core` (the seam) carries real types
today; the two model halves are stubs with READMEs that back-link to the spec.
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
│   ├── generation/      生成半: Abduction inversion + gated Renderer (from VACE/Wan)
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
