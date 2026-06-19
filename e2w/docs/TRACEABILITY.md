# TRACEABILITY — proposal claim → module → test → status

The drift radar (Repo-Design §4②). Read it two ways:

- **novelty with no module/test = a gap** (something promised but unbuilt);
- **module with no proposal anchor = scope creep** (something built but unpromised).

Every PR updates this table. Status legend: ✅ done · ◑ contract/partial ·
⬜ not started · ⏸ deferred (see [SCOPE.md](SCOPE.md)).

The ⬜/◑ model rows are turned green by the build spec — each row maps to an
acceptance gate in [docs/build/05-acceptance-and-tasks.md](build/05-acceptance-and-tasks.md).

## The three true novelties (architecture §A.7 — the reason this beats Bernini)

| claim | module | guarding test | status |
|---|---|---|---|
| ① abduction = source inversion to latent as invariant prior (the U) | `generation/e2w_generation/latent` + `e2w_core.latent` | inversion round-trip error + UNCHANGED-region latent match | ⬜ |
| ② the **indirect / multi-hop** layer of the three-layer mask (the 命门) | `localization` (change A) + `data_engine` (dependency-graph labels) | predicted indirect mask aligns to sim dependency graph | ⬜ |
| ③ abduction-bound invariant-preservation loss | `generation/e2w_generation/losses` | `不改变` region V̂ latent must equal source latent | ⬜ |

## Shared contracts (e2w_core — the seam)

| claim | module | guarding test | status |
|---|---|---|---|
| three-layer mask enum = {direct, indirect, unchanged}; priority 直接>间接>不改变 | `e2w_core.masks` | `resolve_pixel` + enum-membership test | ◑ contract defined, test TODO |
| planner emits vectors (query + edit tokens), not masks; `Operation` matches benchmark | `e2w_core.plan` | Operation↔manifest enum parity test | ◑ |
| model IO `predictions/` shape is one source of truth | `e2w_core.io_contract` | parity test vs `cf_vedit_bench` constants | ◑ |

## Benchmark (P0 — the ruler; lives at `physics_iq_for_simple_eval/` for now)

| claim | module | guarding test | status |
|---|---|---|---|
| two-axis metrics never collapse to one score (保不变量 × 命中后果) | `cf_vedit_bench` scoring | existing spec-test (`保不变量`/`命中后果` keys required) | ✅ |
| `copy_source`/`free_regen` anchors land at opposite corners | `cf_vedit_bench` baselines | existing anchor test | ✅ |
| truth never comes from a generative model | `cf_vedit_bench` data / `provenance.jsonl` | provenance source enum forbids t2v-as-GT | ◑ data is real Physics-IQ; explicit test TODO |
| `target_success` precondition gates consequence/physical | `cf_vedit_bench` scoring | gating tests (added this cycle) | ✅ |

## Boundaries (CI guards)

| claim | module | guarding test | status |
|---|---|---|---|
| B1 benchmark imports 0 model packages | — | import-linter contract (`pyproject.toml`) | ◑ contract written, CI wiring TODO |
| B3 localization ⟂ generation | — | import-linter independence contract | ◑ |
| B4 vendored upstream has no in-place diff | `third_party/` | CI third_party-clean check | ⬜ (no vendor yet) |
| B5 train/eval sources disjoint | `provenance.jsonl` | leakage check | ◑ provenance carries evidence; cross-source check TODO |

## Pipeline & training

| claim | module | guarding test | status |
|---|---|---|---|
| render seam solved in one pass (feather + joint denoise, no 2nd-pass) | `generation/e2w_generation/renderer` | seam-artifact eval | ⬜ |
| three-stage training (align → end-to-end → optional RL) | `integration/pipelines` + `configs/` | training smoke | ⬜ |
| Rung-3 gate: same instruction, two sources → different GT | `cf_vedit_bench` (`pair_id`) | reserved | ⏸ see SCOPE |
