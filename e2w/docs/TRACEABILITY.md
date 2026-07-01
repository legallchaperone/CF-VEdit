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
| ① abduction = source inversion to latent as invariant prior (the U) | `generation/e2w_generation/abduction.py` (+ `encode_only` G1 hook) + `generation/e2w_generation/renderer.py` (`encode_source_to_latent`, noised+feathered paste-back) + `e2w_core.latent` | V0 smoke: Wan source latent materialized; UNCHANGED latent paste-back (noised-to-timestep + feather) | ◑ encode in V0; flow-inversion deferred (ADR-0005) |
| ② the **indirect / multi-hop** layer of the three-layer mask (the 命门) | `localization/e2w_localization/{overlay,teacher_forced,planner}.py` (full: teacher-forced `[SEG_DIR]/[SEG_IND]` → real direct+indirect; vanilla: stock `[SEG]`→direct, indirect empty) + `data_engine` TODO | predicted indirect mask aligns to sim dependency graph | ◑ full path builds a **real (untrained) indirect** mask end-to-end (ADR-0004); semantics need training |
| ③ abduction-bound invariant-preservation loss | `generation/e2w_generation/losses` | `不改变` region V̂ latent must equal source latent | ⏸ no training in V0 |

## Shared contracts (e2w_core — the seam)

| claim | module | guarding test | status |
|---|---|---|---|
| three-layer mask enum = {direct, indirect, unchanged}; priority 直接>间接>不改变 | `e2w_core.masks` | `test_contracts.MaskContractTest` | ✅ |
| planner emits vectors (query + edit tokens), not masks; `Operation` matches benchmark | `e2w_core.plan` | `test_contracts.OperationParityTest` | ✅ |
| model IO `predictions/` shape is one source of truth | `e2w_core.io_contract` | `test_contracts.IoContractParityTest` | ✅ |

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
| B4 vendored upstream has no in-place diff | `third_party/` (`sa2va`, `vace`, `wan2_2`, `videox_fun`); deltas in `e2w_localization/overlay.py` (runtime composition) + `localization/patches/` (unapplied) | CI third_party-clean check | ◑ Sa2VA heads added by runtime composition onto the loaded snapshot; `third_party/` byte-clean (ADR-0004) |
| B5 train/eval sources disjoint | `provenance.jsonl` | leakage check | ◑ provenance carries evidence; cross-source check TODO |

## Pipeline & training

| claim | module | guarding test | status |
|---|---|---|---|
| render seam solved in one pass (feather + joint denoise, no 2nd-pass) | `generation/e2w_generation/renderer.py` (noised-to-timestep paste `paste_noise_to_timestep`; soft feather `mask_feather_latent`) | V0 prediction run validates; A/B toggles vs 0.08 floor pending (ADR-0005) | ◑ seam fixes landed + run end-to-end; preservation A/B + human eval pending |
| full (untrained) A.1: teacher-forced 3-layer mask + edit_tokens → renderer | `localization/e2w_localization/{overlay,teacher_forced}.py` + `generation/.../renderer.py` (edit_tokens via `_get_t5` override) + `integration/{pipelines,adapters}` (`--full`) | **single-sample** `e2w_full_smoke` on GPU: npz carries direct/real-indirect/edit_tokens; video written (ADR-0003). Benchmark-valid 12-sample run pending | ◑ runs end-to-end untrained; quality out of scope until training |
| `edit_tokens` = continuous content condition to the renderer | `localization` `edit_hidden_fcs` (overlay) → `generation` `prompt_embeds` seam | full run feeds `edit_tokens (Nt,4096)` as positive cross-attn | ◑ wired + runs; no gradient until joint stage-② training |
| three-stage training (align → end-to-end → optional RL) | `integration/pipelines` + `configs/` | training smoke | ⏸ no training in V0 |
| V0 vanilla eval path writes benchmark `predictions/<run>/` without benchmark imports | `integration/adapters/e2w_adapter.py` + `integration/pipelines/e2w_pipeline.py` + `configs/vanilla.v0.json` | `bench.py validate e2w_vanilla_v0` | ✅ run produced 12/12 valid videos; human judge launched |
| Rung-3 gate: same instruction, two sources → different GT | `cf_vedit_bench` (`pair_id`) | reserved | ⏸ see SCOPE |
