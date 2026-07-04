# TRACEABILITY — proposal claim → module → test → status

The drift radar (Repo-Design §4②). Read it two ways:

- **novelty with no module/test = a gap** (something promised but unbuilt);
- **module with no proposal anchor = scope creep** (something built but unpromised).

Every PR updates this table. Status legend: ✅ done · ◑ contract/partial ·
⬜ not started · ⏸ deferred (see [SCOPE.md](SCOPE.md)) · 🔁 superseded (see
[ADR-0007](adr/0007-e2w-v0-remove-only-void-renderer.md), row kept for history.

The ⬜/◑ model rows are turned green by the build spec — each row maps to an
acceptance gate in [docs/build/05-acceptance-and-tasks.md](build/05-acceptance-and-tasks.md).
**As of ADR-0007 (2026-07-02) the build spec's target architecture (VACE/Wan
renderer, MLLM abduction) is itself superseded for v0** — see
[`E2W-v0-Remove-Only-Spec.md`](../../E2W-v0-Remove-Only-Spec.md) at repo root
for the current build target; `docs/build/01`/`02` rows below have not yet
been reworked to match it (docs landed first, per ADR-0007's own decision).

## The three true novelties — v0 framing (ADR-0007, current)

Replaces the abduction/Pearl-framed novelties below. See spec §0.

| claim | module | guarding test | status |
|---|---|---|---|
| (a) physics-consequence-aware removal task (vs. photometric-only concurrent work) | benchmark `operation=remove` subset + contract `affected_regions`/`counterfactual_state` | CF-VEdit remove subset IP×CR vs VOID (`results/void/`) | ⬜ not started |
| (b) controlled comparison: renderer + mask mechanism identical to VOID, only conditioning source varies | `generation/e2w_generation/void_renderer.py` (frozen CogVideoX-Fun + `void_pass1.safetensors`, thin VOID-pipeline wrapper) + `integration/{pipelines,adapters}` | VOID-oracle-quadmask+edit-token ablation cell (spec §3) isolates the one variable | ◑ renderer ported (M0–M4): vendored VOID fork + `void_renderer.py`; M2 fidelity pixel-MAE ~1.5 vs VOID pass1; M4 e2e pass (Sa2VA→mask→render, 832×480/21f). A/B eval pending training |
| (c) seg/edit dual-branch, asymmetric differentiability made precise (edit branch end-to-end differentiable through frozen renderer; seg branch is not — quadmask thresholding blocks gradient) | `localization/e2w_localization/{overlay,query_tokens}.py` (6 fixed-position query tokens: `seg_dir,seg_ind,edit_0..3`; custom 4D attention mask; tied RoPE) | Stage 0: held-out mask IoU/Dice. Stage 2: edit-token cosine-collapse check (spec §2 Stage 1 note) | ◑ query-token mechanism exists (ADR-0004/0006); grad-through-frozen-renderer verified (edit_embeds.grad nonzero via `prompt_embeds`, bf16); full edit_tokens→`prompt_embeds` render path runs e2e (M2b, valid 832×480); **localization mechanism GPU-gated** (`gate_query_tokens_gpu.py`: 4D-mask+tied-RoPE forward runs on real Sa2VA, edit→seg attention blocked==0, M-RoPE non-fallback); Stage 0–2 training not started |

## Superseded novelties (pre-ADR-0007, kept for history)

| claim | module | guarding test | status |
|---|---|---|---|
| ① abduction = source inversion to latent as invariant prior (the U) | `generation/e2w_generation/abduction.py` (+ `encode_only` G1 hook) + `generation/e2w_generation/renderer.py` (`encode_source_to_latent`, noised+feathered paste-back) + `e2w_core.latent` | V0 smoke: Wan source latent materialized; UNCHANGED latent paste-back (noised-to-timestep + feather) | 🔁 dropped for v0, not deferred — no MLLM-inversion step in the remove-only design; unchanged-region conditioning now comes from VOID's mask+masked-latent channel-concat instead |
| ② the **indirect / multi-hop** layer of the three-layer mask (the 命门) | `localization/e2w_localization/{overlay,query_tokens,planner}.py` (full: query-token `[SEG_DIR]`/`[SEG_IND]` → real direct+indirect; vanilla: stock `[SEG]`→direct, indirect empty) + `data_engine/e2w_data_engine/davis2017_remove.py` for v0 Stage 0/1 pseudo-labels | `data_engine/tests/test_davis2017_remove.py` guards DAVIS palette-index labels, instruction-mask alignment, integral-pair merging, grey-mask quarantine, and quadmask consistency | 🔁 original sim-dependency-graph target is superseded for v0, but the Stage 0/1 DAVIS+VOID label builder now exists (ADR-0008); Stage 2 paired sim data still not started |
| ③ abduction-bound invariant-preservation loss | `generation/e2w_generation/losses` | `不改变` region V̂ latent must equal source latent | 🔁 dropped for v0 — renderer is frozen throughout v0 training (Stages 0–2), no loss to bind; unchanged-region fidelity is architectural (VOID gating) not learned |

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
| B4 vendored upstream has no in-place diff | `third_party/` (`sa2va`, `void_videox_fun`; the Wan/VACE `vace`/`wan2_2`/`videox_fun` submodules removed with the VACE/Wan renderer, ADR-0007); deltas in `e2w_localization/overlay.py` (runtime composition) + `localization/patches/` (unapplied) | CI third_party-clean check | ◑ Sa2VA heads added by runtime composition onto the loaded snapshot; `third_party/` byte-clean (ADR-0004) |
| B5 train/eval sources disjoint | `provenance.jsonl` | leakage check | ◑ provenance carries evidence; cross-source check TODO |

## Pipeline & training

> **Renderer reworked to frozen CogVideoX-Fun/`void_pass1` (M0–M4, ADR-0007).**
> The v0 renderer is now `e2w_generation/void_renderer.py` — a thin wrapper over
> the vendored VOID fork (`third_party/void_videox_fun/`): edit_tokens →
> `prompt_embeds` hard-replace, mask → VOID channel-concat, paste-back/abduction
> dropped. Verified: M2 pixel-MAE ~1.5 vs VOID pass1, M4 end-to-end pass,
> gradient flows through the frozen renderer to edit_embeds. Deps: Sa2VA remote
> code needs `qwen_vl_utils`. Known gap: `E2WPipeline.edit()` keeps both models
> resident (no `planner.unload()`); fine on 97GB, adapter's two-phase path
> handles small GPUs. The rows below describe the **superseded** VACE/Wan
> implementation (`e2w_generation/renderer.py`/`abduction.py`), **now removed from
> the tree** along with the `vace`/`wan2_2`/`videox_fun` submodules (🔁 ADR-0007);
> rows kept for history, file pointers are historical.
> v0's Stage 0/1/2 training has no code rows yet; Stage -1 pre-experiment done
> (spec §2 result).

| claim | module | guarding test | status |
|---|---|---|---|
| render seam solved in one pass (feather + joint denoise, no 2nd-pass) | `generation/e2w_generation/renderer.py` (noised-to-timestep paste `paste_noise_to_timestep`; soft feather `mask_feather_latent`) | V0 prediction run validates; A/B toggles vs 0.08 floor pending (ADR-0005) | ◑ seam fixes landed + run end-to-end; preservation A/B + human eval pending — **on the superseded VACE/Wan target (🔁 ADR-0007)** |
| full (untrained) A.1: query-token 3-layer mask + edit_tokens → renderer | `localization/e2w_localization/{overlay,query_tokens}.py` + `generation/.../renderer.py` (edit_tokens via `_get_t5` override) + `integration/{pipelines,adapters}` (`--full`) | **single-sample** `e2w_full_smoke` on GPU: npz carries direct/real-indirect/edit_tokens; video written (ADR-0003). Benchmark-valid 12-sample run pending | ◑ runs end-to-end untrained; quality out of scope until training |
| `edit_tokens` = continuous content condition to the renderer | `localization` `edit_hidden_fcs` (overlay) → `generation` `prompt_embeds` seam | full run feeds `edit_tokens (Nt,4096)` as positive cross-attn | ◑ wired + runs; no gradient until joint stage-② training |
| query-token attention: `[EDIT]` slots bidirectional + tied position id, isolated from `[SEG_DIR]`/`[SEG_IND]` | `localization/e2w_localization/query_tokens.py` (`_build_query_attention_mask`, `_continuation_offsets`) | unit test on mask connectivity + offset symmetry (no GPU needed) | ◑ mask + position tying fixed pre-training (ADR-0006); semantic effect only observable once training starts |
| three-stage training (align → end-to-end → optional RL) | `integration/pipelines` + `configs/` | training smoke | ⏸ no training in V0 |
| V0 vanilla eval path writes benchmark `predictions/<run>/` without benchmark imports | `integration/adapters/e2w_adapter.py` + `integration/pipelines/e2w_pipeline.py` + `configs/vanilla.v0.json` | `bench.py validate e2w_vanilla_v0` | ✅ run produced 12/12 valid videos; human judge launched |
| Rung-3 gate: same instruction, two sources → different GT | `cf_vedit_bench` (`pair_id`) | reserved | ⏸ see SCOPE |
