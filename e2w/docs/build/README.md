# BUILD SPEC — how to build E2W from existing models

> **⚠️ Target superseded for v0 (ADR-0007, 2026-07-02).** This build spec
> (and 01/02 below) still describe the VACE/Wan2.2 renderer + MLLM-abduction
> architecture. The current build target is
> [`E2W-v0-Remove-Only-Spec.md`](../../../E2W-v0-Remove-Only-Spec.md) at repo
> root: remove-only, frozen CogVideoX-Fun-V1.5-5b-InP renderer initialized from
> VOID's `void_pass1.safetensors`, no MLLM source-inversion step. `00` and `03`–`05`
> are largely orthogonal to the renderer choice and still apply in outline; `01`'s
> query-token design (changes A/B) carries over but is respecified precisely in
> the v0 spec §1.2–1.4; `02` (VACE/Wan renderer) does not apply to v0 at all.
> Per ADR-0007, docs land first — these files have not yet been rewritten for
> v0; treat the VACE/Wan-specific parts below as historical/pre-pivot, not as
> current instructions to a coding agent.

> **Audience:** a coding agent on a machine with disk + GPUs (≥8×A100 80GB for
> training). **Goal:** turn the `e2w/` skeleton into a working counterfactual
> video editor by *editing Sa2VA* (planner + mask decoder) and *adapting
> VACE/Wan* (abduction + gated renderer), wired through the frozen `e2w_core`
> contracts, then training in three stages.
>
> This is implementation-ready intent, not training code. Where an exact upstream
> API is needed, it is marked **⚠️ verify** — read the pinned upstream source and
> adapt; do not assume signatures from this doc.

## How to use these docs

Read in order; each is one slice of the pipeline:

| doc | slice | upstream reused |
|---|---|---|
| this README | environment, weights, vendoring, the seam rule, milestone map | — |
| [00-vanilla-eval](00-vanilla-eval.md) | **vanilla eval**: run *our* assembled-but-untrained structure on the benchmark (the floor; sequence step "vanilla eval") | benchmark |
| [01-localization-sa2va](01-localization-sa2va.md) | 定位半: Causal Planner + Mask decoder (changes A–D) | **Sa2VA** |
| [02-generation-vace-wan](02-generation-vace-wan.md) | 生成半: Abduction inversion + gated Renderer | **VACE / Wan2.2** |
| [03-data-engine](03-data-engine.md) | sim pairs + dependency-graph labels (supervises the hard layer) | **Kubric** |
| [04-training-and-integration](04-training-and-integration.md) | the bridge, inference pipeline, 3-stage training, benchmark adapter | — |
| [05-acceptance-and-tasks](05-acceptance-and-tasks.md) | per-component acceptance tests + ordered task list + risks | — |

The canonical *why* lives in [docs/proposal](../proposal/README.md). This build
spec implements it; it does not restate the research. If the two ever disagree,
the proposal wins and you change it there first (AGENTS.md).

## The one non-negotiable rule: conform to the seam

All new code must produce/consume the frozen `e2w_core` types — that is what lets
the two halves be built independently and still fit together (boundary B3), and
what lets the benchmark score the result without importing model code (B1):

- localization **emits** [`e2w_core.masks.ThreeLayerMask`](../../packages/e2w_core/e2w_core/masks.py) + [`e2w_core.plan.EditPlan`](../../packages/e2w_core/e2w_core/plan.py).
- generation **consumes** those + its own [`e2w_core.latent.SourceLatent`](../../packages/e2w_core/e2w_core/latent.py).
- integration **writes** runs via [`e2w_core.io_contract`](../../packages/e2w_core/e2w_core/io_contract.py) so the benchmark consumes them.

If you need to change a contract, that is an `e2w_core` change → review + an ADR.

## Environment & weights (pin everything)

Download to a local `weights/` dir; pin exact repos/revisions in `e2w/configs/`
(do **not** commit weights). Suggested set:

| component | source | notes |
|---|---|---|
| MLLM + mask plumbing | `ByteDance/Sa2VA-Qwen2_5-VL-7B` (HF) | variant: Qwen2.5-VL-7B; repo `github.com/bytedance/Sa2VA` (Apache-2.0) |
| SAM2 backbone | `facebook/sam2-hiera-large` (`pretrained/sam2_hiera_large.pt`) | grounding encoder inside Sa2VA |
| renderer DiT + VAE | **Wan2.2** (Wan-14B class) + **Wan VAE**; **VACE** (`ali-vilab/VACE`, paper 2503.07598) | VACE = all-in-one editing on Wan; gives masked-V2V conditioning ⚠️ verify exact API/weights |
| sim | **Kubric** (`google-research/kubric`) | data engine only |

Hardware: training ≥8×A100 80GB (Sa2VA-plan §1.D); inference far less. CUDA + a
recent PyTorch matching the Wan/Sa2VA requirements.

## Vendoring discipline (boundary B4)

```
packages/localization/third_party/sa2va/      git submodule, pinned commit, NEVER edited in place
packages/localization/patches/                 our overlay deltas to Sa2VA (git apply at build)
packages/localization/e2w_localization/        new code (prefer subclassing upstream)
packages/generation/third_party/vace_wan/      git submodule, pinned commit
packages/generation/e2w_generation/            new code
```

**Prefer subclassing/overriding over in-place edits** (e.g. `E2WSa2VAModel(Sa2VAModel)`),
so `third_party/` stays byte-identical to upstream. A `third_party`-clean CI guard
(B4) is **planned, not yet wired** — it only becomes meaningful once an upstream is
vendored, so add that job in the PR that introduces the first submodule (tracked
⬜ in [TRACEABILITY](../TRACEABILITY.md); today's CI runs spec-test + e2w_core
import + import-linter only). Until then this is a discipline, not an enforced
check. Use `patches/` where subclassing genuinely can't reach (e.g. tokenizer/
special-token wiring). Pin submodule commits in `.gitmodules` and the weight
revisions in configs.

## Milestone map (aligns to proposal §5 / Repo-Design §5)

The build sequence is a measured loop: **build structure → vanilla eval → collect
data → train → eval₁**, against an unchanging ruler.

- **Build the structure (no training).** Assemble all four blocks from pretrained
  weights and wire them through `e2w_core`; the pipeline runs in **vanilla mode**
  (pretrained pathways, untrained heads bypassed — see [00] and [04]). End-to-end
  runnable, nothing trained yet. → [01], [02], [04].
- **Vanilla eval — the floor.** Run that untrained structure on the benchmark.
  Expect decent preservation (gating is architectural) and weak consequence (no
  trained indirect layer). → [00-vanilla-eval](00-vanilla-eval.md).
- **Data — collect training data (the hard half).** Reuse before you render
  (VOID → Paint-by-Inpaint → t2v aug → own Kubric for the indirect layer); see [03].
- **P1 — train the shared core on shallow DAG.** Learn masks/tokens + invariant
  pinning on *attribute / add / remove* (stages ①②). → [01], [02], [04].
- **P2 — causal closure depth.** Full data engine; train the **indirect / multi-hop**
  mask layer; temporal rollout for physics (`force_event`). → [03] deep, retrain.
- **Eval₁ — re-measure the lift.** Re-run the *identical* vanilla-eval harness
  post-training; report the delta over the floor + the Rung-3 source-sensitivity
  gate. → [00-vanilla-eval](00-vanilla-eval.md).
- **P3 — alignment (optional).** RL / preference, isolated package, core
  independent of it. → [04] last section.

Detailed, checkable tasks: [05-acceptance-and-tasks](05-acceptance-and-tasks.md).
