# 04 — The bridge, inference pipeline, training, benchmark adapter

> Wires the two halves through `e2w_core`, defines the single-pass inference flow,
> the three-stage training recipe, and the adapter that scores E2W on the
> benchmark. Implements architecture §A.1 and proposal §2.7.

## The bridge (boundary B3)

The two halves never import each other. They meet only at `e2w_core`:

```
[01] localization ──emits──►  ThreeLayerMask + EditPlan.edit_tokens
[02] generation   ──emits──►  SourceLatent
                       │
            integration assembles all three and runs the renderer
```

All assembly lives in `integration/` (the only place that imports both halves).

## Inference pipeline (architecture §A.1 — single pass, no 2nd-pass)

```python
# integration/pipelines/e2w_pipeline.py  (to write)
from e2w_localization.planner import CausalPlanner
from e2w_generation.abduction import WanAbductor
from e2w_generation.renderer import GatedRenderer

def edit(video, instruction: str, *, vanilla: bool = False):
    source = WanAbductor().invert(video)                 # 【1】 abduction → U (pretrained VAE, zero-shot)
    mask, plan = CausalPlanner().plan(video, instruction, vanilla=vanilla)  # 【2】+【3】
    cond = instruction if vanilla else plan.edit_tokens  # vanilla: untrained [EDIT] head bypassed
    return GatedRenderer().render(source, cond, mask)    # 【4】 gated render → V̂
```

## Vanilla mode — run the structure before any training ([00])

The new heads start randomly-initialized with no gradient, so the assembled
structure must be runnable through the **pretrained** pathways only, or the vanilla
eval is meaningless. `vanilla=True` routes around the untrained parts:

| component | normal | vanilla (untrained) |
|---|---|---|
| abduction inversion | Wan VAE | same (pretrained, zero-shot) |
| direct mask | `[SEG_DIR]` (trained) | stock Sa2VA `[SEG]` on `target_ref` (zero-shot) |
| indirect mask | `[SEG_IND]` (trained) | **empty** — no causal closure yet |
| content condition | `EditPlan.edit_tokens` | the **instruction string** → VACE native text cond |
| gated renderer | mask-gated inpaint (pretrained) + invariant loss (trained) | mask-gated inpaint only (pretrained, zero-shot) |

So `CausalPlanner.plan(..., vanilla=True)` returns a direct-only `ThreeLayerMask`
(indirect empty) and lets the caller substitute the instruction for `edit_tokens`.
This is the configuration the **vanilla eval** measures; training then lights up
the bypassed paths, and eval₁ measures the lift over that floor.

## Three-stage training (proposal §2.7 — never all at once; VEGGIE failed that)

Start from pretrained Sa2VA + Wan/VACE; **VAE frozen throughout**.

**Stage ① — Align.** Freeze the MLLM; train the alignment layers
(`text_hidden_fcs`, `edit_hidden_fcs`) + renderer adaptation on simple
image/single-frame edits, so the renderer learns to *obey* the mask and the edit
tokens. Goal: tokens/masks become meaningful conditions.

**Stage ② — End-to-end (the main event).** Train on sim CF videos ([03]); light
MLLM finetune (LoRA or partial unfreeze), unfreeze `sam_mask_decoder`, and
backprop **all three losses jointly**:
1. main flow-matching V̂→V\* ([02]);
2. causal-mask loss: predicted three-layer mask vs sim dependency graph ([01]),
   indirect down-weighted;
3. invariant loss: UNCHANGED-region latent == source latent ([02]).
This is where `edit_tokens` finally get gradient (via the renderer) and where the
**indirect layer** is actually learned. Add temporal rollout for `force_event` in
P2.

**Stage ③ — Alignment (optional, P3).** Human-preference / differentiable reward,
in an **isolated** package; core training must not depend on it.

Configs live in `e2w/configs/`; pin weights + hyperparameters there. Launch via
`e2w/scripts/` over the Sa2VA `dist.sh`-style entry (≥8×A100).

## Benchmark adapter (boundary B1 — how E2W gets scored)

E2W is scored exactly like any other model: it writes a `predictions/<run>/`
directory and the benchmark consumes it. **Never** make the benchmark import E2W.

- Write `integration/adapters/e2w_adapter.py`: read the benchmark `manifest.jsonl`,
  run `edit(...)` per sample, write each clip to `videos/<sample_id>.mp4`, and
  emit `predictions.jsonl` + `run_meta.json` using
  [`e2w_core.io_contract`](../../packages/e2w_core/e2w_core/io_contract.py)
  (`RUN_META_REQUIRED`, `PredictionRow`).
- The boilerplate (manifest iteration, failure rows, run_meta hash/version) is
  already solved by
  [`physics_iq_for_simple_eval/tools/make_prediction_run.py`](../../../physics_iq_for_simple_eval/tools/make_prediction_run.py)
  — reuse its pattern (swap the `cp` command for the `edit(...)` call).
- Then score/report with the benchmark CLI: `bench validate <run>` →
  `score --judge vlm|human` → `report`. Two-axis metrics; compare against the
  `copy_source`/`free_regen` anchors and (later) Bernini.

## What stays frozen vs trains (quick reference)

| stage | MLLM | text/edit fcs | sam_mask_decoder | renderer DiT | VAE |
|---|---|---|---|---|---|
| ① align | ❄️ frozen | 🔥 train | ❄️ | 🔥 adapt | ❄️ |
| ② e2e | 🔥 light (LoRA) | 🔥 | 🔥 | 🔥 | ❄️ |
| ③ align (opt) | 🔥 light | 🔥 | — | 🔥 | ❄️ |
