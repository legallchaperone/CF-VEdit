# 00 — Vanilla eval: the assembled structure, untrained

> **The first eval in the sequence: build structure → vanilla eval → collect data
> → train → eval₁.** "Vanilla" means **our** E2W, assembled from pretrained weights
> with **no training of the new parts** — run on the benchmark as-is. It is the
> **floor**: it isolates what the architecture gives for free, so the post-training
> eval₁ measures the lift attributable to training (not to the backbone models).
>
> This is *not* about running competitors (Bernini/VEGGIE) — those are a separate,
> optional external comparison (bottom of this doc). The vanilla eval is our own
> untrained structure.

## The catch: untrained heads must be routed around ("vanilla mode")

The new components have **randomly-initialized** weights and no gradient yet, so
running them as-is yields garbage. Vanilla mode (a flag in the pipeline, [04])
uses only the **pretrained** pathways and bypasses the untrained ones:

| block | vanilla behavior | trained later? |
|---|---|---|
| Abduction inversion [02] | ✅ pretrained Wan VAE encode/invert — zero-shot | refined in train |
| Mask — **direct** | ✅ use Sa2VA's **existing** pretrained `[SEG]` on `target_ref` → a zero-shot direct mask | replaced by `[SEG_DIR]` |
| Mask — **indirect** | ⛔ `[SEG_IND]` untrained → **empty** in vanilla; no causal closure | the whole point of training (the 命门) |
| Content condition | ⛔ `[EDIT]` head untrained → **bypass**: feed the instruction to VACE's native text conditioning | replaced by edit tokens |
| Gated renderer [04] | ✅ pretrained VACE masked inpainting: inpaint the direct region per the instruction, paste the source latent back everywhere else | refined in train |

So **vanilla E2W = "segment the target with stock Sa2VA, inpaint that region with
stock VACE under the instruction, gate everything else back from the source
latent."** A zero-shot mask-gated editor — real, runnable, untrained.

## What vanilla eval should show (and why the floor is informative)

Read the two axes separately, never one score:

- **Preservation (保不变量): expected DECENT.** The paste-back gating is
  *architectural* — it works without training. This is the architecture's free
  gift, and the headline finding of the vanilla eval: even untrained, E2W should
  not over-edit the way prompt-led models do.
- **Consequence (命中后果): expected WEAK.** No indirect layer (untrained) → causal
  connExtensions (shadows, secondary motion, knock-on effects) are missed. Direct
  edits may land if VACE follows the instruction zero-shot.
- **Edit success:** depends on VACE's zero-shot instruction-following on the masked
  region.

That gap — good preservation, weak consequence — is **exactly what training must
close**: learn the indirect mask + edit tokens + invariant loss so consequence and
physical-plausibility rise *without* preservation collapsing. eval₁ vs this floor
is the experiment.

## How to run it

Same harness as any model — the benchmark never imports E2W ([04] adapter, B1).
Use the dedicated `e2w_adapter` (built per [04]); it calls the pipeline's
`edit(..., vanilla=True)` per sample and writes `predictions/<run>/` via
`e2w_core.io_contract`. The commands below are **illustrative** — the adapter
module does not exist until you build it ([04]):

```bash
# 1. run E2W in vanilla mode -> writes predictions/e2w_vanilla/
python3 -m integration.adapters.e2w_adapter --vanilla --run-name e2w_vanilla

# 2. score + report with the benchmark CLI (it only consumes the directory)
cd physics_iq_for_simple_eval
python3 bench.py validate e2w_vanilla
python3 bench.py score  e2w_vanilla --judge vlm     # OPENROUTER_API_KEY + ffmpeg (or --judge human)
python3 bench.py report e2w_vanilla
```

> For *external* models (the optional baselines below) there is no in-process
> adapter — package their outputs with the generic
> [`tools/make_prediction_run.py`](../../../physics_iq_for_simple_eval/tools/make_prediction_run.py),
> giving `--cmd` a **real** per-sample CLI (its `--cmd` is executed literally per
> sample, so a placeholder string would just record failed predictions).

Needs a GPU (to run E2W) + a judge backend. Record this as the **floor** run;
eval₁ after training uses the **identical** harness (same manifest hash, judge,
prompts) so the delta is attributable to training alone.

## Optional: external baselines (separate from the vanilla eval)

For context you may also run the proposal's §4.3 baselines through the same harness
— the `copy_source`/`free_regen` anchors (ruler sanity: they must bracket the
field) and competitors (Bernini, VEGGIE, VOID-remove, …). Useful, but secondary to
the vanilla→eval₁ comparison, and gated by model access.

## Honest constraints

- **Scale:** 12-clip smoke set today → sanity only; a real vanilla eval needs the
  benchmark grown toward ~300–500 annotated real clips (the *easy* data half —
  annotation on real video, no GT render; see [03]).
- **Judge:** report `agree` (human↔VLM κ) so the judge's reliability is visible.
- **⚠️ vanilla-mode is a deliberate routing choice.** The alternative — running the
  full new heads while untrained — is uninformative (near-random). Confirm the
  routing above matches how you assemble [01]/[04].
