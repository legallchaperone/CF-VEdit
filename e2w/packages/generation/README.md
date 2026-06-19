# generation — 生成半 (Abduction inversion + gated Renderer)

Actually **renders the result**. Built on **VACE / Wan2.2** (Wan-14B class DiT +
Wan VAE). Implements architecture §A.2 【1】+【4】 and §A.5. This is the bigger
engineering lift and is **not** in Sa2VA.

## Two pieces

1. **Abduction source inversion** — encode/invert the source video with the Wan
   VAE → `源latent` = the invariant prior (the engineered U). This is
   [true novelty ①](../../docs/TRACEABILITY.md). Interface: `e2w_core.latent.Abductor`.
2. **Gated Renderer** — a DiT conditioned on `源latent` + `edit-plan tokens` +
   the three-layer mask. **Core mechanism = mask-gated inpainting:** every denoise
   step *pastes the source latent back* in the UNCHANGED region (architecturally
   pinned, not "encouraged"); DIRECT/INDIRECT regions denoise freely under the
   tokens. Minimal-change is therefore structural, not optimized-for.

> No 2nd-pass (architecture deviation from the original proposal): the seam is
> solved in a single denoise via feather + joint denoising of gated and preserved
> regions, avoiding boundary artifacts.

## Training signals (architecture §A.5)

- main: flow-matching / denoise reconstruction against sim CF ground truth;
- **invariant-preservation loss** ([novelty ③](../../docs/TRACEABILITY.md)):
  UNCHANGED-region latent must match `源latent` (cleanest under shared-seed sim);
- causal-mask supervision lives in `localization`, aligned to the sim dependency
  graph.

Note: the `[EDIT]` token path has **no gradient inside Sa2VA** — it only learns
once this renderer is attached, i.e. the two halves train jointly (sa2va-plan
change B).

## Planned layout (boundary B4)

```
generation/
  third_party/vace_wan/   vendored, pinned, never edited in place
  e2w_generation/         inversion → 源latent; mask-gated inpainting; invariant loss; renderer
  tests/
```

Depends on `e2w_core` only (consumes `ThreeLayerMask`, `EditPlan`, `SourceLatent`).
Must never import `localization` (B3).
