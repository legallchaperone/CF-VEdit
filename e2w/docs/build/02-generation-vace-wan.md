# 02 — Generation half: abduction inversion + gated renderer (VACE/Wan)

> **⚠️ Superseded for v0 (ADR-0007).** v0 drops the abduction source-inversion
> step entirely and swaps VACE/Wan for a **frozen CogVideoX-Fun-V1.5-5b-InP**
> initialized from VOID's `void_pass1.safetensors` — see
> [`E2W-v0-Remove-Only-Spec.md`](../../../E2W-v0-Remove-Only-Spec.md) §1.3–1.5.
> Novelties ① and ③ referenced below are dropped for v0, not built here; see
> TRACEABILITY's "Superseded novelties" table. This doc describes the
> pre-pivot target; existing `e2w_generation` code has not yet been ported.

> Builds the **Abduction source inversion** and the **gated Renderer** (actually
> renders the result). Reuses Wan2.2 (DiT + VAE) and VACE (masked-V2V
> conditioning). Implements architecture §A.2【1】+【4】 and §A.5. Home of true
> novelties ① (source latent = U) and ③ (invariant-preservation loss).
>
> **Consumes across the seam:** `e2w_core.masks.ThreeLayerMask` +
> `e2w_core.plan.EditPlan.edit_tokens` (from [01]) and its own
> `e2w_core.latent.SourceLatent`. Never imports the localization half (B3).

## Block 1 — Abduction source inversion (the U prior)

Encode/invert the source video into the renderer's latent space; this latent is
the engineered exogenous **U** — "everything reconstructable from the source",
pasted back in the UNCHANGED region every denoise step.

- Use the **Wan VAE** to encode source frames → latent. For an editable starting
  point, run the renderer's inversion (DDIM/flow inversion) to recover an initial
  noised latent that reconstructs the source under the renderer. ⚠️ verify
  whether VACE/Wan ship an inversion util or you implement flow-inversion against
  their scheduler.
- Implement `e2w_core.latent.Abductor`:

```python
# packages/generation/e2w_generation/abduction.py  (to write)
from e2w_core.latent import Abductor, SourceLatent

class WanAbductor(Abductor):
    def invert(self, video) -> SourceLatent:
        # Wan VAE encode (+ optional flow inversion) -> latent (T, C, H', W')
        return SourceLatent(latent=...)
```

- This is novelty ①: no Bernini/VEGGIE/VOID equivalent. Open-domain inversion is
  only approximate — a known risk ([05]).

## Block 4 — Gated renderer (mask-gated inpainting on VACE/Wan)

A strong DiT (Wan-14B class) conditioned on three things and gated by the mask.

**Conditions:**
1. `SourceLatent.latent` — identity/detail + the pinned invariants.
2. `EditPlan.edit_tokens` — content condition for the changed region (cross-attn
   stream; matches the width `edit_hidden_fcs` projects to in [01] change B).
3. `ThreeLayerMask` — the spatiotemporal gate.

**Core mechanism — mask-gated inpainting (the structural minimal-change):**
at each denoise step, in the **UNCHANGED** region overwrite the working latent with
the source latent noised to the current timestep (replacement, like inpainting);
in **DIRECT/INDIRECT** regions denoise freely under `edit_tokens`. Minimal-change
is then architectural, not optimized-for (architecture §A.2【4】 / B.2: "same U" =
"paste source latent back in unchanged region").

- VACE already does masked video-to-video with reference + mask conditioning —
  map our inputs onto its interface: our UNCHANGED region = VACE's preserve/keep
  region; DIRECT+INDIRECT = the generate/inpaint region; `edit_tokens` = the
  content/condition stream; `SourceLatent` = the reference/source latent. ⚠️
  **verify VACE's exact conditioning API and mask convention** and adapt — do not
  assume names.
- **Seam handling (no 2nd pass; architecture deviation, §A.2 note):** feather the
  mask boundary and **jointly denoise** gated and preserved regions rather than
  hard-pasting only at the final step, to avoid boundary/morphing artifacts.

```python
# packages/generation/e2w_generation/renderer.py  (to write)
from e2w_core.masks import ThreeLayerMask
from e2w_core.latent import SourceLatent

class GatedRenderer:
    def render(self, source: SourceLatent, edit_tokens, mask: ThreeLayerMask):
        # denoise loop with mask-gated replacement in mask.unchanged();
        # feather + joint denoise at the boundary -> edited video V̂
        ...
```

Note `ThreeLayerMask.unchanged()` is a contract stub in `e2w_core` — implement the
complement of `direct ∪ indirect` here on your array backend.

## Training signals owned by this half (architecture §A.5)

- **Main:** flow-matching / denoising loss against sim GT `V*` (latent space).
- **Invariant-preservation loss (novelty ③):** on the UNCHANGED region, the
  generated latent must match `SourceLatent.latent` (L2 in latent space). Cleanest
  under shared-seed sim where non-descendants are bit-identical ([03]).
- The **causal-mask loss** lives in [01] (localization), supervised by [03]'s
  dependency graph — not here.

## Component acceptance (see [05])

- **inversion round-trip:** decode(invert(V)) reconstructs V below an error
  threshold;
- **invariant loss bites:** on a no-op edit, UNCHANGED-region latent ≈ source
  latent → preservation axis ≈ 1 (the `copy_source` lower-bound behavior);
- **gating works:** with a hand-made mask, only masked regions change; preserved
  regions are pixel/identity-stable;
- no visible seam artifacts at mask boundaries on a small qualitative set.

## Honest risk

The bigger engineering lift (not in Sa2VA). Open-domain abduction is approximate;
seam artifacts are real (mitigated by feather + joint denoise); and `edit_tokens`
only acquire gradient once this renderer is attached → **the two halves must train
jointly** in stage ② ([04]).
