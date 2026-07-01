# ADR-0005 — Renderer seam (noised+feathered paste-back) and edit_tokens conditioning

- **Status:** Accepted
- **Date:** 2026-06-20
- **Anchors:** build/02-generation-vace-wan.md §"Gated renderer" (02:48-63, 02:89-97); Architecture §A.2【4】 / §A.2 note; ADR-0003

## Context

Two renderer gaps, both fixable without training:

1. **The gate was wrong vs spec.** V0 hard-pasted the **clean** source latent with a
   nearest-neighbour mask every step (`renderer.py:103`), against `02:52` ("source
   latent **noised to the current timestep**") and `02:61-63` ("**feather** the
   boundary + **jointly denoise**, not hard-paste"). This injects a denoised region
   beside a noisy one each step and is the leading suspect for the preservation
   collapse (0.08, below the `copy_source` floor).
2. **edit_tokens have no clean entry.** The full path must feed `edit_tokens` as the
   positive content condition, but the VACE-Fun pipeline's external `prompt_embeds`
   path is brittle (`batch_size = prompt_embeds.shape[0]` on what is otherwise a
   list, while the CFG concat `negative + positive` expects lists).

## Decision

1. **Noised paste-back:** composite the source latent **noised to the working
   sigma** (`sigmas[step_index+1]`, flow-matching `x_σ=(1-σ)x₀+σ·noise`) using a
   fixed per-call noise drawn from a *separate* generator (pipeline init-noise
   stream untouched). Behind `paste_noise_to_timestep` (default on).
2. **Feather:** downsample the pixel mask to latent resolution with a soft
   (trilinear) kernel so the existing `(1-m)·src + m·latents` composite blends the
   seam. Behind `mask_feather_latent` (default on). This is the one-pass joint
   denoise the proposal mandates after dropping the 2nd pass (§A.2 note).
3. **edit_tokens injection:** `render()` accepts `edit_condition` as a string
   (vanilla) **or** a tensor (`edit_tokens`). For the tensor case, temporarily
   override `pipeline._get_t5_prompt_embeds` so the **positive** prompt returns
   `edit_tokens` and the **negative** still encodes the real negative text (the
   override is keyed on call order — positive first, then negative — and restored in
   a `finally`).

## Consequences

- **+** The seam matches the proposal's one-pass mandate; `edit_tokens` drive
  cross-attention as specified. Validated end-to-end on GPU (full run rendered a
  video with `edit_tokens` as the positive condition).
- **+** Both gate fixes are config toggles, so they can be A/B'd against the 0.08
  floor (and disabled to recover legacy behavior) — important because clean-paste
  was **not proven** to be the dominant cause of the collapse (the add-path 0.00 is
  a *localization* hallucination, not a paste-back bug).
- **−** The `_get_t5_prompt_embeds` override is a localized workaround for an
  upstream API quirk — reversible per call, but coupled to encode_prompt's
  positive-then-negative call order.
- **−** Noised-paste correctness assumes the flow-matching `α_t = 1-σ` convention
  on the configured scheduler; verify the σ range on first use.
