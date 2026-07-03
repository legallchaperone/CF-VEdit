# generation — 生成半 (CogVideoX-Fun/VOID renderer)

Actually **renders the result** for E2W v0 remove-only. Built on frozen
CogVideoX-Fun-V1.5-5b-InP + VOID `void_pass1.safetensors` (ADR-0007), pass1 only.
Depends on `e2w_core` only; must never import `localization` (B3).

## Two pieces

1. **Source payload** — `void_abduction.py` probes the source video and carries
   pixels/metadata inside `SourceLatent`; v0 has no VAE source inversion.
2. **VOID renderer** — `void_renderer.py` maps `ThreeLayerMask` to VOID quadmask
   conditioning and calls the vendored VOID CogVideoX-Fun inpaint pipeline.

## Layout (boundary B4)

```
generation/
  third_party/void_videox_fun/  vendored, pinned, never edited in place
  e2w_generation/              source payload + quadmask renderer wrapper
  tests/
```
