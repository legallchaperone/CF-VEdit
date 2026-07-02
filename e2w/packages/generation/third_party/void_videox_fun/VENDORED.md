# Vendored: VOID's VideoX-Fun fork (CogVideoX-Fun renderer)

- **Upstream:** `netflix/void-model` (arXiv:2604.02296), the `videox_fun/` package.
- **License:** Apache-2.0 (VOID weights+code). Base CogVideoX-Fun = CogVideoX License (academic-research free). See `E2W-v0-Remove-Only-Spec.md` §5.1.
- **Vendored on:** 2026-07-02, from `/data/cwx/void-model/videox_fun`.
- **Why:** e2w's other vendored fork (`../videox_fun`) is stock aigc-apps VideoX-Fun and lacks VOID's quadmask conditioning mods (`use_vae_mask`/`stack_mask` → 48-channel mask-in-latent concat). The E2W v0 renderer (ADR-0007) needs those mods to load `void_pass1.safetensors` and reuse VOID's mask channel-concat.
- **Untouched (boundary B4):** upstream sources are not modified. E2W's renderer sys.path-injects this dir and imports `videox_fun` at runtime; all E2W additions live in `e2w_generation/`, never here.
- **Re-vendor:** `cp -r /data/cwx/void-model/videox_fun <this_dir>/videox_fun` (drop any `__pycache__`).

Provides: `CogVideoXTransformer3DModel`, `AutoencoderKLCogVideoX`, `CogVideoXFunInpaintPipeline` (48-ch inpaint path, `use_vae_mask=True`).
