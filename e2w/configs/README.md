# configs — training / inference configuration

V0 remove-only inference/training configs. External weights and versions are
**pinned here**, not vendored into the repo:

- `weights.v0.json` — resolved local `/data/cwx` paths plus upstream revisions.
- `vanilla.v0.json` — stock Sa2VA `[SEG]` localization + frozen CogVideoX-Fun/VOID pass1.
- `full.v0.json` — query-token localization + frozen CogVideoX-Fun/VOID pass1.

V0 pinned model families:

- MLLM: `ByteDance/Sa2VA-Qwen2_5-VL-7B`
- SAM2 backbone: `facebook/sam2-hiera-large`
- Renderer / VAE: `CogVideoX-Fun-V1.5-5b-InP` + VOID `void_pass1.safetensors`

`wan2_2_vace_fun_a14b` remains in `weights.v0.json` only as a historical/downloaded
weight pointer; current v0 code does not read it.
