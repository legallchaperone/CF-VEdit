# configs — training / inference configuration

Three-stage training configs (proposal §2.7) and inference configs. External
weights and versions are **pinned here**, not vendored into the repo:

- `weights.v0.json` — resolved local `/data/cwx` paths plus upstream revisions for V0 vanilla eval.
- `vanilla.v0.json` — runtime settings for stock Sa2VA `[SEG]` localization and Wan2.2 VACE-Fun rendering.

V0 pinned model families:

- MLLM: `ByteDance/Sa2VA-Qwen2_5-VL-7B`
- SAM2 backbone: `facebook/sam2-hiera-large`
- Renderer / VAE: `alibaba-pai/Wan2.2-VACE-Fun-A14B` + Wan VAE

Stages: ① align (freeze MLLM, train alignment + renderer) → ② end-to-end (light
MLLM finetune on sim CF videos) → ③ optional RL/preference alignment. Don't train
all at once (VEGGIE failed that way).
