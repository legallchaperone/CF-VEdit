# configs — training / inference configuration

Three-stage training configs (proposal §2.7) and inference configs. External
weights and versions are **pinned here**, not vendored into the repo:

- MLLM: `ByteDance/Sa2VA-Qwen2_5-VL-7B`
- SAM2 backbone: `facebook/sam2-hiera-large`
- Renderer / VAE: Wan2.2 (Wan-14B class) + Wan VAE

Stages: ① align (freeze MLLM, train alignment + renderer) → ② end-to-end (light
MLLM finetune on sim CF videos) → ③ optional RL/preference alignment. Don't train
all at once (VEGGIE failed that way).
