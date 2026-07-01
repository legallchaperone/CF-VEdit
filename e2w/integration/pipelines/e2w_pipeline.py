"""E2W V0 vanilla integration pipeline.

Integration is the only layer that imports both localization and generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_generation.abduction import WanAbductor
from e2w_generation.renderer import GatedRenderer, RendererConfig
from e2w_localization.planner import CausalPlanner, PlannerConfig


@dataclass(frozen=True)
class E2WConfig:
    weights_config: dict[str, Any]
    vanilla_config: dict[str, Any]
    config_dir: Path

    @classmethod
    def load(cls, config_path: str | Path) -> "E2WConfig":
        config_path = Path(config_path).resolve()
        vanilla = json.loads(config_path.read_text())
        weights_path = (config_path.parent / vanilla["weights_config"]).resolve()
        weights = json.loads(weights_path.read_text())
        return cls(weights_config=weights, vanilla_config=vanilla, config_dir=config_path.parent)


def _repo_e2w_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_v0_pipeline(config_path: str | Path) -> "E2WPipeline":
    cfg = E2WConfig.load(config_path)
    weights = cfg.weights_config["models"]
    vanilla = cfg.vanilla_config
    e2w_root = _repo_e2w_root()

    planner_cfg = PlannerConfig(
        weights_path=weights["sa2va_qwen2_5_vl_7b"]["path"],
        device=vanilla.get("device", "cuda:0"),
        dtype=vanilla.get("dtype", "bfloat16"),
        **vanilla.get("localization", {}),
    )
    gen = vanilla.get("generation", {})
    renderer_cfg = RendererConfig(
        weights_path=weights["wan2_2_vace_fun_a14b"]["path"],
        videox_fun_root=str((e2w_root / gen["videox_fun_root"]).resolve()),
        config_path=str((e2w_root / gen["config_path"]).resolve()),
        device=vanilla.get("device", "cuda:0"),
        dtype=vanilla.get("dtype", "bfloat16"),
        sample_size=tuple(gen.get("sample_size", [480, 832])),
        fps=int(gen.get("fps", 12)),
        num_inference_steps=int(gen.get("num_inference_steps", 20)),
        guidance_scale=float(gen.get("guidance_scale", 5.0)),
        negative_prompt=gen.get("negative_prompt", "low quality"),
        sampler_name=gen.get("sampler_name", "Flow"),
        shift=float(gen.get("shift", 12.0)),
        gpu_memory_mode=gen.get("gpu_memory_mode", "sequential_cpu_offload"),
        enable_teacache=bool(gen.get("enable_teacache", True)),
        teacache_threshold=float(gen.get("teacache_threshold", 0.10)),
        num_skip_start_steps=int(gen.get("num_skip_start_steps", 5)),
        cfg_skip_ratio=gen.get("cfg_skip_ratio", 0),
        vace_context_scale=float(gen.get("vace_context_scale", 1.0)),
        paste_back_source_latent=bool(gen.get("paste_back_source_latent", True)),
        paste_noise_to_timestep=bool(gen.get("paste_noise_to_timestep", True)),
        mask_feather_latent=bool(gen.get("mask_feather_latent", True)),
        seed=int(gen.get("seed", 43)),
    )
    return E2WPipeline(
        abductor=WanAbductor(),
        planner=CausalPlanner(planner_cfg),
        renderer=GatedRenderer(renderer_cfg),
    )


class E2WPipeline:
    def __init__(self, *, abductor: WanAbductor, planner: CausalPlanner, renderer: GatedRenderer):
        self.abductor = abductor
        self.planner = planner
        self.renderer = renderer

    def edit(self, video_path: str | Path, instruction: str, *, target_ref: str, operation: str,
             out_path: str | Path, vanilla: bool = False) -> Path:
        source = self.abductor.invert(video_path)
        mask, plan = self.planner.plan(
            video_path,
            instruction,
            target_ref=target_ref,
            operation=operation,
            vanilla=vanilla,
        )
        # Vanilla: [EDIT] bypassed, VACE receives the native instruction string.
        # Full A.1: VACE receives the planner's edit_tokens as the content condition.
        edit_condition = instruction if vanilla else plan.edit_tokens
        return self.renderer.render(source, edit_condition, mask, out_path=out_path)
