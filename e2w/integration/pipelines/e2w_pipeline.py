"""E2W V0 vanilla integration pipeline.

Integration is the only layer that imports both localization and generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_generation.void_abduction import CogVideoXAbductor
from e2w_generation.void_renderer import VoidRenderer, VoidRendererConfig
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
    # Renderer = frozen CogVideoX-Fun/VOID pass1 (ADR-0007). Render params
    # (sample_size/num_frames/steps/guidance/negative/out_size) are fixed to
    # reproduce VOID pass1 (verified in M2) and live as VoidRendererConfig
    # defaults; config only overrides gpu_memory_mode (float8 infer vs bf16
    # full-load for training) and seed.
    gen = vanilla.get("generation", {})
    void = weights["cogvideox_fun_void"]
    renderer_cfg = VoidRendererConfig(
        base_path=void["base_path"],
        void_pass1_path=void["void_pass1_path"],
        device=vanilla.get("device", "cuda:0"),
        weight_dtype=vanilla.get("dtype", "bfloat16"),
        gpu_memory_mode=gen.get("gpu_memory_mode", "model_cpu_offload_and_qfloat8"),
        seed=int(gen.get("seed", 42)),
    )
    return E2WPipeline(
        abductor=CogVideoXAbductor(),
        planner=CausalPlanner(planner_cfg),
        renderer=VoidRenderer(renderer_cfg),
    )


class E2WPipeline:
    def __init__(self, *, abductor: CogVideoXAbductor, planner: CausalPlanner, renderer: VoidRenderer):
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
        # Vanilla: [EDIT] bypassed, renderer receives the native instruction string (T5).
        # Full A.1: renderer's T5 text position is hard-replaced by planner edit_tokens.
        edit_condition = instruction if vanilla else plan.edit_tokens
        return self.renderer.render(source, edit_condition, mask, out_path=out_path)
