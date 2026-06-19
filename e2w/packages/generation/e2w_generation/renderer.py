"""Wan2.2 VACE-Fun gated renderer for E2W V0 vanilla mode.

Verified upstream source:
- weights README points to github.com/aigc-apps/VideoX-Fun;
- example `examples/wan2.2_vace_fun/predict_v2v_mask.py` loads
  VaceWanTransformer3DModel + Wan2_2VaceFunPipeline;
- pipeline mask convention: mask_video > 0.5 is the reactive/inpaint region,
  mask_video < 0.5 is preserved source.

E2W addition (outside third_party): after each denoise step a callback pastes the
source Wan latent back where DIRECT∪INDIRECT is false. third_party is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_core.latent import SourceLatent
from e2w_core.masks import ThreeLayerMask

from .abduction import WanSourcePayload


@dataclass(frozen=True)
class RendererConfig:
    weights_path: str
    videox_fun_root: str
    config_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sample_size: tuple[int, int] = (480, 832)
    fps: int = 12
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    negative_prompt: str = "low quality, blurry, artifacts"
    sampler_name: str = "Flow"
    shift: float = 12.0
    gpu_memory_mode: str = "sequential_cpu_offload"
    enable_teacache: bool = True
    teacache_threshold: float = 0.10
    num_skip_start_steps: int = 5
    cfg_skip_ratio: float | None = 0
    vace_context_scale: float = 1.0
    paste_back_source_latent: bool = True
    seed: int = 43


class GatedRenderer:
    def __init__(self, config: RendererConfig):
        self.config = config
        self._backend: dict[str, Any] | None = None

    def render(self, source: SourceLatent, edit_condition: str, mask: ThreeLayerMask, *, out_path: str | Path) -> Path:
        """Render one edited video and save it to out_path.

        `edit_condition` is the native VACE text prompt in vanilla mode. `mask`
        uses E2W seam semantics; DIRECT∪INDIRECT becomes VACE's inpaint region.
        """
        import torch
        from einops import rearrange

        backend = self._load_backend()
        pipeline = backend["pipeline"]
        device = backend["device"]
        weight_dtype = backend["weight_dtype"]
        save_videos_grid = backend["save_videos_grid"]
        get_video_to_video_latent = backend["get_video_to_video_latent"]

        payload = self._payload(source)
        sample_size = tuple(self.config.sample_size)
        video_length = int(payload.num_frames or self._mask_time(mask))
        video_length = max(1, video_length)

        source_video, _, _, _ = get_video_to_video_latent(
            payload.video_path,
            video_length=video_length,
            sample_size=sample_size,
            fps=self.config.fps,
            ref_image=None,
        )
        payload.video_tensor = source_video

        mask_video = self._mask_to_video_tensor(mask, video_length=source_video.shape[2], sample_size=sample_size)

        source_latents, latent_mask = self._materialize_source_latents(
            pipeline=pipeline,
            source_video=source_video,
            mask_video=mask_video,
            height=sample_size[0],
            width=sample_size[1],
            weight_dtype=weight_dtype,
            device=device,
        )
        payload.latents = source_latents

        callback = None
        if self.config.paste_back_source_latent:
            def paste_back_callback(pipe, step_index, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                src = source_latents.to(device=latents.device, dtype=latents.dtype)
                m = latent_mask.to(device=latents.device, dtype=latents.dtype)
                callback_kwargs["latents"] = (1.0 - m) * src + m * latents
                return callback_kwargs
            callback = paste_back_callback

        generator = torch.Generator(device=device).manual_seed(int(self.config.seed))
        with torch.no_grad():
            sample = pipeline(
                edit_condition,
                num_frames=source_video.shape[2],
                negative_prompt=self.config.negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=float(self.config.guidance_scale),
                num_inference_steps=int(self.config.num_inference_steps),
                video=source_video,
                mask_video=mask_video,
                control_video=None,
                subject_ref_images=None,
                boundary=backend["boundary"],
                shift=float(self.config.shift),
                vace_context_scale=float(self.config.vace_context_scale),
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            ).videos

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(sample, str(out_path), fps=int(payload.fps or self.config.fps))
        return out_path

    def _load_backend(self) -> dict[str, Any]:
        if self._backend is not None:
            return self._backend

        import os
        import sys
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        videox_root = Path(self.config.videox_fun_root).resolve()
        if str(videox_root) not in sys.path:
            sys.path.insert(0, str(videox_root))

        from videox_fun.dist import set_multi_gpus_devices
        from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8, VaceWanTransformer3DModel, WanT5EncoderModel
        from videox_fun.pipeline import Wan2_2VaceFunPipeline
        from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper, replace_parameters_by_name
        from videox_fun.utils.utils import filter_kwargs, get_video_to_video_latent, save_videos_grid
        from videox_fun.models.cache_utils import get_teacache_coefficients

        config = OmegaConf.load(str(Path(self.config.config_path).resolve()))
        model_name = str(Path(self.config.weights_path).resolve())
        device = set_multi_gpus_devices(1, 1)
        if self.config.device and self.config.device != "cuda:0":
            device = self.config.device
        weight_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

        transformer = VaceWanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
            transformer_2 = VaceWanTransformer3DModel.from_pretrained(
                os.path.join(model_name, config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer")),
                transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = None

        vae_cls = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
            config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")
        ]
        vae = vae_cls.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()

        schedulers = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }
        scheduler_cls = schedulers[self.config.sampler_name]
        scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
        if self.config.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
            scheduler_kwargs["shift"] = 1
        scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_kwargs))

        pipeline = Wan2_2VaceFunPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )

        mode = self.config.gpu_memory_mode
        if mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation"], device=device)
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif mode == "model_group_offload":
            register_auto_device_hook(pipeline.transformer)
            if transformer_2 is not None:
                register_auto_device_hook(pipeline.transformer_2)
            safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
        elif mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        if self.config.enable_teacache:
            coefficients = get_teacache_coefficients(model_name)
            if coefficients is not None:
                num_steps = int(self.config.num_inference_steps)
                num_skip = max(0, min(int(self.config.num_skip_start_steps), num_steps))
                pipeline.transformer.enable_teacache(
                    coefficients,
                    num_steps,
                    float(self.config.teacache_threshold),
                    num_skip_start_steps=num_skip,
                    offload=False,
                )
                if transformer_2 is not None:
                    pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        if self.config.cfg_skip_ratio is not None:
            pipeline.transformer.enable_cfg_skip(float(self.config.cfg_skip_ratio), int(self.config.num_inference_steps))
            if transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        self._backend = {
            "pipeline": pipeline,
            "device": device,
            "weight_dtype": weight_dtype,
            "boundary": boundary,
            "get_video_to_video_latent": get_video_to_video_latent,
            "save_videos_grid": save_videos_grid,
        }
        return self._backend

    @staticmethod
    def _payload(source: SourceLatent) -> WanSourcePayload:
        payload = source.latent
        if not isinstance(payload, WanSourcePayload):
            raise TypeError(f"GatedRenderer expected WanSourcePayload, got {type(payload)!r}")
        return payload

    @staticmethod
    def _mask_time(mask: ThreeLayerMask) -> int:
        return int(getattr(mask.direct, "shape", [1])[0])

    @staticmethod
    def _mask_to_video_tensor(mask: ThreeLayerMask, *, video_length: int, sample_size: tuple[int, int]):
        import numpy as np
        import torch
        import torch.nn.functional as F

        direct = np.asarray(mask.direct).astype(bool)
        indirect = np.asarray(mask.indirect).astype(bool)
        edit = np.logical_or(direct, indirect).astype("float32")
        if edit.shape[0] != video_length:
            src_idx = np.linspace(0, edit.shape[0] - 1, video_length).round().astype(int)
            edit = edit[src_idx]
        tensor = torch.from_numpy(edit).unsqueeze(0).unsqueeze(0)  # 1,1,T,H,W
        if tuple(edit.shape[-2:]) != tuple(sample_size):
            tensor = F.interpolate(tensor, size=(video_length, sample_size[0], sample_size[1]), mode="nearest")
        return tensor.float()

    @staticmethod
    def _materialize_source_latents(*, pipeline: Any, source_video: Any, mask_video: Any,
                                    height: int, width: int, weight_dtype: Any, device: Any):
        import torch
        import torch.nn.functional as F
        from einops import rearrange

        source_video = source_video.to(dtype=weight_dtype)
        mask_video = mask_video.to(dtype=weight_dtype)
        init_video = pipeline.image_processor.preprocess(
            rearrange(source_video, "b c f h w -> (b f) c h w"), height=height, width=width
        ).to(dtype=weight_dtype)
        init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=source_video.shape[2]).to(device=device)

        source_latents_list = pipeline.vace_encode_frames(init_video, ref_images=None, masks=None, vae=pipeline.vae)
        source_latents = torch.stack(source_latents_list, dim=0).to(device=device, dtype=weight_dtype)

        mask_condition = pipeline.mask_processor.preprocess(
            rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width
        ).to(dtype=torch.float32)
        mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=mask_video.shape[2])
        latent_mask = F.interpolate(mask_condition[:, :1], size=source_latents.shape[-3:], mode="nearest").to(
            device=device, dtype=weight_dtype
        )
        return source_latents, latent_mask
