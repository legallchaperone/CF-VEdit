"""CogVideoX-Fun / VOID `void_pass1` renderer for E2W v0 (remove-only).

Thin wrapper over the vendored VOID fork (boundary B4: third_party untouched).
Implements ADR-0007 / `E2W-v0-Remove-Only-Spec.md`: renderer swaps from VACE/Wan
to CogVideoX-Fun-V1.5-5b-InP initialized from `void_pass1.safetensors`, frozen,
pass1 only. Depends on ``e2w_core`` only; must never import ``e2w_localization``
(boundary B3).

Load logic mirrors VOID's ``inference/cogvideox_fun/predict_v2v.py:load_pipeline``
with the CLI/omegaconf config dependency stripped out. The pipeline-call params
reproduce ``config/quadmask_cogvideox.py`` + ``run_inference`` (pass1).

--------------------------------------------------------------------------------
Three deviations from the porting brief, each forced by the vendored source
(cited inline) — see module-level constants and the mask/temporal helpers:

  1. MASK INVERSION. The brief said "map direct->0 / indirect->127 / else->255,
     /255, feed directly as mask_video". That is the on-disk quadmask FILE
     convention (spec §1.3), but VOID's ``get_video_mask_input`` applies
     ``255 - input_mask`` before the pipeline (utils.py, use_quadmask branch),
     so the pipeline boundary wants REMOVE=1.0 / KEEP=0.0 — the inverse. Feeding
     file-convention values directly would preserve the object and regenerate the
     background. ``three_layer_to_quadmask`` keeps the spec §1.3 file convention;
     ``void_quadmask_to_pipeline_mask`` performs the inversion.
  2. TEMPORAL PADDING. The pipeline sets ``video_length = video.shape[2]`` and
     never pads; with ``num_frames=85`` and a 21-frame source you get a latent
     shape mismatch. VOID pads the source 21->85 by flip-repeat via
     ``temporal_padding`` (utils.py) before the call; we replicate it.
  3. AREA RESIZE. ``get_video_mask_input`` resizes video & mask to sample_size
     with ``F.interpolate(mode="area")`` up front (not the pipeline's internal
     bilinear preprocess); we do the same for pass1 fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_core.latent import SourceLatent
from e2w_core.masks import ThreeLayerMask

from .void_abduction import CogVideoXSourcePayload

# VOID pass1 negative prompt (config/quadmask_cogvideox.py).
VOID_NEGATIVE_PROMPT = (
    "The video is not of a high quality, it has a low resolution. Watermark "
    "present in each frame. The background is solid. Strange body and strange "
    "trajectory. Distortion. "
)

# patch_embed.proj.weight partial-overwrite constants (predict_v2v.load_pipeline).
_LATENT_CH = 16
_FEAT_SCALE = 8
_FEAT_DIM = _LATENT_CH * _FEAT_SCALE  # 128


# --------------------------------------------------------------------------- #
# Pure mask helpers (unit-testable, no torch/GPU needed for the first).
# --------------------------------------------------------------------------- #
def three_layer_to_quadmask(direct, indirect):
    """``ThreeLayerMask`` layers -> VOID single-channel quadmask (uint8).

    Spec §1.3 FILE convention (aligns to the renderer's trained input dist):

        direct              -> 0   (remove)
        indirect \\ direct   -> 127 (affected)
        else                -> 255 (keep)

    ``direct`` / ``indirect`` are boolean ``(T, H, W)`` stacks. Returns a
    ``(T, H, W)`` uint8 array with values in ``{0, 127, 255}``. This is the
    on-disk convention; ``void_quadmask_to_pipeline_mask`` inverts it for the
    pipeline (see module docstring, deviation #1).
    """
    import numpy as np

    direct = np.asarray(direct).astype(bool)
    indirect = np.asarray(indirect).astype(bool)
    if direct.shape != indirect.shape:
        raise ValueError(f"direct {direct.shape} != indirect {indirect.shape}")

    out = np.full(direct.shape, 255, dtype=np.uint8)   # keep
    out[np.logical_and(indirect, np.logical_not(direct))] = 127  # affected
    out[direct] = 0  # remove (highest priority; PIXEL_PRIORITY direct > indirect)
    return out


def void_quadmask_to_pipeline_mask(quadmask_uint8, *, sample_size):
    """VOID quadmask (uint8 file convention) -> pipeline ``mask_video`` tensor.

    Replicates ``get_video_mask_input``'s ``use_quadmask`` branch (vendored
    utils.py) exactly, IN ORDER:
      area-resize to sample_size -> quantize to {0,63,127,255} -> ``255 - x`` ->
      ``/255``.
    Result ``(1, 1, T, H, W)`` float in [0,1] with REMOVE=1.0 / KEEP=0.0 (the
    inversion; module docstring deviation #1). Accepts ``(T,H,W)`` or
    ``(T,H,W,C)`` uint8. Temporal padding is applied separately by the renderer.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    m = np.asarray(quadmask_uint8)
    if m.ndim == 4:  # (T,H,W,C) -> single channel
        m = m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"quadmask must be (T,H,W[,C]); got {m.shape}")

    t = torch.from_numpy(m.astype(np.float32))  # (T,H,W)
    # area-resize to sample_size (get_video_mask_input: F.interpolate mode='area')
    t = F.interpolate(t.unsqueeze(0), size=tuple(sample_size), mode="area").squeeze(0)
    # quadmask quantization to {0,63,127,255}
    t = torch.where(t <= 31, torch.zeros_like(t), t)
    t = torch.where((t > 31) & (t <= 95), torch.full_like(t, 63.0), t)
    t = torch.where((t > 95) & (t <= 191), torch.full_like(t, 127.0), t)
    t = torch.where(t > 191, torch.full_like(t, 255.0), t)
    # invert + normalize: remove(0)->1.0, keep(255)->0.0, affected(127)->~0.5
    t = (255.0 - t) / 255.0
    return t.unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)


def _temporal_pad(tensor, *, target_length: int):
    """Replicate VOID ``temporal_padding`` on dim=2 (flip-repeat to length).

    ``tensor`` is ``(b, c, T, h, w)``. Matches utils.py ``temporal_padding``:
    truncates to ``target_length`` (a no-op when shorter) then flip-repeats.
    """
    import torch

    length = tensor.shape[2]
    if length == target_length:
        return tensor
    tensor = tensor[:, :, :target_length]
    while tensor.shape[2] < target_length:
        tensor = torch.cat([tensor, torch.flip(tensor, [2])], dim=2)[:, :, :target_length]
    return tensor


@dataclass
class VoidRendererConfig:
    base_path: str                       # CogVideoX-Fun-V1.5-5b-InP dir
    void_pass1_path: str                 # void_pass1.safetensors
    device: str = "cuda"
    weight_dtype: str = "bfloat16"
    # "model_cpu_offload_and_qfloat8": compatibility alias; effective bf16 offload.
    # "model_full_load": training, keeps grad to encoder_hidden_states (bf16).
    # "model_cpu_offload": inference, no float8.
    # Stage 2 training HARD-REQUIRES model_full_load (spec §2 Stage 2 / §5.10):
    # offload modes are not valid for the edit-token gradient path.
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8"
    sample_size: tuple[int, int] = (384, 672)     # (H, W)
    num_frames: int = 85
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    denoise_strength: float = 1.0
    seed: int = 42
    fps: int = 12
    negative_prompt: str = VOID_NEGATIVE_PROMPT
    out_size: tuple[int, int] = (832, 480)        # (W, H) benchmark output
    use_vae_mask: bool = True
    stack_mask: bool = False
    use_trimask: bool = True                      # hardcoded True in run_inference
    zero_out_mask_region: bool = False


class VoidRenderer:
    """Frozen CogVideoX-Fun / VOID pass1 renderer (E2W v0 remove-only)."""

    def __init__(self, config: VoidRendererConfig):
        self.config = config
        self._backend: dict[str, Any] | None = None

    # ---------------------- public render API ---------------------------- #
    def render(self, source: SourceLatent, edit_condition: "str | Any",
               mask: ThreeLayerMask, *, out_path: str | Path) -> Path:
        """Render one edited video and save it to ``out_path``.

        ``source.latent`` carries a ``CogVideoXSourcePayload`` (source pixels +
        metadata; no VAE inversion — v0 conditions on VOID's mask+masked-latent
        channel-concat, ADR-0007). ``edit_condition`` is the VOID text prompt
        (vanilla) or the planner's ``edit_tokens`` tensor ``(Nt, 4096)`` (full).
        ``mask`` is the E2W ``ThreeLayerMask``.
        """
        import numpy as np

        payload = self._payload(source)
        video_tensor = self._load_source_tensor(payload)  # (1,3,T,H,W) [0,1]

        direct = np.asarray(mask.direct).astype(bool)
        indirect = np.asarray(mask.indirect).astype(bool)
        quadmask = three_layer_to_quadmask(direct, indirect)  # (T,H,W) uint8
        return self.render_from_quadmask(video_tensor, quadmask, edit_condition, out_path)

    def render_from_quadmask(self, video_tensor, quadmask_uint8,
                             text_or_embeds: "str | Any", out_path: str | Path) -> Path:
        """M2 fidelity entry — render straight from a VOID quadmask (uint8).

        Bypasses ``ThreeLayerMask`` conversion so backend wiring can be checked
        against a known-good VOID quadmask (reproduce pass1). ``video_tensor`` is
        ``(1,3,T,H,W)`` float in [0,1] (source resolution ok); ``quadmask_uint8``
        is the VOID file-convention quadmask ``(T,H,W[,C])`` in {0,63,127,255}.
        ``text_or_embeds`` is a str (vanilla) or ``(Nt,4096)`` tensor (full).
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        cfg = self.config
        sample_size = tuple(cfg.sample_size)  # (H, W)

        # --- video: area-resize spatial to sample_size, temporal-pad to num_frames
        video = video_tensor
        if not torch.is_tensor(video):
            video = torch.as_tensor(video)
        video = video.float()
        b, c, t0, h0, w0 = video.shape
        quadmask_arr = np.asarray(quadmask_uint8)
        if quadmask_arr.ndim not in (3, 4):
            raise ValueError(f"quadmask must be (T,H,W[,C]); got {quadmask_arr.shape}")
        if quadmask_arr.shape[0] != t0:
            raise ValueError(
                f"quadmask frames ({quadmask_arr.shape[0]}) must match video frames ({t0}) "
                "before temporal padding"
            )

        backend = self._load_backend()
        pipeline = backend["pipeline"]
        device = backend["device"]
        weight_dtype = backend["weight_dtype"]

        if (h0, w0) != sample_size:
            # area-resize spatial dims per-frame (matches get_video_mask_input)
            v = video.permute(0, 2, 1, 3, 4).reshape(b * t0, c, h0, w0)
            v = F.interpolate(v, size=sample_size, mode="area")
            video = v.reshape(b, t0, c, sample_size[0], sample_size[1]).permute(0, 2, 1, 3, 4)
        video = _temporal_pad(video, target_length=cfg.num_frames)

        # --- mask: file-convention quadmask -> pipeline mask, temporal-pad
        mask_video = void_quadmask_to_pipeline_mask(quadmask_arr, sample_size=sample_size)
        mask_video = _temporal_pad(mask_video, target_length=cfg.num_frames)

        video = video.to(device=device, dtype=weight_dtype)
        mask_video = mask_video.to(device=device, dtype=weight_dtype)

        # --- edit branch: vanilla (T5 text) vs full (edit_tokens embeds)
        prompt: Any = None
        prompt_embeds = None
        if isinstance(text_or_embeds, str):
            prompt = text_or_embeds
        else:
            emb = text_or_embeds
            if not torch.is_tensor(emb):
                emb = torch.as_tensor(emb)
            emb = emb.to(device=device, dtype=weight_dtype)
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)  # (1, Nt, 4096)
            prompt_embeds = emb  # guidance_scale=1.0 -> CFG off, no negative embeds

        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

        sample = pipeline(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            num_frames=cfg.num_frames,
            negative_prompt=cfg.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=float(cfg.guidance_scale),
            num_inference_steps=int(cfg.num_inference_steps),
            video=video,
            mask_video=mask_video,
            strength=float(cfg.denoise_strength),
            use_trimask=bool(cfg.use_trimask),
            zero_out_mask_region=bool(cfg.zero_out_mask_region),
            use_vae_mask=bool(cfg.use_vae_mask),
            stack_mask=bool(cfg.stack_mask),
            output_type="numpy",
            return_dict=False,
        ).videos  # (b,c,f,h,w); torch tensor in [0,1] (return_dict=False wraps a tensor)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._postprocess_and_save(sample, out_path)
        return out_path

    # ---------------------- post-processing ------------------------------ #
    def _postprocess_and_save(self, sample, out_path: Path) -> None:
        """raw (1,3,85,384,672) -> benchmark 832x480 / 21f @ 12fps h264.

        Frame indices 2,6,...,82 (``2 + 4*i``, i=0..20) then non-aspect resize to
        ``out_size`` (W,H), encode h264/yuv420p (matches ``predictions/void``).
        """
        import numpy as np
        import torch

        if torch.is_tensor(sample):
            sample = sample.detach().float().cpu().numpy()
        sample = np.asarray(sample)
        # (b,c,f,h,w) -> (f,h,w,c)
        vid = np.transpose(sample[0], (1, 2, 3, 0))  # (f,h,w,c)
        n = vid.shape[0]
        idx = [2 + 4 * i for i in range(21)]
        idx = [j for j in idx if j < n]
        vid = vid[idx]
        vid = np.clip(vid * 255.0, 0, 255).astype(np.uint8)

        out_w, out_h = self.config.out_size  # (W, H)
        frames = self._resize_frames(vid, out_w=out_w, out_h=out_h)
        self._write_h264(frames, out_path, fps=int(self.config.fps))

    @staticmethod
    def _resize_frames(frames, *, out_w: int, out_h: int):
        import cv2
        import numpy as np

        out = np.empty((frames.shape[0], out_h, out_w, frames.shape[3]), dtype=np.uint8)
        for i, f in enumerate(frames):
            out[i] = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return out

    @staticmethod
    def _write_h264(frames, out_path: Path, *, fps: int) -> None:
        import imageio

        with imageio.get_writer(
            str(out_path), fps=fps, codec="libx264",
            format="FFMPEG", macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        ) as w:
            for f in frames:
                w.append_data(f)

    # ---------------------- source I/O ----------------------------------- #
    def _load_source_tensor(self, payload: CogVideoXSourcePayload):
        """Return the source video as ``(1,3,T,H,W)`` float [0,1]."""
        import torch

        if payload.video_tensor is not None:
            v = payload.video_tensor
            return v if torch.is_tensor(v) else torch.as_tensor(v)
        return self._read_video(payload.video_path)

    @staticmethod
    def _read_video(path: str):
        import numpy as np
        import torch
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {path}")
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"no frames read from: {path}")
        arr = np.stack(frames).astype(np.float32) / 255.0  # (T,H,W,C)
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,C,T,H,W)
        return t

    # ---------------------- backend load (VOID pass1) -------------------- #
    def _load_backend(self) -> dict[str, Any]:
        if self._backend is not None:
            return self._backend

        import sys
        import torch
        from safetensors.torch import load_file
        from diffusers import DDIMScheduler
        from transformers import T5EncoderModel, T5Tokenizer

        cfg = self.config
        # vendored VOID fork parent dir on sys.path -> `import videox_fun`
        vendored = (Path(__file__).resolve().parent.parent
                    / "third_party" / "void_videox_fun")
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))

        from videox_fun.models import (AutoencoderKLCogVideoX,
                                        CogVideoXTransformer3DModel)
        from videox_fun.pipeline import CogVideoXFunInpaintPipeline

        weight_dtype = torch.bfloat16 if cfg.weight_dtype == "bfloat16" else torch.float16
        device = cfg.device
        base = str(Path(cfg.base_path).resolve())

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            base, subfolder="transformer", low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
            use_vae_mask=cfg.use_vae_mask, stack_mask=cfg.stack_mask,
        ).to(weight_dtype)

        # load void_pass1 with patch_embed 48-ch partial overwrite (predict_v2v)
        state_dict = load_file(str(Path(cfg.void_pass1_path).resolve()))
        state_dict = state_dict.get("state_dict", state_dict)
        p = "patch_embed.proj.weight"
        if (cfg.use_vae_mask or cfg.stack_mask) and \
           state_dict[p].size(1) != transformer.state_dict()[p].size(1):
            new_weight = transformer.state_dict()[p].clone()
            new_weight[:, :_FEAT_DIM] = state_dict[p][:, :_FEAT_DIM]
            new_weight[:, -_FEAT_DIM:] = state_dict[p][:, -_FEAT_DIM:]
            state_dict[p] = new_weight  # middle channels keep base weights
        m, u = transformer.load_state_dict(state_dict, strict=False)
        if len(m) or len(u):
            # pass1 should be missing=0 unexpected=0; surface anything unexpected.
            import warnings
            warnings.warn(f"void_pass1 load: missing={len(m)} unexpected={len(u)}")

        vae = AutoencoderKLCogVideoX.from_pretrained(base, subfolder="vae").to(weight_dtype)
        tokenizer = T5Tokenizer.from_pretrained(base, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=weight_dtype)
        scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")

        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
            transformer=transformer, scheduler=scheduler,
        )

        mode = cfg.gpu_memory_mode
        if mode == "model_cpu_offload_and_qfloat8":
            import warnings
            warnings.warn(
                "gpu_memory_mode=model_cpu_offload_and_qfloat8 is a compatibility no-op "
                "in E2W: running effective bf16 model_cpu_offload for numerical parity "
                "with the VOID baseline; no float8 VRAM saving is claimed.",
                RuntimeWarning,
            )
            pipeline.enable_model_cpu_offload(device=device)
        elif mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:  # model_full_load — keeps grad path for training
            pipeline.to(device=device)

        transformer.requires_grad_(False)  # renderer frozen (infer + train)

        self._backend = {
            "pipeline": pipeline,
            "device": device,
            "weight_dtype": weight_dtype,
        }
        return self._backend

    @staticmethod
    def _payload(source: SourceLatent) -> CogVideoXSourcePayload:
        payload = source.latent
        if not isinstance(payload, CogVideoXSourcePayload):
            raise TypeError(
                f"VoidRenderer expected CogVideoXSourcePayload, got {type(payload)!r}")
        return payload
