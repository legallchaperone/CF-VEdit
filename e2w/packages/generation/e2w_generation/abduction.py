"""Wan/VACE source abduction for E2W V0.

The SourceLatent payload intentionally carries both the source path and the
materialized Wan latent. To avoid loading the Wan VAE twice, V0 lets the renderer
materialize the latent lazily with its already-loaded VAE when `latents` is None.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_core.latent import Abductor, SourceLatent


@dataclass
class WanSourcePayload:
    video_path: str
    latents: Any = None
    video_tensor: Any = None
    fps: float | None = None
    num_frames: int | None = None
    height: int | None = None
    width: int | None = None


class WanAbductor(Abductor):
    """Create a SourceLatent payload for Wan/VACE.

    The actual VAE encode is performed by GatedRenderer.render once the shared VAE
    is loaded. This still preserves the seam: integration passes a SourceLatent;
    generation consumes it and pins unchanged latents during denoising.
    """

    def invert(self, video: str | Path) -> SourceLatent:
        meta = self._probe(video)
        return SourceLatent(latent=WanSourcePayload(video_path=str(video), **meta))

    def encode_only(self, source_video: Any, *, pipeline: Any, sample_size: tuple[int, int],
                    weight_dtype: Any, device: Any) -> Any:
        """Encode a loaded source-video tensor to the Wan latent — the G1 hook.

        Architecture A.2【1】: the source latent is the engineered exogenous U. V0
        materializes it lazily in the renderer to avoid a double VAE load; this
        exposes the *same* encode (``renderer.encode_source_to_latent``) so the G1
        round-trip ``decode(invert(V)) ≈ V`` (02:91) is testable in isolation once a
        pipeline/VAE is loaded. ``source_video`` is a ``(b,c,f,h,w)`` tensor, e.g.
        from the renderer backend's ``get_video_to_video_latent``. Lazy import keeps
        the abduction↔renderer module pair free of an import cycle.
        """
        from .renderer import encode_source_to_latent

        return encode_source_to_latent(
            pipeline, source_video,
            height=sample_size[0], width=sample_size[1],
            weight_dtype=weight_dtype, device=device,
        )

    @staticmethod
    def _probe(video: str | Path) -> dict[str, Any]:
        import cv2
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        cap.release()
        return {"fps": fps, "num_frames": frames, "height": height, "width": width}
