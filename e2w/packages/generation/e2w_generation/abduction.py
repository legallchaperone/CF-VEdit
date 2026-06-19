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
