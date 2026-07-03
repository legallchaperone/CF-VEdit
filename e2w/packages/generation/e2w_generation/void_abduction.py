"""CogVideoX-Fun / VOID source abduction for E2W v0 (remove-only).

ADR-0007: abduction-as-source-inversion is DROPPED for v0 — there is no
MLLM-inversion-to-renderer-latent step. Unchanged-region conditioning comes from
VOID's own mask + masked-video-latent channel-concat. So this "abductor" does not
VAE-encode anything; it only reads the source pixels + probes metadata and packs
them into a ``SourceLatent`` so the renderer seam is preserved.

Depends on ``e2w_core`` only; must never import ``e2w_localization`` (boundary B3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2w_core.latent import Abductor, SourceLatent


@dataclass
class CogVideoXSourcePayload:
    """Source payload carried inside ``SourceLatent.latent`` for the VOID renderer.

    No VAE latent — v0 conditions unchanged regions via VOID's channel-concat, not
    a source-inversion prior (ADR-0007). ``video_tensor`` (optional) is a
    ``(1,3,T,H,W)`` float [0,1] tensor; if absent the renderer reads ``video_path``.
    """

    video_path: str
    video_tensor: Any = None
    fps: float | None = None
    num_frames: int | None = None
    height: int | None = None
    width: int | None = None


class CogVideoXAbductor(Abductor):
    """Read a source video into a ``CogVideoXSourcePayload`` (no VAE encode)."""

    def invert(self, video: str | Path) -> SourceLatent:
        meta = self._probe(video)
        return SourceLatent(latent=CogVideoXSourcePayload(video_path=str(video), **meta))

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
