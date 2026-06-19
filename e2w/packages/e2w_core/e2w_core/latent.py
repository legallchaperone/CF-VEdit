"""Abduction source latent — the engineered exogenous U (invariant prior).

Implements: ``architecture.md §A.2`` 【1】 and Part B.2 (Pearl abduction → 源latent),
``proposal.md §2.6.1``. The source video is inverted into the renderer's VAE
latent; that latent is the invariant prior pasted back into the UNCHANGED region
at every denoise step. "Same U" in Pearl's abduction→action→prediction *is*
"paste the source latent back in the unchanged region".

This is **true novelty ①** (architecture.md §A.7): abduction = source inversion
to latent as an invariant prior — Bernini / VEGGIE / VOID have nothing like it.
Note: this preprocessor lives in the generation half (Wan VAE), not in Sa2VA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# torch.Tensor at runtime; dependency-free contract alias.
Tensor = Any


@dataclass(frozen=True)
class SourceLatent:
    """Everything reconstructable from the source = the pinned U.

    ``latent``: the inverted VAE latent, shape ``(T, C, H', W')``.
    """

    latent: Tensor


class Abductor(Protocol):
    """Generation-half preprocessor: invert a source video to ``SourceLatent``.

    Structural type only — the concrete implementation (Wan VAE inversion) lives
    in ``generation/``; nothing in e2w_core depends on a model.
    """

    def invert(self, video: Tensor) -> SourceLatent: ...
