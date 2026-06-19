"""Three-layer spatiotemporal causal mask — the geometry of the seam.

Implements: ``architecture.md §A.4`` (三层时空 mask) and ``proposal.md``
§"因果分层 mask:三层(直接/间接/不改变)" — a generalization of VOID's quadmask,
cut by causal distance from the intervention point.

This module is **contract only**. It fixes the layer enum, the pixel-conflict
priority, and the data shape that the *localization* half (producer) and the
*generation* half (consumer) agree on. No model logic lives here — this is the
B3 seam, so any change requires review.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

# np.ndarray at runtime; kept as an alias so the contract layer stays
# dependency-free (matches the benchmark's stdlib-only ethos).
Array = Any


class MaskLayer(Enum):
    """The three causal layers. Enum *values* are stable wire labels.

    VOID quadmask correspondence (proposal §"因果分层 mask"):

    ===========  =====================  ===============================
    layer        VOID quadmask          meaning
    ===========  =====================  ===============================
    DIRECT       ⊃ remove (0)           intervention node + 1-hop descendants
    INDIRECT     = affected (127)       multi-hop closure; global lighting /
                                        shadow / reflection live here
    UNCHANGED    = preserve (255)       non-descendants / invariants (the U)
    ===========  =====================  ===============================
    """

    DIRECT = "direct"
    INDIRECT = "indirect"
    UNCHANGED = "unchanged"


# Pixel-conflict priority. Labels are given at object/attribute level and
# *projected* to pixels; when several layers claim one pixel the higher-priority
# layer wins: 直接 > 间接 > 不改变 (architecture.md §A.4).
PIXEL_PRIORITY: tuple[MaskLayer, ...] = (
    MaskLayer.DIRECT,
    MaskLayer.INDIRECT,
    MaskLayer.UNCHANGED,
)


def resolve_pixel(layers: Iterable[MaskLayer]) -> MaskLayer:
    """Pick the winning layer for a pixel claimed by several (priority rule).

    Pure logic, no array backend — safe to rely on and unit-test directly.
    """
    present = set(layers)
    for layer in PIXEL_PRIORITY:
        if layer in present:
            return layer
    raise ValueError("resolve_pixel requires at least one MaskLayer")


@dataclass(frozen=True)
class ThreeLayerMask:
    """Per-frame label maps over time. The seam between the two halves.

    ``direct`` / ``indirect`` are boolean spatiotemporal stacks of shape
    ``(T, H, W)``. The ``UNCHANGED`` layer is implicit — the complement of
    ``direct ∪ indirect`` (Sa2VA-plan change A). The indirect layer activates
    *late* (the frame the consequence occurs), encoding cause-before-effect
    (architecture.md §A.4).

    Invariant: labels are object/attribute-level projections, **not**
    pixel-equality ground truth (proposal §"落实细则"). Identity is checked at
    object level downstream, so an UNCHANGED object's shadow may still move under
    a legal global effect.
    """

    direct: Array
    indirect: Array

    def unchanged(self) -> Array:
        """Invariant region = complement of ``direct ∪ indirect`` (the pinned U).

        Contract stub: implement against the chosen array backend (numpy/torch)
        in the generation half, not in e2w_core.
        """
        raise NotImplementedError(
            "contract stub: unchanged() is the complement of (direct ∪ indirect); "
            "implement on the array backend in generation/"
        )
