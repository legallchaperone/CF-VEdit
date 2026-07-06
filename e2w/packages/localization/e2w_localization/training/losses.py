"""Stage-0 seg-branch mask loss: per-layer Dice + BCE.

Spec anchor: ``E2W-v0-Remove-Only-Spec.md`` §Stage 0 — seg_dir / seg_ind are each
supervised by a binary mask (direct GT / affected GT) with a Dice + BCE loss.
This module is pure (torch only), no model coupling, so it is unit-testable in
isolation and shared by both the seg_dir and seg_ind heads.

Design notes carried from TRAINING_NOTES.md:
- Dice is the region term (forgiving on the coarse/gridified affected labels);
  BCE is the per-pixel term. Weighted sum.
- ``ignore`` masks pixels/samples with unknown supervision (e.g. a
  ``void_vlm_weak`` row whose affected mask came back empty — do NOT train it as
  a confident negative). Callers pass ``ignore=None`` when every pixel is
  supervised (e.g. seg_dir on DAVIS GT).
- ``pos_weight`` counteracts the heavy background imbalance (masks are a few % of
  the frame — see dataset coverage stats).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskLossWeights:
    dice: float = 1.0
    bce: float = 1.0
    # Down-weight seg_ind vs seg_dir at the trainer level (clean DAVIS GT should
    # dominate the shared backbone; the affected GT is noisy). Applied by the
    # trainer when summing the two layers, not here.


def dice_loss(logits: torch.Tensor, target: torch.Tensor, *,
              ignore: torch.Tensor | None = None, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss on sigmoid(logits). Shapes broadcast over (..., H, W).

    ``target`` is {0,1} float; ``ignore`` is a bool mask (True = do not supervise).
    Returns a scalar mean over the batch.
    """
    prob = torch.sigmoid(logits)
    if ignore is not None:
        keep = (~ignore).to(prob.dtype)
        prob = prob * keep
        target = target * keep
    dims = tuple(range(1, prob.dim()))  # reduce over everything but batch
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor, *,
             ignore: torch.Tensor | None = None,
             pos_weight: torch.Tensor | float | None = None) -> torch.Tensor:
    """Per-pixel BCE-with-logits, masking out ``ignore`` pixels before reducing."""
    pw = None
    if pos_weight is not None:
        pw = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(
            float(pos_weight), device=logits.device, dtype=logits.dtype)
    per_px = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none")
    if ignore is not None:
        keep = (~ignore).to(per_px.dtype)
        denom = keep.sum().clamp_min(1.0)
        return (per_px * keep).sum() / denom
    return per_px.mean()


def mask_loss(logits: torch.Tensor, target: torch.Tensor, *,
              ignore: torch.Tensor | None = None,
              weights: MaskLossWeights = MaskLossWeights(),
              pos_weight: torch.Tensor | float | None = None) -> tuple[torch.Tensor, dict]:
    """Combined per-layer mask loss = w_dice * Dice + w_bce * BCE.

    Returns (scalar_loss, components) where components has detached floats for
    logging. ``target`` and ``ignore`` must broadcast to ``logits`` shape.
    """
    d = dice_loss(logits, target, ignore=ignore)
    b = bce_loss(logits, target, ignore=ignore, pos_weight=pos_weight)
    total = weights.dice * d + weights.bce * b
    return total, {"dice": float(d.detach()), "bce": float(b.detach()),
                   "total": float(total.detach())}


@torch.no_grad()
def mask_iou(logits: torch.Tensor, target: torch.Tensor, *,
             ignore: torch.Tensor | None = None, threshold: float = 0.5) -> float:
    """Held-out IoU of thresholded prediction vs target (the Stage-0 eval metric).

    ignore pixels are excluded from both intersection and union.
    """
    pred = (torch.sigmoid(logits) > threshold)
    tgt = target > 0.5
    if ignore is not None:
        valid = ~ignore
        pred = pred & valid
        tgt = tgt & valid
    inter = (pred & tgt).sum().float()
    union = (pred | tgt).sum().float()
    if union == 0:
        return 1.0  # both empty (e.g. a true-negative frame) → perfect
    return float(inter / union)
