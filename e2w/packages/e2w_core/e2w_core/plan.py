"""Causal Planner outputs — the two vector types fed downstream (NOT masks).

Implements: ``architecture.md §A.2`` 【2】 and ``proposal.md §2.6.2``. The planner
parses ``do(X=x)`` and emits two *different* things (do not conflate them):

* **region-query vectors** — sparse concept pointers → the mask decoder
  (``masks.py``); they say *which regions get hit*, they are NOT pixel-aligned.
* **edit-plan tokens** — continuous content condition → the gated renderer; they
  say *what the changed region should look like* (the ``[EDIT]`` path,
  Sa2VA-plan change B).

Pixel masks are decoded later — the LLM never emits a mask directly (its tokens
are sparse and not pixel-aligned). Both vector types are differentiable and
trained end-to-end (no RL).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# torch.Tensor at runtime; dependency-free contract alias.
Tensor = Any


class Operation(Enum):
    """Intervention type.

    MUST stay in lockstep with the benchmark manifest enum
    (``cf_vedit_bench`` manifest.schema.json and ``io_contract`` here).
    ``attribute`` / ``force_event`` are reserved breadth (see SCOPE.md).
    """

    ADD = "add"
    REMOVE = "remove"
    ATTRIBUTE = "attribute"
    FORCE_EVENT = "force_event"


EDIT_TOKEN_DIM = 4096


def validate_edit_tokens_shape(shape: Any, *, slots: int, dim: int = EDIT_TOKEN_DIM) -> tuple[int, int]:
    """Assert an ``edit_tokens`` array has shape ``(slots, dim)``; return it.

    Pure / dependency-free so the full-path failure mode ("edit_tokens missing or
    malformed must NOT silently fall back to text conditioning") is unit-testable
    without numpy/torch. Raises ``ValueError`` on mismatch.
    """
    got = tuple(int(x) for x in shape)
    want = (int(slots), int(dim))
    if got != want:
        raise ValueError(f"edit_tokens must have shape {want}, got {got}")
    return want


@dataclass(frozen=True)
class Intervention:
    """Parsed ``do(X=x)`` — the Action step of abduction→action→prediction."""

    operation: Operation
    target_ref: str
    instruction: str


@dataclass(frozen=True)
class EditPlan:
    """Planner output bundle handed across the seam.

    ``region_query``: sparse concept pointers, shape ``(Nq, Dq)`` — feed to the
    mask decoder. ``edit_tokens``: continuous tokens, shape ``(Nt, Dt)`` — feed
    to the renderer as the content condition for the changed region.
    """

    intervention: Intervention
    region_query: Tensor
    edit_tokens: Tensor
