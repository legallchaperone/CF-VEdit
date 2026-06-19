"""e2w_core — the frozen contract layer (the seam). Depends on nothing internal.

This is the "narrow waist" of the monorepo (Repo-Design §2). Two boundaries are
enforced through it:

* **B3** — the localization half and the generation half meet ONLY through these
  types: the three-layer mask (``masks``), the planner's two vectors (``plan``),
  and the abduction source latent (``latent``).
* **B1 / B2** — ``io_contract`` defines the ``predictions/`` directory shape the
  benchmark consumes; the benchmark never imports model code.

Because everything depends on it and it depends on nothing internal, any PR that
touches e2w_core should trip review. Keep it dependency-light (stdlib + typing).
"""
from e2w_core import io_contract
from e2w_core.latent import Abductor, SourceLatent
from e2w_core.masks import PIXEL_PRIORITY, MaskLayer, ThreeLayerMask, resolve_pixel
from e2w_core.plan import EditPlan, Intervention, Operation

__all__ = [
    "MaskLayer",
    "ThreeLayerMask",
    "PIXEL_PRIORITY",
    "resolve_pixel",
    "Operation",
    "Intervention",
    "EditPlan",
    "SourceLatent",
    "Abductor",
    "io_contract",
]
