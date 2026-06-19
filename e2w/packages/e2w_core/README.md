# e2w_core — shared contract layer (the seam)

Implements the "narrow waist" of [Repo-Design §2](../../docs/proposal/README.md).
This package is the **only** place the two halves of the model meet, and it
depends on no other internal package. Changing it should always trigger review.

## Modules

| module | what it fixes | proposal anchor |
|---|---|---|
| `masks.py` | three-layer spatiotemporal mask: `MaskLayer{DIRECT,INDIRECT,UNCHANGED}`, pixel-conflict priority (`直接>间接>不改变`), `ThreeLayerMask` shape | architecture §A.4 |
| `plan.py` | the planner's two outputs (region-query vectors + edit-plan tokens, **not** masks) and the `Operation` enum | architecture §A.2, proposal §2.6.2 |
| `latent.py` | `SourceLatent` = the inverted VAE latent = engineered exogenous **U** (invariant prior) | architecture §A.2【1】 / B.2 |
| `io_contract.py` | the `predictions/<run>/` disk shape the benchmark consumes (B1/B2 seam) | benchmark-spec §4 |

## Why this is its own package

- **B3** (localization ↔ generation): the two halves import only these types —
  three-layer mask + edit tokens + source latent. They must never import each
  other.
- **B1/B2** (benchmark ↔ model, assets ↔ outputs): `io_contract` is the single
  definition of the directory the benchmark reads; the benchmark never imports
  model code.

## Status

Contract **stubs** — types, enums, priorities, and the IO shape are real and
importable; behavior that needs an array/model backend (`ThreeLayerMask.unchanged`,
`Abductor.invert`) raises `NotImplementedError` and is implemented in
`generation/`. `resolve_pixel`, `Operation`, and `PredictionRow.to_json` are pure
and usable today.

```python
from e2w_core import MaskLayer, resolve_pixel, Operation, io_contract
resolve_pixel([MaskLayer.UNCHANGED, MaskLayer.DIRECT])  # -> MaskLayer.DIRECT
```
