# data_engine — simulated paired data + dependency-graph labels

A Kubric-style simulation engine that renders, from a **shared seed**, a
`factual` / `counterfactual` pair plus an **object-level causal dependency graph**
(proposal §3, architecture §A.3). The dependency graph is what teaches the hard
part — it is the supervision for the **indirect / multi-hop** mask layer that no
pretrained component knows ([novelty ②](../../docs/TRACEABILITY.md)).

## What one sample yields (the "three-piece set")

- source video **V** (no intervention),
- ground-truth **V\*** (same seed + `do(X=x)`),
- label map **M** (per-frame, from the sim log): direct / indirect / unchanged.

Because factual and counterfactual share the seed, non-descendants are
bit-identical → exact invariant labels and the cleanest invariant-loss signal.

## Boundaries

- **B5 — train/eval disjoint:** this engine is **dev/val only** and never appears
  in the report. Evaluation uses real held-out video. Keep the two strictly
  separate; record provenance.
- Depends on `e2w_core` only (emits `ThreeLayerMask`-shaped labels + `Operation`
  intervention metadata).

## Planned layout

```
data_engine/
  e2w_data_engine/   scene gen, shared-seed factual/CF render, dependency-graph → masks/E/I
  tests/
```

Scope note: the engine only covers the physics it models — semantic/social/
biological counterfactuals are out of coverage ([SCOPE.md](../../docs/SCOPE.md)).
