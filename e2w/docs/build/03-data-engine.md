# 03 — Data engine: sim pairs + dependency-graph labels

> Builds the supervision for the hard part. A Kubric-style simulator renders, from
> a **shared seed**, a factual/counterfactual pair plus an object-level causal
> dependency graph, from which the three-layer masks and E/I labels are derived.
> Implements proposal §3 and architecture §A.3. **Dev/val only — never in the
> report (boundary B5).**
>
> **Produces:** the "three-piece set" (V, V\*, M) and labels convertible to
> `e2w_core.masks.ThreeLayerMask` + `e2w_core.plan.Operation`. Consumed by [01]'s
> CF dataset and by [02]'s invariant loss.

## Data sourcing — reality check (Kubric from scratch is hard)

Building a Kubric pipeline from zero is genuinely hard. Two things make it less
scary: the difficulty is **inverted** between eval and training, and you can
**reuse before you render**.

- **Eval data is the easy half.** The benchmark needs only *annotations* on real
  video (invariant set I + consequence list E) — no GT output video. That is what
  `physics_iq_for_simple_eval/` already is. Scaling eval is annotation, not
  simulation.
- **Training data is the hard half** — the only place you need exact masks + the
  dependency graph. So spend Kubric effort *only* where nothing else can give you
  the labels.

**Reuse ladder (do these before building your own engine):**

| source | gives you | for |
|---|---|---|
| **VOID** (Netflix, open, arXiv:2604.02296) | already-rendered Kubric+HUMOTO counterfactual pairs with causal (quad)masks | the **remove** subset — possibly reuse data/pipeline directly |
| **Paint-by-Inpaint** (2404.18212) | cheap add pairs via "delete-then-learn-to-add" (image-level) | the **add** direction |
| **t2v generation** | synthetic pairs | **training augmentation only** — never GT, never eval (proposal §3) |
| **your own Kubric** (below) | shared-seed pairs + dependency graph | the **indirect/multi-hop + physics** layer nothing else supplies (the 命门) |

So the recommended order is: reuse VOID for remove, Paint-by-Inpaint for add, get
the pipeline training end-to-end on those, and build the Kubric engine when you hit
the indirect/physics layer that needs the dependency graph. The rest of this doc
specs that engine — treat it as the *last* data step, not the first.

## Why this is the project's ceiling

Pretrained components give the *mechanism* to turn a query into a mask; none of
them know **how far causation propagates**. The indirect/multi-hop mask layer
(novelty ②) can only be learned from labels that encode the causal closure — and
those come from here. Label quality on the indirect layer ≈ the model's ceiling.
This is the 命门; invest here.

## What one sample yields (architecture §A.3, proposal §2.7)

For a scenario with a chosen intervention `do(X=x)`:

1. **V** — factual render (no intervention).
2. **V\*** — counterfactual render: **same random seed + initial conditions**, with
   the intervention applied. Shared seed ⇒ non-descendants are bit-identical ⇒
   exact invariants.
3. **M** — per-frame label maps from the sim's object-level log → causal layers:
   - **direct** = intervention node + 1-hop descendants;
   - **indirect** = multi-hop descendants (global lighting/shadow/reflection
     included); **activates late** (the frame the consequence occurs) — preserve
     this timing, it encodes cause-before-effect;
   - **unchanged** = everything else (non-descendants).
   - Also emit eval-style **E** (consequence list) and **I** (invariant list) so
     sim can cross-check the benchmark's annotation metrics.

Derive layers from the **dependency graph**, not from pixel diffs (diff masks are
only a fast self-check; they smear identity — proposal §"落实细则"). Apply the
pixel-conflict priority `直接>间接>不改变`.

## Build steps

- Wrap **Kubric** (`google-research/kubric`) in `packages/data_engine/e2w_data_engine/`.
- Per scenario: sample seed + scene; record an object-level causal log
  (contacts, forces, occlusions) → build the per-frame dependency graph → project
  to the three layers.
- **Interventions, phased:** start with `attribute`, `add`, `remove` (shallow DAG,
  P1). Add `force_event` / physics as a **temporal rollout** in P2 (the indirect
  layer fires across frames).
- **Output format** must (a) load cleanly into [01]'s `sa2va_data_cf.py`, and
  (b) be convertible to `e2w_core.masks.ThreeLayerMask`. Keep a documented on-disk
  schema (frames + per-layer mask stacks + intervention metadata + E/I).

```python
# packages/data_engine/e2w_data_engine/sample.py  (to write)
from e2w_core.masks import ThreeLayerMask
from e2w_core.plan import Operation

class SimSample:
    factual_frames: ...      # V
    counterfactual_frames: ...  # V*
    masks: ThreeLayerMask    # M (direct/indirect; unchanged is the complement)
    operation: Operation
    consequences: list[str]  # E
    invariants: list[str]    # I
```

## Boundaries

- **B5 (train/eval disjoint):** this engine is dev/val only. The benchmark
  evaluates on real held-out video (Physics-IQ). Never let sim assets/scripts/
  seeds used here leak into the eval set; record provenance.
- Depends on `e2w_core` only.

## Scope

The engine only covers the physics it models — semantic/social/biological
counterfactuals are out of coverage ([SCOPE.md](../SCOPE.md)). t2v-generated pairs
are **training augmentation only**, never ground truth (proposal §3).

## Component acceptance (see [05])

- shared-seed invariance: on a rendered pair, non-descendant pixels are identical
  (within codec tolerance) → confirms the invariant signal is clean;
- the derived indirect layer matches the dependency graph by construction
  (round-trip check);
- a thin P1 slice (a few hundred attribute/add/remove pairs) loads through [01]'s
  dataset and trains without shape errors.
