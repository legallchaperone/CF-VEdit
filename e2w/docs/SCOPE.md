# SCOPE — deferred & reserved scope

Things the proposal explicitly puts off. The point of this register: deferred
items neither sneak in half-built nor get forgotten. Each keeps a **placeholder
location** so the seam is ready when we pick it up. Nothing here is implemented
beyond its placeholder field.

| item | why deferred | placeholder location | status | source |
|---|---|---|---|---|
| **Rung-3 pair examples** (same instruction, two sources → different GT) | the sharpest proof we're testing counterfactuals, but expensive to collect | manifest `pair_id` field (currently always `null`); benchmark leaderboard notes it as reserved | ⏸ field reserved | proposal §4.2–4.3 |
| **Edit breadth: attribute / force_event** | start with add/remove; broaden later | `Operation` enum (`e2w_core.plan`) and manifest `operation` enum already include both; no samples or training yet | ⏸ enum reserved | proposal §4.3 |
| **Cycle training (add↔remove self-supervision)** | not novel (Ouroboros/Paint-by-Inpaint); provably breaks on non-bijective interactions; only ever an auxiliary signal on lazy/reversible examples | none yet; would enter `data_engine` as an optional augmentation, not the main signal | ⏸ idea only | proposal §2.8 |
| **RL / preference alignment (P3)** | optional last layer; core training must not depend on it | future isolated `alignment` package; not in main dependency path | ⏸ not built | proposal §5, architecture §A.5 |
| **Semantic / social / biological counterfactuals** | the sim engine only covers the physics it models; out of coverage | documented limitation; not a benchmark category | 🚫 out of scope | proposal §3 |

When an item moves from reserved to in-progress, it graduates to a row in
[TRACEABILITY.md](TRACEABILITY.md) and (if it changes a contract) gets an ADR.
