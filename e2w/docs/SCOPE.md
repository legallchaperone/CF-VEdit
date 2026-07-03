# SCOPE — deferred & reserved scope

Things the proposal explicitly puts off. The point of this register: deferred
items neither sneak in half-built nor get forgotten. Each keeps a **placeholder
location** so the seam is ready when we pick it up. Nothing here is implemented
beyond its placeholder field.

| item | why deferred | placeholder location | status | source |
|---|---|---|---|---|
| **Rung-3 pair examples** (same instruction, two sources → different GT) | the sharpest proof we're testing counterfactuals, but expensive to collect | manifest `pair_id` field (currently always `null`); benchmark leaderboard notes it as reserved | ⏸ field reserved | proposal §4.2–4.3 |
| **Edit breadth: add / attribute / force_event** | v0 (ADR-0007) hard-scopes to remove-only, not "add/remove now, broaden later" — VOID's mask semantics and renderer are built around removal; unverified whether they transfer | `Operation` enum (`e2w_core.plan`) and manifest `operation` enum already include all four; no samples or training yet for anything but remove | ⏸ enum reserved | [E2W-v0-Remove-Only-Spec.md](../../E2W-v0-Remove-Only-Spec.md) §4; ADR-0007 |
| **VOID pass2 (deformation repair)** | v0 uses pass1 only; pass2 adds complexity not needed for the remove-only controlled comparison | none; would be a renderer config swap if picked up | ⏸ not used | v0 spec §1.5, §4 |
| **`seg` attends `edit` (cross-branch attention)** | semantically motivated (indirect-mask judgment relates to physical-consequence reasoning) but breaks Stage 0's gradient isolation (mask loss would leak into edit params via the shared channel); VOID's fully decoupled architecture already hits 0.83 on the physically-affected dimension without it | would be a change to `_build_query_attention_mask` in `query_tokens.py`; independent ablation, not v0 | ⏸ ablation only, post-Stage-2 | v0 spec §4 |
| **Gated mixing `α·edit_embed + (1-α)·text_embed` as default renderer conditioning** | hard replacement (edit tokens fully occupy the T5 text-condition slot) is the v0 default; α-mixing demoted to a fallback only if hard replacement proves unstable early in Stage 2 | `1.4` describes both paths; α path unimplemented until/unless triggered | ⏸ fallback only | v0 spec §1.4, §4 |
| **4D attention mask batched training** | current mask construction assumes `batch_size=1` | `query_tokens.py` `_build_query_attention_mask` | ⏸ known limitation | v0 spec §1.2 |
| **Soft/differentiable seg mask path** (skip thresholding, feed SAM2's probability map to the renderer directly so video loss reaches `seg_dir`/`seg_ind`) | frozen renderer only ever saw VOID's discrete quadmask values in training — continuous input is out-of-distribution risk; uncertain benefit vs. real generation-quality regression risk | none; would replace the quadmask construction in `1.3` | ⏸ not attempted in v0 | v0 spec §4 |
| **Cycle training (add↔remove self-supervision)** | not novel (Ouroboros/Paint-by-Inpaint); provably breaks on non-bijective interactions; only ever an auxiliary signal on lazy/reversible examples | none yet; would enter `data_engine` as an optional augmentation, not the main signal | ⏸ idea only | proposal §2.8 |
| **RL / preference alignment (P3)** | optional last layer; core training must not depend on it | future isolated `alignment` package; not in main dependency path | ⏸ not built | proposal §5, architecture §A.5 |
| **Semantic / social / biological counterfactuals** | the sim engine only covers the physics it models; out of coverage | documented limitation; not a benchmark category | 🚫 out of scope | proposal §3 |

When an item moves from reserved to in-progress, it graduates to a row in
[TRACEABILITY.md](TRACEABILITY.md) and (if it changes a contract) gets an ADR.
