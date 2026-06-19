# 05 — Acceptance tests, task order, and risks

> The checkable backbone. Each acceptance test corresponds to a row in
> [TRACEABILITY.md](../TRACEABILITY.md) — when it passes, flip the row to ✅. Build
> in the listed order; dependencies are noted.

## Per-component acceptance (the spec-as-test gates)

| # | component | acceptance test | TRACEABILITY row |
|---|---|---|---|
| A1 | e2w_core | mask enum/priority + Operation↔manifest + io_contract↔bench parity (pure unit tests, runnable now, no GPU) | contracts ◑→✅ |
| L1 | localization | direct-layer mask IoU vs sim GT above threshold on val | novelty ② (direct) |
| L2 | localization | **indirect-layer** mask IoU vs sim dependency graph (tracked research metric; expect lower) | novelty ② (indirect) |
| L3 | localization | referring-seg sanity unchanged (no base-skill regression) | — |
| G1 | generation | inversion round-trip: decode(invert(V)) ≈ V below threshold | novelty ① |
| G2 | generation | invariant loss bites: no-op edit → UNCHANGED latent ≈ source → preservation≈1 | novelty ③ |
| G3 | generation | gating: with a hand mask, only masked regions change; no seam artifacts | render seam |
| V0 | end-to-end (vanilla) | the **untrained** assembled structure runs end-to-end on the benchmark in vanilla mode; preservation decent (gating is architectural), consequence weak — the **floor** eval₁ must beat | pipeline / floor |
| E1 | end-to-end | E2W runs the benchmark via the adapter; two-axis metrics emitted | B1 / pipeline |
| E2 | end-to-end | beats `copy_source` (consequence>0 at high preservation) and isn't `free_regen` (preservation doesn't collapse) | benchmark anchors |
| E3 | end-to-end | beats Bernini on IP×CR **and** source-sensitivity (Rung-3 gate: same instruction, two sources → different output) | Rung-3 gate ⏸→active |

## Task order (dependency-aware)

The build sequence is a measured loop: **build structure → vanilla eval → collect
data → train → eval₁**, against an unchanging ruler.

**P0 (done):** benchmark + e2w_core contracts.

**Build the structure — no training** (target: V0): assemble all four blocks from
pretrained weights, runnable end-to-end in **vanilla mode** ([00], [04]).
1. **A1** — write `e2w_core` tests (no GPU; do this first, it locks the seam).
2. Vendor Sa2VA + VACE/Wan as pinned submodules; download weights ([README]).
3. **[02] block 1** — `WanAbductor` inversion (pretrained VAE; → G1).
4. **[02] block 4** — gated renderer *mechanism* (pretrained VACE masked
   inpainting; paste-back gating works zero-shot; the invariant **loss** comes with
   training, step 9).
5. **[01] base path + heads** — wire stock `[SEG]` for the zero-shot direct mask;
   add `[SEG_DIR]`/`[SEG_IND]`/`[EDIT]` + projection heads **initialized but
   untrained** (change A/B scaffolding).
6. **[04] vanilla-mode pipeline + adapter** — single-pass pipeline with vanilla
   routing (stock `[SEG]` direct mask + instruction→VACE text cond) + `e2w_adapter`
   writing `predictions/` (→ E1 path exists).

**Vanilla eval — the floor** (target: V0): run the untrained structure on the
benchmark ([00]); expect decent preservation, weak consequence. Record as the
floor. (Grow the eval set — the easy, annotation-only data half — in parallel.)

**Data — collect training data** (the hard half): reuse ladder in [03] (VOID
remove → Paint-by-Inpaint add → t2v augmentation → own Kubric for the
indirect/physics layer). Gates all training below.

**P1 — train the shared core on shallow DAG** (target: L1, G2, E2 on
attribute/add/remove):
7. **[03] thin** — a few hundred attribute/add/remove pairs with layer labels.
8. **[01] A+C+D** — three-layer mask training + CF dataset + config.
9. **[02]** — turn on the invariant-preservation loss (→ G2).
10. **[04] stage ① then ②** — align, then joint end-to-end on the thin set (where
    `[01]B` edit tokens finally get gradient and the direct layer is learned).

**P2 — causal closure depth** (target: L2, physics):
11. **[03] full** — scale the engine; add `force_event` temporal rollout.
12. retrain stage ② with temporal rollout; push the **indirect layer** (→ L2).
13. broaden eval; grow the benchmark toward eval scale (real annotated clips).

**Eval₁ — re-measure the lift**: re-run the *identical* vanilla-eval harness (same
manifest hash, judge, prompts) and report the delta over the floor + the Rung-3
source-sensitivity gate (→ E2, E3). Any harness change invalidates the comparison.

**P3 — alignment (optional):**
14. **[04] stage ③** — isolated RL/preference package (→ E3 polish).

Generation (steps 3–4) and localization (step 5) can proceed in parallel once A1
fixes the seam — that's the whole point of building the contracts first.

## Risks (carry these into every stage)

1. **Indirect/multi-hop mask = the 命门.** At inference the planner predicts the
   mask with no GT; under-predict → missed consequences, over-predict → over-edit.
   Capability comes only from [03]'s dependency-graph labels, not from Sa2VA's
   mechanism. This caps the model.
2. **Open-domain abduction is approximate.** Inversion won't perfectly reconstruct
   arbitrary real video; the invariant pin is only as good as the latent.
3. **Seam artifacts.** Mitigated by feather + joint denoise (no 2nd pass); watch
   boundaries qualitatively (G3).
4. **`edit_tokens` gradient coupling.** They don't learn in Sa2VA-only training —
   the halves **must** co-train in stage ②. Don't ship [01] expecting B to work
   standalone.
5. **sim-to-real gap (B5).** Train on sim, evaluate on real held-out; keep sources
   disjoint and report scope honestly.
6. **⚠️ upstream API drift.** VACE/Wan conditioning API and Sa2VA internals are
   the `⚠️ verify` points in [01]/[02] — confirm against the pinned commits before
   coding against them.

## Note for the executing agent

Start with **A1** (pure, no GPU) to make the contracts enforceable, then vendor +
download. Keep `third_party/` byte-clean (subclass, don't edit in place — B4).
Update [TRACEABILITY.md](../TRACEABILITY.md) as gates go green; any deviation from
the proposal needs an ADR (AGENTS.md).
