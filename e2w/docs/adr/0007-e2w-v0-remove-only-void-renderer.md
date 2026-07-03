# ADR-0007 — E2W v0 narrows to remove-only on a frozen VOID/CogVideoX-Fun renderer

- **Status:** Accepted
- **Date:** 2026-07-02
- **Anchors:** replaces proposal §2.6 (5-block model, abduction-as-latent-prior),
  architecture §A.1–A.5, Sa2VA-Modification-Plan §2–3; supersedes TRACEABILITY
  novelties ①③ as previously stated; narrows SCOPE's "edit breadth" row.

## Context

The root proposal (`Counterfactual-Video-Editing-Proposal.md`,
`CF-VEdit-Architecture-and-Narrative...md`) frames E2W around a Pearl Rung-3
abduction→intervention→closure→render pipeline: an MLLM inverts the source
video into a renderer VAE latent (the "U" prior), a gated DiT (VACE/Wan2.2)
pastes that latent back in unchanged regions each denoising step, and three
true novelties are claimed (abduction-as-latent-prior, indirect/multi-hop
mask, invariant-preservation loss bound to abduction).

`Sa2VA-Modification-Plan.md` details the localization-half changes (A: split
`[SEG]` into `[SEG_DIR]`/`[SEG_IND]`; B: add `[EDIT]` tokens) against that
VACE/Wan target.

Two problems surfaced while scoping toward a ~1-month, AAAI-class submission:

1. **Novelty over-claiming.** The "learnable query tokens couple an MLLM to a
   diffusion renderer" mechanism is now a well-populated design space
   (MetaQuery 2025, InstructX 2025, MetaCanvas) — not a standalone
   contribution. Holding the full 5-block abduction/Pearl framing as *the*
   novelty risks the paper's central claim being subsumed by concurrent work.
2. **Scope vs. timeline.** VACE/Wan2.2 integration, a from-scratch gated
   renderer, and full open-domain edit breadth (add/attribute/force_event) is
   not buildable, trained, and evaluated in ~1 month by one person.

VOID (Netflix/INSAIT, arXiv:2604.02296) already ships a trained,
license-checked-pending renderer (CogVideoX-Fun-V1.5-5b-InP,
`void_pass1.safetensors`) plus a quadmask input convention for exactly the
remove/object-interaction-deletion task, and the benchmark already has a VOID
baseline run (`results/void/`) to compare against directly.

## Decision

1. **Scope narrows to remove-only** for v0. Add/attribute/force_event stay
   reserved (already placeholder fields per SCOPE.md) but are explicitly out
   of v0, not "later this sprint."
2. **Renderer swaps from VACE/Wan2.2 to CogVideoX-Fun-V1.5-5b-InP initialized
   from `void_pass1.safetensors`, frozen, pass1 only** (no pass2 deformation
   repair). This is the single largest concrete change: `e2w_generation`'s
   existing VACE/Wan-targeted code (`renderer.py`, `abduction.py`, the
   `third_party/vace_wan` vendoring plan) is now stale relative to this ADR —
   the code has not yet been ported to CogVideoX-Fun; see Consequences.
3. **Abduction-as-source-inversion is dropped, not deferred.** There is no
   MLLM-inversion-to-renderer-latent step in v0. Unchanged-region conditioning
   comes from VOID's own mechanism: mask + masked-video-latent channel
   concatenation, reusing VOID's verified implementation as-is. TRACEABILITY
   novelty ① ("abduction = source inversion to latent as invariant prior") no
   longer describes what v0 builds.
4. **Invariant-preservation loss claim (novelty ③) is dropped**, not merely
   deferred to a later training stage — v0's renderer is frozen throughout
   (Stages 0–2), so there is no loss that could bind an invariant-preservation
   term to abduction. Unchanged-region fidelity is architectural (VOID's
   channel-concat gating), same status as the original claim intended, but
   achieved by reused mechanism rather than a novel loss.
5. **Novelty claim is renarrowed to three items** (see spec §0): (a) the
   physics-consequence-aware removal task itself, (b) a controlled comparison
   holding renderer + mask mechanism identical to VOID and varying only the
   conditioning source, (c) the seg/edit dual-branch — one shared planner
   driving both a segmentation head and a generation renderer, with the
   asymmetric-differentiability property made precise (edit branch is
   end-to-end differentiable through the frozen renderer; seg branch is not,
   because quadmask construction thresholds and is non-differentiable).
6. **`[SEG_DIR]`/`[SEG_IND]`/`[EDIT]` design in Sa2VA-Modification-Plan §1
   (changes A, B) is retained in spirit** but respecified precisely in the new
   doc: 6 fixed-position, non-vocabulary query tokens
   (`seg_dir, seg_ind, edit_0..3`), a custom 4D attention mask (edit tokens
   bidirectional among themselves, mutually masked from seg tokens), and tied
   RoPE position ids across `edit_0..3`. Sa2VA-Modification-Plan's changes C/D
   (dataset, training config) still apply in outline; its §2 (generation half
   = VACE/Wan) does not.
7. New canonical doc: `E2W-v0-Remove-Only-Spec.md` at repo root. It is the
   authoritative spec for the current build target. Root proposal/architecture/
   Sa2VA-plan docs keep their content (removed nothing — historical record,
   still the long-run P1–P3 thesis if the team returns to open-domain editing)
   but now carry a superseded banner pointing here.

## Consequences

- **+** Renderer risk collapses to near-zero: reusing a trained, benchmarked
  checkpoint instead of building/training a gated DiT from VACE/Wan means v0
  is buildable in ~1 month.
- **+** The controlled A/B against VOID (same renderer, same mask mechanism,
  only conditioning source varies) is a much cleaner experimental design than
  anything the open-domain framing offered — isolates exactly one variable.
- **+** Novelty claim is now precise and defensible against 2025–2026
  concurrent work (MetaQuery/InstructX/MetaCanvas already occupy the generic
  "query-token MLLM↔diffusion coupling" space).
- **−** `e2w_generation`'s existing code (renderer.py's VACE/Wan paste-back
  gating, `abduction.py`) targets the now-superseded architecture and needs a
  rework pass (new adapter for CogVideoX-Fun + `void_pass1` weights, drop the
  MLLM-inversion step) — tracked as a TRACEABILITY status change, not done by
  this ADR. Per repo discipline ("代码再徐徐图之" — code catches up slowly),
  this ADR intentionally lands docs first; TRACEABILITY rows referencing
  VACE/Wan or abduction latent are marked stale until the corresponding code
  PR lands.
- **−** The long-run open-domain Pearl/abduction thesis (P1–P3, non-remove
  operations) is not abandoned but is now explicitly deferred past v0 — if
  picked back up, it needs its own ADR reconciling it with whatever v0 code
  exists by then (e.g. does the CogVideoX-Fun renderer get kept for remove and
  a separate path added for other ops, or does the team return to VACE/Wan).
- **Open blocker (tracked, not resolved by this ADR):** VOID's
  `void_pass1.safetensors` license is unconfirmed. If it turns out
  incompatible with a submission, this ADR's renderer choice (Decision #2) is
  invalidated and needs revisiting before any Stage 0 training starts.
