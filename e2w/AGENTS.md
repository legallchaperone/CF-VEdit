# AGENTS.md — e2w repo constitution

One page for every collaborator (human or AI). If you read nothing else, read
this. The repo's shape *is* the proposal's shape; these rules make the proposal's
boundaries physically un-crossable rather than verbal.

## The five boundaries (B1–B5)

| # | boundary | rule | guarded by |
|---|---|---|---|
| **B1** | benchmark ↔ model | `cf_vedit_bench` only consumes the `predictions/` directory; it **never imports** localization/generation/integration | import-linter + spec-test |
| **B2** | data assets ↔ run outputs | `manifest/contracts/annotations/videos` are read-only; model outputs go to `predictions/`, scores to `results/` | dir layout + `bench validate` |
| **B3** | localization ↔ generation | the two halves **never import each other**; the only seam is `e2w_core` (three-layer mask + edit tokens + source latent) | import-linter |
| **B4** | upstream ↔ our deltas | `third_party/` (Sa2VA, VACE/Wan) is vendored and **not edited in place**; our changes live in `patches/` or new `e2w_*` files | CI: third_party has no diff |
| **B5** | train source ↔ eval source | sim engine A is dev/val only, never in the report; eval = real held-out; the two are strictly disjoint | `provenance.jsonl` + leakage check |

## Change discipline

1. **Truth source leads implementation.** Change `docs/proposal/` *before* you
   change code. Implementation must never quietly get ahead of the spec.
2. **Every deviation gets an ADR.** A decision that departs from the proposal is
   not mergeable without a `docs/adr/NNNN-*.md` (context / decision / consequences),
   back-linked from the code and TRACEABILITY.
3. **Every PR updates TRACEABILITY.** New capability → add its row (claim → module
   → test → status). A module with no proposal anchor is scope creep; a novelty
   with no module/test is a gap.
4. **Reserved scope stays a placeholder, not a half-build.** Items in `SCOPE.md`
   (cycle training, Rung-3 pair examples, attribute/force_event breadth) keep
   their reserved fields and do not get partially implemented.
5. **e2w_core changes trigger review.** It is the narrow waist; everything depends
   on it.

## Definition of Done (per PR)

Passes CI (import-linter boundaries + schema validate + spec-test + third_party
clean) **and** updates `TRACEABILITY.md` **and** (if it deviates) ships an ADR.
