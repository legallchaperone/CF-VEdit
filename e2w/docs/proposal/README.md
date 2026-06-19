# docs/proposal — proposal-as-truth

The four design notes are the **canonical source of truth**. The repo's shape is
derived from them; when design and code disagree, the design wins, and you change
the design here *first* (AGENTS.md, discipline rule 1).

These notes currently live at the **repository root** and are referenced (not
duplicated) from here to avoid two drifting copies in one repo — see
[ADR-0002](../adr/0002-scaffold-alongside-without-migration.md). When the
benchmark/monorepo is split into its own repo, they get materialized as copies.

| canonical doc | file | covers |
|---|---|---|
| proposal | [proposal](<../../../Counterfactual-Video-Editing-Proposal.md>) | research proposal: white space, unified abduction→do→closure→render framework, training, evaluation, P0–P3 |
| architecture | [architecture](<../../../CF-VEdit-Architecture-and-Narrative (给人看的）.md>) | the 4-block pipeline + Pearl narrative; the three true novelties (§A.7) |
| benchmark-spec | [benchmark-spec](<../../../CF-VEdit-Benchmark-Spec.md>) | executable spec for `cf_vedit_bench` (assets vs outputs, contracts, IO, metrics) |
| sa2va-plan | [sa2va-plan](<../../../Sa2VA-Modification-Plan.md>) | the localization half: concrete Sa2VA deltas (changes A–D) + honest risk |

Module READMEs back-link to specific sections of these notes (e.g. "implements
architecture §A.4"). Keep those anchors accurate — they are how
[TRACEABILITY.md](../TRACEABILITY.md) stays honest.
