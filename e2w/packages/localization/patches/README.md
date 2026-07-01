# localization/patches/ — Sa2VA deltas (auditability, **unapplied**)

This directory holds the source-level diff of the Sa2VA changes (changes A/B from
`Sa2VA-Modification-Plan.md`: `[SEG_DIR]/[SEG_IND]/[EDIT]` tokens, dual-seg forward,
`edit_hidden_fcs`) **for the eventual training fork** — kept here so the deltas are
auditable separately from vendored upstream.

**The V0 runtime does NOT apply these patches.** Per ADR-0004, the runtime Sa2VA
code is the checkpoint snapshot (loaded via `trust_remote_code`), not
`third_party/sa2va` — so the heads are composed onto the *loaded instance* at
runtime by `e2w_localization/overlay.py` (held `nn.Parameter` tokens, vocab
untouched). That keeps `third_party/` byte-clean (B4) and the vanilla `[SEG]` path
byte-identical, and sidesteps the transformers≥4.51 embedding-resize hazard.

When training stands up (stage ②), the model that owns real token
registration/resizing materializes from these patches against a pinned Sa2VA
commit; until then this is a placeholder, not a build step.
