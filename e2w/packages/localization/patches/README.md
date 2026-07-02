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

When training stands up (stage ②), the training fork attaches **LoRA** to the
Sa2VA backbone and sets the overlay's non-vocab query-token `nn.Parameter`s
(`seg_dir_embed`/`seg_ind_embed`/`edit_embeds`) `requires_grad=True`. It does
**NOT** register tokens or resize embeddings — query tokens stay non-vocab per
spec §1.2, keeping the transformers≥4.51 embedding-resize hazard sidestepped.
These patches (dual-seg forward, `edit_hidden_fcs`) are the auditable source-level
record of that fork's mechanism; until it stands up this is a placeholder, not a
build step.
