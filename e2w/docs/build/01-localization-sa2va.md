# 01 — Localization half: editing Sa2VA

> Builds the **Causal Planner + Mask decoder** (decides *which regions change*).
> Reuses Sa2VA's instruction→mask machinery; we add the three-layer split, the
> edit-token path, the CF dataset, and the config. Implements architecture
> §A.2【2】+【3】 and Sa2VA-plan changes A–D.
>
> **Emits across the seam:** `e2w_core.masks.ThreeLayerMask` (direct, indirect)
> and `e2w_core.plan.EditPlan` (region_query + edit_tokens). Consumes nothing
> from the generation half (B3).

## What Sa2VA already gives you (do not rebuild)

Read `projects/sa2va/models/sa2va.py` (`Sa2VAModel`) first. Confirmed structure
(Sa2VA-plan §0):

- `self.mllm` — Qwen2.5-VL-7B: understanding + emits `[SEG]` tokens.
- `self.grounding_encoder` — SAM2 (Hiera-L encoder + prompt encoder + mask
  decoder + memory for temporal propagation); default frozen, optionally unfreeze
  `sam_mask_decoder`.
- `self.text_hidden_fcs` — MLP projecting the LLM hidden at `[SEG]` positions to
  SAM2's prompt dim. **This is the planner→mask-decoder glue, already written.**

forward data flow: MLLM → `hidden_states[-1]`; `seg_token_mask = input_ids ==
seg_token_idx`; `pred_embeddings = text_hidden_fcs(hidden)[seg_token_mask]`;
`grounding_encoder.inject_language_embd(...)` (in `models/extension/sam2_base.py`)
appends it as a sparse prompt → `sam_mask_decoder` → mask; SAM2 memory propagates
across frames. Loss = `loss_mask` (BCE) + `loss_dice` + `llm_loss`.

So the query-vector → temporal-mask path is **free**. We add four deltas.

## Change A — single mask → three-layer causal mask

**Idea:** split the one `[SEG]` into `[SEG_DIR]` and `[SEG_IND]`; each runs the
existing SAM2 path to its own mask. `UNCHANGED` is the complement of the union
(computed in generation; here we only supervise direct + indirect).

Implementation:
- `__init__`: register `special_tokens=['[SEG_DIR]','[SEG_IND]']`; add to the
  tokenizer; resize MLLM embeddings; store `seg_dir_idx`, `seg_ind_idx`. ⚠️ verify
  how Sa2VA registers `seg_token_idx` and mirror it exactly.
- `forward`: build two masks from the two token ids, e.g. for each layer compute
  `pred_embeddings_<layer> = text_hidden_fcs(hidden)[mask_<layer>]` and call
  `inject_language_embd` per layer (batch the two if practical) → `pred_masks_dir`,
  `pred_masks_ind`, each a per-frame stack.
- Loss: GT masks carry a layer label (from the data engine); compute
  `loss_mask`/`loss_dice` per layer. Down-weight indirect (e.g. ×0.5) — it is
  noisier and often underdetermined (proposal §"落实细则").
- **Boundary output:** package `(pred_masks_dir, pred_masks_ind)` into
  `e2w_core.masks.ThreeLayerMask(direct=..., indirect=...)`. Keep labels at
  object/attribute level; the mask is their projection. Honor priority
  `直接>间接>不改变` when flattening for visualization/eval.

Keep edits in a subclass `E2WSa2VAModel(Sa2VAModel)` where possible; tokenizer
wiring may need a `patches/` overlay (B4).

## Change B — add edit-plan tokens (content condition for the renderer)

Sa2VA has no token describing *what the changed region should look like*. Add one.

- New special token `[EDIT]` (support N slots, e.g. 4–8). New projection head
  `self.edit_hidden_fcs` (mirror `text_hidden_fcs`) projecting `[EDIT]` hidden →
  the renderer's condition dim (Wan cross-attention dim). ⚠️ verify Wan condition
  width in [02].
- `forward`: collect `[EDIT]` hidden, project, return as `EditPlan.edit_tokens`.
  **Do not** feed these to SAM2.
- ⚠️ **Gradient caveat (critical):** `edit_hidden_fcs` gets **no gradient inside
  Sa2VA** — nothing supervises it here. It only learns once the renderer is
  attached and backprops its denoise loss (Sa2VA-plan §1.B). So this head is
  initialized in this stage but *trained jointly* with generation in stage ②
  ([04]). Don't expect it to do anything from Sa2VA-only training.

`region_query` in `EditPlan` is the sparse `pred_embeddings` used for masks; expose
it if generation wants it, but the load-bearing seam outputs are the mask + edit
tokens.

## Change C — CF dataset (sim pairs + causal-layer labels)

Sa2VA trains on referring-seg (RefCOCO/MeViS/ReVOS/Ref-SAV). Add ours.

- New `projects/sa2va/datasets/sa2va_data_cf.py`, modeled on
  `sa2va_data_03_refvos.py` / `sa2va_data_finetune.py`.
- Each sample yields `(video frames, instruction, direct-layer mask seq,
  indirect-layer mask seq)`. Layer labels come from the data engine's dependency
  graph ([03]).
- Reuse the existing `gt_masks` / `frames_per_batch` pipeline, extended to two
  layers. ⚠️ verify the exact collate/keys Sa2VA expects.
- The dataset is the **only** place the indirect layer's supervision enters — its
  label quality is the project's ceiling (the 命门). Keep loader output convertible
  to `e2w_core.masks.ThreeLayerMask`.

## Change D — training config

- Copy `projects/sa2va/configs/sa2va_qwen_finetune.py`. Change: our CF dataset,
  `special_tokens` (the three new tokens), `frozen_sam2_decoder=False` (unfreeze
  `sam_mask_decoder` so it can learn causal regions), per-layer loss weights.
- Entry (unchanged): `bash tools/dist.sh train <config> 8` (≥8×A100).
- Start weights: `ByteDance/Sa2VA-Qwen2_5-VL-7B` + `pretrained/sam2_hiera_large.pt`.

## Local interface (what the rest of E2W imports)

```python
# packages/localization/e2w_localization/planner.py  (to write)
from e2w_core.masks import ThreeLayerMask
from e2w_core.plan import EditPlan, Intervention

class CausalPlanner:
    """Wraps E2WSa2VAModel. Stateless w.r.t. the renderer (B3)."""
    def plan(self, frames, instruction: str, *, vanilla: bool = False
             ) -> tuple[ThreeLayerMask, EditPlan]:
        # vanilla=True (pre-training): use stock [SEG] for a direct-only mask,
        # indirect empty, [EDIT] head bypassed (see 00-vanilla-eval / 04).
        ...  # run MLLM + mask decoder; return the seam types
```

## Component acceptance (see [05])

- three-layer mask enum/priority round-trips through `e2w_core` (unit test);
- on sim val, **direct-layer** IoU vs GT high; **indirect-layer** IoU is the
  tracked research metric (expect lower; it's the hard layer);
- the model still produces valid single-frame masks on a referring-seg sanity set
  (didn't regress Sa2VA's base skill).

## Honest risk

Sa2VA hands over the *mechanism* (query→mask, temporal propagation, BCE/dice,
training scaffold) — the tedious, error-prone third. It does **not** give the
*capability* to know how far causation propagates: the indirect layer must be
taught by [03]'s dependency-graph labels. Mechanism free ≠ capability free
(Sa2VA-plan §3).
