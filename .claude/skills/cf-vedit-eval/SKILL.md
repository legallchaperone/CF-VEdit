---
name: cf-vedit-eval
description: Run the CF-VEdit physics-iq benchmark (cf_vedit_bench) on any video-editing model — generate edited clips, package into predictions/, validate, score with a judge, and report. Use when asked to evaluate/benchmark a video-editing or counterfactual-editing model on CF-VEdit, or to score/compare runs. Examples of models already run this way: bernini, void.
---

# Running eval on the CF-VEdit physics-iq benchmark

The benchmark lives in `physics_iq_for_simple_eval/` (aka `cf_vedit_bench`). It is a
**file-based** evaluation: it never imports or calls a model. You run the model
*externally*, write its outputs to `predictions/<run>/`, and the benchmark consumes
directories. Everything below is model-agnostic; `bernini` (full generative v2v,
instruction-driven) and `void` (removal-only, mask-conditioned) are worked examples.

**Run every `bench.py` command from inside `physics_iq_for_simple_eval/`** — it resolves
all paths relative to itself. Pure stdlib; no pip install. The only external needs are
`ffmpeg` on PATH and a judge backend (below).

## Mental model (read once)

- 12 samples: **6 `remove` + 6 `add`** (see `python3 bench.py list`). Each has a
  `contracts/<id>.json` describing what must **change** (consequence) and **stay**
  (preservation), and a `videos/source/<id>.mp4`.
- **Two axes, never collapsed:** `preservation_axis` (保不变量) and `consequence_axis`
  (命中后果), plus `物理可信`/`编辑落地`/`质量`. Scoring gates on `target_success`: if the
  edit didn't land, consequence/physical are forced to 0.
- **Read-only vs run outputs is a hard boundary.** Inputs (`manifest.jsonl`,
  `contracts/`, `videos/source/`, `annotations/`, `judges/`, `schemas/`) must not be
  mutated. Model outputs go under `predictions/<run>/`; scores under `results/<run>/`.
- **`run_meta.json` is a reproducibility lock** — `validate` rejects a run whose
  `manifest_sha256` ≠ the live manifest or `benchmark_version` ≠ `BENCHMARK_VERSION`.
  Editing the manifest invalidates every prior run.

## Workflow

### 1. Generate edited videos (externally, per sample)

For each of the 12 samples, produce one edited clip from the source video +
`instruction`/`target_ref`/`operation` (from `manifest.jsonl` / the contract).

- **Respect capability limits — never fake an edit.** If the model can't do an
  operation (e.g. VOID does removal only), let that sample **fail** rather than emitting
  garbage. Failed samples get `status:"failed"`, `video:null`, and score 0. This is
  honest and shows up in `by_operation` (see step 5).
- **Match the source shape.** If the model changes resolution or frame count,
  post-process each output back to the source's `video_meta` (w×h, num_frames, fps) so
  preservation/temporal scoring is fair. `validate` only checks the file exists, but the
  judge compares source↔edited.
- Keep model working files **outside** the benchmark tree (e.g. a scratch/run dir).

### 2. Package into `predictions/<run>/`

Use the generic adapter — it writes `videos/<id>.mp4`, `predictions.jsonl`, and
`run_meta.json` for you, and marks a sample `ok` only if the command exits 0 **and**
the output file exists:

```bash
cd physics_iq_for_simple_eval
python3 tools/make_prediction_run.py \
  --run-name <run> --model-name <Model> --model-version <ver> \
  --cmd '<your-wrapper> {sample_id} {operation} {source} {instruction} {target_ref} {out}'
```

Placeholders (each substituted as one argv token): `{source} {out} {sample_id}
{instruction} {operation} {target_ref}`. Write a tiny wrapper that produces `{out}`
(or exits non-zero for unsupported ops). Two common shapes:

- **Precomputed outputs** (recommended when generation is slow/manual): generate all
  clips first into a run dir, then the wrapper just `cp`s the finished clip for supported
  ops and `exit`s non-zero otherwise. (This is how `void` was packaged.)
- **Live per-sample CLI**: the `--cmd` invokes the model directly per sample.

Then **document provenance** by patching `predictions/<run>/run_meta.json` `notes`
(it's a run output, safe to edit; extra keys are allowed as long as `RUN_META_REQUIRED`
stay intact). Record: what the model is, which operations it covers, any pre/post-
processing, and known per-sample limitations. Future readers rely on this.

### 3. Validate

```bash
python3 bench.py validate <run>     # 12 valid; prints ok / failed counts
```

Fix anything it rejects (missing run_meta fields, hash mismatch, name mismatches) before
scoring.

### 4. Score with a judge

```bash
python3 bench.py score <run> --judge vlm     # OpenRouter google/gemini-2.5-pro; needs OPENROUTER_API_KEY + ffmpeg
python3 bench.py score <run> --judge human   # built-in local web UI (stdlib), you label manually
python3 bench.py score <run> --judge vlm --judge-output <rows.jsonl>   # import any judge's rows
```

Both judges emit the shared per-sample schema (`target_success`, `preservation_success`,
`physical_effect_success`, `temporal_consistency`, `major_artifacts`, `overall_pass`,
`effect_hits`, `short_reason`). Choosing a judge:

- **Compare models under the SAME judge**, or the numbers aren't comparable. If an
  existing run was human-judged and you want parity, either human-judge yours too, or
  re-judge the old run with your judge (`bench.py agree` then makes them cross-checkable).
- **If OpenRouter is unreachable** (region/IP blocks are common), use the bundled
  `scripts/gemini_judge.py` — it reuses `judges/vlm_judge.py`'s frame-extraction/prompt/
  parse logic but calls Gemini's OpenAI-compat endpoint (works through a proxy), writing
  a raw JSONL you import via `--judge-output`. This keeps the read-only `judges/` dir
  untouched. It needs `GEMINI_API_KEY` (and honors `http(s)_proxy` env):
  ```bash
  python3 scripts/gemini_judge.py <run> --root <path-to>/physics_iq_for_simple_eval --out /tmp/<run>_raw.jsonl
  python3 bench.py score <run> --judge vlm --judge-output /tmp/<run>_raw.jsonl
  ```
- Deterministic baselines (`copy_source`/`free_regen`, via `baseline_type` in run_meta)
  score offline with fixed anchors — no key, no network.

### 5. Report and interpret

```bash
python3 bench.py report <run> --judge <vlm|human>   # writes results/<run>/summary.json + leaderboard.md
python3 bench.py agree <run>                          # human↔VLM accuracy + Cohen's κ (needs both per_sample files)
```

Reading `summary.json` correctly is the crux:

- **For a capability-limited model, the real number is `by_operation.<op>`, not the
  aggregate.** A removal-only model fails all 6 `add` samples by design, dragging the
  headline `failure_rate` to ~0.5 and the aggregate axes down. Report the
  `by_operation.remove` row as the model's actual performance and say the aggregate is
  dragged by out-of-scope operations.
- **Trust rank over absolute.** Two models under the same judge rank reliably; absolute
  pass-rates depend on judge leniency. If you have any run with both human and VLM
  labels, `agree` gives κ — a low κ means treat absolute pass-rates as uncertain and lean
  on the ranking.
- Back up a prior `summary.json` before re-reporting with a different `--judge` (report
  overwrites it; the raw `*_per_sample.jsonl` are preserved). Run `agree` **before** the
  final `report` so agreement folds into the summary.

## Worked examples

| aspect | `bernini` (full generative v2v) | `void` (removal-only) |
|---|---|---|
| step 1 generate | one instruction-driven v2v call per sample, native res | mask-conditioned: build a per-clip quadmask (SAM2 point-prompt → dilation quadmask), then run the model; only `remove` samples |
| capability | attempts all 12 | 6 `remove` only; 6 `add` → `status:failed` (wrapper exits non-zero for op≠remove) |
| post-process | native res kept | model emits fixed-size/length → resized+resampled back to source w×h / num_frames |
| judge | first human, later re-judged via `gemini_judge.py` for parity | `gemini_judge.py` (OpenRouter blocked on host) |
| report focus | full 12-sample table | `by_operation.remove` (aggregate dragged by the add half) |

Both are packaged identically (steps 2–5); they differ only in step 1 (how the edited
clips are produced) and which operations they cover. A new model slots in the same way:
implement step 1 for it, declare which operations it supports, and the rest is unchanged.

## Gotchas

- Run `bench.py` from inside `physics_iq_for_simple_eval/`.
- Don't mutate read-only inputs; if you must change `manifest.jsonl`, regenerate every
  run afterward (the sha256 lock invalidates them).
- `gemini_judge.py` falls back to the source's `num_frames` (default 21) for the edited
  clip when `ffprobe` is absent — fine if your outputs match source length; otherwise
  pass/point it at the true count so it doesn't sample a truncated view.
- Keep judge and prompts fixed across runs you intend to compare.
