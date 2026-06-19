# CF-VEdit Benchmark

This repository packages the 12 Physics-IQ simple-eval clips as a small
CF-VEdit benchmark. The benchmark is file based: it does not call editing
models. Any model can read `manifest.jsonl`, write edited videos under
`predictions/<run_name>/`, and then use `bench.py` for validation, scoring
imports, aggregation, and reports.

## Assets

- `manifest.jsonl`: lightweight sample index.
- `videos/source/<sample_id>.mp4`: source clips.
- `contracts/<sample_id>.json`: counterfactual contracts with preserve regions
  and required consequences.
- `annotations/provenance.jsonl`: source and leakage evidence.
- `judges/vlm_prompts.jsonl`: reproducible VLM judge prompts.
- `schemas/`: manifest and contract schemas.

Generated outputs are separated from assets:

- `predictions/<run_name>/`: model outputs, prediction status, and run metadata.
- `results/<run_name>/`: per-sample judge rows, summaries, agreement, and
  leaderboard output.

## Model Output Contract

Each run must write:

```text
predictions/<run_name>/
  videos/<sample_id>.mp4
  predictions.jsonl
  run_meta.json
```

`predictions.jsonl` must contain every `sample_id`. Failed samples stay in the
file with `status != "ok"` and `video: null`; they are counted in the failure
rate and receive zero scores.

`run_meta.json` must include `benchmark_version`, `manifest_sha256`,
`model_version`, and the command used to create the run.

## CLI

```bash
python bench.py validate-manifest
python bench.py list
python bench.py validate <run_name>
python bench.py score <run_name> --judge vlm     # OpenRouter / Gemini 2.5 Pro
python bench.py score <run_name> --judge human   # built-in local web UI
python bench.py report <run_name> --judge vlm    # aggregate VLM scores (default)
python bench.py report <run_name> --judge human  # aggregate human scores
python bench.py agree <run_name>                 # human vs VLM agreement
```

Pick the judge with `--judge`. Both judges write the same per-sample schema, so
you can run either (or both, then `agree`). `score --judge vlm` writes
`per_sample.jsonl` and `score --judge human` writes `human_per_sample.jsonl`;
`report --judge` selects which of the two to aggregate (it defaults to `vlm`, so
a human-only run must be reported with `report <run_name> --judge human`). The
chosen judge is recorded in `summary.json` and the leaderboard.

### VLM judge (OpenRouter / Gemini 2.5 Pro)

The only built-in VLM backend is OpenRouter with `google/gemini-2.5-pro`. You
must supply your own key:

```bash
export OPENROUTER_API_KEY=sk-or-...
python bench.py score <run_name> --judge vlm
# options: --judge-model <slug>  --frames <n>  --api-key <key>
```

Each video is sent as `--frames` evenly sampled JPEG frames (source first, then
edited), extracted with **ffmpeg** (must be on `PATH`). Results are written to
`results/<run>/per_sample.jsonl`; the raw model verdicts are kept in
`results/<run>/vlm_raw_judge.jsonl` and the backend/model/frame count are
recorded in `results/<run>/vlm_judge_meta.json`.

Deterministic baseline runs (`copy_source` / `free_regen`, detected via
`baseline_type` in `run_meta.json`) are scored offline with fixed anchor rows
and need no API key. Advanced users can bypass the built-in judge entirely with
`--judge-output <jsonl>`.

### Human judge (built-in web UI)

`score --judge human` starts a dependency-free local web app (Python stdlib
only) and opens it in your browser. Source and edited videos play side by side;
tick the preserve regions left unchanged and the counterfactual effects hit, set
the flags, and click **Save & next**. Labels stream to
`results/<run>/human_per_sample.jsonl` and resume where you left off.

## Metrics

Reports keep the required two-axis view:

- `preservation_axis` / `保不变量`: preserve regions remain unchanged.
- `consequence_axis` / `命中后果`: required affected regions and
  counterfactual-state effects are hit.

The report also includes `physical_effect`, `edit_success`, `quality`,
`failure_rate`, and splits by category, operation, and difficulty. It never
collapses the benchmark into a single total score.

## Baseline Smoke Test

```bash
python baselines/copy_source.py --run-name copy_source
python bench.py validate copy_source
python bench.py score copy_source --judge vlm
python bench.py report copy_source
```

Expected anchor shape: high preservation, low edit success, low consequence
score.
