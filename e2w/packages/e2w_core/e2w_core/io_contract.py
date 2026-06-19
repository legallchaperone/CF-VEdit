"""Model IO disk contract — the B1/B2 seam (benchmark ↔ model).

Implements: ``benchmark-spec.md §4``. The benchmark NEVER imports model code; it
consumes the ``predictions/<run_name>/`` directory. This module is the single
definition of that directory's shape — imported by the ``integration`` adapters
(producers) and mirrored by ``cf_vedit_bench`` (consumer). Keeping it here makes
the contract one source of truth instead of two drifting copies.

The directory shape::

    predictions/<run_name>/
        videos/<sample_id>.mp4     # edited clip; filename == sample_id
        predictions.jsonl          # one row per sample, failures included
        run_meta.json              # reproducibility lock (fields below)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Bump in lockstep with cf_vedit_bench BENCHMARK_VERSION; a run is only valid
# against the benchmark version it was produced for (benchmark-spec §4.2).
BENCHMARK_VERSION = "0.1.0"

PREDICTIONS_VIDEO_DIR = "videos"
PREDICTIONS_INDEX = "predictions.jsonl"
RUN_META = "run_meta.json"

# run_meta.json must carry all of these or `bench validate` rejects the run.
RUN_META_REQUIRED: tuple[str, ...] = (
    "run_name",
    "model_name",
    "model_version",
    "benchmark_version",
    "manifest_sha256",
    "command",
    "created_at",
    "num_samples",
)

STATUS_OK = "ok"


@dataclass
class PredictionRow:
    """One line of ``predictions.jsonl``.

    Failed samples stay in the file with ``status != "ok"`` and ``video=None``;
    they are counted in ``failure_rate`` and scored 0 (benchmark-spec §4.1).
    """

    sample_id: str
    status: str
    video: Optional[str] = None  # "videos/<sample_id>.mp4" when status == ok
    error: Optional[str] = None

    def expected_video(self) -> str:
        return f"{PREDICTIONS_VIDEO_DIR}/{self.sample_id}.mp4"

    def to_json(self) -> dict:
        row: dict = {"sample_id": self.sample_id, "status": self.status, "video": self.video}
        if self.error is not None:
            row["error"] = self.error
        return row
