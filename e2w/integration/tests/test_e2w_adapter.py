import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

E2W_ROOT = Path(__file__).resolve().parents[2]
for rel in ("packages/e2w_core", "packages/localization", "packages/generation", "."):
    path = E2W_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integration.adapters import e2w_adapter


class ModeConfigContractTest(unittest.TestCase):
    def test_full_mode_rejects_vanilla_config_before_writing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_root = Path(tmp)
            (benchmark_root / "manifest.jsonl").write_text("")

            with self.assertRaises(SystemExit) as cm:
                e2w_adapter.main([
                    "--full",
                    "--config", str(E2W_ROOT / "configs" / "vanilla.v0.json"),
                    "--benchmark-root", str(benchmark_root),
                    "--run-name", "bad_mode_config",
                ])

            self.assertEqual(cm.exception.code, 2)
            self.assertFalse((benchmark_root / "predictions" / "bad_mode_config").exists())


class RenderWorkerBatchTest(unittest.TestCase):
    def test_batch_worker_loads_pipeline_once_and_writes_each_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "render_results.jsonl"
            jobs = []
            for sid in ("a", "b"):
                mask = root / f"{sid}.npz"
                np.savez_compressed(
                    mask,
                    direct=np.zeros((1, 2, 2), dtype=bool),
                    indirect=np.zeros((1, 2, 2), dtype=bool),
                )
                jobs.append({
                    "sample_id": sid,
                    "source_path": f"{sid}.mp4",
                    "mask_npz": str(mask),
                    "instruction": "remove object",
                    "out_path": str(root / f"{sid}.mp4"),
                    "video": f"videos/{sid}.mp4",
                    "vanilla": True,
                    "edit_slots": 4,
                })
            batch = root / "batch.json"
            batch.write_text(e2w_adapter.json.dumps({
                "config": "fake_config.json",
                "results_path": str(results),
                "jobs": jobs,
            }))

            load_calls = []

            class FakeAbductor:
                def invert(self, source_path):
                    return ("source", source_path)

            class FakeRenderer:
                def render(self, source, edit_condition, mask, *, out_path):
                    Path(out_path).write_text(f"{source}:{edit_condition}")

            class FakePipeline:
                abductor = FakeAbductor()
                renderer = FakeRenderer()

            old_build = e2w_adapter.build_v0_pipeline
            try:
                e2w_adapter.build_v0_pipeline = lambda config: load_calls.append(config) or FakePipeline()
                self.assertEqual(e2w_adapter.render_worker(batch), 0)
            finally:
                e2w_adapter.build_v0_pipeline = old_build

            self.assertEqual(load_calls, ["fake_config.json"])
            rows = [e2w_adapter.json.loads(line) for line in results.read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
            self.assertTrue(all(row["status"] == "ok" for row in rows))


if __name__ == "__main__":
    unittest.main()
