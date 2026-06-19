import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CfVEditBenchmarkShapeTest(unittest.TestCase):
    def test_spec_is_saved_as_markdown(self):
        spec_path = ROOT / "CF_VEDIT_BENCHMARK_SPEC.md"
        self.assertTrue(spec_path.exists())
        text = spec_path.read_text(encoding="utf-8")
        self.assertIn("CF-VEdit Benchmark", text)
        self.assertIn("bench validate-manifest", text)

    def test_manifest_is_lightweight_and_points_to_external_assets(self):
        rows = read_jsonl(ROOT / "manifest.jsonl")
        self.assertEqual(len(rows), 12)

        forbidden_manifest_fields = {
            "converted_video",
            "expected_physical_effect",
            "expected_visible_outcome",
            "source_full_video",
            "source_metadata",
            "leakage_exclusion_evidence",
            "must_preserve",
            "vlm_judge_prompt",
        }

        provenance_rows = {
            row["sample_id"]: row
            for row in read_jsonl(ROOT / "annotations" / "provenance.jsonl")
        }

        for row in rows:
            self.assertFalse(forbidden_manifest_fields.intersection(row))
            self.assertEqual(row["split"], "test")
            self.assertEqual(row["identifiability"], "identifiable")
            self.assertIsNone(row["pair_id"])

            source_path = ROOT / row["source_video"]
            contract_path = ROOT / row["contract"]
            self.assertTrue(source_path.exists(), row["source_video"])
            self.assertTrue(contract_path.exists(), row["contract"])
            self.assertEqual(source_path.name, f"{row['sample_id']}.mp4")

            meta = row["video_meta"]
            self.assertGreater(meta["fps"], 0)
            self.assertGreater(meta["num_frames"], 0)
            self.assertGreater(meta["width"], 0)
            self.assertGreater(meta["height"], 0)

            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(contract["sample_id"], row["sample_id"])
            self.assertEqual(contract["operation"], row["operation"])
            self.assertEqual(contract["target_ref"], row["target_ref"])
            self.assertTrue(contract["counterfactual_state"])
            self.assertTrue(contract["affected_regions"])
            self.assertTrue(contract["preserve_regions"])

            provenance = provenance_rows[row["sample_id"]]
            self.assertTrue(provenance["leakage_checked"])
            self.assertFalse(provenance["leaked"])


class CfVEditCliTest(unittest.TestCase):
    run_name = "unittest_copy_source"

    def tearDown(self):
        shutil.rmtree(ROOT / "predictions" / self.run_name, ignore_errors=True)
        shutil.rmtree(ROOT / "results" / self.run_name, ignore_errors=True)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "bench.py", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_validate_manifest_cli(self):
        completed = self.run_cli("validate-manifest")
        self.assertIn("valid manifest rows: 12", completed.stdout)

    def test_copy_source_baseline_validates_scores_and_reports(self):
        subprocess.run(
            [
                sys.executable,
                "baselines/copy_source.py",
                "--run-name",
                self.run_name,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        validate = self.run_cli("validate", self.run_name)
        self.assertIn("valid predictions: 12", validate.stdout)

        score = self.run_cli("score", self.run_name, "--judge", "vlm")
        self.assertIn("wrote results", score.stdout)

        report = self.run_cli("report", self.run_name)
        self.assertIn("wrote summary", report.stdout)

        summary = json.loads(
            (ROOT / "results" / self.run_name / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["n"], 12)
        self.assertEqual(summary["missing"], 0)
        self.assertEqual(summary["failure_rate"], 0)
        self.assertEqual(summary["preservation_axis"], 1)
        self.assertEqual(summary["consequence_axis"], 0)
        self.assertEqual(summary["edit_success"], 0)
        self.assertIn("保不变量", summary)
        self.assertIn("命中后果", summary)


if __name__ == "__main__":
    unittest.main()
