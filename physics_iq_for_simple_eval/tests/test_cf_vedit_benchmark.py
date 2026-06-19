import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench  # noqa: E402  (import after sys.path is extended)


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


class SchemaValidatorTest(unittest.TestCase):
    def test_additional_properties_schema_is_enforced(self):
        # counterfactual_state declares additionalProperties: {"type": "string"};
        # a non-string value must be rejected, not silently accepted.
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        bench.validate_against_schema({"physical_effect": "ok"}, schema)
        with self.assertRaises(bench.ValidationError):
            bench.validate_against_schema({"physical_effect": 123}, schema)

    def test_additional_properties_false_rejects_unknown_keys(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        bench.validate_against_schema({"a": "x"}, schema)
        with self.assertRaises(bench.ValidationError):
            bench.validate_against_schema({"a": "x", "b": "y"}, schema)

    def test_additional_properties_true_allows_extras(self):
        schema = {"type": "object", "additionalProperties": True}
        bench.validate_against_schema({"anything": 1, "more": [2]}, schema)

    def test_contract_schema_rejects_non_string_counterfactual_value(self):
        schema = bench.read_json(ROOT / "schemas" / "contract.schema.json")
        contract = {
            "sample_id": "x_add",
            "operation": "add",
            "target_ref": "thing",
            "counterfactual_state": {"physical_effect": 123},
            "affected_regions": ["a"],
            "preserve_regions": ["b"],
        }
        with self.assertRaises(bench.ValidationError):
            bench.validate_against_schema(contract, schema, "contract")


class NormalizeGateTest(unittest.TestCase):
    contract = {"affected_regions": ["a", "b"], "counterfactual_state": {"x": "y"}}

    def test_failed_target_zeros_consequence_and_physical(self):
        row = bench.normalize_score_row(
            {
                "target_success": 0,
                "preservation_success": 1,
                "physical_effect_success": 1,
                "effect_hits": ["a", "b"],
                "overall_pass": 1,
            },
            "s_add",
            "human",
            "ok",
            self.contract,
        )
        self.assertEqual(row["target_success"], 0)
        self.assertEqual(row["physical_effect_success"], 0)
        self.assertEqual(row["effect_hits"], [])
        # preservation / overall_pass are NOT gated by target_success.
        self.assertEqual(row["preservation_success"], 1)
        self.assertEqual(row["overall_pass"], 1)

    def test_landed_target_keeps_consequence_and_physical(self):
        row = bench.normalize_score_row(
            {
                "target_success": 1,
                "physical_effect_success": 1,
                "effect_hits": ["a"],
            },
            "s_add",
            "human",
            "ok",
            self.contract,
        )
        self.assertEqual(row["physical_effect_success"], 1)
        self.assertEqual(row["effect_hits"], ["a"])


class CfVEditCliTest(unittest.TestCase):
    run_name = "unittest_copy_source"

    def tearDown(self):
        shutil.rmtree(ROOT / "predictions" / self.run_name, ignore_errors=True)
        shutil.rmtree(ROOT / "results" / self.run_name, ignore_errors=True)

    def make_baseline(self):
        subprocess.run(
            [sys.executable, "baselines/copy_source.py", "--run-name", self.run_name],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

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

    def test_human_only_run_reports_with_judge_human(self):
        self.make_baseline()
        sample_ids = [row["sample_id"] for row in read_jsonl(ROOT / "manifest.jsonl")]
        human_input = ROOT / "predictions" / self.run_name / "human_input.jsonl"
        with human_input.open("w", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                handle.write(json.dumps({
                    "sample_id": sample_id,
                    "target_success": 0,
                    "preservation_success": 1,
                    "effect_hits": [],
                    "physical_effect_success": 0,
                    "temporal_consistency": 1,
                    "major_artifacts": 0,
                    "overall_pass": 0,
                }) + "\n")

        # Reporting before any human scores must point at the right file.
        missing = subprocess.run(
            [sys.executable, "bench.py", "report", self.run_name, "--judge", "human"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("human_per_sample.jsonl", missing.stderr)

        self.run_cli("score", self.run_name, "--judge", "human", "--judge-output", str(human_input))
        self.assertTrue((ROOT / "results" / self.run_name / "human_per_sample.jsonl").exists())

        self.run_cli("report", self.run_name, "--judge", "human")
        summary = json.loads(
            (ROOT / "results" / self.run_name / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["judge"], "human")
        self.assertEqual(summary["n"], 12)
        self.assertEqual(summary["preservation_axis"], 1)
        self.assertEqual(summary["consequence_axis"], 0)

    def test_duplicate_prediction_row_rejected(self):
        self.make_baseline()
        predictions_path = ROOT / "predictions" / self.run_name / "predictions.jsonl"
        lines = predictions_path.read_text(encoding="utf-8").splitlines(keepends=True)
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(lines[0])  # duplicate the first sample_id row

        result = subprocess.run(
            [sys.executable, "bench.py", "validate", self.run_name],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate sample_id", result.stderr)


if __name__ == "__main__":
    unittest.main()
