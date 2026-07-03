import dataclasses
import importlib.util
import json
import shutil
import unittest
from pathlib import Path

from e2w_core import io_contract
from e2w_core.masks import PIXEL_PRIORITY, MaskLayer, ThreeLayerMask, resolve_pixel
from e2w_core.plan import EDIT_TOKEN_DIM, EditPlan, Intervention, Operation, validate_edit_tokens_shape


# tests/ -> e2w_core -> packages -> e2w -> repo root == parents[4]; if this file
# moves, update the depth. The parity tests require the benchmark beside e2w/.
REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_ROOT = REPO_ROOT / "physics_iq_for_simple_eval"


def setUpModule():
    if not BENCHMARK_ROOT.is_dir():
        raise AssertionError(
            f"benchmark not found at {BENCHMARK_ROOT}; the e2w_core parity tests "
            "must run inside the monorepo (path computed via parents[4])."
        )


def load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "cf_vedit_bench_for_contract_tests", BENCHMARK_ROOT / "bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MaskContractTest(unittest.TestCase):
    def test_mask_layer_wire_values_are_stable(self):
        self.assertEqual(
            {layer.name: layer.value for layer in MaskLayer},
            {
                "DIRECT": "direct",
                "INDIRECT": "indirect",
                "UNCHANGED": "unchanged",
            },
        )

    def test_pixel_priority_is_direct_then_indirect_then_unchanged(self):
        self.assertEqual(
            PIXEL_PRIORITY,
            (MaskLayer.DIRECT, MaskLayer.INDIRECT, MaskLayer.UNCHANGED),
        )
        self.assertIs(resolve_pixel([MaskLayer.UNCHANGED]), MaskLayer.UNCHANGED)
        self.assertIs(
            resolve_pixel([MaskLayer.UNCHANGED, MaskLayer.INDIRECT]),
            MaskLayer.INDIRECT,
        )
        self.assertIs(
            resolve_pixel([MaskLayer.UNCHANGED, MaskLayer.INDIRECT, MaskLayer.DIRECT]),
            MaskLayer.DIRECT,
        )
        with self.assertRaises(ValueError):
            resolve_pixel([])


class SeamPayloadContractTest(unittest.TestCase):
    """The seam must carry the Phase-2 planner payloads, not just the vanilla stubs.

    `region_query` / `edit_tokens` are typed `Any` (tensor at runtime), so this is a
    structural contract check — it pins that the dataclasses transport the fields the
    full A.1 planner emits, and that the vanilla `None` bypass stays valid.
    """

    def test_editplan_carries_region_query_and_edit_tokens(self):
        # Stand-in objects for the runtime tensors — identity must survive the seam.
        region_query = ("region_query", (4, 256))
        edit_tokens = ("edit_tokens", (4, 4096))
        plan = EditPlan(
            intervention=Intervention(
                operation=Operation.ADD, target_ref="red ball", instruction="Add a red ball."
            ),
            region_query=region_query,
            edit_tokens=edit_tokens,
        )
        self.assertIs(plan.region_query, region_query)
        self.assertIs(plan.edit_tokens, edit_tokens)
        self.assertEqual(plan.intervention.operation, Operation.ADD)

    def test_editplan_vanilla_bypass_keeps_none(self):
        plan = EditPlan(
            intervention=Intervention(
                operation=Operation.REMOVE, target_ref="ball", instruction="Remove the ball."
            ),
            region_query=None,
            edit_tokens=None,
        )
        self.assertIsNone(plan.region_query)
        self.assertIsNone(plan.edit_tokens)

    def test_threelayermask_carries_direct_and_indirect(self):
        direct = ("direct", (21, 480, 832))
        indirect = ("indirect", (21, 480, 832))
        mask = ThreeLayerMask(direct=direct, indirect=indirect)
        self.assertIs(mask.direct, direct)
        self.assertIs(mask.indirect, indirect)
        # unchanged() is a contract stub implemented in the generation half, not core.
        with self.assertRaises(NotImplementedError):
            mask.unchanged()


class EditTokenValidationTest(unittest.TestCase):
    """Guards the full-path failure mode: edit_tokens missing/malformed must raise,
    never silently fall back to text conditioning (the adapter + render worker both
    call this before reaching VACE)."""

    def test_valid_shape_passes(self):
        self.assertEqual(validate_edit_tokens_shape((4, EDIT_TOKEN_DIM), slots=4), (4, 4096))

    def test_wrong_slots_raises(self):
        with self.assertRaises(ValueError):
            validate_edit_tokens_shape((3, EDIT_TOKEN_DIM), slots=4)

    def test_wrong_dim_raises(self):
        with self.assertRaises(ValueError):
            validate_edit_tokens_shape((4, 2048), slots=4)

    def test_wrong_rank_raises(self):
        with self.assertRaises(ValueError):
            validate_edit_tokens_shape((4,), slots=4)


class OperationParityTest(unittest.TestCase):
    def test_operation_enum_matches_benchmark_schemas(self):
        operation_values = [operation.value for operation in Operation]
        manifest_schema = json.loads(
            (BENCHMARK_ROOT / "schemas" / "manifest.schema.json").read_text(encoding="utf-8")
        )
        contract_schema = json.loads(
            (BENCHMARK_ROOT / "schemas" / "contract.schema.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            set(operation_values), set(manifest_schema["properties"]["operation"]["enum"])
        )
        self.assertEqual(
            set(operation_values), set(contract_schema["properties"]["operation"]["enum"])
        )

    def test_live_manifest_operations_are_declared_by_core(self):
        declared = {operation.value for operation in Operation}
        manifest_ops = {
            json.loads(line)["operation"]
            for line in (BENCHMARK_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertLessEqual(manifest_ops, declared)

    def test_edit_plan_carries_vectors_not_masks(self):
        # backs the contract claim: the planner emits vectors, not masks.
        field_names = {field.name for field in dataclasses.fields(EditPlan)}
        self.assertEqual(field_names, {"intervention", "region_query", "edit_tokens"})
        self.assertFalse([name for name in field_names if "mask" in name])


class IoContractParityTest(unittest.TestCase):
    run_name = "e2w_core_contract_unittest"

    def setUp(self):
        self.bench = load_bench_module()
        self.run_dir = BENCHMARK_ROOT / "predictions" / self.run_name
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_constants_match_benchmark_validator(self):
        self.assertEqual(io_contract.BENCHMARK_VERSION, self.bench.BENCHMARK_VERSION)
        self.assertEqual(set(io_contract.RUN_META_REQUIRED), self.bench.RUN_META_REQUIRED)

    def test_prediction_row_uses_benchmark_video_path_shape(self):
        ok_row = io_contract.PredictionRow(
            "sample_001", io_contract.STATUS_OK, "videos/sample_001.mp4"
        )
        self.assertEqual(ok_row.expected_video(), "videos/sample_001.mp4")
        self.assertEqual(
            ok_row.to_json(),
            {"sample_id": "sample_001", "status": "ok", "video": "videos/sample_001.mp4"},
        )

        failed_row = io_contract.PredictionRow("sample_001", "failed", None, "model crashed")
        self.assertEqual(
            failed_row.to_json(),
            {"sample_id": "sample_001", "status": "failed", "video": None, "error": "model crashed"},
        )

    def _write_run(self, prediction_rows, num_samples):
        videos_dir = self.run_dir / io_contract.PREDICTIONS_VIDEO_DIR
        videos_dir.mkdir(parents=True, exist_ok=True)
        self.bench.write_jsonl(self.run_dir / io_contract.PREDICTIONS_INDEX, prediction_rows)
        self.bench.write_json(
            self.run_dir / io_contract.RUN_META,
            {
                "run_name": self.run_name,
                "model_name": "e2w_core_contract_test",
                "model_version": "test",
                "benchmark_version": io_contract.BENCHMARK_VERSION,
                "manifest_sha256": self.bench.manifest_sha256(),
                "command": "python -m unittest discover -s tests -v",
                "created_at": self.bench.utc_now(),
                "num_samples": num_samples,
            },
        )

    def test_benchmark_accepts_directory_written_with_core_contract_names(self):
        rows = self.bench.load_manifest(validate=True)
        prediction_rows = [
            io_contract.PredictionRow(
                sample_id=row["sample_id"],
                status="failed",
                video=None,
                error="intentional contract-test failure row",
            ).to_json()
            for row in rows
        ]
        self._write_run(prediction_rows, len(rows))

        manifest_rows, predictions, run_meta, run_dir = self.bench.validate_predictions(self.run_name)
        self.assertEqual(len(manifest_rows), len(rows))
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(run_meta["run_name"], self.run_name)
        self.assertEqual(run_dir, self.run_dir)

    def test_benchmark_rejects_ok_row_with_missing_video(self):
        # negative parity: an "ok" row whose video file was never written must be
        # rejected, so the predictions/ shape is locked (not just the happy path).
        rows = self.bench.load_manifest(validate=True)
        prediction_rows = []
        for index, row in enumerate(rows):
            if index == 0:
                prediction_rows.append(
                    io_contract.PredictionRow(
                        row["sample_id"],
                        io_contract.STATUS_OK,
                        f"{io_contract.PREDICTIONS_VIDEO_DIR}/{row['sample_id']}.mp4",
                    ).to_json()
                )
            else:
                prediction_rows.append(
                    io_contract.PredictionRow(row["sample_id"], "failed", None, "skip").to_json()
                )
        self._write_run(prediction_rows, len(rows))

        with self.assertRaises(self.bench.ValidationError):
            self.bench.validate_predictions(self.run_name)


if __name__ == "__main__":
    unittest.main()
