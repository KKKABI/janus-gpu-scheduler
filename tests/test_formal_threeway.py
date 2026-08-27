from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
FORMAL = REPO / "experiments" / "formal_threeway_20260827"
sys.path[:0] = [str(FORMAL), str(REPO / "experiments"), str(REPO)]

import common
from build_ncu_median_cache import REPEAT_COUNT, merge_repeated_caches
from run_threeway_latency import (
    FORMAL_REPEATS,
    formal_latency_mean,
    policy_environment,
    validate_result,
    write_cross_policy_audits,
)
from select_same_ready_pairs import group_class, paired_subset_filter, unique_ready_map


class FormalThreewayTests(unittest.TestCase):
    @staticmethod
    def _ncu_repeat(metric_value: float) -> dict:
        identity = {
            "requested_model": "GoogLeNet",
            "model_class": "GoogLeNet",
            "input_shapes": [[1, 3, 224, 224]],
            "input_dtypes": ["torch.float32"],
            "device_name": "NVIDIA RTX A5000",
            "device_capability": [8, 6],
            "torch_version": "2.4.0+cu124",
            "cuda_version": "12.4",
            "cudnn_version": 90100,
            "capture_backend": "dynamo_explain",
            "fx_code_sha256": "fx",
            "fx_node_names": ["x"],
            "profile_path": "/repo/profile.json",
            "profile_sha256": "profile",
            "git_head": "commit",
            "correctness": {"ok": True},
        }
        return {
            "schema_version": 2,
            "identity": identity,
            "kernels": [
                {
                    "launch_id": 1,
                    "op_name": "x",
                    "name": "kernel",
                    "grid_size": 4,
                    "block_size": 32,
                    "metrics": {
                        "mem_thru": metric_value,
                        "dram_thru": metric_value,
                        "l2_thru": metric_value,
                        "comp_thru": metric_value,
                        "dur_ns": metric_value,
                    },
                }
            ],
        }

    def test_three_repeat_ncu_cache_uses_exact_launch_median(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat, value in enumerate((1.0, 9.0, 3.0)):
                path = root / f"repeat_{repeat}.json"
                path.write_text(
                    json.dumps(self._ncu_repeat(value)), encoding="utf-8"
                )
                paths.append(path)
            output = root / "median.json"
            report = merge_repeated_caches(
                model="GoogLeNet", cache_paths=paths, output_path=output
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["repeat_count"], 3)
            self.assertEqual(
                payload["aggregation"]["method"],
                "identity-checked per-launch median",
            )
            self.assertEqual(payload["kernels"][0]["metrics"]["dram_thru"], 3.0)

    def test_three_repeat_ncu_cache_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [self._ncu_repeat(value) for value in (1.0, 2.0, 3.0)]
            payloads[2]["identity"]["profile_sha256"] = "different"
            paths = []
            for repeat, payload in enumerate(payloads):
                path = root / f"repeat_{repeat}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                merge_repeated_caches(
                    model="GoogLeNet",
                    cache_paths=paths,
                    output_path=root / "median.json",
                )

    def test_yolo_is_backbone_identity_in_frozen_config(self):
        config = json.loads(
            (REPO / "experiments" / "repro_config.json").read_text(
                encoding="utf-8"
            )
        )
        yolo = config["models"]["YOLOv8x"]
        self.assertEqual(yolo["capture_backend"], "dynamo_explain")
        self.assertTrue(yolo["profile_file"].startswith("BackboneWrapper_"))
        self.assertNotIn(
            "DetectionModel_torch.Size([1, 3, 320, 320]).pt.trace.json",
            (REPO / "experiments" / "profile_manifest.sha256").read_text(
                encoding="utf-8"
            ),
        )
        experiment_readme = (REPO / "experiments" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("YOLOv8x BackboneWrapper", experiment_readme)
        self.assertNotIn("corrected `YOLOv8x` task captures the complete", experiment_readme)

    def test_policy_environment_is_fail_closed_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ncu = root / "ncu"
            empty = root / "empty"
            solo = root / "solo"
            for path in (ncu, empty, solo):
                path.mkdir()
            janus = policy_environment(
                "janus", ncu_cache=ncu, empty_cache=empty, solo_root=solo
            )
            drt = policy_environment(
                "newtd_drt", ncu_cache=ncu, empty_cache=empty, solo_root=solo
            )
            ncu_env = policy_environment(
                "newtd_ncu_drt",
                ncu_cache=ncu,
                empty_cache=empty,
                solo_root=solo,
            )
            self.assertEqual(Path(janus["JANUS_NCU_CACHE_DIR"]), empty)
            self.assertNotIn("JANUS_NEW_TD_PAIR_EXTENSION", janus)
            self.assertEqual(drt["JANUS_NEW_TD_FINAL_SELECTOR"], "strategy")
            self.assertEqual(drt["JANUS_NEW_TD_MIN_OVERLAP_US"], "2.0")
            self.assertEqual(Path(drt["JANUS_NCU_CACHE_DIR"]), empty)
            self.assertEqual(
                ncu_env["JANUS_NEW_TD_FINAL_SELECTOR"],
                "risk_adjusted_interference",
            )
            self.assertEqual(ncu_env["JANUS_REQUIRE_VALID_NCU"], "1")
            self.assertNotIn("JANUS_ALLOW_LEGACY_NCU", ncu_env)

    def test_exact_ordered_ready_and_unique_rule(self):
        result = {
            "scheduler": {
                "calls": [
                    {"ready_ops": ["a", "b"], "selected_resource": ["a"]},
                    {"ready_ops": ["b", "a"], "selected_resource": ["b"]},
                    {"ready_ops": ["a", "b"], "selected_resource": ["b"]},
                ]
            }
        }
        mapped = unique_ready_map(result)
        self.assertNotIn(("a", "b"), mapped)
        self.assertIn(("b", "a"), mapped)

    def test_resource_classes_and_paired_subset_filter(self):
        self.assertEqual(group_class(["Conv", "GEMM"]), "pure_compute")
        self.assertEqual(group_class(["Pool", "Elementwise"]), "pure_memory")
        self.assertEqual(group_class(["Conv", "Pool"]), "mixed_resource")
        rows = [
            {
                "raw_pair_id": "small",
                "comparison": "x",
                "model": "m",
                "left_group": ["a"],
                "right_group": ["b"],
            },
            {
                "raw_pair_id": "large",
                "comparison": "x",
                "model": "m",
                "left_group": ["a", "c"],
                "right_group": ["b", "d"],
            },
        ]
        kept, removed = paired_subset_filter(rows)
        self.assertEqual([row["raw_pair_id"] for row in kept], ["large"])
        self.assertEqual([row["raw_pair_id"] for row in removed], ["small"])

    def test_frozen_model_and_policy_sets(self):
        self.assertEqual(len(common.MODELS), 7)
        self.assertEqual(len(common.POLICIES), 3)
        self.assertEqual(FORMAL_REPEATS, 10)
        self.assertEqual(REPEAT_COUNT, 3)
        self.assertEqual(len(common.MODELS) * REPEAT_COUNT, 21)
        self.assertEqual(
            len(common.MODELS) * len(common.POLICIES) * FORMAL_REPEATS,
            210,
        )
        self.assertEqual(common.MODEL_CLASSES["YOLOv8x"], "BackboneWrapper")
        stage_a = (
            FORMAL / "run_stage_a_profiles.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--repeats 3", stage_a)
        self.assertNotIn("REUSE_NCU", stage_a)

    def test_formal_latency_is_one_arithmetic_mean(self):
        values = [float(index) for index in range(1, 11)]
        self.assertEqual(formal_latency_mean(values), 5.5)
        with self.assertRaisesRegex(ValueError, "requires 10"):
            formal_latency_mean(values[:-1])

    def test_formal_ncu_result_is_fail_closed_on_mapping_coverage(self):
        payload = {
            "status": "completed",
            "task": {"model": "GoogLeNet"},
            "correctness": {"ok": True},
            "effective_parameters": {
                "max_ready": 6,
                "selection_mode": "static_interference",
                "time_domain": True,
                "final_selector": "risk_adjusted_interference",
            },
            "scheduler": {"summary": {"max_ready": 6}},
            "new_td_admission": {
                "mode": "static_union_frozen_td_pair_v1",
                "minimum_predicted_overlap_us": 2.0,
                "launch_gap_ms": 0.004096,
            },
            "ncu_report": {
                "experimental_valid": True,
                "cache_sha256": "cache",
                "duration_coverage": 0.49,
                "aggregation": {
                    "method": "identity-checked per-launch median",
                    "repeat_count": 3,
                },
            },
            "timing": {"statistics": {"count": 3}},
            "model_spec": {"timed_iterations": 3},
        }
        with self.assertRaisesRegex(RuntimeError, "duration coverage"):
            validate_result(
                payload,
                model="GoogLeNet",
                policy="newtd_ncu_drt",
                expected_cache_hashes={"GoogLeNet": "cache"},
            )

    def test_formal_ncu_result_rejects_old_single_cache(self):
        payload = {
            "status": "completed",
            "task": {"model": "GoogLeNet"},
            "correctness": {"ok": True},
            "effective_parameters": {
                "max_ready": 6,
                "selection_mode": "static_interference",
                "time_domain": True,
                "final_selector": "risk_adjusted_interference",
            },
            "scheduler": {"summary": {"max_ready": 6}},
            "new_td_admission": {
                "mode": "static_union_frozen_td_pair_v1",
                "minimum_predicted_overlap_us": 2.0,
                "launch_gap_ms": 0.004096,
            },
            "ncu_report": {
                "experimental_valid": True,
                "cache_sha256": "old-cache",
                "duration_coverage": 1.0,
            },
            "timing": {"statistics": {"count": 3}},
            "model_spec": {"timed_iterations": 3},
        }
        with self.assertRaisesRegex(RuntimeError, "three-repeat median"):
            validate_result(
                payload,
                model="GoogLeNet",
                policy="newtd_ncu_drt",
                expected_cache_hashes={"GoogLeNet": "old-cache"},
            )

    def test_cross_policy_audit_is_written_beside_all_three_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "trial": 0,
                "model": "GoogLeNet",
                "all_policy_inputs_byte_identical": True,
                "output_max_absolute_difference": {
                    policy: 0.0 for policy in common.POLICIES
                },
                "comparisons": [{"changed_selection": True}],
            }
            self.assertEqual(write_cross_policy_audits(root, [row]), 3)
            for policy in common.POLICIES:
                path = (
                    root
                    / "tasks"
                    / "trial_00"
                    / "googlenet"
                    / policy
                    / "cross_policy_audit.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["current_policy"], policy)
                self.assertTrue(payload["all_policy_inputs_byte_identical"])


if __name__ == "__main__":
    unittest.main()
