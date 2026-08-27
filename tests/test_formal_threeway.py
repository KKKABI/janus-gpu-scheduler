from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
FORMAL = REPO / "experiments" / "formal_threeway_20260827"
sys.path[:0] = [str(FORMAL), str(REPO / "experiments"), str(REPO)]

import common
from aggregate_same_ready import completion_status, split_resource_pair_rows
from build_ncu_v2_cache import REQUIRED_METRICS, incomplete_launch_ids
from build_ncu_median_cache import REPEAT_COUNT, merge_repeated_caches
from profile_group_resources import prepare_target_interpreter
from run_threeway_latency import (
    FORMAL_REPEATS,
    formal_latency_mean,
    policy_environment,
    validate_model_reference_identities,
    validate_result,
    write_cross_policy_audits,
)
from select_same_ready_pairs import (
    SAME_CLASS_QUOTAS,
    group_class,
    kernel_family,
    load_profiles,
    paired_subset_filter,
    select_with_frozen_quotas,
    unique_ready_map,
    validate_stage_b_provenance,
)
from verify_stage_entrypoints import SCRIPT_NAMES


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

    def test_single_ncu_cache_requires_memory_and_duration_metrics(self):
        launch = {
            "launch_id": 1,
            "grid_size": 1,
            "block_size": 32,
            "metrics": {name: 1.0 for name in REQUIRED_METRICS},
        }
        self.assertEqual(incomplete_launch_ids([launch]), [])
        for field in ("mem_thru", "dur_ns"):
            incomplete = json.loads(json.dumps(launch))
            incomplete["metrics"].pop(field)
            self.assertEqual(incomplete_launch_ids([incomplete]), [1])

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

    def test_unknown_kernel_is_not_silently_memory_classified(self):
        self.assertEqual(kernel_family("brand_new_kernel_xyz"), "Unclassified")
        self.assertEqual(kernel_family("ampere_sgemm_128x64"), "GEMM")
        self.assertEqual(group_class(["Unclassified"]), "unclassified")

    def test_profile_classification_accounting_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = {
                "schema_version": 2,
                "identity": {
                    "model_class": "GoogLeNet",
                    "profile_sha256": "profile",
                    "fx_code_sha256": "fx",
                },
                "aggregation": {
                    "method": "identity-checked per-launch median",
                    "repeat_count": 3,
                },
                "kernels": [
                    {
                        "launch_id": 1,
                        "op_name": "known_op",
                        "name": "ampere_sgemm",
                        "grid_size": 1,
                        "block_size": 32,
                        "metrics": {"dur_ns": 30.0},
                    },
                    {
                        "launch_id": 2,
                        "op_name": "unknown_op",
                        "name": "brand_new_kernel_xyz",
                        "grid_size": 1,
                        "block_size": 32,
                        "metrics": {"dur_ns": 10.0},
                    },
                ],
            }
            (root / "GoogLeNet.ncu.v2.json").write_text(
                json.dumps(cache), encoding="utf-8"
            )
            with mock.patch("select_same_ready_pairs.MODELS", ("GoogLeNet",)):
                profiles, sources = load_profiles(root)
            audit = sources[0]["classification"]
            self.assertEqual(
                audit["kernel_launches"],
                audit["classified_kernel_launches"]
                + audit["unclassified_kernel_launches"],
            )
            self.assertEqual(
                audit["total_duration_ns"],
                audit["classified_duration_ns"]
                + audit["unclassified_duration_ns"],
            )
            self.assertAlmostEqual(audit["duration_classification_coverage"], 0.75)
            self.assertEqual(
                profiles[("GoogLeNet", "unknown_op")]["family"],
                "Unclassified",
            )

    @staticmethod
    def _quota_row(index: int, resource_class: str) -> dict:
        return {
            "raw_pair_id": f"row_{resource_class}_{index:02d}",
            "comparison": "newtd_drt_vs_newtd_ncu",
            "comparison_role": "primary",
            "model": "GoogLeNet",
            "same_resource_class": True,
            "paired_resource_class": resource_class,
            "left_group_metadata": {
                "profiled_duration_us": float(index + 1),
                "work_items_proxy": float(index + 1),
            },
            "right_group_metadata": {
                "profiled_duration_us": float(index + 2),
                "work_items_proxy": float(index + 2),
            },
        }

    def test_frozen_3712_quotas_are_bounded_and_order_independent(self):
        rows = []
        for resource_class, available in (
            ("pure_compute", 5),
            ("pure_memory", 9),
            ("mixed_resource", 15),
        ):
            rows.extend(
                self._quota_row(index, resource_class)
                for index in range(available)
            )
        first, rejected, audit = select_with_frozen_quotas(rows)
        shuffled = list(rows)
        random.Random(42).shuffle(shuffled)
        second, _, _ = select_with_frozen_quotas(shuffled)
        self.assertEqual(
            [row["raw_pair_id"] for row in first],
            [row["raw_pair_id"] for row in second],
        )
        selected_counts = {
            resource_class: sum(
                row["paired_resource_class"] == resource_class for row in first
            )
            for resource_class in SAME_CLASS_QUOTAS
        }
        self.assertEqual(selected_counts, SAME_CLASS_QUOTAS)
        self.assertEqual(len(first), 22)
        self.assertEqual(len(rejected), 7)
        primary_audit = [
            row for row in audit
            if row["comparison"] == "newtd_drt_vs_newtd_ncu"
            and row["bucket"] in SAME_CLASS_QUOTAS
        ]
        self.assertEqual(sum(row["selected"] for row in primary_audit), 22)

    def test_stage_c_completion_requires_valid_primary_pair(self):
        secondary_valid = {
            "pair_id": "secondary",
            "comparison_role": "secondary",
            "same_resource_class": True,
            "paired_resource_class": "pure_compute",
            "valid_paired_interference_result": True,
        }
        self.assertEqual(completion_status([secondary_valid])["status"], "inconclusive")
        primary_invalid = {
            **secondary_valid,
            "pair_id": "primary",
            "comparison_role": "primary",
            "valid_paired_interference_result": False,
        }
        self.assertEqual(completion_status([primary_invalid])["status"], "inconclusive")
        primary_valid = {
            **primary_invalid,
            "valid_paired_interference_result": True,
        }
        status = completion_status([primary_valid])
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["primary_valid_pairs"], 1)

    def test_same_class_and_heterogeneous_tables_do_not_overlap(self):
        same = {
            "pair_id": "same",
            "same_resource_class": True,
            "paired_resource_class": "pure_memory",
        }
        cross = {
            "pair_id": "cross",
            "same_resource_class": False,
            "paired_resource_class": None,
        }
        same_rows, cross_rows = split_resource_pair_rows([same, cross])
        self.assertEqual([row["pair_id"] for row in same_rows], ["same"])
        self.assertEqual([row["pair_id"] for row in cross_rows], ["cross"])

    def test_fx_partial_closure_binds_second_placeholder_to_second_input(self):
        class Node:
            def __init__(self, name, op, parents=()):
                self.name = name
                self.op = op
                self.all_input_nodes = list(parents)

        first = Node("first", "placeholder")
        second = Node("second", "placeholder")
        target = Node("double_second", "call_function", (second,))
        output = Node("output", "output", (target,))
        module = types.SimpleNamespace(
            graph=types.SimpleNamespace(nodes=[first, second, target, output])
        )

        class Interpreter:
            def __init__(self, _module):
                self.env = {}

            def run_node(self, node):
                if node is target:
                    return self.env[second] * 2
                raise AssertionError(f"unexpected node execution: {node.name}")

        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        fake_torch = types.SimpleNamespace(
            fx=types.SimpleNamespace(Interpreter=Interpreter),
            inference_mode=lambda: InferenceMode(),
            cuda=types.SimpleNamespace(synchronize=lambda: None),
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            interpreter, targets, _ = prepare_target_interpreter(
                module, (3, 7), ["double_second"], synchronize=False
            )
        self.assertEqual(targets, [target])
        self.assertEqual(interpreter.run_node(target), 14)

    def test_stage_b_reference_identity_must_match_all_processes_and_policies(self):
        results = {}
        identity = [{"shape": [1, 2], "dtype": "torch.float32", "sha256": "same"}]
        for model in common.MODELS:
            for policy in common.POLICIES:
                for trial in range(FORMAL_REPEATS):
                    results[(model, policy, trial)] = {
                        "reference_output_identity": identity
                    }
        audits = validate_model_reference_identities(results, FORMAL_REPEATS)
        self.assertEqual(len(audits), 7)
        results[("GoogLeNet", "newtd_ncu_drt", 9)] = {
            "reference_output_identity": [
                {"shape": [1, 2], "dtype": "torch.float32", "sha256": "changed"}
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "reference output identity differs"):
            validate_model_reference_identities(results, FORMAL_REPEATS)

    def test_stage_c_provenance_rejects_head_or_cache_identity_change(self):
        cache_sources = []
        summary_rows = []
        audits = []
        for model in common.MODELS:
            observed = {
                "model": model,
                "model_class": common.MODEL_CLASSES[model],
                "sha256": f"cache-{model}",
                "profile_sha256": f"profile-{model}",
                "fx_code_sha256": f"fx-{model}",
                "aggregation_method": "identity-checked per-launch median",
                "aggregation_repeat_count": 3,
            }
            cache_sources.append(observed)
            summary_rows.append(
                {
                    "model": model,
                    "model_class": observed["model_class"],
                    "ncu_cache_sha256": observed["sha256"],
                    "profile_sha256": observed["profile_sha256"],
                    "ncu_fx_code_sha256": observed["fx_code_sha256"],
                    "aggregation_method": observed["aggregation_method"],
                    "aggregation_repeat_count": 3,
                }
            )
            audits.append(
                {
                    "model": model,
                    "process_policy_results": 30,
                    "all_reference_output_identities_identical": True,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            verification = Path(directory) / "asset.json"
            verification.write_text("{}", encoding="utf-8")
            import hashlib

            summary = {
                "git_head": "frozen",
                "ncu_cache_identity": summary_rows,
                "asset_verification": str(verification),
                "asset_verification_sha256": hashlib.sha256(b"{}").hexdigest(),
                "model_reference_identity_audit": audits,
            }
            valid = validate_stage_b_provenance(summary, cache_sources, "frozen")
            self.assertTrue(valid["stage_b_ncu_identities_match"])
            with self.assertRaisesRegex(RuntimeError, "git HEAD differs"):
                validate_stage_b_provenance(summary, cache_sources, "other")
            changed = [dict(row) for row in cache_sources]
            changed[0]["sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "NCU identity differs"):
                validate_stage_b_provenance(summary, changed, "frozen")

            latency = Path(directory) / "latency"
            latency.mkdir()
            (latency / "COMPLETE").write_text("done", encoding="utf-8")
            tasks = [
                {"model": model, "policy": policy, "trial": trial}
                for model in common.MODELS
                for policy in common.POLICIES
                for trial in range(FORMAL_REPEATS)
            ]
            (latency / "plan.json").write_text(
                json.dumps(
                    {
                        "git_head": "frozen",
                        "repeats": FORMAL_REPEATS,
                        "task_count": len(tasks),
                        "tasks": tasks,
                    }
                ),
                encoding="utf-8",
            )
            (latency / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed": len(tasks),
                        "total": len(tasks),
                    }
                ),
                encoding="utf-8",
            )
            (latency / "asset.json").write_text("{}", encoding="utf-8")
            task_records = []
            for task in tasks:
                relative = (
                    Path("tasks")
                    / f"trial_{task['trial']:02d}"
                    / common.MODEL_SLUGS[task["model"]]
                    / task["policy"]
                    / "result.json"
                )
                path = latency / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(task), encoding="utf-8")
                task_records.append(
                    {
                        **task,
                        "result_relative_path": str(relative),
                        "result_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                )
            summary["asset_verification_relative_path"] = "asset.json"
            summary["task_records"] = task_records
            audited = validate_stage_b_provenance(
                summary, cache_sources, "frozen", latency
            )
            self.assertEqual(audited["stage_b_task_results_verified"], 210)
            (latency / task_records[0]["result_relative_path"]).write_text(
                "tampered", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "result SHA differs"):
                validate_stage_b_provenance(
                    summary, cache_sources, "frozen", latency
                )

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
        stage_a = (FORMAL / "run_stage_a_profiles.sh").read_text(encoding="utf-8")
        self.assertIn("--repeats 3", stage_a)
        self.assertNotIn("REUSE_NCU", stage_a)

    def test_stage_entrypoints_self_resolve_and_pin_the_commit(self):
        self.assertEqual(len(SCRIPT_NAMES), 3)
        for name in SCRIPT_NAMES:
            source = (FORMAL / name).read_text(encoding="utf-8")
            self.assertIn('SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")', source)
            self.assertIn('DEFAULT_REPO=$(realpath "$SCRIPT_DIR/../..")', source)
            self.assertIn("JANUS_FORMAL_EXPECTED_COMMIT", source)
            self.assertIn('ACTUAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)', source)
            self.assertIn('[[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]', source)
            self.assertIn("EXPECTED_SCRIPT=$(realpath", source)
            self.assertNotIn("janus_release_newtd_ncu_20260827", source)
            self.assertLess(source.index("ACTUAL_COMMIT="), source.index("OUT="))
            self.assertLess(source.index("ACTUAL_COMMIT="), source.index("nvidia-smi"))

        readme = (FORMAL / "README.md").read_text(encoding="utf-8")
        self.assertIn("/public_0/LYX/janus_formal_threeway_20260827", readme)
        self.assertNotIn("/public_0/LYX/janus_release_newtd_ncu_20260827", readme)
        self.assertIn("verify_stage_entrypoints.py", readme)
        self.assertIn("REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT", readme)

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
