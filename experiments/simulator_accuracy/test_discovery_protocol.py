#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("discover_simulator_candidates.py")
SPEC = importlib.util.spec_from_file_location("discover_simulator_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeKernel:
    def __init__(self, name):
        self.name = name


class FakeOperator:
    def __init__(self, name):
        self.name = name
        self.kernels = [FakeKernel(name + "_kernel")]


class FakeResourceModel:
    time_domain = False

    def __deepcopy__(self, memo):
        copied = type(self)()
        copied.time_domain = self.time_domain
        return copied

    def update_time(self, now):
        self.now = now

    def can_apply_launch(self, operator, now):
        return not operator.name.startswith("static_reject")

    def apply_launch(self, operator, now):
        pass

    def total_utilization(self):
        return 0.5

    def evaluate_initial_combo(self, group, now):
        rejected = any(operator.name.startswith("td_reject") for operator in group)
        return {
            "feasible": not rejected,
            "failure_reason": "test_reject" if rejected else None,
            "initial_utilization": 0.25 if not rejected else -1.0,
            "initial_resident_blocks": {
                operator.name: 1 for operator in group if not rejected
            },
        }


class DiscoveryProtocolTests(unittest.TestCase):
    def test_enumerates_all_sizes_two_through_five(self):
        operators = [FakeOperator(f"op_{index}") for index in range(6)]
        rows = MODULE.enumerate_candidates(
            model="Test",
            reference_variant="Baseline",
            call=1,
            ready_ops=operators,
            max_group_size=5,
            resource_model=FakeResourceModel(),
            now=0.0,
        )
        self.assertEqual(len(rows), 56)
        self.assertEqual({row["group_size"] for row in rows}, {2, 3, 4, 5})

    def test_static_and_td_are_independent_labels(self):
        operators = [
            FakeOperator("static_reject_a"),
            FakeOperator("td_reject_b"),
        ]
        row = MODULE.candidate_row(
            model="Test",
            reference_variant="TD+DRT",
            call=7,
            ready_ops=operators,
            group=operators,
            resource_model=FakeResourceModel(),
            now=0.0,
        )
        self.assertFalse(row["static_prediction"])
        self.assertFalse(row["td_prediction"])
        self.assertEqual(row["call"], 7)

    def test_identity_changes_with_reference_state(self):
        operators = [FakeOperator("a"), FakeOperator("b")]
        common = dict(
            model="Test",
            reference_variant="Baseline",
            ready_ops=operators,
            group=operators,
            resource_model=FakeResourceModel(),
            now=0.0,
        )
        first = MODULE.candidate_row(call=1, **common)
        second = MODULE.candidate_row(call=2, **common)
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])

    def test_sample_stratum_is_paired_prediction_and_width(self):
        row = {
            "model": "Test",
            "static_prediction": False,
            "td_prediction": True,
            "group_size": 4,
        }
        selector_path = Path(__file__).with_name("select_positive_sample.py")
        spec = importlib.util.spec_from_file_location(
            "select_positive_sample", selector_path
        )
        selector = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(selector)
        self.assertEqual(
            selector.stratum_key(row), ("Test", False, True, 4)
        )


if __name__ == "__main__":
    unittest.main()
