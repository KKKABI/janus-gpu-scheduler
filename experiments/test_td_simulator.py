import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "opara_scheduler", ROOT / "Opara" / "Scheduler.py"
)
SCHEDULER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER_MODULE)

KernelProfile = SCHEDULER_MODULE.KernelProfile
OperatorTask = SCHEDULER_MODULE.OperatorTask
ResourceModel = SCHEDULER_MODULE.ResourceModel
Scheduler = SCHEDULER_MODULE.Scheduler
clear_candidate_stats = SCHEDULER_MODULE.clear_candidate_stats
get_candidate_stats = SCHEDULER_MODULE.get_candidate_stats


SM_SPECS = {
    "shared_mem_total": 100,
    "register_total": 100,
    "warp_total": 4,
}


def make_operator(name, *, blocks=1, warps=2, shared_mem=0, registers=0):
    return OperatorTask(name, [KernelProfile(
        f"{name}_kernel",
        duration=1.0,
        shared_mem=shared_mem,
        registers=registers,
        warps=warps,
        blocks=blocks,
    )])


def add_ncu_pressure(operator, *, dram=0.0, l2=0.0, compute=0.0):
    kernel = operator.kernels[0]
    kernel.dram_thru = dram
    kernel.l2_thru = l2
    kernel.comp_thru = compute
    return operator


class TimeDomainSimulatorTests(unittest.TestCase):
    def test_large_single_operator_uses_block_waves(self):
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        large = make_operator("large", blocks=10, warps=2)

        self.assertTrue(model.can_apply_launch(large, 0.0))
        self.assertTrue(model.try_apply_concurrent_combo([large], 0.0))
        self.assertEqual(len(model.sms[0].running_blocks), 2)
        self.assertEqual(model.sms[0].warps_used, 4)

    def test_static_still_requires_all_blocks_to_co_reside(self):
        model = ResourceModel(1, SM_SPECS, time_domain=False)
        large = make_operator("large", blocks=10, warps=2)

        self.assertFalse(model.can_apply_launch(large, 0.0))

    def test_two_operators_can_initially_co_reside(self):
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        operators = [make_operator("a"), make_operator("b")]

        self.assertTrue(model.try_apply_concurrent_combo(operators, 0.0))
        self.assertEqual(len(model.sms[0].running_blocks), 2)

    def test_third_operator_is_rejected_without_initial_slot(self):
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        operators = [
            make_operator("a"),
            make_operator("b"),
            make_operator("c"),
        ]

        self.assertFalse(model.try_apply_concurrent_combo(operators, 0.0))
        self.assertEqual(len(model.sms[0].running_blocks), 0)
        self.assertEqual(model.sms[0].warps_used, 0)

    def test_scheduler_records_non_feasible_triple(self):
        clear_candidate_stats()
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        scheduler = Scheduler(
            model,
            alpha=0.9,
            selection_mode="max_occupancy",
            time_domain=True,
        )
        operators = [
            make_operator("a"),
            make_operator("b"),
            make_operator("c"),
        ]

        scheduler.schedule(operators, 0.0)
        stats = get_candidate_stats(clear=True)

        self.assertEqual(stats[0]["enumerated_count"], 7)
        self.assertEqual(stats[0]["feasible_count"], 6)
        self.assertEqual(stats[0]["candidate_score_kind"], "initial_occupancy")
        self.assertEqual(stats[0]["final_score_kind"], "predicted_speedup")
        self.assertEqual(stats[0]["candidate_score_max"], 1.0)
        self.assertEqual(
            stats[0]["selected_timeline"]["predicted_speedup"], 2.0
        )

    def test_strategy_final_selector_does_not_get_overridden_by_timeline(self):
        operators = [
            make_operator("a"),
            make_operator("b"),
            make_operator("c"),
        ]
        operators[0].kernels[0].duration = 10.0
        operators[1].kernels[0].duration = 1.0
        operators[2].kernels[0].duration = 10.0

        with mock.patch.dict(os.environ, {
                "OPARA_TD_FINAL_SELECTOR": "strategy"}):
            scheduler = Scheduler(
                ResourceModel(1, SM_SPECS, time_domain=True),
                alpha=0.0,
                selection_mode="max_occupancy",
                time_domain=True,
            )
            selected = scheduler.schedule(operators, 0.0)

        self.assertEqual([operator.name for operator in selected], ["a", "b"])

    def test_timeline_final_selector_can_override_stage1_rank(self):
        operators = [
            make_operator("a"),
            make_operator("b"),
            make_operator("c"),
        ]
        operators[0].kernels[0].duration = 10.0
        operators[1].kernels[0].duration = 1.0
        operators[2].kernels[0].duration = 10.0

        with mock.patch.dict(os.environ, {
                "OPARA_TD_FINAL_SELECTOR": "timeline"}):
            scheduler = Scheduler(
                ResourceModel(1, SM_SPECS, time_domain=True),
                alpha=0.0,
                selection_mode="max_occupancy",
                time_domain=True,
            )
            selected = scheduler.schedule(operators, 0.0)

        self.assertEqual({operator.name for operator in selected}, {"a", "c"})

    def test_ncu_complementary_pair_has_lower_interference_risk(self):
        scheduler = Scheduler(
            ResourceModel(1, SM_SPECS, time_domain=True),
            alpha=0.0,
            selection_mode="max_occupancy",
            time_domain=True,
        )
        compute_a = add_ncu_pressure(
            make_operator("compute_a"), compute=90.0
        )
        compute_b = add_ncu_pressure(
            make_operator("compute_b"), compute=90.0
        )
        memory = add_ncu_pressure(
            make_operator("memory"), dram=90.0
        )

        same_risk = scheduler._combo_interference_metrics(
            [compute_a, compute_b]
        )["risk"]
        complementary_risk = scheduler._combo_interference_metrics(
            [compute_a, memory]
        )["risk"]

        self.assertGreater(same_risk, complementary_risk)

    def test_guarded_selector_prefers_complementary_equal_speedup_pair(self):
        clear_candidate_stats()
        operators = [
            add_ncu_pressure(
                make_operator("compute_a"), compute=90.0
            ),
            add_ncu_pressure(
                make_operator("compute_b"), compute=90.0
            ),
            add_ncu_pressure(
                make_operator("memory"), dram=90.0
            ),
        ]
        with mock.patch.dict(os.environ, {
                "OPARA_TD_FINAL_SELECTOR": "guarded_interference",
                "OPARA_TD_SPEEDUP_GUARD": "0.5"}):
            scheduler = Scheduler(
                ResourceModel(1, SM_SPECS, time_domain=True),
                alpha=0.0,
                selection_mode="max_occupancy",
                time_domain=True,
            )
            selected = scheduler.schedule(operators, 0.0)

        stats = get_candidate_stats(clear=True)[0]
        self.assertEqual(
            {operator.name for operator in selected},
            {"compute_a", "memory"},
        )
        self.assertEqual(
            stats["selected_timeline"]["final_selector"],
            "guarded_interference",
        )
        self.assertEqual(
            stats["selected_timeline"]["selector_selected_speedup_loss"],
            0.0,
        )

    def test_shared_timeline_runs_large_grid_in_waves(self):
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        large = make_operator("large", blocks=5, warps=2)

        metrics = model.simulate_combo_timeline([large], 0.0)

        self.assertTrue(metrics["feasible"])
        self.assertEqual(metrics["initial_resident_blocks"]["large"], 2)
        self.assertEqual(metrics["event_count"], 3)
        self.assertEqual(metrics["makespan"], 3.0)
        self.assertAlmostEqual(metrics["average_utilization"], 5.0 / 6.0)

    def test_shared_timeline_keeps_operator_kernels_sequential(self):
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        operator_a = OperatorTask("a", [
            KernelProfile("a0", 1.0, 0, 0, 2, 2),
            KernelProfile("a1", 2.0, 0, 0, 4, 1),
        ])
        operator_b = make_operator("b", blocks=1, warps=2)
        operator_b.kernels[0].duration = 2.0

        metrics = model.simulate_combo_timeline(
            [operator_a, operator_b], 0.0
        )

        self.assertTrue(metrics["feasible"])
        self.assertEqual(metrics["makespan"], 4.0)
        self.assertEqual(metrics["operator_completion_times"]["b"], 2.0)
        self.assertEqual(metrics["operator_completion_times"]["a"], 4.0)
        self.assertGreater(metrics["overlap_duration"], 0.0)

    def test_shared_timeline_is_independent_of_candidate_order(self):
        operators = [
            make_operator("a", blocks=5, warps=1),
            make_operator("b", blocks=3, warps=2),
        ]
        forward = ResourceModel(
            2, SM_SPECS, time_domain=True
        ).simulate_combo_timeline(operators, 0.0)
        reverse = ResourceModel(
            2, SM_SPECS, time_domain=True
        ).simulate_combo_timeline(list(reversed(operators)), 0.0)

        self.assertEqual(forward["feasible"], reverse["feasible"])
        self.assertEqual(forward["makespan"], reverse["makespan"])
        self.assertEqual(
            forward["average_utilization"],
            reverse["average_utilization"],
        )

    def test_zero_resource_nodes_bypass_gpu_candidate_enumeration(self):
        clear_candidate_stats()
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        scheduler = Scheduler(
            model,
            alpha=0.9,
            selection_mode="max_occupancy",
            time_domain=True,
        )
        resource_op = make_operator("resource")
        metadata_ops = [
            OperatorTask("metadata_a", []),
            OperatorTask("metadata_b", []),
        ]

        selected = scheduler.schedule(
            [resource_op] + metadata_ops, 0.0
        )
        stats = get_candidate_stats(clear=True)[0]

        self.assertEqual({op.name for op in selected}, {
            "resource", "metadata_a", "metadata_b"
        })
        self.assertEqual(stats["enumerated_count"], 1)
        self.assertEqual(stats["resource_ready_count"], 1)
        self.assertEqual(stats["passthrough_count"], 2)

    def test_only_zero_resource_nodes_are_returned_without_combinations(self):
        clear_candidate_stats()
        model = ResourceModel(1, SM_SPECS, time_domain=True)
        scheduler = Scheduler(
            model,
            alpha=0.9,
            selection_mode="max_occupancy",
            time_domain=True,
        )
        metadata_ops = [
            OperatorTask("metadata_a", []),
            OperatorTask("metadata_b", []),
        ]

        selected = scheduler.schedule(metadata_ops, 0.0)
        stats = get_candidate_stats(clear=True)[0]

        self.assertEqual({op.name for op in selected}, {
            "metadata_a", "metadata_b"
        })
        self.assertEqual(stats["enumerated_count"], 0)
        self.assertEqual(stats["passthrough_count"], 2)


if __name__ == "__main__":
    unittest.main()
