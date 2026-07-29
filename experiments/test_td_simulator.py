import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
