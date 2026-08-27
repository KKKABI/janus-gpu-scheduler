import unittest
from types import SimpleNamespace

from Opara.ncu_profiler import merge_ncu_v2_to_nodes


def info(name, duration, grid=(1, 1, 1), block=(32, 1, 1)):
    return {
        'name': name,
        'dur': duration,
        'args': {'grid': grid, 'block': block},
    }


def source(name, grid=1, block=32, dram=40.0, l2=20.0, comp=30.0):
    return {
        'name': name,
        'grid_size': grid,
        'block_size': block,
        'metrics': {
            'dram_thru': dram,
            'l2_thru': l2,
            'comp_thru': comp,
        },
    }


class NcuProfilerV2Tests(unittest.TestCase):
    def test_exact_launch_occurrences_are_zipped_in_order(self):
        first = info('void foo<int>()', 3.0)
        second = info('void foo<int>()', 1.0)
        nodes = [SimpleNamespace(info=[first, second])]
        data = {'kernels': [
            source('void foo<int>()', dram=10.0),
            source('void foo<int>()', dram=90.0),
        ]}
        report = merge_ncu_v2_to_nodes(nodes, data, 0.5)
        self.assertEqual(report['status'], 'accepted')
        self.assertEqual(report['mapped_kernels'], 2)
        self.assertEqual(first['dram_thru'], 10.0)
        self.assertEqual(second['dram_thru'], 90.0)

    def test_count_mismatch_is_ambiguous_not_guessed(self):
        target = info('void generic<float>()', 1.0)
        nodes = [SimpleNamespace(info=[target])]
        data = {'kernels': [
            source('void generic<float>()'),
            source('void generic<float>()'),
        ]}
        report = merge_ncu_v2_to_nodes(nodes, data, 0.0)
        self.assertEqual(report['ambiguous_kernels'], 1)
        self.assertNotIn('dram_thru', target)

    def test_namespace_difference_uses_ordered_geometry_fallback(self):
        target = info('void vendor::foo<int>()', 1.0, grid=(4, 1, 1))
        nodes = [SimpleNamespace(info=[target])]
        data = {'kernels': [source(
            'void other_vendor::foo<float>()', grid=4, dram=55.0
        )]}
        report = merge_ncu_v2_to_nodes(nodes, data, 0.5)
        self.assertEqual(report['ordered_fallback_mapped_kernels'], 1)
        self.assertEqual(target['ncu_match'], 'ordered_fallback_v2')
        self.assertEqual(target['dram_thru'], 55.0)

    def test_op_nvtx_identity_uses_duration_weighted_operator_profile(self):
        first = info('kernel_a', 2.0)
        second = info('kernel_b', 3.0)
        missing = info('kernel_c', 5.0)
        nodes = [
            SimpleNamespace(name='x_1', info=[first, second]),
            SimpleNamespace(name='x_2', info=[missing]),
        ]
        left = source('ncu_a', dram=20.0)
        left.update({'op_name': 'x_1', 'metrics': {**left['metrics'], 'dur_ns': 1.0}})
        right = source('ncu_b', dram=60.0)
        right.update({'op_name': 'x_1', 'metrics': {**right['metrics'], 'dur_ns': 3.0}})
        report = merge_ncu_v2_to_nodes(nodes, {'kernels': [left, right]}, 0.5)
        self.assertEqual(report['mapping_mode'], 'op_nvtx_v2')
        self.assertEqual(report['mapped_operators'], 1)
        self.assertAlmostEqual(report['duration_coverage'], 0.5)
        self.assertAlmostEqual(first['dram_thru'], 50.0)
        self.assertAlmostEqual(second['dram_thru'], 50.0)
        self.assertNotIn('dram_thru', missing)

    def test_duration_gate_prevents_partial_merge(self):
        mapped = info('void mapped()', 1.0)
        missing = info('void missing()', 9.0)
        nodes = [SimpleNamespace(info=[mapped, missing])]
        data = {'kernels': [source('void mapped()')]}
        report = merge_ncu_v2_to_nodes(nodes, data, 0.5)
        self.assertEqual(report['status'], 'coverage_below_threshold')
        self.assertAlmostEqual(report['duration_coverage'], 0.1)
        self.assertNotIn('dram_thru', mapped)


if __name__ == '__main__':
    unittest.main()
