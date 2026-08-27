#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import unittest

from td_v2_simulator import simulate_strict_overlap


@dataclass
class Kernel:
    name: str
    duration: float
    shared_mem: int
    registers: int
    warps: int
    blocks: int


@dataclass
class Operator:
    name: str
    kernels: list[Kernel]


class SM:
    shared_mem_total = 100
    register_total = 100
    warp_total = 100
    shared_mem_used = 0
    registers_used = 0
    warps_used = 0
    running_blocks = []


class Model:
    def __init__(self, count=1):
        self.sms = [SM() for _ in range(count)]


def op(name, duration, resource, blocks=1):
    return Operator(
        name,
        [Kernel(name + "_k", duration, resource, resource, resource, blocks)],
    )


class TDV2Tests(unittest.TestCase):
    def test_leftover_capacity_allows_overlap(self):
        result = simulate_strict_overlap(
            [op("a", 10, 40), op("b", 10, 40)],
            Model(),
            launch_gap=1,
        )
        self.assertTrue(result["strict_parallel"])
        self.assertGreater(result["strict_overlap_duration"], 0)

    def test_no_future_reservation(self):
        result = simulate_strict_overlap(
            [op("a", 10, 100), op("b", 10, 100)],
            Model(),
            launch_gap=1,
        )
        self.assertFalse(result["strict_parallel"])
        self.assertEqual(result["max_concurrent_operators"], 1)

    def test_short_first_kernel_finishes_before_second_launch(self):
        result = simulate_strict_overlap(
            [op("a", 1, 20), op("b", 10, 20)],
            Model(),
            launch_gap=2,
        )
        self.assertFalse(result["strict_parallel"])

    def test_multiple_sms_preserve_leftover_parallelism(self):
        result = simulate_strict_overlap(
            [op("a", 10, 100), op("b", 10, 100)],
            Model(count=2),
            launch_gap=1,
        )
        self.assertTrue(result["strict_parallel"])


if __name__ == "__main__":
    unittest.main()
