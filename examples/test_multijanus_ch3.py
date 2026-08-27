#!/usr/bin/env python3

import unittest

from build_compatibility_table import build_entry
from multi_janus_benchmark import pair_key, percentile, stats
from run_ch3_matrix import pair_slug


def fake_result(mode, response, throughput, gpu, trial=0):
    models = ["GoogLeNet", "GoogLeNet"]
    return {
        "models": models,
        "batch_sizes": [1, 1],
        "trial": trial,
        "mode_effective": mode,
        "overall": {
            "correctness_ok": True,
            "response_ms": {"mean": response},
            "throughput_requests_per_second": throughput,
        },
        "clients": [
            {
                "client_id": index,
                "model": model,
                "batch_size": 1,
                "summary": {"gpu_event_ms": {"median": gpu[index]}},
            }
            for index, model in enumerate(models)
        ],
    }


class MultiJanusUnitTests(unittest.TestCase):
    def test_pair_key_is_order_independent(self):
        left = pair_key(["GoogLeNet", "NASNetALarge"], [1, 4])
        right = pair_key(["NASNetALarge", "GoogLeNet"], [4, 1])
        self.assertEqual(left, right)

    def test_pair_slug_is_order_independent(self):
        left = pair_slug("GoogLeNet", "NASNetALarge", 1)
        right = pair_slug("NASNetALarge", "GoogLeNet", 1)
        self.assertEqual(left, right)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(percentile([7], 0.99), 7)

    def test_stats_keeps_tail(self):
        result = stats([1, 1, 1, 10])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 1)
        self.assertGreater(result["p95"], 8)

    def test_lookup_accepts_good_pair(self):
        serial = [fake_result("sequential", 2.0, 100.0, [1.0, 1.0])]
        parallel = [fake_result("concurrent", 1.0, 190.0, [1.1, 1.1])]
        _, entry = build_entry(serial, parallel, 1.10, 1.05, 1.75)
        self.assertTrue(entry["allow_concurrent"])

    def test_lookup_rejects_heavy_slowdown(self):
        serial = [fake_result("sequential", 2.0, 100.0, [1.0, 1.0])]
        parallel = [fake_result("concurrent", 1.5, 120.0, [1.9, 1.9])]
        _, entry = build_entry(serial, parallel, 1.10, 1.05, 1.75)
        self.assertFalse(entry["allow_concurrent"])
        self.assertIn("service_slowdown", entry["reason"])


if __name__ == "__main__":
    unittest.main()
