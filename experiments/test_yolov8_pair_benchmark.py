import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_yolov8_pairs import (
    derive_pair_metrics,
    load_pair_specs,
)


class PairMetricTests(unittest.TestCase):
    def test_derive_pair_metrics(self):
        metrics = derive_pair_metrics(
            solo_a=2.0,
            solo_b=1.0,
            corun_a=3.0,
            corun_b=2.0,
            corun_makespan=3.0,
        )

        self.assertEqual(metrics["slowdown_a"], 1.5)
        self.assertEqual(metrics["slowdown_b"], 2.0)
        self.assertEqual(metrics["mean_slowdown"], 1.75)
        self.assertEqual(metrics["max_slowdown"], 2.0)
        self.assertEqual(metrics["makespan_dilation"], 1.5)
        self.assertEqual(metrics["measured_pair_speedup"], 1.0)

    def test_load_external_pair_specs(self):
        payload = {
            "pairs": [{
                "id": "a_b",
                "a": "a",
                "b": "b",
                "predicted_risk": 0.2,
                "predicted_speedup": 1.1,
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specs = load_pair_specs(path)

        self.assertEqual(specs[0]["id"], "a_b")
        self.assertEqual(specs[0]["scheduler_decision"], "candidate")


if __name__ == "__main__":
    unittest.main()
