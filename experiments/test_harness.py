from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import PRIMARY_VARIANTS, expand_tasks, load_config, stats_from_samples, task_id, validate_run_id, verify_manifest


class HarnessTests(unittest.TestCase):
    def setUp(self): self.config = load_config()

    def test_primary_matrix_has_unambiguous_alpha_rows(self):
        tasks = expand_tasks(self.config, ["GoogLeNet"], PRIMARY_VARIANTS, 1)
        self.assertEqual(len(tasks), 11)
        baseline = [task for task in tasks if task.variant == "Baseline"]
        drt = [task for task in tasks if task.variant.endswith("DRT")]
        cosine = [task for task in tasks if task.variant.endswith("Cos")]
        self.assertEqual(len(baseline), 1); self.assertIsNone(baseline[0].alpha)
        self.assertEqual(len(drt), 2); self.assertTrue(all(task.alpha is None for task in drt))
        self.assertEqual(len(cosine), 8); self.assertTrue(all(task.alpha in self.config["alpha_grid"] for task in cosine))

    def test_shuffle_is_reproducible_and_ids_are_unique(self):
        first = expand_tasks(self.config, ["GoogLeNet", "BERT"], PRIMARY_VARIANTS, 2)
        second = expand_tasks(self.config, ["GoogLeNet", "BERT"], PRIMARY_VARIANTS, 2)
        self.assertEqual(first, second); ids = [task_id(task) for task in first]; self.assertEqual(len(ids), len(set(ids)))

    def test_statistics(self):
        stats = stats_from_samples([1.0, 2.0, 3.0])
        self.assertEqual((stats["median_ms"], stats["mean_ms"], stats["sample_std_ms"], stats["mad_ms"]), (2.0, 2.0, 1.0, 1.0))

    def test_manifest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload = root / "payload.bin"; payload.write_bytes(b"janus")
            digest = hashlib.sha256(b"janus").hexdigest(); manifest = root / "manifest.sha256"
            manifest.write_text(f"{digest}  payload.bin\n", encoding="utf-8")
            self.assertTrue(verify_manifest(manifest, root)[0]["ok"])

    def test_frozen_model_profiles_exist(self):
        profile_root = Path(__file__).resolve().parents[1] / "Opara" / "profile_result"
        for spec in self.config["models"].values():
            self.assertTrue((profile_root / spec["profile_file"]).is_file())

    def test_run_id_validation(self):
        self.assertEqual(validate_run_id("run_20260728-a"), "run_20260728-a")
        with self.assertRaises(ValueError): validate_run_id("../escape")


if __name__ == "__main__": unittest.main()
