from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.core.formal_null_evaluation import (
    _clopper_pearson,
    prepare,
    select_prefix_window,
)


ROOT = Path(__file__).resolve().parents[1]


class FormalNullEvaluationTests(unittest.TestCase):
    def test_window_rule_uses_first_eligible_hard_prefix(self) -> None:
        sentences = [
            "This sentence contains exactly enough ordinary words for testing."
            for _ in range(12)
        ]
        window = select_prefix_window(" ".join(sentences))
        self.assertIsNotNone(window)
        self.assertGreaterEqual(window["word_count"], 100)
        self.assertEqual(window["hard_unit_count"], 12)

    source_root = ROOT / "experiments/swan_count_neutral_final1/sealed"

    @unittest.skipUnless(
        (source_root / "calibration_sources.json").is_file()
        and (source_root / "test_sources.json").is_file(),
        "raw frozen allocations are not deployed on this host",
    )
    def test_real_frozen_allocations_reproduce_preregistered_selection(self) -> None:
        source_root = self.source_root
        with tempfile.TemporaryDirectory(prefix="formal-null-") as folder:
            manifest = prepare(
                source_root / "calibration_sources.json",
                source_root / "test_sources.json",
                Path(folder),
                sample_count=1000,
            )
            calibration = manifest["summaries"]["calibration"]
            test = manifest["summaries"]["independent_test"]
            self.assertEqual(calibration["eligible_rows"], 2760)
            self.assertEqual(
                calibration["last_selected_allocation_ordinal"], 1447
            )
            self.assertEqual(test["eligible_rows"], 2764)
            self.assertEqual(test["last_selected_allocation_ordinal"], 1448)
            self.assertFalse(
                any(manifest["identity_overlap_counts"].values())
            )

            calibration_rows = [
                json.loads(line)
                for line in (
                    Path(folder) / "inputs/calibration_windows.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(calibration_rows), 1000)
            self.assertAlmostEqual(
                sum(row["word_count"] for row in calibration_rows) / 1000,
                154.245,
            )

    def test_exact_binomial_interval_is_reported(self) -> None:
        interval = _clopper_pearson(0, 1000)
        self.assertEqual(interval["method"], "clopper_pearson_exact")
        self.assertEqual(interval["lower"], 0.0)
        self.assertLess(interval["one_sided_upper_95"], 0.003)


if __name__ == "__main__":
    unittest.main()
