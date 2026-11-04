"""Endpoint denominator and frozen generation contract tests; no provider I/O."""
import random
import unittest
from types import SimpleNamespace

import run_swan_path0_completion1 as run


class Path0CompletionTests(unittest.TestCase):
    def test_template_indices_match_actual_official_choices(self):
        bank = list(range(100))
        for slot in range(4):
            rng = random.Random(run.content_seed("template", "source-0001", slot))
            self.assertEqual(run.template_indices("source-0001", slot, 100),
                             [rng.choice(bank) for _ in range(10)])

    def test_call_and_word_budget_covers_all_frozen_allocations(self):
        counts = (13, 13, 12, 12)
        total = sum(n * contract[0] for n, contract in zip(counts, run.BIN_CONTRACTS))
        self.assertEqual(total, 161)
        self.assertEqual(50 * total, run.CALL_CAP)
        self.assertLessEqual(run.CALL_CAP * run.budget.MAX_RESERVE, run.COST_CAP)
        for (n, lo, hi), (lower, upper) in zip(
                run.BIN_CONTRACTS, ((17, 24), (25, 32), (33, 40), (41, 48))):
            self.assertGreaterEqual(n * lo, lower)
            self.assertLessEqual(n * hi, upper)

    def test_full_denominator_and_joint_survival_are_distinct(self):
        rows = []
        for index in range(50):
            source = {"allocation_id": str(index), "length_bin": 0}
            for variant in run.VARIANTS:
                row = run.miss_row(source, variant, {}, "generation_failed")
                # Two baseline detections; each loses a different attack.
                if index < 2 and not ((index == 0 and variant == "semicolon_all") or
                                      (index == 1 and variant == "comma_all")):
                    row["word_count"] = 20
                    for arm in row["arms"]:
                        row["arms"][arm] = {"status": "ok", "statistic": 1}
                rows.append(row)
        calibrator = SimpleNamespace(
            calibrate=lambda *args: SimpleNamespace(is_watermarked=True))
        result = run.positive_summary(rows, {"control": calibrator})["control"]["primary"]
        self.assertEqual(result["full_denominator_TPR"]["baseline"]["trials"], 50)
        self.assertEqual(result["full_denominator_TPR"]["baseline"]["successes"], 2)
        self.assertEqual(result["worst_condition_survival"], .5)
        self.assertEqual(result["all_five_survival"]["successes"], 0)
        self.assertEqual(result["all_five_survival"]["trials"], 2)
        self.assertEqual(result["abstain_full_denominator"]["baseline"], 48)

    def test_no_baseline_detections_has_undefined_survival(self):
        rows = [run.miss_row({"allocation_id": str(i), "length_bin": 0}, v, {}, "failed")
                for i in range(50) for v in run.VARIANTS]
        calibrator = SimpleNamespace(calibrate=lambda *args: self.fail("Abstain scored"))
        result = run.positive_summary(rows, {"path0": calibrator})["path0"]["primary"]
        self.assertIsNone(result["worst_condition_survival"])
        self.assertIsNone(result["all_five_survival"]["rate"])
        with self.assertRaises(run.IntegrityHalt):
            run.positive_summary(rows[:-1], {"path0": calibrator})


if __name__ == "__main__":
    unittest.main()
