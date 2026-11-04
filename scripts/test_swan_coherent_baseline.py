"""Offline tests for the coherent SWAN baseline qualification and detection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_swan_coherent_baseline as baseline


class CleanModule:
    def parse_amr(self, text):
        return "(a / approve-01)"

    def normalize_amr_variables(self, graph):
        return graph

    def generate_template_amr(self, graph):
        return graph


class SourceTests(unittest.TestCase):
    def test_source_is_ten_unique_strict_acceptances(self):
        inputs = baseline.source_inputs()
        selected = inputs["selected"]
        checks = inputs["structural_coherence_checks"]
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({row["text_sha256"] for row in selected}), 10)
        self.assertTrue(all(row["target_score"] >= 0.75 for row in selected))
        self.assertTrue(checks["pass"])
        self.assertEqual(len(checks["expected_predicate_hits"]), 10)
        self.assertFalse(inputs["source_prior_auxiliary_gate_pass"])


class ReviewTests(unittest.TestCase):
    GOOD = {
        "coherence": 4,
        "naturalness": 3,
        "nonredundancy": 4,
        "task_consistency": 4,
        "severe_problem": False,
    }

    def test_event_log_threshold_passes(self):
        row = baseline.parse_review(json.dumps(self.GOOD))
        self.assertTrue(baseline.review_pass(row))

    def test_invalid_or_materially_weak_review_fails(self):
        self.assertFalse(
            baseline.review_pass({**self.GOOD, "task_consistency": 3})
        )
        self.assertFalse(
            baseline.review_pass({**self.GOOD, "severe_problem": True})
        )
        self.assertIsNone(
            baseline.parse_review(json.dumps({**self.GOOD, "extra": 1}))
        )


class DetectionTests(unittest.TestCase):
    SOURCE = {
        "slot": 0,
        "text": "Maria Lopez approved the recommendations.",
        "text_sha256": "unused-in-test",
        "target_template_index": 47,
        "target_score": 0.9,
    }

    def test_clean_unit_is_green(self):
        with mock.patch.object(
            baseline,
            "score_triple",
            return_value=(0.7, 0.8, 0.75),
        ):
            row = baseline.score_unit(
                CleanModule(),
                ["(a / approve-01)"],
                "doc",
                0,
                self.SOURCE,
            )
        self.assertTrue(row["green"])
        self.assertEqual(row["max_score"], 0.75)

    def test_detection_exception_is_frozen(self):
        class FailingModule(CleanModule):
            def parse_amr(self, text):
                raise RuntimeError("parse failure")

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(baseline, "OUT", Path(folder)):
                with self.assertRaisesRegex(RuntimeError, "parse failure"):
                    baseline.score_unit(
                        FailingModule(),
                        ["(a / approve-01)"],
                        "doc",
                        0,
                        self.SOURCE,
                    )
                self.assertTrue(
                    (Path(folder) / "detection_failure_unit_00.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
