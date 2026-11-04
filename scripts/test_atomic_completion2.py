"""Negative evidence and source-leakage tests; synthetic data only."""
import unittest

from collect_atomic_alignment_oracle2 import resolve
from swan_atomic_calibration1 import calibrate, evaluate, reject, threshold, validate_rows


PAIR = {"candidate_index": 0, "event_id": "e0", "event_candidate_index": 0}


def packet(reviewer, value):
    return {"reviewer_code": reviewer, "human_attestation": True,
            "independent_attestation": True, "no_model_or_target_answers_used": True,
            "freeze_sha256": "f", "prediction_seal_sha256": "s",
            "completed_at": "2026-11-02T10:00:00+00:00",
            "rows": [{"row_id": "r", "pairs": [
                {**PAIR, "same_fact_supported": value, "rationale": "SYNTHETIC TEST ONLY"}]}]}


class AdjudicationTests(unittest.TestCase):
    def test_disagreement_stays_pending(self):
        labels, disputes, status = resolve({"r": [PAIR]}, [packet("a", True), packet("b", False)],
                                           ["h1", "h2"], "f", "s")
        self.assertEqual(labels, {})
        self.assertEqual(status["status"], "PENDING_ADJUDICATION")
        self.assertEqual(len(disputes), 1)

    def test_third_review_binds_both_originals(self):
        third = {**packet("c", False), "original_submission_sha256": ["h1", "h2"]}
        args = ({"r": [PAIR]}, [packet("a", True), packet("b", False)], ["h1", "h2"], "f", "s")
        labels, _, status = resolve(*args, third)
        self.assertFalse(labels["r"][0, "e0"])
        self.assertEqual(status["adjudicated_pairs"], 1)
        third["original_submission_sha256"] = ["foreign", "h2"]
        self.assertEqual(resolve(*args, third)[0], {})
        third["original_submission_sha256"] = ["h1", "h2"]
        third["reviewer_code"] = "a"
        self.assertEqual(resolve(*args, third)[0], {})

    def test_third_cannot_fill_unknown_original(self):
        third = {**packet("c", True), "original_submission_sha256": ["h1", "h2"]}
        labels, _, status = resolve({"r": [PAIR]}, [packet("a", None), packet("b", False)],
                                    ["h1", "h2"], "f", "s", third)
        self.assertFalse(labels)
        self.assertEqual(status["status"], "PENDING_COMPLETE_REVIEWS")

    def test_empty_universe_can_be_reviewed(self):
        packets = [packet("a", True), packet("b", True)]
        for p in packets:
            p["rows"][0]["pairs"] = []
        self.assertEqual(resolve({"r": []}, packets, ["h1", "h2"], "f", "s")[0], {"r": {}})


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.expected = {str(i): {"role": role, "kind": kind, "source_sha256": str(i)}
                         for i, (role, kind) in enumerate(
                             [("calibration", "human")] * 100
                             + [("null_test", "human")] * 10
                             + [("positive_test", "watermarked")] * 10)}
        self.rows = [
            {"source_id": sid, "condition": condition, **meta,
             "scores": {arm: int(sid) for arm in ("A0", "A1", "A2", "single_view")}}
            for sid, meta in self.expected.items() for condition in ("clean", "split")]

    def test_strict_ties_match_finite_sample_pvalue(self):
        values = [i // 3 for i in range(100)]
        cut = threshold(values, .01)
        for score in range(40):
            p = (1 + sum(v >= score for v in values)) / 101
            self.assertEqual(reject(score, cut), p <= .01)
        self.assertFalse(reject(100, threshold([1, 2], .01)))

    def test_max_condition_keeps_source_denominator(self):
        fitted = calibrate(self.rows, self.expected, ["clean", "split"])
        self.assertEqual(fitted["thresholds"]["A1"]["calibration_n"], 100)
        result = evaluate(self.rows, self.expected, fitted, "split")
        self.assertEqual(result["TPR_full_denominator"]["A0"]["split"]["n"], 10)
        self.assertEqual(result["primary_paired_differences"]["A1_minus_A0"]["difference"], 0)

    def test_duplicate_and_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(self.rows + self.rows[:1], self.expected, ["clean", "split"])
        self.expected["110"]["source_sha256"] = self.expected["0"]["source_sha256"]
        with self.assertRaises(ValueError):
            validate_rows(self.rows, self.expected, ["clean", "split"])

    def test_missing_human_labels_are_not_abstention(self):
        self.rows[0]["scores"]["A2"] = None
        with self.assertRaises(ValueError):
            validate_rows(self.rows, self.expected, ["clean", "split"])
        self.rows[0]["abstentions"] = {"A2": "PENDING_HUMANS"}
        with self.assertRaises(ValueError):
            validate_rows(self.rows, self.expected, ["clean", "split"])
        self.rows[0]["abstentions"]["A2"] = "no_usable_observation"
        self.assertEqual(len(validate_rows(self.rows, self.expected, ["clean", "split"])), 240)

    def test_threshold_drift_rejected(self):
        fitted = calibrate(self.rows, self.expected, ["clean", "split"])
        self.rows[0]["scores"]["A0"] = 1000
        with self.assertRaises(ValueError):
            evaluate(self.rows, self.expected, fitted, "split")


if __name__ == "__main__":
    unittest.main()
