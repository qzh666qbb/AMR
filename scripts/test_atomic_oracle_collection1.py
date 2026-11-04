"""Synthetic negative checks for real-human Oracle evidence gates."""
import unittest
from collect_atomic_alignment_oracle1 import pair_labels, review_errors


class ReviewTests(unittest.TestCase):
    def test_unfilled_or_assistant_packet_rejected(self):
        self.assertIn("human_attestation", review_errors({}, "freeze", "prediction"))
        p = {"reviewer_code": "synthetic-test", "human_attestation": False,
             "independent_attestation": True, "no_model_or_target_answers_used": True,
             "completed_at": "2026-11-02T00:00:00+00:00",
             "freeze_sha256": "freeze", "prediction_seal_sha256": "prediction"}
        self.assertEqual(review_errors(p, "freeze", "prediction"), ["human_attestation"])

    def test_dropped_or_unknown_support_not_negative(self):
        common = [{"candidate_index": 0, "event_id": "e0"}]
        with self.assertRaises(ValueError):
            pair_labels({"pairs": []}, common)
        with self.assertRaises(ValueError):
            pair_labels({"pairs": [{**common[0], "same_fact_supported": None, "rationale": "unknown"}]}, common)

    def test_duplicate_support_rejected(self):
        common = [{"candidate_index": 0, "event_id": "e0"}]
        p = {**common[0], "same_fact_supported": True, "rationale": "synthetic test only"}
        with self.assertRaises(ValueError):
            pair_labels({"pairs": [p, p]}, common)


if __name__ == "__main__":
    unittest.main()
