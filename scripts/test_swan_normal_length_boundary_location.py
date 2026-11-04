"""Offline tests for the normal-length boundary-location pilot."""
import json
import unittest

import run_swan_normal_length_boundary_location as pilot


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = pilot.build_inputs()

    def test_standard_c2_fails_without_natural_split(self):
        construction = self.inputs["construction"]
        self.assertEqual(construction["merge_candidate_count"], 4)
        self.assertEqual(construction["split_candidate_count"], 0)
        self.assertFalse(construction["standard_C2_constructed"])
        self.assertEqual(
            construction["standard_C2_failure"],
            "no_eligible_internal_split",
        )

    def test_location_pair_is_nonadjacent_and_count_neutral(self):
        construction = self.inputs["construction"]
        left, right = construction["selected_boundaries"]
        self.assertGreater(abs(left - right), 1)
        r0 = self.inputs["conditions"]["R0"]
        r1 = self.inputs["conditions"]["R1"]
        self.assertEqual(r0["num_detection_units"], 19)
        self.assertEqual(r1["num_detection_units"], 19)
        self.assertEqual(r0["word_tokens"], r1["word_tokens"])
        self.assertEqual(r0["punctuation"], r1["punctuation"])
        self.assertEqual(r0["character_length"], r1["character_length"])
        self.assertNotEqual(r0["text_sha256"], r1["text_sha256"])

    def test_restoration_is_byte_exact(self):
        self.assertEqual(
            self.inputs["conditions"]["C0"]["text"],
            self.inputs["conditions"]["R4"]["text"],
        )
        self.assertEqual(
            self.inputs["conditions"]["C0"]["text_sha256"],
            self.inputs["conditions"]["R4"]["text_sha256"],
        )


class ReviewTests(unittest.TestCase):
    GOOD = {
        "meaning_preserved": True,
        "same_events_and_roles": True,
        "left_naturalness": 3,
        "right_naturalness": 3,
        "severe_problem": False,
    }

    def test_exact_review_schema_passes(self):
        parsed = pilot.parse_review(json.dumps(self.GOOD))
        self.assertEqual(parsed, self.GOOD)
        self.assertTrue(pilot.review_pass(parsed))

    def test_semantic_or_naturalness_failure_rejects(self):
        self.assertFalse(pilot.review_pass({
            **self.GOOD,
            "meaning_preserved": False,
        }))
        self.assertFalse(pilot.review_pass({
            **self.GOOD,
            "left_naturalness": 2,
        }))
        self.assertIsNone(pilot.parse_review(json.dumps({
            **self.GOOD,
            "extra": 1,
        })))


if __name__ == "__main__":
    unittest.main()
