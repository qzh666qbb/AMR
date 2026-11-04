"""Offline tests for the normal-length coherent SWAN pilot."""
import unittest
from unittest import mock

import run_swan_normal_length_coherent_pilot as pilot
from swan_final1_transform import text_hash


class DesignTests(unittest.TestCase):
    GOOD_REVIEW = {
        "coherence": 4,
        "naturalness": 3,
        "nonredundancy": 4,
        "task_consistency": 4,
        "severe_problem": False,
    }

    def test_schedule_and_budget_are_exact(self):
        self.assertEqual(pilot.TEMPLATE_SCHEDULE, (47,) * 20)
        self.assertEqual(len(pilot.SLOT_PURPOSES), 20)
        self.assertEqual(pilot.MAX_CALLS, 603)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 1_703_354_400)

    def test_context_binds_event_and_recent_history(self):
        previous = [
            f"Laura Chen reviewed report stage {index}."
            for index in range(4)
        ]
        context = pilot.context_for(4, previous)
        self.assertNotIn(previous[0], context)
        self.assertIn(previous[1], context)
        self.assertIn(previous[3], context)
        self.assertIn(pilot.SLOT_PURPOSES[4], context)
        self.assertIn("EVENT 5/20", context)

    def test_surface_uses_five_to_nine_word_gate(self):
        patches = {
            "MIN_WORDS": pilot.MIN_WORDS,
            "MAX_WORDS": pilot.MAX_WORDS,
            "THEME_TOKENS": pilot.THEME_TOKENS,
        }
        with mock.patch.multiple(pilot.base, **patches):
            accepted = pilot.surface_result(
                "Laura Chen reviewed the report.",
                [],
                set(),
            )
            rejected = pilot.surface_result(
                "Laura Chen reviewed it.",
                [],
                set(),
            )
        self.assertTrue(accepted["word_count_5_9"])
        self.assertTrue(accepted["pass"])
        self.assertFalse(rejected["word_count_5_9"])
        self.assertFalse(rejected["pass"])

    def test_document_requires_twenty_units_and_one_hundred_words(self):
        slots = []
        for index in range(20):
            text = f"Laura Chen reviewed report stage{index}."
            slots.append({
                "accepted": True,
                "text": text,
                "text_sha256": text_hash(text),
            })
        document = pilot.document_from_slots(slots)
        self.assertEqual(document["word_count"], 100)
        self.assertEqual(len(document["segmentation_spans"]), 20)

    def test_review_threshold_and_historical_collision_set(self):
        self.assertTrue(pilot.review_pass(self.GOOD_REVIEW))
        self.assertFalse(
            pilot.review_pass({**self.GOOD_REVIEW, "coherence": 3})
        )
        self.assertEqual(len(pilot.historical_hashes()), 24)


if __name__ == "__main__":
    unittest.main()
