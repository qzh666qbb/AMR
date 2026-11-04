"""Offline tests for the targeted coherent SWAN injection pilot."""
import json
import unittest

import run_swan_coherent_injection_pilot as pilot
from swan_final1_transform import text_hash


class FrozenDesignTests(unittest.TestCase):
    def test_schedule_and_budget_are_exact(self):
        self.assertEqual(
            [pilot.target_for(slot) for slot in range(10)],
            [47, 47, 27, 11, 14, 43, 25, 37, 27, 11],
        )
        self.assertEqual(pilot.MAX_CALLS, 503)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 1_420_874_400)
        self.assertEqual(pilot.budget()["max_cost_usd"], 1.4208744)

    def test_context_binds_slot_and_only_three_recent_sentences(self):
        previous = [
            "United Nations released a report.",
            "Alice Morgan received the report.",
            "Keep the briefing concise.",
            "Summarize the report about New York for Alice.",
        ]
        context = pilot.context_for(4, previous)
        self.assertNotIn(previous[0], context)
        self.assertIn(previous[1], context)
        self.assertIn(previous[3], context)
        self.assertIn(pilot.FIXED_TOPIC, context)
        self.assertIn(pilot.SLOT_PURPOSES[4], context)

    def test_context_rejects_history_length_drift(self):
        with self.assertRaisesRegex(ValueError, "history"):
            pilot.context_for(2, ["Only one sentence."])


class SurfaceTests(unittest.TestCase):
    def test_accepts_topic_bound_unique_sentence(self):
        result = pilot.surface_result(
            "Keep the briefing concise.",
            ["United Nations released a report."],
            set(),
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["theme_overlap"], ["briefing", "concise"])

    def test_rejects_off_topic_historical_and_repetitive_text(self):
        off_topic = pilot.surface_result(
            "Keep the device simple.",
            [],
            set(),
        )
        self.assertFalse(off_topic["theme_vocabulary_overlap"])
        historical_text = "Alice Morgan received the report."
        historical = pilot.surface_result(
            historical_text,
            [],
            {text_hash(historical_text)},
        )
        self.assertFalse(historical["not_historical_strict_acceptance"])
        repetitive = pilot.surface_result(
            "Keep the briefing concise.",
            ["Keep the briefing concise."],
            set(),
        )
        self.assertFalse(repetitive["prior_token_jaccard_at_most_point8"])

    def test_rejects_bad_length_segmentation_and_punctuation(self):
        self.assertFalse(
            pilot.surface_result("Brief report.", [], set())["word_count_3_18"]
        )
        self.assertFalse(
            pilot.surface_result(
                "Alice reviewed the report. Bob approved it.",
                [],
                set(),
            )["one_punkt_unit"]
        )
        self.assertFalse(
            pilot.surface_result(
                "Alice reviewed the report",
                [],
                set(),
            )["terminal_punctuation"]
        )


class ReviewTests(unittest.TestCase):
    GOOD = {
        "coherence": 4,
        "naturalness": 5,
        "nonredundancy": 4,
        "task_consistency": 5,
        "severe_problem": False,
    }

    def test_exact_valid_review_passes(self):
        parsed = pilot.parse_review(json.dumps(self.GOOD))
        self.assertEqual(parsed, self.GOOD)
        self.assertTrue(pilot.review_pass(parsed))

    def test_low_score_or_severe_problem_fails(self):
        low = {**self.GOOD, "coherence": 3}
        severe = {**self.GOOD, "severe_problem": True}
        self.assertFalse(pilot.review_pass(pilot.parse_review(json.dumps(low))))
        self.assertFalse(pilot.review_pass(pilot.parse_review(json.dumps(severe))))

    def test_extra_key_wrong_type_and_markdown_are_invalid(self):
        extra = {**self.GOOD, "reason": "extra"}
        wrong = {**self.GOOD, "coherence": True}
        self.assertIsNone(pilot.parse_review(json.dumps(extra)))
        self.assertIsNone(pilot.parse_review(json.dumps(wrong)))
        self.assertIsNone(
            pilot.parse_review("```json\n" + json.dumps(self.GOOD) + "\n```")
        )


if __name__ == "__main__":
    unittest.main()
