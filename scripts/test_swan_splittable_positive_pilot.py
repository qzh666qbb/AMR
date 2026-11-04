"""Offline tests for the splittable SWAN-positive sentence pilot."""
import unittest

import run_swan_splittable_positive_pilot as pilot


class CompleteSyntax:
    def __call__(self, text):
        return {
            "complete": True,
            "capitalization_valid_mid_sentence": True,
            "content_lemmas": ["aster", "plan"],
        }


class SurfaceTests(unittest.TestCase):
    def test_accepts_exact_semicolon_shape(self):
        text = (
            "Aster Labs approved the plan; "
            "Aster Labs confirmed the plan."
        )
        result = pilot.surface_result(text, "Aster Labs", CompleteSyntax())
        self.assertTrue(result["pass"])
        self.assertEqual(result["word_count"], 10)
        self.assertEqual(len(result["split_candidates"]), 1)

    def test_rejects_comma_and_wrong_subject(self):
        text = (
            "Aster Labs approved the plan; "
            "Meridian Press confirmed it, today."
        )
        result = pilot.surface_result(text, "Aster Labs", CompleteSyntax())
        self.assertFalse(result["pass"])
        self.assertFalse(result["no_comma"])
        self.assertFalse(result["right_subject_exact"])


class FrozenDesignTests(unittest.TestCase):
    def test_template_schedule_is_fixed_and_rotated(self):
        self.assertEqual(len(pilot.schedule_for(0)), 50)
        self.assertEqual(pilot.schedule_for(0)[:5], [45] * 5)
        self.assertEqual(pilot.schedule_for(1)[:5], [16] * 5)
        self.assertEqual(
            sorted(set(pilot.schedule_for(3))),
            sorted(pilot.TEMPLATE_SCHEDULE),
        )

    def test_budget_cap(self):
        self.assertEqual(pilot.MAX_CALLS, 201)
        self.assertEqual(pilot.PILOT_CAP, (201, 567_784_800))
        self.assertEqual(pilot.budget()["max_cost_usd"], 0.5677848)


if __name__ == "__main__":
    unittest.main()
