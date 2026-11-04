"""Offline tests for coherent SWAN injection Pilot 4."""
import unittest

import run_swan_coherent_injection_pilot4 as pilot


class DesignTests(unittest.TestCase):
    GOOD = {
        "coherence": 4,
        "naturalness": 3,
        "nonredundancy": 4,
        "task_consistency": 4,
        "severe_problem": False,
    }

    def test_schedule_realizations_and_budget_are_exact(self):
        self.assertEqual(pilot.TEMPLATE_SCHEDULE, (47,) * 10)
        self.assertEqual(len(pilot.PREFERRED_REALIZATIONS), 10)
        self.assertEqual(len(set(pilot.PREFERRED_REALIZATIONS)), 10)
        self.assertEqual(pilot.MAX_CALLS, 203)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 573_434_400)

    def test_event_log_has_stable_actors_and_distinct_stages(self):
        text = " ".join(pilot.PREFERRED_REALIZATIONS)
        for value in ("United Nations", "Sarah Kim", "James Lee"):
            self.assertIn(value, text)
        for value in ("released", "reviewed", "revised", "approved"):
            self.assertIn(value, text)

    def test_context_includes_preferred_realization_and_recent_history(self):
        prior = list(pilot.PREFERRED_REALIZATIONS[:4])
        context = pilot.context_for(4, prior)
        self.assertIn(pilot.PREFERRED_REALIZATIONS[4], context)
        self.assertNotIn(prior[0], context)
        self.assertIn(prior[1], context)
        self.assertIn(prior[3], context)

    def test_event_log_review_thresholds(self):
        self.assertTrue(pilot.review_pass(self.GOOD))
        self.assertFalse(pilot.review_pass({**self.GOOD, "coherence": 3}))
        self.assertFalse(pilot.review_pass({**self.GOOD, "naturalness": 2}))
        self.assertFalse(
            pilot.review_pass({**self.GOOD, "severe_problem": True})
        )

    def test_historical_collision_set_is_complete(self):
        self.assertEqual(len(pilot.historical_hashes()), 24)


if __name__ == "__main__":
    unittest.main()
