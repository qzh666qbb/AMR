"""Offline tests for coherent SWAN injection Pilot 3."""
import unittest

import run_swan_coherent_injection_pilot3 as pilot


class DesignTests(unittest.TestCase):
    def test_schedule_realizations_and_budget_are_exact(self):
        self.assertEqual(pilot.TEMPLATE_SCHEDULE, (47,) * 10)
        self.assertEqual(len(pilot.PREFERRED_REALIZATIONS), 10)
        self.assertEqual(len(set(pilot.PREFERRED_REALIZATIONS)), 10)
        self.assertEqual(pilot.MAX_CALLS, 203)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 573_434_400)

    def test_realizations_form_a_linked_workflow(self):
        text = " ".join(pilot.PREFERRED_REALIZATIONS)
        for value in (
            "Initially",
            "then",
            "subsequently",
            "consequently",
            "Finally",
        ):
            self.assertIn(value, text)
        self.assertIn("United Nations", text)
        self.assertIn("Maria Lopez", text)
        self.assertIn("David Chen", text)

    def test_context_includes_exact_preferred_realization(self):
        prior = list(pilot.PREFERRED_REALIZATIONS[:4])
        context = pilot.context_for(4, prior)
        self.assertIn(pilot.PREFERRED_REALIZATIONS[4], context)
        self.assertNotIn(prior[0], context)
        self.assertIn(prior[1], context)
        self.assertIn(prior[3], context)

    def test_score_seed_namespace_changes(self):
        old = pilot.canonical_digest(
            ["swan_coherent_injection_pilot1", 0, 0, 47]
        )
        new = pilot.pilot_digest(
            ["swan_coherent_injection_pilot1", 0, 0, 47]
        )
        self.assertNotEqual(old, new)

    def test_historical_collision_set_includes_all_prior_acceptances(self):
        self.assertEqual(len(pilot.historical_hashes()), 24)


if __name__ == "__main__":
    unittest.main()
