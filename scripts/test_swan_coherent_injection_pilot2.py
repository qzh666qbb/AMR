"""Offline tests for coherent SWAN injection Pilot 2."""
import unittest

import run_swan_coherent_injection_pilot2 as pilot


class DesignTests(unittest.TestCase):
    def test_all_slots_use_target_47(self):
        self.assertEqual(pilot.TEMPLATE_SCHEDULE, (47,) * 10)
        self.assertEqual(len(pilot.SLOT_PURPOSES), 10)

    def test_budget_is_exact(self):
        self.assertEqual(pilot.MAX_CALLS, 303)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 855_914_400)
        self.assertEqual(pilot.PILOT_CAP, (303, 855_914_400))

    def test_story_has_stable_actors_and_progression(self):
        purposes = " ".join(pilot.SLOT_PURPOSES)
        self.assertIn("United Nations", purposes)
        self.assertIn("Maria Lopez", purposes)
        self.assertIn("David Chen", purposes)
        self.assertIn("released", purposes)
        self.assertIn("approved", purposes)

    def test_score_seed_namespace_changes(self):
        old = pilot.canonical_digest(
            ["swan_coherent_injection_pilot1", 0, 0, 47]
        )
        new = pilot.pilot_digest(
            ["swan_coherent_injection_pilot1", 0, 0, 47]
        )
        self.assertNotEqual(old, new)
        self.assertEqual(
            new,
            pilot.canonical_digest(
                ["swan_coherent_injection_pilot2", 0, 0, 47]
            ),
        )

    def test_historical_collision_set_includes_both_pilots(self):
        hashes = pilot.historical_hashes()
        self.assertEqual(len(hashes), 14)


if __name__ == "__main__":
    unittest.main()
