"""Focused tests for the v2 WP2 runner and independent auditor."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_semstamp_certified_defense_v2_wp2 as auditor
import run_semstamp_certified_defense_v2_wp2 as runner


class RunnerTests(unittest.TestCase):
    def test_synthetic_gap_generator_authorizes_every_zero(self):
        for case_index in range(100):
            original = (1, 3, 5)
            lengths = runner.gap_lengths_for(7, original, case_index)
            zeros = {
                boundary
                for boundary, length in enumerate(lengths, start=1)
                if length == 0
            }
            self.assertTrue(zeros.issubset(original))

    def test_disclosed_sample_indices_are_exact(self):
        self.assertEqual(len(runner.SYSTEMATIC_SAMPLE_INDICES), 32)
        self.assertEqual(len(runner.RANDOM_SAMPLE_INDICES), 32)


class IndependentAuditorTests(unittest.TestCase):
    def deletion_descriptor(self):
        descriptor = {
            "case_id": "zero-deletion",
            "atom_count": 5,
            "original_boundaries": [1, 3],
            "gap_byte_lengths": [0, 1, 1, 2],
            "radius": 1,
            "allowed_spans": [
                [0, 1], [1, 3], [3, 5], [0, 3],
            ],
            "hit_triples": [[0, 1, 3], [1, 3, 5]],
            "threshold": None,
        }
        return descriptor

    def test_independent_enumerator_handles_zero_gap_deletion(self):
        result = auditor.independently_solve_descriptor(
            self.deletion_descriptor()
        )
        self.assertEqual(result["worst_boundaries"], [3])
        self.assertEqual((result["insertions"], result["deletions"]), (0, 1))
        self.assertEqual(
            result["refined_topology"]["zero_byte_gap_boundaries"], [1]
        )

    def test_independent_enumerator_rejects_unauthorized_zero(self):
        descriptor = self.deletion_descriptor()
        descriptor["gap_byte_lengths"] = [1, 0, 1, 2]
        with self.assertRaisesRegex(
            AssertionError, "unauthorized zero gap"
        ):
            auditor.independently_solve_descriptor(descriptor)

    def test_cross_edge_count_comparison_is_exact(self):
        self.assertEqual(
            auditor.compare_scores((0, 2), (1, 8)),
            0,
        )


if __name__ == "__main__":
    unittest.main()
