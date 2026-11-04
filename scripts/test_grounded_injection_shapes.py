"""Triple bounds do not assert that topology or semantics match."""

import unittest

from diagnose_grounded_injection_shapes import triple_bound


class BoundTests(unittest.TestCase):
    def test_equal_counts_only_give_unit_upper_bound(self):
        self.assertEqual(triple_bound([5, 1, 4], [5, 1, 4]), 1.)

    def test_different_sizes_can_force_failure(self):
        self.assertAlmostEqual(triple_bound([20, 5, 19], [4, 2, 3]), 18 / 53)
        self.assertLess(triple_bound([20, 5, 19], [4, 2, 3]), .75)

    def test_types_cannot_be_exchanged(self):
        self.assertAlmostEqual(triple_bound([5, 0, 0], [0, 5, 0]), 0.)
        self.assertEqual(triple_bound([0, 0, 0], [0, 0, 0]), 0.)


if __name__ == "__main__":
    unittest.main()
