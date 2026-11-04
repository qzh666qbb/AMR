"""Focused standard-library tests for the evidence audit."""

import math
from pathlib import Path
import struct
import tempfile
import unittest

from audit_feedback_evidence import (
    exact_budget, feedback_interval, group_candidates, policy_success, read_vector, z_score,
)


def row(candidate, quality, score, index=0):
    return {"index": index, "candidate": candidate, "quality_valid": quality,
            "z_score": score, "source": "same source"}


class AuditTests(unittest.TestCase):
    def test_single_sentence_ceiling(self):
        self.assertAlmostEqual(z_score(1, 1), math.sqrt(3))
        self.assertLess(z_score(1, 1), 2.33)
        self.assertGreater(z_score(2, 2), 2.33)

    def test_invalid_counts(self):
        for counts in ((0, 0), (1, 2), (2, -1)):
            with self.assertRaises(ValueError):
                z_score(*counts)

    def test_quality_and_feedback_are_distinct(self):
        pool = [row(0, False, -2), row(1, True, 0), row(2, True, 3), row(3, False, 4)]
        self.assertEqual(policy_success(pool, "random"), .25)
        self.assertEqual(policy_success(pool, "quality_only"), .5)
        self.assertEqual(policy_success(pool, "quality_then_min_z"), 1)
        self.assertEqual(policy_success(pool, "min_z"), 0)

    def test_no_quality_candidate_is_failure(self):
        pool = [row(i, False, -1) for i in range(4)]
        for policy in ("random", "quality_only", "min_z", "quality_then_min_z"):
            self.assertEqual(policy_success(pool, policy), 0)

    def test_threshold_equality_and_tie_break(self):
        pool = [row(1, False, 2.33), row(0, True, 2.33)]
        self.assertEqual(policy_success(pool, "min_z"), 1)
        self.assertEqual(policy_success(list(reversed(pool)), "min_z"), 1)

    def test_exact_enumeration(self):
        groups = group_candidates([row(i, True, 0 if i == 0 else 3) for i in range(4)], 4)
        result = exact_budget(groups)
        for k in range(1, 5):
            self.assertAlmostEqual(result[str(k)]["random"], .25)
            self.assertAlmostEqual(result[str(k)]["quality_only"], .25)
            self.assertAlmostEqual(result[str(k)]["quality_then_min_z"], k / 4)
        self.assertEqual(feedback_interval(groups), feedback_interval(groups))

    def test_group_validation(self):
        good = [row(i, True, 0) for i in range(4)]
        self.assertEqual(len(group_candidates(good, 4)), 1)
        for bad in (good[:3], good + good[:1], [row(i, True, float("nan")) for i in range(4)]):
            with self.assertRaises(ValueError):
                group_candidates(bad, 4)
        bad = [dict(r) for r in good]
        bad[0]["source"] = "different"
        with self.assertRaises(ValueError):
            group_candidates(bad, 4)

    def test_npy_float_vector_and_bad_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npy"
            for version, header_fmt in ((1, "<H"), (2, "<I")):
                header = repr({"descr": "<f8", "fortran_order": False, "shape": (2,)}).encode()
                path.write_bytes(b"\x93NUMPY" + bytes([version, 0])
                                 + struct.pack(header_fmt, len(header)) + header
                                 + struct.pack("<2d", -1.0, 2.33))
                self.assertEqual(read_vector(path), [-1.0, 2.33])
            path.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                read_vector(path)


if __name__ == "__main__":
    unittest.main()
