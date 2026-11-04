"""Regression and fail-closed tests for the WP2-v5 external score table."""

import math
from pathlib import Path
import subprocess
import unittest

from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)
from certified_resegmentation_wp2_v3 import SparsePairwiseProblem
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)


CORE = (
    Path(__file__).resolve().parents[1]
    / "build"
    / "wp2_v5"
    / "certified_resegmentation_wp2_v5_core"
)


def frozen_score(left, middle, right):
    return float(
        ((left + 3) * 101 + (middle + 5) * 1009 + (right + 7) * 9176)
        % 100003
    ) / 100003.0


def make_reference():
    scores = {}

    def recording_scorer(left, middle, right):
        value = frozen_score(left, middle, right)
        scores[(left, middle, right)] = value
        return value

    problem = SparsePairwiseProblem(
        token_count=9,
        original_boundaries=(2, 5, 7),
        max_boundary_moves=3,
        span_allowed=lambda start, end: 0 <= start < end <= 9,
        pair_scorer=recording_scorer,
        normalization=3.0,
        bias=0.125,
        threshold=0.25,
    )
    result = exact_grouped_sparse_pairwise_certificate(problem)
    return problem, result, scores


def serialize(problem, score_rows):
    fields = [
        problem.token_count,
        len(problem.original_boundaries),
        problem.max_boundary_moves,
        4,
        1,
        format(float(problem.normalization), ".17g"),
        format(float(problem.bias), ".17g"),
        format(float(problem.threshold), ".17g"),
        *problem.original_boundaries,
        len(score_rows),
    ]
    lines = ["1", " ".join(map(str, fields))]
    lines.extend(
        f"{left} {middle} {right} {value}"
        for left, middle, right, value in score_rows
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def run_core(payload):
    return subprocess.run(
        [str(CORE)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ExternalTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CORE.is_file():
            raise unittest.SkipTest(f"compiled v5 core is absent: {CORE}")
        cls.problem, cls.reference, scores = make_reference()
        cls.rows = [
            (*key, format(value, ".17g"))
            for key, value in sorted(scores.items())
        ]

    def assert_fails(self, rows, expected_message):
        completed = run_core(serialize(self.problem, rows))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(expected_message, completed.stderr.decode("utf-8"))
        self.assertEqual(completed.stdout, b"")

    def test_external_table_matches_grouped_python_reference(self):
        completed = run_core(serialize(self.problem, self.rows))
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8"),
        )
        observed = parse_core_output(completed.stdout, 1)[0]
        self.assertLessEqual(
            abs(
                observed["original_score"]
                - self.reference.original_score
            ),
            1e-12,
        )
        self.assertLessEqual(
            abs(
                observed["certified_score"]
                - self.reference.certified_score
            ),
            1e-12,
        )
        fields = (
            "worst_boundaries",
            "exact_move_histogram",
            "threat_set_size",
            "reachable_state_count",
            "transitions_evaluated",
            "peak_layer_states",
            "candidate_boundary_positions",
            "unique_span_checks",
            "unique_pair_scores",
        )
        for field in fields:
            self.assertEqual(
                observed[field],
                getattr(self.reference, field),
            )

    def test_missing_score_fails_closed(self):
        self.assert_fails(
            self.rows[1:],
            "missing external pair score",
        )

    def test_duplicate_score_fails_closed(self):
        self.assert_fails(
            [*self.rows, self.rows[0]],
            "duplicate external pair score",
        )

    def test_nonfinite_scores_fail_closed(self):
        key = self.rows[0][:3]
        for value in ("nan", "inf"):
            with self.subTest(value=value):
                rows = [(*key, value), *self.rows[1:]]
                completed = run_core(serialize(self.problem, rows))
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, b"")

    def test_out_of_bounds_key_fails_closed(self):
        self.assert_fails(
            [*self.rows, (0, 1, 10, "0.5")],
            "invalid external pair score",
        )

    def test_unused_key_fails_closed(self):
        unused = (0, 1, self.problem.token_count)
        self.assertNotIn(unused, {row[:3] for row in self.rows})
        self.assert_fails(
            [*self.rows, (*unused, "0.5")],
            "unused external pair score",
        )

    def test_test_fixture_contains_only_finite_scores(self):
        self.assertTrue(
            all(math.isfinite(float(row[3])) for row in self.rows)
        )


if __name__ == "__main__":
    unittest.main()
