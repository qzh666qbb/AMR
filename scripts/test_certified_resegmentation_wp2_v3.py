"""Offline tests for the WP2-v3 typed-list adapter."""

from itertools import combinations
import random
import re
import unittest

from certified_resegmentation_wp2_v3 import canonicalize_structured
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    brute_force_sparse_pairwise_certificate,
    exact_sparse_pairwise_certificate,
    rank_candidate_boundaries,
)


_ATOM_RE = re.compile(r"\S+")


class StructuredAdapterTests(unittest.TestCase):
    def test_edge_whitespace_p0_round_trips_byte_exactly(self):
        units = ["  aa  bb  ", "cc dd", "\tee ff\n"]
        document = canonicalize_structured(units)
        self.assertEqual(document.original_boundaries, (2, 4))
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            tuple(units),
        )
        self.assertEqual(
            document.preprocess_partition("semstamp"),
            tuple(units),
        )
        self.assertEqual(
            document.preprocess_partition("simmark"),
            (units[0], " " + units[1], " " + units[2]),
        )

    def test_missing_raw_whitespace_uses_virtual_separator_when_merged(self):
        document = canonicalize_structured(["aa bb", "cc dd", "ee ff"])
        self.assertEqual(
            document.render_partition((1, 4)),
            ("aa", " bb cc dd", "ee ff"),
        )
        self.assertIn("bb cc", document.render_partition((1, 4))[1])
        self.assertEqual(
            document.render_partition((2, 4)),
            ("aa bb", "cc dd", "ee ff"),
        )

    def test_new_boundary_right_attributes_existing_whitespace(self):
        document = canonicalize_structured([" aa \t bb ", " cc dd "])
        self.assertEqual(
            document.render_partition((1,)),
            (" aa", " \t bb   cc dd "),
        )
        self.assertEqual(
            document.partition_atoms((1,)),
            ("aa", "bb", "cc", "dd"),
        )

    def test_string_splitter_is_called_once(self):
        calls = []

        def splitter(value):
            calls.append(value)
            return ["First sentence.", "Second sentence."]

        document = canonicalize_structured(
            "First sentence. Second sentence.", splitter
        )
        self.assertEqual(calls, ["First sentence. Second sentence."])
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            ("First sentence.", "Second sentence."),
        )

    def test_all_small_partitions_preserve_atoms(self):
        document = canonicalize_structured(
            [" a  b c", "d e ", "\tf g h "]
        )
        for boundary_count in range(document.atom_count):
            for boundaries in combinations(
                range(1, document.atom_count), boundary_count
            ):
                self.assertEqual(
                    document.partition_atoms(boundaries),
                    document.atom_payloads,
                )

    def test_random_whitespace_layouts_round_trip(self):
        rng = random.Random(90210)
        whitespace = ("", " ", "  ", "\t", "\n")
        for _ in range(200):
            units = []
            for unit_index in range(rng.randint(2, 7)):
                atoms = [
                    f"u{unit_index}a{atom_index}"
                    for atom_index in range(rng.randint(1, 6))
                ]
                unit = rng.choice(whitespace)
                unit += "".join(
                    atom + (
                        rng.choice(whitespace[1:])
                        if index < len(atoms) - 1
                        else ""
                    )
                    for index, atom in enumerate(atoms)
                )
                unit += rng.choice(whitespace)
                units.append(unit)
            document = canonicalize_structured(units)
            self.assertEqual(
                document.render_partition(document.original_boundaries),
                tuple(units),
            )
            self.assertEqual(
                document.partition_atoms(document.original_boundaries),
                tuple(
                    atom
                    for unit in units
                    for atom in _ATOM_RE.findall(unit)
                ),
            )

    def test_invalid_units_and_boundaries_fail_closed(self):
        with self.assertRaises(ValueError):
            canonicalize_structured([])
        with self.assertRaises(ValueError):
            canonicalize_structured(["aa", "  "])
        with self.assertRaises(TypeError):
            canonicalize_structured(["aa", 3])
        document = canonicalize_structured(["aa bb", "cc dd"])
        for boundaries in ((0,), (4,), (2, 2), (3, 2), (True,)):
            with self.subTest(boundaries=boundaries):
                with self.assertRaises((TypeError, ValueError)):
                    document.render_partition(boundaries)


class SparsePairwiseTests(unittest.TestCase):
    def test_rank_candidates_equal_exhaustive_projection(self):
        for atom_count in range(4, 10):
            gaps = tuple(range(1, atom_count))
            for boundary_count in range(1, min(4, atom_count - 1) + 1):
                for original in combinations(gaps, boundary_count):
                    for radius in range(boundary_count + 1):
                        expected = [set() for _ in range(boundary_count)]
                        for candidate in combinations(gaps, boundary_count):
                            if len(set(candidate) - set(original)) <= radius:
                                for index, boundary in enumerate(candidate):
                                    expected[index].add(boundary)
                        observed = rank_candidate_boundaries(
                            atom_count, original, radius
                        )
                        self.assertEqual(
                            observed,
                            tuple(
                                tuple(sorted(values)) for values in expected
                            ),
                        )

    def test_sparse_dp_matches_oracle_with_ties(self):
        problem = SparsePairwiseProblem(
            token_count=9,
            original_boundaries=(2, 5, 7),
            max_boundary_moves=3,
            span_allowed=lambda start, end: end - start <= 5,
            pair_scorer=lambda left, middle, right: 0.0,
        )
        exact = exact_sparse_pairwise_certificate(problem)
        oracle = brute_force_sparse_pairwise_certificate(problem)
        self.assertEqual(exact.certified_score, oracle.certified_score)
        self.assertEqual(exact.worst_boundaries, oracle.worst_boundaries)
        self.assertEqual(
            exact.exact_move_histogram,
            oracle.exact_move_histogram,
        )
        self.assertEqual(exact.threat_set_size, oracle.threat_set_size)

    def test_random_sparse_dp_matches_oracle(self):
        rng = random.Random(731)
        for _ in range(500):
            atom_count = rng.randint(4, 11)
            boundary_count = rng.randint(
                1, min(4, atom_count - 1)
            )
            original = tuple(sorted(rng.sample(
                range(1, atom_count), boundary_count
            )))
            radius = rng.randint(0, min(3, boundary_count))
            original_spans = set(zip(
                (0, *original),
                (*original, atom_count),
            ))
            legal = {
                (start, end): (
                    (start, end) in original_spans
                    or rng.random() >= 0.25
                )
                for start in range(atom_count)
                for end in range(start + 1, atom_count + 1)
            }

            def scorer(left, middle, right):
                return float(
                    (
                        (left + 3) * 1009
                        + (middle + 5) * 9176
                        + (right + 7) * 65537
                    )
                    % 1000003
                )

            problem = SparsePairwiseProblem(
                token_count=atom_count,
                original_boundaries=original,
                max_boundary_moves=radius,
                span_allowed=lambda start, end: legal[(start, end)],
                pair_scorer=scorer,
                threshold=17.0,
            )
            exact = exact_sparse_pairwise_certificate(problem)
            oracle = brute_force_sparse_pairwise_certificate(problem)
            self.assertEqual(
                exact.exact_move_histogram,
                oracle.exact_move_histogram,
            )
            self.assertEqual(exact.threat_set_size, oracle.threat_set_size)
            self.assertAlmostEqual(
                exact.original_score, oracle.original_score, places=12
            )
            self.assertAlmostEqual(
                exact.certified_score, oracle.certified_score, places=12
            )
            self.assertEqual(
                exact.worst_boundaries,
                oracle.worst_boundaries,
            )


if __name__ == "__main__":
    unittest.main()
