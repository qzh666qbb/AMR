from __future__ import annotations

import math
import re
import unittest

from src.core.elastic_v2 import (
    CountStatePathOptimizer,
    ElasticBoundaryProposerV2,
    SwanBankLatticeDetectorV2,
    recovery_edge_validity,
)
from src.core.models import BankMatch, Boundary, Lattice, LatticeEdge
from src.core.sqlite_cache import normalize_text, text_sha256


def match(score: float) -> BankMatch:
    return BankMatch(
        template_index=0,
        score=score,
        concept_score=score,
        relation_score=score,
        path_score=score,
    )


def small_lattice() -> Lattice:
    return Lattice(
        text="alpha beta",
        boundaries=(
            Boundary(index=0, char_offset=0, kind="terminal"),
            Boundary(index=1, char_offset=5, kind="test"),
            Boundary(index=2, char_offset=10, kind="terminal"),
        ),
        edges=(
            LatticeEdge(0, 1, "alpha", 1),
            LatticeEdge(1, 2, "beta", 1),
            LatticeEdge(0, 2, "alpha beta", 2),
        ),
    )


def recovery_records(lattice: Lattice, error_span: str = "") -> dict:
    records = {}
    for edge in lattice.edges:
        records[text_sha256(normalize_text(edge.span_text))] = {
            "selected_beam": 0,
            "events": (
                [{
                    "level": "ERROR",
                    "beam_index": 0,
                    "message": "synthetic selected-beam error",
                }]
                if edge.span_text == error_span
                else []
            ),
        }
    return records


class StaticScorer:
    def __init__(self, scores):
        self.scores = scores

    def score_many(self, texts):
        return tuple(match(self.scores[text]) for text in texts)


class ElasticBoundaryProposerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposer = ElasticBoundaryProposerV2()

    def test_five_punctuation_forms_retain_internal_candidates(self):
        forms = (
            "Alpha does one two three. Beta does four five six. "
            "Gamma does seven eight nine.",
            "Alpha does one two three; Beta does four five six; "
            "Gamma does seven eight nine.",
            "Alpha does one two three, Beta does four five six, "
            "Gamma does seven eight nine.",
            "Alpha does one two three\nBeta does four five six\n"
            "Gamma does seven eight nine.",
            "Alpha does one two three Beta does four five six "
            "Gamma does seven eight nine.",
        )
        for text in forms:
            with self.subTest(text=text):
                lattice = self.proposer.build_lattice(text)
                tokens = tuple(re.finditer(r"\w+", text))
                token_boundaries = {
                    sum(token.end() <= boundary.char_offset for token in tokens)
                    for boundary in lattice.boundaries
                }
                self.assertTrue({5, 10}.issubset(token_boundaries))
                self.assertTrue(
                    all(
                        any(
                            edge.start == index and edge.end == index + 1
                            for edge in lattice.edges
                        )
                        for index in range(lattice.node_count - 1)
                    )
                )

    def test_capitalized_comma_fallback_proposes_before_vocative(self):
        text = (
            "The question concerns Alice Smith "
            "Alice, question the concept by Alice Smith "
            "Alice, question the concept by Alice Smith."
        )
        boundaries = self.proposer.candidate_boundaries(text)
        offsets = {boundary.char_offset for boundary in boundaries}
        second_alice = text.index("Alice,", text.index("Alice,") + 1)
        self.assertIn(second_alice, offsets)

    def test_long_unpunctuated_text_has_bounded_full_coverage_path(self):
        text = " ".join(f"word{i}" for i in range(205))
        lattice = self.proposer.build_lattice(text)
        adjacent = {(edge.start, edge.end) for edge in lattice.edges}
        self.assertTrue(
            all(
                (index, index + 1) in adjacent
                for index in range(lattice.node_count - 1)
            )
        )
        self.assertTrue(
            all(len(edge.span_text.split()) <= 80 for edge in lattice.edges)
        )
        self.assertGreater(lattice.node_count, 2)

    def test_empty_text_has_single_terminal_boundary(self):
        lattice = self.proposer.build_lattice("  ")
        self.assertEqual(lattice.node_count, 1)
        self.assertEqual(lattice.edges, ())


class CountStatePathOptimizerTests(unittest.TestCase):
    def test_optimizes_final_normalized_statistic_not_raw_utility(self):
        result = CountStatePathOptimizer(score_center=0.65).optimize(
            small_lattice(),
            (match(0.85), match(0.85), match(0.95)),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unit_count, 1)
        self.assertAlmostEqual(result.raw_utility, 0.30)
        self.assertAlmostEqual(result.statistic, 0.30)
        self.assertEqual(result.units[0].edge, small_lattice().edges[2])

    def test_invalid_edge_is_removed_before_optimization(self):
        result = CountStatePathOptimizer(score_center=0.65).optimize(
            small_lattice(),
            (match(0.85), match(0.85), match(1.0)),
            valid_edges=(True, True, False),
        )
        self.assertFalse(result.abstained)
        self.assertEqual(result.unit_count, 2)
        self.assertEqual(result.invalid_edge_count, 1)
        self.assertAlmostEqual(
            result.statistic,
            0.40 / math.sqrt(2),
        )

    def test_no_complete_valid_path_abstains(self):
        result = CountStatePathOptimizer().optimize(
            small_lattice(),
            (match(0.9), match(0.9), match(0.9)),
            valid_edges=(False, False, False),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstain_reason, "no_complete_valid_path")
        self.assertIsNone(result.statistic)
        self.assertEqual(result.invalid_edge_count, 3)

    def test_count_state_dp_matches_complete_path_enumeration(self):
        lattice = Lattice(
            text="a b c d",
            boundaries=tuple(
                Boundary(i, i * 2, "test" if i not in (0, 4) else "terminal")
                for i in range(5)
            ),
            edges=(
                LatticeEdge(0, 1, "a", 1),
                LatticeEdge(0, 2, "a b", 2),
                LatticeEdge(1, 2, "b", 1),
                LatticeEdge(1, 3, "b c", 2),
                LatticeEdge(2, 3, "c", 1),
                LatticeEdge(2, 4, "c d", 2),
                LatticeEdge(3, 4, "d", 1),
            ),
        )
        scores = (0.92, 0.68, 0.81, 0.97, 0.72, 0.88, 0.76)
        matches = tuple(match(score) for score in scores)
        outgoing = {}
        for edge, bank_match in zip(lattice.edges, matches):
            outgoing.setdefault(edge.start, []).append((edge, bank_match))

        paths = []

        def enumerate_paths(node, edges, utility):
            if node == lattice.node_count - 1:
                paths.append((utility / math.sqrt(len(edges)), tuple(edges)))
                return
            for edge, bank_match in outgoing.get(node, ()):
                enumerate_paths(
                    edge.end,
                    edges + [edge],
                    utility + bank_match.score - 0.65,
                )

        enumerate_paths(0, [], 0.0)
        oracle_statistic, oracle_edges = max(paths, key=lambda item: item[0])
        result = CountStatePathOptimizer(score_center=0.65).optimize(
            lattice,
            matches,
        )
        self.assertAlmostEqual(result.statistic, oracle_statistic)
        self.assertEqual(
            tuple(unit.edge for unit in result.units),
            oracle_edges,
        )

    def test_input_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "matches"):
            CountStatePathOptimizer().optimize(
                small_lattice(),
                (match(0.9),),
            )

    def test_recovery_binding_rejects_selected_beam_error(self):
        lattice = small_lattice()
        validity = recovery_edge_validity(
            lattice,
            recovery_records(lattice, error_span="alpha beta"),
        )
        self.assertEqual(validity, (True, True, False))
        detector = SwanBankLatticeDetectorV2(
            builder=ElasticBoundaryProposerV2(),
            scorer=StaticScorer({
                "alpha": 0.85,
                "beta": 0.85,
                "alpha beta": 1.0,
            }),
        )
        result = detector.detect_lattice_with_recovery(
            lattice,
            recovery_records(lattice, error_span="alpha beta"),
        )
        self.assertEqual(result.unit_count, 2)
        self.assertEqual(result.invalid_edge_count, 1)

    def test_missing_recovery_evidence_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "lacks bound"):
            recovery_edge_validity(small_lattice(), {})


if __name__ == "__main__":
    unittest.main()
