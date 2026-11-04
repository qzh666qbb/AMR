"""规则与容量测试；手写图不作为检测性能数据。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.core.multiview_alignment import (
    AlignmentConfig, align, event_conflicts, event_groups, surface_conflicts,
)
from src.core.multiview_candidates import candidate_windows, select_budget
from src.core.multiview_views import extract_events
from scripts.run_swan_multiview import require_cache


ROOT = Path(__file__).resolve().parents[1]
GRAPH = '(g / give-01 :ARG0 (a / person :name (n / name :op1 "Alice")) :ARG1 (b / book) :ARG2 (j / person :name (m / name :op1 "Bob")))'


class MultiviewTests(unittest.TestCase):
    def test_real_preflight_offsets_and_candidate_identity(self):
        rows = [json.loads(line) for line in (
            ROOT / "experiments/swan_multiview_development1/inputs/development_rows.jsonl"
        ).read_text().splitlines()]
        expected = [json.loads(line) for line in (
            ROOT / "experiments/swan_multiview_development1/inputs/candidate_rows.jsonl"
        ).read_text().splitlines()]
        for row, reference in zip(rows, expected):
            actual = candidate_windows(row["text"])
            self.assertEqual(actual["candidates"], reference["candidates"])
            for candidate in actual["candidates"]:
                self.assertEqual(row["text"][candidate["start_char"]:candidate["end_char"]],
                                 candidate["text"])
        self.assertEqual(len(rows), 84)

    def test_budget_is_deterministic_and_not_prefix(self):
        text = " ".join(f"word{index}" for index in range(180))
        candidates = candidate_windows(text)["candidates"]
        selected = select_budget(candidates, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected, select_budget(candidates, 12))
        self.assertTrue(any(candidate["start_char"] > len(text) // 2 for candidate in selected))
        self.assertEqual(len({(c["start_char"], c["end_char"]) for c in selected}), 12)

    def test_entity_and_event_details_not_erased(self):
        event = extract_events(GRAPH, 0)[0]
        self.assertEqual(event.predicate, "give-01")
        self.assertEqual(event.entities, ("alice", "bob"))
        self.assertIn("person[Alice]", dict(event.roles)[":ARG0"])
        self.assertIn("book", dict(event.roles)[":ARG1"])

    def test_negation_and_role_exchange_are_conflicts(self):
        first = extract_events(GRAPH, 0)[0]
        negative = extract_events(GRAPH[:-1] + " :polarity -)", 1)[0]
        swapped = extract_events(GRAPH.replace('"Alice"', '"TEMP"').replace(
            '"Bob"', '"Alice"').replace('"TEMP"', '"Bob"'), 2)[0]
        self.assertIn("negation_conflict", event_conflicts(first, negative))
        self.assertIn("role_conflict::ARG0", event_conflicts(first, swapped))
        self.assertEqual(len(event_groups([first, negative, swapped])), 3)

    def test_number_conflict(self):
        left = extract_events("(c / count-01 :ARG1 (b / book :quant 2))", 0)[0]
        right = extract_events("(c / count-01 :ARG1 (b / book :quant 3))", 1)[0]
        self.assertIn("number_conflict", event_conflicts(left, right))

    def test_variable_names_and_inverse_role(self):
        left = extract_events('(l / love-01 :ARG0 (p / person))', 0)[0]
        inverse = extract_events('(x / person :ARG0-of (y / love-01))', 1)[0]
        self.assertEqual(left.signature, inverse.signature)
        self.assertEqual(len(event_groups([left, inverse])), 1)

    def test_surface_explicit_contradiction_and_unknown(self):
        event = extract_events(GRAPH, 0)[0]
        self.assertEqual(surface_conflicts({"clauses": []}, event), [])
        surface = {"clauses": [{"predicate": "give", "negated": True,
                                "roles": {":ARG0": "alice"}}]}
        self.assertIn("surface_negation_conflict", surface_conflicts(surface, event))
        surface["clauses"][0]["negated"] = False
        surface["clauses"][0]["roles"][":ARG0"] = "bob"
        self.assertIn("surface_role_conflict::ARG0", surface_conflicts(surface, event))

    def test_actual_spacy_constraints_and_passive(self):
        from src.core.multiview_views import SurfaceAnalyzer
        surface = SurfaceAnalyzer()
        texts = ["Alice gave Bob two books.", "Bob gave Alice two books.",
                 "Alice did not give Bob two books.", "Two books were given to Bob by Alice.",
                 "Alice gave Bob three books."]
        event = extract_events(
            GRAPH.replace("(b / book)", "(b / book :quant 2)"), 0)[0]
        results = [surface_conflicts(row, event) for row in surface.analyze_many(texts)]
        self.assertEqual(results[0], [])
        self.assertIn("surface_role_conflict::ARG0", results[1])
        self.assertIn("surface_negation_conflict", results[2])
        self.assertEqual(results[3], [])
        self.assertIn("surface_number_conflict", results[4])

    def test_duplicate_observations_do_not_increase_capacity(self):
        candidates = [
            {"start_char": 0, "end_char": 20, "span_sha256": "a"},
            {"start_char": 0, "end_char": 20, "span_sha256": "a"}]
        events = extract_events(GRAPH, 0) + extract_events(GRAPH, 1)
        result = align(candidates, events, np.ones((2, 1)), np.ones((2, 1)),
                       [{"clauses": []}] * 2)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(len(result["selected"]), 1)

    def test_modifier_variant_groups_require_overlap_and_preserve_time(self):
        first = extract_events("(r / read-01 :ARG0 (p / person))", 0)[0]
        variant = extract_events("(r / read-01 :ARG0 (p / person) :mode imperative)", 1)[0]
        candidates = [{"start_char": 0, "end_char": 20}, {"start_char": 5, "end_char": 25}]
        self.assertEqual(len(event_groups([first, variant], candidates)), 1)
        candidates[1] = {"start_char": 50, "end_char": 70}
        self.assertEqual(len(event_groups([first, variant], candidates)), 2)
        timed = extract_events("(r / read-01 :ARG0 (p / person) :time (d / date-entity :year 2026))", 1)[0]
        candidates[1] = {"start_char": 5, "end_char": 25}
        self.assertEqual(len(event_groups([first, timed], candidates)), 2)

    def test_tied_support_uses_shorter_graph_without_bank_feedback(self):
        candidates = [
            {"start_char": 0, "end_char": 20, "span_sha256": "a", "word_count": 4},
            {"start_char": 0, "end_char": 40, "span_sha256": "b", "word_count": 9}]
        events = extract_events(GRAPH, 0) + extract_events(GRAPH, 1)
        result = align(candidates, events, np.ones((2, 1)), np.ones((2, 1)),
                       [{"clauses": []}] * 2)
        self.assertEqual(result["selected"][0]["event_candidate_index"], 0)

    def test_capacity_and_incompatible_events(self):
        candidates = [{"start_char": 0, "end_char": 20, "span_sha256": "a"}]
        events = extract_events(GRAPH, 0) + extract_events(
            GRAPH[:-1] + " :polarity -)", 0)
        result = align(candidates, events, np.ones((1, 1)), np.ones((2, 1)),
                       [{"clauses": []}], AlignmentConfig(max_events_per_candidate=3))
        self.assertEqual(len(result["selected"]), 1)
        independent = events[:1] + extract_events("(r / read-01 :ARG0 (p / person))", 0)
        result = align(candidates, independent, np.ones((1, 1)), np.ones((2, 1)),
                       [{"clauses": []}], AlignmentConfig(max_events_per_candidate=1))
        self.assertEqual(len(result["selected"]), 1)

    def test_missing_view_stays_unmatched(self):
        candidates = [{"start_char": 0, "end_char": 20, "span_sha256": "a"}]
        result = align(candidates, [], np.ones((1, 1)), np.zeros((0, 1)), [{"clauses": []}])
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["unmatched_candidates"], [0])

    def test_cache_only_miss_never_parses(self):
        from src.core.sqlite_cache import AMRCache
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            cache = AMRCache(str(Path(temporary) / "empty.db"))
            with self.assertRaisesRegex(RuntimeError, "cache-only"):
                require_cache(["Alice gave Bob a book."],
                              SimpleNamespace(fingerprint="missing"),
                              SimpleNamespace(cache=cache))


if __name__ == "__main__":
    unittest.main()
