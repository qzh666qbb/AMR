"""Scorer boundary tests; fake pair scores are not empirical detection results."""
import unittest
from types import SimpleNamespace
import warnings
from unittest.mock import patch

from src.core.embedding_amr_official import OfficialBankScorer
from src.core.swan_blind_protocol import digest


class OfficialScorerTests(unittest.TestCase):
    def scorer(self, function):
        scorer = object.__new__(OfficialBankScorer)
        scorer.templates = ("bank-a", "bank-b", "bank-c")
        scorer.matcher = SimpleNamespace(
            compute_s2match_from_strings=function, ERROR_LOG=None, DEBUG_LOG=None)
        return scorer

    def test_all_bank_entries_scored_after_exact_match(self):
        calls = []
        def pair(graph, template, **kwargs):
            calls.append(template)
            return (1., 1., 1.)
        scores = self.scorer(pair)._scores("graph")
        self.assertEqual(calls, ["bank-a", "bank-b", "bank-c"])
        self.assertEqual(scores, [1., 1., 1.])

    def test_warning_and_nonfinite_scores_fail_closed(self):
        for value in (float("nan"), float("inf"), -1, 1.1):
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                self.scorer(lambda *a, **k: (value, value, value))._scores("graph")
        def pair(*args, **kwargs):
            warnings.warn("bound test warning", RuntimeWarning)
            return (1, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "diagnostic"):
            self.scorer(pair)._scores("graph")

    def test_seed_uses_frozen_official_serialization(self):
        from src.core.swan_blind_runtime import fixed_seed
        seeds = []
        def record(seed):
            seeds.append(seed)
            return fixed_seed(seed)
        with patch("src.core.embedding_amr_official.fixed_seed", record):
            self.scorer(lambda *a, **k: (1, 1, 1))._scores("graph")
        self.assertEqual(seeds, [int(digest(["sha256-graph-bank-v1", "graph", template])[:8], 16)
                                 for template in ("bank-a", "bank-b", "bank-c")])

    def test_raw_graph_only_normalized_at_bank_boundary(self):
        scorer = self.scorer(lambda *a, **k: (1, 1, 1))
        normalized_inputs, scored_inputs = [], []
        def normalize(raw):
            normalized_inputs.append(raw)
            return {"graph": "normalized", "normalization_ok": True}
        def score(graph):
            scored_inputs.append(graph)
            return [.2, .9, .7]
        scorer.normalizer, scorer._scores = normalize, score
        scorer.memo, scorer.audit, scorer.exact_replays = {}, {}, 0
        result = scorer.score_amr("raw_graph_with_roles")
        self.assertEqual(normalized_inputs, ["raw_graph_with_roles"] * 2)
        self.assertEqual(scored_inputs, ["normalized"] * 2)
        self.assertEqual((result.template_index, result.score), (1, .9))
        self.assertEqual(len(scorer.audit), 1)
        self.assertEqual(scorer.score_amr("raw_graph_with_roles"), result)
        self.assertEqual(len(scored_inputs), 2)

    def test_normalization_failure_excludes_candidate_and_replay_must_match(self):
        scorer = self.scorer(lambda *a, **k: (1, 1, 1))
        scorer.memo, scorer.audit, scorer.exact_replays = {}, {}, 0
        scorer.normalizer = lambda raw: {"graph": None, "normalization_ok": False, "events": []}
        self.assertIsNone(scorer.score_amr("raw"))
        self.assertFalse(next(iter(scorer.audit.values()))["normalization_ok"])
        self.assertIsNone(scorer.score_amr("raw"))
        self.assertEqual(len(scorer.audit), 1)
        scorer.normalizer = lambda raw: {"graph": "normalized", "normalization_ok": True}
        scores = iter([[.1, .2, .3], [.1, .2, .4]])
        scorer._scores = lambda graph: next(scores)
        with self.assertRaisesRegex(RuntimeError, "replay"):
            scorer.score_amr("other_raw")


if __name__ == "__main__":
    unittest.main()
