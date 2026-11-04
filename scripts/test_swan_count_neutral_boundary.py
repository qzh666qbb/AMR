"""Tests for the count-neutral SWAN boundary-exchange gate."""

import json
import os
from pathlib import Path
import unittest

from run_segmentation_intervention import ROOT
from run_swan_count_neutral_boundary import (
    build_exchange,
    construct_exchanges,
    find_merge_candidates,
    find_split_candidates,
    lexical_sequence,
)

os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))


class BoundaryExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import nltk
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")
        cls.tokenize = staticmethod(nltk.sent_tokenize)

    def test_explicit_subject_coordination_is_splittable(self):
        sentence = "Carol stayed home, and Dan left early."
        candidates = find_split_candidates(
            sentence, self.nlp, self.tokenize,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["parts"],
            ["Carol stayed home.", "And Dan left early."],
        )
        self.assertEqual(
            lexical_sequence(candidates[0]["text"]),
            lexical_sequence(sentence),
        )

    def test_shared_subject_coordination_is_not_splittable(self):
        sentence = "Carol stayed home and reviewed the report."
        self.assertEqual(
            find_split_candidates(sentence, self.nlp, self.tokenize),
            [],
        )

    def test_unrelated_sentences_are_not_merged(self):
        sentences = [
            "Alice filed the report.",
            "Clouds covered the mountain.",
        ]
        self.assertEqual(find_merge_candidates(sentences, self.nlp), [])

    def test_count_neutral_exchange_preserves_lexical_sequence(self):
        sentences = [
            "Alice filed the report.",
            "Alice reviewed the report.",
            "Carol stayed home, and Dan left early.",
            "Eve approved the plan.",
        ]
        splits, merges, exchanges = construct_exchanges(
            sentences, self.nlp, self.tokenize,
        )
        self.assertEqual(len(splits), 1)
        self.assertEqual(len(merges), 1)
        self.assertEqual(len(exchanges), 1)
        exchange = exchanges[0]
        self.assertEqual(len(exchange["units"]), len(sentences))
        self.assertEqual(
            lexical_sequence(exchange["text"]),
            lexical_sequence(" ".join(sentences)),
        )
        self.assertEqual(
            self.tokenize(exchange["text"]),
            exchange["units"],
        )

    def test_merge_cannot_consume_split_source(self):
        sentences = [
            "Alice filed the report.",
            "Alice reviewed the report, and Bob approved it.",
        ]
        split = {
            "sentence_index": 1,
            "kind": "explicit_subject_coordination",
            "parts": [
                "Alice reviewed the report.",
                "And Bob approved it.",
            ],
        }
        merge = {
            "boundary_index": 0,
            "text": (
                "Alice filed the report; "
                "Alice reviewed the report, and Bob approved it."
            ),
        }
        with self.assertRaises(ValueError):
            build_exchange(sentences, merge, split, self.tokenize)


class FrozenGateTests(unittest.TestCase):
    def test_frozen_gate_result(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/swan_count_neutral_boundary1/summary.json"
        )
        if not path.exists():
            self.skipTest("Count-neutral experiment has not been prepared")
        summary = json.loads(path.read_text())
        self.assertEqual(
            summary["status"],
            "complete_construction_gate_failed",
        )
        self.assertEqual(summary["coverage"]["documents"], 30)
        self.assertEqual(
            summary["coverage"]["documents_with_split_candidate"],
            0,
        )
        self.assertEqual(
            summary["coverage"]["eligible_detected_exchanges"],
            0,
        )
        self.assertEqual(summary["parser_calls"], 0)
        self.assertEqual(summary["s2match_calls"], 0)


if __name__ == "__main__":
    unittest.main()
