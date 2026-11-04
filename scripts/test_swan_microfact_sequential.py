"""No-network tests for sequential microfact construction."""

import unittest

from run_swan_microfact_sequential import (
    FACTS,
    graph_document,
    slot_body,
    structure_metrics,
)


def tokenize(text):
    return [
        sentence.strip() + "."
        for sentence in text.split(".")
        if sentence.strip()
    ]


class SequentialMicrofactTests(unittest.TestCase):
    def facts(self):
        return [
            f"Subject{i} records one verified source detail clearly today."
            for i in range(FACTS)
        ]

    def responses(self, facts=None):
        return [
            {
                "content": fact,
                "finish_reason": "stop",
            }
            for fact in (facts or self.facts())
        ]

    def test_structure_accepts_exact_short_fact_document(self):
        facts = self.facts()
        result = structure_metrics(facts, self.responses(facts), tokenize)
        self.assertTrue(result["pass"])
        self.assertEqual(result["words"], 120)

    def test_structure_rejects_long_or_internally_dotted_fact(self):
        facts = self.facts()
        facts[0] = (
            "Subject0 records one unusually detailed verified source fact very clearly today."
        )
        self.assertFalse(
            structure_metrics(facts, self.responses(facts), tokenize)["pass"]
        )
        facts = self.facts()
        facts[0] = "Subject0 references the U.S. source detail today."
        self.assertFalse(
            structure_metrics(facts, self.responses(facts), tokenize)["pass"]
        )

    def test_slot_body_binds_position_and_prior_facts(self):
        body = slot_body("Frozen source.", ["Earlier fact."], 1)
        prompt = body["messages"][-1]["content"]
        self.assertIn("Write fact 2 of 15", prompt)
        self.assertIn("Earlier fact.", prompt)
        self.assertEqual(body["max_tokens"], 48)
        with self.assertRaises(ValueError):
            slot_body("Frozen source.", [], 1)

    def test_graph_document_uses_frozen_thresholds(self):
        passing = {
            "sentences": [
                {
                    "scale_compatible": index < 10,
                    "compatible_bank_templates": 10,
                    "template_triples": 12,
                }
                for index in range(FACTS)
            ]
        }
        result = graph_document(passing)
        self.assertTrue(result["pass"])
        self.assertEqual(result["scale_compatible"], 10)
        failing = {
            "sentences": [
                {
                    "scale_compatible": False,
                    "compatible_bank_templates": 1,
                    "template_triples": 12,
                }
                for _ in range(FACTS)
            ]
        }
        self.assertFalse(graph_document(failing)["pass"])


if __name__ == "__main__":
    unittest.main()
