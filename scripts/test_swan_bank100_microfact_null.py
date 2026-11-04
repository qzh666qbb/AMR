"""No-network tests for the matched-form bank100 microfact null replication."""

import json
from pathlib import Path
import tempfile
import unittest

from run_swan_bank100_microfact_null import (
    FACTS,
    decode_facts,
    generation_body,
    review_gates,
    screen_source,
    structure_metrics,
    write_report,
)


def tokenize(text):
    return [
        sentence.strip() + "."
        for sentence in text.split(".")
        if sentence.strip()
    ]


class Bank100MicrofactNullTests(unittest.TestCase):
    def facts(self):
        return [
            f"Subject{i} records one verified source detail clearly today."
            for i in range(FACTS)
        ]

    def test_decoder_requires_exact_schema_and_count(self):
        facts = self.facts()
        self.assertEqual(
            decode_facts(json.dumps({"facts": facts})),
            facts,
        )
        with self.assertRaises(ValueError):
            decode_facts(json.dumps({"facts": facts[:-1]}))
        with self.assertRaises(ValueError):
            decode_facts(json.dumps({"facts": facts, "extra": True}))

    def test_structure_requires_short_unique_single_sentences(self):
        result = structure_metrics(self.facts(), tokenize)
        self.assertTrue(result["pass"])
        self.assertEqual(result["fact_count"], FACTS)
        invalid = self.facts()
        invalid[0] = "Subject0 uses an internal abbreviation U.S. today."
        self.assertFalse(structure_metrics(invalid, tokenize)["pass"])

    def test_review_gates_are_strict(self):
        style = {
            "fluency": 4,
            "coherence": 4,
            "nonredundancy": 4,
            "severe_problem": False,
            "reason": "fixture",
        }
        grounding = {
            "faithfulness": 4,
            "unsupported_claim": False,
            "contradiction": False,
            "all_atomic": True,
            "all_self_contained": True,
            "duplicate_pairs": [],
            "issues": [],
        }
        self.assertTrue(all(review_gates(style, grounding).values()))
        self.assertFalse(
            review_gates({**style, "coherence": 3}, grounding)["style"]
        )
        self.assertFalse(
            review_gates(
                style,
                {**grounding, "duplicate_pairs": [[0, 1]]},
            )["atomicity"]
        )

    def test_source_screen_requires_rich_complete_document(self):
        sentences = [
            f"Topic sentence {index} contains enough distinct source words today."
            for index in range(45)
        ]
        source = " ".join(sentences)
        result = screen_source(source, tokenize)
        self.assertTrue(result["eligible"])
        self.assertGreaterEqual(result["substantial_sentences"], 15)
        self.assertFalse(screen_source("Too short.", tokenize)["eligible"])

    def test_generation_requests_json_object_mode(self):
        body = generation_body("A frozen source document.")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 900)

    def test_report_handles_empty_detector_cohort(self):
        metrics = {"documents": 0, "detected": 0, "rate": None}
        result = {
            "machine_quality_pass": 0,
            "joint_gate": False,
            "machine_gate_pass_counts": {
                "structure": 0,
                "style": 0,
                "facts": 0,
                "atomicity": 0,
            },
            "detector": {
                "calibrated": metrics,
                "fixed_reference": metrics,
            },
            "representation": {
                "sentences": 0,
                "count_feasible_at_075": 0,
                "count_feasible_fraction": None,
            },
            "api_attempts": 91,
            "neural_parser_sentences": 0,
            "s2match_calls": 0,
            "decision": "stop_bank100_matched_form_route",
        }
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory), result)
            self.assertIn("n/a", (Path(directory) / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
