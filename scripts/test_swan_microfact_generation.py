"""No-network tests for source-grounded microfact generation."""

import json
from pathlib import Path
import tempfile
import unittest

from run_confirmation_experiment import freeze
from run_swan_microfact_generation import (
    FACTS,
    decode_grounding_review,
    decode_microfacts,
    generation_body,
    review_gates,
    scale_metrics,
    structure_metrics,
    summarize,
)


def tokenize(text):
    return [sentence.strip() + "." for sentence in text.split(".") if sentence.strip()]


def facts(count=FACTS, words=7):
    return [
        " ".join([f"Subject{index}", "states"] + ["plain"] * (words - 3) + ["facts."])
        for index in range(count)
    ]


class MicrofactGenerationTests(unittest.TestCase):
    def test_generation_prompt_treats_source_as_json_data(self):
        body = generation_body('Source says "quoted text."')
        prompt = body["messages"][-1]["content"]
        self.assertIn("SOURCE_JSON", prompt)
        self.assertIn('\\"quoted text.\\"', prompt)
        self.assertEqual(body["temperature"], 0)

    def test_decode_microfacts_has_exact_schema(self):
        expected = facts()
        self.assertEqual(decode_microfacts(json.dumps({"facts": expected})), expected)
        self.assertEqual(
            decode_microfacts("```json\n" + json.dumps({"facts": expected}) + "\n```"),
            expected,
        )
        with self.assertRaises(ValueError):
            decode_microfacts(json.dumps({"facts": expected, "extra": True}))
        with self.assertRaises(ValueError):
            decode_microfacts(json.dumps({"facts": expected + ["extra."]}))

    def test_structure_gate_checks_count_length_identity_and_person(self):
        good = structure_metrics(facts(), tokenize)
        self.assertTrue(good["structure_pass"])
        self.assertEqual(good["words"], FACTS * 7)
        duplicate = facts()
        duplicate[-1] = duplicate[0]
        self.assertFalse(structure_metrics(duplicate, tokenize)["checks"]["unique"])
        personal = facts()
        personal[-1] = "We state several plain facts here today."
        self.assertFalse(
            structure_metrics(personal, tokenize)["checks"]["no_first_second_person"]
        )

    def test_grounding_review_schema_and_indices_are_strict(self):
        valid = {
            "faithfulness": 5,
            "unsupported_claim": False,
            "contradiction": False,
            "all_atomic": True,
            "all_self_contained": True,
            "duplicate_pairs": [],
            "issues": [],
        }
        self.assertEqual(decode_grounding_review(json.dumps(valid), FACTS), valid)
        with self.assertRaises(ValueError):
            decode_grounding_review(
                json.dumps({**valid, "duplicate_pairs": [[0, FACTS]]}), FACTS
            )
        with self.assertRaises(ValueError):
            decode_grounding_review(
                json.dumps({**valid, "all_atomic": False}), FACTS
            )

    def test_review_gates_keep_dimensions_separate(self):
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
            "all_atomic": False,
            "all_self_contained": True,
            "duplicate_pairs": [],
            "issues": [{"index": 0, "claim": "x", "evidence": "x", "reason": "compound"}],
        }
        result = review_gates(style, grounding)
        self.assertTrue(result["style"])
        self.assertTrue(result["factual"])
        self.assertFalse(result["atomicity"])

    def test_scale_gate_requires_all_facts_and_eighty_percent(self):
        rows = [
            {"template_triples": 10, "compatible_bank_templates": 20,
             "scale_compatible": index < 12}
            for index in range(FACTS)
        ]
        self.assertTrue(scale_metrics(rows)["scale_pass"])
        self.assertFalse(scale_metrics(rows[:-1])["scale_pass"])
        rows[11]["scale_compatible"] = False
        self.assertFalse(scale_metrics(rows)["scale_pass"])

    def test_partial_summary_cannot_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            freeze(folder / "manifest.json", {
                "sources": [{"prompt_id": f"p{index}", "shard_row": index,
                             "source_sha256": str(index)} for index in range(5)]
            })
            result = summarize(folder)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["continuation_gate"])
            self.assertEqual(result["decision"], "pending")


if __name__ == "__main__":
    unittest.main()
