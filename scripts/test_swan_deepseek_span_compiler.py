"""No-network tests for the DeepSeek span compiler."""

import json
from pathlib import Path
import tempfile
import unittest

from run_swan_deepseek_span_compiler import (
    PLANS_PER_BATCH,
    compile_plan,
    decode_plans,
    plan_body,
    prepare,
    process_batches,
)


def plan(subject, predicate, object_text, evidence):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_text,
        "evidence": evidence,
    }


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.evidence = (
            "Judge Thomas Pender found Apple infringed one Qualcomm patent Friday"
        )
        self.source = {
            "prompt_id": "source",
            "source": self.evidence + ".",
            "source_sha256": "unused",
        }
        self.valid = plan(
            "Judge Thomas Pender",
            "found",
            "Apple infringed one Qualcomm patent",
            self.evidence,
        )

    def test_compiler_accepts_exact_ordered_spans(self):
        result = compile_plan(self.valid, self.source["source"], [])
        self.assertTrue(result["eligible"])
        self.assertEqual(
            result["compiled"],
            "Judge Thomas Pender found Apple infringed one Qualcomm patent.",
        )
        self.assertEqual(result["words"], 9)

    def test_compiler_rejects_paraphrase_and_reordered_spans(self):
        paraphrase = dict(self.valid)
        paraphrase["predicate"] = "determined"
        self.assertFalse(
            compile_plan(paraphrase, self.source["source"], [])["eligible"]
        )
        reordered = dict(self.valid)
        reordered["subject"] = "Apple"
        reordered["object"] = "Judge Thomas Pender"
        self.assertFalse(
            compile_plan(reordered, self.source["source"], [])["eligible"]
        )

    def test_compiler_rejects_duplicate_or_pronoun(self):
        first = compile_plan(self.valid, self.source["source"], [])
        self.assertTrue(first["eligible"])
        duplicate = compile_plan(
            self.valid,
            self.source["source"],
            [first["compiled"]],
        )
        self.assertFalse(duplicate["checks"]["normalized_unique"])
        pronoun_evidence = "They found Apple infringed one Qualcomm patent Friday"
        pronoun = plan(
            "They",
            "found",
            "Apple infringed one Qualcomm patent Friday",
            pronoun_evidence,
        )
        result = compile_plan(pronoun, pronoun_evidence + ".", [])
        self.assertFalse(result["checks"]["resolved_subject"])

    def test_fixed_order_selection_has_no_fallback(self):
        plans = [self.valid, self.valid]
        batches = [{"batch": 0, "plans": plans}]
        result = process_batches(batches, self.source)
        self.assertEqual(len(result["selected"]), 1)
        self.assertTrue(result["candidates"][0]["selected"])
        self.assertFalse(result["candidates"][1]["selected"])
        self.assertEqual(
            result["candidates"][1]["selection_reason"],
            "compiler_rejected",
        )


class DecoderTests(unittest.TestCase):
    def valid_plan(self, index):
        evidence = (
            f"Subject{index} records one distinct verified source fact today"
        )
        return plan(
            f"Subject{index}",
            "records",
            "one distinct verified source fact today",
            evidence,
        )

    def test_decoder_requires_exact_batch_and_schema(self):
        value = {
            "plans": [
                self.valid_plan(index)
                for index in range(PLANS_PER_BATCH)
            ]
        }
        decoded = decode_plans(json.dumps(value))
        self.assertEqual(len(decoded), PLANS_PER_BATCH)
        value["plans"].pop()
        with self.assertRaises(ValueError):
            decode_plans(json.dumps(value))

    def test_decoder_rejects_extra_key(self):
        value = {
            "plans": [
                self.valid_plan(index)
                for index in range(PLANS_PER_BATCH)
            ]
        }
        value["plans"][0]["score"] = 1
        with self.assertRaises(ValueError):
            decode_plans(json.dumps(value))

    def test_second_batch_binds_avoid_claims(self):
        body = plan_body("Frozen source.", 1, ["Prior compiled fact."])
        prompt = body["messages"][-1]["content"]
        self.assertIn("Prior compiled fact.", prompt)
        self.assertEqual(body["max_tokens"], 6000)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        with self.assertRaises(ValueError):
            plan_body("Frozen source.", 2, [])


class ManifestTests(unittest.TestCase):
    def test_prepare_freezes_paired_cohort_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(Path(directory))
            self.assertEqual(len(manifest["sources"]), 5)
            self.assertEqual(manifest["protocol"]["api_cap"], 21)
            self.assertFalse(
                manifest["protocol"]["compiler"]["amr_or_detector_inputs"]
            )
            self.assertIn(
                "not an independent validation",
                manifest["scope"],
            )


if __name__ == "__main__":
    unittest.main()
