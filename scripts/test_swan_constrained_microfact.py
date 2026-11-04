"""No-network tests for constrained microfact construction."""

import json
from pathlib import Path
import tempfile
import unittest

from audit_swan_constrained_microfact import validate_token_path
from run_swan_constrained_microfact import (
    ConstrainedFactLogitsProcessor,
    candidate_metrics,
    choose_candidate,
    decode_grounding_review,
    document_structure,
    generation_prompt,
    generation_seed,
    hard_structure,
    prepare,
    safe_token_ids,
)


def tokenize(text):
    return [
        sentence.strip() + "."
        for sentence in text.split(".")
        if sentence.strip()
    ]


class SurfaceTests(unittest.TestCase):
    def test_hard_structure_accepts_only_bounded_single_sentence(self):
        accepted = "Microsoft changed the Kerberos ticket format in Windows."
        self.assertTrue(hard_structure(accepted, tokenize)["pass"])
        self.assertFalse(
            hard_structure(
                "Microsoft changed Kerberos tickets in Windows.",
                tokenize,
            )["pass"]
        )
        self.assertFalse(
            hard_structure(
                "Microsoft changed Kerberos. Windows used proprietary tickets.",
                tokenize,
            )["pass"]
        )

    def test_candidate_selection_is_source_grounded_and_novel(self):
        source = (
            "Microsoft changed the Kerberos ticket format in Windows. "
            "The company used a proprietary access certificate."
        )
        previous = [
            "Microsoft changed the Kerberos ticket format in Windows."
        ]
        candidates = [
            "Microsoft changed the Kerberos ticket format in Windows.",
            "Microsoft used a proprietary access certificate for Windows tickets.",
            "Unknown researchers invented unrelated systems during winter.",
        ]
        selection = choose_candidate(
            candidates,
            source,
            previous,
            tokenize,
        )
        self.assertEqual(selection["selected_candidate"], 1)
        self.assertTrue(selection["selection_pass"])
        self.assertTrue(
            candidate_metrics(
                candidates[1],
                source,
                previous,
                tokenize,
            )["eligible"]
        )

    def test_selection_retains_fixed_fallback(self):
        source = "Microsoft changed the Kerberos ticket format in Windows."
        duplicate = "Microsoft changed the Kerberos ticket format in Windows."
        selection = choose_candidate(
            [duplicate, duplicate],
            source,
            [duplicate],
            tokenize,
        )
        self.assertEqual(selection["selected_candidate"], 0)
        self.assertFalse(selection["selection_pass"])
        self.assertEqual(selection["fallback_reason"], "no_eligible_candidate")

    def test_document_structure_requires_unique_facts(self):
        facts = [
            f"Subject{i} records one verified source detail clearly today."
            for i in range(15)
        ]
        self.assertTrue(document_structure(facts, tokenize)["pass"])
        facts[-1] = facts[0]
        self.assertFalse(document_structure(facts, tokenize)["pass"])


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            "models/qwen2.5-0.5b-instruct",
            local_files_only=True,
        )
        cls.safe = safe_token_ids(cls.tokenizer)

    def scores(self):
        import torch

        return torch.zeros((1, len(self.tokenizer)))

    def process(self, text):
        import torch

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        processor = ConstrainedFactLogitsProcessor(
            self.tokenizer,
            0,
            self.safe,
        )
        return processor(torch.tensor([ids]), self.scores())

    def test_period_and_eos_are_forced_at_frozen_boundaries(self):
        period = self.tokenizer.encode(".", add_special_tokens=False)[0]
        eos = self.tokenizer.eos_token_id
        before_min = self.process("one two three four five six")
        self.assertTrue(before_min[0, period].isneginf())
        self.assertTrue(before_min[0, eos].isneginf())
        at_min = self.process("one two three four five six seven")
        self.assertFalse(at_min[0, period].isneginf())
        at_max = self.process(
            "one two three four five six seven eight nine ten"
        )
        self.assertFalse(at_max[0, period].isneginf())
        self.assertEqual(int((~at_max.isneginf()).sum()), 1)
        after_period = self.process(
            "one two three four five six seven."
        )
        self.assertFalse(after_period[0, eos].isneginf())
        self.assertEqual(int((~after_period.isneginf()).sum()), 1)

    def test_saved_token_path_validation(self):
        ids = self.tokenizer.encode(
            "one two three four five six seven.",
            add_special_tokens=False,
        )
        ids.append(self.tokenizer.eos_token_id)
        self.assertEqual(
            validate_token_path(self.tokenizer, ids),
            "one two three four five six seven.",
        )
        with self.assertRaises(ValueError):
            validate_token_path(
                self.tokenizer,
                self.tokenizer.encode(
                    "one two three.",
                    add_special_tokens=False,
                ),
            )


class ProtocolTests(unittest.TestCase):
    def test_seed_and_prompt_are_bound_to_slot(self):
        self.assertEqual(
            generation_seed("source", 1),
            generation_seed("source", 1),
        )
        self.assertNotEqual(
            generation_seed("source", 1),
            generation_seed("source", 2),
        )
        prompt = generation_prompt("Frozen source.", ["Prior fact."], 1)
        self.assertIn("fact 2 of 15", prompt)
        self.assertIn("Prior fact.", prompt)
        with self.assertRaises(ValueError):
            generation_prompt("Frozen source.", [], 1)

    def test_grounding_decoder_is_strict(self):
        valid = {
            "faithfulness": 5,
            "atomicity": 4,
            "self_containment": 4,
            "unsupported_claim": False,
            "contradiction": False,
            "duplicate_fact": False,
            "issues": [],
            "reason": "Supported.",
        }
        self.assertEqual(
            decode_grounding_review(json.dumps(valid)),
            valid,
        )
        invalid = dict(valid)
        invalid["faithfulness"] = 6
        with self.assertRaises(ValueError):
            decode_grounding_review(json.dumps(invalid))

    def test_prepare_freezes_paired_development_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = prepare(folder)
            self.assertEqual(len(manifest["sources"]), 5)
            self.assertEqual(
                manifest["protocol"]["generated_candidates"],
                600,
            )
            self.assertIn(
                "not an independent validation",
                manifest["scope"],
            )


if __name__ == "__main__":
    unittest.main()
