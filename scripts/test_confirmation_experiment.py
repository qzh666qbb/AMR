"""Offline tests for independent prompt selection and three-arm budgets."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_confirmation_experiment import (
    ARMS, attack_templates, check, choose_prompts, historical_template, normalized_hash, prepare,
)


def corpus(count=285):
    return [{"text": f"Article{i} describes a different event in detail.", "url": f"https://example.org/{i}"}
            for i in range(count)]


class ConfirmationTests(unittest.TestCase):
    def test_historical_250_excluded(self):
        selected, rejected, scanned = choose_prompts(corpus(), 30, [])
        self.assertEqual([r["shard_row"] for r in selected], list(range(250, 280)))
        self.assertEqual(scanned, 280)
        self.assertFalse(rejected)
        self.assertTrue(all(r["role"] == "unwatermarked_generation_prompt_only" for r in selected))

    def test_text_prompt_and_url_duplicates_excluded(self):
        records = corpus()
        records[250] = records[0].copy()
        records[251]["text"] = records[1]["text"].upper() + " "
        records[252]["text"] = records[2]["text"] + " Additional text follows."
        records[253]["url"] = records[3]["url"]
        selected, rejected, _ = choose_prompts(records, 2, [records[254]["text"]])
        self.assertEqual([r["shard_row"] for r in selected], [255, 256])
        self.assertEqual(len(rejected), 5)

    def test_insufficient_inputs_rejected(self):
        with self.assertRaises(ValueError):
            choose_prompts(corpus(251), 30, [])

    def test_templates_preserve_full_e2_and_candidate_identity(self):
        templates = attack_templates()
        self.assertEqual(set(templates), set(ARMS))
        for k in range(4):
            actual = templates["full_e2"].format("SOURCE {literal}", candidate=k)
            self.assertEqual(actual, historical_template().format("SOURCE {literal}") + f"\nCandidate variant: {k}")
            for template in templates.values():
                prompt = template.format("A source paragraph.", candidate=k)
                self.assertIn("Preserve every entity, number, date, polarity, modality, attribution, and event.", prompt)
                self.assertIn("Keep similar length.", prompt)

    def test_unicode_whitespace_and_case_normalization(self):
        self.assertEqual(normalized_hash("Ａ Test \n text"), normalized_hash("a test text"))

    def test_prepare_is_prompt_only_and_refuses_overwrite_or_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch("prepare_confirmation_experiment.stream_validation", return_value=iter_with_close()):
                prepare(folder, 30)
            status = check(folder)
            self.assertEqual(status["planned_attack_candidates"], 360)
            self.assertFalse(status["ready_for_attack_generation"])
            self.assertFalse(status["generation_started"])
            data = json.loads((folder / "preparation.json").read_text())
            self.assertEqual(data["attack"]["planned_max_output_tokens"], 504000)
            self.assertFalse(any("attack" in r for r in data["prompts"]))
            with self.assertRaises(FileExistsError):
                prepare(folder, 30)
            data["prompts"][0]["generation_prompt"] = "Changed."
            (folder / "preparation.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                check(folder)


def iter_with_close():
    yield from corpus()


if __name__ == "__main__":
    unittest.main()
