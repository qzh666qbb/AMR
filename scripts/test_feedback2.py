"""Offline regression tests for exploratory metrics and bounded probe execution."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from analyze_feedback2_diagnostics import edit_distance, lexical_tokens, ngram_retention, text_metrics, validate_group
from feedback2_runtime import CachedDetector, Heartbeat, claim
from run_confirmation_experiment import digest, freeze
from run_swan_length_probe import LENGTHS, SHORTEST, assess, length_prompt, summarize


class MetricTests(unittest.TestCase):
    def test_exact_edit_distance(self):
        self.assertEqual(edit_distance(["a", "b"], ["a", "c", "b"]), 1)
        self.assertEqual(edit_distance([], ["a"]), 1)
        self.assertEqual(edit_distance(["b"], []), 1)
        self.assertEqual(edit_distance(["a", "b"], ["b", "a"]), 2)

    def test_punctuation_preserves_lexical_sequence(self):
        a, b = "Alice waited. Bob left.", "Alice waited; Bob left."
        self.assertEqual(lexical_tokens(a), lexical_tokens(b))
        metrics = text_metrics(a, b, lambda s: [x for x in s.split(".") if x.strip()])
        self.assertEqual(metrics["lexical_edit_fraction"], 0)
        self.assertEqual(metrics["sentence_delta"], -1)
        self.assertEqual(metrics["lexical_4gram_retention"], 1)

    def test_multiset_retention_not_set_reuse(self):
        self.assertEqual(ngram_retention(["a"] * 4, ["a"] * 2, 2), 1 / 3)
        self.assertIsNone(ngram_retention(["a"], ["a"], 4))

    def test_selection_and_fixed_candidate_budget(self):
        group = {"prompt_id": "p", "no_feedback_candidate": 1, "rows": [
            {"prompt_id": "p", "candidate": i, "quality_valid": i > 0} for i in range(4)]}
        validate_group(group)
        group["no_feedback_candidate"] = 2
        with self.assertRaises(ValueError):
            validate_group(group)
        group["no_feedback_candidate"] = 1
        group["rows"].pop()
        with self.assertRaises(ValueError):
            validate_group(group)


class RuntimeTests(unittest.TestCase):
    def test_exclusive_claim_and_release_after_error(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "task.lock"
            with claim(lock):
                with self.assertRaises(RuntimeError):
                    with claim(lock):
                        self.fail("Duplicate claim")
            with self.assertRaises(ValueError):
                with claim(lock):
                    raise ValueError("fixture")
            with claim(lock):
                pass

    def test_persistent_cache_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("feedback2_runtime.Detector.__init__", return_value=None):
                cache = CachedDetector(Path(d), "fixture")
                self.assertEqual(cache.memo("parse", {"text": "x"}, lambda: [1]), [1])
                resumed = CachedDetector(Path(d), "fixture")
                self.assertEqual(resumed.memo("parse", {"text": "x"}, lambda: self.fail("Recomputed")), [1])
                self.assertEqual(resumed.counts["parse_hits"], 1)
                with self.assertRaises(ValueError):
                    cache.memo("parse", {"text": "bad"}, lambda: (_ for _ in ()).throw(ValueError()))
                self.assertFalse((cache.cache / "parse" / f'{digest({"text": "bad"})}.json').exists())
                path = next((cache.cache / "parse").glob("*.json"))
                saved = json.loads(path.read_text())
                saved["value"] = [2]
                path.write_text(json.dumps(saved))
                with self.assertRaises(ValueError):
                    resumed.memo("parse", {"text": "x"}, lambda: [])

    def test_heartbeat_finishes_and_joins(self):
        with tempfile.TemporaryDirectory() as d:
            with Heartbeat(Path(d), interval=.01) as heartbeat:
                heartbeat.update(stage="fixture")
                time.sleep(.03)
            payload = json.loads(next((Path(d) / "run_logs").glob("*.json")).read_text())
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["stage"], "fixture")
            self.assertFalse(heartbeat.thread.is_alive())


class ProbeTests(unittest.TestCase):
    def test_prompt_changes_only_length_instruction(self):
        original = "Before\n" + SHORTEST + "\nAfter"
        amended = length_prompt(original, 24, 36)
        self.assertTrue(amended.startswith("Before\n"))
        self.assertTrue(amended.endswith("\nAfter"))
        self.assertIn("24 and 36", amended)
        self.assertNotIn(SHORTEST, amended)
        with self.assertRaises(ValueError):
            length_prompt("Changed upstream", 24, 36)

    def test_joint_success_not_just_score(self):
        text = "Alice sent a letter today."
        self.assertTrue(assess(text, "stop", [{}], .75, [5, 10])["joint_success"])
        for finish, details, score, bounds in (
            ("length", [{}], 1., [5, 10]), ("stop", [{}, {}], 1., [5, 10]),
            ("stop", [{}], .74, [5, 10]), ("stop", [{}], 1., [24, 36]),
            ("stop", [], None, [5, 10]),
        ):
            self.assertFalse(assess(text, finish, details, score, bounds)["joint_success"])
        self.assertFalse(assess("NE sent a letter today.", "stop", [{}], 1., [5, 10])["joint_success"])

    def test_partial_probe_cannot_pass_expansion_gate(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            jobs = [{"id": f"{s}-{i}"} for s in LENGTHS for i in range(6)]
            freeze(folder / "manifest.json", {"jobs": jobs, "protocol": {"no_asr": "No ASR"}})
            result = summarize(folder)
            self.assertEqual(result["status"], "incomplete_preflight")
            self.assertIsNone(result["paragraph_pilot_gate_passed"])
            self.assertEqual(result["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
