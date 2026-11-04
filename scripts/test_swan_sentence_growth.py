"""Offline checks for sentence-growth budgets, replay and feasibility decisions."""

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from run_confirmation_experiment import freeze, request_body
from run_swan_sentence_growth import (
    BASELINE_REVIEW, BoundedJournal, BudgetReached, baseline_gate,
    decode_baseline_review, injection_adapter, pair_merge, summarize,
)
from test_confirmation_execution import response


class GrowthTests(unittest.TestCase):
    def test_baseline_review_strict_types(self):
        row = {"fluency": 4, "coherence": 4, "nonredundancy": 5,
               "severe_problem": False, "reason": "fixture"}
        self.assertEqual(decode_baseline_review(json.dumps(row)), row)
        row["fluency"] = True
        with self.assertRaises(ValueError):
            decode_baseline_review(json.dumps(row))
        row["fluency"] = 4
        row["severe_problem"] = "false"
        with self.assertRaises(ValueError):
            decode_baseline_review(json.dumps(row))

    def test_gate_not_detector_conditioned(self):
        review = {"fluency": 4, "coherence": 4, "nonredundancy": 4, "severe_problem": False}
        sentences = [{"text": "Example.", "finish_reason": "stop", "actual_sentences": 1}] * 20
        self.assertTrue(baseline_gate(review, 120, "medium", sentences))
        self.assertFalse(baseline_gate(review, 90, "medium", sentences))
        self.assertFalse(baseline_gate({**review, "nonredundancy": 3}, 120, "medium", sentences))
        self.assertFalse(baseline_gate({**review, "severe_problem": True}, 120, "medium", sentences))
        self.assertFalse(baseline_gate(review, 120, "medium", [
            {**sentences[0], "finish_reason": "length"}]))

    def test_pair_merge_preserves_odd_tail(self):
        self.assertEqual(pair_merge(["Alice left.", "Bob stayed.", "Carol waited."]),
                         "Alice left; Bob stayed. Carol waited.")

    def test_cumulative_cap_blocks_new_calls_but_allows_replay(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            with patch("run_swan_sentence_growth.INJECTION_CAP", 1), patch("run_swan_sentence_growth.TOTAL_CAP", 2):
                api = BoundedJournal(folder, "not-real", Mock())
                body = request_body("OK", 0, 16)
                with patch("requests.post", return_value=response()) as post:
                    api.request("inject-000", body)
                    with self.assertRaises(BudgetReached):
                        api.request("inject-001", body)
                    self.assertEqual(post.call_count, 1)
                resumed = BoundedJournal(folder, "not-real", Mock())
                with patch("requests.post", side_effect=AssertionError("Unexpected network")):
                    resumed.request("inject-000", body)
                    with self.assertRaises(BudgetReached):
                        resumed.request("inject-001", body)
                self.assertEqual(resumed.sent, 0)

    def test_total_cap_includes_non_injection_calls(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("run_swan_sentence_growth.TOTAL_CAP", 1):
                api = BoundedJournal(Path(d), "not-real", Mock())
                with patch("requests.post", return_value=response()):
                    api.request("probe", request_body("OK", 0, 16))
                    with self.assertRaises(BudgetReached):
                        api.request("baseline-review-short", request_body(BASELINE_REVIEW, 0, 700))

    def test_adapter_restores_globals_and_replays_cached_scores(self):
        import numpy as np
        original_rng = random
        parse, match = Mock(return_value="(x / example)"), Mock(return_value=(np.bool_(True), np.float64(.8)))
        manager = object()
        injection = SimpleNamespace(BedrockManager=manager, random=original_rng,
                                    parse_amr=parse, decide_s2match_match=match)
        cache = {}
        def memo(kind, key, compute):
            index = kind, json.dumps(key, sort_keys=True)
            if index not in cache:
                cache[index] = compute()
                json.dumps(cache[index], allow_nan=False)
            return cache[index]
        detector = SimpleNamespace(memo=memo, parser=SimpleNamespace(
            tokenizer=lambda text: {"input_ids": [1, 2]}, max_sent_len=100))
        api = Mock()
        api.request.return_value = {"content": "Example.", "finish_reason": "stop"}
        choices = []
        for _ in range(2):
            with injection_adapter(injection, detector, api, "inject-000", 123) as responses:
                choices.append(injection.random.choice(range(50)))
                self.assertEqual(injection.parse_amr("Example."), "(x / example)")
                self.assertEqual(injection.decide_s2match_match("a", "b", .75), [True, .8])
                self.assertEqual(injection.BedrockManager().generate("u", "s", 50, .7, .9), "Example.")
                self.assertEqual(len(responses), 1)
            self.assertIs(injection.random, original_rng)
            self.assertIs(injection.BedrockManager, manager)
        self.assertEqual(choices[0], choices[1])
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(match.call_count, 1)
        with self.assertRaises(ValueError):
            with injection_adapter(injection, detector, api, "inject-000", 123):
                raise ValueError("fixture")
        self.assertIs(injection.parse_amr, parse)
        self.assertIs(injection.decide_s2match_match, match)

    def test_incomplete_prefix_is_not_counted_as_success_or_failure(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"prompts": [{"prompt_id": "p"}]})
            result = summarize(folder)
            self.assertEqual(result["status"], "incomplete_development_run")
            self.assertTrue(result["no_population_asr"])
            self.assertEqual(result["independent_documents"], 1)
            self.assertTrue(all(r["status"] == "not_completed" for r in result["prefixes"].values()))


if __name__ == "__main__":
    unittest.main()
