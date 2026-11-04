"""No-network regression tests for source screening and paired injection."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from run_confirmation_experiment import freeze, request_body
from run_grounded_injection_pilot import (
    COMMON, Journal, context_for, execute_slot, grounded_prompt, screen_source, select_sources,
    source_metrics, summarize,
)
from run_swan_generator_comparison import load_upstream
from test_confirmation_execution import response


def tokenize(text):
    return [s.strip() + "." for s in text.split(".") if s.strip()]


def article(label="source"):
    return " ".join(" ".join([f"{label}{i}"] * 30) + "." for i in range(10))


class PilotTests(unittest.TestCase):
    def test_source_length_complete_and_unique(self):
        self.assertTrue(screen_source(article(), tokenize)["eligible"])
        self.assertIn("incomplete_terminal", screen_source(article()[:-1], tokenize)["reasons"])
        self.assertIn("incomplete_terminal", screen_source(article() + "..", tokenize)["reasons"])
        self.assertIn("words_outside_250_800", screen_source("Too short.", tokenize)["reasons"])
        self.assertIn("exact_sentence_repetition", screen_source(("word " * 30 + ". ") * 10, tokenize)["reasons"])

    def test_selection_before_generation_keeps_rejection_provenance(self):
        records = [{"text": "historical", "url": f"u{i}"} for i in range(288)]
        records += [{"text": "short.", "url": "short"}, {"text": article("old"), "url": "used"}]
        records += [{"text": article(f"new{i}"), "url": f"new{i}"} for i in range(5)]
        chosen, rejected, scanned = select_sources(records, [{"text": "prior text", "url": "used"}], tokenize)
        self.assertEqual([r["shard_row"] for r in chosen], list(range(290, 295)))
        self.assertEqual(len(rejected), 2)
        self.assertIn("prior_or_duplicate_source", rejected[1]["reasons"])
        self.assertEqual(scanned, 295)

    def test_structured_context_has_only_own_prior_sentences(self):
        context = json.loads(context_for("source", [{"text": "previous"}], 1))
        self.assertEqual(context, {"source": "source", "previous_sentences": ["previous"],
                                   "sentence_position": 2, "total_sentences": 5})

    def bindings(self):
        return {
            "BedrockManager": object(), "parse_amr": lambda text: "(x / test)",
            "normalize_amr_variables": lambda graph: graph, "generate_template_amr": lambda graph: graph,
            "decide_s2match_match": lambda *args: (False, .2),
        }

    def test_prompt_wrapper_restores_after_exception(self):
        module = load_upstream(self.bindings())
        original = module.build_user_prompt
        with self.assertRaises(ValueError):
            with grounded_prompt(module):
                self.assertTrue(module.build_user_prompt("t", [], "c").endswith(COMMON))
                raise ValueError("fixture")
        self.assertIs(module.build_user_prompt, original)

    def test_rejection_fallback_trace_and_budget(self):
        module = load_upstream(self.bindings())
        detector = SimpleNamespace(
            bank=["(t / template)"], parser=SimpleNamespace(tokenizer=lambda _: {"input_ids": [0]}, max_sent_len=100),
            memo=lambda kind, key, compute: compute())
        api = Mock()
        api.request.return_value = {"content": "A factual sentence.", "finish_reason": "stop"}
        job = {"prompt_id": "p", "arm": "swan_grounded", "slot": 0}
        original = module.build_user_prompt
        with patch("nltk.sent_tokenize", side_effect=tokenize):
            row = execute_slot(job, "source", [], module, detector, api)
        self.assertEqual(row["attempts"], 6)
        self.assertFalse(row["accepted"])
        self.assertEqual(len(row["match_events"]), 6)
        self.assertIs(module.build_user_prompt, original)
        self.assertEqual(api.request.call_count, 6)

    def test_control_never_queries_matcher(self):
        module = load_upstream(self.bindings())
        api = Mock()
        api.request.return_value = {"content": "A factual sentence.", "finish_reason": "stop"}
        with patch("nltk.sent_tokenize", side_effect=tokenize):
            row = execute_slot({"prompt_id": "p", "arm": "control", "slot": 0}, "source", [], module, None, api)
        self.assertIsNone(row["accepted"])
        self.assertEqual(row["attempts"], 1)

    def test_quality_excludes_score_and_exact_sentence_length(self):
        style = {"fluency": 4, "coherence": 4, "nonredundancy": 4, "severe_problem": False}
        facts = {"faithfulness": 5, "unsupported_claim": False, "contradiction": False, "issues": []}
        sentences = [{"text": " ".join(["word"] * n) + ".", "finish_reason": "stop",
                      "actual_sentences": 1, "accepted": False} for n in (36, 20, 20, 20, 20)]
        result = source_metrics(sentences, style, facts, tokenize)
        self.assertTrue(result["source_pass"])
        self.assertEqual(result["words"], 116)
        self.assertNotIn("sentence_length", result["gates"])
        sentences[0]["text"] = "1. " + sentences[0]["text"]
        self.assertFalse(source_metrics(sentences, style, facts, tokenize)["gates"]["structure"])

    def test_budget_across_restarts(self):
        with tempfile.TemporaryDirectory() as d, patch("run_grounded_injection_pilot.CAP", 1):
            folder = Path(d)
            body = request_body("OK", 0, 16)
            with patch("requests.post", return_value=response()):
                Journal(folder, "", Mock()).request("probe", body)
            with patch("requests.post", side_effect=AssertionError("No network")):
                api = Journal(folder, "", Mock())
                api.request("probe", body)
                with self.assertRaises(RuntimeError):
                    api.request("another", body)
                self.assertEqual(api.sent, 0)

    def test_partial_pilot_has_no_continue_decision(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"prompts": [{"prompt_id": f"p{i}"} for i in range(5)]})
            result = summarize(folder)
            self.assertEqual(result["decision"], "pending")
            self.assertEqual(result["allocated_documents"], 5)


if __name__ == "__main__":
    unittest.main()
