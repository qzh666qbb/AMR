"""No-network tests of upstream extraction, budgets and generation labels."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from run_confirmation_experiment import request_body
from run_swan_generator_comparison import (
    CappedJournal, FUNCTIONS, context_for, execute_slot, load_upstream, source_gates, upstream_ast,
)
from test_confirmation_execution import response


class ComparisonTests(unittest.TestCase):
    def bindings(self, match=True):
        return {
            "BedrockManager": object(),
            "parse_amr": lambda text: "(x / data)",
            "normalize_amr_variables": lambda graph: graph,
            "generate_template_amr": lambda graph: graph,
            "decide_s2match_match": lambda *args: (match, .8 if match else .2),
        }

    def detector(self):
        return SimpleNamespace(
            bank=["(a / template)", "(b / template)"],
            parser=SimpleNamespace(tokenizer=lambda text: {"input_ids": [1]}, max_sent_len=1000),
            memo=lambda kind, key, compute: json.loads(json.dumps(compute())),
        )

    def test_only_pinned_definitions_loaded(self):
        tree = upstream_ast()
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in tree.body))
        self.assertEqual([n.name for n in tree.body], list(FUNCTIONS))
        module = load_upstream(self.bindings())
        self.assertNotIn("generate_through_HF", module.__dict__)
        self.assertNotIn("torch", module.__dict__)
        prompt = module.build_user_prompt("(a / template)", [], "CONTEXT")
        self.assertIn("CONTEXT", prompt)
        self.assertNotIn("shortest", prompt)
        self.assertNotIn("Parsing feedback", prompt)

    def test_upstream_acceptance_comes_from_matching_event(self):
        module = load_upstream(self.bindings())
        original_match = module.decide_s2match_match
        api = Mock()
        api.request.return_value = {"content": "A device processes data.", "finish_reason": "stop"}
        with patch("nltk.sent_tokenize", side_effect=lambda text: [text]):
            row = execute_slot("upstream", 0, "context", "p", module, self.detector(), api)
        self.assertEqual(row["attempts"], 1)
        self.assertTrue(row["accepted"])
        self.assertEqual(row["target_score"], .8)
        self.assertEqual(len(row["match_events"]), 1)
        self.assertIs(module.decide_s2match_match, original_match)

    def test_upstream_fallback_keeps_all_failed_trials(self):
        module = load_upstream(self.bindings(False))
        api = Mock()
        api.request.return_value = {"content": "A device processes data.", "finish_reason": "stop"}
        with patch("nltk.sent_tokenize", side_effect=lambda text: [text]):
            row = execute_slot("upstream", 0, "context", "p", module, self.detector(), api)
        self.assertEqual(row["attempts"], 15)
        self.assertFalse(row["accepted"])
        self.assertEqual(api.request.call_count, 15)
        self.assertEqual(row["text"], "A device processes data.")

    def test_control_never_calls_parser_or_matcher(self):
        module = load_upstream(self.bindings())
        api = Mock()
        api.request.return_value = {"content": "A device processes data.", "finish_reason": "stop"}
        with patch("nltk.sent_tokenize", side_effect=lambda text: [text]):
            row = execute_slot("unwatermarked", 0, "context", "p", module, None, api)
        self.assertEqual(row["attempts"], 1)
        self.assertIsNone(row["accepted"])
        self.assertEqual(row["match_events"], [])

    def test_source_gate_independent_of_acceptance_and_score(self):
        review = {"fluency": 4, "coherence": 4, "nonredundancy": 4, "severe_problem": False}
        sentences = [{"text": "word " * 20, "finish_reason": "stop", "actual_sentences": 1,
                      "accepted": False, "target_score": 0.} for _ in range(5)]
        self.assertTrue(source_gates(review, sentences)["source_feasibility_pass"])
        self.assertFalse(source_gates(review, sentences[:4])["source_feasibility_pass"])
        sentences[0]["finish_reason"] = "length"
        self.assertFalse(source_gates(review, sentences)["source_feasibility_pass"])

    def test_context_does_not_include_other_arms(self):
        prompt = {"generation_prompt": "Start."}
        self.assertEqual(context_for(prompt, []), "Start.")
        self.assertEqual(context_for(prompt, [{"text": "Own previous."}]), "Start. Own previous.")

    def test_budget_is_cumulative_across_resume(self):
        with tempfile.TemporaryDirectory() as d, patch("run_swan_generator_comparison.CAP", 1):
            folder = Path(d)
            body = request_body("OK", 0, 16)
            with patch("requests.post", return_value=response()):
                CappedJournal(folder, "not-real", Mock()).request("probe", body)
            resumed = CappedJournal(folder, "not-real", Mock())
            with patch("requests.post", side_effect=AssertionError("Network forbidden")):
                resumed.request("probe", body)
                with self.assertRaises(RuntimeError):
                    resumed.request("new-scope", body)
            self.assertEqual(resumed.sent, 0)


if __name__ == "__main__":
    unittest.main()
