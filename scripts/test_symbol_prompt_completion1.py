import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_semantic_sequence_stage1b as old
import run_symbol_prompt_completion1 as run
import symbol_prompt_completion1_budget as budget
from swan_final1_common import freeze


def response(text="Kept.", tokens=10):
    return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": tokens},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class PromptCompletion(unittest.TestCase):
    def test_old_arm_exactly_matches_original(self):
        p = old.build_manifests({"symbols": {s: {"templates": list(range(i * 6, i * 6 + 6))}
                                            for i, s in enumerate("ABCD")},
                                 "template_to_symbol": {str(i): "ABCD"[i // 6] for i in range(24)}})[0][0]["prompt"]
        for slot in range(8):
            self.assertEqual(run.messages(p, "Previous.", slot, "(v / V)", "old"),
                             old.carrier_messages(p, "Previous.", slot, "(v / V)"))
        revised = run.messages(p, "", 0, "(v / V)", "revised")[-1]["content"]
        self.assertNotIn("in the past tense", revised)
        self.assertNotIn("another unspecified concept", revised)
        self.assertIn("amr-unknown", revised)

    def test_empty_first_response_is_replayed_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            journal = budget.Journal(d, "hash", lambda b: calls.append(b) or response(""))
            self.assertEqual(journal.request("c0", [])["text"], "")
            restored = budget.Journal(d, "hash", lambda b: self.fail("repeated transport"))
            self.assertEqual(restored.request("c0", [])["text"], "")
            self.assertEqual(len(calls), 1)
            self.assertEqual(restored.totals()["uncertain_calls"], 0)

    def test_uncertain_dispatch_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            def fail(body):
                raise TimeoutError()
            journal = budget.Journal(d, "hash", fail)
            with self.assertRaises(TimeoutError):
                journal.request("c0", [])
            self.assertEqual(journal.totals()["uncertain_calls"], 1)
            with self.assertRaises(budget.IntegrityError):
                budget.Journal(d, "hash")

    def test_replay_rejects_changed_request_or_raw_response(self):
        with tempfile.TemporaryDirectory() as d:
            journal = budget.Journal(d, "hash", lambda b: response())
            journal.request("c0", [])
            with self.assertRaises(budget.IntegrityError):
                journal.request("c0", [{"role": "user", "content": "changed"}])
            path = Path(d) / "private/calls/c0.json"
            path.write_text(json.dumps(response("tampered")))
            with self.assertRaises(budget.IntegrityError):
                budget.Journal(d, "hash")

    def test_usage_schema_bound_and_model_are_enforced(self):
        allowance = budget.reserve(dict(budget.PARAMETERS, messages=[]))
        for payload in (response(tokens=1601), {**response(), "model": "changed"},
                        {**response(), "usage": {}}, {**response(), "choices": []}):
            with self.assertRaises(budget.IntegrityError):
                budget.parse(payload, allowance)
        self.assertGreater(allowance["nanodollars"], budget.parse(response(tokens=1600), allowance)["nanodollars"])

    def test_budget_reservation_precedes_send(self):
        with tempfile.TemporaryDirectory() as d:
            def send(body):
                events = (Path(d) / "private/calls/journal.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(events[-1])["kind"], "intent")
                return response()
            budget.Journal(d, "hash", send).request("c0", [])
        with self.assertRaises(budget.BudgetPause):
            budget.reserve(dict(budget.PARAMETERS, messages=[{"content": "x" * 40000}]))

    def test_contracts_keep_ambiguous_unknown_polarity(self):
        bank = [{"normalized_template_amr": '(v / V :polarity (x / X) :ARG0 (p / N :name (n / N :op1 "NE" :op2 "NE")))'}]
        row = run.template_contracts(bank, [0])[0]
        self.assertTrue(row["polarity_X"])
        self.assertEqual(row["name_operand_counts"], [2])
        self.assertTrue(row["mechanical_description_only"])

    def test_paired_bootstrap_is_by_source_and_retains_negative(self):
        rows = [{"old": .5, "revised": .25} for _ in range(24)]
        result = run.paired_summary(rows)
        self.assertEqual(result["mean_difference"], -.25)
        self.assertEqual(result["bootstrap95_by_source"], [-.25, -.25])

    def test_empty_and_native_limit_are_retained_erasures(self):
        parser = object.__new__(run.Parser)
        parser.freeze_sha = "frozen"
        parser.model = SimpleNamespace(max_sent_len=1024,
                                       tokenizer=lambda text, truncation: {"input_ids": [0] * len(text)})
        for text, reason in (("", "empty_text"), ("x" * 1024, "native_input_limit")):
            record = parser.fresh(text)
            self.assertFalse(record["valid"])
            self.assertEqual(record["erasure_reason"], reason)

    def test_full_generation_and_segmentation_replay_no_live_requests(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            symbol_map = {"symbols": {s: {"templates": list(range(i * 6, i * 6 + 6))}
                                     for i, s in enumerate("ABCD")},
                          "template_to_symbol": {str(i): "ABCD"[i // 6] for i in range(24)}}
            prompts, assignments = old.build_manifests(symbol_map)
            freeze(out / "FREEZE.json", {"test_fixture": True})
            (out / "private").mkdir()
            old.jsonl(out / "private/PROMPTS.jsonl", prompts)
            old.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl", assignments)
            freeze(out / "private/BANK.json", [{"normalized_template_amr": "(v / V)"}] * 24)
            freeze(out / "TEMPLATE_CONTRACTS.json", [])
            order = [{"doc_id": doc, "arm": arm, "slot": slot} for doc in range(24)
                     for slot in list(range(8)) + ["paraphrase"] for arm in run.ARMS]
            freeze(out / "private/DISPATCH_ORDER.json", order)
            fake_response = SimpleNamespace(status_code=200, json=lambda: response("A small fictional event happened."))
            with patch.object(run, "verify"), patch.object(run, "read_credential", return_value="TEST"), \
                    patch("requests.post", return_value=fake_response) as transport:
                run.generate(out, Path(d) / "unused")
                self.assertEqual(transport.call_count, 432)
                run.generate(out, Path(d) / "unused")
                self.assertEqual(transport.call_count, 432)
            run.materialize(out)
            streams = run.load(out / "private/INPUTS.json")
            self.assertEqual(len(streams), 144)
            self.assertEqual(sum(len(s["units"]) for s in streams if s["arm"].endswith("/slots")), 384)
            self.assertNotIn("intended_template", json.dumps(streams))
            run.materialize(out)
            run.collect(out)
            audit = run.load(out / "HUMAN_COLLECTION_AUDIT.json")
            self.assertEqual(audit["valid_submissions"], [0, 0])
            self.assertFalse(audit["human_semantic_collection_complete"])


if __name__ == "__main__":
    unittest.main()
