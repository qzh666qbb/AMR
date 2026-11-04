"""Meaning/novelty contracts and real durable variable-output accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import swan_writer_repair3 as w
import swan_writer3_budget as b
from swan_final1_common import freeze


def payload(text="{}", output=400):
    return {"id": "synthetic", "model": b.base.MODEL,
            "usage": {"prompt_tokens": 12, "completion_tokens": output},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class Contracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        freeze(self.out / "FREEZE.json", {"parent_budget": {
            "calls": 11223, "exposure_nanodollars": 1903284300},
            "caps": {"new_calls": 1100, "new_nanodollars": 3000000000}})

    def test_long_output_reserved_and_replayed_without_send(self):
        sent = []
        def send(body):
            sent.append(body)
            return payload()
        first = b.Journal(self.out, send).request("plan", "p", "s", 2400)
        journal = b.Journal(self.out, send)
        self.assertEqual(first, journal.request("plan", "p", "s", 2400))
        self.assertEqual(len(sent), 1)
        self.assertEqual(journal.calls["plan"]["reserve"]["reserve"]["output_token_bound"], 2400)
        self.assertEqual(journal.accounting()["cumulative_calls"], 11224)
        with self.assertRaises(b.base.GlobalStop):
            journal.request("plan", "p", "s", 600)

    def test_over_limit_usage_is_durable_halt(self):
        journal = b.Journal(self.out, lambda _: payload(output=601))
        with self.assertRaises(b.base.GlobalStop):
            journal.request("review", "p", "s", 600)
        self.assertEqual(journal.uncertain, 1)
        with self.assertRaises(b.base.GlobalStop):
            b.Journal(self.out, lambda _: self.fail("sent"))

    def test_timeout_keeps_reserve_never_resends(self):
        def send(_):
            raise TimeoutError()
        journal = b.Journal(self.out, send)
        with self.assertRaises(b.base.DocumentFailure):
            journal.request("plan", "p", "s", 2400)
        self.assertGreater(journal.exposure, 2400 * 1200)
        with self.assertRaises(b.base.GlobalStop):
            b.Journal(self.out, lambda _: self.fail("sent"))

    def test_null_response_keeps_halt(self):
        journal = b.Journal(self.out, lambda _: None)
        with self.assertRaises(b.base.GlobalStop):
            journal.request("plan", "p", "s", 2400)
        self.assertEqual(journal.calls["plan"]["terminal"]["kind"], "invalid_response")

    def test_parent_money_gate_accounts_variable_output(self):
        journal = b.Journal(self.out, lambda _: self.fail("sent"))
        with patch.object(b.previous, "PARENT_MONEY_CAP", 1903284300 + b.MAX_RESERVE - 1):
            with self.assertRaises(b.base.Pause):
                journal.request("plan", "p", "s", 2400)

    def test_repetition_or_missing_meaning_cannot_pass(self):
        fields = ("meaning", "novel", "topic", "plausible", "coherent")
        good = {**dict.fromkeys(fields, True), "duplicate_of": [], "issues": []}
        self.assertTrue(w.review_pass(good, fields, "duplicate_of"))
        self.assertFalse(w.review_pass({**good, "novel": False}, fields, "duplicate_of"))
        self.assertFalse(w.review_pass({**good, "meaning": "true"}, fields, "duplicate_of"))
        self.assertFalse(w.review_pass({**good, "duplicate_of": [2]}, fields, "duplicate_of"))

    def test_plan_duplicate_facts_are_not_sixteen_facts(self):
        stages = ("setup", "development", "response", "conclusion")
        plan = {"premise": "one topic", "slots": [
            {"stage": stages[i // 4], "fact": "same fact", "key": str(i),
             "sentence": "The team has been practicing the routine."} for i in range(16)]}
        with self.assertRaisesRegex(ValueError, "duplicate_plan_fact"):
            w.validate_plan(plan, lambda text: [text])

    def test_truncated_or_nonobject_review_fails(self):
        for response in ({"stop_reason": "length", "text": "{}"},
                         {"stop_reason": "stop", "text": "null"}):
            with self.assertRaises(ValueError):
                w.decode(response)


if __name__ == "__main__":
    unittest.main()
