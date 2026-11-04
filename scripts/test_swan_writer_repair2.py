"""Request replay, malformed response, and Bank membership contracts."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import swan_writer_repair2 as writer
from swan_final1_common import freeze, sha256


def response():
    return {"id": "synthetic", "model": writer.budget.MODEL,
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
            "choices": [{"message": {"content": "The scientist carefully examined the new sample."},
                         "finish_reason": "stop"}]}


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        freeze(self.out / "FREEZE.json", {"parent_budget": {
            "calls": 10938, "exposure_nanodollars": 1866770400}})

    def test_complete_replay_never_resends(self):
        sent = []
        def send(body):
            sent.append(body)
            return response()
        j = writer.Journal(self.out, send)
        value = j.request("candidate", "prompt", "system")
        replay = writer.Journal(self.out, send)
        self.assertEqual(value, replay.request("candidate", "prompt", "system"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(replay.accounting()["cumulative_calls"], 10939)
        with self.assertRaises(writer.budget.GlobalStop):
            replay.request("candidate", "changed", "system")

    def test_json_null_is_durable_halt(self):
        j = writer.Journal(self.out, lambda _: None)
        with self.assertRaises(writer.budget.GlobalStop):
            j.request("candidate", "prompt", "system")
        self.assertEqual(j.calls["candidate"]["terminal"]["kind"], "invalid_response")
        self.assertGreater(j.accounting()["new_version"]["exposure_nanodollars"], 0)
        with self.assertRaises(writer.budget.GlobalStop):
            writer.Journal(self.out, lambda _: self.fail("redispatched"))

    def test_unknown_response_never_resends(self):
        def fail(_):
            raise TimeoutError("synthetic")
        j = writer.Journal(self.out, fail)
        with self.assertRaises(writer.budget.DocumentFailure):
            j.request("candidate", "prompt", "system")
        with self.assertRaises(writer.budget.GlobalStop):
            writer.Journal(self.out, lambda _: self.fail("redispatched"))

    def test_cumulative_cap_retains_parent_exposure(self):
        j = writer.Journal(self.out, lambda _: self.fail("sent"))
        with patch.object(writer, "PARENT_MONEY_CAP", 1866770400):
            with self.assertRaises(writer.budget.Pause):
                j.request("candidate", "prompt", "system")

    def test_full_bank_gate_preserves_threshold(self):
        scores = [.1] * 50
        scores[23] = .75
        self.assertTrue(writer.accept_scores(scores, True))
        self.assertFalse(writer.accept_scores(scores, False))
        scores[23] = .7499999
        self.assertFalse(writer.accept_scores(scores, True))
        with self.assertRaises(writer.budget.GlobalStop):
            writer.accept_scores(scores[:49], True)

    def test_truncated_sentence_cannot_pass(self):
        text = response()["choices"][0]["message"]["content"]
        self.assertTrue(writer.sentence_valid(text, "stop", lambda _: [text]))
        self.assertFalse(writer.sentence_valid(text, "length", lambda _: [text]))
        self.assertFalse(writer.sentence_valid(text, "stop", lambda _: [text, "Extra."]))


if __name__ == "__main__":
    unittest.main()
