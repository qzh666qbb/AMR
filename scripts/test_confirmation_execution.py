"""Network-free regression coverage for paid-call journaling and review parsing."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from run_confirmation_experiment import CallJournal, decode_reviews, freeze, recovery_scope, request_body, summarize


def response():
    result = Mock(status_code=200)
    result.json.return_value = {
        "id": "fixture", "model": "deepseek-flash",
        "usage": {"prompt_tokens": 8, "completion_tokens": 1},
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    }
    return result


class JournalTests(unittest.TestCase):
    def test_complete_call_replayed_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            body = request_body("OK", 0, 16)
            with patch("requests.post", return_value=response()) as post:
                result = CallJournal(folder, "test-secret-not-real", 1).request("probe", body)
                self.assertEqual(post.call_count, 1)
            with patch("requests.post", side_effect=AssertionError("Unexpected network")):
                self.assertEqual(CallJournal(folder, "new-secret-not-real", 1).request("probe", body), result)
            serialized = (folder / "calls/probe-000.json").read_text()
            self.assertNotIn("test-secret-not-real", serialized)
            self.assertNotIn("Authorization", serialized)

    def test_uncertain_call_stops_resume(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            body = request_body("OK", 0, 16)
            with patch("requests.post", side_effect=TimeoutError("secret must not appear")):
                with self.assertRaises(RuntimeError):
                    CallJournal(folder, "not-real", 1).request("probe", body)
            self.assertNotIn("secret must not appear", (folder / "calls/probe-000.json").read_text())
            with patch("requests.post", side_effect=AssertionError("Unexpected retry")):
                with self.assertRaises(RuntimeError):
                    CallJournal(folder, "not-real", 1).request("probe", body)

    def test_call_cap_and_request_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            body = request_body("OK", 0, 16)
            journal = CallJournal(folder, "not-real", 1)
            with patch("requests.post", return_value=response()) as post:
                journal.request("a", body)
                with self.assertRaises(RuntimeError):
                    journal.request("b", body)
                self.assertEqual(post.call_count, 1)
            with self.assertRaises(ValueError):
                CallJournal(folder, "not-real", 1).request("a", request_body("different", 0, 16))

    def test_model_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            journal = CallJournal(Path(d), "not-real", 1)
            journal.model_returned = "another-model"
            with patch("requests.post", return_value=response()):
                with self.assertRaises(ValueError):
                    journal.request("a", request_body("OK", 0, 16))

    def test_freeze_refuses_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "freeze.json"
            freeze(p, {"a": 1})
            freeze(p, {"a": 1})
            with self.assertRaises(ValueError):
                freeze(p, {"a": 2})

    def test_partial_summary_has_no_valid_asr(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "preparation.json", {"prompts": [{}] * 30})
            result = summarize(folder)
            self.assertEqual(result["status"], "integration_pilot_incomplete")
            self.assertTrue(all(row["valid_asr"] is None for row in result["arms"].values()))

    def test_recovery_counts_prior_calls_and_never_expands_budget(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.assertEqual(recovery_scope(folder, "inject-a-0", False), ("inject-a-0", 0))
            for i in range(17):
                freeze(folder / "calls" / f"inject-a-0-{i:03d}.json", {"status": "failed_or_uncertain"})
            with self.assertRaises(RuntimeError):
                recovery_scope(folder, "inject-a-0", False)
            self.assertEqual(recovery_scope(folder, "inject-a-0", True), ("inject-a-0-recovery17", 17))
            for i in range(33):
                freeze(folder / "calls" / f"inject-a-0-recovery17-{i:03d}.json", {"status": "complete"})
            with self.assertRaises(RuntimeError):
                recovery_scope(folder, "inject-a-0", True)


class ReviewTests(unittest.TestCase):
    def test_explicit_string_labels_normalize_without_guessing(self):
        rows = [{"candidate": f"CANDIDATE {k}", "score": 4, "major_error": False} for k in range(4)]
        parsed = decode_reviews(json.dumps(rows))
        self.assertEqual([r["candidate"] for r in parsed.values()], list(range(4)))
        rows[3]["candidate"] = "3"
        self.assertEqual(decode_reviews(json.dumps(rows))[3]["score"], 4)
        rows[3]["candidate"] = "Candidate three"
        with self.assertRaises(ValueError):
            decode_reviews(json.dumps(rows))
        rows[3]["candidate"] = "CANDIDATE 0"
        with self.assertRaises(ValueError):
            decode_reviews(json.dumps(rows))

    def test_strict_identity_and_types(self):
        rows = [{"candidate": k, "score": 5, "major_error": False, "reason": "same"} for k in range(4)]
        self.assertEqual(set(decode_reviews("```json\n" + json.dumps(rows) + "\n```")), set(range(4)))
        rows[3]["candidate"] = 0
        with self.assertRaises(ValueError):
            decode_reviews(json.dumps(rows))
        rows[3]["candidate"] = 3
        rows[0]["major_error"] = "false"
        with self.assertRaises(ValueError):
            decode_reviews(json.dumps(rows))

    def test_same_request_parameters(self):
        body = request_body("text", .7, 1400)
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["max_tokens"], 1400)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("top_p", body)


if __name__ == "__main__":
    unittest.main()
