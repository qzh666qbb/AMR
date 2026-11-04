"""Generation invariants, using no model calls or live credentials."""
import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import run_semantic_sequence_stage1b as g


class Stage1BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbol_map = json.loads((g.RUN / "SYMBOL_MAP.json").read_text())
        cls.prompts, cls.assignments = g.build_manifests(cls.symbol_map)

    def test_balance_replay_and_split(self):
        self.assertEqual((self.prompts, self.assignments), g.build_manifests(self.symbol_map))
        self.assertEqual(len({p["prompt"] for p in self.prompts}), 24)
        self.assertEqual(len({e for p in self.prompts for e in p["entities"]}), 72)
        for split in ("development", "readout"):
            rows = [a for a in self.assignments if a["split"] == split]
            self.assertEqual(len(rows), 12)
            self.assertEqual(Counter(s["intended_template"] for a in rows for s in a["slots"]),
                             Counter({int(t): 4 for t in self.symbol_map["template_to_symbol"]}))
            domains = Counter(p["domain"] for p in self.prompts if p["split"] == split)
            self.assertEqual(set(domains.values()), {4})
        for a in self.assignments:
            self.assertEqual(Counter(s["intended_state"] for s in a["slots"]),
                             Counter(dict.fromkeys("ABCD", 2)))

    def test_negative_has_no_target_or_positive_context(self):
        messages = g.carrier_messages("fiction scenario", "own-arm-context", 2)
        text = messages[-1]["content"]
        self.assertIn("own-arm-context", text)
        self.assertNotIn("AMR", text)
        self.assertNotIn("template", text)
        self.assertNotIn("symbol", text)

    def make_runner(self, directory):
        runner = object.__new__(g.Generator)
        runner.run = Path(directory)
        runner.private = runner.run / "private"
        runner.calls = runner.private / "calls"
        runner.calls.mkdir(parents=True)
        runner.key = "test-placeholder"
        runner.mutex = threading.RLock()
        runner.stop = threading.Event()
        runner.started = {"started_at": g.now(), "started_epoch": time.time()}
        return runner

    def response(self, text):
        payload = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                   "model": "mock", "usage": {"total_tokens": 10}}
        class Response:
            status_code = 200
            content = json.dumps(payload).encode()

            def json(self):
                return payload
        return Response()

    def test_single_dispatch_replay_retains_format_failure(self):
        with tempfile.TemporaryDirectory() as d:
            runner = self.make_runner(d)
            with patch.object(g.requests, "post", return_value=self.response("Two. Sentences.")) as post:
                first = runner.call("d00-positive-s0", [], "carrier")
                second = runner.call("d00-positive-s0", [], "carrier")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first["text"], "Two. Sentences.")
            self.assertFalse(first["surface_audit"]["single_sentence_heuristic"])

    def test_transport_failure_permanently_halts_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            runner = self.make_runner(d)
            with patch.object(g.requests, "post", side_effect=g.requests.Timeout) as post:
                with self.assertRaises(g.PermanentHalt):
                    runner.call("d00-positive-s0", [], "carrier")
                with self.assertRaises(g.PermanentHalt):
                    runner.call("d01-positive-s0", [], "carrier")
            self.assertEqual(post.call_count, 1)
            self.assertTrue((runner.run / "GENERATION_HALT.json").exists())
            record = json.loads((runner.calls / "d00-positive-s0.response.json").read_text())
            self.assertFalse(record["completion_received"])

    def test_orphan_intent_never_redispatched(self):
        with tempfile.TemporaryDirectory() as d:
            runner = self.make_runner(d)
            g.atomic_json(runner.calls / "d00-positive-s0.intent.json", {"dispatch": "uncertain"})
            with patch.object(g.requests, "post") as post:
                with self.assertRaises(g.PermanentHalt):
                    runner.call("d00-positive-s0", [], "carrier")
            post.assert_not_called()

    def test_changed_request_cannot_reuse_response(self):
        with tempfile.TemporaryDirectory() as d:
            runner = self.make_runner(d)
            with patch.object(g.requests, "post", return_value=self.response("Retained text.")) as post:
                runner.call("d00-positive-s0", [], "carrier")
                with self.assertRaises(g.PermanentHalt):
                    runner.call("d00-positive-s0", [{"role": "user", "content": "changed"}], "carrier")
            self.assertEqual(post.call_count, 1)

    def test_pair_own_context_and_full_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            runner = self.make_runner(d)
            runner.prompts = self.prompts
            runner.assignments = self.assignments
            runner.bank = json.loads(g.BANK.read_text())
            requests_seen = []

            def fake_call(call_id, messages, kind):
                requests_seen.append((call_id, messages[-1]["content"]))
                # Deliberately empty one positive slot: preserve all planned positions.
                text = "" if call_id == "d00-positive-s2" else call_id + " completed."
                return {"call_id": call_id, "text": text, "finish_reason": "stop",
                        "surface_audit": g.surface_audit(text)}
            runner.call = fake_call
            runner.run_pair(0)
            self.assertEqual(len(requests_seen), 18)
            for call_id, prompt in requests_seen:
                if "unwatermarked" in call_id:
                    self.assertNotIn("d00-positive", prompt)
                    self.assertNotIn("TARGET AMR", prompt)
            doc = json.loads((runner.private / "documents/d00.json").read_text())
            self.assertEqual(len(doc["arms"]["positive"]["slots"]), 8)
            self.assertEqual(doc["arms"]["positive"]["slots"][2]["text"], "")
            self.assertEqual(len(doc["arms"]["unwatermarked"]["slots"]), 8)


if __name__ == "__main__":
    unittest.main()
