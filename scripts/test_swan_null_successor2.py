"""Synthetic contracts: no provider network access."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import swan_null_successor2 as new
from swan_final1_common import freeze, sha256


def response(text="Alice carefully inspected the newly repaired laboratory equipment."):
    return {"id": "synthetic", "model": new.budget.MODEL,
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / "parent"
        self.out = Path(self.temp.name) / "successor"
        for path in (self.parent, self.out):
            freeze(path / "FREEZE.json", {"path": path.name, "synthetic": True})
        new.budget.CAPS = dict(new.normal.CAPS)
        self.old = new.budget.Journal(self.parent, "llm_null", 100,
                                     lambda _: response(), sha256(self.parent / "FREEZE.json"))
        self.cid = "source-null-slot-00-attempt-00"
        self.args = ("context", "", 50, .7, .9)

    def test_old_complete_never_sent_again(self):
        self.old.request(self.cid, *self.args)
        sent = []
        j = new.Journal(self.out, self.parent, lambda body: sent.append(body))
        with self.assertRaises(new.budget.GlobalStop):
            j.request(self.cid, *self.args)
        self.assertEqual(sent, [])
        self.assertEqual(j.totals()["calls"], 1)
        self.assertEqual(j.totals()["exposure_nanodollars"], 17400)

    def test_unknown_reserve_carried_once(self):
        def disconnect(_):
            raise ConnectionError("synthetic")
        self.old.send = disconnect
        with self.assertRaises(new.budget.DocumentFailure):
            self.old.request(self.cid, *self.args)
        j = new.Journal(self.out, self.parent, lambda _: response())
        self.assertEqual(new.unavailable_sources(j.previous), {"source"})
        self.assertEqual(j.totals()["exposure_nanodollars"], self.old.totals()["exposure_nanodollars"])
        self.assertEqual(j.totals()["uncertain_or_invalid_calls"], 1)
        j.request("other-null-slot-00-attempt-00", *self.args)
        self.assertEqual(j.totals()["calls"], 2)
        self.assertEqual(j.totals()["exposure_nanodollars"],
                         self.old.totals()["exposure_nanodollars"] + 17400)

    def test_own_checkpoint_replay_has_no_network(self):
        sent = []
        def send(body):
            sent.append(body)
            return response()
        j = new.Journal(self.out, self.parent, send)
        first = j.request(self.cid, *self.args)
        again = new.Journal(self.out, self.parent, send)
        self.assertEqual(first, again.request(self.cid, *self.args))
        self.assertEqual(len(sent), 1)
        with self.assertRaises(new.budget.GlobalStop):
            again.request(self.cid, "different", "", 50, .7, .9)

    def test_new_uncertain_is_nonreplayable(self):
        def disconnect(_):
            raise TimeoutError("synthetic")
        j = new.Journal(self.out, self.parent, disconnect)
        with self.assertRaises(new.budget.DocumentFailure):
            j.request(self.cid, *self.args)
        with self.assertRaises(new.budget.DocumentFailure):
            new.Journal(self.out, self.parent, lambda _: self.fail("network")).request(
                self.cid, *self.args)
        self.assertEqual(new.unavailable_sources(j), {"source"})

    def test_invalid_model_remains_permanent_halt(self):
        bad = response()
        bad["model"] = "different-model"
        j = new.Journal(self.out, self.parent, lambda _: bad)
        with self.assertRaises(new.budget.GlobalStop):
            j.request(self.cid, *self.args)
        with self.assertRaises(new.budget.GlobalStop):
            new.Journal(self.out, self.parent, None)

    def test_empty_paid_candidate_is_not_retried(self):
        calls = []
        def send(body):
            calls.append(body)
            return response("")
        j = new.Journal(self.out, self.parent, send)
        self.assertEqual(j.request(self.cid, *self.args)["text"], "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(j.totals()["uncertain_or_invalid_calls"], 0)

    def test_original_exposure_enforces_total_cap(self):
        self.old.request(self.cid, *self.args)
        j = new.Journal(self.out, self.parent, lambda _: self.fail("network"))
        with patch.dict(new.budget.CAPS, {"llm_null": (640000, 17400)}):
            with self.assertRaises(new.budget.Pause):
                j.request("other-null-slot-00-attempt-00", *self.args)
        self.assertEqual(len(j.calls), 0)

    def test_identity_rejects_unmapped_calls(self):
        for cid in ("invalid", "source-null-slot-xx-attempt-00", "source-null-slot-00-attempt-0"):
            with self.assertRaises(new.budget.GlobalStop):
                new.source_of(cid)
        self.assertEqual(new.source_of(self.cid), "source")


if __name__ == "__main__":
    unittest.main()
