"""Offline contracts for missing calls, original records and all 250 allocations."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import swan_confirmation_successor2 as new
from swan_final1_common import digest, freeze, load, sha256


def response(text="Alice carefully inspected the newly repaired laboratory equipment."):
    return {"id": "synthetic", "model": new.BUDGET.MODEL,
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent, self.out = [Path(self.temp.name) / n for n in ("parent", "successor")]
        for p in (self.parent, self.out):
            freeze(p / "FREEZE.json", {"synthetic": p.name})
        new.BUDGET.CAPS = {"pilot": (40550, 25_000_000_000)}
        self.old = new.BUDGET.Journal(self.parent, "pilot", 40550, lambda _: response(),
                                     sha256(self.parent / "FREEZE.json"))
        self.cid = "source-slot-00-attempt-00"
        self.args = ("context", "", 50, .7, .9)

    def test_no_old_request_redispatch(self):
        self.old.request(self.cid, *self.args)
        j = new.Journal(self.out, self.parent, lambda _: self.fail("network"))
        with self.assertRaises(new.BUDGET.GlobalStop):
            j.request(self.cid, *self.args)
        self.assertEqual(j.totals()["calls"], 1)
        self.assertEqual(j.totals()["exposure_nanodollars"], 17400)

    def missing(self, journal, cid):
        def disconnect(_):
            raise ConnectionError("synthetic")
        journal.send = disconnect
        with self.assertRaises(new.BUDGET.DocumentFailure):
            journal.request(cid, *self.args)

    def test_missing_cost_once_and_full_cap(self):
        self.missing(self.old, self.cid)
        j = new.Journal(self.out, self.parent, lambda _: self.fail("network"))
        self.assertEqual(new.unavailable(j.previous), {"source"})
        self.assertEqual(j.totals()["exposure_nanodollars"], self.old.totals()["exposure_nanodollars"])
        with patch.dict(new.BUDGET.CAPS, {"pilot": (40550, j.totals()["exposure_nanodollars"])}):
            with self.assertRaises(new.BUDGET.Pause):
                j.request("other-slot-00-attempt-00", *self.args)
        self.assertEqual(len(j.calls), 0)

    def test_new_missing_never_replayed(self):
        j = new.Journal(self.out, self.parent)
        self.missing(j, self.cid)
        again = new.Journal(self.out, self.parent, lambda _: self.fail("network"))
        with self.assertRaises(new.BUDGET.DocumentFailure):
            again.request(self.cid, *self.args)
        self.assertEqual(new.unavailable(again), {"source"})

    def test_complete_checkpoint_and_empty_response(self):
        sent = []
        j = new.Journal(self.out, self.parent, lambda body: (sent.append(body), response(""))[1])
        first = j.request(self.cid, *self.args)
        self.assertEqual(first["text"], "")
        again = new.Journal(self.out, self.parent, lambda _: self.fail("network"))
        self.assertEqual(again.request(self.cid, *self.args), first)
        self.assertEqual(len(sent), 1)
        with self.assertRaises(new.BUDGET.GlobalStop):
            again.request(self.cid, "changed", "", 50, .7, .9)

    def test_invalid_model_still_halts(self):
        j = new.Journal(self.out, self.parent, lambda _: {**response(), "model": "wrong"})
        with self.assertRaises(new.BUDGET.GlobalStop):
            j.request(self.cid, *self.args)
        with self.assertRaises(new.BUDGET.GlobalStop):
            new.Journal(self.out, self.parent)

    def test_call_mapping(self):
        self.assertEqual(new.source_of(self.cid), "source")
        for cid in ("bad", "source-slot-0-attempt-00", "source-slot-00-attempt-x0"):
            with self.assertRaises(new.BUDGET.GlobalStop):
                new.source_of(cid)

    def test_full_cohort_inheritance_missingness_and_restart(self):
        records = [{"allocation_id": f"s-{i}", "length_bin": 0,
                    "window_sha256": digest(str(i)), "generation_prompt": "Synthetic context."}
                   for i in range(250)]
        self.old.request("s-0-slot-00-attempt-00", *self.args)
        self.missing(self.old, "s-1-slot-00-attempt-00")
        inherited = {
            "allocation_id": "s-0", "source_window_sha256": records[0]["window_sha256"],
            "contract_sha256": sha256(self.parent / "FREEZE.json"), "length_bin": 0,
            "text": "", "sentences": [], "automatic_qualified": False,
            "failure": "slot_attempts_exhausted", "slot_sha256": {}, "human_qualified": None,
            "detector_feedback_used": False}
        relative = "private/documents/s-0.json"
        freeze(self.parent / relative, inherited)
        freeze(self.out / relative, inherited)
        freeze(self.out / "SOURCE_REFERENCE.json", {
            "inherited_documents": {relative: sha256(self.parent / relative)}})
        bank = list(range(100))
        freeze(self.out / "GENERATION_ASSIGNMENT.json", new.run.shared.assignment(records, bank))
        sent = []

        def send(body, key):
            sent.append(body)
            return response()

        @contextmanager
        def runtime(out):
            yield SimpleNamespace(evaluate=lambda *a: None)

        def slot(module, source, slot, context, bank, journal):
            cid = source["allocation_id"] + "-slot-00-attempt-00"
            result = journal.request(cid, *self.args)
            identity = {"allocation_id": source["allocation_id"], "slot": slot,
                        "contract_sha256": sha256(self.out / "FREEZE.json"),
                        "context_sha256": digest(context)}
            attempt = self.out / "private/attempts" / (cid + ".json")
            freeze(attempt, {"response_sha256": result["call_sha256"], **identity})
            return freeze(self.out / "private/slots" / (source["allocation_id"] + "-slot-00.json"), {
                **identity, "text": result["text"], "accepted": False,
                "attempt_file_hashes": {cid: sha256(attempt)}})

        with ExitStack() as stack:
            for obj, name, value in (
                (new, "OUT", self.out), (new, "PARENT", self.parent),
                (new, "verify", lambda: None), (new, "gpu_guard", lambda: None),
                (new.run, "OUT", self.out), (new.run, "journal", new.journal),
                (new.run, "validate_document", new.validate_document),
                (new.run, "read_sources", lambda phase: records),
                (new.run, "review_packet", lambda: None),
                (new.run.shared, "read_bank", lambda: bank),
                (new.run.shared, "tokenizer", lambda: lambda t: [t] if t else []),
                (new.run.shared, "runtime", runtime), (new.run.shared, "space_check", lambda: None),
                (new.run.shared, "readiness", lambda *a: {"synthetic": True}),
                (new.run.GENERATOR, "run_slot", slot),
                (new.run.GENERATOR.pilot.credential, "read_credential", lambda p: "synthetic"),
                (new.BUDGET, "transport", send),
            ):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(patch.object(new.run.GENERATOR.pilot, "LENGTH",
                                           new.run.GENERATOR.pilot.LENGTH))
            new.generate(None)
            complete = new.run.verify_generation()
            self.assertEqual(len(complete["documents"]), 250)
            self.assertEqual(complete["budget"]["calls"], 250)
            self.assertEqual(complete["budget"]["uncertain_or_invalid_calls"], 1)
            self.assertEqual(len(sent), 248)
            self.assertEqual(complete["automatic_qualified"], 0)
            self.assertEqual(sha256(self.out / relative), sha256(self.parent / relative))
            self.assertEqual(load(self.out / "private/documents/s-1.json")["failure"],
                             "missing_response_source_unavailable")
            new.generate(None)
            self.assertEqual(len(sent), 248)
            new.run.verify_generation()
            (self.out / relative).write_text("{}")
            with self.assertRaises(new.run.IntegrityHalt):
                new.validate_document(records[0], self.out / relative)


if __name__ == "__main__":
    unittest.main()
