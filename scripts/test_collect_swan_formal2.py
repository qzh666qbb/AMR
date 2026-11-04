"""Synthetic monitor tests; never contact a worker or use research observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_swan_formal2 as monitor
import swan_successors2_snapshot as remote


def snapshot(phase="parsing", complete=False):
    jobs = {}
    for name in monitor.METRIC_JOBS:
        jobs[name] = {
            "machine_closure": complete,
            "completion_contract_verified": complete,
            "HEARTBEAT.json": {"phase": phase},
            "queue/QUEUE_STATUS.json": {"status": "success" if complete else "running"},
            "RESULT.json": {"sha256": "result"},
            "AUDIT.json": {"sha256": "audit"},
        }
        if complete:
            jobs[name]["aggregate_metrics"] = {"synthetic": True}
    return {"at_unix": 1., "jobs": jobs}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.output = patch.object(monitor, "OUT", self.out)
        self.output.start()
        self.addCleanup(self.output.stop)

    def events(self):
        return [json.loads(line) for line in
                (self.out / "MONITOR_EVENTS.jsonl").read_text().splitlines()]

    @patch.object(monitor.time, "sleep")
    def test_phase_transitions_do_not_end_monitor(self, sleep):
        sequence = [snapshot(), snapshot("scoring"), snapshot("done", True)]
        with patch.object(monitor, "fetch", side_effect=sequence) as fetch:
            monitor.watch(1)
        self.assertEqual(fetch.call_count, 3)
        self.assertIn("phase_transition", [e["event"] for e in self.events()])
        self.assertEqual(self.events()[-1]["event"], "all_machine_metrics_audited")
        self.assertFalse(json.loads((self.out / "MACHINE_METRICS.json").read_text())[
            "scientific_closure"])

    @patch.object(monitor.time, "sleep")
    def test_repeated_query_errors_alert_and_recover(self, sleep):
        with patch.object(monitor, "fetch", side_effect=[
                TimeoutError(), TimeoutError(), TimeoutError(), snapshot(complete=True)]):
            monitor.watch(1)
        events = self.events()
        error = next(e for e in events if e["event"] == "observation_unavailable")
        self.assertFalse(error["experiment_failure_inferred"])
        self.assertIn("observation_restored", [e["event"] for e in events])
        self.assertEqual(events[-1]["event"], "all_machine_metrics_audited")

    def test_receipt_and_metrics_both_required(self):
        data = snapshot(complete=True)
        data["jobs"]["normal"]["completion_contract_verified"] = False
        data["jobs"]["confirmation"]["machine_closure"] = False
        data["jobs"]["endpoint"].pop("aggregate_metrics")
        ready = monitor.audited_metrics(data)
        self.assertNotIn("normal", ready)
        self.assertNotIn("confirmation", ready)
        self.assertNotIn("endpoint", ready)

    def test_parent_failure_and_pause_are_attention_not_success(self):
        data = snapshot()
        data["jobs"]["normal"]["PAUSED.json"] = {"status": "paused"}
        data["jobs"]["endpoint"]["queue/endpoint_TERMINAL.json"] = {"status": "blocked"}
        data["jobs"]["formal_alignment"]["COORDINATOR_STATUS.json"] = {"status": "blocked"}
        self.assertEqual(set(monitor.blockers(data)), {"normal", "endpoint", "formal_alignment"})
        self.assertFalse(monitor.audited_metrics(data))

    def test_receipt_cannot_outlive_mutated_artifact(self):
        runtime = self.out / "runtime"
        queue = self.out / "queue"
        runtime.mkdir()
        queue.mkdir()
        monitor.atomic(runtime / "RESULT.json", {"status": "DONE"})
        monitor.atomic(runtime / "AUDIT.json", {"synthetic": True})
        monitor.atomic(runtime / "TERMINAL.json", {
            "status": "DONE", "result_sha256": remote.sha(runtime / "RESULT.json"),
            "audit_sha256": remote.sha(runtime / "AUDIT.json")})
        job = {"id": "test", "completion": {
            "terminal": {"path": str(runtime / "TERMINAL.json"), "equals": {"status": "DONE"},
                         "sha256_fields": {"result_sha256": "result", "audit_sha256": "seal"}},
            "result": {"path": str(runtime / "RESULT.json"), "equals": {"status": "DONE"}},
            "seal": {"path": str(runtime / "AUDIT.json")}}}
        monitor.atomic(queue / "MANIFEST.json", {"jobs": [job]})
        monitor.atomic(queue / "test_TERMINAL.json", {
            "status": "success", "returncode": 0, "job": "test",
            "schema": "research-completion-queue/v2",
            "manifest_sha256": remote.sha(queue / "MANIFEST.json"),
            "artifacts": {k: {"path": v["path"], "sha256": remote.sha(Path(v["path"]))}
                          for k, v in job["completion"].items()}})
        self.assertTrue(remote.verify_receipt(runtime, queue, "test"))
        monitor.atomic(runtime / "RESULT.json", {"status": "DONE", "changed": True})
        with self.assertRaises(AssertionError):
            remote.verify_receipt(runtime, queue, "test")

    @patch.object(monitor.time, "sleep")
    def test_alert_does_not_stop_independent_progress(self, sleep):
        blocked = snapshot()
        blocked["jobs"]["endpoint"]["HALT.json"] = {"status": "HALT"}
        with patch.object(monitor, "fetch", side_effect=[blocked, snapshot(complete=True)]):
            monitor.watch(1)
        self.assertIn("experiment_attention_required", [e["event"] for e in self.events()])
        self.assertEqual(self.events()[-1]["event"], "all_machine_metrics_audited")

    def test_notification_ignores_phase_changes(self):
        monitor.emit("phase_transition")
        monitor.emit("audited_metrics_available", jobs=["calibration"])
        before = copy.deepcopy(self.events())
        with patch("builtins.print") as output:
            monitor.notify(0)
        self.assertEqual(json.loads(output.call_args.args[0])["event"], "audited_metrics_available")
        self.assertEqual(self.events(), before)


if __name__ == "__main__":
    unittest.main()
