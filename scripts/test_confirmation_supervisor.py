"""Network-free tests for the operational wrapper, not the frozen protocol."""

import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import supervise_confirmation_experiment as supervisor
from run_segmentation_intervention import write_json


class SupervisorTests(unittest.TestCase):
    def test_existing_uncertain_calls_do_not_trigger_new_run_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            call = {"status": "in_flight", "started_at": "2026-09-13T05:00:00+00:00"}
            write_json(folder / "calls/old.json", call)
            write_json(folder / "calls/new.json", call)
            now = datetime(2026, 9, 13, 5, 10, tzinfo=timezone.utc).timestamp()
            state = supervisor.snapshot(folder, {"old.json"}, now)
            self.assertEqual(state["active_calls"], [{"file": "new.json", "age_seconds": 600}])
            self.assertEqual(state["counts"]["evaluated"], 0)

    def test_completed_request_not_timed_out(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "calls/done.json", {"status": "complete"})
            self.assertEqual(supervisor.snapshot(folder, set())["active_calls"], [])

    def test_command_keeps_frozen_budget_and_explicit_recovery(self):
        command = supervisor.runner_command(Path("/tmp/cohort"), 30, 7000)
        self.assertIn("--recover-incomplete-injection", command)
        self.assertNotIn("--amend-unstarted-runtime", command)
        self.assertNotIn("--key-stdin", command)
        self.assertEqual(command[command.index("--limit") + 1], "30")
        self.assertEqual(command[command.index("--max-calls") + 1], "7000")

    def test_guard_verified_and_bound_to_supervisor(self):
        guard = Mock(pid=123, poll=Mock(return_value=None))
        with patch.object(supervisor.subprocess, "Popen", return_value=guard) as launch, \
                patch.object(supervisor.subprocess, "check_output",
                             return_value="pid 123(caffeinate): PreventUserIdleSystemSleep"):
            self.assertIs(supervisor.start_sleep_guard(), guard)
        self.assertEqual(launch.call_args.args[0][:3], ["/usr/bin/caffeinate", "-i", "-w"])

    def test_failed_guard_is_not_accepted(self):
        guard = Mock(pid=123, poll=Mock(return_value=1))
        with patch.object(supervisor.subprocess, "Popen", return_value=guard):
            with self.assertRaisesRegex(RuntimeError, "guard exited"):
                supervisor.start_sleep_guard()

    def test_stop_escalates_only_owned_process(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 10), 0]
        with patch.object(supervisor.os, "killpg") as kill:
            supervisor.stop_process(process, group=True)
        self.assertEqual([c.args for c in kill.call_args_list],
                         [(123, supervisor.signal.SIGTERM), (123, supervisor.signal.SIGKILL)])

    def test_failure_status_persisted_without_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            keyfile = folder / "credential.txt"
            keyfile.write_text("sk-test-only-not-a-real-key\n")
            child = Mock(pid=567, returncode=1, poll=Mock(return_value=1))
            guard = Mock(pid=123, poll=Mock(return_value=None))
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), \
                    patch.object(supervisor, "start_sleep_guard", return_value=guard), \
                    patch.object(supervisor.subprocess, "Popen", return_value=child) as launch, \
                    patch.object(supervisor, "stop_process") as stop:
                result = supervisor.supervise(folder, keyfile, 30, 7000, 300)
            self.assertEqual(result, 1)
            status = json.loads((folder / "run_logs/status.json").read_text())
            self.assertEqual(status["status"], "runner_failed")
            self.assertEqual(status["runner_returncode"], 1)
            self.assertNotIn("sk-test", capture.getvalue())
            for path in (folder / "run_logs").iterdir():
                self.assertNotIn("sk-test", path.read_text())
            self.assertEqual(launch.call_args.kwargs["env"]["DEEPSEEK_API_KEY"],
                             "sk-test-only-not-a-real-key")
            stop.assert_any_call(child, group=True)
            stop.assert_any_call(guard)

    def test_wall_timeout_records_status_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            keyfile = folder / "credential.txt"
            keyfile.write_text("sk-test-only-not-a-real-key\n")
            child = Mock(pid=567, returncode=None, poll=Mock(return_value=None))
            guard = Mock(pid=123, poll=Mock(return_value=None))
            state = {"counts": {}, "active_calls": [{"file": "new.json", "age_seconds": 301}]}
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(supervisor, "start_sleep_guard", return_value=guard), \
                    patch.object(supervisor.subprocess, "Popen", return_value=child) as launch, \
                    patch.object(supervisor, "stop_process"), \
                    patch.object(supervisor, "snapshot", return_value=state):
                self.assertEqual(supervisor.supervise(folder, keyfile, 30, 7000, 300), 1)
            self.assertEqual(launch.call_count, 1)
            status = json.loads((folder / "run_logs/status.json").read_text())
            self.assertEqual(status["status"], "request_wall_timeout")


if __name__ == "__main__":
    unittest.main()
