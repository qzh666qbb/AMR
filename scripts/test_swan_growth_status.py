"""Status inspection must distinguish completion from an abandoned heartbeat."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_confirmation_experiment import freeze
from swan_growth_status import snapshot


class StatusTests(unittest.TestCase):
    def test_not_started(self):
        with tempfile.TemporaryDirectory() as d:
            result = snapshot(Path(d))
            self.assertEqual(result["state"], "not_started")
            self.assertEqual(result["api_attempted"], 0)

    def fixture(self, folder, status=None, complete=False):
        freeze(folder / "run_logs/run.json", {
            "pid": 123, "stage": "injection", "status": status,
            "updated_at": "2026-09-15T00:00:00+00:00"})
        freeze(folder / "summary.json", {
            "status": "complete_development_run" if complete else "incomplete_development_run"})

    def test_completed_run_with_old_heartbeat(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.fixture(folder, "complete", True)
            with patch("swan_growth_status.process_exists", return_value=False):
                self.assertEqual(snapshot(folder)["state"], "complete")

    def test_dead_process_not_reported_running(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.fixture(folder)
            with patch("swan_growth_status.process_exists", return_value=False):
                self.assertEqual(snapshot(folder)["state"], "stopped_without_completion")

    def test_live_process_stale_heartbeat_and_call_counts(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.fixture(folder)
            for index, status in enumerate(("complete", "in_flight", "failed_or_uncertain")):
                freeze(folder / "calls" / f"{index}.json", {"status": status})
            now = datetime(2026, 9, 15, 0, 2, tzinfo=timezone.utc)
            with patch("swan_growth_status.process_exists", return_value=True):
                result = snapshot(folder, now)
            self.assertEqual(result["state"], "process_present_heartbeat_stale")
            self.assertEqual(result["heartbeat_age_seconds"], 120)
            self.assertEqual((result["api_complete"], result["api_in_flight"],
                              result["api_failed_or_uncertain"]), (1, 1, 1))

    def test_comparison_layout_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"protocol": {
                "sentence_slots_per_arm": 5, "arms": ["upstream", "local_adapted", "unwatermarked"]}})
            freeze(folder / "summary.json", {"status": "complete_development_comparison"})
            freeze(folder / "run_logs/run.json", {
                "pid": 123, "stage": "complete", "status": "complete",
                "updated_at": "2026-09-15T00:00:00+00:00"})
            freeze(folder / "sentences/upstream/000.json", {})
            freeze(folder / "evaluated/upstream.json", {})
            with patch("swan_growth_status.process_exists", return_value=False):
                result = snapshot(folder)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["completed_sentence_slots"], 1)
            self.assertEqual(result["planned_sentence_slots"], 15)
            self.assertEqual(result["baseline_prefixes"], 1)

    def test_grounded_workflow_counts_calls_not_sentence_slots(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"protocol": {
                "workflow": "grounded_generation", "generation_call_count": 18}})
            freeze(folder / "summary.json", {"status": "complete_grounded_comparison"})
            freeze(folder / "run_logs/run.json", {
                "pid": 123, "stage": "complete", "status": "complete",
                "updated_at": "2026-09-15T00:00:00+00:00"})
            freeze(folder / "generated/p-onepass-0.json", {})
            freeze(folder / "calls/style-p-0.json", {"scope": "style-p", "status": "complete"})
            freeze(folder / "calls/facts-p-0.json", {"scope": "facts-p", "status": "complete"})
            freeze(folder / "calls/facts-p-1.json", {"scope": "facts-p", "status": "in_flight"})
            with patch("swan_growth_status.process_exists", return_value=False):
                result = snapshot(folder)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["completed_generation_calls"], 1)
            self.assertEqual(result["planned_generation_calls"], 18)
            self.assertIsNone(result["completed_sentence_slots"])
            self.assertEqual(result["quality_reviews"], 2)

    def test_injection_layout_total_all_documents(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            freeze(folder / "manifest.json", {"protocol": {"workflow": "grounded_injection", "total_sentence_slots": 50}})
            freeze(folder / "summary.json", {"status": "complete_grounded_injection_pilot"})
            freeze(folder / "run_logs/run.json", {"pid": 123, "stage": "complete", "status": "complete",
                                                 "updated_at": "2026-09-15T00:00:00+00:00"})
            freeze(folder / "sentences/p-control-0.json", {})
            freeze(folder / "evaluated/p-control.json", {})
            with patch("swan_growth_status.process_exists", return_value=False):
                result = snapshot(folder)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["planned_sentence_slots"], 50)
            self.assertEqual(result["completed_sentence_slots"], 1)
            self.assertEqual(result["baseline_prefixes"], 1)


if __name__ == "__main__":
    unittest.main()
