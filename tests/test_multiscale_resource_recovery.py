"""Resource admission fixtures do not provide scientific performance evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_multiscale_transfer.py"
spec = importlib.util.spec_from_file_location("transfer_resource_test", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ResourceRecoveryTests(unittest.TestCase):
    def test_shared_gpu_pressure_records_pause_without_scientific_halt(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            freeze = {"execution": {"resumable_gpu_pressure": True},
                      "resource_gate": {"vram_fraction_max": 0.8}}
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
                    stdout="19500, 24576\n")):
                with self.assertRaises(runner.ResourcePause):
                    runner.gpu_capacity(out, freeze, "before_parse", 512)
            snapshot = json.loads((out / "GPU_RESOURCE_STATE.json").read_text())
            self.assertFalse(snapshot["admitted"])
            self.assertEqual(snapshot["used_mib"], 19500)
            self.assertFalse((out / "HALT.json").exists())

    def test_available_gpu_admits_without_changing_legacy_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(runner.subprocess, "run") as command:
                runner.gpu_capacity(out, {}, "before_parse", 512)
                command.assert_not_called()
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
                    stdout="946, 24576\n")):
                runner.gpu_capacity(out, {
                    "execution": {"resumable_gpu_pressure": True},
                    "resource_gate": {"vram_fraction_max": 0.8}}, "before_model_load", 3072)
            self.assertTrue(json.loads((out / "GPU_RESOURCE_STATE.json").read_text())["admitted"])


if __name__ == "__main__":
    unittest.main()
