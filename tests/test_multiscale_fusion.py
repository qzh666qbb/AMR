"""Engineering fixtures only; no synthetic detection-performance evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from src.core.multiscale_data import load_cached_scales, validate_split
from src.core.multiscale_fusion import EmpiricalThreshold, MultiScaleAttentionFusion


ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.train_neg = torch.rand(12, 3, 5)
        self.model = MultiScaleAttentionFusion(n_templates=5)
        self.model.fit_normalizer(self.train_neg)
        self.model.eval()

    def test_requires_normalization_and_refuses_implicit_refit(self):
        with self.assertRaisesRegex(RuntimeError, "normalizer"):
            MultiScaleAttentionFusion(n_templates=5)(self.train_neg)
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            self.model.fit_normalizer(self.train_neg)
        before = self.model.center.clone()
        self.model(torch.ones(4, 3, 5))
        torch.testing.assert_close(self.model.center, before)

    def test_masking_abstention_and_finite_backward(self):
        values = torch.rand(4, 3, 5)
        values[0, 0] = torch.nan
        values[1, :, 2] = torch.nan
        values[2] = torch.nan
        result = self.model(values, return_attention=True)
        self.assertEqual(result["abstain"].tolist(), [False, False, True, False])
        self.assertEqual(float(result["statistic"][2]), -float("inf"))
        self.assertTrue(torch.isfinite(result["scale_weights"]).all())
        self.assertTrue(torch.isfinite(result["cross_attention"]).all())
        self.assertEqual(int(torch.count_nonzero(result["scale_weights"][0, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(result["cross_attention"][0, :, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(result["cross_attention"][0, :, 0, :])), 0)
        torch.testing.assert_close(result["scale_weights"].sum(1),
                                   result["valid_templates"].float())
        # Backpropagate valid rows; abstentions are never fabricated as negatives.
        result["statistic"][[0, 1, 3]].sum().backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(self.model.cross_attention.in_proj_weight.grad.abs().sum()), 0)

    def test_cross_scale_interaction_is_nonzero(self):
        values = torch.rand(1, 3, 5, requires_grad=True)
        result = self.model(values)
        # A local-scale gate responds to a different scale of the same template.
        derivative = torch.autograd.grad(result["scale_weights"][0, 0, 0], values)[0]
        self.assertGreater(float(derivative[0, 1, 0].abs()), 1e-8)

    def test_batching_determinism_checkpoint_and_column_equivariance(self):
        values = torch.rand(7, 3, 5)
        result = self.model(values)["statistic"]
        replay = torch.cat([self.model(x)["statistic"] for x in values.split(2)])
        torch.testing.assert_close(result, replay)
        self.assertTrue(torch.equal(result, self.model(values)["statistic"]))
        restored = MultiScaleAttentionFusion(n_templates=5)
        restored.load_state_dict(self.model.state_dict())
        permutation = torch.tensor([3, 1, 4, 0, 2])
        restored.center.copy_(restored.center[:, permutation])
        restored.spread.copy_(restored.spread[:, permutation])
        permuted = restored(values[:, :, permutation])
        torch.testing.assert_close(result, permuted["statistic"])
        torch.testing.assert_close(self.model(values)["template_scores"][:, permutation],
                                   permuted["template_scores"])

    def test_single_scale_and_all_invalid_never_fall_back(self):
        model = MultiScaleAttentionFusion(n_templates=5, mode="s6")
        model.fit_normalizer(self.train_neg)
        values = torch.rand(3, 3, 5)
        values[0, 0] = torch.nan
        result = model(values)
        self.assertTrue(result["abstain"][0])
        self.assertEqual(int(torch.count_nonzero(result["scale_weights"][:, 1:])), 0)
        z = (values[1:, 0] - model.center[0]) / model.spread[0]
        torch.testing.assert_close(result["template_scores"][1:], z)

    def test_raw_short_baseline_preserves_original_bank_max(self):
        model = MultiScaleAttentionFusion(n_templates=5, mode="raw_s6")
        model.fit_normalizer(self.train_neg)
        values = torch.rand(3, 3, 5)
        result = model(values)
        torch.testing.assert_close(result["template_scores"], values[:, 0])
        torch.testing.assert_close(result["template_scores"].amax(1),
                                   values[:, 0].amax(1))

    def test_bad_inputs_fail(self):
        for values in (torch.full((2, 3, 5), float("inf")),
                       torch.full((2, 3, 5), 1.1), torch.zeros(2, 5),
                       torch.zeros(0, 3, 5)):
            with self.assertRaises(ValueError):
                self.model(values)

    def test_threshold_ties_and_resolution(self):
        threshold = EmpiricalThreshold.fit([1.0] * 100, 0.01)
        self.assertEqual(threshold.predict([1.0, 1.001, -float("inf")]),
                         [False, True, False])
        scarce = EmpiricalThreshold.fit([float(i) for i in range(15)], 0.01)
        self.assertIsNone(scarce.threshold)
        with self.assertRaises(RuntimeError):
            scarce.predict([1.0])
        with self.assertRaises(ValueError):
            EmpiricalThreshold.fit([float("nan")], 0.05)
        # Calibration abstentions stay in n; they do not become finite scores.
        threshold = EmpiricalThreshold.fit([-float("inf")] * 10 + list(range(90)), 0.05)
        self.assertEqual((threshold.n, threshold.n_abstain), (100, 10))
        self.assertLessEqual(sum(threshold.predict([-float("inf")] * 10 + list(range(90)))) / 100,
                             0.05)


class AdapterTests(unittest.TestCase):
    def fixture(self, root, prefix, missing=False):
        main, short = [], []
        for doc in range(8):
            for slot in range(2):
                row = {"doc": f"d{doc}", "attack": "baseline", "slot": slot,
                       "vec_d1": [0.1 + slot * 0.1 + doc * 0.01] * 100,
                       "vec_d2": [0.4 + slot * 0.1 + doc * 0.01] * 100}
                # These fields must not become model inputs.
                row.update(template_index=doc, score_d1_true=999, rank_d1=doc)
                main.append(row)
                if not (missing and doc == 7 and slot == 1):
                    short.append(dict(row))
        mp, sp = root / f"{prefix}_main.json", root / f"{prefix}_short.json"
        mp.write_text(json.dumps({"per_instance": main}))
        sp.write_text(json.dumps({"per_instance": short}))
        return mp, sp

    def test_union_no_oracle_fields_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = self.fixture(root, "negative", missing=True)
            data = load_cached_scales(*paths)
            self.assertEqual(len(data.keys), 16)
            self.assertEqual(data.missing_short_rows, 1)
            self.assertTrue(torch.isnan(data.scores[-1, 0]).all())
            raw = json.loads(paths[0].read_text())
            for row in raw["per_instance"]:
                row["template_index"] = 50000
                row["score_d1_true"] = -999
            paths[0].write_text(json.dumps(raw))
            torch.testing.assert_close(data.scores, load_cached_scales(*paths).scores,
                                       equal_nan=True)
            split = {"train": ["d0", "d1"], "validation": ["d2", "d3"],
                     "calibration": ["d4", "d5"], "test": ["d6", "d7"]}
            validate_split(split, data, data)
            split["test"].append("d0")
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_split(split, data, data)
            raw["per_instance"].append(raw["per_instance"][0])
            paths[0].write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_cached_scales(*paths)

    def test_cli_fit_checkpoint_and_insufficient_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pm, ps = self.fixture(root, "positive")
            nm, ns = self.fixture(root, "negative", missing=True)
            split = root / "split.json"
            split.write_text(json.dumps({
                "train": ["d0", "d1"], "validation": ["d2", "d3"],
                "calibration": ["d4", "d5"], "test": ["d6", "d7"]}))
            output = root / "output"
            command = [sys.executable, str(ROOT / "scripts/run_multiscale_fusion.py"),
                       "--mode", "fit", "--positive-main", str(pm), "--positive-short", str(ps),
                       "--negative-main", str(nm), "--negative-short", str(ns),
                       "--split", str(split), "--epochs", "1", "--batch-size", "4",
                       "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "RESULT.json").read_text())
            self.assertEqual(report["status"], "exploratory_oracle_slot_evaluation")
            self.assertEqual(report["evaluation"]["test_negative_n"], 4)
            for working_point in report["evaluation"]["working_points"].values():
                self.assertEqual(working_point["calibration"]["status"],
                                 "insufficient_calibration")
                self.assertNotIn("tpr", working_point)
            checkpoint = torch.load(output / "MODEL.pt", weights_only=True)
            model = MultiScaleAttentionFusion(**checkpoint["config"])
            model.load_state_dict(checkpoint["state_dict"])
            self.assertTrue(model.normalizer_ready)
            # Existing result directories are immutable to the CLI.
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
