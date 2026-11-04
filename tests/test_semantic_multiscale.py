"""Safety/correctness checks on real sealed cached scores, not performance tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/core"), str(ROOT / "scripts")]
from semantic_multiscale import MultiScaleSymbolReadout, scores_from_observation
from semantic_sequence import exact_costs
from audit_semantic_readout import full_dp


class SemanticMultiScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        mapping = json.loads((ROOT / "experiments/semantic_sequence_readout1/SYMBOL_MAP.json").read_text())
        ids = sorted(map(int, mapping["template_to_symbol"]))
        cls.groups = [[j for j, t in enumerate(ids)
                       if mapping["template_to_symbol"][str(t)] == c] for c in "ABCD"]
        source = Path.home() / ".cache/amr/semantic_channel_audit1/predictions"
        cls.row = json.loads((source / "d12-positive-clean.json").read_text())
        cls.x = torch.tensor([scores_from_observation(o) for o in cls.row["observations"]])

    def model(self, mode="attention"):
        torch.manual_seed(20261026)
        model = MultiScaleSymbolReadout(self.groups, mode)
        model.fit_normalizer(self.x)
        return model

    def test_checkpoint_and_batch_replay(self):
        model = self.model().eval()
        with torch.no_grad():
            whole = model(self.x)["q"]
            split = torch.cat([model(self.x[:3])["q"], model(self.x[3:])["q"]])
        self.assertTrue(torch.allclose(whole, split, atol=1e-6, rtol=0))
        other = MultiScaleSymbolReadout(self.groups)
        other.load_state_dict(copy.deepcopy(model.state_dict()))
        other.eval()
        with torch.no_grad():
            replay = other(self.x)["q"]
        self.assertTrue(torch.equal(whole, replay))

    def test_partial_and_all_missing_evidence(self):
        model = self.model()
        x = self.x[:4].clone()
        x[0] = float("nan")
        x[1, 0] = float("nan")
        output = model(x)
        self.assertTrue(bool(output["abstain"][0]))
        self.assertTrue(torch.isfinite(output["q"]).all())
        self.assertTrue(torch.equal(output["scale_weights"][0],
                                    torch.zeros_like(output["scale_weights"][0])))
        self.assertTrue(torch.equal(output["scale_weights"][1, 0],
                                    torch.zeros_like(output["scale_weights"][1, 0])))
        (-output["q"].clamp_min(1e-9).log().mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_template_reordering_preserves_public_symbols(self):
        model = self.model().eval()
        perm = list(reversed(range(24)))
        inverse = {old: new for new, old in enumerate(perm)}
        groups = [[inverse[old] for old in g] for g in self.groups]
        reordered = MultiScaleSymbolReadout(groups)
        state = copy.deepcopy(model.state_dict())
        for name in ("fusion.center", "fusion.spread", "fusion.normalizer_count"):
            state[name] = state[name][:, perm]
        reordered.load_state_dict(state)
        reordered.eval()
        self.assertTrue(torch.allclose(model(self.x)["q"], reordered(self.x[:, :, perm])["q"],
                                       atol=1e-6, rtol=0))

    def test_forbidden_target_inputs_and_invalid_scores(self):
        model = self.model()
        with self.assertRaises(TypeError):
            model(self.x, gold_symbol=0)
        with self.assertRaises(ValueError):
            model(torch.full_like(self.x, float("inf")))
        obs = copy.deepcopy(self.row["observations"][0])
        obs.update(gold_symbol=3, true_position=99, condition="ignored", template_index=47)
        self.assertEqual(scores_from_observation(obs), scores_from_observation(self.row["observations"][0]))

    def test_full_candidate_dp_matches_independent_layout(self):
        model = self.model().eval()
        output = model(self.x)
        q = output["q"].detach().numpy().astype(np.float64)
        q /= q.sum(1, keepdims=True)
        weights = [None if missing else value.tolist()
                   for value, missing in zip(q, output["abstain"].tolist())]
        indices = np.asarray([0, 137, 2520, 65535], dtype=np.int32)
        candidates = ((indices[:, None] // (4 ** np.arange(7, -1, -1))) % 4).astype(np.int8)
        self.assertTrue(np.allclose(exact_costs(weights, candidates),
                                    full_dp(weights, candidates), atol=1e-10, rtol=0))


if __name__ == "__main__":
    unittest.main()
