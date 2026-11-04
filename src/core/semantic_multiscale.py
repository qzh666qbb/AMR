"""Score-level cross-scale fusion for semantic symbols, without target inputs.

This adapts the existing SWAN fusion architecture to a public four-symbol
dictionary. It does not import its binary detector head or old checkpoints.
The normalizer is fitted on all *training* candidate scores in this task;
reference-unmatched candidates are not asserted to be semantic negatives.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from multiscale_fusion import MultiScaleAttentionFusion


VERSION = "semantic-score-msa-v1"
SCALES = ("W6", "W12", "E1_12")
MODES = ("s6", "s12", "ssub", "mean", "gate", "attention", "gate_wide")


def scores_from_observation(observation):
    """Use the same text-only anchor for all three views; never read labels."""
    raw = observation["raw_scores"]
    vectors = (raw["W"][0], raw["W"][1], raw["E1"][1])
    widths = [window["width"] for window in observation["windows"]]
    if widths != [6, 12]:
        raise ValueError("Expected the frozen 6/12-word candidate windows")
    n_templates = next((len(v) for v in vectors if v is not None), 24)
    result = []
    for vector in vectors:
        if vector is None:
            result.append([float("nan")] * n_templates)
        else:
            if len(vector) != n_templates:
                raise ValueError("Inconsistent dictionary dimensions")
            result.append([float("nan") if x is None else float(x) for x in vector])
    return result


class MultiScaleSymbolReadout(nn.Module):
    """Return four soft weights per candidate from [batch, 3, templates].

    Public template groups determine the alphabet; no template-index embedding,
    source message, gold position, document ID or condition enters forward().
    Incomplete views are masked. Missing all evidence produces an erasure.
    Hard core-fact constraints, when available, must be applied upstream to
    every compared method; attention cannot override a forbidden mapping.
    """

    def __init__(self, groups, mode="attention"):
        super().__init__()
        members = [x for group in groups for x in group]
        if (mode not in MODES or len(groups) != 4
                or len(set(map(len, groups))) != 1 or not groups[0]
                or sorted(members) != list(range(len(members)))):
            raise ValueError("Four equal public groups must partition the dictionary")
        self.groups = tuple(tuple(group) for group in groups)
        self.mode = mode
        # The wider gate is a capacity control for attention, without Q/K/V.
        hidden = 36 if mode == "gate_wide" else 16
        self.fusion = MultiScaleAttentionFusion(
            n_templates=len(members), hidden_dim=hidden, n_heads=2,
            mode="gate" if mode == "gate_wide" else mode)
        self.fusion.log_scale.requires_grad_(False)
        self.fusion.bias.requires_grad_(False)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.class_bias = nn.Parameter(torch.zeros(4))

    @torch.no_grad()
    def fit_normalizer(self, training_candidate_scores):
        # Numerically the same training-only mean/std routine as the original
        # architecture; this task deliberately does not use a negative label.
        self.fusion.fit_normalizer(training_candidate_scores)

    def forward(self, scores):
        output = self.fusion(scores)
        pooled, class_valid = [], []
        for group in self.groups:
            values = output["template_scores"][:, group]
            valid = output["valid_templates"][:, group]
            present = valid.any(1)
            # Prevent logsumexp(-inf,...) from creating NaN gradients on a
            # fully missing class. It stays masked after pooling.
            safe = values.masked_fill(~present[:, None], 0.0)
            value = torch.logsumexp(safe, dim=1) - math.log(len(group))
            pooled.append(value)
            class_valid.append(present)
        class_valid = torch.stack(class_valid, dim=1)
        pooled = torch.stack(pooled, dim=1)
        logits = F.softplus(self.log_scale) * pooled + self.class_bias
        logits = logits.masked_fill(~class_valid, -torch.inf)
        abstain = ~class_valid.any(1)
        logits = logits.masked_fill(abstain[:, None], 0.0)
        q = torch.softmax(logits, dim=1)
        return {
            "logits": logits, "q": q, "abstain": abstain,
            "class_valid": class_valid, "scale_weights": output["scale_weights"],
        }

    def configuration(self):
        return {
            "version": VERSION, "mode": self.mode, "groups": self.groups,
            "scales": SCALES, "fusion": self.fusion.config,
            "normalizer_population": "all training candidate observations",
            "trainable_parameters": sum(p.numel() for p in self.parameters()
                                        if p.requires_grad),
        }
