"""Experimental score-level cross-scale attention, downstream of S2Match.

Inputs are [batch, scale, bank_template], never target-template indices.
This is not token/graph attention and does not modify the frozen SWAN detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


VERSION = "swan-score-msa-v1"
SCALES = ("S6", "S12", "Ssub")
MODES = ("attention", "gate", "mean", "s6", "s12", "ssub", "raw_s6")


class MultiScaleAttentionFusion(nn.Module):
    """Exchange information across scales *for each bank template*.

    Normalization must be fitted on training negatives, not validation,
    calibration or test data. Missing scores are NaN; a missing view has zero
    weight. All-missing examples abstain, with statistic=-inf.
    """

    def __init__(self, n_templates=100, hidden_dim=16, n_heads=2,
                 mode="attention", std_floor=0.05):
        super().__init__()
        if (n_templates < 1 or hidden_dim < 1 or n_heads < 1
                or hidden_dim % n_heads or mode not in MODES
                or not math.isfinite(std_floor) or std_floor <= 0):
            raise ValueError("Invalid fusion configuration")
        self.config = dict(n_templates=n_templates, hidden_dim=hidden_dim,
                           n_heads=n_heads, mode=mode, std_floor=std_floor)
        self.mode = mode
        self.register_buffer("center", torch.zeros(3, n_templates))
        self.register_buffer("spread", torch.ones(3, n_templates))
        self.register_buffer("normalizer_count", torch.zeros(3, n_templates, dtype=torch.long))
        self.register_buffer("normalizer_ready", torch.tensor(False))
        # No template embedding: a bank-column index cannot become a learned label.
        if mode in ("attention", "gate"):
            self.project = nn.Linear(2, hidden_dim)
            self.scale_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
            nn.init.normal_(self.scale_embedding, std=0.02)
            if mode == "attention":
                self.cross_attention = nn.MultiheadAttention(
                    hidden_dim, n_heads, dropout=0.0, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.Tanh(), nn.Linear(hidden_dim, 1))
        # A monotone affine logit head permits BCE training. Its output is NOT
        # a calibrated probability or the official SWAN z statistic.
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-2.5))

    def _validate(self, scores):
        if (scores.ndim != 3 or tuple(scores.shape[1:]) != tuple(self.center.shape)
                or scores.shape[0] == 0 or not scores.is_floating_point()):
            raise ValueError("Expected nonempty floating [batch, 3, n_templates]")
        if torch.isinf(scores).any():
            raise ValueError("Infinity is not a missing score; use NaN")
        finite = torch.isfinite(scores)
        if ((scores[finite] < 0) | (scores[finite] > 1)).any():
            raise ValueError("S2Match scores must lie in [0, 1]")
        return finite

    @torch.no_grad()
    def fit_normalizer(self, training_negative_scores):
        if bool(self.normalizer_ready):
            raise RuntimeError("Normalizer already fitted; create a new model to refit")
        valid = self._validate(training_negative_scores)
        count = valid.sum(0)
        if (count < 2).any():
            raise ValueError("Every scale/template needs >=2 training-negative scores")
        scores = torch.where(valid, training_negative_scores,
                             torch.zeros_like(training_negative_scores))
        mean = scores.sum(0) / count
        variance = torch.where(valid, (scores - mean).square(),
                               torch.zeros_like(scores)).sum(0) / count
        self.center.copy_(mean)
        self.spread.copy_(variance.sqrt().clamp_min(self.config["std_floor"]))
        self.normalizer_count.copy_(count)
        self.normalizer_ready.fill_(True)

    def forward(self, scores, return_attention=False):
        if not bool(self.normalizer_ready):
            raise RuntimeError("Fit the normalizer on training negatives first")
        valid = self._validate(scores)
        z = torch.where(valid, (scores - self.center) / self.spread,
                        torch.zeros_like(scores))
        if self.mode == "raw_s6":
            z = torch.where(valid, scores, torch.zeros_like(scores))
        if self.mode in ("s6", "s12", "ssub", "raw_s6"):
            selected = 0 if self.mode == "raw_s6" else (
                ("s6", "s12", "ssub").index(self.mode))
            active = torch.zeros_like(valid)
            active[:, selected] = valid[:, selected]
        else:
            active = valid
        template_valid = active.any(1)
        attention = None
        if self.mode in ("attention", "gate"):
            # Within-scale context is the template's margin to that scale's
            # strongest bank match. It contains no injected-template identity.
            strongest = z.masked_fill(~valid, -torch.inf).amax(2, keepdim=True)
            strongest = torch.where(torch.isfinite(strongest), strongest, 0.0)
            features = torch.stack((z, z - strongest), dim=-1)
            features = torch.where(valid.unsqueeze(-1), features, 0.0)
            tokens = self.project(features) + self.scale_embedding[None, :, None, :]
            batch, _, templates, hidden = tokens.shape
            tokens = tokens.permute(0, 2, 1, 3).reshape(-1, 3, hidden)
            present = valid.permute(0, 2, 1).reshape(-1, 3)
            if self.mode == "attention":
                # PyTorch softmax over three masked keys would produce NaNs.
                # Give empty rows a dummy key, then zero all their outputs.
                padding = ~present
                padding = padding.clone()
                padding[~present.any(1), 0] = False
                tokens = tokens.masked_fill(~present.unsqueeze(-1), 0.0)
                exchanged, weights = self.cross_attention(
                    tokens, tokens, tokens, key_padding_mask=padding,
                    need_weights=return_attention)
                tokens = self.norm(tokens + exchanged)
                if return_attention:
                    weights = weights.masked_fill(
                        ~(present.unsqueeze(2) & present.unsqueeze(1)), 0.0)
                    attention = weights.reshape(batch, templates, 3, 3)
            logits = self.gate(tokens).squeeze(-1)
            logits = logits.reshape(batch, templates, 3).permute(0, 2, 1)
        else:
            logits = torch.zeros_like(z)
        logits = logits.masked_fill(~active, -torch.inf)
        logits = torch.where(template_valid.unsqueeze(1), logits, 0.0)
        weights = torch.softmax(logits, dim=1).masked_fill(~active, 0.0)
        # Convex pooling avoids summing duplicated/correlated evidence as if
        # it were independent. The final statistic still needs calibration.
        fused = (weights * z).sum(1).masked_fill(~template_valid, -torch.inf)
        abstain = ~template_valid.any(1)
        # Mask BEFORE the learned affine head: 0 * -inf in its backward pass
        # otherwise corrupts gradients even when abstaining rows leave the loss.
        maximum = fused.amax(1).masked_fill(abstain, 0.0)
        statistic = (F.softplus(self.log_scale) * maximum + self.bias).masked_fill(
            abstain, -torch.inf)
        return {
            "template_scores": fused, "statistic": statistic,
            "scale_weights": weights, "cross_attention": attention,
            "valid_templates": template_valid, "abstain": abstain,
        }


@dataclass(frozen=True)
class EmpiricalThreshold:
    """Conservative upper-tail threshold; ties never count as exceedances.

    No IID guarantee is asserted for correlated slots/attacks. -inf represents
    an abstention and stays in the calibration denominator.
    """
    alpha: float
    n: int
    n_abstain: int
    threshold: float | None
    status: str

    @classmethod
    def fit(cls, calibration_negative_scores, alpha):
        values = [float(x) for x in calibration_negative_scores]
        if not 0 < alpha < 1 or not values:
            raise ValueError("Nonempty calibration negatives and alpha in (0,1) required")
        if any(math.isnan(x) or x == math.inf for x in values):
            raise ValueError("Invalid calibration statistic")
        n_abstain = sum(x == -math.inf for x in values)
        rank = math.ceil((len(values) + 1) * (1 - alpha))
        if rank > len(values) or n_abstain == len(values):
            return cls(alpha, len(values), n_abstain, None, "insufficient_calibration")
        threshold = sorted(values)[rank - 1]
        if threshold == -math.inf:
            return cls(alpha, len(values), n_abstain, None, "insufficient_valid_calibration")
        return cls(alpha, len(values), n_abstain, threshold, "calibrated_exploratory")

    def predict(self, values):
        if self.threshold is None:
            raise RuntimeError(self.status)
        values = [float(x) for x in values]
        if any(math.isnan(x) or x == math.inf for x in values):
            raise ValueError("Invalid prediction statistic")
        return [math.isfinite(x) and x > self.threshold for x in values]

    def to_dict(self):
        return asdict(self)
