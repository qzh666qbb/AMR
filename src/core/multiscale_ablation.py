"""Confidence-feature fusion and training-only negative weighting.

Independent successor: the frozen v1 module and its outputs remain unchanged.
No injected-template identity or perturbation label enters the fusion model.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.core.multiscale_fusion import MultiScaleAttentionFusion

VERSION = "swan-score-msa-ablation-v1"
FEATURES = (
    "normalized_score", "margin_to_scale_maximum", "scale_top_two_gap",
    "scale_concentration", "template_scale_mean", "template_scale_variance",
    "template_available_scale_fraction",
)


def confidence_features(z, valid):
    """[B,3,T] -> [B,3,T,7]; permutation-equivariant, missing-safe."""
    if z.ndim != 3 or z.shape != valid.shape or z.shape[1] != 3:
        raise ValueError("Expected matching [B,3,T] scores and validity mask")
    safe = torch.where(valid, z, 0.0)
    count = valid.sum(2, keepdim=True)
    ranked = safe.masked_fill(~valid, -torch.inf)
    top = ranked.topk(min(2, z.shape[2]), dim=2).values
    top = torch.where(torch.isfinite(top), top, 0.0)
    maximum = top[:, :, :1]
    gap = (top[:, :, :1] - top[:, :, 1:2]) if z.shape[2] >= 2 else top * 0
    gap = torch.where(count >= 2, gap, 0.0)
    logits = torch.where(count > 0, ranked, 0.0)
    probability = torch.softmax(logits, dim=2).masked_fill(~valid, 0.0)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(2, keepdim=True)
    concentration = torch.where(
        count >= 2, 1.0 - entropy / count.clamp_min(2).float().log(), 0.0,
    ).clamp(0.0, 1.0)
    scale_count = valid.sum(1, keepdim=True)
    mean = safe.sum(1, keepdim=True) / scale_count.clamp_min(1)
    variance = torch.where(valid, (safe - mean).square(), 0.0).sum(1, keepdim=True)
    variance = variance / scale_count.clamp_min(1)
    features = torch.stack((
        safe, safe - maximum, gap.expand_as(safe),
        concentration.expand_as(safe), mean.expand_as(safe),
        variance.expand_as(safe), (scale_count / 3.0).expand_as(safe),
    ), dim=-1)
    return features.masked_fill(~valid.unsqueeze(-1), 0.0)


class ConfidenceFusion(MultiScaleAttentionFusion):
    """Same gate initialization as v1; five new projection columns start at 0."""

    def __init__(self, n_templates=100, hidden_dim=16, n_heads=2,
                 mode="gate", std_floor=0.05):
        if mode not in ("gate", "attention"):
            raise ValueError("Only gate and attention are valid ablation modes")
        # Initialize shared layers before optional attention to couple seeds.
        super().__init__(n_templates, hidden_dim, n_heads, "gate", std_floor)
        original = self.project
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(len(FEATURES), hidden_dim)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :2].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        self.project = expanded
        self.mode = mode
        self.config["mode"] = mode
        if mode == "attention":
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=0.0, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, scores, return_attention=False):
        if not bool(self.normalizer_ready):
            raise RuntimeError("Fit the normalizer on training negatives first")
        valid = self._validate(scores)
        z = torch.where(valid, (scores - self.center) / self.spread, 0.0)
        features = confidence_features(z, valid)
        tokens = self.project(features) + self.scale_embedding[None, :, None, :]
        batch, _, templates, hidden = tokens.shape
        tokens = tokens.permute(0, 2, 1, 3).reshape(-1, 3, hidden)
        present = valid.permute(0, 2, 1).reshape(-1, 3)
        attention = None
        if self.mode == "attention":
            padding = (~present).clone()
            padding[~present.any(1), 0] = False
            tokens = tokens.masked_fill(~present.unsqueeze(-1), 0.0)
            exchanged, weights = self.cross_attention(
                tokens, tokens, tokens, key_padding_mask=padding,
                need_weights=return_attention,
            )
            tokens = self.norm(tokens + exchanged)
            if return_attention:
                weights = weights.masked_fill(
                    ~(present.unsqueeze(2) & present.unsqueeze(1)), 0.0)
                attention = weights.reshape(batch, templates, 3, 3)
        logits = self.gate(tokens).squeeze(-1)
        logits = logits.reshape(batch, templates, 3).permute(0, 2, 1)
        template_valid = valid.any(1)
        logits = logits.masked_fill(~valid, -torch.inf)
        logits = torch.where(template_valid.unsqueeze(1), logits, 0.0)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
        fused = (weights * z).sum(1).masked_fill(~template_valid, -torch.inf)
        abstain = ~template_valid.any(1)
        maximum = fused.amax(1).masked_fill(abstain, 0.0)
        statistic = (F.softplus(self.log_scale) * maximum + self.bias).masked_fill(
            abstain, -torch.inf)
        return {
            "template_scores": fused, "statistic": statistic,
            "scale_weights": weights, "cross_attention": attention,
            "valid_templates": template_valid, "abstain": abstain,
        }


def hard_negative_weights(base_weights, labels, keys, negative_statistics,
                          fraction=0.1, multiplier=3.0):
    """Mine only supplied training negatives; preserve class/document mass."""
    if not 0 < fraction <= 1 or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Invalid hard-negative configuration")
    if len(base_weights) != len(labels) or len(keys) != len(labels):
        raise ValueError("Mismatched training inputs")
    negative_indices = (labels == 0).nonzero().flatten()
    if (len(negative_statistics) != len(negative_indices)
            or not torch.isfinite(negative_statistics).all()
            or not ((labels == 0) | (labels == 1)).all()
            or not torch.isfinite(base_weights).all() or (base_weights <= 0).any()):
        raise ValueError("Invalid negative statistics, labels or base weights")
    weights = base_weights.clone()
    selected = torch.zeros(len(labels), dtype=torch.bool)
    negative_docs = [keys[i][0] for i in negative_indices.tolist()]
    audit = {"per_document": {}, "negative_rows": len(negative_indices)}
    for doc in sorted(set(negative_docs)):
        local = torch.tensor([i for i, value in enumerate(negative_docs) if value == doc])
        rows = negative_indices[local]
        scores = negative_statistics[local]
        requested = math.ceil(len(rows) * fraction)
        cutoff = torch.sort(scores, descending=True).values[requested - 1]
        chosen = scores >= cutoff
        selected[rows[chosen]] = True
        before = base_weights[rows].sum()
        weights[rows[chosen]] *= multiplier
        weights[rows] *= before / weights[rows].sum()
        after = weights[rows].sum()
        if abs(float(before - after)) > 1e-7:
            raise RuntimeError("Document weight mass changed")
        audit["per_document"][doc] = {
            "negative_rows": len(rows), "requested": requested,
            "selected_with_ties": int(chosen.sum()), "cutoff": float(cutoff),
            "weight_mass_before": float(before), "weight_mass_after": float(after),
        }
    chosen_keys = [list(key) for key, flag in zip(keys, selected.tolist()) if flag]
    audit["selected_rows"] = len(chosen_keys)
    audit["selected_keys_sha256"] = hashlib.sha256(
        json.dumps(chosen_keys, separators=(",", ":")).encode()).hexdigest()
    if not torch.equal(weights[labels == 1], base_weights[labels == 1]):
        raise RuntimeError("Positive weights changed")
    return weights, selected, audit
