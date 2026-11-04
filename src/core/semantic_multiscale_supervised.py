"""Separate event selection from conditional ABCD identity; old sources unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semantic_multiscale import MultiScaleSymbolReadout

EVENT_CLASSES = ("carrier", "background", "duplicate", "unknown")
VERSION = "semantic-score-supervised:v2"


class SupervisedSymbolReadout(nn.Module):
    def __init__(self, groups, mode):
        super().__init__()
        self.symbol = MultiScaleSymbolReadout(groups, mode)
        # Same small selection head and evidence for every fusion architecture.
        self.event = nn.Linear(9, 4)

    def fit_normalizer(self, training_scores):
        self.symbol.fit_normalizer(training_scores)

    def forward(self, scores):
        result = self.symbol(scores)
        # Relative class weights plus absolute evidence: softmax alone loses strength.
        logits = result["logits"].masked_fill(~result["class_valid"], 0.0)
        valid_fraction = torch.isfinite(scores).float().mean((1, 2))[:, None]
        features = torch.cat((result["q"], logits, valid_fraction), 1)
        event_logits = self.event(features)
        return {**result, "event_logits": event_logits,
                "event_q": torch.softmax(event_logits, 1)}

    def loss(self, scores, kinds, symbols):
        output = self(scores)
        valid = ~output["abstain"]
        if not valid.any():
            raise ValueError("No usable supervised evidence; cannot silently skip a batch")
        event_loss = F.cross_entropy(output["event_logits"][valid], kinds[valid])
        known = valid & ((kinds == 0) | (kinds == 2)) & (symbols >= 0)
        if known.any():
            expected_valid = output["class_valid"][known, symbols[known]]
            if not expected_valid.all():
                raise ValueError("Gold class lacks legal template evidence")
            identity_loss = F.cross_entropy(output["logits"][known], symbols[known])
        else:
            identity_loss = event_loss * 0
        # Unknown supervises the event head, never gets forced into a fact class.
        return event_loss + identity_loss

    def configuration(self):
        return {**self.symbol.configuration(), "version": VERSION,
                "event_classes": EVENT_CLASSES, "event_parameters": 40,
                "unknown_symbol_supervision": False}


def emit(output, presence=0.0, confidence=0.0):
    """No labels/positions in inference; fixed argmax gate precedes confidence rules."""
    q, event = output["q"].detach().cpu(), output["event_q"].detach().cpu()
    missing = output["abstain"].detach().cpu()
    predictions = []
    for i in range(len(q)):
        class_max = q[i].max()
        unique = torch.isclose(q[i], class_max, atol=1e-12, rtol=0).sum() == 1
        event_max = event[i].max()
        event_unique = torch.isclose(event[i], event_max, atol=1e-12, rtol=0).sum() == 1
        accepted = (not missing[i] and bool(unique) and bool(event_unique)
                    and int(event[i].argmax()) == 0
                    and float(event[i, 0]) >= presence and float(class_max) >= confidence)
        predictions.append("ABCD"[int(q[i].argmax())] if accepted else None)
    return predictions
