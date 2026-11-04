"""Text-only shared candidates and exact equal-composition order diagnostics.

This is a readability experiment, not a watermark detector or a certificate.
No gold event positions, source sequences, attack labels or template IDs enter
the candidate extractor.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

import numpy as np

VERSION = "semantic-sequence-shared-candidates/v1"
ALPHABET = "ABCD"


def equal_composition_sequences():
    """All 8!/(2!)^4 arrangements, including the true arrangement."""
    rows = []

    def visit(prefix, counts):
        if len(prefix) == 8:
            rows.append(prefix)
            return
        for symbol in range(4):
            if counts[symbol] < 2:
                updated = list(counts)
                updated[symbol] += 1
                visit(prefix + (symbol,), updated)

    visit((), [0] * 4)
    return np.asarray(rows, dtype=np.int8)


def centered_window(text, tokens, anchor, width):
    index = next((i for i, t in enumerate(tokens)
                  if t.start() <= anchor < t.end()), None)
    if index is None:
        raise ValueError("Anchor is not contained in a whitespace token")
    start = min(max(0, index - width // 2), max(0, len(tokens) - width))
    stop = min(len(tokens), start + width)
    a, b = tokens[start].start(), tokens[stop - 1].end()
    return {"span": [a, b], "text": text[a:b], "width": width,
            "text_sha256": hashlib.sha256(text[a:b].encode()).hexdigest()}


def shared_candidates(text, nlp, max_anchors=32):
    """One observation per non-AUX VERB occurrence; repeated text stays distinct."""
    if not isinstance(text, str):
        raise TypeError("The extractor accepts only text")
    tokens = list(re.finditer(r"\S+", text))
    anchors = [(t.idx, t.idx + len(t.text), t.lemma_)
               for t in nlp(text) if t.pos_ == "VERB"]
    anchors = sorted(set(anchors))
    status = "ok" if anchors else "no_observable_anchor"
    if len(anchors) > max_anchors:
        status = "budget_abstain"
    observations = []
    if status == "ok":
        for start, end, lemma in anchors:
            observations.append({
                "observation_id": f"event-{start}-{end}",
                "anchor_span": [start, end], "lemma": lemma,
                "windows": [centered_window(text, tokens, start, width)
                            for width in (6, 12)],
            })
    return {"version": VERSION, "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "status": status, "observed_anchor_count": len(anchors),
            "observations": observations, "truncated": False}


def symbol_weights(scores, groups, temperature):
    """Complete-dictionary weights; missing evidence is a local erasure."""
    values = np.asarray(scores, dtype=float)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Invalid temperature")
    members = [i for group in groups for i in group]
    if len(groups) != 4 or len(set(map(len, groups))) != 1:
        raise ValueError("Four equally sized groups required")
    if sorted(members) != list(range(len(values))):
        raise ValueError("Groups must partition every score exactly once")
    if not np.isfinite(values).all():
        return None
    exp = np.exp((values - values.max()) / temperature)
    return np.asarray([exp[group].sum() for group in groups]) / exp.sum()


def average_window_weights(weights):
    """Both windows must be valid; one failed window erases only its anchor."""
    if len(weights) != 2:
        raise ValueError("Exactly two common windows required")
    if any(w is None for w in weights):
        return None
    values = np.asarray(weights, dtype=float)
    if values.shape != (2, 4) or not np.isfinite(values).all():
        raise ValueError("Invalid window weights")
    if (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("Weights must be normalized")
    return values.mean(0)


def exact_costs(weights, candidates, mode="soft"):
    """Exact insertion/deletion/substitution DP, vectorized over candidates.

    An explicit None is an erased observation: substitution costs one for all
    classes. No candidates or alignment paths are sampled. Hard tied predictions
    also erase rather than introduce alphabet-order preference.
    """
    candidates = np.asarray(candidates)
    if (candidates.ndim != 2 or candidates.shape[0] == 0
            or not np.issubdtype(candidates.dtype, np.integer)
            or ((candidates < 0) | (candidates > 3)).any()):
        raise ValueError("Invalid candidate sequences")
    if mode not in ("hard", "soft"):
        raise ValueError("Unknown edit mode")
    costs = []
    for q in weights:
        if q is None:
            costs.append(np.ones(4))
            continue
        q = np.asarray(q, dtype=float)
        if (q.shape != (4,) or not np.isfinite(q).all() or (q < 0).any()
                or not np.isclose(q.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("Invalid observation weights")
        if mode == "soft":
            costs.append(-np.log(np.maximum(q, 1e-6)) / math.log(4))
        else:
            winners = np.flatnonzero(np.isclose(q, q.max(), atol=1e-12, rtol=0))
            row = np.ones(4)
            if len(winners) == 1:
                row[winners[0]] = 0
            costs.append(row)
    count, length = candidates.shape
    previous = np.broadcast_to(np.arange(length + 1, dtype=float), (count, length + 1)).copy()
    for i, cost in enumerate(costs, 1):
        current = np.empty_like(previous)
        current[:, 0] = i
        for j in range(1, length + 1):
            current[:, j] = np.minimum(
                previous[:, j - 1] + cost[candidates[:, j - 1]],
                np.minimum(previous[:, j] + 1, current[:, j - 1] + 1))
        previous = current
    return previous[:, -1]


def unordered_costs(weights, candidates):
    """Only symbol counts matter; all 2,520 candidates must tie exactly."""
    observed = np.zeros(4)
    for q in weights:
        if q is not None:
            values = np.asarray(q, dtype=float)
            if (values.shape != (4,) or not np.isfinite(values).all()
                    or (values < 0).any() or not np.isclose(values.sum(), 1)):
                raise ValueError("Invalid weights")
            observed += values
    return np.asarray([sum(abs(observed[i] - Counter(row)[i]) for i in range(4))
                       for row in candidates])


def rank_truth(costs, candidates, truth, tolerance=1e-10):
    """Truth is joined only after the prediction file is sealed."""
    candidates, costs = np.asarray(candidates), np.asarray(costs, dtype=float)
    truth = tuple(ALPHABET.index(x) if isinstance(x, str) else int(x) for x in truth)
    indices = np.flatnonzero(np.all(candidates == truth, axis=1))
    if len(indices) != 1 or not np.isfinite(costs).all() or len(costs) != len(candidates):
        raise ValueError("Truth must occur exactly once and scores must be finite")
    value = costs[indices[0]]
    equal = np.isclose(costs, value, atol=tolerance, rtol=0)
    better = (costs < value) & ~equal
    worse = (costs > value) & ~equal
    ties = int(equal.sum())
    return {"candidate_count": len(costs), "alternative_count": len(costs) - 1,
            "true_cost": float(value), "average_rank": float(better.sum() + (ties + 1) / 2),
            "top1_credit": 1 / ties if not better.any() else 0.0,
            "pairwise_win_rate": float((worse.sum() + (ties - 1) / 2) / (len(costs) - 1)),
            "tie_count_including_truth": ties, "tie_tolerance": tolerance}
