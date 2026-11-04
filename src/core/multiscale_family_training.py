"""Training-only carrier-family mining, max-format loss and model selection."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

VERSION = "swan-family-hard-development-v1"


def family_groups(keys):
    """Group row indices by (source document, carrier slot), never by format."""
    groups = {}
    if len(set(tuple(k) for k in keys)) != len(keys):
        raise ValueError("Duplicate row keys")
    for i, (doc, attack, slot) in enumerate(keys):
        groups.setdefault((doc, int(slot)), []).append(i)
    return [(key, sorted(rows, key=lambda i: keys[i][1]))
            for key, rows in sorted(groups.items())]


def family_negative_weights(base, labels, keys, negative_scores,
                            fraction=0.1, multiplier=3.0):
    """Same nominal ROW budget as old mining; at most one row per family.

    Format ties: lexicographic attack. Family ties: ascending slot.
    All non-selected formats retain their positive base training weight.
    """
    ni = (labels == 0).nonzero().flatten()
    if (not 0 < fraction <= 1 or not math.isfinite(multiplier) or multiplier < 1
            or len(base) != len(labels) or len(keys) != len(labels)
            or len(ni) != len(negative_scores)
            or not torch.isfinite(negative_scores).all()
            or not torch.isfinite(base).all() or (base <= 0).any()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("Invalid family mining input")
    nk = [keys[i] for i in ni.tolist()]
    grouped = family_groups(nk)
    candidates = {}
    for (doc, slot), local in grouped:
        representative = min(local, key=lambda i: (-float(negative_scores[i]), nk[i][1]))
        candidates.setdefault(doc, []).append((representative, slot))
    selected = torch.zeros(len(base), dtype=torch.bool)
    weights = base.clone()
    audit = {"negative_rows": len(ni), "negative_families": len(grouped),
             "per_document": {}, "budget_basis": "ceil(fraction * negative_rows)"}
    for doc, choices in sorted(candidates.items()):
        local = [i for i, key in enumerate(nk) if key[0] == doc]
        rows = ni[local]
        requested = math.ceil(len(rows) * fraction)
        if requested > len(choices):
            raise ValueError("Row hard budget exceeds number of carrier families")
        choices.sort(key=lambda pair: (-float(negative_scores[pair[0]]), pair[1]))
        chosen = ni[[i for i, _ in choices[:requested]]]
        selected[chosen] = True
        before = base[rows].sum()
        weights[chosen] *= multiplier
        weights[rows] *= before / weights[rows].sum()
        if abs(float(weights[rows].sum() - before)) > 1e-7:
            raise RuntimeError("Document training mass changed")
        audit["per_document"][doc] = {
            "negative_rows": len(rows), "negative_families": len(choices),
            "requested": requested, "selected": len(chosen),
            "weight_mass_before": float(before),
            "weight_mass_after": float(weights[rows].sum()),
        }
    selected_keys = [list(k) for k, flag in zip(keys, selected.tolist()) if flag]
    audit["selected_rows"] = len(selected_keys)
    audit["selected_keys_sha256"] = hashlib.sha256(
        json.dumps(selected_keys, separators=(",", ":")).encode()).hexdigest()
    if not torch.equal(weights[labels == 1], base[labels == 1]):
        raise RuntimeError("Positive weight changed")
    return weights, selected, audit


def max_format_loss(statistics, groups, family_weights):
    """Weighted softplus(max logit) for negatives, with exact max gradients.

    -inf is an abstention; it remains in the family denominator. NaN and +inf
    are invalid. torch.amax shares gradients equally across exact max ties.
    """
    if (statistics.ndim != 1 or len(groups) != len(family_weights)
            or not len(groups) or torch.isnan(statistics).any()
            or torch.isposinf(statistics).any()
            or not torch.isfinite(family_weights).all() or (family_weights <= 0).any()):
        raise ValueError("Invalid max-format input")
    maxima = torch.stack([statistics[rows].amax() for rows in groups])
    return (F.softplus(maxima) * family_weights).sum() / family_weights.sum()


def guarded_epoch(history, alpha=0.05, tpr_tolerance=0.02, positive_bce_tolerance=0.05):
    """Validation selection at train-derived thresholds; final cal is absent."""
    reference = min(history, key=lambda h: (h["validation_bce"], h["epoch"]))
    eligible = [h for h in history
                if h["validation_tpr"] >= reference["validation_tpr"] - tpr_tolerance
                and h["validation_positive_bce"] <=
                reference["validation_positive_bce"] + positive_bce_tolerance]
    feasible = [h for h in eligible if h["validation_worst_fpr"] <= alpha]
    if feasible:
        chosen = min(feasible, key=lambda h: (-h["validation_tpr"],
                                            h["validation_worst_fpr"],
                                            h["validation_bce"], h["epoch"]))
    else:
        chosen = min(eligible, key=lambda h: (h["validation_worst_fpr"],
                                            -h["validation_tpr"],
                                            h["validation_bce"], h["epoch"]))
    return {
        "bce_epoch": reference["epoch"], "guarded_epoch": chosen["epoch"],
        "bce_reference": reference, "guarded": chosen,
        "eligible_epochs": [h["epoch"] for h in eligible],
        "feasible_epochs": [h["epoch"] for h in feasible],
        "status": "feasible" if feasible else "no_feasible_epoch_with_positive_guard",
        "threshold_partition": "training_negatives_in_sample_proxy",
        "ranking_partition": "validation_only",
    }


def select_coefficient(candidate_rows, alpha=0.05, tpr_tolerance=0.02,
                       positive_bce_tolerance=0.05):
    """One coefficient per architecture, shared by all seeds; zero is allowed."""
    stats = {}
    for name, rows in candidate_rows.items():
        stats[name] = {
            "mean_validation_tpr": float(np.mean([r["validation_tpr"] for r in rows])),
            "mean_validation_positive_bce": float(np.mean(
                [r["validation_positive_bce"] for r in rows])),
            "mean_validation_bce": float(np.mean([r["validation_bce"] for r in rows])),
            "mean_validation_worst_fpr": float(np.mean(
                [r["validation_worst_fpr"] for r in rows])),
            "max_seed_validation_worst_fpr": max(r["validation_worst_fpr"] for r in rows),
        }
    reference = stats["L000"]
    eligible = [name for name, r in stats.items()
                if r["mean_validation_tpr"] >= reference["mean_validation_tpr"] - tpr_tolerance
                and r["mean_validation_positive_bce"] <=
                reference["mean_validation_positive_bce"] + positive_bce_tolerance]
    feasible = [name for name in eligible
                if stats[name]["max_seed_validation_worst_fpr"] <= alpha]
    def rank(name):
        r = stats[name]
        return ((-r["mean_validation_tpr"], r["mean_validation_worst_fpr"],
                 r["mean_validation_bce"], name) if feasible else
                (r["mean_validation_worst_fpr"], -r["mean_validation_tpr"],
                 r["mean_validation_bce"], name))
    chosen = min(feasible or eligible, key=rank)
    return {"selected": chosen, "candidates": stats, "eligible": eligible,
            "feasible": feasible, "selection_partition": "validation_only",
            "status": "feasible" if feasible else "no_all_seed_feasible_coefficient"}
