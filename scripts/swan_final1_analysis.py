"""Frozen final1 matching and source-level inference. No model/scorer imports."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy.stats import beta

from swan_reframing_statistics import (
    conservative_threshold, exact_mcnemar, holm_adjust, z_score,
)

CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 20260918


def feature_bin(row):
    w, n = row["word_count"], row["num_detection_units"]
    if not 100 <= w <= 220 or not 8 <= n <= 24:
        raise ValueError("Matching requires structurally qualified baseline features")
    return ((0 if w < 140 else 1 if w < 180 else 2),
            (0 if n < 12 else 1 if n < 16 else 2))


def match_humans(targets, candidates):
    """Two per target, greedily in allocation order; no scores or backtracking."""
    targets = sorted(targets, key=lambda r: r["ordinal"])
    candidates = sorted(candidates, key=lambda r: r["ordinal"])
    for group in (targets, candidates):
        if len({r["allocation_id"] for r in group}) != len(group):
            raise ValueError("Duplicate allocation in matching")
    used, rows, exact = set(), [], 0
    for target in targets:
        eligible = [r for r in candidates if r["allocation_id"] not in used
                    and feature_bin(r) == feature_bin(target)]
        eligible.sort(key=lambda r: (
            r["domain"] != target["domain"],
            abs(r["word_count"] - target["word_count"])
            + 5 * abs(r["num_detection_units"] - target["num_detection_units"]),
            r["ordinal"],
        ))
        chosen = eligible[:2]
        used.update(r["allocation_id"] for r in chosen)
        exact += sum(r["domain"] == target["domain"] for r in chosen)
        rows.append({"target_id": target["allocation_id"],
                     "human_ids": [r["allocation_id"] for r in chosen]})
    covered = sum(bool(r["human_ids"]) for r in rows)
    def bins(group):
        return dict(sorted(Counter(str(feature_bin(r)) for r in group).items()))
    return {
        "pairs": rows, "target_count": len(targets), "covered_targets": covered,
        "human_count": len(used), "exact_hostname_count": exact,
        "exact_hostname_rate": exact / len(used) if used else None,
        "target_bins": bins(targets), "eligible_human_bins": bins(candidates),
        "matched_human_bins": bins([r for r in candidates if r["allocation_id"] in used]),
        "passed": bool(targets) and covered * 10 >= len(targets) * 9 and len(used) >= 400,
    }


def proportion(successes, n):
    if type(successes) is not int or type(n) is not int or not 0 <= successes <= n:
        raise ValueError("Invalid proportion counts")
    if not n:
        return {"successes": successes, "n": n, "estimate": None, "ci95": None,
                "upper95_one_sided": None}
    lower = float(beta.ppf(.025, successes, n-successes+1)) if successes else 0.0
    upper = float(beta.ppf(.975, successes+1, n-successes)) if successes < n else 1.0
    bound = float(beta.ppf(.95, successes+1, n-successes)) if successes < n else 1.0
    return {"successes": successes, "n": n, "estimate": successes/n,
            "ci95": [lower, upper], "upper95_one_sided": bound}


def aggregate_vectors(vectors):
    if not vectors or any(len(row) != 50 for row in vectors):
        raise ValueError("Every detection unit needs all 50 bank scores")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or not 0 <= v <= 1 for row in vectors for v in row):
        raise ValueError("Missing/nonfinite score is not a red unit")
    maxima = [max(row) for row in vectors]
    green = sum(v >= .65 for v in maxima)
    n = len(vectors)
    return {"N": n, "G": green, "green_rate": green/n, "z": z_score(green, n),
            "mean_max_bank_score": sum(maxima)/n}


def bootstrap_mean(differences):
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires complete finite source pairs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for start in range(0, BOOTSTRAPS, 1000):
        index = rng.integers(0, len(values), size=(min(1000, BOOTSTRAPS-start), len(values)))
        draws.extend(np.mean(values[index], axis=1).tolist())
    return {"mean": float(np.mean(values)),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "source"}


def sign_test(differences):
    if any(not math.isfinite(v) for v in differences):
        raise ValueError("Missing paired difference")
    positive = sum(v > 0 for v in differences)
    negative = sum(v < 0 for v in differences)
    return {"positive": positive, "negative": negative,
            "zero": len(differences)-positive-negative,
            "p_one_sided": exact_mcnemar(positive, negative)}


def continuous_pair(left, right, field):
    if len(left) != len(right) or not left:
        raise ValueError("Incomplete continuous pairs")
    differences = [a[field]-b[field] for a, b in zip(left, right)]
    return {**bootstrap_mean(differences), **sign_test(differences)}


def binary_pair(left, right):
    if len(left) != len(right) or not left or any(type(v) is not bool for v in left+right):
        raise ValueError("Complete boolean pairs required")
    counts = Counter((a, b) for a, b in zip(left, right))
    losses, gains = counts[True, False], counts[False, True]
    return {**bootstrap_mean([int(a)-int(b) for a, b in zip(left, right)]),
            "n": len(left), "both_detected": counts[True, True],
            "left_only": losses, "right_only": gains, "neither": counts[False, False],
            "p_one_sided": exact_mcnemar(losses, gains)}


def missing_effect_bounds(left, right):
    """Each element is a boolean or None; do not discard unknown sources."""
    if len(left) != len(right) or not left:
        raise ValueError("Aligned full denominator required")
    low = high = unknown = 0
    for a, b in zip(left, right):
        if any(v is not None and type(v) is not bool for v in (a, b)):
            raise ValueError("Invalid detection value")
        low += (int(a) if a is not None else 0) - (int(b) if b is not None else 1)
        high += (int(a) if a is not None else 1) - (int(b) if b is not None else 0)
        unknown += a is None or b is None
    return {"n": len(left), "pairs_with_missing": unknown,
            "effect_bounds": [low/len(left), high/len(left)]}


def rates(rows, threshold):
    return proportion(sum(r["z"] > threshold for r in rows), len(rows))


def calibrate(rows):
    threshold = conservative_threshold([r["z"] for r in rows])
    return {"threshold": threshold, "comparison": "strict_greater",
            "calibration": rates(rows, threshold), "official_2_33": rates(rows, 2.33)}


def primary_analysis(common, thresholds, test_rows):
    """common maps C0..C4 to the same ordered, quality-frozen sources."""
    n = len(common["C0"])
    ids = [r["allocation_id"] for r in common["C0"]]
    if n < 200 or len(set(ids)) != n:
        raise ValueError("At least 200 unique common pairs required")
    for condition in CONDITIONS:
        if [r["allocation_id"] for r in common[condition]] != ids:
            raise ValueError("Conditions use different source denominators")
    if any(a["N"] != b["N"] for a, b in zip(common["C0"], common["C2"])):
        raise ValueError("Count-neutral invariant failed")
    detected = {c: [r["z"] > thresholds[c] for r in common[c]] for c in CONDITIONS}
    hypotheses = {}
    for name, left, right in (("H1", "C0", "C1"), ("H2", "C0", "C2"), ("H3", "C4", "C2")):
        hypotheses[name] = {"left": left, "right": right,
                            **binary_pair(detected[left], detected[right])}
    for row, adjusted in zip(hypotheses.values(), holm_adjust(
            [r["p_one_sided"] for r in hypotheses.values()])):
        row["p_holm"] = adjusted
    h2 = hypotheses["H2"]
    effect_gate = (h2["left_only"]-h2["right_only"])*10 >= n
    gate_e = h2["p_holm"] <= .05 and effect_gate
    gate_b = sum(detected["C0"])*5 >= n*4
    fpr = {c: rates(test_rows[c], thresholds[c]) for c in CONDITIONS}
    fpr_gate = all(r["n"] >= 400 and r["upper95_one_sided"] <= .03 for r in fpr.values())
    continuous = {c: {field: continuous_pair(common["C0"], common[c], field)
                       for field in ("z", "green_rate", "mean_max_bank_score")}
                  for c in ("C1", "C2", "C3")}
    # (C0-C2)-(C0-C3) simplifies to C3-C2; positive supports stronger loss at C2.
    mechanism = {field: continuous_pair(common["C3"], common["C2"], field)
                 for field in ("mean_max_bank_score", "green_rate")}
    for row, adjusted in zip(mechanism.values(), holm_adjust(
            [r["p_one_sided"] for r in mechanism.values()])):
        row["p_holm"] = adjusted
    mechanism_gate = all(r["mean"] > 0 and r["p_holm"] <= .05 for r in mechanism.values())
    # Also require the actual C0→C2 losses to have positive means.
    actual_direction = all(continuous["C2"][f]["mean"] > 0 for f in mechanism)
    recovery = hypotheses["H3"]["p_holm"] <= .05 and hypotheses["H3"]["mean"] > 0
    branch = ("C" if not gate_b or not fpr_gate else
              "A" if gate_e and mechanism_gate and actual_direction and recovery else "B")
    return {
        "branch": branch, "common_n": n,
        "gates": {"B_common": gate_b, "E": gate_e, "test_FPR": fpr_gate,
                  "mechanism": mechanism_gate and actual_direction, "recovery": recovery},
        "hypotheses": hypotheses, "continuous_C0_minus_condition": continuous,
        "mechanism_C3_minus_C2": mechanism,
        "mechanism_interpretation": "secondary_support" if gate_e and mechanism_gate
        and actual_direction else "exploratory_only",
        "common_TPR": {c: rates(common[c], thresholds[c]) for c in CONDITIONS},
        "test_FPR": fpr,
        "official_2_33": {"common_TPR": {c: rates(common[c], 2.33) for c in CONDITIONS},
                          "test_FPR": {c: rates(test_rows[c], 2.33) for c in CONDITIONS}},
        "oracle_recovery_not_independent_defense_evidence": True,
    }


def asr_denominators(baseline, qualified, transformed, baseline_threshold, threshold):
    detected = {r["allocation_id"] for r in baseline if r["z"] > baseline_threshold}
    known = {r["allocation_id"]: r["z"] > threshold for r in transformed}
    qualified = set(qualified)
    successes = sum(a in qualified and known.get(a) is False for a in detected)
    unknown = sum(a in qualified and a not in known for a in detected)
    eligible_detected = detected & qualified
    return {
        "valid_ASR": proportion(successes, len(detected)),
        "quality_qualified_ASR": proportion(successes, len(eligible_detected)),
        "confirmed_construction_or_quality_failures": len(detected-qualified),
        "unknown_detection": unknown,
        "quality_qualified_detected": len(eligible_detected),
        "bounds_over_all_baseline_detected": [
            successes/len(detected), (successes+unknown)/len(detected)] if detected else None,
        "interpretation": "complete" if not unknown else "lower_bound_with_unknowns",
    }
