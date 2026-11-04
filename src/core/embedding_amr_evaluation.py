"""Source-separated, condition-specific calibration for the entire new flow."""
from __future__ import annotations

import math

from scipy.stats import beta, binomtest

from .embedding_amr import ARMS, digest

VERSION = "embedding-amr-calibration/v2"


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty score cohort")
    keys, ids, fingerprints = set(), set(), set()
    source_bindings, text_bindings, literal_bindings = {}, {}, {}
    for row in rows:
        for key in ("row_id", "source_id", "source_sha256", "condition", "kind",
                    "text_sha256", "detector_fingerprint"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("Missing score metadata: " + key)
        key = row["source_id"], row["condition"]
        if key in keys or row["row_id"] in ids:
            raise ValueError("Duplicate source-condition or row ID")
        keys.add(key)
        ids.add(row["row_id"])
        fingerprints.add(row["detector_fingerprint"])
        source = row["source_sha256"]
        if row["source_id"] in source_bindings and source_bindings[row["source_id"]] != source:
            raise ValueError("Source ID maps to different content hashes")
        source_bindings[row["source_id"]] = source
        binding = row["source_id"], row["kind"], row.get("split")
        if source in text_bindings and text_bindings[source] != binding:
            raise ValueError("Source hash reused under different identity or kind")
        text_bindings[source] = binding
        text_hash = row["text_sha256"]
        if text_hash == digest(""):
            if any(a["statistic"] is not None for a in row["arms"].values()):
                raise ValueError("Unavailable input must abstain in every arm")
        else:
            if text_hash in literal_bindings and literal_bindings[text_hash] != binding:
                raise ValueError("Same text reused under different source, kind or split")
            literal_bindings[text_hash] = binding
        if set(row["arms"]) != set(ARMS):
            raise ValueError("Complete shared arms required")
        for arm in row["arms"].values():
            score = arm["statistic"]
            if score is not None and (isinstance(score, bool) or not math.isfinite(score)):
                raise ValueError("Nonfinite score must not become a positive decision")
    if len(fingerprints) != 1:
        raise ValueError("Mixed detector fingerprints")
    return next(iter(fingerprints))


def fit(rows, alpha=0.05):
    fingerprint = validate_rows(rows)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if any(r["kind"] != "human" or r.get("split") != "calibration" for r in rows):
        raise ValueError("Only independent human calibration rows accepted")
    conditions = sorted({r["condition"] for r in rows})
    sources = {r["source_id"] for r in rows}
    if len(rows) != len(conditions) * len(sources):
        raise ValueError("Incomplete source-condition calibration product")
    if 1 / (len(sources) + 1) > alpha:
        raise ValueError("Calibration cohort cannot resolve requested alpha")
    result = {
        "version": VERSION, "detector_fingerprint": fingerprint, "alpha": alpha,
        "conditions": conditions,
        "source_ids": sorted(sources),
        "source_sha256": sorted({r["source_sha256"] for r in rows}),
        "text_sha256": sorted({r["text_sha256"] for r in rows} - {digest("")}),
        "calibration_scores_sha256": digest(rows),
        "nulls": {condition: {arm: [
            r["arms"][arm]["statistic"] for r in rows if r["condition"] == condition
        ] for arm in ARMS} for condition in conditions},
        "scope": "condition_specific_marginal_not_simultaneous",
    }
    result["artifact_sha256"] = digest(result)
    return result


def decision(score, nulls, alpha):
    if (not nulls or not 0 < alpha < 1
            or any(s is not None and (isinstance(s, bool) or not math.isfinite(s))
                   for s in [score, *nulls])):
        raise ValueError("Invalid score/calibration values")
    if score is None:
        return {"p_value": 1.0, "is_watermarked": False, "abstain": True}
    # Abstentions remain in the full calibration denominator and act as -inf.
    exceedances = sum(s is not None and s >= score for s in nulls)
    p = (1 + exceedances) / (1 + len(nulls))
    return {"p_value": p, "is_watermarked": p <= alpha, "abstain": False}


def interval(k, n, alpha=.05):
    return [0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))]


def paired_comparisons(predictions):
    """Paired document decisions; fixed contrasts, no test-driven arm selection.

    The difference CI bounds the two discordant probabilities with 97.5% exact
    binomial intervals, giving a conservative 95% interval by union bound.
    Holm adjusts all reported comparisons (conditions, kinds and both controls).
    """
    comparisons = []
    groups = sorted({(r["condition"], r["kind"]) for r in predictions})
    for condition, kind in groups:
        group = [r for r in predictions if (r["condition"], r["kind"]) == (condition, kind)]
        for control in ("amr_only", "embedding_without_relation"):
            pairs = [(r["decisions"]["embedding_amr"]["is_watermarked"],
                      r["decisions"][control]["is_watermarked"]) for r in group]
            wins = sum(a and not b for a, b in pairs)
            losses = sum(b and not a for a, b in pairs)
            n = len(pairs)
            left, right = interval(wins, n, .025), interval(losses, n, .025)
            comparisons.append({
                "condition": condition, "kind": kind, "control": control, "denominator": n,
                "fusion_only_positive": wins, "control_only_positive": losses,
                "rate_difference_fusion_minus_control": (wins - losses) / n,
                "difference_ci95_conservative": [left[0] - right[1], left[1] - right[0]],
                "mcnemar_exact_two_sided_p": float(binomtest(wins, wins + losses).pvalue)
                if wins + losses else 1.0,
                "favorable_direction": "positive" if kind == "watermarked" else "negative",
            })
    adjusted = 0.0
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["mcnemar_exact_two_sided_p"])
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1.0, (len(order) - rank) *
                                     comparisons[i]["mcnemar_exact_two_sided_p"]))
        comparisons[i]["holm_p_all_reported_contrasts"] = adjusted
    return comparisons


def evaluate(rows, calibration):
    fingerprint = validate_rows(rows)
    if calibration.get("artifact_sha256") != digest(
            {k: v for k, v in calibration.items() if k != "artifact_sha256"}):
        raise ValueError("Calibration artifact checksum mismatch")
    if (calibration.get("version") != VERSION
            or calibration["detector_fingerprint"] != fingerprint):
        raise ValueError("Calibration/detector fingerprint mismatch")
    if any(r.get("split") != "test" or r["kind"] not in ("human", "watermarked", "llm") for r in rows):
        raise ValueError("Only labeled independent test rows accepted")
    for field, key in (("source_id", "source_ids"), ("source_sha256", "source_sha256"),
                       ("text_sha256", "text_sha256")):
        if {r[field] for r in rows} & set(calibration[key]):
            raise ValueError("Calibration/test overlap: " + field)
    # A source can reappear as a condition text with a new source identifier.
    if ({r[k] for r in rows for k in ("source_sha256", "text_sha256")} &
            set(calibration["source_sha256"] + calibration["text_sha256"])):
        raise ValueError("Calibration/test overlap across source/text hashes")
    conditions = set(calibration["conditions"])
    n = len(calibration["source_ids"])
    if (n != len(set(calibration["source_ids"])) or n != len(set(calibration["source_sha256"]))
            or not 0 < calibration["alpha"] < 1 or 1 / (n + 1) > calibration["alpha"]
            or set(calibration["nulls"]) != conditions):
        raise ValueError("Invalid calibration artifact")
    for condition in conditions:
        if set(calibration["nulls"][condition]) != set(ARMS):
            raise ValueError("Invalid calibration arm set")
        if any(len(values) != n for values in calibration["nulls"][condition].values()):
            raise ValueError("Invalid calibration denominator")
    if {r["condition"] for r in rows} != conditions:
        raise ValueError("Test conditions must equal frozen calibration conditions")
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], set()).add(row["condition"])
    if any(cs != conditions for cs in by_source.values()):
        raise ValueError("Missing test rows; retain failures as abstentions")
    if not any(r["kind"] == "human" for r in rows):
        raise ValueError("Independent human-test FPR is required")
    predictions, metrics = [], []
    for row in rows:
        decisions = {arm: decision(row["arms"][arm]["statistic"],
                                  calibration["nulls"][row["condition"]][arm],
                                  calibration["alpha"]) for arm in ARMS}
        predictions.append({**{k: row[k] for k in ("row_id", "source_id", "condition", "kind")},
                            "decisions": decisions})
    for condition in sorted(conditions):
        for kind in sorted({r["kind"] for r in rows}):
            group = [r for r in predictions if r["condition"] == condition and r["kind"] == kind]
            for arm in ARMS:
                n = len(group)
                hits = sum(r["decisions"][arm]["is_watermarked"] for r in group)
                metrics.append({
                    "condition": condition, "kind": kind, "arm": arm,
                    "metric": "TPR" if kind == "watermarked" else "FPR",
                    "hits": hits, "denominator": n, "rate": hits / n,
                    "ci95_clopper_pearson": interval(hits, n),
                    "abstentions": sum(r["decisions"][arm]["abstain"] for r in group),
                })
    return {
        "status": "independent_evaluation_complete",
        "detector_fingerprint": fingerprint, "alpha": calibration["alpha"],
        "metrics": metrics, "predictions": predictions,
        "paired_comparisons": paired_comparisons(predictions),
        "confidence_intervals": "pointwise_not_multiple_comparison_adjusted",
        "paired_difference_intervals": "conservative_pointwise_95_percent",
        "paired_p_adjustment": "Holm_all_conditions_kinds_and_two_fixed_controls",
        "fusion_benefit_established": False,
    }
