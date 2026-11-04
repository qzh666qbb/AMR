"""Source-isolated calibration/evaluation for sealed complete atomic-arm scores."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy.stats import beta

ARMS = ("A0", "A1", "A2", "single_view")


def validate_rows(rows, expected, conditions, arms=ARMS):
    """`expected` maps source IDs to presealed role/kind and optional source hashes."""
    if not expected or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Complete frozen cohort and conditions required")
    required = {(sid, condition) for sid in expected for condition in conditions}
    mapped = {}
    for row in rows:
        key = row["source_id"], row["condition"]
        if key in mapped or key not in required:
            raise ValueError("Duplicate or foreign source-condition row")
        meta = expected[key[0]]
        if row["role"] != meta["role"] or row["kind"] != meta["kind"]:
            raise ValueError("Frozen source role/kind changed")
        if set(row["scores"]) != set(arms):
            raise ValueError("Full requested arm set required")
        for arm, score in row["scores"].items():
            if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
                raise ValueError("Invalid statistic")
            if score is None and row.get("abstentions", {}).get(arm) not in (
                    "no_usable_observation", "generation_failure", "no_complete_valid_path"):
                raise ValueError("Missing execution or human labels cannot become abstention")
        mapped[key] = row
    if set(mapped) != required:
        raise ValueError("Incomplete full-denominator source-condition product")
    identities = {}
    for sid, meta in expected.items():
        if meta["role"] not in ("development", "calibration", "null_test", "positive_test"):
            raise ValueError("Unknown source role")
        for field in ("source_sha256", "url_sha256", "dataset_locator"):
            value = meta.get(field)
            if value:
                identity = field, value
                if identity in identities and identities[identity] != sid:
                    raise ValueError("Source reuse or cross-role identity leakage")
                identities[identity] = sid
    return mapped


def threshold(values, alpha):
    if not values or not 0 < alpha < 1:
        raise ValueError("Finite-sample calibration needs rows and alpha in (0,1)")
    ordered = sorted(float("-inf") if v is None else float(v) for v in values)
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    # JSON-safe infinity representation. Ties reject only strictly above threshold.
    result = ordered[rank - 1] if rank <= len(ordered) else float("inf")
    return {"value": result if math.isfinite(result) else None,
            "infinity": None if math.isfinite(result) else ("positive" if result > 0 else "negative"),
            "calibration_n": len(ordered), "order_statistic_1based": rank,
            "comparison": "strictly_greater"}


def reject(score, cut):
    if score is None or cut["infinity"] == "positive":
        return False
    return True if cut["infinity"] == "negative" else score > cut["value"]


def calibrate(rows, expected, conditions, arms=ARMS, alpha=.01):
    mapped = validate_rows(rows, expected, conditions, arms)
    groups = [sid for sid, meta in expected.items() if meta["role"] == "calibration"]
    if not groups:
        raise ValueError("Independent Null calibration missing")
    cuts = {}
    for arm in arms:
        # One contribution per original document. Max across declared conditions
        # gives one fixed threshold and avoids multiplying the calibration n.
        values = []
        for sid in groups:
            scores = [mapped[sid, condition]["scores"][arm] for condition in conditions]
            available = [s for s in scores if s is not None]
            values.append(max(available) if available else None)
        cuts[arm] = threshold(values, alpha)
    return {"arms": list(arms), "conditions": list(conditions), "alpha": alpha,
            "calibration_sources": sorted(groups),
            "calibration_composition": dict(Counter(expected[s]["kind"] for s in groups)),
            "condition_aggregation": "per_source_max", "thresholds": cuts,
            "thresholds_are_not_test_FPR_guarantees": True}


def binomial(success, total, tail):
    if not 0 <= success <= total or total <= 0:
        raise ValueError("Valid full binomial denominator required")
    return {
        "count": success, "n": total, "rate": success / total,
        "CP95": [0.0 if success == 0 else float(beta.ppf(.025, success, total - success + 1)),
                 1.0 if success == total else float(beta.ppf(.975, success + 1, total - success))],
        "simultaneous_one_sided_upper": 1.0 if success == total else float(
            beta.ppf(1 - tail, success + 1, total - success)),
    }


def paired_difference(left, right, seed=20261102, repetitions=10000):
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if len(delta) == 0:
        raise ValueError("Empty independent positive source denominator")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions)
    for start in range(0, repetitions, 100):
        end = min(start + 100, repetitions)
        bootstrap[start:end] = delta[rng.integers(0, len(delta), (end - start, len(delta)))].mean(1)
    return {"difference": float(delta.mean()), "source_n": len(delta),
            "paired_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrap_repetitions": repetitions, "seed": seed,
            "interval_is_exploratory_when_discordances_are_sparse": True}


def evaluate(rows, expected, frozen_calibration, primary_condition):
    arms, conditions = frozen_calibration["arms"], frozen_calibration["conditions"]
    mapped = validate_rows(rows, expected, conditions, arms)
    if primary_condition not in conditions:
        raise ValueError("Primary condition was not frozen")
    replay = calibrate(rows, expected, conditions, arms, frozen_calibration["alpha"])
    if replay != frozen_calibration:
        raise ValueError("Calibration sources, statistics or threshold drift")
    null_kinds = sorted({m["kind"] for m in expected.values() if m["role"] == "null_test"})
    positive = [sid for sid, meta in expected.items() if meta["role"] == "positive_test"]
    if not null_kinds or not positive:
        raise ValueError("Independent positive and negative tests required")
    tests = len(arms) * len(conditions) * len(null_kinds)
    fpr, tpr, paired = {}, {}, {}
    for arm in arms:
        cut = frozen_calibration["thresholds"][arm]
        fpr[arm], tpr[arm] = {}, {}
        for condition in conditions:
            fpr[arm][condition] = {}
            for kind in null_kinds:
                sources = [sid for sid, m in expected.items()
                           if m["role"] == "null_test" and m["kind"] == kind]
                count = sum(reject(mapped[s, condition]["scores"][arm], cut) for s in sources)
                fpr[arm][condition][kind] = binomial(count, len(sources), .05 / tests)
            count = sum(reject(mapped[s, condition]["scores"][arm], cut) for s in positive)
            tpr[arm][condition] = binomial(count, len(positive), .05)
        paired[arm] = [reject(mapped[s, primary_condition]["scores"][arm], cut) for s in positive]
    comparisons = {
        "A1_minus_" + other: paired_difference(paired["A1"], paired[other])
        for other in ("A0", "single_view") if other in paired and "A1" in paired
    }
    return {"FPR_by_condition_and_kind": fpr, "TPR_full_denominator": tpr,
            "primary_condition": primary_condition, "primary_paired_differences": comparisons,
            "FPR_simultaneous_test_count": tests, "FPR_multiplicity": "Bonferroni",
            "A2_FPR_interpretation": "Oracle-conditional diagnostic, not deployable guarantee",
            "identity_endpoints": None, "independent_quality_gate": None,
            "formal_Go": None, "additional_requirements": [
                "Complete independent fact-instance inventory and alignment endpoints",
                "Human source-faithfulness and meaning-preservation evidence",
                "Pretest power decision and full frozen model/source provenance",
            ]}
