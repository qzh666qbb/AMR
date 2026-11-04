"""Calibrate only on frozen negatives, retain all document denominators."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.swan_blind_protocol import digest, official_two_stage_summary

ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def unit_threshold(values, q=.01):
    ordered = sorted(values, reverse=True)
    threshold = ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return threshold, threshold == max(ordered)


def doc_threshold(values, q):
    return sorted(values)[max(0, math.ceil((1 - q) * len(values)) - 1)]


def interval(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.,
            float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.]


def cluster_interval(rows, decisions, seed=20261023):
    import numpy as np
    ids = sorted({r["document_id"] for r in rows})
    groups = [[int(decisions.get(r["record_id"]) is True) for r in rows
               if r["document_id"] == doc] for doc in ids]
    counts = np.array([sum(g) for g in groups])
    denominators = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ids), (2000, len(ids)))
    rates = counts[samples].sum(axis=1) / denominators[samples].sum(axis=1)
    return [float(v) for v in np.quantile(rates, [.025, .975])]


def metrics(rows, decisions, statistics, threshold, available):
    n = len(rows)
    if not n:
        return None
    unresolved = sum(decisions.get(r["record_id"]) is None for r in rows)
    k = sum(decisions.get(r["record_id"]) is True for r in rows)
    successful = n - unresolved
    return {
        "decisions_positive": k, "denominator": n, "unresolved": unresolved,
        "unresolved_rate": unresolved / n,
        "rate": k / n if available else None,
        "conditional_rate": k / successful if successful and available else None,
        "complete_decision_documents": successful,
        "ties_at_threshold": sum(statistics.get(r["record_id"]) == threshold
                                 for r in rows) if available else None,
        "binomial_interval_95": interval(k, n) if available and
        len({r["attack"] for r in rows}) == 1 else None,
        "binomial_interval_scope": "single-condition-independent-source-documents"
        if len({r["attack"] for r in rows}) == 1 else
        "not_applicable_to_correlated-pooled-transformations",
        "cluster_bootstrap_95": cluster_interval(rows, decisions) if available else None,
        "source_documents": len({r["document_id"] for r in rows}),
        "failure_reasons": dict(Counter(reason for r in rows
                                       for reason in r["failure_counts"])),
    }


def analyze(folder):
    manifest = json.loads((folder / "FREEZE.json").read_text())
    rows = json.loads((folder / "DOCUMENT_SCORES.json").read_text())["documents"]
    methods = sorted({r["method"] for r in rows}, key=lambda m:
                     ["P0_raw", "P0_strict", "S6", "S8", "S10", "S12"].index(m))
    result = {
        "schema": "swan-blind-eval/v1", "freeze_sha256": digest(manifest),
        "complete_methods": methods, "planned_methods": manifest["evaluation_methods"],
        "scope": "explored-dataset5-paired-generated-negatives-development-evaluation",
        "primary_test": "last-15-paired-source-document-ids",
        "targets_are_nominal": True, "decision_rule": "statistic > document_threshold",
        "bootstrap_replicates": 2000, "independent_human_test": False,
        "per_condition_negative_resolution": "1/15 = 6.67 percentage points",
        "zero_of_15_fpr_upper_95": interval(0, 15)[1],
        "methods": {},
    }
    csv_rows, calibration = [], {}
    for method in methods:
        data = [r for r in rows if r["method"] == method]
        negatives = [r for r in data if r["arm"] == "paired_negative"
                     and r["split"] == "calibration"]
        complete = [r for r in negatives if r["complete"]]
        values = [s for r in complete for s in r["unit_bank_max"]]
        saturated = False
        unit = .65 if method.startswith("P0") else None
        if not method.startswith("P0") and values:
            unit, saturated = unit_threshold(values)
        available = bool(complete) and unit is not None and not saturated
        stats = {}
        for r in data:
            if r["complete"] and unit is not None:
                if method.startswith("P0"):
                    stats[r["record_id"]] = official_two_stage_summary(
                        r["unit_bank_max"])["z"]
                else:
                    stats[r["record_id"]] = sum(
                        s >= unit for s in r["unit_bank_max"]) / r["N_expected"]
        cal_values = [stats[r["record_id"]] for r in complete
                      if stats.get(r["record_id"]) is not None]
        available = available and bool(cal_values)
        method_result = {
            "configuration_sha256": data[0]["configuration_sha256"],
            "unit_threshold": unit, "unit_target": .01 if not method.startswith("P0")
            else None, "unit_threshold_saturated": saturated,
            "calibration_documents": len(negatives),
            "complete_calibration_documents": len(complete),
            "calibration_source_documents": len({r["document_id"] for r in negatives}),
            "calibration_unit_count": len(values),
            "calibration_unit_green_rate": sum(
                s is not None and s >= unit for s in values) / len(values)
            if values and unit is not None else None,
            "availability": "available" if available else "unavailable",
            "unavailability_reason": "unit_1pct_threshold_saturated" if saturated
            else ("no_complete_calibration" if not available else None),
            "operating_points": {},
        }
        cal_manifest = digest([(r["record_id"], r["text_sha256"]) for r in negatives])
        calibration[method] = {"configuration_sha256": data[0]["configuration_sha256"],
                               "negative_manifest_sha256": cal_manifest,
                               "unit_threshold": unit, "operating_points": {}}
        for q in (.01, .05):
            c = doc_threshold(cal_values, q) if available else None
            decisions = {r["record_id"]: stats.get(r["record_id"]) > c
                         if available and stats.get(r["record_id"]) is not None
                         else None for r in data}
            op = {
                "nominal_document_fpr": q, "document_threshold": c,
                "calibration": metrics(negatives, decisions, stats, c, available),
                "conditions": {},
            }
            for attack in (*ATTACKS, "pooled"):
                condition_rows = [r for r in data if attack == "pooled"
                                  or r["attack"] == attack]
                test_positive = [r for r in condition_rows if r["arm"] == "positive"
                                 and r["split"] == "test"]
                test_negative = [r for r in condition_rows
                                 if r["arm"] == "paired_negative" and r["split"] == "test"]
                auxiliary = [r for r in condition_rows if r["arm"] == "positive"]
                measure = {
                    "TPR": metrics(test_positive, decisions, stats, c, available),
                    "FPR": metrics(test_negative, decisions, stats, c, available),
                    "all30_positive_auxiliary": metrics(
                        auxiliary, decisions, stats, c, available),
                }
                op["conditions"][attack] = measure
                csv_rows.append({
                    "method": method, "nominal_document_fpr": q, "condition": attack,
                    "unit_threshold": unit, "document_threshold": c,
                    "TP": measure["TPR"]["decisions_positive"],
                    "positive_denominator": measure["TPR"]["denominator"],
                    "TPR": measure["TPR"]["rate"],
                    "positive_unresolved": measure["TPR"]["unresolved"],
                    "FP": measure["FPR"]["decisions_positive"],
                    "negative_denominator": measure["FPR"]["denominator"],
                    "FPR": measure["FPR"]["rate"],
                    "negative_unresolved": measure["FPR"]["unresolved"],
                })
            name = f"document_{int(q * 100)}pct"
            method_result["operating_points"][name] = op
            calibration[method]["operating_points"][name] = c
        if method.startswith("P0") and cal_values:
            reference = {}
            for q, fallback in ((.01, 2.33), (.05, 1.645)):
                hits = [i * .005 for i in range(1200)
                        if sum(z > i * .005 for z in cal_values) / len(cal_values) <= q]
                c = hits[0] if hits else fallback
                decisions = {r["record_id"]: stats[r["record_id"]] > c
                             if r["record_id"] in stats else None for r in data}
                reference[str(q)] = {
                    "threshold": c, "fallback_used": not bool(hits),
                    "calibration_fpr": sum(z > c for z in cal_values) / len(cal_values),
                    "test_pooled_FPR": metrics(
                        [r for r in data if r["arm"] == "paired_negative"
                         and r["split"] == "test"], decisions, stats, c, True),
                    "test_pooled_TPR": metrics(
                        [r for r in data if r["arm"] == "positive" and r["split"] == "test"],
                        decisions, stats, c, True),
                }
            method_result["official_grid_auxiliary"] = reference
        result["methods"][method] = method_result
    save_json(folder / "CALIBRATION.json", calibration)
    save_json(folder / "RESULTS.json", result)
    stream = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (folder / "METRICS.csv").write_text(stream.getvalue())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.output)
    print(json.dumps({m: v["operating_points"]["document_1pct"]["conditions"]["pooled"]
                      for m, v in summary["methods"].items()}, indent=2))
