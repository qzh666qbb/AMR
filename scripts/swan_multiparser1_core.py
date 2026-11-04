"""Paired parser statistics; thresholds fit only to independent calibration sources."""
import math
from swan_independent_calibration1_core import (
    CONDITIONS, fit_threshold as official_fit, detected, binomial_interval)

FLOWS = ("amrlib", "amrbart", "ensemble")


def statistics(first, second):
    if len(first) != len(second):
        raise ValueError("Parsers must see the identical sentence candidates")
    n = len(first)
    green = {key: sum(v is not None and v >= .65 for v in values)
             for key, values in (("amrlib", first), ("amrbart", second))}
    z = {key: (value - .25 * n) / math.sqrt(n * .25 * .75) if n >= 2 else None
         for key, value in green.items()}
    z["ensemble"] = (z["amrlib"] + z["amrbart"]) / math.sqrt(2) if n >= 2 else None
    return {"scores": z, "sentences": n, "green": green, "abstain": n < 2,
            "erasures": {"amrlib": sum(v is None for v in first),
                         "amrbart": sum(v is None for v in second)},
            "complementarity": {
                "both_green": sum(a is not None and b is not None and a >= .65 and b >= .65
                                  for a, b in zip(first, second)),
                "amrlib_only_green": sum(a is not None and a >= .65 and (b is None or b < .65)
                                        for a, b in zip(first, second)),
                "amrbart_only_green": sum(b is not None and b >= .65 and (a is None or a < .65)
                                         for a, b in zip(first, second))}}


def thresholds(rows, ids):
    result = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        result[condition] = {}
        for flow in FLOWS:
            converted = [{**r, "scores": {"public": r["scores"][flow]}} for r in selected]
            result[condition][flow] = {
                **official_fit(converted, ids, "public", condition), "flow": flow}
    return result


def null_summary(rows, ids, fitted):
    wanted = {(i, c) for i in ids for c in CONDITIONS}
    if (len(rows) != len(wanted) or {(r["source_id"], r["condition"]) for r in rows} != wanted
            or any(r["role"] != "null" for r in rows)):
        raise ValueError("Full independent Null denominator required")
    result = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        for flow in FLOWS:
            for reference in ("clean", condition) if condition != "clean" else ("clean",):
                k = sum(detected(r["scores"][flow], fitted[reference][flow]) for r in selected)
                result.append({"flow": flow, "condition": condition, "threshold_condition": reference,
                               "false_positives": k, "sources": len(ids), "FPR": k / len(ids),
                               "FPR_CP95": binomial_interval(k, len(ids))})
    return result
