"""Shared detection statistics and source-bound calibration; no target selection."""
import math
import re

CONDITIONS = ("clean", "semicolon_all", "comma_all", "newline_all", "space_all")
FLOWS = ("public", "a", "b", "joint", "pooled")
BANK_KEYS = ("public", "a", "b")
MATCH_THRESHOLD = .65
LAMBDA = .25
ALPHA = .01


def variants(text, tokenizer):
    spans = list(tokenizer.span_tokenize(text))
    if not spans:
        raise ValueError("Empty source")
    result = {"clean": text}
    for condition, replacement in (("semicolon_all", ";"), ("comma_all", ","),
                                    ("newline_all", "\n"), ("space_all", " ")):
        value = text
        for start, end in reversed(spans[:-1]):
            segment = text[start:end]
            match = re.search(r"[.!?]+(?=[\"')\]]*$)", segment)
            if not match:
                raise ValueError("Frozen source boundary lacks terminal punctuation")
            a, b = start + match.start(), start + match.end()
            value = value[:a] + replacement + value[b:]
        # Exact ordered lexical sequence, including numbers, contractions, names.
        if re.findall(r"\w+", value, re.UNICODE) != re.findall(r"\w+", text, re.UNICODE):
            raise ValueError("Boundary transform altered words")
        result[condition] = value
    return result


def document_statistics(rows):
    """Official >=.65 green count; invalid AMRs remain in N; N<2 abstains."""
    n = len(rows)
    greens = {key: sum(r[key] is not None and r[key] >= MATCH_THRESHOLD for r in rows)
              for key in BANK_KEYS}
    greens["pooled"] = sum(any(r[k] is not None and r[k] >= MATCH_THRESHOLD
                               for k in ("a", "b")) for r in rows)
    z = {key: ((value - LAMBDA * n) / math.sqrt(n * LAMBDA * (1 - LAMBDA))
               if n >= 2 else None) for key, value in greens.items()}
    z["joint"] = (z["a"] + z["b"]) / math.sqrt(2) if n >= 2 else None
    return {"sentences": n, "green": greens, "scores": z,
            "erased": {k: sum(r[k] is None for r in rows) for k in BANK_KEYS},
            "abstain": n < 2}


def fit_threshold(rows, expected_ids, flow, condition, alpha=ALPHA):
    if flow not in FLOWS or condition not in CONDITIONS or not 0 < alpha < 1:
        raise ValueError("Unknown flow, condition or alpha")
    if (len(rows) != len(expected_ids) or len(set(expected_ids)) != len(expected_ids)
            or {r["source_id"] for r in rows} != set(expected_ids)):
        raise ValueError("Calibration source denominator differs")
    if any(r["role"] != "calibration" or r["condition"] != condition for r in rows):
        raise ValueError("Calibration/test or condition leakage")
    values = [r["scores"][flow] for r in rows]
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values):
        raise ValueError("Invalid score")
    n = len(values)
    # Abstentions rank below finite scores and remain in the source denominator.
    rank = math.ceil((n + 1) * (1 - alpha))
    ordered = sorted(v for v in values if v is not None)
    rank_finite = rank - sum(v is None for v in values)
    disabled = rank > n or not ordered or rank_finite <= 0
    threshold = None if disabled else ordered[rank_finite - 1]
    return {"threshold": threshold, "disabled": disabled, "rank": rank, "n": n,
            "abstentions": sum(v is None for v in values), "alpha": alpha,
            "comparison": "strictly_greater", "flow": flow, "condition": condition}


def detected(value, threshold):
    return bool(value is not None and not threshold["disabled"]
                and value > threshold["threshold"])


def binomial_interval(k, n):
    from scipy.stats import beta
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid denominator")
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.,
            float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.]


def evaluate_null(rows, expected_ids, thresholds):
    wanted = {(i, c) for i in expected_ids for c in CONDITIONS}
    if (len(rows) != len(wanted) or {(r["source_id"], r["condition"]) for r in rows} != wanted
            or any(r["role"] != "null" for r in rows)):
        raise ValueError("Independent Null denominator differs")
    result = []
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        for flow in FLOWS:
            for reference in ("clean", condition) if condition != "clean" else ("clean",):
                threshold = thresholds[reference][flow]
                positives = sum(detected(r["scores"][flow], threshold) for r in subset)
                result.append({"flow": flow, "condition": condition,
                               "threshold_condition": reference, "false_positives": positives,
                               "sources": len(subset), "FPR": positives / len(subset),
                               "FPR_CP95": binomial_interval(positives, len(subset)),
                               "abstentions": sum(r["scores"][flow] is None for r in subset)})
    return result
