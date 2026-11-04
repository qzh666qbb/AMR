"""Pure statistical calculations; no parser, scorer, model or detector imports."""

import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

from scipy.stats import binom, binomtest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party/SWAN_official_2ca0c207_clean/detection"


def z_score(green, n, lambda_null=0.25):
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("N must be a nonnegative integer")
    if not isinstance(green, int) or isinstance(green, bool) or not 0 <= green <= n:
        raise ValueError("G must be an integer in [0,N]")
    if not 0 < lambda_null < 1:
        raise ValueError("lambda must lie in (0,1)")
    return (green - lambda_null * n) / math.sqrt(
        n * lambda_null * (1 - lambda_null)
    ) if n else math.nan


def z_max(n, lambda_null=0.25):
    return z_score(n, n, lambda_null)


def minimum_green(n, threshold=2.33, lambda_null=0.25):
    return next((g for g in range(n + 1)
                 if z_score(g, n, lambda_null) > threshold), None)


def exact_mcnemar(losses, gains):
    if any(type(x) is not int or x < 0 for x in (losses, gains)):
        raise ValueError("Discordant counts must be nonnegative integers")
    return binomtest(losses, losses + gains, 0.5, alternative="greater").pvalue \
        if losses + gains else 1.0


def holm_adjust(pvalues):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Invalid p value")
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def conservative_threshold(scores, target_fpr=0.01):
    """Smallest observed threshold with strict exceedance <= target FPR."""
    if not scores or not 0 <= target_fpr < 1:
        raise ValueError("Nonempty calibration and valid FPR required")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("NaN/inf must be recorded as failures, never sorted away")
    ordered = sorted(scores, reverse=True)
    permitted = math.floor(len(ordered) * target_fpr + 1e-12)
    return ordered[permitted]


def paired_power(n, loss_probability, gain_probability, alpha=0.05 / 3,
                 minimum_effect=0.10):
    """Exact joint distribution of discordance count and its direction.

    Planning probability: one-sided exact McNemar p<=alpha AND observed
    marginal TPR loss >= minimum_effect. Bonferroni alpha is conservative
    for the registered three-test Holm family.
    """
    discordance = loss_probability + gain_probability
    if not 0 < discordance <= 1 or min(loss_probability, gain_probability) < 0:
        raise ValueError("Invalid discordance probabilities")
    power = 0.0
    for m in range(1, n + 1):
        # P(Binomial(m,.5)>=k) <= alpha, with integer/tie handling.
        critical = int(binom.ppf(1 - alpha, m, 0.5)) + 1
        effect_critical = math.ceil((m + minimum_effect * n) / 2 - 1e-12)
        k = max(critical, effect_critical)
        power += binom.pmf(m, n, discordance) * binom.sf(
            k - 1, m, loss_probability / discordance
        )
    return float(power)


def extracted_official_function(filename, function_name, namespace):
    """Extract only the named function; never execute upstream module imports."""
    tree = ast.parse((OFFICIAL / filename).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == function_name]
    if len(functions) != 1:
        raise ValueError("Official function lookup ambiguous")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(OFFICIAL / filename), "exec"), namespace)
    return namespace[function_name]


def verify_official():
    scores = []
    namespace = {
        "np": SimpleNamespace(sqrt=math.sqrt),
        "logger": logging.getLogger("formula_only"),
        "parse_amr": lambda text: text,
        "normalize_amr_variables": lambda text: text,
        "generate_template_amr": lambda text: text,
        "compute_s2match_score": lambda *_: scores.pop(0),
    }
    direct = extracted_official_function(
        "detection_utils.py", "detect_amr_sentences", namespace.copy()
    )
    preparsed = extracted_official_function(
        "detect_from_parsed_amrs.py", "compute_zscore_for_doc", namespace.copy()
    )
    checked = 0
    for n in range(33):
        for g in range(n + 1):
            for lam in (0.10, 0.25, 0.50):
                scores[:] = [1.0] * g + [0.0] * (n - g)
                actual = preparsed(0, ["unit"] * n, ["bank"], .65, lam, False)[1]
                expected = z_score(g, n, lam)
                assert math.isnan(actual) if not n else math.isclose(actual, expected)
                scores[:] = [1.0] * g + [0.0] * (n - g)
                actual = direct(["unit"] * n, ["bank"], .65, lam)
                assert math.isnan(actual) if n < 2 else math.isclose(actual, expected)
                checked += 2
    return {
        "comparisons_with_mocked_graph_functions": checked,
        "real_model_parser_s2match_detector_calls": 0,
        "official_source_sha256": {
            filename: hashlib.sha256((OFFICIAL / filename).read_bytes()).hexdigest()
            for filename in ("detection_utils.py", "detect_from_parsed_amrs.py")
        },
        "n1_direct": "NaN", "n1_preparsed_max": z_max(1),
    }


def report():
    assumptions = [(0.25, 0.05), (0.30, 0.10), (0.35, 0.15)]
    # A fixed 200-pair design is evaluated rather than choosing N from outcomes.
    return {
        "kind": "pre_execution_formula_and_design_calculation",
        "official_verification": verify_official(),
        "z_table": [{"N": n, "lambda": lam, "z_max": z_max(n, lam),
                     "minimum_green_strict_2_33": minimum_green(n, 2.33, lam)}
                    for lam in (0.10, 0.25, 0.50)
                    for n in (1, 2, 3, 5, 8, 10, 12, 16, 24, 32, 40)],
        "planning": {
            "minimum_complete_quality_pairs": 200,
            "assumptions_not_estimated_from_pilot": True,
            "family": ["H1", "H2", "H3"], "holm_fwer": 0.05,
            "power_calculation_alpha": 0.05 / 3,
            "minimum_observed_effect": 0.10,
            "design_effect": 0.20,
            "sensitivity": [
                {"n": n, "p_loss": loss, "p_gain": gain,
                 "joint_power": paired_power(n, loss, gain)}
                for n in (100, 150, 200, 250) for loss, gain in assumptions
            ],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(json.dumps(report(), indent=2) + "\n")
    print(args.output)
