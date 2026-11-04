"""Independently verify saved operating-point thresholds, rates and selection."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_multiscale_validation import sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/operating_points1"


def order_threshold(values, alpha):
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        return None
    return float(np.sort(values)[rank - 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    base = args.input
    if (base / "ANALYSIS_CHECKS.json").exists():
        raise FileExistsError(base / "ANALYSIS_CHECKS.json")
    summary = json.loads((base / "SUMMARY.json").read_text())
    protocol = json.loads((base / "RUN_PROTOCOL.json").read_text())
    predictions = np.load(base / "PREDICTIONS.npz", allow_pickle=False)
    assert sha(base / "PREDICTIONS.npz") == summary["predictions_sha256"]
    assert sha(base / "SELECTION.json") == summary["selection_sha256"]
    count_thresholds = count_rates = 0
    policies = ("pooled_05", "pooled_03", "pooled_02", "condition_max_05")
    for name, run in summary["runs"].items():
        calibration = predictions[f"{name}__calibration_negative"]
        calibration_keys = predictions["keys__calibration"]
        for policy, row in run["policies"].items():
            alpha = {"pooled_05": .05, "pooled_03": .03,
                     "pooled_02": .02, "condition_max_05": .05}[policy]
            thresholds = [order_threshold(calibration, alpha)]
            if policy == "condition_max_05":
                for attack in np.unique(calibration_keys[:, 1]):
                    value = order_threshold(calibration[calibration_keys[:, 1] == attack], .05)
                    thresholds.append(value)
                    assert value == row["calibration"]["sources"][attack]["threshold"]
            threshold = max(thresholds)
            assert threshold == row["calibration"]["threshold"]
            count_thresholds += 1
            for part, rates in row["partition_metrics"].items():
                keys = predictions[f"keys__{part}"]
                for side, metric in (("negative", "fpr"), ("positive", "tpr")):
                    if part == "calibration" and side == "positive":
                        continue
                    values = predictions[f"{name}__{part}_{side}"]
                    flags = np.isfinite(values) & (values > threshold)
                    for attack, point in rates.items():
                        mask = np.ones(len(values), dtype=bool) if attack == "all" else keys[:, 1] == attack
                        assert abs(float(flags[mask].mean()) - point[metric]) < 1e-12
                        count_rates += 1
    for stage in ("A2", "A3"):
        eligible = []
        utilities = {}
        for policy in policies:
            usable = True
            tpr = []
            for seed in protocol["seeds"]:
                rates = summary["runs"][f"{stage}_{seed}"]["policies"][policy]["partition_metrics"]
                tpr.append(rates["validation"]["all"]["tpr"])
                usable &= all(point["fpr"] <= .05 for part in ("validation", "calibration")
                              for point in rates[part].values())
            assert usable == summary["selection"][stage]["candidates"][policy]["eligible"]
            utilities[policy] = sum(tpr) / len(tpr)
            if usable:
                eligible.append(policy)
        selected = max(eligible, key=lambda p: utilities[p]) if eligible else None
        assert selected == summary["selection"][stage]["chosen_policy"]
    result = {
        "status": "passed", "models_checked": len(summary["runs"]),
        "unified_thresholds_checked": count_thresholds,
        "partition_and_condition_rates_checked": count_rates,
        "selection_recomputed_without_test_rates": True,
        "saved_baseline_and_prediction_replay_exact": summary["analysis"]["baseline_and_prediction_replay_exact"],
        "summary_sha256": sha(base / "SUMMARY.json"), "predictions_sha256": sha(base / "PREDICTIONS.npz"),
        "verification_script_sha256": sha(__file__),
        "independent_human_test_fpr": None,
    }
    with (base / "ANALYSIS_CHECKS.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
