"""Independent arithmetic checks on saved predictions, without model fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/validation1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    base = args.input
    target = base / "ANALYSIS_CHECKS.json"
    if target.exists():
        raise FileExistsError(target)
    summary = json.loads((base / "SUMMARY.json").read_text())
    assert sha(base / "PREDICTIONS.npz") == summary["predictions_sha256"]
    predictions = np.load(base / "PREDICTIONS.npz", allow_pickle=False)
    keys = predictions["keys"]
    attacks = np.unique(keys[:, 1])
    checked_thresholds = checked_rates = 0
    for name, run in summary["runs"].items():
        positive = predictions[name + "__positive"].astype(float)
        negative = predictions[name + "__negative"].astype(float)
        calibration = predictions[name + "__calibration_negative"].astype(float)
        # Direct pairwise comparisons, independently of the rank-count AUC code.
        auc = sum(float(((chunk[:, None] > negative[None, :]).astype(float)
                         + 0.5 * (chunk[:, None] == negative[None, :])).sum())
                  for chunk in np.array_split(positive, 16))
        auc /= len(positive) * len(negative)
        assert abs(auc - run["auroc"]) < 1e-12
        for alpha, point in run["working_points"].items():
            rank = math.ceil((len(calibration) + 1) * (1 - float(alpha)))
            threshold = np.sort(calibration)[rank - 1]
            assert threshold == point["calibration_threshold"]
            checked_thresholds += 1
            for scores, metric in ((positive, "tpr"), (negative, "fpr")):
                flags = np.isfinite(scores) & (scores > threshold)
                assert abs(float(flags.mean()) - point[metric]) < 1e-12
                checked_rates += 1
        assert len(positive) == len(negative) == len(keys)
    result = {
        "status": "passed", "checks_are_on_real_saved_predictions": True,
        "checkpoint_replay_verified_by_summary": summary["analysis"]["checkpoint_replay_tpr_fpr_matches_original"],
        "pairwise_AUCs_checked": len(summary["runs"]),
        "thresholds_checked": checked_thresholds, "TPR_FPR_checked": checked_rates,
        "test_rows_per_class": len(keys),
        "test_rows_per_class_by_attack": {a: int(np.sum(keys[:, 1] == a)) for a in attacks},
        "summary_sha256": sha(base / "SUMMARY.json"),
        "predictions_sha256": sha(base / "PREDICTIONS.npz"),
        "verification_script_sha256": sha(__file__),
        "formal_detection_validated": False,
    }
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
