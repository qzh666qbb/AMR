# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
evaluate_z_scores.py

Standalone evaluation script: loads pre-computed z-scores from .npy files
and computes detection metrics (AUROC, TPR@1%FPR, TPR@5%FPR).

Usage:
    python detection/evaluate_z_scores.py \\
        --machine_z_scores results/machine_z_scores.npy \\
        --human_z_scores results/human_z_scores.npy \\
        --output_dir results/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detection.detection_utils import evaluate_z_scores_amr

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AMR-based watermark detection from saved z-scores.")
    parser.add_argument("--machine_z_scores", type=str, default="machine_z_scores.npy",
                        help="Path to the .npy file containing machine-generated text z-scores.")
    parser.add_argument("--human_z_scores", type=str, default="human_z_scores.npy",
                        help="Path to the .npy file containing human text z-scores.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the ROC curve plot and results.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.machine_z_scores):
        raise FileNotFoundError(f"Machine z-scores file not found at {args.machine_z_scores}")
    if not os.path.exists(args.human_z_scores):
        raise FileNotFoundError(f"Human z-scores file not found at {args.human_z_scores}")

    # 1) Load the saved z-scores
    machine_scores = np.load(args.machine_z_scores)
    human_scores = np.load(args.human_z_scores)

    # 2) Evaluate performance
    print(f"Computing AUC for files:\n  Machine: {args.machine_z_scores}\n  Human: {args.human_z_scores}")
    auroc, tpr_at_fpr1, tpr_at_fpr5 = evaluate_z_scores_amr(machine_scores, human_scores, args.output_dir)

    # 3) Print results to console
    print("\n--- Detection Metrics ---")
    print(f"AUROC: {auroc:.3f}")
    print(f"TPR at 1% FPR: {tpr_at_fpr1:.3f}")
    print(f"TPR at 5% FPR: {tpr_at_fpr5:.3f}")

    # 4) Save metrics to CSV
    results_path = os.path.join(args.output_dir, "results.csv")
    columns = ["auroc", "tpr_at_fpr1", "tpr_at_fpr5"]
    metrics = [f"{auroc:.3f}", f"{tpr_at_fpr1:.3f}", f"{tpr_at_fpr5:.3f}"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)
    print(f"\n[Done] Results saved to {results_path}")

if __name__ == "__main__":
    main()
