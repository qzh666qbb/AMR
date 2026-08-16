# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
detection_utils.py

Shared utilities for AMR-based watermark detection (§3.3).

Provides:
  - detect_amr_sentences():    Parse sentences → compare to bank → compute z-score
  - evaluate_z_scores_amr():   Compute AUROC, TPR@1%FPR, TPR@5%FPR from z-scores
  - find_fpr_thresholds_amr(): Find z-score thresholds for target FPR levels
"""

import os
import sys
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

from utils.amr_utils import (
    parse_amr,
    normalize_amr_variables,
    generate_template_amr,
    compute_s2match_score
)


def detect_amr_sentences(sents, amr_bank, threshold, lmbd, is_human=False):
    """
    For each sentence in `sents`:
      1) Parse to AMR string.
      2) Normalize + template + normalize => 'cand_final' (the 'normalized template AMR').
      3) Compare cand_final with each AMR in `amr_bank` via s2match; take best score.
      4) If best score >= threshold, count sentence as "green".
    After that, compute one-proportion z-score:
        z = (count_green - lmbd*N) / sqrt(N*lmbd*(1-lmbd))

    Note: `sents` should not include the original prompt. Callers (e.g. the
    end-to-end detection script) are expected to pass only the sentences that
    should participate in the z-test, consistent with how parsed AMRs are
    stored in the two-stage detection pipeline.

    Args:
        sents (list of str): The text split into sentences.
        amr_bank (set or list of str): AMRs stored in normalized template form.
        threshold (float): s2match threshold above which a sentence is "green".
        lmbd (float): Expected fraction of green sentences under random chance.

    Returns:
        float: z-score for the entire text (document).
    """
    n_sents = len(sents)
    if n_sents < 2:
        # If there's 0 or 1 sentence, we can't do a meaningful z-test
        return float('nan')

    count_green = 0

    for sent in sents:
        try:
            # 1) Parse to AMR
            candidate_amr = parse_amr(sent)
            if not candidate_amr.strip():
                # If parsing fails or the model returned empty, skip
                continue
        except Exception as e:
            # On any parse error, skip
            continue

        # 2) Normalize + template + normalize again
        cand_norm = normalize_amr_variables(candidate_amr)
        cand_templ = generate_template_amr(cand_norm)
        cand_final = normalize_amr_variables(cand_templ)

        # 3) Compare with all AMRs in the bank (already normalized template form)
        best_score = 0.0
        for bank_amr in amr_bank:
            score = compute_s2match_score(cand_final, bank_amr)
            if score > best_score:
                best_score = score

        logger.debug(f"{'HUMAN' if is_human else 'MACHINE'} best_score: {best_score:.3f}")
        # 4) Check threshold
        if best_score >= threshold:
            count_green += 1

    logger.debug(f"Counted {count_green} green sentences out of {n_sents} total")

    # 5) One-proportion z-score
    numerator = count_green - (lmbd * n_sents)
    denominator = np.sqrt(n_sents * lmbd * (1 - lmbd))
    if denominator < 1e-9:
        return float('nan')

    return numerator / denominator


def evaluate_z_scores_amr(machine_scores, human_scores, output_dir):
    """
    Evaluate detection performance using only:
      - machine_scores (positive class)
      - human_scores   (negative class)
    
    We compute:
      1) AUROC (Area Under ROC curve)
      2) Z-score thresholds at ~1% and ~5% FPR among human text
      3) TPR at those thresholds (fraction of machine texts correctly detected)

    Args:
        machine_scores (list or np.array): z-scores for machine-generated texts
        human_scores (list or np.array): z-scores for human texts
        output_dir (str): directory to save the ROC plot (amr_roc_curve.png)

    Returns:
        (float, float, float):
          - roc_auc: area under ROC
          - tpr_at_fpr1: true positive rate (fraction of machine texts detected) at 1% FPR
          - tpr_at_fpr5: true positive rate (fraction of machine texts detected) at 5% FPR
    """
    machine_scores = np.array(machine_scores)
    human_scores = np.array(human_scores)

    # 1) Build label + score arrays for ROC
    labels = np.concatenate([
        np.ones(len(machine_scores)),   # 1 for machine
        np.zeros(len(human_scores))     # 0 for human
    ])
    scores = np.concatenate([machine_scores, human_scores])

    # 2) Compute ROC curve, AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 3) Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.title("ROC Curve (AMR-based Watermark Detection)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, "amr_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # 4) Find the thresholds that give ~1% and ~5% FPR among human scores
    fpr_1_thresh, fpr_5_thresh = find_fpr_thresholds_amr(human_scores)

    # 5) TPR = fraction of machine texts above those thresholds
    tpr_at_fpr1 = np.mean(machine_scores > fpr_1_thresh)
    tpr_at_fpr5 = np.mean(machine_scores > fpr_5_thresh)

    return roc_auc, tpr_at_fpr1, tpr_at_fpr5


def find_fpr_thresholds_amr(human_scores, step=0.005):
    """
    Scans from z=0 to z=6 in increments of `step`.
    Finds thresholds where ~1% and ~5% of human scores exceed that threshold.

    If none is found, we fallback to standard z-scores (2.33 for ~1%, 1.645 for ~5%).

    Returns:
        (float, float): (threshold_for_1pct, threshold_for_5pct)
    """
    hs = np.array(human_scores)
    if len(hs) == 0:
        # fallback if no human data available
        return 2.33, 1.645

    fpr_1_threshold = 0
    fpr_5_threshold = 0

    # Sweep thresholds
    for z_threshold in np.arange(0, 6, step):
        # fraction of human texts with z-score > z_threshold
        frac_exceed = np.mean(hs > z_threshold)
        # ~1%
        if 0.0095 <= frac_exceed <= 0.0105:
            fpr_1_threshold = z_threshold
        # ~5%
        if 0.045 <= frac_exceed <= 0.055:
            fpr_5_threshold = z_threshold

    # If we didn't find an exact threshold, fallback
    if fpr_1_threshold == 0:
        fpr_1_threshold = 2.33  # ~z-value for 1%
    if fpr_5_threshold == 0:
        fpr_5_threshold = 1.645 # ~z-value for 5%

    return fpr_1_threshold, fpr_5_threshold