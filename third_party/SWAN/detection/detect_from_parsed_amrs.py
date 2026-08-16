# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
detect_from_parsed_amrs.py

Stage 2 of the two-stage detection pipeline (§3.3):

1) Load pre-parsed machine AMRs (JSON) from parse_machine_text.py
2) Load pre-parsed human AMRs (JSON) from parse_human_text.py
3) Load the curated AMR bank (normalized template AMRs)
4) Compare each doc's AMRs to the bank via S2match, count fraction >= threshold
5) Compute z-score per document, evaluate detection metrics (AUROC, TPR@1%FPR, TPR@5%FPR)

Usage:
    python detection/detect_from_parsed_amrs.py \\
        --parsed_machine output/watermarked/parsed_amrs.json \\
        --parsed_human amr_bank/artifacts/human_parsed_amrs_250.json \\
        --amr_bank_path amr_bank/banks/amr_bank_50.json \\
        --output_dir results/detection
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

from detection.detection_utils import evaluate_z_scores_amr
from utils.amr_utils import load_amr_bank, compute_s2match_score


def parse_args():
    parser = argparse.ArgumentParser(description="CPU-based detection on pre-parsed AMRs.")
    parser.add_argument(
        '--parsed_machine',
        required=True,
        help='Path to JSON of final machine AMRs (from parse_machine_text.py).'
    )
    parser.add_argument(
        '--parsed_human',
        required=True,
        help='Path to JSON of final human AMRs (from parse_human_text.py).'
    )
    parser.add_argument(
        '--amr_bank_path',
        type=str,
        default="amr_bank/banks/amr_bank_50.json",
        help='Path to your curated AMR bank (normalized template AMRs).'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.65,
        help='s2match threshold above which a sentence is counted as "green".'
    )
    parser.add_argument(
        '--lmbd',
        type=float,
        default=0.25,
        help='Expected fraction of green sentences if random (the λ).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save detection outputs (z-scores, ROC plot, etc.).'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=48,
        help='Number of CPU workers to parallelize detection.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the AMR bank
    amr_bank = load_amr_bank(args.amr_bank_path)
    print("Threshold: ", args.threshold)

    # 2) Load pre-parsed AMRs (list-of-lists)
    #   machine_amrs[i] => list of final AMRs for doc i (machine)
    #   human_amrs[i]   => list of final AMRs for doc i (human)
    with open(args.parsed_machine, 'r') as f:
        machine_amrs = json.load(f)
    with open(args.parsed_human, 'r') as f:
        human_amrs = json.load(f)

    print(f"[INFO] #Machine docs = {len(machine_amrs)}, #Human docs = {len(human_amrs)}")

    # 3) Detect: compute z-scores for machine docs
    print("Computing z-scores for machine text ...")
    machine_scores, machine_best_matches = compute_doc_zscores_parallel(
        docs_amrs=machine_amrs,
        amr_bank=amr_bank,
        threshold=args.threshold,
        lmbd=args.lmbd,
        is_human=False,
        num_workers=args.num_workers
    )

    # 4) Detect: compute z-scores for human docs
    print("Computing z-scores for human text ...")
    human_scores, human_best_matches = compute_doc_zscores_parallel(
        docs_amrs=human_amrs,
        amr_bank=amr_bank,
        threshold=args.threshold,
        lmbd=args.lmbd,
        is_human=True,
        num_workers=args.num_workers
    )

    # 5) Save z-scores to .npy
    machine_score_path = os.path.join(args.output_dir, "machine_z_scores.npy")
    human_score_path = os.path.join(args.output_dir, "human_z_scores.npy")
    np.save(machine_score_path, machine_scores)
    np.save(human_score_path, human_scores)

    # 6) Save AMR match details
    machine_matches_dict = {
        "matched_amrs": [
            [
                {
                    "score": match_score,
                    "best_matching_amr": best_amr
                }
                for match_score, best_amr in doc_matches
            ]
            for doc_matches in machine_best_matches
        ]
    }

    machine_matches_out_path = os.path.join(args.output_dir, "machine_amr_matches.json")
    with open(machine_matches_out_path, 'w') as f:
        json.dump(machine_matches_dict, f, indent=2)
    

    # 7) Evaluate detection performance
    print("Evaluating z-scores for AMR-based watermark detection...")
    auroc, tpr_at_fpr1, tpr_at_fpr5 = evaluate_z_scores_amr(machine_scores, human_scores, args.output_dir)

    # 8) Save metrics (AUROC, TPR@1%FPR, TPR@5%FPR) to results.csv
    results_path = os.path.join(args.output_dir, "results.csv")
    columns = ["auroc", "tpr_at_fpr1", "tpr_at_fpr5"]
    metrics = [f"{auroc:.3f}", f"{tpr_at_fpr1:.3f}", f"{tpr_at_fpr5:.3f}"]
    df = pd.DataFrame([metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)

    print(f"\n[Done] Results saved to {results_path}")


def compute_doc_zscores_parallel(docs_amrs, amr_bank, threshold, lmbd, is_human, num_workers):
    """
    docs_amrs: list of docs, each doc is a list of final AMRs (one per sentence).
    is_human: if True, we still skip the first sentence (prompt),
              but there's no further truncation here since it was done in parse_human_text.py.
    Return: list of z-scores, one per doc.
    """
    results = [None] * len(docs_amrs)
    best_matches = [None] * len(docs_amrs)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, doc_amrs in enumerate(docs_amrs):
            futures.append(executor.submit(
                compute_zscore_for_doc,
                i,
                doc_amrs,
                amr_bank,
                threshold,
                lmbd,
                is_human
            ))
        for f in as_completed(futures):
            idx, z_score, doc_best_matches = f.result()
            results[idx] = z_score
            best_matches[idx] = doc_best_matches
    return results, best_matches


def compute_zscore_for_doc(doc_idx, doc_amrs, amr_bank, threshold, lmbd, is_human):
    """
    Compare each AMR in the document to all templates in amr_bank (s2match).
    Count how many >= threshold => compute z-score:
         z = (count_green - lmbd*N) / sqrt(N*lmbd*(1-lmbd))
    Returns (doc_idx, z_score, best_matches).
    """
    # All AMRs in the doc are watermarked sentences (prompt is not included in the saved text)
    relevant_amrs = doc_amrs
    count_green = 0
    best_matches = []
    
    for cand_final in relevant_amrs:
        if not cand_final.strip():
            continue
        best_score = 0.0
        best_match = None

        for bank_amr in amr_bank:
            score = compute_s2match_score(cand_final, bank_amr)
            if score > best_score:
                best_score = score
                best_match = bank_amr

        logger.debug(f"{'HUMAN' if is_human else 'MACHINE'} best_score: {best_score:.3f}")
        if best_score >= threshold:
            count_green += 1
        best_matches.append((best_score, best_match))

    N = len(relevant_amrs)
    logger.debug(f"Counted {count_green} green sentences out of {N} total")

    numerator = count_green - (lmbd * N)
    denominator = np.sqrt(N * lmbd * (1 - lmbd))
    if denominator < 1e-9:
        z_score = float('nan')
    else:
        z_score = numerator / denominator

    return (doc_idx, z_score, best_matches)


if __name__ == '__main__':
    main()
