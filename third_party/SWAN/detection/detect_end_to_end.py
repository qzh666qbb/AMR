# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
detect_end_to_end.py

End-to-end watermark detection (§3.3): parses text to AMR and detects in one pass.

For each document, sentence-tokenizes, parses each sentence to AMR, compares
against the AMR bank via S2match, counts "green" sentences, and computes a
z-score. Evaluates AUROC, TPR@1%FPR, TPR@5%FPR.

Usage:
    python detection/detect_end_to_end.py output/watermarked \\
        --human_text data/c4-human \\
        --amr_bank_path amr_bank/banks/amr_bank_50.json
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datasets import load_from_disk
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detection.detection_utils import detect_amr_sentences, evaluate_z_scores_amr
from utils.amr_utils import load_amr_bank

def parse_args():
    parser = argparse.ArgumentParser(description="AMR-based watermark detection with doc-level parallelism.")
    parser.add_argument('dataset_path', help='HF dataset containing text column for watermarked (machine) text')
    parser.add_argument('--human_text', required=True,
                        help='HF dataset containing text column for human text')
    parser.add_argument('--amr_bank_path', type=str,
                        default="amr_bank/banks/amr_bank_50.json",
                        help='Path to your curated AMR bank')
    parser.add_argument('--threshold', type=float, default=0.65, 
                        help='s2match threshold to mark a sentence as green')
    parser.add_argument('--lmbd', type=float, default=0.25, 
                        help='expected fraction of green sentences if random (the λ)')
    parser.add_argument('--output_dir', type=str, default="detection_results", 
                        help='where to save detection outputs')
    parser.add_argument('--num_workers', type=int, default=96, 
                        help='How many processes to run in parallel')
    return parser.parse_args()

def process_doc(index, text, amr_bank, threshold, lmbd, is_human=False):
    """
    Process a single document:
      1) tokenize sentences
      2) if human, truncate to 5
      3) call detect_amr_sentences to parse each sentence and compute the
         paragraph-level z-score
      4) return (index, z_score)
    """
    from nltk.tokenize import sent_tokenize  # re-import inside process if needed
    sents = text if isinstance(text, list) else sent_tokenize(text)

    if is_human:
        # Truncate to 5 sentences total
        sents = sents[:5]
    
    z_score = detect_amr_sentences(sents, amr_bank, threshold, lmbd, is_human=is_human)
    return (index, z_score)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1) Load your curated AMR bank once
    amr_bank = load_amr_bank(args.amr_bank_path)
    print(f"Loaded {len(amr_bank)} AMR templates from '{args.amr_bank_path}'")

    # 2) Load watermarked text dataset
    dataset = load_from_disk(args.dataset_path)
    if 'text' not in dataset.column_names:
        raise ValueError("The dataset must have a 'text' column.")
    watermarked_texts = dataset['text']
    print(f"Loaded {len(watermarked_texts)} watermarked texts from '{args.dataset_path}'")

    # 3) Load human text dataset
    human_dataset = load_from_disk(args.human_text)
    human_texts = human_dataset['text'][:len(watermarked_texts)]  # match lengths

    # 4) Detect: compute z-scores for watermarked text in parallel
    print("[Detection] Computing z-scores for machine-generated text with doc-level parallelism ...")
    z_scores = [None]*len(watermarked_texts)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i, text in enumerate(watermarked_texts):
            futures.append(executor.submit(process_doc, i, text, amr_bank, 
                                           args.threshold, args.lmbd, 
                                           is_human=False))
        for f in as_completed(futures):
            idx, z_score = f.result()
            z_scores[idx] = z_score

    # 5) Detect: compute z-scores for human text in parallel
    print("[Detection] Computing z-scores for human text in parallel ...")
    human_scores = [None]*len(human_texts)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i, text in enumerate(human_texts):
            futures.append(executor.submit(process_doc, i, text, amr_bank, 
                                           args.threshold, args.lmbd, 
                                           is_human=True))
        for f in as_completed(futures):
            idx, z_score = f.result()
            human_scores[idx] = z_score

    # 6) Save z-scores
    z_score_path = os.path.join(args.output_dir, "machine_z_scores.npy")
    human_score_path = os.path.join(args.output_dir, "human_z_scores.npy")

    np.save(z_score_path, z_scores)
    np.save(human_score_path, human_scores)

    # 7) Evaluate detection performance
    print("Evaluating z-scores for AMR-based watermark detection...")
    auroc, tpr_at_fpr1, tpr_at_fpr5 = evaluate_z_scores_amr(z_scores, human_scores, args.output_dir)
    
    # 8) Save metrics
    results_path = os.path.join(args.output_dir, "results.csv")
    columns = ["auroc", "tpr_at_fpr1", "tpr_at_fpr5"]
    metrics = [f"{auroc:.3f}", f"{tpr_at_fpr1:.3f}", f"{tpr_at_fpr5:.3f}"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)

    print(f"\n[Done] Results saved to {results_path}")

if __name__ == '__main__':
    main()
