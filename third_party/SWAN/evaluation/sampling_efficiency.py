# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
sampling_efficiency.py

Compute the average number of rejection sampling attempts per sentence (§4.3).
Reads sampled_data.json files produced by watermark_generation.py.

Usage:
    python evaluation/sampling_efficiency.py \
        --sampled_data_paths output/watermarked/sampled_data.json

    # Multiple shards (if generation was split across runs):
    python evaluation/sampling_efficiency.py \
        --sampled_data_paths output/shard_0/sampled_data.json output/shard_1/sampled_data.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Compute sampling efficiency statistics.")
    parser.add_argument("--sampled_data_paths", nargs="+", required=True,
                        help="Paths to sampled_data.json files from each generation shard")
    args = parser.parse_args()

    all_samples = []
    for path in args.sampled_data_paths:
        with open(path) as f:
            data = json.load(f)
        all_samples.extend(data["sentences_sampled"])
        print(f"  Loaded {len(data['sentences_sampled'])} docs from {path}")

    flat = [x for sublist in all_samples for x in sublist]

    if not flat:
        print("No sampling data found.")
        return

    average = sum(flat) / len(flat)
    print(f"\nTotal sentences: {len(flat)}")
    print(f"Total documents: {len(all_samples)}")
    print(f"Average attempts per sentence: {average:.2f}")

    # Efficiency distribution: group trials into bins of 5
    bin_size = 5
    max_trials = max(flat)
    bins = list(range(1, max_trials + bin_size, bin_size))

    print(f"\n--- Efficiency Distribution ---")
    cumulative = 0
    for start in bins:
        end = start + bin_size - 1
        count = sum(1 for x in flat if start <= x <= end)
        cumulative += count
        pct = 100.0 * cumulative / len(flat)
        print(f"Trials {start:>3}-{end:>3}: {count:>5} sentences  (cumulative: {pct:.1f}%)")


if __name__ == "__main__":
    main()
