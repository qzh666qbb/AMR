# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
build_curated_amr_bank.py

AMR Bank Creation — Step 2 of 2 (§3.1)

Takes the template distribution produced by analyze_amr_distribution.py and
selects templates that (a) occur within a frequency range [min_freq, max_freq]
and (b) contain at least min_nodes concept nodes. The result is the curated
AMR bank B used as the secret key for watermark injection and detection.

Usage:
    python amr_bank/build_curated_amr_bank.py \\
        --distribution_json amr_bank/artifacts/normalized_template_amr_distribution.json \\
        --bank_size 50 --output_dir amr_bank/banks
"""

import json
import argparse
import os
import penman

def parse_args():
    parser = argparse.ArgumentParser(description="Build a curated AMR bank from distribution.json using frequency-based selection and optional fallback.")
    parser.add_argument('--distribution_json', type=str, default='amr_bank/artifacts/normalized_template_amr_distribution.json', help='Path to distribution JSON')
    parser.add_argument('--output_dir', type=str, default='amr_bank/banks', help='Directory to save the curated AMR bank')
    parser.add_argument('--output_name', type=str, default='curated_amr_bank.json', help='Name of the output JSON file')
    parser.add_argument('--min_freq', type=int, default=3, help='Minimum frequency threshold for inclusion')
    parser.add_argument('--max_freq', type=int, default=20, help='Maximum frequency threshold for inclusion')
    parser.add_argument('--bank_size', type=int, default=50, help='Desired number of templates in the curated bank')
    parser.add_argument('--fallback', action='store_true', help='Enable fallback logic to relax min_freq if not enough templates found')
    parser.add_argument('--min_nodes', type=int, default=3, help='Minimum number of nodes/concepts required in the AMR')
    return parser.parse_args()

def filter_templates(dist, min_freq, max_freq):
    return [entry for entry in dist if min_freq <= entry['count'] <= max_freq]

def count_nodes(amr_str):
    # Decode AMR and count how many nodes it has
    # Nodes are given by the number of :instance triples
    # Penman graph: each triple with :instance is a concept node
    g = penman.decode(amr_str)
    node_count = sum(1 for t in g.triples if t[1] == ':instance')
    return node_count

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    # Load distribution
    with open(args.distribution_json, 'r', encoding='utf-8') as f:
        dist = json.load(f)

    dist.sort(key=lambda x: x["count"], reverse=True)

    # Filter by frequency
    filtered = filter_templates(dist, args.min_freq, args.max_freq)

    # Filter by minimum number of nodes
    final_filtered = []
    for entry in filtered:
        amr_str = entry["normalized_template_amr"]
        nodes = count_nodes(amr_str)
        if nodes >= args.min_nodes:
            final_filtered.append(entry)

    # If we have fewer than bank_size templates and fallback is enabled, lower min_freq
    if len(final_filtered) < args.bank_size and args.fallback:
        print("Not enough templates found with current frequency range and node count. Attempting fallback strategy...")
        freq_to_try = args.min_freq - 1
        while freq_to_try > 0 and len(final_filtered) < args.bank_size:
            print(f"Lowering min_freq to {freq_to_try}...")
            # Re-filter by frequency with the lowered min_freq
            re_filtered = filter_templates(dist, freq_to_try, args.max_freq)
            # Also filter by node count again
            re_final = []
            for entry in re_filtered:
                amr_str = entry["normalized_template_amr"]
                nodes = count_nodes(amr_str)
                if nodes >= args.min_nodes:
                    re_final.append(entry)

            final_filtered = re_final
            freq_to_try -= 1

    # Truncate if needed
    if len(final_filtered) > args.bank_size:
        final_filtered = final_filtered[:args.bank_size]
    else:
        if len(final_filtered) < args.bank_size:
            print(f"Warning: Only {len(final_filtered)} templates found after all filtering.")

    # Save the curated bank
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(final_filtered, out_f, ensure_ascii=False, indent=2)

    print(f"Curated AMR bank saved to {output_path}")
    print(f"Criteria: min_freq={args.min_freq}, max_freq={args.max_freq}, bank_size={args.bank_size}, fallback={args.fallback}, min_nodes={args.min_nodes}")
    print(f"Templates selected: {len(final_filtered)}")

if __name__ == "__main__":
    main()
