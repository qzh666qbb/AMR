# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
analyze_amr_distribution.py

AMR Bank Creation — Step 1 of 2 (§3.1)

Reads the MASSIVE-AMR corpus and builds a frequency distribution of template
AMRs. For each raw AMR graph the pipeline is:
    normalize variables → generate template → normalize again
Identical templates are counted, producing two outputs:
  - A distribution JSON: list of {normalized_template_amr, count}
  - A template-groups JSON: same, plus the raw AMR examples per template

Usage:
    python amr_bank/analyze_amr_distribution.py \\
        --input_jsonl amr_bank/data/massive_amr.jsonl \\
        --output_dir amr_bank/artifacts
"""

import json
import argparse
import penman
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.amr_utils import remove_amr_comments, generate_template_amr, normalize_amr_variables

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AMR distributions and produce frequency counts.")
    parser.add_argument('--input_jsonl', type=str, default='amr_bank/data/massive_amr.jsonl', help='Path to input JSONL file with raw_amr field')
    parser.add_argument('--output_json', type=str, default='normalized_template_amr_distribution.json', help='Path to save the AMR distribution JSON')
    parser.add_argument('--output_dir', type=str, default='amr_bank/artifacts', help='Directory to save output files')
    parser.add_argument('--template_groups_json', type=str, default='normalized_template_groups.json', help='Path to save template to raw AMR groups')
    return parser.parse_args()

def is_valid_amr(amr_str):
    return amr_str.strip().startswith('(')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_json)
    groups_path = os.path.join(args.output_dir, args.template_groups_json)

    # Load and parse AMRs
    template_counter = Counter()
    template_to_raw = defaultdict(list)  # map from final_template_amr -> list of raw_amrs

    total_amrs = 0
    valid_amrs = 0

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            raw_amr = entry.get("raw_amr", "").strip()
            if not raw_amr:
                continue

            clean_amr = remove_amr_comments(raw_amr)

            if not is_valid_amr(clean_amr):
                continue

            total_amrs += 1
            try:
                # Decode to ensure it's valid AMR
                g = penman.decode(clean_amr)

                # Now apply the pipeline:
                # 1. Normalize AMR variables
                norm_amr = normalize_amr_variables(clean_amr)
                # 2. Generate template AMR
                templ_amr = generate_template_amr(norm_amr)
                # 3. Normalize again after templating
                final_amr = normalize_amr_variables(templ_amr)

                template_counter[final_amr] += 1
                template_to_raw[final_amr].append(raw_amr)
                valid_amrs += 1
            except Exception:
                continue

    # Convert counter to a list of {template_amr, count}
    distribution = [{"normalized_template_amr": amr, "count": cnt} for amr, cnt in template_counter.items()]

    # Sort by count descending
    distribution.sort(key=lambda x: x["count"], reverse=True)

    # Save distribution
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(distribution, out_f, ensure_ascii=False, indent=2)

    # Build template groups
    template_groups = []
    for item in distribution:
        t_amr = item["normalized_template_amr"]
        count = item["count"]
        examples = template_to_raw[t_amr]
        template_groups.append({
            "normalized_template_amr": t_amr,
            "count": count,
            "examples": examples
        })

    with open(groups_path, 'w', encoding='utf-8') as out_g:
        json.dump(template_groups, out_g, ensure_ascii=False, indent=2)

    print(f"Total AMRs processed: {total_amrs}, Valid: {valid_amrs}")
    print(f"Saved distribution to {output_path}")
    print(f"Saved template groups to {groups_path}")
    print(f"Total unique template AMRs: {len(distribution)}")

if __name__ == "__main__":
    main()
