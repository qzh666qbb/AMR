# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
display_top_templates.py

Utility to inspect the most frequent template AMRs in the corpus.

Reads the template-groups JSON produced by analyze_amr_distribution.py,
picks the top-N templates by frequency, and writes them (with example
raw AMRs) to a human-readable text file. Useful for understanding what
kinds of semantic structures dominate the AMR bank.

Usage:
    python amr_bank/display_top_templates.py \\
        --input_json amr_bank/artifacts/normalized_template_groups.json \\
        --top_n 10
"""

import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Display top templates from template_groups.json and save examples to a text file.")
    parser.add_argument('--input_json', type=str, default='amr_bank/artifacts/normalized_template_groups.json', help='Path to template_groups.json')
    parser.add_argument('--output_txt', type=str, default='amr_bank/artifacts/top_templates.txt', help='Path to save the output text file')
    parser.add_argument('--top_n', type=int, default=3, help='Number of top templates to display')
    parser.add_argument('--max_examples', type=int, default=5, help='Number of examples per template to show')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data is a list of { "normalized_template_amr", "count", "examples": [...] }
    # Assuming it's already sorted by count in descending order.
    # If not sorted, we can sort it:
    data.sort(key=lambda x: x["count"], reverse=True)

    top_n = min(args.top_n, len(data))

    with open(args.output_txt, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Top {top_n} Template AMRs:\n\n")
        for i in range(top_n):
            template = data[i]
            template_amr = template["normalized_template_amr"]
            count = template["count"]
            examples = template["examples"]
            out_f.write(f"Rank {i+1}:\n")
            out_f.write(f"Count: {count}\n")
            out_f.write("Template AMR:\n")
            out_f.write(template_amr + "\n\n")
            out_f.write("Examples:\n")
            max_ex = min(args.max_examples, len(examples))
            for j in range(max_ex):
                out_f.write(f"{j+1}. {examples[j]}\n\n")
            out_f.write("=" * 40 + "\n\n")

    print(f"Top {top_n} templates written to {args.output_txt}")

if __name__ == "__main__":
    main()
