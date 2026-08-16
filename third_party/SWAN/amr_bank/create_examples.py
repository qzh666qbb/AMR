# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
create_examples.py

Generate few-shot examples for the watermark injection prompt (Appendix A.1).

Reads raw sentence + AMR pairs from the MASSIVE-AMR corpus, applies the
normalize → template → normalize pipeline, then selects a balanced mix of
short, medium, and long sentences. The output JSON is used by
injection_amr_utils.build_user_prompt() to give the LLM concrete examples
of how template AMRs map to English sentences.

Usage:
    python amr_bank/create_examples.py \\
        --input_jsonl amr_bank/data/massive_amr.jsonl \\
        --output_json amr_bank/artifacts/normalized_amr_examples.json
"""

import json
import argparse
import os
import sys
import penman

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.amr_utils import generate_template_amr, normalize_amr_variables, remove_amr_comments

def parse_args():
    parser = argparse.ArgumentParser(description="Generate sentence, AMR, and template AMR tuples from curated AMR bank with normalization.")
    parser.add_argument('--input_jsonl', type=str, default='amr_bank/data/massive_amr.jsonl', help='Path to the raw AMR data jsonl file')
    parser.add_argument('--output_json', type=str, default='amr_bank/artifacts/normalized_amr_examples.json', help='Path to save the output JSON with examples')
    parser.add_argument('--num_candidates', type=int, default=50, help='Number of candidates to load before filtering')
    parser.add_argument('--short_max', type=int, default=5, help='Max number of words considered as short sentence')
    parser.add_argument('--medium_max', type=int, default=8, help='Max number of words considered as medium sentence')
    parser.add_argument('--num_short', type=int, default=2, help='Number of short examples to pick')
    parser.add_argument('--num_medium', type=int, default=2, help='Number of medium examples to pick')
    parser.add_argument('--num_long', type=int, default=1, help='Number of long examples to pick')
    return parser.parse_args()

def main():
    args = parse_args()

    candidates = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.num_candidates:
                break
            entry = json.loads(line.strip())
            sentence = entry.get('utt', '').strip()
            raw_amr = entry.get('raw_amr', '').strip()

            # Basic validation
            if not sentence or not raw_amr:
                continue

            # Remove comments
            clean_amr = remove_amr_comments(raw_amr)

            # Check if it starts with '(' to be a valid AMR
            if not clean_amr.strip().startswith('('):
                continue

            # Try to parse the AMR
            try:
                g = penman.decode(clean_amr)
            except Exception:
                continue

            # Apply normalization pipeline:
            # 1. normalize
            norm_amr = normalize_amr_variables(clean_amr)
            # 2. template
            templ_amr = generate_template_amr(norm_amr)
            # 3. normalize again
            final_amr = normalize_amr_variables(templ_amr)

            candidates.append({
                "sentence": sentence,
                "raw_amr": raw_amr,
                "template_amr": templ_amr.strip(),
                "normalized_template_amr": final_amr.strip()
            })

    # Categorize candidates by length
    short_examples = []
    medium_examples = []
    long_examples = []

    for ex in candidates:
        word_count = len(ex["sentence"].split())
        if word_count <= args.short_max:
            short_examples.append(ex)
        elif word_count <= args.medium_max:
            medium_examples.append(ex)
        else:
            long_examples.append(ex)

    # Select examples according to the desired distribution
    final_examples = []

    # Short examples
    final_examples.extend(short_examples[:args.num_short])

    # Medium examples
    final_examples.extend(medium_examples[:args.num_medium])

    # Long examples
    final_examples.extend(long_examples[:args.num_long])

    # If not enough examples, fill with whatever is left
    needed = (args.num_short + args.num_medium + args.num_long) - len(final_examples)
    if needed > 0:
        remaining_candidates = [c for c in candidates if c not in final_examples]
        final_examples.extend(remaining_candidates[:needed])

    # Save final chosen examples
    with open(args.output_json, 'w', encoding='utf-8') as out_f:
        json.dump(final_examples, out_f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_examples)} examples to {args.output_json}")
    print("Final Distribution:")
    print(f"Short examples picked: {len([e for e in final_examples if len(e['sentence'].split()) <= args.short_max])}")
    print(f"Medium examples picked: {len([e for e in final_examples if args.short_max < len(e['sentence'].split()) <= args.medium_max])}")
    print(f"Long examples picked: {len([e for e in final_examples if len(e['sentence'].split()) > args.medium_max])}")

if __name__ == '__main__':
    main()
