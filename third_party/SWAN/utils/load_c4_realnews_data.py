# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
load_c4_realnews_data.py

Download the REALNEWS subset of the C4 corpus for evaluation.

Saves two splits:
  - data/c4-val:   first k validation examples (used as prompts for watermark generation)
  - data/c4-human: next 5000 train examples (human reference text for detection)

We use the validation split for generation prompts to match the SemStamp baseline
evaluation setup, and the train split for human reference text.

Uses streaming mode to avoid downloading the full C4 dataset.

Usage:
    python utils/load_c4_realnews_data.py --k 250
"""

from datasets import load_dataset, Dataset
import argparse


def stream_n_examples(split, n, skip=0):
    """Stream n examples from a dataset split, optionally skipping the first `skip`."""
    ds = load_dataset("allenai/c4", "realnewslike", split=split, streaming=True)
    examples = []
    for i, example in enumerate(ds):
        if i < skip:
            continue
        if len(examples) >= n:
            break
        examples.append(example)
    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download REALNEWS subset of C4.")
    parser.add_argument("--k", type=int, default=250)
    args = parser.parse_args()

    print(f"Streaming {args.k} validation examples...")
    val_dataset = stream_n_examples("validation", n=args.k)
    val_dataset.save_to_disk("data/c4-val")
    print(f"Saved {args.k} validation examples to data/c4-val")

    print(f"Streaming 5000 train examples (skipping first {args.k})...")
    human_dataset = stream_n_examples("train", n=5000, skip=args.k)
    human_dataset.save_to_disk("data/c4-human")
    print(f"Saved 5000 human reference examples to data/c4-human")
