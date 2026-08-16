# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
paraphrase_gen.py

Generate paraphrased versions of watermarked text for robustness evaluation (§4.1).

Paraphrases each sentence in a watermarked paragraph using Claude in a zero-shot
setting, preserving context from preceding sentences (Appendix A.2).

Usage:
    python evaluation/paraphrase_gen.py \
        --dataset_path output/watermarked \
        --output_dir output/watermarked_paraphrased \
        --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --region us-east-1
"""

import argparse
import os
import sys
import json
import logging
import time

from datasets import load_from_disk, Dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk

nltk.download('punkt_tab', quiet=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.bedrock_utils import BedrockManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def paraphrase_paragraph(sents, bm):
    """Paraphrase each sentence in a paragraph using Claude, with preceding context.

    Implements the zero-shot Claude paraphrase attack from Appendix A.2:
        Previous context: {preceding sentences}
        Current sentence to paraphrase: {sent}
        Rewrite the sentence above while preserving its meaning.
    """
    para_sents = []
    for i, sent in enumerate(sents):
        context = sents[:i]
        prompt = (
            f"Previous context: {' '.join(context)}\n"
            f"Current sentence to paraphrase: {sent}\n"
            "Rewrite the sentence above while preserving its meaning.\n"
            "Do not provide any explanation or extra commentary.\n"
            "Return only the new sentence."
        )

        try:
            response = bm.generate(
                user_text=prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            para_sents.append(response.strip())
        except Exception as e:
            logger.warning(f"Paraphrase failed for sentence {i}: {e}")
            para_sents.append(sent)  # fallback to original on failure

    return para_sents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paraphrased versions of watermarked text (§4.1)."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to HF dataset with a 'text' column (watermarked paragraphs).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the paraphrased dataset.",
    )
    parser.add_argument(
        "--model_id", type=str,
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Bedrock model ID for paraphrasing.",
    )
    parser.add_argument(
        "--region", type=str, default="us-east-1",
        help="AWS region for Bedrock.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of samples to paraphrase (default: all).",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="Seconds to sleep between API calls to avoid throttling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must have a 'text' column.")

    texts = dataset["text"]
    if args.max_samples is not None:
        texts = texts[: args.max_samples]

    logger.info(f"Paraphrasing {len(texts)} paragraphs with {args.model_id}")

    bm = BedrockManager(region=args.region, model_id=args.model_id)

    original_texts = []
    paraphrased_texts = []
    failed = 0

    for i, text in enumerate(tqdm(texts, desc="Paraphrasing")):
        sents = sent_tokenize(text)
        para_sents = paraphrase_paragraph(sents, bm)

        original_texts.append(text)
        paraphrased_texts.append(" ".join(para_sents))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Save as HF dataset with both original and paraphrased text
    new_dataset = Dataset.from_dict({
        "text": original_texts,
        "para_text": paraphrased_texts,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    new_dataset.save_to_disk(args.output_dir)

    logger.info(f"Saved {len(paraphrased_texts)} paraphrased paragraphs to '{args.output_dir}'")
    logger.info(f"Columns: 'text' (original), 'para_text' (paraphrased)")


if __name__ == "__main__":
    main()
