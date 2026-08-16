# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
text_quality_eval.py

LLM-as-a-judge text quality evaluation (§4.5, Appendix A.3).

Rates watermarked paragraphs on three dimensions:
  - Coherence (logical organization and clarity)
  - Fluency (grammatical correctness and readability)
  - Diversity (variety of vocabulary and sentence structures)

Each dimension is scored on a [0, 5] scale. Uses Claude 3.7 Sonnet
in a zero-shot setting as the judge.

Usage:
    python text_quality_eval.py \
        --dataset_path /path/to/hf_dataset \
        --output_path results/quality_scores.json \
        --model_id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --region us-east-1
"""

import argparse
import json
import os
import sys
import time
import logging

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.bedrock_utils import BedrockManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EVAL_PROMPT_TEMPLATE = """You are an expert writing quality evaluator.

You will assess a GENERATED PARAGRAPH using the following criteria. For each, assign a score from 1 to 5 (decimals allowed), using the descriptions below.

1. **Coherence**: Measures how logically and clearly the ideas are organized and connected.
   - 1: Incoherent; sentences are unrelated or confusing.
   - 2: Poor transitions or unclear relationships between ideas.
   - 3: Basic logical flow, but some awkward connections.
   - 4: Mostly logical and clear, with minor lapses.
   - 5: Highly logical and seamless flow of ideas.

2. **Fluency**: Assesses the grammatical correctness and naturalness of the language.
   - 1: Grammatically broken or unreadable.
   - 2: Understandable but awkward or error-prone.
   - 3: Generally readable, some minor grammatical errors or odd phrasing.
   - 4: Well-written with only occasional issues.
   - 5: Grammatically correct and naturally flowing throughout.

3. **Diversity**: Use of varied vocabulary and sentence structure, avoiding repetition.
   - 1: Extremely repetitive or formulaic.
   - 2: Some repetition with occasional variation.
   - 3: Moderate variety; not monotonous.
   - 4: Good diversity in language and structure.
   - 5: Highly expressive and varied without redundancy.

**Scoring Instructions**:
- Return a score for each of the three dimensions above.
- You may use decimal values (e.g., 2.5, 4.7).

**Output Format**:
Respond with a **valid JSON object only** in this exact format:

{{
    "coherence_score": float,
    "fluency_score": float,
    "diversity_score": float
}}

**GENERATED PARAGRAPH**:
{paragraph}"""


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-judge text quality evaluation.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HF dataset with a 'text' column containing paragraphs to evaluate.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="quality_scores.json",
        help="Path to save the evaluation results JSON.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Bedrock model ID for the judge LLM.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls to avoid throttling.",
    )
    return parser.parse_args()


def parse_judge_response(response_text: str) -> dict:
    """Parse the JSON response from the judge LLM. Handles markdown fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        scores = json.loads(text)
        return {
            "coherence_score": float(scores["coherence_score"]),
            "fluency_score": float(scores["fluency_score"]),
            "diversity_score": float(scores["diversity_score"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {e}\nRaw response: {response_text}")
        return None


def main():
    args = parse_args()

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must have a 'text' column.")

    texts = dataset["text"]
    if args.max_samples is not None:
        texts = texts[: args.max_samples]

    logger.info(f"Evaluating {len(texts)} paragraphs with {args.model_id}")

    # Initialize Bedrock
    bm = BedrockManager(region=args.region, model_id=args.model_id)

    results = []
    failed = 0

    for i, paragraph in enumerate(tqdm(texts, desc="Evaluating text quality")):
        prompt = EVAL_PROMPT_TEMPLATE.format(paragraph=paragraph)

        try:
            response = bm.generate(
                user_text=prompt,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as e:
            logger.error(f"[{i}] API call failed: {e}")
            failed += 1
            results.append({"index": i, "paragraph": paragraph, "scores": None, "error": str(e)})
            continue

        scores = parse_judge_response(response)
        if scores is None:
            failed += 1
            results.append({"index": i, "paragraph": paragraph, "scores": None, "raw_response": response})
        else:
            results.append({"index": i, "paragraph": paragraph, "scores": scores})

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Compute aggregate statistics
    valid_results = [r for r in results if r.get("scores") is not None]
    if valid_results:
        coherence = [r["scores"]["coherence_score"] for r in valid_results]
        fluency = [r["scores"]["fluency_score"] for r in valid_results]
        diversity = [r["scores"]["diversity_score"] for r in valid_results]

        summary = {
            "num_evaluated": len(texts),
            "num_valid": len(valid_results),
            "num_failed": failed,
            "coherence": {
                "mean": float(np.mean(coherence)),
                "std": float(np.std(coherence)),
                "min": float(np.min(coherence)),
                "max": float(np.max(coherence)),
            },
            "fluency": {
                "mean": float(np.mean(fluency)),
                "std": float(np.std(fluency)),
                "min": float(np.min(fluency)),
                "max": float(np.max(fluency)),
            },
            "diversity": {
                "mean": float(np.mean(diversity)),
                "std": float(np.std(diversity)),
                "min": float(np.min(diversity)),
                "max": float(np.max(diversity)),
            },
        }
    else:
        summary = {"num_evaluated": len(texts), "num_valid": 0, "num_failed": failed}

    output = {"summary": summary, "results": results}

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"TEXT QUALITY EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Evaluated: {len(texts)} | Valid: {len(valid_results)} | Failed: {failed}")
    if valid_results:
        logger.info(f"Coherence:  {summary['coherence']['mean']:.2f} ± {summary['coherence']['std']:.2f} ({summary['coherence']['min']:.1f}–{summary['coherence']['max']:.1f})")
        logger.info(f"Fluency:    {summary['fluency']['mean']:.2f} ± {summary['fluency']['std']:.2f} ({summary['fluency']['min']:.1f}–{summary['fluency']['max']:.1f})")
        logger.info(f"Diversity:  {summary['diversity']['mean']:.2f} ± {summary['diversity']['std']:.2f} ({summary['diversity']['min']:.1f}–{summary['diversity']['max']:.1f})")
    logger.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
