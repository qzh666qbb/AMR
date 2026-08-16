# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
watermark_generation.py

Main watermark injection script implementing Algorithm 1 (§3.2).

For each example in the dataset, extracts the first sentence as a prompt,
then generates N watermarked sentences using AMR-guided rejection sampling.
Saves the generated text as a HuggingFace dataset and sampling metadata as JSON.

Usage:
    python injection/watermark_generation.py \\
        --data data/c4-val \\
        --amr_bank_path amr_bank/banks/amr_bank_50.json \\
        --amr_examples_path amr_bank/artifacts/normalized_amr_examples.json \\
        --output_dir output/watermarked
"""

import argparse
import os
import sys
import json
import random
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_from_disk, Dataset
from utils.text_utils import extract_prompt_from_text
from utils.amr_utils import load_amr_bank
from injection_amr_utils import amr_instruction_completion, non_watermark_generation
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction-based approach for sentence-by-sentence generation with AMR templates.")
    parser.add_argument('--data', required=True, help='Path to HF dataset with a "text" column (use utils/load_c4_realnews_data.py to download)')
    parser.add_argument('--amr_bank_path', type=str, default='amr_bank/banks/amr_bank_50.json', help='Path to curated AMR bank JSON')
    parser.add_argument('--model_id', type=str, default='us.anthropic.claude-sonnet-4-5-20250929-v1:0', help='Bedrock model ID or DeepSeek API model ID')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--len_prompt', type=int, default=32)
    parser.add_argument('--num_sentences', type=int, default=5, help='Number of sentences to generate per example')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_trials', type=int, default=50)
    parser.add_argument('--amr_examples_path', type=str, required=True, help='Path to the AMR examples file')
    parser.add_argument('--output_dir', type=str, default="output/watermarked", help='Directory to save generated output')
    parser.add_argument('--hf_model', type=bool, default=False)
    parser.add_argument('--no_watermark', action='store_true', help="Disable watermarking (generate without AMR guidance)")
    parser.add_argument('--max_samples', type=int, default=None, help='Limit examples for smoke/pilot runs')
    parser.add_argument('--seed', type=int, default=20260721, help='Python RNG seed for template selection')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    print("Using data from:", args.data)
    print("Using model ID:", args.model_id)
    print("The output will be saved to:", args.output_dir)

    dataset = load_from_disk(args.data)
    texts = dataset['text']
    if args.max_samples is not None:
        texts = texts[:args.max_samples]

    amr_bank = load_amr_bank(args.amr_bank_path)
    no_watermark = args.no_watermark
    hf_model = args.hf_model
    model = None
    tokenizer = None
    if hf_model:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "generation_checkpoint.json")
    generated_texts = []
    generated_texts_sentences_sampled = []
    generated_texts_amrs = []
    generated_texts_scores = []
    generated_texts_accepted = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        generated_texts = checkpoint["texts"]
        generated_texts_sentences_sampled = checkpoint["sentences_sampled"]
        generated_texts_amrs = checkpoint["chosen_amrs"]
        generated_texts_scores = checkpoint["s2match_scores"]
        generated_texts_accepted = checkpoint["accepted"]
        if checkpoint.get("random_state"):
            random.setstate(ast.literal_eval(checkpoint["random_state"]))
        print(f"Resuming from checkpoint after {len(generated_texts)} examples")

    start_index = len(generated_texts)
    remaining_texts = texts[start_index:]

    for text in tqdm(remaining_texts, desc="Processing examples", initial=start_index, total=len(texts)):
        prompt = extract_prompt_from_text(text, args.len_prompt)
        current_sentences = []
        current_sentences_sampled = []
        current_sentences_amrs = []
        current_scores = []
        current_accepted = []
        current_prompt = prompt

        for s_idx in range(args.num_sentences):
            if no_watermark == False:
                if(hf_model == False):
                    gen_text, total_attempts, chosen_template, match_score, accepted = amr_instruction_completion(
                        context=current_prompt,
                        amr_bank=amr_bank,
                        model_id=args.model_id,
                        region=args.region,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_trials=args.max_trials,
                        amr_examples_path=args.amr_examples_path
                    )
                else:
                    gen_text, total_attempts, chosen_template, match_score, accepted = amr_instruction_completion(
                        context=current_prompt,
                        amr_bank=amr_bank,
                        model_id=args.model_id,
                        region=args.region,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_trials=args.max_trials,
                        amr_examples_path=args.amr_examples_path,
                        model = model,
                        tokenizer = tokenizer,
                        hf_model = hf_model
                    )
            else:
                gen_text = non_watermark_generation(
                        context=current_prompt,
                        model_id=args.model_id,
                        region=args.region,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        model = model,
                        tokenizer = tokenizer,
                        hf_model = hf_model,
                    )
                total_attempts = 0
                chosen_template = ""
                match_score = None
                accepted = None
            if not gen_text.strip():
                break
            # Clean DeepSeek think-tag artifacts if present (only affects R1-Distill models)
            if "No text found after </think> tag" in gen_text:
                gen_text = gen_text.replace("No text found after </think> tag", "").strip()
            if not gen_text:
                break
            current_sentences.append(gen_text)
            current_sentences_sampled.append(total_attempts)
            current_sentences_amrs.append((gen_text.strip(), chosen_template))
            current_scores.append(match_score)
            current_accepted.append(accepted)
            current_prompt = prompt + " " + " ".join(current_sentences)

        final_text = " ".join(current_sentences)
        generated_texts.append(final_text)
        generated_texts_sentences_sampled.append(current_sentences_sampled)
        generated_texts_amrs.append(current_sentences_amrs)
        generated_texts_scores.append(current_scores)
        generated_texts_accepted.append(current_accepted)
        checkpoint = {
            "texts": generated_texts,
            "sentences_sampled": generated_texts_sentences_sampled,
            "chosen_amrs": generated_texts_amrs,
            "s2match_scores": generated_texts_scores,
            "accepted": generated_texts_accepted,
            "random_state": repr(random.getstate()),
            "complete": False,
        }
        tmp_checkpoint = checkpoint_path + ".tmp"
        with open(tmp_checkpoint, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        os.replace(tmp_checkpoint, checkpoint_path)

    new_dataset = Dataset.from_dict({'text': generated_texts})

    new_dataset.save_to_disk(args.output_dir)
    print(f"Generated dataset saved to {args.output_dir}")

    # Save sampling metadata
    sampling_data = {
        "sentences_sampled": generated_texts_sentences_sampled,
        "chosen_amrs": generated_texts_amrs,
        "s2match_scores": generated_texts_scores,
        "accepted": generated_texts_accepted,
    }
    sampled_output_file_path = os.path.join(args.output_dir, "sampled_data.json")
    with open(sampled_output_file_path, "w") as f:
        json.dump(sampling_data, f, indent = 2)
    checkpoint["complete"] = True
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
