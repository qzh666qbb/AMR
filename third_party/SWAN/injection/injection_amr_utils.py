# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
injection_amr_utils.py

Core watermark injection logic implementing Algorithm 1 from §3.2.

Provides:
  - amr_instruction_completion():  Rejection sampling with AMR-guided generation
  - non_watermark_generation():    Baseline generation without watermark guidance
  - build_user_prompt():           Construct the LLM prompt with AMR template + examples
"""

import os
import sys
import json
import re
import random
import logging

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.amr_utils import parse_amr, generate_template_amr, normalize_amr_variables, decide_s2match_match
from utils.bedrock_utils import BedrockManager

logger = logging.getLogger(__name__)


def load_amr_examples(path: str, num_examples: int = 3):
    """
    Load a few AMR examples from a JSON file.
    Each entry in the file should have: "sentence", "raw_amr", "template_amr".
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:num_examples]


def build_system_prompt() -> str:
    """Build a minimal system prompt for sentence generation."""
    system_prompt = (
        "You are a helpful assistant. Follow the user's instructions to produce exactly one sentence.\n"
        "Do not provide any explanations of your reasoning."
    )

    return system_prompt.strip()


def build_user_prompt(chosen_template: str, examples: list, context: str = "") -> str:
    """
    Build the user prompt containing:
      - optional "context"
      - the example AMRs + example English sentences
      - instructions about placeholders
      - the chosen template AMR
    """
    example_text = ""
    for ex in examples:
        example_text += (
            f"\nTemplate AMR:\n{ex['normalized_template_amr']}\n"
            f"Possible sentence: {ex['sentence']}\n"
        )

    user_prompt = f"""AMR (Abstract Meaning Representation) is a graph-based representation of a sentence's meaning.
        Each node is a concept and edges represent semantic roles or relationships.

        Below are some examples of template AMRs and corresponding sentences:
        {example_text}

        In the provided AMR, there are placeholders:
        - "V" for a verb or event. Realize it as the main predicate.
        - "NE" for named entities (e.g., "Alice", "France", "Google").
        - "N" for generic nouns (e.g., "a device", "an object").
        - "X" for unspecified concepts (e.g., "something", "an idea").
        - "ADJ" for an adjective or state.
        - "NUM" for a number or quantity.
        - "LOC" for a location.

        Instructions:

        - Do not write "NE", "N", or "X" literally. Instead, replace them with appropriate English words to form a natural, meaningful sentence.
        - Preserve the AMR graph structure: realize every listed role (such as :ARG0, :ARG1, :ARG2, :name, and :op1) and do not add extra clauses.
        - Do not introduce semantic roles that are absent from the AMR. In particular, if the root has no :ARG0, do not invent an agent; use a natural agentless passive or another construction whose parser omits :ARG0.
        - For a verbal root, use canonical active English: realize :ARG0 as the standalone grammatical subject, :ARG1 as the direct object/theme, and :ARG2 as a secondary complement.
        - Never turn an :ARG0 named entity into a possessive modifier. For example, use "IBM Corporation launched a server", not "IBM Corporation's server launched something".
        - Quoted :op1, :op2, and :op3 values under :name indicate the required number of words in that entity's name.
        - Use the context only to choose topical words and entities. Do not summarize or copy all facts from the context.
        - Prefer the shortest natural sentence that realizes the complete AMR structure.
        - Do not produce multiple sentences or lists.
        - Produce exactly one coherent sentence.

        AMR:
        {chosen_template}

        Context: {context}

        Please output only that one sentence."""

    return user_prompt

def chat_template(model_id, user_text, system_text):
    # All DeepSeek-R1-Distill variants share the same <think> prompt format.
    if "DeepSeek-R1-Distill" in model_id:
        prompt_to_model = f"""{user_text}<think>\n"""

    return prompt_to_model

def clean(text):
    match = re.search(r'</think>\s*(.*?)(?:<\|end▁of▁sentence\|>|$)', text, re.DOTALL)
    
    if not match:
        return "No text found after </think> tag"
    
    # Get the extracted text
    extracted_text = match.group(1)
    
    # Remove newlines and leading/trailing whitespace
    cleaned_text = ' '.join(extracted_text.split())
    
    # Remove EOS token if present (as a safety check)
    cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
    return cleaned_text

def generate_through_HF(model_id, model, tokenizer, user_text, system_text, max_tokens, temperature, top_p):
    if "DeepSeek-R1-Distill" in model_id:
        # DeepSeek-R1 distilled variants are tuned for T=0.6.
        temperature = 0.6
    model.generation_config.temperature=temperature
    model.generation_config.top_p=top_p
    model.eval()
    model.to(torch.device('cuda'))
    
    prompt_to_model = chat_template(model_id, user_text, system_text)
    tokenized_input = tokenizer(prompt_to_model, padding=False, truncation=False, return_tensors="pt")
    with torch.no_grad():
            generated_output = model.generate(input_ids = tokenized_input["input_ids"].cuda(), attention_mask = tokenized_input['attention_mask'].cuda(),  max_new_tokens = 1000, do_sample= True, pad_token_id=tokenizer.eos_token_id)
       
    tokenized_output = generated_output[0][len(tokenized_input["input_ids"][0]):]
    decoded_output = tokenizer.decode(tokenized_output)

    return clean(decoded_output)


# watermark injection 
def amr_instruction_completion(
    context: str,
    amr_bank: list,
    model_id: str,
    region: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_trials: int,
    amr_examples_path: str,
    s2match_threshold: float = 0.75,
    max_templates_to_try: int = 10,          # how many different templates to attempt
    max_retries_per_template: int = 5,       # how many times we retry each template
    model = None,
    tokenizer = None,
    hf_model = False
) -> str:
    """
    Generate a sentence from a randomly chosen AMR in the bank that matches it,
    using a simpler system prompt and a user prompt that includes the examples/explanation.
    Implements a basic "exact match" approach (if desired).
    """
    # Load a few examples to improve the user prompt
    examples = load_amr_examples(amr_examples_path, num_examples=3)

    # Initialize Bedrock (only needed for API-based models)
    bm = None
    if not hf_model:
        bm = BedrockManager(region=region, model_id=model_id)

    # Build the minimal system prompt
    system_prompt = build_system_prompt()

    total_attempts = 0
    templates_tried = 0

    last_candidate = ""
    last_template_tried = ""

    # We keep trying new templates up to max_templates_to_try
    while templates_tried < max_templates_to_try and total_attempts < max_trials:
        chosen_template = random.choice(list(amr_bank))
        user_prompt = build_user_prompt(chosen_template, examples, context=context)
        attempt_prompt = user_prompt


        # We'll attempt generating up to max_retries_per_template times
        for attempt in range(max_retries_per_template):
            if total_attempts >= max_trials:
                break  # stop if we reached overall limit

            total_attempts += 1
            logger.debug(f"Attempt {total_attempts}/{max_trials} (Template attempt {attempt+1}/{max_retries_per_template})")
            
            if hf_model == True:
                candidate = generate_through_HF(model_id, model, tokenizer, user_text=attempt_prompt,
                    system_text=system_prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p)
            else:
                candidate = bm.generate(
                    user_text=attempt_prompt,
                    system_text=system_prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

            logger.debug(f"Candidate: {candidate}")
            placeholder_match = re.search(
                r"(?:\b(?:NE|ADJ|NUM|LOC|AMR)\b|:(?:ARG\d|op\d|name|mode|time|location)\b)",
                candidate,
            )
            if placeholder_match:
                logger.debug("Rejected candidate containing an AMR placeholder or role marker.")
                last_template_tried = chosen_template
                attempt_prompt = user_prompt + f"""

Correction feedback: your previous sentence contained the forbidden literal marker {placeholder_match.group(0)!r}.
Replace every AMR placeholder with a real English word or name. Return only a natural sentence with no AMR notation."""
                continue
            # Parse the candidate
            candidate_amr = parse_amr(candidate)
            cand_norm = normalize_amr_variables(candidate_amr)
            cand_templ = generate_template_amr(cand_norm)
            cand_final = normalize_amr_variables(cand_templ)


            # Check S2match
            is_match, score = decide_s2match_match(
                cand_final,
                chosen_template,
                threshold=s2match_threshold
            )

            logger.debug(f"S2match = {score:.3f} (Threshold={s2match_threshold})")
            
            if is_match:
                logger.debug(f"Accepted! (S2match >= {s2match_threshold})")
                return candidate.strip(), total_attempts, chosen_template, float(score), True
            else:
                logger.debug("Rejected. Trying again or switching template.")
                last_candidate = candidate
                last_template_tried = chosen_template
                attempt_prompt = user_prompt + f"""

Correction feedback from the official AMR parser:
- Your previous sentence was: {candidate}
- Its normalized template was:
{cand_final}
- Its S2MATCH score against the target was {score:.3f}, below {s2match_threshold:.3f}.

Generate a different sentence. Correct the role, node-type, name-token-count, and extra-edge mismatches so the parser output matches the target AMR. Return only the corrected sentence."""

        # Done with the current template, try another
        templates_tried += 1

    # If we exit the while loop, we either ran out of templates or attempts
    if last_candidate:
        final_amr = parse_amr(last_candidate)
        final_norm = normalize_amr_variables(final_amr)
        final_templ = generate_template_amr(final_norm)
        final_score = decide_s2match_match(
            normalize_amr_variables(final_templ), last_template_tried,
            threshold=s2match_threshold
        )[1]
        return last_candidate.strip(), total_attempts, last_template_tried, float(final_score), False
    else:
        return "", total_attempts, last_template_tried, None, False


def build_user_prompt_no_watermark(context):
    user_prompt = f"""Given the following context, generate exactly one complete and grammatically correct sentence as a continuation.

        Context: {context}

        Please output only one sentence."""

    return user_prompt


def generate_through_HF_no_watermark(model_id, model, tokenizer, user_text, system_text, max_tokens, temperature, top_p):
    if "DeepSeek-R1-Distill" in model_id:
        # DeepSeek-R1 distilled variants are tuned for T=0.6.
        temperature = 0.6
    model.generation_config.temperature=temperature
    model.generation_config.top_p=top_p
    model.eval()
    model.to(torch.device('cuda'))
    
    prompt_to_model = chat_template(model_id, user_text, system_text)
    tokenized_input = tokenizer(prompt_to_model, padding=False, truncation=False, return_tensors="pt")
    with torch.no_grad():
            generated_output = model.generate(input_ids = tokenized_input["input_ids"].cuda(), attention_mask = tokenized_input['attention_mask'].cuda(),  max_new_tokens = 1000, do_sample= True, pad_token_id=tokenizer.eos_token_id)
       
    tokenized_output = generated_output[0][len(tokenized_input["input_ids"][0]):]
    decoded_output = tokenizer.decode(tokenized_output)

    return clean(decoded_output)


def non_watermark_generation(
    context: str,
    model_id: str,
    region: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model = None,
    tokenizer = None,
    hf_model = False,
):
    user_prompt = build_user_prompt_no_watermark(context=context)
    system_prompt = ""
    if hf_model:
        candidate = generate_through_HF_no_watermark(
            model_id, model, tokenizer, user_text=user_prompt,
            system_text=system_prompt, max_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p
        )
    else:
        candidate = BedrockManager(region=region, model_id=model_id).generate(
            user_text=user_prompt, system_text=system_prompt,
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
    return candidate.strip()
