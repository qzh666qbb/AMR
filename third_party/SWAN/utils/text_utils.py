# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
text_utils.py

Shared text processing utilities used across injection and detection scripts.
"""

PUNCTS = '!.?'

def extract_prompt_from_text(text, len_prompt):
    """Extract the first sentence from text as a prompt for generation.

    Tokenizes by whitespace, takes the first ``len_prompt`` tokens, then
    finds the earliest sentence-ending punctuation to form a clean prompt.
    Falls back to appending a period if no punctuation is found.
    """
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            # has to be greater than a minimum prompt
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    # select first (sub)sentence, delimited by any of PUNCTS
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt
