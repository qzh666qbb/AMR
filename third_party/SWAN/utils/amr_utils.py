# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
amr_utils.py

Core AMR processing utilities shared across injection, detection, and bank creation.

Provides:
  - parse_amr():                  Sentence → AMR string (via amrlib stog model)
  - normalize_amr_variables():    BFS-based variable renaming for canonical form
  - generate_template_amr():      Replace concepts with placeholders (V, N, NE, X, ...)
  - compute_s2match_score():      S2match F1 between two AMR strings
  - decide_s2match_match():       Threshold-based accept/reject using S2match
  - load_amr_bank():              Load curated bank from JSON
  - remove_amr_comments():        Strip comment lines from AMR strings
  - classify_concept():           Map AMR concept to placeholder type
"""

import os
import sys
import json
import re
import logging
import penman
import amrlib
import spacy
import nltk
from nltk.corpus import propbank
from collections import defaultdict, deque

# S2match lives inside the cloned amr-metric-suite repo, not in this package.
# Users clone amr-metric-suite at the project root and copy utils/s2match.py
# (our modified version, adds compute_s2match_from_strings) into
# amr-metric-suite/py3-Smatch-and-S2match/smatch/. See README §2.
# Adding that directory to sys.path lets s2match resolve its sibling imports
# (amr_py3, helpers) the same way upstream does.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S2MATCH_DIR = os.path.join(_PROJECT_ROOT, "amr-metric-suite", "py3-Smatch-and-S2match", "smatch")
if _S2MATCH_DIR not in sys.path:
    sys.path.insert(0, _S2MATCH_DIR)

from s2match import compute_s2match_from_strings

logger = logging.getLogger(__name__)

# Use provisioned local corpora in offline/reproducible runs.  Calling the
# downloader here refreshes the online index and can fail even when the
# corpus is already installed.
try:
    nltk.data.find('corpora/propbank')
except LookupError:
    try:
        nltk.data.find('corpora/propbank.zip')
    except LookupError:
        raise RuntimeError('Local NLTK propbank is required; refusing online download during detection.')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the AMR model only when parsing is requested. Detection workers only need
# S2MATCH and should not each allocate a BART-large parser on Windows spawn.
stog = None

nlp = spacy.load("en_core_web_sm")

# Build a set of verb lemmas from PropBank
verb_lemmas = set()
for inst in propbank.instances():
    roleset_id = inst.roleset
    lemma = roleset_id.split('.')[0]
    verb_lemmas.add(lemma)

verb_pattern = re.compile(r'(.*)-\d+$')

def classify_concept(concept: str) -> str:
    # Special hard-coded mappings
    if concept == 'amr-unknown':
        return 'X'
    if concept in ['date-entity', 'temporal-quantity', 'monetary-quantity']:
        return 'NUM'
    if concept in ['city', 'country', 'continent', 'state', 'province']:
        return 'LOC'

    # Attempt to detect verbs via PropBank
    match = verb_pattern.match(concept)
    if match:
        lemma = match.group(1)
        if lemma in verb_lemmas:
            return 'V'
    
    # Fallback to spaCy POS tagging
    # Remove the -XX sense if present
    lemma = re.sub(r'-\d+$', '', concept)
    doc = nlp(lemma)
    if not doc:
        # If spaCy can't parse
        return 'N'
    pos = doc[0].pos_
    if pos == 'PROPN':
        return 'NE'
    elif pos == 'VERB':
        return 'V'
    elif pos == 'NUM':
        return 'NUM'
    elif pos == 'ADJ':
        return 'ADJ'

    # Default fallback
    return 'N'

def generate_template_amr(amr_str):
    try:
        if not amr_str.strip().startswith('('):
            logger.warning(f"Invalid AMR: {amr_str}")
            return amr_str

        graph = penman.decode(amr_str)
        named_entity_nodes = set()

        # Identify named entity nodes
        for triple in graph.triples:
            if triple[1] == ':name':
                named_entity_nodes.add(triple[0])

        template_triples = []
        for triple in graph.triples:
            rel = triple[1]
            concept = triple[2]

            if rel == ':instance':
                node = triple[0]
                if node in named_entity_nodes:
                    # Named entity node with a name role
                    placeholder = 'NE'
                else:
                    placeholder = classify_concept(concept)
                template_triples.append((node, rel, placeholder))
            elif concept.startswith('"') and concept.endswith('"'):
                template_triples.append((triple[0], rel, '"NE"'))
            else:
                template_triples.append(triple)

        template_graph = penman.Graph(template_triples)
        return penman.encode(template_graph)
    except Exception as e:
        logger.warning(f"Error generating template AMR: {e}")
        return amr_str


def parse_amr(sentence: str) -> str:
    """
    Parse a sentence into AMR using the stog model.
    Returns the AMR string (without comments), ensuring we extract only the AMR part.
    """
    global stog
    # Lazy-load in each process if not loaded
    if stog is None:
        logger.info("Loading stog model...")
        stog = amrlib.load_stog_model(device='cuda')
        logger.info("stog model loaded.")

    amrs = stog.parse_sents([sentence])

    if amrs and amrs[0]:
        amr_output = amrs[0].strip().split('\n')
        # Find the first line starting with '(' which marks the start of the AMR
        start_idx = None
        for i, line in enumerate(amr_output):
            if line.strip().startswith('('):
                start_idx = i
                break
        if start_idx is not None:
            # Extract only the lines from the AMR start to the end
            amr_str = '\n'.join(amr_output[start_idx:])
            return remove_amr_comments(amr_str)
        else:
            # No AMR found
            return ""
    return ""


def normalize_amr_variables(amr_str):
    """
    Normalize AMR variable names by performing a BFS traversal from the top node.
    Each encountered node is assigned a new variable name in discovery order.
    Outgoing edges are sorted by (relation, concept) to ensure stable ordering.
    """
    try:
        graph = penman.decode(amr_str)

        # Extract structure for traversal
        # Build adjacency list: var -> list of (rel, target_var_or_constant)
        # Also record concept for each variable for stable sorting.
        var_to_concept = {}
        adjacency = defaultdict(list)

        for triple in graph.triples:
            src, rel, tgt = triple
            if rel == ':instance':
                # This triple defines the concept for src
                # tgt is a concept like 'want-01', 'person', etc.
                var_to_concept[src] = tgt
            else:
                # Add edge
                adjacency[src].append((rel, tgt))

        # We will do a BFS from the top node
        top = graph.top
        # Assign new variable names in BFS order
        # Start with top node as x0
        new_names = {}
        queue = deque([top])
        new_names[top] = 'x0'
        next_index = 1

        while queue:
            current = queue.popleft()
            # Sort children by a stable criterion to ensure consistency
            # We can sort by (relation, concept_of_tgt_if_var, tgt_str)
            children = adjacency[current]

            # We'll define a helper to get sorting keys
            def sort_key(edge):
                rel, tgt = edge
                # If tgt is a variable, get its concept, otherwise use tgt as-is
                if tgt in var_to_concept:
                    return (rel, var_to_concept[tgt], tgt)
                else:
                    # Tgt might be a string literal or something else
                    return (rel, str(tgt), str(tgt))

            children.sort(key=sort_key)

            # Assign names and enqueue new variables
            for (rel, tgt) in children:
                if tgt in var_to_concept:  # It's a variable node
                    if tgt not in new_names:
                        new_names[tgt] = f"x{next_index}"
                        next_index += 1
                        queue.append(tgt)

        # Now rebuild triples with the new names
        new_triples = []
        for (src, rel, tgt) in graph.triples:
            new_src = new_names[src] if src in new_names else src
            if tgt in new_names:
                new_tgt = new_names[tgt]
            else:
                new_tgt = tgt
            new_triples.append((new_src, rel, new_tgt))

        # Update top in case it changed
        new_top = new_names.get(top, top)
        new_graph = penman.Graph(new_triples, top=new_top)
        normalized = penman.encode(new_graph)
        return normalized
    except Exception as e:
        logger.warning(f"Error normalizing AMR: {e}")
        return amr_str


def remove_amr_comments(amr_str):
    """
    Remove comments from an AMR string.
    """
    # Split the AMR string into lines
    lines = amr_str.strip().split('\n')
    # Keep only lines that do not start with '#'
    non_comment_lines = [line for line in lines if not line.strip().startswith('#')]
    # Join the lines back into a single string
    amr_no_comments = '\n'.join(non_comment_lines)
    return amr_no_comments


def load_amr_bank(path: str):
    """Load curated AMRs in stable JSON order, removing duplicates."""
    with open(path, 'r') as f:
        data = json.load(f)

    amr_list = [entry['normalized_template_amr'] for entry in data if 'normalized_template_amr' in entry]
    template_bank = list(dict.fromkeys(amr_list))

    logger.info(f"Loaded {len(template_bank)} template AMRs from '{path}'")
    return template_bank


###############################
# S2match-based matching logic
###############################
def compute_s2match_score(amr1_str: str, amr2_str: str) -> float:
    """
    Compute the S2match F1 score between two AMR strings.
    Set the GLOVE_VECTORS_PATH environment variable to point to your
    glove.6B.100d.txt file.
    """
    vectors_path = os.environ.get("GLOVE_VECTORS_PATH", "glove.6B.100d.txt")
    precision, recall, f_score = compute_s2match_from_strings(
        amr1_str,
        amr2_str,
        vectors_file=vectors_path,
        similarity_function="cosine",
        cutoff=0.5,
        diffsense=0.5,
        mwp="split",
        do_not_mark_quotes=False,
        verbose=False,
    )
    return f_score


def decide_s2match_match(
    amr1_str: str,
    amr2_str: str,
    threshold: float
):
    """
    Decide if the two AMR strings match according to S2match.
    Returns (is_match, s2match_score).
    """
    score = compute_s2match_score(amr1_str, amr2_str)
    return (score >= threshold, score)


if __name__ == "__main__":
    # Test cases for normalize_amr_variables

    test_amr_1 = """
    (w / want-01
       :ARG0 (p / person :name (n / name :op1 "Alice"))
       :ARG1 (h / help-01
                :ARG0 p
                :ARG1 (b / boy :name (n2 / name :op1 "Brian"))))
    """

    # Same structure as test_amr_1 but different variable names
    test_amr_2 = """
    (x1 / want-01
        :ARG0 (x2 / person :name (x3 / name :op1 "Alice"))
        :ARG1 (x4 / help-01
                 :ARG0 x2
                 :ARG1 (x5 / boy :name (x6 / name :op1 "Brian"))))
    """

    # After normalization, these two should produce identical normalized AMR strings
    norm_1 = normalize_amr_variables(test_amr_1)
    norm_2 = normalize_amr_variables(test_amr_2)

    print("Normalized AMR 1:\n", norm_1)
    print("Normalized AMR 2:\n", norm_2)
    assert norm_1.strip() == norm_2.strip(), "Normalization test failed: AMRs do not match after normalization."

    # Another test case with a slightly different AMR
    test_amr_3 = """
    (a / see-01
       :ARG0 (c / cat)
       :ARG1 (d / dog))
    """

    test_amr_4 = """
    (z / see-01
       :ARG0 (y / cat)
       :ARG1 (x / dog))
    """

    norm_3 = normalize_amr_variables(test_amr_3)
    norm_4 = normalize_amr_variables(test_amr_4)
    print("Normalized AMR 3:\n", norm_3)
    print("Normalized AMR 4:\n", norm_4)
    assert norm_3.strip() == norm_4.strip(), "Normalization test failed: Cat/Dog AMRs do not match after normalization."

    print("All normalization tests passed!")

