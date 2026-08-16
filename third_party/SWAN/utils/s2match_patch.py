# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
s2match_patch.py

Patch for s2match from the amr-metric-suite repository
(https://github.com/flipz357/amr-metric-suite).

This file contains only the additional entry point `compute_s2match_from_strings`
that SWAN uses. The upstream `s2match.py` exposes a file-based / document-level
API; SWAN needs a string-based API that returns (precision, recall, f_score)
for two AMR strings.

Setup (see README §2):
    git clone https://github.com/flipz357/amr-metric-suite.git
    cat utils/s2match_patch.py >> amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py

After appending, utils/amr_utils.py imports compute_s2match_from_strings from
the patched upstream file via sys.path manipulation.
"""

def compute_s2match_from_strings(amr_string1, amr_string2,
                                 vectors_file="../vectors/glove.6B.100d.txt",
                                 similarity_function="cosine", cutoff=0.5,
                                 diffsense=0.5, mwp="split",
                                 do_not_mark_quotes=False, verbose=False):
    """
    Compute the S2match score between two AMR strings.

    Args:
        amr_string1 (str): The first AMR string.
        amr_string2 (str): The second AMR string.
        vectors_file (str): Path to the vector file.
        similarity_function (str): Similarity function to use ("cosine", "euclidean", "cityblock").
        cutoff (float): Similarity cutoff threshold.
        diffsense (float): Coefficient for concepts with different senses.
        mwp (str): Strategy for multi-word concepts ("split" or "None").
        do_not_mark_quotes (bool): Treat quotes as tokens if True.
        verbose (bool): Enable verbose output.

    Returns:
        tuple: A tuple containing precision, recall, and F1 score.
    """
    global match_triple_dict
    global iteration_num

    # Initialize variables
    match_triple_dict = {}
    iteration_num = 5  # Number of iterations for hill-climbing

    # Load vectors and similarity function
    vectors = load_vecs(vectors_file)
    simfun = get_sim_fun(similarity_function)

    # Parse AMR strings
    amr1 = amr.AMR.parse_AMR_line(amr_string1, do_not_mark_quotes)
    amr2 = amr.AMR.parse_AMR_line(amr_string2, do_not_mark_quotes)

    # Rename nodes
    prefix1 = "a"
    prefix2 = "b"
    amr1.rename_node(prefix1)
    amr2.rename_node(prefix2)

    # Get triples
    instance1, attributes1, relation1 = amr1.get_triples()
    instance2, attributes2, relation2 = amr2.get_triples()

    # Compute best mapping
    best_mapping, best_match_num_soft = get_best_match(
        instance1, attributes1, relation1,
        instance2, attributes2, relation2,
        prefix1, prefix2, vectors, cutoff, diffsense, simfun, mwp
    )

    # Compute triple counts
    test_triple_num = len(instance1) + len(attributes1) + len(relation1)
    gold_triple_num = len(instance2) + len(attributes2) + len(relation2)

    # Compute precision, recall, F1
    precision, recall, f_score = compute_f(
        best_match_num_soft, test_triple_num, gold_triple_num
    )

    # Clear match triple dictionary
    match_triple_dict.clear()

    return precision, recall, f_score
