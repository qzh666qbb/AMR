"""S2Match string scoring with one immutable vector load per process."""
from __future__ import annotations

import importlib

import run_swan_positive_collection as swan


class CachedS2Matcher:
    def __init__(self, vectors_file):
        from utils.amr_utils import compute_s2match_from_strings

        module = importlib.import_module(compute_s2match_from_strings.__module__)
        self.module = module
        self.vectors_file = str(vectors_file)
        self.vectors = module.load_vecs(self.vectors_file)
        self.simfun = module.get_sim_fun("cosine")

    def score(self, candidate, bank_graph, score_seed):
        module = self.module
        with swan.fixed_score_seed(score_seed):
            module.match_triple_dict = {}
            module.iteration_num = 5
            try:
                amr1 = module.amr.AMR.parse_AMR_line(candidate, False)
                amr2 = module.amr.AMR.parse_AMR_line(bank_graph, False)
                amr1.rename_node("a")
                amr2.rename_node("b")
                instance1, attributes1, relation1 = amr1.get_triples()
                instance2, attributes2, relation2 = amr2.get_triples()
                _, best_match_num_soft = module.get_best_match(
                    instance1,
                    attributes1,
                    relation1,
                    instance2,
                    attributes2,
                    relation2,
                    "a",
                    "b",
                    self.vectors,
                    0.5,
                    0.5,
                    self.simfun,
                    "split",
                )
                test_triple_num = (
                    len(instance1) + len(attributes1) + len(relation1)
                )
                gold_triple_num = (
                    len(instance2) + len(attributes2) + len(relation2)
                )
                return module.compute_f(
                    best_match_num_soft,
                    test_triple_num,
                    gold_triple_num,
                )
            finally:
                module.match_triple_dict.clear()
