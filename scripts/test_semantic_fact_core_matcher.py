import itertools
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.semantic_fact_core_matcher import (
    SearchBudgetExceeded, exact_match, graph_payload, representation, score_view,
)

CONTRACT = {"role_one": "Alice", "role_two": "Bob", "object": "box"}
GRAPH = """(r / receive-01
 :ARG0 (b / person :name (n / name :op1 "Bob"))
 :ARG1 (x / box)
 :ARG2 (a / person :name (m / name :op1 "Alice")))"""


def brute(left, right):
    lc = {s: t for s, r, t in left["triples"] if r == ":instance"}
    rc = {s: t for s, r, t in right["triples"] if r == ":instance"}
    rt = set(map(tuple, right["triples"]))
    best = 0
    for values in itertools.product(*[[n for n, c in rc.items() if c == lc[v]] + [None] for v in lc]):
        if len([v for v in values if v is not None]) != len({v for v in values if v is not None}):
            continue
        mapping = dict(zip(lc, values))
        count = 0
        for s, rel, t in left["triples"]:
            end = mapping.get(t) if rel != ":instance" and t in lc else t
            count += mapping[s] is not None and (mapping[s], rel, end) in rt
        best = max(best, count)
    return best


class MatcherTests(unittest.TestCase):
    def test_small_exhaustive_oracle(self):
        rng = random.Random(42)
        for _ in range(32):
            pair = []
            for prefix in ("a", "b"):
                triples = [(prefix + str(i), ":instance", rng.choice(("person", "thing"))) for i in range(3)]
                triples += [(prefix + str(i), ":ARG1", prefix + str(j))
                            for i in range(3) for j in range(3) if i != j and rng.random() < .3]
                pair.append(graph_payload(triples, {}))
            self.assertEqual(exact_match(*pair)["matched_triples"], brute(*pair))

    def test_variable_renaming(self):
        one = representation(GRAPH, CONTRACT, "receive", ["receive"])
        two = representation(GRAPH.replace("(r /", "(rr /"), CONTRACT, "receive", ["receive"])
        self.assertEqual(exact_match(one["full"], two["full"])["score"], 1.)

    def test_role_reversal_does_not_get_perfect_score(self):
        one = representation(GRAPH, CONTRACT, "receive", ["receive"])
        reversed_graph = GRAPH.replace('"Bob"', '"TMP"').replace('"Alice"', '"Bob"').replace('"TMP"', '"Alice"')
        two = representation(reversed_graph, CONTRACT, "receive", ["receive"])
        self.assertLess(exact_match(one["full"], two["full"])["score"], 1.)

    def test_negation_is_not_completed(self):
        result = representation(GRAPH.replace(":ARG0", ":polarity - :ARG0", 1), CONTRACT, "receive", ["receive"])
        self.assertEqual(result["status"], "invalid")

    def test_unknown_anchor_never_borrows_full_document_fact(self):
        self.assertEqual(representation(GRAPH, CONTRACT, "watch", ["watch"])["status"], "invalid")
        self.assertEqual(representation(GRAPH, CONTRACT, "receive", ["receive", "receive"])["status"], "invalid")

    def test_budget_never_returns_incumbent(self):
        one = representation(GRAPH, CONTRACT, "receive", ["receive"])
        with self.assertRaises(SearchBudgetExceeded):
            exact_match(one["full"], one["full"], 1)
        self.assertEqual(score_view(one, [one] * 24, "full", 1)["status"], "budget_abstain")


if __name__ == "__main__":
    unittest.main()
