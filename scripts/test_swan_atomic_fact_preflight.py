"""No-network tests for the SWAN atomic-fact scale preflight."""

from types import SimpleNamespace
import unittest

from run_swan_atomic_fact_preflight import (
    choose_fact_prefix,
    count_upper_bound,
    document_summary,
    surface_fact_candidates,
    syntax_accepts,
)


def tokenize(text):
    return [sentence.strip() + "." for sentence in text.split(".") if sentence.strip()]


def fact(index, words=10):
    text = " ".join([f"Fact{index}"] + ["plain"] * (words - 2) + ["ends."])
    return {
        "sentence_index": index,
        "text": text,
        "text_sha256": str(index),
        "normalized_sha256": f"normalized-{index}",
        "words": words,
    }


class AtomicFactPreflightTests(unittest.TestCase):
    def test_surface_filter_keeps_only_bounded_declarative_text(self):
        valid = "Researchers describe several stable patterns across the collected samples."
        rows = surface_fact_candidates(
            " ".join([
                valid,
                "This sentence begins with an excluded anaphoric token today.",
                "Analysts recorded 12 outcomes across the complete study sample.",
                "Readers can inspect the source, method, and conclusion carefully.",
            ]),
            tokenize,
        )
        self.assertEqual([row["text"] for row in rows], [valid])

    def test_syntax_requires_root_with_direct_subject(self):
        root = SimpleNamespace(dep_="ROOT", pos_="VERB")
        subject = SimpleNamespace(dep_="nsubj", pos_="NOUN", head=root)
        object_token = SimpleNamespace(dep_="dobj", pos_="NOUN", head=root)
        self.assertTrue(syntax_accepts([subject, root, object_token]))
        subject.head = object_token
        self.assertFalse(syntax_accepts([subject, root, object_token]))

    def test_fact_prefix_stops_at_frozen_minimum(self):
        chosen, words, eligible = choose_fact_prefix([fact(index) for index in range(14)])
        self.assertTrue(eligible)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(words, 100)

    def test_count_bound_is_typed_and_symmetric(self):
        self.assertEqual(count_upper_bound([2, 1, 1], [2, 1, 1]), 1.0)
        self.assertEqual(count_upper_bound([2, 0, 0], [0, 2, 0]), 0.0)
        self.assertEqual(
            count_upper_bound([3, 1, 2], [2, 2, 2]),
            count_upper_bound([2, 2, 2], [3, 1, 2]),
        )

    def test_document_gate_uses_all_frozen_facts(self):
        source = {"prompt_id": "p", "shard_row": 1200, "facts": [fact(i) for i in range(10)]}
        rows = []
        for index in range(10):
            rows.append({
                **fact(index),
                "template_triples": 10,
                "compatible_bank_templates": 17,
                "scale_compatible": index < 8,
            })
        result = document_summary(source, {"facts": rows})
        self.assertTrue(result["document_gate"])
        rows[-1]["template_triples"] = 30
        rows[-2]["template_triples"] = 30
        rows[-3]["template_triples"] = 30
        rows[-4]["template_triples"] = 30
        rows[-5]["template_triples"] = 30
        rows[-6]["template_triples"] = 30
        self.assertFalse(document_summary(source, {"facts": rows})["document_gate"])


if __name__ == "__main__":
    unittest.main()
