"""Tests for the WP2-v6 score-blind workload audit."""

import unittest

from audit_certified_resegmentation_wp2_v6_workload import (
    adjacent_profile_indices,
    audit_document,
    batched_token_counts,
    conservative_rate,
    sidecar_bytes,
)
from certified_resegmentation_wp2_v3 import canonicalize_structured


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs.get("truncation") is not False:
            raise AssertionError("workload audit requested truncation")
        if isinstance(text, list):
            return {
                "input_ids": [
                    list(range(len(value.split()) + 2))
                    for value in text
                ]
            }
        return {"input_ids": list(range(len(text.split()) + 2))}


class WorkloadAuditTests(unittest.TestCase):
    def setUp(self):
        self.profiles = [
            {
                "atom_count": 10,
                "selection_slot": "small",
                "rate": 1.0,
            },
            {
                "atom_count": 20,
                "selection_slot": "middle",
                "rate": 3.0,
            },
            {
                "atom_count": 30,
                "selection_slot": "large",
                "rate": 2.0,
            },
        ]

    def test_adjacent_profiles_are_conservative_at_boundaries(self):
        self.assertEqual(
            adjacent_profile_indices(self.profiles, 5),
            (0, 1),
        )
        self.assertEqual(
            adjacent_profile_indices(self.profiles, 15),
            (0, 1),
        )
        self.assertEqual(
            adjacent_profile_indices(self.profiles, 20),
            (0, 1, 2),
        )
        self.assertEqual(
            adjacent_profile_indices(self.profiles, 35),
            (1, 2),
        )
        selected = conservative_rate(
            self.profiles,
            20,
            "rate",
        )
        self.assertEqual(selected["selected_slot"], "middle")
        self.assertEqual(selected["rate"], 3.0)

    def test_sidecar_size_accounts_for_only_legal_features(self):
        self.assertEqual(
            sidecar_bytes("semstamp", 3, 2),
            28 + 64 + 3 * 9 + 2,
        )
        self.assertEqual(
            sidecar_bytes("simmark", 3, 2),
            28 + 3 * 9 + 2 * 768 * 4,
        )

    def test_batched_token_counts_match_scalar_calls(self):
        tokenizer = FakeTokenizer()
        texts = ["alpha", "beta gamma", "delta epsilon zeta"]
        observed = batched_token_counts(tokenizer, texts)
        expected = [
            len(tokenizer(text, truncation=False)["input_ids"])
            for text in texts
        ]
        self.assertEqual(observed, expected)

    def test_streaming_topology_counts_without_model_features(self):
        document = canonicalize_structured([
            "alpha beta",
            " gamma delta",
            " epsilon zeta",
        ])
        result = audit_document(
            document,
            "semstamp",
            FakeTokenizer(),
        )
        self.assertGreater(result["unique_query_spans"], 0)
        self.assertEqual(
            result["unique_query_spans"],
            result["legal_query_spans"],
        )
        self.assertGreater(result["actual_unique_pairs"], 0)
        self.assertGreater(result["deduplicated_input_tokens"], 0)
        self.assertLess(
            result["deduplicated_feature_texts"],
            result["unique_query_spans"] + 1,
        )


if __name__ == "__main__":
    unittest.main()
