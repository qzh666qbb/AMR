"""No-network tests for the SWAN oracle-composition diagnostic."""

import math
import unittest

from run_confirmation_mechanism import lexical_tokens
from run_swan_oracle_composition import (
    batch_ids,
    count_only_z,
    detection_from_details,
    pair_intervention,
    primary_decision,
    z_score,
)


def detail(text, score):
    scores = [0.1] * 49 + [score]
    return {
        "text": text,
        "raw_amr": "(x / example)",
        "amr": "(x / N)",
        "bank_scores": scores,
        "score": score,
    }


class SwanOracleCompositionTests(unittest.TestCase):
    def test_pair_intervention_preserves_tokens_and_tail(self):
        sentences = [
            "Alice left.",
            "Bob stayed.",
            "Carol waited.",
            "Dan returned.",
            "Eve agreed.",
        ]
        units = pair_intervention(sentences)
        self.assertEqual(
            units,
            [
                "Alice left; Bob stayed.",
                "Carol waited; Dan returned.",
                "Eve agreed.",
            ],
        )
        self.assertEqual(
            lexical_tokens(" ".join(units)),
            lexical_tokens(" ".join(sentences)),
        )

    def test_z_score_matches_detector_formula(self):
        self.assertAlmostEqual(z_score(5, 5), math.sqrt(15))
        self.assertAlmostEqual(z_score(0, 1), -1 / math.sqrt(3))

    def test_count_only_retains_original_green_rate(self):
        expected = (0.8 - 0.25) * math.sqrt(15 / (0.25 * 0.75))
        self.assertAlmostEqual(count_only_z(20, 25, 15), expected)

    def test_detection_reconstructs_green_from_bank_scores(self):
        result = detection_from_details(
            [detail("One.", 0.65), detail("Two.", 0.64)]
        )
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["green"], 1)
        self.assertFalse(result["detected"])

    def test_batches_are_fixed_consecutive_groups(self):
        groups = batch_ids()
        self.assertEqual(len(groups), 6)
        self.assertEqual(groups[0], [f"c4-validation-{i}" for i in range(250, 255)])
        self.assertEqual(groups[-1], [f"c4-validation-{i}" for i in range(275, 280)])

    def test_primary_decision_requires_five_pair_failures(self):
        groups = []
        for index in range(6):
            groups.append(
                {
                    "conditions": {
                        "pair": {
                            "count_only_z": 4.0,
                            "z": 1.0 if index < 5 else 3.0,
                            "detected": index == 5,
                        }
                    }
                }
            )
        result = primary_decision(groups, True)
        self.assertTrue(result["supports_boundary_sensitive_detector_mechanics"])
        groups[4]["conditions"]["pair"]["detected"] = True
        groups[4]["conditions"]["pair"]["z"] = 3.0
        self.assertFalse(
            primary_decision(groups, True)[
                "supports_boundary_sensitive_detector_mechanics"
            ]
        )


if __name__ == "__main__":
    unittest.main()
