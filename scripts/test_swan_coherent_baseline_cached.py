"""Offline tests for cached coherent-baseline detection."""
import unittest

import run_swan_coherent_baseline_cached as baseline


class CleanModule:
    def parse_amr(self, text):
        return "(a / approve-01)"

    def normalize_amr_variables(self, graph):
        return graph

    def generate_template_amr(self, graph):
        return graph


class FixedCachedScorer:
    def score(self, candidate, bank_graph, score_seed):
        return 0.7, 0.8, 0.75


class SourceTests(unittest.TestCase):
    def test_source_preserves_passed_quality_and_ten_units(self):
        inputs = baseline.source_inputs()
        self.assertTrue(inputs["quality_summary"]["qualified_for_detection"])
        self.assertEqual(inputs["quality_summary"]["review_passes"], 2)
        self.assertEqual(len(inputs["selected"]), 10)
        self.assertEqual(len(inputs["document"]["segmentation_spans"]), 10)

    def test_equivalence_contract_is_fixed(self):
        self.assertEqual(
            baseline.EQUIVALENCE_PAIRS,
            ((0, 0), (4, 25), (9, 47)),
        )
        self.assertEqual(baseline.EQUIVALENCE_TOLERANCE, 1e-12)


class DetectionTests(unittest.TestCase):
    SOURCE = {
        "slot": 0,
        "text": "Maria Lopez approved the recommendations.",
        "text_sha256": "unused-in-test",
        "target_template_index": 47,
        "target_score": 0.9,
    }

    def test_cached_unit_is_green(self):
        row = baseline.score_unit(
            CleanModule(),
            FixedCachedScorer(),
            ["(a / approve-01)"],
            "doc",
            0,
            self.SOURCE,
        )
        self.assertTrue(row["green"])
        self.assertEqual(row["max_score"], 0.75)
        self.assertEqual(row["diagnostics"]["stderr"], "")


if __name__ == "__main__":
    unittest.main()
