"""Failure attribution against actual facts, including erasures and pending data."""
import unittest

from src.core.semantic_fact_layers import compare_fact_layers

CONTRACT = {"role_one": "Mina", "role_two": "Theo", "object": "label box"}
FACTS = {"sender": "Mina", "recipient": "Theo", "object": "label box",
         "transfer_status": "completed", "readability": "clear", "explicit_status": "yes"}
REVERSE = {**FACTS, "sender": "Theo", "recipient": "Mina"}


class FactLayerTests(unittest.TestCase):
    def compare(self, **kwargs):
        kwargs.setdefault("raw_graph_facts_independent_confirmed", True)
        return compare_fact_layers(
            contract=CONTRACT, planned_symbol="A", text_facts=FACTS, **kwargs)

    def test_model_annotation_cannot_confirm_layers(self):
        r = self.compare(raw_graph_facts=FACTS, raw_prediction="B")
        self.assertEqual(r["status"], "PENDING_TEXT_CONFIRMATION")
        self.assertEqual(r["supported_failure_layers"], [])

    def test_model_graph_facts_cannot_confirm_classifier_failure(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=FACTS,
                         raw_prediction="B", raw_graph_facts_independent_confirmed=False)
        self.assertEqual(r["status"], "PENDING_RAW_GRAPH_FACT_CONFIRMATION")
        self.assertEqual(r["supported_failure_layers"], [])

    def test_absent_graph_is_pending_not_failed(self):
        r = self.compare(text_human_confirmed=True)
        self.assertEqual(r["status"], "PENDING_RAW_GRAPH")
        self.assertIsNone(r["raw_graph_matches_actual_text"])

    def test_revision_is_not_implicitly_independent(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=REVERSE,
                         revised_graph_facts=FACTS)
        self.assertEqual(r["status"], "PENDING_INDEPENDENT_REVISION")
        self.assertNotIn("parser_or_graph_representation", r["supported_failure_layers"])

    def test_verified_correction_supports_parser_loss(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=REVERSE,
                         revised_graph_facts=FACTS, revision_independent_confirmed=True,
                         raw_prediction="B", revised_prediction="A")
        self.assertEqual(r["supported_failure_layers"], ["parser_or_graph_representation"])
        self.assertEqual(r["status"], "LAYER_COMPARISON_COMPLETE")

    def test_correct_graph_wrong_symbol_is_classifier_failure(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=FACTS,
                         revised_graph_facts=FACTS, revision_independent_confirmed=True,
                         raw_prediction="B", revised_prediction="B")
        self.assertEqual(r["supported_failure_layers"], ["raw_classifier", "revised_classifier"])

    def test_erasure_on_correct_graph_is_classifier_failure(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=FACTS, raw_prediction=None)
        self.assertIn("raw_classifier", r["supported_failure_layers"])
        self.assertFalse(r["raw_output_matches_actual_text"])

    def test_incorrect_revision_cannot_establish_parser_failure(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts=REVERSE,
                         revised_graph_facts=REVERSE, revision_independent_confirmed=True)
        self.assertEqual(r["status"], "REVISION_DISAGREES_WITH_ACTUAL_TEXT")
        self.assertNotIn("parser_or_graph_representation", r["supported_failure_layers"])

    def test_unknown_graph_counts_as_loss_when_revision_restores_it(self):
        r = self.compare(text_human_confirmed=True, raw_graph_facts={**FACTS, "sender": "unknown"},
                         revised_graph_facts=FACTS, revision_independent_confirmed=True,
                         revised_prediction="A")
        self.assertIn("parser_or_graph_representation", r["supported_failure_layers"])

    def test_does_not_compare_graph_to_wrong_plan(self):
        r = compare_fact_layers(contract=CONTRACT, planned_symbol="A", text_facts=REVERSE,
                                text_human_confirmed=True, raw_graph_facts=REVERSE,
                                raw_graph_facts_independent_confirmed=True,
                                revised_graph_facts=REVERSE, revision_independent_confirmed=True,
                                raw_prediction="B", revised_prediction="B")
        self.assertEqual(r["text_symbol"], "B")
        self.assertEqual(r["supported_failure_layers"], ["writing"])
        self.assertTrue(r["raw_graph_matches_actual_text"])

    def test_ambiguous_text_is_unresolved(self):
        r = compare_fact_layers(contract=CONTRACT, planned_symbol="A",
                                text_facts={**FACTS, "readability": "unclear"},
                                text_human_confirmed=True)
        self.assertEqual(r["status"], "TEXT_UNRESOLVED")

    def test_planned_and_completed_are_different_states(self):
        r = self.compare(text_human_confirmed=True,
                         raw_graph_facts={**FACTS, "transfer_status": "planned_not_occurred"},
                         revised_graph_facts=FACTS, revision_independent_confirmed=True,
                         raw_prediction="C", revised_prediction="A")
        self.assertFalse(r["raw_graph_matches_actual_text"])
        self.assertIn("parser_or_graph_representation", r["supported_failure_layers"])


if __name__ == "__main__":
    unittest.main()
