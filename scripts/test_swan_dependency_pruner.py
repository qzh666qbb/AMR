"""No-network tests for dependency-path microfact pruning."""

from pathlib import Path
import tempfile
import unittest

from run_swan_dependency_pruner import (
    base_plan_metrics,
    prepare,
    prune_body,
    prune_plan,
)


def plan(subject, predicate, object_text, evidence):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_text,
        "evidence": evidence,
    }


class DependencyPrunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")

    def test_prunes_to_complete_bounded_dependency_path(self):
        body = (
            "Other Kerberos implementations also put user identification data "
            "in the auth-data field"
        )
        result = prune_body(body, self.nlp)
        self.assertTrue(result["validation"]["pass"])
        self.assertEqual(
            result["text"],
            "Other Kerberos implementations put user identification data.",
        )
        self.assertEqual(result["validation"]["words"], 7)

    def test_passthrough_preserves_negation_and_modal(self):
        body = "Apple should not be banned from the market"
        result = prune_body(body, self.nlp)
        self.assertTrue(result["validation"]["pass"])
        self.assertIn("should", result["text"])
        self.assertIn("not", result["text"])
        self.assertEqual(result["mode"], "passthrough")

    def test_rejects_path_that_cannot_fit_without_semantic_loss(self):
        body = (
            "Judge Thomas Pender found that Apple infringed one of three "
            "Qualcomm patents in the case"
        )
        result = prune_body(body, self.nlp)
        self.assertIsNone(result["text"])
        self.assertEqual(
            result["reason"],
            "mandatory_path_exceeds_word_budget",
        )

    def test_base_checks_reject_pronoun_question_and_conditional(self):
        question = "If you license the PAC, how open is Kerberos?"
        value = plan(
            "you",
            "license",
            "the PAC",
            question,
        )
        result = base_plan_metrics(value, question)
        self.assertFalse(result["checks"]["resolved_subject"])
        self.assertFalse(result["checks"]["not_question_evidence"])
        self.assertFalse(result["checks"]["not_initial_conditional_clause"])

    def test_prune_plan_preserves_exact_source_contract(self):
        evidence = (
            "Other Kerberos implementations also put user identification data "
            "in the auth-data field."
        )
        value = plan(
            "Other Kerberos implementations",
            "also put",
            "user identification data in the auth-data field",
            evidence,
        )
        result = prune_plan(value, evidence, [], self.nlp)
        self.assertTrue(result["eligible"])
        self.assertTrue(result["checks"]["base_plan"])
        self.assertTrue(result["checks"]["dependency_prune"])


class ManifestTests(unittest.TestCase):
    def test_prepare_freezes_zero_new_planning_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare(Path(directory))
            self.assertEqual(len(manifest["sources"]), 5)
            self.assertEqual(manifest["protocol"]["api_cap"], 11)
            self.assertEqual(
                manifest["protocol"]["input"]["new_semantic_planning_calls"],
                0,
            )
            self.assertTrue(
                manifest["protocol"]["pruner"][
                    "explicit_second_person_rejection"
                ]
            )


if __name__ == "__main__":
    unittest.main()
