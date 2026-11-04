"""Engineering checks only: handwritten graphs are not performance evidence."""
from dataclasses import replace
import math
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from src.core.embedding_amr import (
    EmbeddingAMRDetector, FusionConfig, GraphObservation, _fact_graph,
    _semantic_graph, digest, same_fact, windows,
)
from src.core.embedding_amr_evaluation import decision, evaluate, fit, validate_rows
from src.core.embedding_amr_runtime import atomic, file_sha, sealed_scores
from src.core.multiview_candidates import WindowConfig

GRAPH = ('(l / lend-01 :ARG0 (a / person :name (n / name :op1 "Alice")) '
         ':ARG1 (b / book) :ARG2 (j / person :name (m / name :op1 "Bob")))')
SWAPPED = GRAPH.replace('"Alice"', '"TEMP"').replace('"Bob"', '"Alice"').replace('"TEMP"', '"Bob"')


class Encoder:
    fingerprint = "synthetic-constant-test-only"

    def fits(self, text):
        return True

    def encode(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32) * 3


class Scorer:
    fingerprint = "test-only-bank"

    def __init__(self, score=.9):
        self.value, self.calls = score, []

    def score_amr(self, amr):
        self.calls.append(amr)
        return SimpleNamespace(score=self.value, template_index=0)


def observations(text, graphs):
    candidates, _ = windows(text)
    return [GraphObservation(digest(c["text"]), graphs[c["text"]], "parser", True)
            for c in candidates]


def fact(graph, anchor):
    return _fact_graph(_semantic_graph(graph, 256), anchor)


class FusionTests(unittest.TestCase):
    def detector(self, scorer=None, encoder=None, config=FusionConfig()):
        return EmbeddingAMRDetector(encoder or Encoder(), scorer or Scorer(), "parser", config)

    def test_role_swap_negation_number_scope_rejected(self):
        self.assertFalse(same_fact(fact(GRAPH, "l"), fact(SWAPPED, "l")))
        negative = GRAPH[:-1] + " :polarity -)"
        self.assertFalse(same_fact(fact(GRAPH, "l"), fact(negative, "l")))
        for other in (
            GRAPH.replace("(b / book)", "(b / book :quant 2)"),
            "(w / want-01 :ARG1 " + GRAPH + ")",
        ):
            self.assertFalse(same_fact(fact(GRAPH, "l"), fact(other, "l")))

    def test_inverse_roles_and_renaming_preserve_fact(self):
        a = "(l / lend-01 :ARG0 (p / person) :ARG1 (b / book))"
        b = "(x / person :ARG0-of (v / lend-01 :ARG1 (o / book)))"
        self.assertTrue(same_fact(fact(a, "l"), fact(b, "v")))

    def test_reentrancy_and_distinct_people_differ(self):
        a = "(l / love-01 :ARG0 (p / person) :ARG1 p)"
        b = "(l / love-01 :ARG0 (p / person) :ARG1 (q / person))"
        self.assertFalse(same_fact(fact(a, "l"), fact(b, "l")))

    def test_role_edge_multiplicity_is_preserved(self):
        a = "(l / love-01 :ARG0 (p / person) :ARG1 p)"
        b = "(l / love-01 :ARG0 (p / person) :ARG2 p)"
        self.assertFalse(same_fact(fact(a, "l"), fact(b, "l")))

    def test_high_similarity_does_not_override_relation_check(self):
        text = "Alice lends a book. Bob lends a book."
        graphs = {text: "(t / thing)", "Alice lends a book.": GRAPH, "Bob lends a book.": SWAPPED}
        result = self.detector().detect(text, observations(text, graphs))
        self.assertTrue(any(not pair["relation_supported"] for pair in result["pairs"]))
        self.assertTrue(any(pair["relation_supported"] for pair in result["pairs"]))
        self.assertTrue(all(pair["similarity"] > .99 for pair in result["pairs"]))
        self.assertIsNone(result["arms"]["embedding_amr"]["is_watermarked"])

    def test_duplicate_facts_and_merged_graph_count_once(self):
        text = "Alice lends a book. A book is lent by Alice."
        merged = "(a0 / and :op1 " + GRAPH + ")"
        graphs = {text: merged, "Alice lends a book.": GRAPH, "A book is lent by Alice.": GRAPH}
        result = self.detector().detect(text, observations(text, graphs))
        for arm in result["arms"].values():
            self.assertEqual(arm["evidence_count"], 1)
            self.assertAlmostEqual(arm["statistic"], .25)
        self.assertEqual(result, self.detector().detect(text, observations(text, graphs)))

    def test_bank_changes_do_not_change_selection(self):
        text = "Alice lends a book."
        obs = observations(text, {text: GRAPH})
        low, high = self.detector(Scorer(.1)).detect(text, obs), self.detector(Scorer(1)).detect(text, obs)
        self.assertEqual(low["pairs"], high["pairs"])
        for name in low["arms"]:
            self.assertEqual([e["candidate_index"] for e in low["arms"][name]["evidence"]],
                             [e["candidate_index"] for e in high["arms"][name]["evidence"]])
        self.assertLess(low["arms"]["embedding_amr"]["statistic"], 0)

    def test_multisentence_scaffold_does_not_create_new_fact(self):
        renamed = GRAPH.replace("l / lend", "z / lend").replace("a / person", "a2 / person").replace(
            "n / name", "n2 / name").replace("b / book", "b2 / book").replace(
            "j / person", "j2 / person").replace("m / name", "m2 / name")
        merged = "(s / multi-sentence :snt1 " + GRAPH + " :snt2 " + renamed + ")"
        self.assertTrue(same_fact(fact(merged, "l"), fact(GRAPH, "l")))
        text = "Alice lends a book. A book is lent by Alice."
        graphs = {text: merged, "Alice lends a book.": GRAPH, "A book is lent by Alice.": GRAPH}
        result = self.detector().detect(text, observations(text, graphs))
        self.assertEqual(result["arms"]["embedding_amr"]["evidence_count"], 1)

    def test_large_finite_vectors_normalize_without_overflow(self):
        class LargeEncoder(Encoder):
            def encode(self, texts):
                return np.full((len(texts), 2), 1e308)
        text = "Alice lends a book."
        result = self.detector(encoder=LargeEncoder()).detect(text, observations(text, {text: GRAPH}))
        self.assertAlmostEqual(result["pairs"][0]["similarity"], 1)

    def test_score_uses_destination_graph_not_query_graph(self):
        text = "Alice lends a book. Bob lends a book."
        graphs = {text: "(t / thing)", "Alice lends a book.": GRAPH, "Bob lends a book.": SWAPPED}
        scorer = Scorer()
        result = self.detector(scorer).detect(text, observations(text, graphs))
        for evidence in result["arms"]["embedding_amr"]["evidence"]:
            ci = evidence["candidate_index"]
            self.assertEqual(evidence["graph_body_sha256"], digest(observations(text, graphs)[ci].amr))
        self.assertNotIn("(t / thing)", scorer.calls)

    def test_parse_failure_never_calls_bank(self):
        text = "Alice lends a book."
        obs = observations(text, {text: GRAPH})
        obs[0] = replace(obs[0], usable=False, reasons=("parser_warning",))
        scorer = Scorer()
        result = self.detector(scorer).detect(text, obs)
        self.assertEqual(scorer.calls, [])
        self.assertEqual(result["candidate_count"], 1)
        self.assertIsNone(result["arms"]["embedding_amr"]["statistic"])

    def test_normalization_unavailable_candidate_is_excluded_not_fatal(self):
        text = "Alice lends a book."

        class UnavailableScorer(Scorer):
            def score_amr(self, amr):
                self.calls.append(amr)
                return None

        scorer = UnavailableScorer()
        result = self.detector(scorer).detect(text, observations(text, {text: GRAPH}))
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(result["candidate_count"], 1)
        self.assertIsNone(result["arms"]["embedding_amr"]["statistic"])
        self.assertEqual(result["arms"]["embedding_amr"]["evidence_count"], 0)
        self.assertIn("official_normalization_unavailable",
                      {reason for item in result["exclusions"] for reason in item["reasons"]})

    def test_all_nonmatching_bank_stays_negative_despite_embedding(self):
        text = "Alice lends a book."
        result = self.detector(Scorer(0)).detect(text, observations(text, {text: GRAPH}))
        self.assertLess(result["arms"]["embedding_amr"]["statistic"], 0)
        self.assertIsNone(result["arms"]["embedding_amr"]["is_watermarked"])

    def test_sha_and_full_denominator_guards(self):
        text = "Alice lends a book."
        obs = observations(text, {text: GRAPH})
        with self.assertRaisesRegex(ValueError, "denominator"):
            self.detector().detect(text, [])
        with self.assertRaisesRegex(ValueError, "binding"):
            self.detector().detect(text, [replace(obs[0], input_sha256="wrong")])

    def test_nan_vector_and_score_fail_closed(self):
        class BadEncoder(Encoder):
            def encode(self, texts):
                return np.full((len(texts), 2), np.nan)
        text = "Alice lends a book."
        obs = observations(text, {text: GRAPH})
        with self.assertRaisesRegex(ValueError, "vectors"):
            self.detector(encoder=BadEncoder()).detect(text, obs)
        with self.assertRaisesRegex(RuntimeError, "Bank"):
            self.detector(Scorer(math.nan)).detect(text, obs)

    def test_context_overflow_shared_erasure(self):
        class LongEncoder(Encoder):
            def fits(self, text):
                return False
        text = "Alice lends a book."
        result = self.detector(encoder=LongEncoder()).detect(text, observations(text, {text: GRAPH}))
        self.assertEqual(result["usable_candidates"], 0)
        self.assertTrue(all(a["statistic"] is None for a in result["arms"].values()))

    def test_event_cap_never_releases_partial(self):
        text = "Alice lends a book. Bob lends a book."
        graphs = {text: "(t / thing)", "Alice lends a book.": GRAPH, "Bob lends a book.": SWAPPED}
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            self.detector(config=FusionConfig(max_events=1)).detect(text, observations(text, graphs))

    def test_no_original_reference_api_and_budget(self):
        text = " ".join("token" + str(i) for i in range(200))
        candidates, budget = windows(text, FusionConfig(candidate_budget=5), WindowConfig())
        self.assertEqual(len(candidates), 5)
        self.assertGreater(budget["budget_omitted"], 0)
        self.assertTrue(all(text[c["start_char"]:c["end_char"]] == c["text"] for c in candidates))


def score_row(i, split="calibration", kind="human", condition="clean", value=.1):
    return {
        "row_id": f"{split}-{kind}-{i}-{condition}", "source_id": f"{split}-{kind}-{i}",
        "source_sha256": digest(f"{split}-{kind}-{i}"), "condition": condition,
        "kind": kind, "split": split, "text_sha256": digest(f"{split}-{kind}-{i}-{condition}"),
        "detector_fingerprint": "frozen",
        "arms": {name: {"statistic": value} for name in (
            "amr_only", "embedding_amr", "embedding_without_relation")},
    }


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [score_row(i) for i in range(19)]
        self.calibration = fit(self.rows, .05)

    def test_plus_one_ties_and_abstain(self):
        self.assertEqual(decision(.2, [.1] * 19, .05)["p_value"], .05)
        self.assertEqual(decision(.1, [.1] * 19, .05)["p_value"], 1)
        self.assertFalse(decision(None, [.1] * 19, .05)["is_watermarked"])

    def test_independent_full_denominator(self):
        result = evaluate([score_row(0, "test", value=None),
                           score_row(1, "test", value=.9)], self.calibration)
        metric = result["metrics"][0]
        self.assertEqual((metric["denominator"], metric["hits"], metric["abstentions"]), (2, 1, 1))
        self.assertLess(metric["ci95_clopper_pearson"][0], .5)
        self.assertGreater(metric["ci95_clopper_pearson"][1], .5)

    def test_source_text_alias_leakage_rejected(self):
        for field in ("source_id", "source_sha256", "text_sha256"):
            row = score_row(0, "test")
            row[field] = self.rows[0][field]
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate([row], self.calibration)

    def test_calibration_positive_and_low_sample_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "human"):
            fit([score_row(i, kind="watermarked") for i in range(19)])
        with self.assertRaisesRegex(ValueError, "resolve"):
            fit(self.rows[:18])

    def test_changed_detector_and_missing_conditions_rejected(self):
        row = score_row(0, "test")
        row["detector_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            evaluate([row], self.calibration)
        rows = self.rows + [score_row(i, condition="split") for i in range(19)]
        with self.assertRaisesRegex(ValueError, "conditions"):
            evaluate([score_row(0, "test")], fit(rows))

    def test_duplicate_sources_nonfinite_and_missing_human_test(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            fit([score_row(0, value=math.nan)])
        with self.assertRaisesRegex(ValueError, "human-test"):
            evaluate([score_row(0, "test", kind="watermarked")], self.calibration)

    def test_same_text_under_new_source_id_is_not_independent(self):
        rows = self.rows + [score_row(20)]
        rows[-1]["text_sha256"] = rows[0]["text_sha256"]
        with self.assertRaisesRegex(ValueError, "Same text"):
            fit(rows)

    def test_corrupt_calibration_nulls_fail_closed(self):
        self.calibration["nulls"]["clean"]["embedding_amr"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate([score_row(0, "test")], self.calibration)

    def test_paired_rate_difference_uses_discordant_documents(self):
        rows = [score_row(i, "test", value=.1) for i in range(20)]
        for row in rows[:8]:
            row["arms"]["embedding_amr"]["statistic"] = .9
        result = evaluate(rows, self.calibration)
        comparison = result["paired_comparisons"][0]
        self.assertEqual(comparison["fusion_only_positive"], 8)
        self.assertEqual(comparison["control_only_positive"], 0)
        self.assertAlmostEqual(comparison["rate_difference_fusion_minus_control"], .4)
        self.assertAlmostEqual(comparison["mcnemar_exact_two_sided_p"], 2 / 256)
        self.assertGreaterEqual(comparison["holm_p_all_reported_contrasts"], 2 / 256)
        low, high = comparison["difference_ci95_conservative"]
        self.assertLess(low, .4)
        self.assertGreater(high, .4)
        self.assertEqual(comparison["favorable_direction"], "negative")

    def test_source_reappearing_as_condition_text_rejected(self):
        row = score_row(0, "test")
        row["text_sha256"] = self.rows[0]["source_sha256"]
        with self.assertRaisesRegex(ValueError, "across source/text"):
            evaluate([row], self.calibration)

    def test_unavailable_inputs_keep_full_denominator(self):
        for row in self.rows[:2]:
            row["text_sha256"] = digest("")
            for arm in row["arms"].values():
                arm["statistic"] = None
        cal = fit(self.rows)
        rows = [score_row(i, "test", value=None) for i in range(3)]
        for row in rows:
            row["text_sha256"] = digest("")
        result = evaluate(rows, cal)
        self.assertEqual(result["metrics"][0]["denominator"], 3)
        self.assertEqual(result["metrics"][0]["abstentions"], 3)
        rows[0]["arms"]["embedding_amr"]["statistic"] = .8
        with self.assertRaisesRegex(ValueError, "Unavailable"):
            validate_rows(rows)


class SealTests(unittest.TestCase):
    def test_modified_scores_and_replay_are_not_calibration_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            rows = [score_row(0)]
            manifest = {"row_count": 1, "detector_fingerprint": "frozen",
                        "bank_sha256": "bank", "replay_parent_manifest_sha256": None}
            audit = {"rows_scored": 1, "rows_expected": 1, "detector_exact_replay": True}
            def seal():
                for name, value in (("MANIFEST.json", manifest), ("SCORES.json", rows),
                                    ("AUDIT.json", audit), ("OBSERVATION_SEAL.json", {})):
                    atomic(run / name, value)
                atomic(run / "TERMINAL.json", {
                    "status": "score_complete", **{key: file_sha(run / name) for name, key in
                    (("MANIFEST.json", "manifest_sha256"), ("SCORES.json", "scores_sha256"),
                     ("AUDIT.json", "audit_sha256"), ("OBSERVATION_SEAL.json", "observations_sha256"))}})
            seal()
            self.assertEqual(sealed_scores(run / "SCORES.json"), rows)
            (run / "SCORES.json").write_text(json.dumps([]))
            with self.assertRaisesRegex(ValueError, "seal"):
                sealed_scores(run / "SCORES.json")
            manifest["replay_parent_manifest_sha256"] = "parent"
            seal()
            with self.assertRaisesRegex(ValueError, "fresh"):
                sealed_scores(run / "SCORES.json")
            manifest["replay_parent_manifest_sha256"] = None
            rows[0]["kind"] = "watermarked"
            seal()
            with self.assertRaisesRegex(ValueError, "Bank"):
                sealed_scores(run / "SCORES.json")


if __name__ == "__main__":
    unittest.main()
