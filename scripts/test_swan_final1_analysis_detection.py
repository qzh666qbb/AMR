"""Synthetic evidence only: no network, neural inference or real S2Match calls."""
import json
import logging
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from swan_final1_analysis import (
    aggregate_vectors, asr_denominators, binary_pair, bootstrap_mean, calibrate,
    feature_bin, match_humans, missing_effect_bounds, primary_analysis, proportion,
)
from swan_final1_common import digest, freeze
from swan_final1_detection import (
    MissingOperation, OperationStore, condition_input, content_identity, freeze_plan,
    graph_properties, score_job, stage_prerequisites, unit_score, validate_score,
)
from swan_final1_transform import transform
from swan_final1_workflow import verified_transforms
from swan_reframing_statistics import extracted_official_function, z_score
from test_swan_final1_transform import TEXT, marked_syntax


def features(i, prefix="h", words=110, n=10, domain="a.invalid"):
    return {"allocation_id": f"{prefix}{i:04d}", "ordinal": i,
            "word_count": words, "num_detection_units": n, "domain": domain}


class AnalysisTests(unittest.TestCase):
    def test_matching_priority_allocation_order_no_reuse(self):
        targets = [features(1, "p"), features(0, "p")]
        human = [features(0, domain="b.invalid"), features(1, words=130),
                 features(2, words=120), features(3, words=150)]
        result = match_humans(targets, human)
        self.assertEqual(result["pairs"][0], {"target_id": "p0000", "human_ids": ["h0002", "h0001"]})
        self.assertEqual(result["pairs"][1]["human_ids"], ["h0000"])
        self.assertFalse(result["passed"])

    def test_matching_integer_count_and_coverage_gates(self):
        targets = [features(i, "p") for i in range(200)]
        result = match_humans(targets, [features(i) for i in range(400)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["human_count"], 400)
        self.assertFalse(match_humans(targets, [features(i) for i in range(399)])["passed"])
        with self.assertRaises(ValueError):
            feature_bin(features(0, words=99))
        with self.assertRaises(ValueError):
            match_humans(targets, [features(0)]*2)

    def test_conservative_threshold_ties_and_missing(self):
        rows = [{"z": 0.0}]*395 + [{"z": 3.0}]*5
        result = calibrate(rows)
        self.assertEqual(result["threshold"], 3)
        self.assertEqual(result["calibration"]["successes"], 0)
        with self.assertRaises(ValueError):
            calibrate([{"z": math.nan}])

    def test_clopper_pearson_exact_edge_cases(self):
        row = proportion(0, 400)
        self.assertAlmostEqual(row["upper95_one_sided"], 1-.05**(1/400))
        self.assertLess(row["upper95_one_sided"], .03)
        self.assertEqual(proportion(400, 400)["upper95_one_sided"], 1)
        self.assertIsNone(proportion(0, 0)["estimate"])
        self.assertGreater(proportion(10, 400)["upper95_one_sided"], .03)

    def test_all_50_scores_and_inclusive_green_threshold(self):
        row = aggregate_vectors([[.65]+[0.0]*49, [.649]*50])
        self.assertEqual((row["N"], row["G"]), (2, 1))
        self.assertEqual(row["z"], z_score(1, 2))
        for values in ([[0]*49], [[math.nan]*50], []):
            with self.assertRaises(ValueError):
                aggregate_vectors(values)

    def test_discordance_no_discordance_and_source_bootstrap(self):
        row = binary_pair([True, True, False, False], [True, False, True, False])
        self.assertEqual((row["left_only"], row["right_only"]), (1, 1))
        self.assertEqual(row["mean"], 0)
        self.assertEqual(binary_pair([True]*5, [True]*5)["p_one_sided"], 1)
        first = bootstrap_mean([1.0, 0.0, -1.0])
        self.assertEqual(first, bootstrap_mean([1.0, 0.0, -1.0]))
        self.assertEqual(first["resamples"], 20000)

    def test_asr_full_denominator_and_unknowns(self):
        baseline = [{"allocation_id": str(i), "z": 3} for i in range(4)]
        changed = [{"allocation_id": "0", "z": 1}, {"allocation_id": "1", "z": 3}]
        row = asr_denominators(baseline, ["0", "1", "2"], changed, 2.33, 2.33)
        self.assertEqual(row["valid_ASR"]["estimate"], .25)
        self.assertEqual(row["quality_qualified_ASR"]["estimate"], 1/3)
        self.assertEqual(row["unknown_detection"], 1)
        self.assertEqual(row["confirmed_construction_or_quality_failures"], 1)
        self.assertEqual(row["bounds_over_all_baseline_detected"], [.25, .5])
        bounds = missing_effect_bounds([True, None], [None, False])
        self.assertEqual(bounds["effect_bounds"], [0, 1])

    def test_main_hypotheses_mechanism_and_fpr_branch_precedence(self):
        common = {c: [] for c in ("C0", "C1", "C2", "C3", "C4")}
        for i in range(200):
            for c in common:
                green = (3 if c == "C2" and i < 100 else
                         3 if c == "C1" and i < 120 else 8)
                common[c].append({"allocation_id": str(i), "N": 10,
                                  "z": z_score(green, 10), "green_rate": green/10,
                                  "mean_max_bank_score": .6 if green == 3 else .9})
        threshold = {c: 2.33 for c in common}
        human = {c: [{"z": 0.0}]*400 for c in common}
        with patch("swan_final1_analysis.BOOTSTRAPS", 200):
            report = primary_analysis(common, threshold, human)
            self.assertEqual(report["branch"], "A")
            self.assertEqual(report["hypotheses"]["H2"]["mean"], .5)
            human["C2"] = [{"z": 3.0}]*400
            self.assertEqual(primary_analysis(common, threshold, human)["branch"], "C")
            common["C2"][0]["N"] = 9
            with self.assertRaises(ValueError):
                primary_analysis(common, threshold, human)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.revision = {"bank": "fixture", "parser": "fixture"}
        self.bank = ["(b / N)"]*50
        self.unit = {"text": "Alice runs.", "span": [0, 11], "target_overlaps": []}
        self.module = SimpleNamespace(
            parse_amr=Mock(return_value="(a / run-01 :ARG0 (b / boy))"),
            normalize_amr_variables=Mock(side_effect=lambda x: x),
            generate_template_amr=Mock(return_value="(a / V :ARG0 (b / N))"))

    def tearDown(self):
        self.temp.cleanup()

    def test_durable_operation_reuse_and_interruption(self):
        store = OperationStore(self.root)
        callback = Mock(return_value={"x": 1})
        self.assertEqual(store.operation("a", {"input": 1}, callback), {"x": 1})
        self.assertEqual(OperationStore(self.root).operation("a", {"input": 1}), {"x": 1})
        callback.assert_called_once()
        store.event("b", "start", identity={"input": 2})
        never = Mock(side_effect=AssertionError("retry"))
        with self.assertRaises(MissingOperation):
            OperationStore(self.root).operation("b", {"input": 2}, never)
        never.assert_not_called()
        with self.assertRaises(ValueError):
            OperationStore(self.root).operation("a", {"input": 2})

    def test_journal_tamper_or_partial_bytes_halts(self):
        store = OperationStore(self.root)
        store.operation("raw", {}, lambda: "fixture")
        with store.path.open("a") as f:
            f.write('{"partial":')
        with self.assertRaises(ValueError):
            OperationStore(self.root)

    def test_full_vector_cache_replays_without_parser_or_scorer(self):
        with patch("swan_final1_runtime.score_triple", return_value=(.8, .8, .8)) as score:
            row = unit_score(self.root, self.unit, self.revision, self.bank, self.module)
            self.assertEqual(score.call_count, 50)
        self.assertEqual(len(row["bank_scores"]), 50)
        self.module.parse_amr.assert_called_once()
        self.assertEqual(self.module.normalize_amr_variables.call_count, 2)
        with patch("swan_final1_runtime.score_triple", side_effect=AssertionError("scorer")):
            restored = unit_score(self.root, self.unit, self.revision, self.bank)
        self.assertEqual(restored, row)
        changed_span = {**self.unit, "span": [100, 111]}
        self.assertNotEqual(content_identity(changed_span, self.revision),
                            content_identity(self.unit, self.revision))
        self.assertNotEqual(content_identity(self.unit, self.revision),
                            content_identity(self.unit, {"bank": "changed"}))

    def test_failed_score_is_not_red_or_retried(self):
        with patch("swan_final1_runtime.score_triple", side_effect=RuntimeError("fixture")) as score:
            with self.assertRaises(MissingOperation):
                unit_score(self.root, self.unit, self.revision, self.bank, self.module)
            with self.assertRaises(MissingOperation):
                unit_score(self.root, self.unit, self.revision, self.bank, self.module)
            self.assertEqual(score.call_count, 1)

    def test_graph_and_precision_recall_validation(self):
        self.assertTrue(graph_properties("(a / N :ARG0 (b / V))")["connected"])
        self.assertEqual(graph_properties("(a / N :ARG0 (b / V))")["nodes"], 2)
        with self.assertRaises(ValueError):
            validate_score({"precision": 1, "recall": .5, "f1": 1})

    def test_complete_pipeline_matches_official_ast_with_mock_graphs(self):
        events = []
        def parse(text):
            events.append("parse")
            return "(a / N)"
        def normalize(graph):
            events.append("normalize")
            return graph
        def template(graph):
            events.append("template")
            return graph
        namespace = {"np": SimpleNamespace(sqrt=math.sqrt), "logger": logging.getLogger("fixture"),
                     "parse_amr": parse, "normalize_amr_variables": normalize,
                     "generate_template_amr": template, "compute_s2match_score": lambda a, b: .8}
        official = extracted_official_function("detection_utils.py", "detect_amr_sentences", namespace)
        expected = official(["Alice runs.", "Bob runs."], self.bank, .65, .25)
        official_events = events[:]
        events.clear()
        module = SimpleNamespace(parse_amr=parse, normalize_amr_variables=normalize,
                                 generate_template_amr=template)
        job = {"allocation_id": "mock", "condition": "C0",
               "document_sha256": "fixture",
               "units": [self.unit, {**self.unit, "text": "Bob runs.", "span": [12, 21]}]}
        with patch("swan_final1_runtime.score_triple", return_value=(.8, .8, .8)):
            actual = score_job(self.root, job, self.revision, self.bank, module)
        self.assertEqual(events, official_events)
        self.assertEqual(actual["z"], expected)

    def test_gate_order_prevents_condition_detection(self):
        plan = {"matching_pass": True}
        with patch("swan_final1_detection.require_stage", return_value=[]), \
             patch("swan_final1_detection.baseline_gate", return_value={"passed": False}):
            with self.assertRaises(PermissionError):
                stage_prerequisites(self.root, "positive", plan, {}, [])
        with patch("swan_final1_detection.require_stage", side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                stage_prerequisites(self.root, "gate_b_positive", plan, {}, [])
        with self.assertRaises(PermissionError):
            stage_prerequisites(self.root, "gate_b_calibration", {"matching_pass": False}, {}, [])
        (self.root / "detection").mkdir()
        with self.assertRaises(PermissionError):
            freeze_plan(self.root)

    def test_target_mapping_retains_all_overlapping_source_slots(self):
        base = features(0, "p")
        row = {"text": "Alice runs; Bob runs.", "segmentation_spans": [[0, 21]]}
        targets = [{"slot": 0, "span": [0, 11], "bank_indices": [0]},
                   {"slot": 1, "span": [12, 21], "bank_indices": [1]}]
        result = condition_input(base, row, "C2", targets)
        self.assertEqual([r["slot"] for r in result["units"][0]["target_overlaps"]], [0, 1])

    def test_selected_transform_must_replay_first_pair(self):
        evidence = {}
        def syntax(text):
            evidence[text] = marked_syntax(text)
            return evidence[text]
        base = {"allocation_id": "mock", "text": TEXT}
        row = {"allocation_id": "mock", "baseline_sha256": digest(base),
               **transform(TEXT, syntax), "syntax_evidence": evidence}
        path = self.root / "transforms/main.json"
        freeze(path, [row])
        self.assertEqual(verified_transforms(self.root, "main", [base]), [row])
        row["selected_pair"]["merge"]["position"] += 1
        path.write_text(json.dumps([row]))
        with self.assertRaises(ValueError):
            verified_transforms(self.root, "main", [base])


if __name__ == "__main__":
    unittest.main()
