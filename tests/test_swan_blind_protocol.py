"""Engineering fixtures only; no AMR inference or empirical detection evaluation."""
import ast
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import types
import unittest

from src.core.swan_blind_protocol import (
    Calibration, CandidateScores, METHODS, ProtocolConfig, aggregate_document,
    official_two_stage_summary, plan_document, summarize_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SHA = "a" * 64


def frozen_config(method="S8", **kwargs):
    return ProtocolConfig(
        method=method, bank_sha256=FIXTURE_SHA, parser_sha256=FIXTURE_SHA,
        parser_tokenizer_sha256=FIXTURE_SHA, normalizer_sha256=FIXTURE_SHA,
        matcher_sha256=FIXTURE_SHA, parser_context_tokens=64, **kwargs)


def fixture_records(plan, value=0.8):
    return [CandidateScores(
        c.candidate_id, (value,) * plan.config.bank_size,
        plan.configuration_sha256, plan.text_sha256, native_input_tokens=10)
        for c in plan.candidates]


def fixture_calibration(plan, **kwargs):
    args = dict(configuration_sha256=plan.configuration_sha256,
                negative_manifest_sha256=FIXTURE_SHA,
                unit_threshold=0.65 if plan.config.method == "P0" else 0.75,
                document_threshold=1.0 if plan.config.method == "P0" else 0.5,
                document_statistic="z" if plan.config.method == "P0" else "green_fraction")
    return Calibration(**(args | kwargs))


class GeometryTests(unittest.TestCase):
    def test_only_text_and_config_are_inputs(self):
        self.assertEqual(list(inspect.signature(plan_document).parameters), ["text", "config"])
        with self.assertRaises(TypeError):
            plan_document("A asks B.", ProtocolConfig("S8"), slots=[])
        with self.assertRaises(TypeError):
            plan_document({"text": "A asks B.", "template_index": 0}, ProtocolConfig("S8"))

    def test_full_windows_beyond_old_128_limit(self):
        text = " \t" + " \n".join(f"w{i}" for i in range(180)) + " \t"
        for method in METHODS[1:]:
            with self.subTest(method=method):
                length = int(method[1:])
                plan = plan_document(text, ProtocolConfig(method, batch_size=7))
                self.assertEqual(len(plan.candidates), 180 - length + 1)
                self.assertGreater(len(plan.candidates), 128)
                self.assertEqual(plan.candidates[-1].text.split()[-1], "w179")
                self.assertEqual(plan.uncovered_nonwhitespace, 0)
                self.assertEqual([c for batch in plan.batches() for c in batch],
                                 list(plan.candidates))

    def test_short_and_empty_text(self):
        for method in METHODS:
            with self.subTest(method=method):
                plan = plan_document(" \tAlice asks Bob.\n", ProtocolConfig(method))
                self.assertEqual(len(plan.candidates), 1)
                self.assertEqual(plan.uncovered_nonwhitespace, 0)
                empty = plan_document(" \t\n", ProtocolConfig(method))
                self.assertEqual(empty.reasons, ("empty_text",))
                self.assertEqual(empty.batches(), ())

    def test_budget_abstains_without_truncating(self):
        text = " ".join(f"w{i}" for i in range(50))
        plan = plan_document(text, ProtocolConfig("S6", candidate_budget=10, batch_size=4))
        self.assertEqual(len(plan.candidates), 45)
        self.assertEqual(plan.status, "abstain")
        self.assertEqual(plan.batches(), ())
        result = aggregate_document(plan, [])
        self.assertEqual(result["N_expected"], 45)
        self.assertEqual(result["reasons"], ["candidate_budget_exceeded"])
        self.assertTrue(result["include_in_document_denominator"])

    def test_p0_matches_real_nltk_full_text(self):
        import nltk
        text = "Dr. Smith asks Bob. Bob leaves. Carol waits. Dan smiles. Eve runs. Finn rests."
        plan = plan_document(text, ProtocolConfig("P0"))
        self.assertEqual([c.text for c in plan.candidates], nltk.sent_tokenize(text))
        self.assertEqual(len(plan.candidates), 6)  # No human-only five-sentence truncation.
        for suffix in (";", ",", ""):
            attacked = f"Alice asks Bob{suffix} Carol leaves{suffix}"
            self.assertEqual(len(plan_document(attacked, ProtocolConfig("P0")).candidates), 1)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_document(" ".join(f"w{i}" for i in range(12)), frozen_config())
        self.records = fixture_records(self.plan)
        self.calibration = fixture_calibration(self.plan)

    def test_missing_calibration_never_decides(self):
        result = aggregate_document(self.plan, self.records)
        self.assertEqual(result["status"], "uncalibrated")
        self.assertIsNone(result["G"])
        self.assertIsNone(result["decision"])
        unfrozen = plan_document("A asks B.", ProtocolConfig("S8"))
        result = aggregate_document(unfrozen, fixture_records(unfrozen))
        self.assertEqual(result["status"], "execution_unfrozen")
        self.assertIsNone(result["decision"])

    def test_complete_bank_max_fraction_and_strict_document_threshold(self):
        records = [replace(r, bank_scores=(0.0,) * 99 + (0.75 if i < 2 else 0.1,))
                   for i, r in enumerate(self.records)]
        result = aggregate_document(self.plan, records, self.calibration)
        self.assertEqual(result["G"], 2)  # Unit comparison is inclusive.
        self.assertEqual(result["green_fraction"], 2 / 5)
        self.assertIsNone(result["z"])  # Overlapping windows have no binomial z.
        cal = replace(self.calibration, document_threshold=2 / 5)
        self.assertFalse(aggregate_document(self.plan, records, cal)["decision"])
        self.assertTrue(aggregate_document(self.plan, self.records, cal)["decision"])

    def test_numpy_scores_remain_json_serializable_and_counted(self):
        import numpy as np
        records = [replace(r, bank_scores=tuple(np.float64(v) for v in r.bank_scores))
                   for r in self.records]
        cal = replace(self.calibration, unit_threshold=np.float64(0.75),
                      document_threshold=np.float64(0.5))
        result = aggregate_document(self.plan, records, cal)
        self.assertIs(result["decision"], True)
        self.assertEqual(summarize_decisions([result])["positive_decisions"], 1)
        json.dumps(result, allow_nan=False)
        json.dumps(official_two_stage_summary([np.float64(0.8), None]), allow_nan=False)

    def test_partial_duplicate_and_wrong_provenance_abstain(self):
        cases = [
            self.records[:-1],
            [self.records[0]] * len(self.records),
            [replace(self.records[0], text_sha256="b" * 64)] + self.records[1:],
        ]
        for records in cases:
            result = aggregate_document(self.plan, records, self.calibration)
            self.assertEqual(result["status"], "abstain")
            self.assertIsNone(result["decision"])
            self.assertIsNone(result["green_fraction"])

    def test_invalid_scores_and_parse_failure_abstain(self):
        bad = [
            replace(self.records[0], bank_scores=(0.8,) * 99),
            replace(self.records[0], bank_scores=(float("nan"),) * 100),
            replace(self.records[0], bank_scores=(float("inf"),) * 100),
            replace(self.records[0], bank_scores=(None,) * 100),
            replace(self.records[0], bank_scores=(1.1,) * 100),
            replace(self.records[0], bank_scores=(True,) * 100),
            replace(self.records[0], parse_ok=False),
            replace(self.records[0], error="parser_recovery"),
            replace(self.records[0], native_input_tokens=65),
            replace(self.records[0], native_input_tokens=None),
        ]
        for first in bad:
            result = aggregate_document(self.plan, [first] + self.records[1:], self.calibration)
            self.assertEqual(result["status"], "abstain")
            self.assertIsNone(result["decision"])

    def test_calibration_bound_to_method_and_whole_pipeline(self):
        for calibration in (
            replace(self.calibration, configuration_sha256="b" * 64),
            replace(self.calibration, document_statistic="z"),
        ):
            self.assertEqual(aggregate_document(self.plan, self.records, calibration)["reasons"],
                             ["calibration_configuration_mismatch"])
        different = plan_document(" ".join(f"w{i}" for i in range(12)), frozen_config("S10"))
        self.assertNotEqual(different.configuration_sha256, self.plan.configuration_sha256)

    def test_unconditional_denominator_includes_unresolved(self):
        decided = aggregate_document(self.plan, self.records, self.calibration)
        failed = aggregate_document(self.plan, self.records[:-1], self.calibration)
        uncalibrated = aggregate_document(self.plan, self.records)
        summary = summarize_decisions([decided, failed, uncalibrated])
        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["positive_decision_rate_all_documents"], 1 / 3)
        self.assertEqual(summary["unresolved_documents"], 2)

    def test_p0_one_sentence_and_calibration_rule(self):
        plan = plan_document("Alice asks Bob.", frozen_config("P0"))
        records = fixture_records(plan)
        raw = aggregate_document(plan, records)
        self.assertAlmostEqual(raw["z"], math.sqrt(3))
        self.assertEqual(raw["status"], "uncalibrated")
        cal = fixture_calibration(plan)
        self.assertTrue(aggregate_document(plan, records, cal)["decision"])
        bad = replace(cal, unit_threshold=0.7)
        self.assertEqual(aggregate_document(plan, records, bad)["status"], "abstain")


class OfficialCountingTests(unittest.TestCase):
    def test_n1_blank_slot_and_empty_match_pinned_official_source(self):
        # Extract just the numeric function and replace S2Match with fixture lookup.
        # No official module import, parser, vectors, or detector scoring is executed.
        import numpy as np
        source = ROOT / "third_party/SWAN_official_2ca0c207_clean/detection/detect_from_parsed_amrs.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "compute_zscore_for_doc")
        namespace = {"np": np, "logger": types.SimpleNamespace(debug=lambda *a: None),
                     "compute_s2match_score": lambda amr, bank: {"green": 0.8, "red": 0.2}[amr]}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        for slots in ([0.8], [0.8, None, 0.2], [None], []):
            amrs = ["" if s is None else "green" if s == 0.8 else "red" for s in slots]
            _, official, _ = namespace["compute_zscore_for_doc"](0, amrs, ["fixture"], 0.65, 0.25, False)
            summary = official_two_stage_summary(slots)
            self.assertEqual(summary["N"], len(slots))
            if summary["z"] is None:
                self.assertTrue(math.isnan(official))
            else:
                self.assertAlmostEqual(summary["z"], official)
        self.assertAlmostEqual(official_two_stage_summary([0.8, None])["z"], math.sqrt(2 / 3))


if __name__ == "__main__":
    unittest.main()
