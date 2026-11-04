"""Offline tests for the SWAN injection-only pilot."""
from contextlib import nullcontext
import logging
import sys
import unittest
from unittest import mock
import warnings

import run_swan_injection_only_pilot as pilot


class CleanModule:
    def parse_amr(self, text):
        return "(a / approve-01)"

    def normalize_amr_variables(self, graph):
        return graph

    def generate_template_amr(self, graph):
        return graph

    def decide_s2match_match(self, candidate, target, threshold):
        return True, 0.8


class DiagnosticModule(CleanModule):
    def parse_amr(self, text):
        logging.getLogger(pilot.PARSER_LOGGERS[0]).warning("serializer warning")
        warnings.warn("python warning", RuntimeWarning)
        print("parser stderr", file=sys.stderr)
        return super().parse_amr(text)


class FailingModule(CleanModule):
    def parse_amr(self, text):
        warnings.warn("warning before exception", RuntimeWarning)
        raise RuntimeError("candidate parse failed")


class InvalidScoreModule(CleanModule):
    def decide_s2match_match(self, candidate, target, threshold):
        return True, float("nan")


class SurfaceTests(unittest.TestCase):
    def test_accepts_one_plain_punkt_unit(self):
        result = pilot.surface_result("The committee approved the plan.")
        self.assertTrue(result["pass"])

    def test_rejects_multiple_units_and_placeholder_leak(self):
        self.assertFalse(pilot.surface_result("One passed. Two passed.")["pass"])
        self.assertFalse(pilot.surface_result("The X passed.")["pass"])


class FrozenDesignTests(unittest.TestCase):
    def test_target_schedule_repeats_exactly(self):
        self.assertEqual(
            [pilot.target_for(slot) for slot in range(20)],
            list(pilot.TEMPLATE_SCHEDULE) * 2,
        )

    def test_budget_cap(self):
        self.assertEqual(pilot.MAX_CALLS, 601)
        self.assertEqual(pilot.PILOT_CAP, (601, 1_697_704_800))
        self.assertEqual(pilot.budget()["max_cost_usd"], 1.6977048)


class DiagnosticTests(unittest.TestCase):
    def assess(self, module):
        with mock.patch.object(
            pilot.swan,
            "fixed_score_seed",
            return_value=nullcontext(),
        ):
            return pilot.score_with_diagnostics(
                module,
                "The committee approved the plan.",
                "(a / approve-01)",
                123,
            )

    def test_clean_score_is_usable(self):
        result = self.assess(CleanModule())
        self.assertTrue(result["clean"])
        self.assertEqual(result["scored"]["score"], 0.8)
        self.assertFalse(result["diagnostics_present"])

    def test_all_diagnostic_channels_reject_candidate(self):
        result = self.assess(DiagnosticModule())
        self.assertFalse(result["clean"])
        self.assertTrue(result["diagnostics_present"])
        self.assertEqual(result["scored"]["score"], 0.8)
        self.assertEqual(
            result["diagnostics"]["logged_warnings"][0]["message"],
            "serializer warning",
        )
        self.assertEqual(
            result["diagnostics"]["python_warnings"][0]["message"],
            "python warning",
        )
        self.assertEqual(result["diagnostics"]["stderr"], "parser stderr\n")

    def test_parser_exception_is_candidate_local(self):
        result = self.assess(FailingModule())
        self.assertFalse(result["clean"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(
            result["diagnostics"]["python_warnings"][0]["message"],
            "warning before exception",
        )

    def test_nonfinite_score_is_global_integrity_failure(self):
        with self.assertRaises(pilot.GlobalStop):
            self.assess(InvalidScoreModule())


if __name__ == "__main__":
    unittest.main()
