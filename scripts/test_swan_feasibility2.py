"""Offline tests for the normal-length feasibility2 experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swan_anchor1_sources import identity
from swan_feasibility2_budget import MAX_RESERVE, PILOT_CALLS, PILOT_CAP, budget_estimate
from swan_feasibility2_common import freeze
from swan_feasibility2_generation import SLOTS, topic_context
from swan_feasibility2_sources import (
    HUMAN_RANGE,
    PILOT_COUNT,
    POSITIVE_RANGE,
    human_locators,
    select,
)
import run_swan_normal_length_feasibility2 as runner


def sentence(prefix, index, words=12):
    middle = " ".join(f"{prefix}{index}word{part}" for part in range(words - 2))
    return f"Reporter {middle} confirmed."


def article(prefix):
    return " ".join(sentence(prefix, index) for index in range(10))


class SourceTests(unittest.TestCase):
    def test_source_ranges_and_control_disjointness(self):
        self.assertEqual(POSITIVE_RANGE, ("train", 1_600_000, 1_650_000))
        self.assertEqual(HUMAN_RANGE, ("train", 1_650_000, 1_650_800))
        controls = human_locators()
        self.assertEqual(len(controls), 800)
        self.assertEqual(controls[0]["row"], 1_650_000)
        self.assertEqual(controls[-1]["row"], 1_650_799)
        self.assertNotIn("source_text", controls[0])

    def test_selection_excludes_historical_identity(self):
        old = article("old")
        rows = [
            ("train", 2_200_000, {"text": old, "url": "https://old.test/a"}),
            ("train", 2_200_001, {
                "text": article("fresh"), "url": "https://fresh.test/a"
            }),
        ]
        selected, screened = select(rows, {
            "normalized_text_sha256": [identity(old)],
            "url_sha256": [],
        }, count=1)
        self.assertFalse(screened[0]["selected"])
        self.assertEqual(selected[0]["allocation_id"], "feasibility2-pilot-0000")
        self.assertEqual(selected[0]["row"], 2_200_001)


class GenerationDesignTests(unittest.TestCase):
    def test_fixed_ten_slots_and_bounded_context(self):
        self.assertEqual(SLOTS, 10)
        source = {"source_anchors": ["A source sentence about a city council vote."]}
        context = topic_context(source, ["first.", "second.", "third.", "fourth."])
        self.assertNotIn("first.", context)
        self.assertIn("second.", context)
        self.assertIn("fourth.", context)
        self.assertIn("target AMR is the primary semantic constraint", context)
        self.assertIn("10 to 22 words", context)


class BudgetAndAuthorizationTests(unittest.TestCase):
    def test_exact_frozen_cap(self):
        self.assertEqual(PILOT_COUNT, 20)
        self.assertEqual(PILOT_CALLS, 10_001)
        self.assertEqual(PILOT_CAP, (10_001, 10_001 * MAX_RESERVE))
        self.assertEqual(budget_estimate()["pilot"]["max_cost_usd"], 28.2508248)

    def test_authorization_requires_rotation_and_exact_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            freeze(folder / "manifest.json", {
                "frozen_at": "2026-09-20T00:00:00+00:00"
            })
            freeze(folder / "budget_estimate.json", {"fixture": True})
            with patch.object(runner, "check", return_value={}):
                with self.assertRaises(PermissionError):
                    runner.authorize(
                        folder, "researcher", "approve",
                        PILOT_CAP[0], str(PILOT_CAP[1] / 1e9), False,
                    )
                with self.assertRaises(PermissionError):
                    runner.authorize(
                        folder, "researcher", "approve",
                        PILOT_CAP[0] - 1, str(PILOT_CAP[1] / 1e9), True,
                    )

    def test_run_stops_before_key_without_authorization(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "check", return_value={}):
            with self.assertRaises(PermissionError):
                runner.run_phase(Path(directory), 50)


if __name__ == "__main__":
    unittest.main()
