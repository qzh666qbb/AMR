"""Offline tests: no network, neural parser, S2Match, detector, or paid call."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from swan_anchor1_budget import CAPS, MAX_RESERVE, Journal, budget_estimate
from swan_anchor1_common import freeze
from swan_anchor1_generation import (
    MAX_ATTEMPTS,
    run_slot,
    source_context,
    surface_screen,
    template_schedule,
    token_jaccard,
)
from swan_anchor1_sources import (
    control_locators,
    history_snapshot,
    identity,
    select,
    source_window,
)
import run_swan_source_anchored1 as runner


def sentence(index, words=12):
    body = " ".join(f"anchor{index}word{part}" for part in range(words - 2))
    return f"Reporter {body} confirmed."


def article(prefix=""):
    return " ".join(prefix + sentence(index) for index in range(10))


def syntax(text):
    words = {
        token.strip(".,!?").casefold()
        for token in text.split()
        if len(token.strip(".,!?")) > 3
    }
    return {
        "complete": bool(text),
        "capitalization_valid_mid_sentence": True,
        "content_lemmas": sorted(words),
    }


class FakeJournal:
    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.calls = []

    def request(self, call_id, user, system, max_tokens, temperature, top_p):
        self.calls.append((call_id, user, system, max_tokens, temperature, top_p))
        text = next(self.candidates)
        return {
            "text": text,
            "text_sha256": identity(text),
            "stop_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "response_metadata": {"model": "deepseek-flash"},
            "call_sha256": identity(call_id),
        }


class FakeModule:
    def __init__(self, scores):
        self.scores = iter(scores)

    @staticmethod
    def build_system_prompt():
        return "system"

    @staticmethod
    def build_user_prompt(template, examples, context=""):
        return f"{template}|{len(examples)}|{context}"

    @staticmethod
    def parse_amr(candidate):
        return f"(a / report :value {len(candidate)})"

    @staticmethod
    def normalize_amr_variables(graph):
        return graph

    @staticmethod
    def generate_template_amr(graph):
        return graph

    def decide_s2match_match(self, candidate, target, threshold):
        del candidate, target
        score = next(self.scores)
        return score >= threshold, score


class SourceTests(unittest.TestCase):
    def test_first_qualifying_ten_sentence_window(self):
        short = "Too short." * 2
        result = source_window(short + " " + article())
        self.assertIsNotNone(result)
        self.assertEqual(result["num_units"], 10)
        self.assertEqual(len(result["anchors"]), 10)
        self.assertGreaterEqual(result["word_count"], 100)
        self.assertLessEqual(result["word_count"], 220)

    def test_select_is_ordered_and_disjoint(self):
        old = article("Old ")
        rows = [
            (17, 0, {"text": old, "url": "https://example.test/old"}),
            (17, 1, {"text": article("FreshA "), "url": "https://example.test/a"}),
            (17, 2, {"text": article("FreshB "), "url": "https://example.test/b"}),
        ]
        snapshot = {
            "normalized_text_sha256": [identity(old)],
            "url_sha256": [],
        }
        selected, screened = select(
            rows, snapshot, (("pilot", 1), ("main", 1))
        )
        self.assertFalse(screened[0]["selected"])
        self.assertEqual([row["row"] for row in selected], [1, 2])
        self.assertEqual(
            [row["allocation_id"] for row in selected],
            ["anchor1-pilot-0000", "anchor1-main-0000"],
        )

    def test_human_locators_are_fixed_without_content(self):
        rows = control_locators()
        self.assertEqual(len(rows), 8000)
        self.assertEqual(rows[0]["row"], 1_700_000)
        self.assertEqual(rows[-1]["row"], 1_707_999)
        self.assertEqual(rows[3999]["phase"], "calibration")
        self.assertEqual(rows[4000]["phase"], "test")
        self.assertNotIn("source_text", rows[0])
        self.assertFalse(rows[0]["content_materialized"])

    def test_history_does_not_read_final1_sealed_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "experiments/swan_count_neutral_final1/sealed"
            sealed.mkdir(parents=True)
            (sealed / "invalid.json").write_text("{not valid json")
            (root / "api.txt").write_text("secret")
            freeze(
                root / "artifacts/readable.json",
                {"text": "A readable historical source."},
            )
            snapshot = history_snapshot(root, root / "experiments/new")
        self.assertIn(
            identity("A readable historical source."),
            snapshot["normalized_text_sha256"],
        )
        excluded = {row["path"] for row in snapshot["excluded_paths"]}
        self.assertIn(
            "experiments/swan_count_neutral_final1/sealed/invalid.json",
            excluded,
        )
        self.assertIn("api.txt", excluded)


class SurfaceTests(unittest.TestCase):
    def test_anchor_overlap_and_command_rejection(self):
        anchor = "Officials confirmed the harbor closure after the storm passed."
        good = "Officials confirmed the harbor closure after inspectors completed their review today."
        result = surface_screen(good, anchor, [], syntax)
        self.assertTrue(result["pass"])
        command = "You should confirm the harbor closure with inspectors before reporting it today."
        self.assertFalse(surface_screen(command, anchor, [], syntax)["pass"])

    def test_repetition_gate_and_context_tail(self):
        prior = "Officials confirmed the harbor closure after inspectors completed review today."
        candidate = "Officials confirmed the harbor closure after inspectors completed review today."
        self.assertEqual(token_jaccard(prior, candidate), 1)
        self.assertFalse(surface_screen(candidate, prior, [prior], syntax)["pass"])
        source = {
            "source_window": article(),
            "source_anchors": [sentence(index) for index in range(10)],
        }
        context = source_context(source, 3, ["first.", "second.", "third."])
        self.assertNotIn("first.", context)
        self.assertIn("second.", context)
        self.assertIn("third.", context)
        self.assertIn(source["source_anchors"][3], context)

    def test_template_schedule_is_deterministic_and_slot_specific(self):
        bank = list(range(50))
        first = template_schedule(bank, "anchor1-pilot-0000", 0)
        self.assertEqual(first, template_schedule(bank, "anchor1-pilot-0000", 0))
        self.assertNotEqual(first, template_schedule(bank, "anchor1-pilot-0000", 1))
        self.assertEqual(len(first), 10)
        self.assertTrue(all(0 <= value < 50 for value in first))


class SlotSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        freeze(self.folder / "manifest.json", {"fixture": "contract"})
        freeze(self.folder / "source_manifest.json", {"fixture": "sources"})
        self.source = {
            "allocation_id": "anchor1-pilot-0000",
            "source_window": article(),
            "source_anchors": [sentence(index) for index in range(10)],
        }
        self.bank = [f"(t / template-{index})" for index in range(50)]

    def tearDown(self):
        self.temp.cleanup()

    def valid(self, index):
        anchor_word = f"anchor0word{index % 10}"
        return (
            f"Reporter confirmed {anchor_word} with verified local evidence "
            f"during the public briefing number {index}."
        )

    def test_first_surface_and_score_pass_is_selected(self):
        candidates = [
            "You should confirm anchor0word0 with officials before reporting the event today.",
            self.valid(1),
        ]
        row = run_slot(
            self.folder,
            "pilot",
            self.source,
            0,
            [],
            FakeJournal(candidates),
            FakeModule([.8]),
            self.bank,
            [{}, {}, {}],
            syntax,
        )
        self.assertEqual(row["selected_attempt"], 1)
        self.assertTrue(row["accepted"])
        self.assertFalse(row["fallback"])

    def test_highest_surface_score_fallback_uses_earliest_tie(self):
        candidates = [self.valid(index) for index in range(MAX_ATTEMPTS)]
        scores = [.2] * MAX_ATTEMPTS
        scores[7] = scores[11] = .7
        row = run_slot(
            self.folder,
            "pilot",
            self.source,
            0,
            [],
            FakeJournal(candidates),
            FakeModule(scores),
            self.bank,
            [{}, {}, {}],
            syntax,
        )
        self.assertEqual(row["attempts"], 30)
        self.assertEqual(row["selected_attempt"], 7)
        self.assertFalse(row["accepted"])
        self.assertTrue(row["fallback"])


class BudgetAndAuthorizationTests(unittest.TestCase):
    def test_exact_phase_caps(self):
        self.assertEqual(CAPS["pilot"], (2401, 2401 * MAX_RESERVE))
        self.assertEqual(CAPS["main"], (150000, 150000 * MAX_RESERVE))
        estimate = budget_estimate()
        self.assertEqual(estimate["phases"]["pilot"]["max_cost_usd"], 6.7823448)
        self.assertEqual(estimate["phases"]["main"]["max_cost_usd"], 423.72)

    def test_anchor_journal_uses_anchor_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(
                Path(directory), "pilot", MAX_ATTEMPTS, Mock(), "contract"
            )
            with patch(
                "swan_anchor1_budget.CAPS",
                {"pilot": (2401, MAX_RESERVE - 1), "main": CAPS["main"]},
            ):
                self.assertEqual(journal.capacity(1), "phase_money_cap")

    def test_authorization_requires_rotation_attestation_and_exact_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            freeze(folder / "manifest.json", {"frozen_at": "2026-09-19T00:00:00+00:00"})
            freeze(folder / "budget_estimate.json", {"fixture": True})
            with patch.object(runner, "check", return_value={}):
                with self.assertRaises(PermissionError):
                    runner.authorize(
                        folder, "pilot", "person", "approve",
                        CAPS["pilot"][0], str(CAPS["pilot"][1] / 1e9), False,
                    )
                with self.assertRaises(PermissionError):
                    runner.authorize(
                        folder, "pilot", "person", "approve",
                        CAPS["pilot"][0] - 1, str(CAPS["pilot"][1] / 1e9), True,
                    )

    def test_run_stops_before_sources_or_environment_without_authorization(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "check", return_value={}), \
                patch.object(runner, "sources_for") as sources:
            with self.assertRaises(PermissionError):
                runner.run_phase(Path(directory), "pilot", 30)
            sources.assert_not_called()


if __name__ == "__main__":
    unittest.main()
