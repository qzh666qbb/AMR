"""Network-free tests for the SWAN length-gate tradeoff diagnostic."""

from __future__ import annotations

from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import analyze_swan_length_gate_tradeoff as experiment


def fixture(
    *,
    word_count: int,
    gate_pass: bool,
    evaluated: bool,
    score: float,
) -> dict:
    allocation_id = "dslength2-pilot-000"
    slot = 0
    attempt = 3
    return {
        "call_id": f"{allocation_id}-slot-00-attempt-03",
        "allocation_id": allocation_id,
        "slot": slot,
        "attempt": attempt,
        "template_index": 7,
        "template": "(a / N)",
        "template_sha256": "target",
        "candidate": "A candidate sentence.",
        "candidate_sha256": "candidate",
        "candidate_template_amr": "(b / N)",
        "candidate_template_amr_sha256": "candidate-amr",
        "candidate_gate": {
            "word_count": word_count,
            "word_range_pass": 7 <= word_count <= 12,
            "sentence_units": 1,
            "single_sentence_pass": True,
            "terminal_punctuation_pass": True,
            "candidate_gate_pass": gate_pass,
        },
        "threshold": 0.75,
        "score_seed": (
            experiment.swan.score_seed(allocation_id, slot, attempt)
            if evaluated
            else None
        ),
        "score": score if evaluated else 0.0,
        "accepted": bool(evaluated and score >= 0.75),
        "s2match_evaluated": evaluated,
    }


def ten_triples(_source: str) -> dict:
    return {
        "instances": 5,
        "attributes": 1,
        "relations": 4,
        "triples": 10,
    }


class AttemptTests(unittest.TestCase):
    def test_gate_rejection_zero_is_not_treated_as_a_real_score(self):
        row = fixture(
            word_count=13,
            gate_pass=False,
            evaluated=False,
            score=0.0,
        )

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Network access attempted")

        with (
            patch.object(socket, "socket", forbidden),
            patch.object(experiment, "sha256", return_value="trace"),
        ):
            result = experiment.enrich_attempt(
                experiment.ROOT / "fixture.json",
                row,
                lambda *_args: (0.8, 0.8, 0.8),
                ten_triples,
            )
        self.assertIsNone(result["historical_score"])
        self.assertTrue(result["gate_rejected_threshold_pass"])
        self.assertEqual(result["length_class"], "over_12")

    def test_historical_score_and_seed_reproduce(self):
        row = fixture(
            word_count=8,
            gate_pass=True,
            evaluated=True,
            score=0.75,
        )
        with patch.object(experiment, "sha256", return_value="trace"):
            result = experiment.enrich_attempt(
                experiment.ROOT / "fixture.json",
                row,
                lambda *_args: (0.75, 0.75, 0.75),
                ten_triples,
            )
        self.assertEqual(result["score_delta_from_history"], 0.0)
        self.assertTrue(result["historical_accepted"])
        self.assertTrue(result["threshold_pass"])

    def test_historical_score_mismatch_is_fatal(self):
        row = fixture(
            word_count=8,
            gate_pass=True,
            evaluated=True,
            score=0.75,
        )
        with (
            patch.object(experiment, "sha256", return_value="trace"),
            self.assertRaisesRegex(ValueError, "score mismatch"),
        ):
            experiment.enrich_attempt(
                experiment.ROOT / "fixture.json",
                row,
                lambda *_args: (0.7, 0.7, 0.7),
                ten_triples,
            )

    def test_invalid_gate_reconstruction_is_fatal(self):
        row = fixture(
            word_count=13,
            gate_pass=True,
            evaluated=True,
            score=0.75,
        )
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            experiment.enrich_attempt(
                experiment.ROOT / "fixture.json",
                row,
                lambda *_args: (0.75, 0.75, 0.75),
                ten_triples,
            )


class StatisticsTests(unittest.TestCase):
    def test_tied_average_ranks_and_spearman(self):
        self.assertEqual(
            experiment.average_ranks([1, 1, 3]),
            [1.5, 1.5, 3.0],
        )
        rows = [
            {"x": 1, "y": 3},
            {"x": 2, "y": 2},
            {"x": 3, "y": 1},
        ]
        rho = experiment.spearman(
            rows,
            lambda row: row["x"],
            lambda row: row["y"],
        )
        self.assertEqual(rho, -1.0)

    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(experiment.quantile([0, 10], 0.25), 2.5)
        self.assertEqual(experiment.quantile([0, 10], 0.5), 5.0)

    def test_length_classes_are_frozen(self):
        self.assertEqual(experiment.length_class(6), "under_7")
        self.assertEqual(experiment.length_class(7), "target_7_12")
        self.assertEqual(experiment.length_class(12), "target_7_12")
        self.assertEqual(experiment.length_class(13), "over_12")

    def test_slot_opportunity_uses_only_the_first_threshold_pass(self):
        rows = [
            {
                "allocation_id": "a",
                "slot": 0,
                "attempt": 0,
                "call_id": "a-0",
                "threshold_pass": False,
                "historical_accepted": False,
                "candidate_gate_pass": True,
                "word_count": 8,
                "f1": 0.7,
            },
            {
                "allocation_id": "a",
                "slot": 0,
                "attempt": 1,
                "call_id": "a-1",
                "threshold_pass": True,
                "historical_accepted": False,
                "candidate_gate_pass": False,
                "word_count": 6,
                "f1": 0.8,
            },
            {
                "allocation_id": "a",
                "slot": 0,
                "attempt": 2,
                "call_id": "a-2",
                "threshold_pass": True,
                "historical_accepted": False,
                "candidate_gate_pass": False,
                "word_count": 5,
                "f1": 0.9,
            },
        ]
        result = experiment.observed_slot_opportunities(rows)
        self.assertEqual(
            result[
                "slots_whose_first_observed_threshold_pass_was_gate_rejected"
            ],
            1,
        )
        self.assertEqual(
            result["recovered_slot_prefix_details"][0][
                "first_threshold_pass_call_id"
            ],
            "a-1",
        )


class SourceTests(unittest.TestCase):
    def test_analysis_source_has_no_network_client(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("requests.", source)
        self.assertNotIn("urlopen(", source)
        self.assertNotIn("DeepSeekJournal", source)

    def test_completed_artifact_check_is_network_free(self):
        if not experiment.MANIFEST.is_file():
            self.skipTest("Completed diagnostic artifacts are absent")

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Network access attempted")

        with patch.object(socket, "socket", forbidden):
            result = experiment.check()
        self.assertEqual(
            result["integrity"]["candidate_attempts"],
            experiment.EXPECTED_CANDIDATES,
        )
        self.assertTrue(
            result["historical_reproduction"][
                "all_exact_within_tolerance"
            ]
        )


if __name__ == "__main__":
    unittest.main()
