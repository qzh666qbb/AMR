"""Offline tests for the count-neutral confirmation protocol."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault(
    "NLTK_DATA",
    str(Path(__file__).resolve().parents[1] / ".nltk_data"),
)

import run_swan_count_neutral_confirmation as runner


SENTENCES = [
    "Alice carefully reviewed the report.",
    "Bob promptly approved the proposal.",
    "Carol quietly updated the schedule.",
    "Daniel clearly explained the policy.",
    "Evelyn recently inspected the equipment.",
    "Frank immediately repaired the machine.",
    "Grace thoroughly checked the records.",
    "Henry calmly answered the question.",
    "Isabel accurately measured the distance.",
    "James carefully organized the files.",
    "Karen successfully completed the survey.",
    "Louis formally announced the decision.",
    "Maria patiently trained the interns.",
    "Nathan quickly resolved the dispute.",
    "Olivia publicly supported the initiative.",
    "Peter finally delivered the package.",
]


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import nltk
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")
        cls.tokenize = staticmethod(nltk.sent_tokenize)

    def build(self):
        return runner.build_variants(
            "fixture-positive-000",
            SENTENCES,
            self.nlp,
            self.tokenize,
        )

    def test_a_b_are_exactly_count_neutral(self):
        result = self.build()
        self.assertEqual(len(result["original_sentences"]), 16)
        self.assertEqual(len(result["variants"]["A"]["units"]), 15)
        self.assertEqual(len(result["variants"]["B"]["units"]), 15)
        self.assertEqual(
            self.tokenize(result["variants"]["A"]["text"]),
            result["variants"]["A"]["units"],
        )
        self.assertEqual(
            self.tokenize(result["variants"]["B"]["text"]),
            result["variants"]["B"]["units"],
        )

    def test_lexical_case_order_and_punctuation_multiset_match(self):
        result = self.build()
        a = result["variants"]["A"]["text"]
        b = result["variants"]["B"]["text"]
        self.assertEqual(
            runner.lexical_sequence(a),
            runner.lexical_sequence(b),
        )
        self.assertEqual(
            runner.punctuation_multiset(a),
            runner.punctuation_multiset(b),
        )
        self.assertEqual(Counter(a)[";"], 1)
        self.assertEqual(Counter(b)[";"], 1)

    def test_selected_boundaries_are_nonadjacent_and_deterministic(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first["selection"], second["selection"])
        self.assertGreater(
            abs(first["selection"]["a"] - first["selection"]["b"]),
            1,
        )
        self.assertNotEqual(
            first["variants"]["A"]["text"],
            first["variants"]["B"]["text"],
        )

    def test_invalid_lexical_change_is_rejected(self):
        record = {
            "id": "fixture",
            **self.build(),
        }
        record["variants"]["B"]["text"] = (
            record["variants"]["B"]["text"].replace("Peter", "Paul")
        )
        with self.assertRaisesRegex(ValueError, "Invalid frozen"):
            runner.validate_construction(record)

    def test_detector_and_mechanism_preserve_fifteen_units(self):
        record = {
            "id": "fixture-positive-000",
            "cohort": "positive",
            "target_templates": ["(t / target)"] * 16,
            **self.build(),
        }
        module = SimpleNamespace(
            parse_amr=lambda _text: "(x / example)",
            normalize_amr_variables=lambda graph: graph,
            generate_template_amr=lambda graph: graph,
        )
        globals_ = {
            "compute_s2match_score": lambda _left, _right: 0.7,
            "compute_s2match_from_strings": (
                lambda _left, _right, **_kwargs: (1.0, 1.0, 1.0)
            ),
        }
        detection = runner.detect_record(
            record,
            module,
            globals_,
            [f"(t{i} / target)" for i in range(50)],
            {},
        )
        self.assertEqual(detection["arms"]["A"]["n"], 15)
        self.assertEqual(detection["arms"]["B"]["n"], 15)
        self.assertEqual(detection["arms"]["A"]["green"], 15)
        self.assertEqual(detection["arms"]["B"]["green"], 15)
        with tempfile.TemporaryDirectory() as directory:
            detection_path = Path(directory) / "detection.json"
            runner.write_json(detection_path, detection)
            with patch.object(
                runner,
                "detection_path",
                return_value=detection_path,
            ):
                mechanism = runner.mechanism_record(
                    record,
                    detection,
                    module,
                    globals_,
                )
        self.assertEqual(
            mechanism["resource_counts"]["new_parser_calls"],
            0,
        )
        self.assertEqual(
            mechanism["resource_counts"]["new_s2match_observations"],
            60,
        )


class CalibrationAndStatisticsTests(unittest.TestCase):
    def calibration_rows(self):
        high = runner.z_score(8)
        low = runner.z_score(3)
        rows = []
        for index in range(runner.CALIBRATION_CONTROLS):
            rows.append(
                {
                    "status": "complete",
                    "arms": {
                        "A": {"z": high if index < 3 else low},
                        "B": {"z": high if index < 2 else low},
                    },
                }
            )
        return rows

    def test_calibration_uses_only_calibration_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            rows = self.calibration_rows()
            with (
                patch.object(runner, "CALIBRATION", path),
                patch.object(
                    runner,
                    "CONTROL_MANIFEST",
                    Path(__file__),
                ),
                patch.object(
                    runner,
                    "terminal_detection_rows",
                    return_value=rows,
                ) as terminal,
            ):
                result = runner.calibrate()
        terminal.assert_called_once_with("human_calibration")
        self.assertGreater(
            result["selected_common_threshold"],
            runner.PUBLIC_Z_THRESHOLD,
        )
        selected = next(
            row
            for row in result["diagnostics"]
            if row["threshold"]
            == result["selected_common_threshold"]
        )
        self.assertTrue(selected["eligible"])

    def test_exact_directional_probability(self):
        self.assertEqual(runner.exact_directional_p(0, 0), 1.0)
        self.assertEqual(runner.exact_directional_p(3, 3), 1.0)
        self.assertAlmostEqual(
            runner.exact_directional_p(10, 0),
            2**-10,
        )

    def test_wilson_retains_full_denominator(self):
        result = runner.wilson(2, 10)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["total"], 10)
        self.assertEqual(result["rate"], 0.2)
        self.assertLess(result["ci95"][0], result["rate"])
        self.assertGreater(result["ci95"][1], result["rate"])


class SafetySurfaceTests(unittest.TestCase):
    def test_runner_has_no_credential_interface(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--key-file", source)
        self.assertNotIn("read_credential", source)
        self.assertNotIn("requests.post", source)


if __name__ == "__main__":
    unittest.main()
