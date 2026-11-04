"""Offline tests for the injection-only accepted baseline."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_swan_injection_only_baseline as baseline
from swan_final1_transform import sentence_spans, unit_texts


class CleanModule:
    def parse_amr(self, text):
        return "(a / approve-01)"

    def normalize_amr_variables(self, graph):
        return graph

    def generate_template_amr(self, graph):
        return graph


class WarningModule(CleanModule):
    def parse_amr(self, text):
        logging.getLogger(
            "amrlib.models.parse_xfm.penman_serializer"
        ).warning("bad graph")
        return super().parse_amr(text)


class SourceTests(unittest.TestCase):
    def test_source_is_ten_unique_strict_acceptances(self):
        inputs = baseline.source_inputs()
        selected = inputs["selected"]
        document = inputs["document"]
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({row["text_sha256"] for row in selected}), 10)
        self.assertTrue(all(row["target_score"] >= 0.75 for row in selected))
        self.assertEqual(
            unit_texts(document["text"], sentence_spans(document["text"])),
            [row["text"] for row in selected],
        )


class DetectionTests(unittest.TestCase):
    SOURCE = {
        "slot": 0,
        "text": "The committee approved the plan.",
        "text_sha256": (
            "unused-in-test"
        ),
        "target_template_index": 27,
        "target_score": 0.8,
    }

    def test_clean_unit_is_green(self):
        with mock.patch.object(
            baseline,
            "score_triple",
            return_value=(0.7, 0.8, 0.75),
        ):
            row = baseline.score_unit(
                CleanModule(),
                ["(a / approve-01)"],
                "doc",
                0,
                self.SOURCE,
            )
        self.assertTrue(row["green"])
        self.assertEqual(row["max_score"], 0.75)
        self.assertEqual(row["diagnostics"]["stderr"], "")

    def test_logger_warning_aborts_detection_and_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(baseline, "OUT", Path(folder)):
                with mock.patch.object(
                    baseline,
                    "score_triple",
                    return_value=(0.7, 0.8, 0.75),
                ):
                    with self.assertRaisesRegex(RuntimeError, "unit 0"):
                        baseline.score_unit(
                            WarningModule(),
                            ["(a / approve-01)"],
                            "doc",
                            0,
                            self.SOURCE,
                        )
                self.assertTrue(
                    (Path(folder) / "detection_failure_unit_00.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
