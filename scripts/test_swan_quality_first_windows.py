"""No-network tests for the quality-first natural-window preflight."""

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import run_swan_quality_first_windows as experiment
from test_swan_natural_narrative_windows import FakeNLP, tokenize


def long_candidate_text(document=0):
    rows = []
    for index in range(12):
        bridge = f"link{index // 2}"
        rows.append(
            f"Writer{document}_{index} reports {bridge} evidence{index} "
            f"detail{index} context{index} clearly{index} locally{index} "
            f"readers{index} today{index} update{index}."
        )
    return "\n".join(
        " || ".join(rows[index:index + 3])
        for index in range(0, len(rows), 3)
    )


class WindowTests(unittest.TestCase):
    def test_quality_selection_ignores_short_slot_requirement(self):
        window = experiment.find_quality_window(
            long_candidate_text(),
            tokenize,
            FakeNLP(),
        )
        self.assertTrue(window["pass"])
        self.assertFalse(any("short" in key for key in window))
        self.assertFalse(any(
            "short" in key
            for row in window["sentences"]
            for key in row
        ))
        self.assertTrue(all(row["words"] > 10 for row in window["sentences"]))

    def test_coverage_is_revealed_separately(self):
        window = experiment.find_quality_window(
            long_candidate_text(),
            tokenize,
            FakeNLP(),
        )
        source = {
            "prompt_id": "source",
            "source_sha256": "source",
            "shard_index": 2,
            "shard_row": 0,
            "window": window,
        }
        generated = experiment.build_quality_document(source)
        self.assertNotIn("short_slots", generated)
        coverage = experiment.build_coverage(generated)
        self.assertEqual(coverage["slot_count"], 0)
        self.assertFalse(coverage["coverage_pass"])


class AcquisitionTests(unittest.TestCase):
    def test_acquisition_records_zero_slot_predicate_evaluations(self):
        rows = [
            {
                "text": long_candidate_text(index),
                "url": f"https://example.test/{index}",
            }
            for index in range(3)
        ]

        def stream(_index):
            yield from rows

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(experiment, "SHARD_INDICES", (2,)),
                patch.object(experiment, "DOCUMENTS", 3),
                patch.object(
                    experiment,
                    "prior_identities",
                    return_value=(set(), set()),
                ),
                patch.object(
                    experiment.train,
                    "stream_train_shard",
                    side_effect=stream,
                ),
            ):
                sources, screening, _scanned = experiment.acquire_sources(
                    Path(directory),
                    tokenize,
                    FakeNLP(),
                )
        self.assertEqual(len(sources), 3)
        self.assertEqual(screening["short_slot_predicate_evaluations"], 0)
        self.assertTrue(all(
            "short_slot_count" not in source["window"]
            for source in sources
        ))


class GateTests(unittest.TestCase):
    def test_parser_stops_below_joint_quality_coverage_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sources = []
            for index in range(2):
                prompt_id = f"source-{index}"
                sources.append({"prompt_id": prompt_id})
                review_path = folder / "reviews" / f"{prompt_id}.json"
                coverage_path = folder / "coverage" / f"{prompt_id}.json"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                coverage_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(json.dumps({"quality_pass": True}))
                coverage_path.write_text(json.dumps({"coverage_pass": True}))
            manifest = {"sources": sources}
            result = experiment.parse_eligible_documents(
                folder,
                manifest,
                {},
                None,
            )
        self.assertEqual(result, 0)

    def test_quality_review_body_has_no_coverage_metadata(self):
        body = experiment.mixed.style_body("A frozen passage.")
        serialized = json.dumps(body, sort_keys=True)
        self.assertNotIn("short_slots", serialized)
        self.assertNotIn("coverage", serialized)
        self.assertNotIn("sentence_words", serialized)

    def test_protocol_uses_independent_shards_and_fixed_budget(self):
        self.assertEqual(experiment.SHARD_INDICES, (2, 3, 4, 5))
        self.assertEqual(experiment.DOCUMENTS, 30)
        self.assertEqual(experiment.API_CAP, 31)
        self.assertEqual(experiment.MIN_JOINT_COVERAGE_DOCUMENTS, 3)


if __name__ == "__main__":
    unittest.main()
