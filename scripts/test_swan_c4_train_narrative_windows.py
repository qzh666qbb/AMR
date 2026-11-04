"""No-network tests for the held-out C4-train narrative preflight."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import run_swan_c4_train_narrative_windows as experiment
from test_swan_natural_narrative_windows import (
    FakeNLP,
    candidate_text,
    tokenize,
)


def row(text, index):
    return {
        "text": text,
        "url": f"https://example.test/{index}",
    }


class AcquisitionTests(unittest.TestCase):
    def test_excludes_upstream_prefix_and_selects_in_order(self):
        shard = [
            row("Too short.", 0),
            row("Too short.", 1),
        ]
        shard.extend(
            row(candidate_text(index), index + 2)
            for index in range(experiment.DOCUMENTS)
        )

        def stream(_index):
            yield from shard

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(experiment, "SHARD_INDICES", (0,)),
                patch.object(experiment, "FIRST_SHARD_SKIP", 2),
                patch.object(
                    experiment,
                    "prior_identities",
                    return_value=(set(), set()),
                ),
                patch.object(
                    experiment,
                    "stream_train_shard",
                    side_effect=stream,
                ),
            ):
                sources, screening, scanned = experiment.acquire_sources(
                    Path(directory),
                    tokenize,
                    FakeNLP(),
                )
        self.assertEqual(len(sources), experiment.DOCUMENTS)
        self.assertEqual(sources[0]["shard_row"], 2)
        self.assertEqual(scanned, {"0": 6})
        self.assertEqual(screening["excluded_prefix_rows"], 2)
        self.assertEqual(
            screening["eligible_rows_scanned"],
            experiment.DOCUMENTS,
        )

    def test_scans_across_shards_in_order(self):
        shards = {
            0: [row("Too short.", 0)],
            1: [
                row(candidate_text(index), 10 + index)
                for index in range(experiment.DOCUMENTS)
            ],
        }

        def stream(index):
            yield from shards[index]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(experiment, "SHARD_INDICES", (0, 1)),
                patch.object(experiment, "FIRST_SHARD_SKIP", 0),
                patch.object(
                    experiment,
                    "prior_identities",
                    return_value=(set(), set()),
                ),
                patch.object(
                    experiment,
                    "stream_train_shard",
                    side_effect=stream,
                ),
            ):
                sources, screening, scanned = experiment.acquire_sources(
                    Path(directory),
                    tokenize,
                    FakeNLP(),
                )
        self.assertEqual(sources[0]["shard_index"], 1)
        self.assertEqual(scanned, {"0": 0, "1": 4})
        self.assertEqual(screening["eligible_rows_scanned"], 6)

    def test_hard_row_limit_stops_before_extra_row(self):
        shard = [row("Too short.", index) for index in range(5)]

        def stream(_index):
            yield from shard

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(experiment, "SHARD_INDICES", (0,)),
                patch.object(experiment, "FIRST_SHARD_SKIP", 0),
                patch.object(experiment, "MAX_ELIGIBLE_ROWS", 3),
                patch.object(
                    experiment,
                    "prior_identities",
                    return_value=(set(), set()),
                ),
                patch.object(
                    experiment,
                    "stream_train_shard",
                    side_effect=stream,
                ),
            ):
                sources, screening, scanned = experiment.acquire_sources(
                    Path(directory),
                    tokenize,
                    FakeNLP(),
                )
        self.assertEqual(sources, [])
        self.assertEqual(screening["eligible_rows_scanned"], 3)
        self.assertEqual(scanned, {"0": 3})

    def test_failed_attempt_records_no_partial_selection(self):
        def stream(_index):
            yield row(candidate_text(0), 0)
            raise TimeoutError("frozen failure")

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (
                patch.object(experiment, "SHARD_INDICES", (0,)),
                patch.object(experiment, "FIRST_SHARD_SKIP", 0),
                patch.object(
                    experiment,
                    "prior_identities",
                    return_value=(set(), set()),
                ),
                patch.object(
                    experiment,
                    "stream_train_shard",
                    side_effect=stream,
                ),
            ):
                with self.assertRaises(TimeoutError):
                    experiment.acquire_sources(
                        folder,
                        tokenize,
                        FakeNLP(),
                    )
            journal = __import__("json").loads(
                (folder / "source_acquisition.json").read_text()
            )
        self.assertEqual(journal["attempts"][0]["status"], "failed")
        self.assertFalse(journal["attempts"][0]["manifest_written"])


class ProtocolTests(unittest.TestCase):
    def test_reuses_all_frozen_narrative_thresholds(self):
        self.assertEqual(
            experiment.narrative.MIN_SHORT_SLOTS,
            8,
        )
        self.assertEqual(experiment.narrative.MAX_SOURCE_LINES, 8)
        self.assertEqual(
            experiment.narrative.MIN_MULTI_SENTENCE_LINES,
            3,
        )
        self.assertEqual(experiment.narrative.MIN_VERBAL_ROOTS, 10)
        self.assertEqual(experiment.narrative.MAX_PAIR_JACCARD, 0.65)
        self.assertEqual(experiment.narrative.MIN_ADJACENT_OVERLAP, 4)

    def test_shard_identity_is_pinned(self):
        self.assertEqual(
            experiment.shard_name(3),
            "realnewslike/c4-train.00003-of-00512.json.gz",
        )
        self.assertEqual(experiment.FIRST_SHARD_SKIP, 5250)
        self.assertEqual(experiment.MAX_ELIGIBLE_ROWS, 100000)


if __name__ == "__main__":
    unittest.main()
