"""No-network tests for natural mixed-window screening."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from run_swan_natural_mixed_windows import (
    DOCUMENTS,
    acquire_sources,
    build_document,
    find_window,
    graph_document,
    style_gate,
)


def short_sentence(index):
    return f"Topic{index} records one distinct verified natural source fact today."


def long_sentence(index):
    return (
        f"Article{index} provides several additional details about the same "
        "documented event for readers today."
    )


def candidate_sentences(short_count=8):
    rows = [short_sentence(index) for index in range(short_count)]
    rows.extend(
        long_sentence(index)
        for index in range(12 - short_count)
    )
    return rows


class WindowTests(unittest.TestCase):
    def test_finds_contiguous_mixed_window(self):
        sentences = candidate_sentences()
        result = find_window("ignored", lambda _text: sentences)
        self.assertTrue(result["pass"])
        self.assertEqual(result["window_start"], 0)
        self.assertEqual(result["window_end_exclusive"], 12)
        self.assertEqual(result["short_slot_count"], 8)
        self.assertGreaterEqual(result["total_words"], 100)

    def test_rejects_insufficient_short_slots(self):
        sentences = candidate_sentences(short_count=7)
        self.assertIsNone(find_window("ignored", lambda _text: sentences))

    def test_rejects_duplicate_sentences(self):
        sentences = candidate_sentences()
        sentences[1] = sentences[0]
        self.assertIsNone(find_window("ignored", lambda _text: sentences))

    def test_build_document_keeps_full_denominator(self):
        window = find_window(
            "ignored",
            lambda _text: candidate_sentences(),
        )
        source = {
            "prompt_id": "candidate",
            "shard_row": 1700,
            "source_sha256": "source",
            "window": window,
        }
        result = build_document(source)
        self.assertEqual(len(result["sentences"]), 12)
        self.assertEqual(len(result["facts"]), 8)
        self.assertEqual(result["fact_source_indices"], list(range(8)))

    def test_graph_gate_uses_fraction_and_absolute_minimum(self):
        parsed = {
            "sentences": [
                {
                    "scale_compatible": index < 7,
                    "template_triples": 10,
                }
                for index in range(9)
            ]
        }
        result = graph_document(parsed)
        self.assertTrue(result["pass"])
        self.assertEqual(result["required_compatible"], 7)
        parsed["sentences"][6]["scale_compatible"] = False
        self.assertFalse(graph_document(parsed)["pass"])

    def test_style_gate_requires_structure_and_all_dimensions(self):
        generated = {"structure": {"pass": True}}
        style = {
            "fluency": 4,
            "coherence": 4,
            "nonredundancy": 4,
            "severe_problem": False,
        }
        self.assertTrue(all(style_gate(generated, style).values()))
        style["coherence"] = 3
        self.assertFalse(all(style_gate(generated, style).values()))


class AcquisitionTests(unittest.TestCase):
    def test_selects_first_five_rows_after_frozen_start(self):
        prefix = [
            {"text": "Too short.", "url": f"https://prefix/{index}"}
            for index in range(1652)
        ]
        candidates = [
            {
                "text": " || ".join(candidate_sentences()).replace(
                    "Topic0",
                    f"Document{index}",
                ),
                "url": f"https://candidate/{index}",
            }
            for index in range(DOCUMENTS)
        ]

        def stream():
            yield from prefix
            yield from candidates

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "run_swan_natural_mixed_windows.stream_validation",
                side_effect=stream,
            ):
                sources, screening, scanned = acquire_sources(
                    Path(directory),
                    lambda text: text.split(" || "),
                )
        self.assertEqual(len(sources), DOCUMENTS)
        self.assertEqual(sources[0]["shard_row"], 1652)
        self.assertEqual(scanned, 1656)
        self.assertEqual(screening["eligible_rows_scanned"], DOCUMENTS)


if __name__ == "__main__":
    unittest.main()
