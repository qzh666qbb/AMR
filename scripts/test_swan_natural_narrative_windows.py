"""No-network tests for natural narrative window screening."""

import re
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import run_swan_natural_narrative_windows as narrative


class FakeToken:
    def __init__(self, text, pos="NOUN"):
        self.lemma_ = text.casefold()
        self.is_stop = text.casefold() in {"about", "for"}
        self.is_punct = False
        self.is_space = False
        self.pos_ = pos


class FakeSpan(list):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.root = FakeToken("reports", "VERB")


class FakeDoc(list):
    def __init__(self, text):
        tokens = [
            FakeToken(token)
            for token in re.findall(r"[a-z0-9']+", text.casefold())
        ]
        super().__init__(tokens)
        self._span = FakeSpan(tokens)

    @property
    def sents(self):
        return iter([self._span])


class FakeNLP:
    def pipe(self, texts):
        return [FakeDoc(text) for text in texts]


def candidate_sentences(document=0):
    rows = []
    for index in range(12):
        bridge = f"link{index // 2}"
        rows.append(
            f"Writer{document}_{index} reports {bridge} evidence about "
            f"issue{index} clearly for readers."
        )
    return rows


def candidate_text(document=0, one_sentence_per_line=False):
    rows = candidate_sentences(document)
    if one_sentence_per_line:
        return "\n".join(rows)
    return "\n".join(
        " || ".join(rows[index:index + 3])
        for index in range(0, len(rows), 3)
    )


def tokenize(text):
    return [
        row.strip()
        for row in text.replace("\n", " || ").split(" || ")
        if row.strip()
    ]


class WindowTests(unittest.TestCase):
    def test_finds_contiguous_narrative_window(self):
        result = narrative.find_window(
            candidate_text(),
            tokenize,
            FakeNLP(),
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["window_start"], 0)
        self.assertEqual(result["short_slot_count"], 12)
        self.assertEqual(result["narrative"]["source_lines"], 4)
        self.assertGreaterEqual(
            result["narrative"]["adjacent_overlap_pairs"],
            narrative.MIN_ADJACENT_OVERLAP,
        )

    def test_rejects_one_sentence_per_line_list_shape(self):
        result = narrative.find_window(
            candidate_text(one_sentence_per_line=True),
            tokenize,
            FakeNLP(),
        )
        self.assertIsNone(result)

    def test_rejects_repeated_prefixes(self):
        rows = [
            (
                f"Same prefix reports link{index // 2} evidence about "
                f"issue{index} clearly for readers."
            )
            for index in range(12)
        ]
        text = "\n".join(
            " || ".join(rows[index:index + 3])
            for index in range(0, len(rows), 3)
        )
        self.assertIsNone(
            narrative.find_window(text, tokenize, FakeNLP())
        )

    def test_identity_normalizes_apostrophes_and_case(self):
        self.assertEqual(
            narrative.identity_text("  Editor\u2019s Report  "),
            narrative.identity_text("editor's report"),
        )


class AcquisitionTests(unittest.TestCase):
    def test_selects_first_five_held_out_rows(self):
        rows = [
            {"text": "Too short.", "url": f"https://prefix/{index}"}
            for index in range(2)
        ]
        rows.extend(
            {
                "text": candidate_text(index),
                "url": f"https://candidate/{index}",
            }
            for index in range(narrative.DOCUMENTS)
        )

        def stream():
            yield from rows

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(narrative, "SOURCE_START", 2),
                patch.object(narrative, "SOURCE_END", 20),
                patch.object(
                    narrative,
                    "stream_validation",
                    side_effect=stream,
                ),
            ):
                sources, screening, scanned = narrative.acquire_sources(
                    Path(directory),
                    tokenize,
                    FakeNLP(),
                )
        self.assertEqual(len(sources), narrative.DOCUMENTS)
        self.assertEqual(sources[0]["shard_row"], 2)
        self.assertEqual(scanned, 6)
        self.assertEqual(
            screening["selected_documents"],
            narrative.DOCUMENTS,
        )
        self.assertEqual(
            screening["base_window_rows"],
            narrative.DOCUMENTS,
        )


if __name__ == "__main__":
    unittest.main()
