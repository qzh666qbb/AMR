import copy
import unittest

import materialize_semantic_sequence_stage1b as m


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.doc = {"doc_id": 3, "split": "development", "arms": {}}
        for arm, prefix in (("positive", "P"), ("unwatermarked", "N")):
            slots = [{"slot": i, "text": f"{prefix}{i} carried parcel {i}."} for i in range(8)]
            self.doc["arms"][arm] = {"slots": slots, "text": " ".join(s["text"] for s in slots),
                                     "paraphrase": f"An independently supplied {prefix} paraphrase."}

    def lexical(self, text):
        return m.re.findall(r"\w+", text)

    def test_replay_and_word_preserving_format_edits(self):
        for condition in m.CONDITIONS:
            self.assertEqual(m.variant(self.doc, "positive", condition),
                             m.variant(self.doc, "positive", condition))
        clean = self.doc["arms"]["positive"]["text"]
        for condition in ("merge", "split"):
            text, spans, _, failures = m.variant(self.doc, "positive", condition)
            self.assertEqual(self.lexical(text), self.lexical(clean))
            self.assertEqual(len(spans), 8)
            self.assertFalse(failures)
            for span in spans:
                self.assertTrue(text[span["start"]:span["end"]])

    def test_delete_insert_swap_keep_planned_identities(self):
        _, spans, operations, _ = m.variant(self.doc, "positive", "delete-2")
        self.assertEqual(len(spans), 6)
        self.assertEqual({s["slot"] for s in spans}, set(range(8)) - set(operations["deleted_slots"]))
        _, spans, operations, _ = m.variant(self.doc, "positive", "insert-2")
        self.assertEqual(len(spans), 10)
        self.assertEqual([s["slot"] for s in spans if s["origin"] == "positive"], list(range(8)))
        self.assertEqual(sum(s["origin"] == "paired_unwatermarked" for s in spans), 2)
        self.assertEqual(len({r["negative_slot"] for r in operations["insertions"]}), 2)
        _, spans, operations, _ = m.variant(self.doc, "positive", "swap-1")
        planned = list(range(8))
        a, b = operations["swapped_slots"]
        planned[a], planned[b] = planned[b], planned[a]
        self.assertEqual([s["slot"] for s in spans], planned)
        self.assertEqual(b - a, 1)

    def test_empty_slot_kept_and_reported_not_replaced(self):
        self.doc["arms"]["positive"]["slots"][2]["text"] = ""
        _, spans, _, failures = m.variant(self.doc, "positive", "split")
        self.assertEqual(len(spans), 8)
        self.assertEqual(spans[2]["start"], spans[2]["end"])
        self.assertEqual(failures, [{"slot": 2, "reason": "fewer_than_two_whitespace_words"}])

    def test_compacted_source_slots_are_rejected(self):
        self.doc["arms"]["positive"]["slots"].pop(2)
        with self.assertRaises(AssertionError):
            m.variant(self.doc, "positive", "clean")

    def test_paraphrase_has_no_fabricated_mapping(self):
        text, spans, operations, _ = m.variant(self.doc, "positive", "paraphrase")
        self.assertEqual(text, self.doc["arms"]["positive"]["paraphrase"])
        self.assertEqual(spans, [])
        self.assertEqual(operations["semantic_mapping"], "PENDING_INDEPENDENT_HUMAN_REVIEW")

    def test_cross_split_duplication_case_sensitive_whitespace_normalized(self):
        rows = [{"text": "Same   words.", "split": "development", "slot": 0},
                {"text": " Same words. ", "split": "readout", "slot": 1},
                {"text": "same words.", "split": "readout", "slot": 2}]
        groups = m.duplicate_groups(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["occurrences"]), 2)
        self.assertTrue(m.is_cross_split(groups[0]))
        self.assertNotIn("text", groups[0]["occurrences"][0])


if __name__ == "__main__":
    unittest.main()
