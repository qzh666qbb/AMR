"""Synthetic-only tests: no generation, neural parser, S2Match or detector."""
import unittest

from swan_final1_transform import (
    apply_edits, document_structure, punctuation, restore_original,
    sentence_spans, stop_document, transform, word_tokens,
)


TEXT = ("Alice repairs the boat. Eve checks the boat. Alice inspects the boat. "
        "Bob opens the gate; Carol guards the gate. David locks the gate.")


def marked_syntax(text):
    """An explicit test annotation stub, not a language-quality classifier."""
    complete = text.strip() not in {"Carol near the gate.", "Bob near the gate."}
    return {"complete": complete,
            "capitalization_valid_mid_sentence": not text.startswith(("She ", "The ")),
            "content_lemmas": [w for w in ("boat", "gate") if w in text]}


class BoundaryTests(unittest.TestCase):
    def test_all_conditions_and_exact_boundaries(self):
        row = transform(TEXT, marked_syntax)
        self.assertEqual(row["failures"], {})
        conditions = row["conditions"]
        self.assertEqual(set(conditions), {"C0", "C1", "C2", "C3", "C4"})
        for condition in ("C1", "C2", "C3"):
            changed = conditions[condition]
            self.assertEqual(word_tokens(changed["text"]), word_tokens(TEXT))
            self.assertEqual(changed["num_detection_units"], 5 - (condition == "C1"))
            self.assertEqual(sum(e["kind"] == "merge" for e in changed["punctuation_edits"]), 1)
            self.assertEqual(sum(e["kind"] == "split" for e in changed["punctuation_edits"]),
                             int(condition != "C1"))
        self.assertEqual(punctuation(conditions["C2"]["text"]),
                         punctuation(conditions["C3"]["text"]))
        self.assertNotEqual(conditions["C2"]["segmentation_spans"],
                            conditions["C3"]["segmentation_spans"])
        self.assertEqual(conditions["C4"]["unit_texts"], conditions["C0"]["unit_texts"])
        self.assertEqual(conditions["C4"]["restored_text"], TEXT)

    def test_comma_and_semicolon_share_counts(self):
        for punctuation_mark in (",", ";"):
            text = TEXT.replace("; ", punctuation_mark + " ")
            row = transform(text, marked_syntax)["conditions"]
            self.assertEqual(punctuation(row["C2"]["text"]), punctuation(row["C3"]["text"]))
            self.assertEqual(row["C4"]["restored_text"], text)

    def test_lowercase_and_conjunction_not_promoted(self):
        for replacement in ("carol guards", "And Carol guards", "While Carol guards"):
            row = transform(TEXT.replace("Carol guards", replacement), marked_syntax)
            self.assertNotIn("C2", row["conditions"])
            self.assertEqual(row["failures"]["C2"], "no_eligible_swap")

    def test_fragments_fail_syntax_gate(self):
        row = transform(TEXT.replace("Carol guards the gate", "Carol near the gate"), marked_syntax)
        self.assertEqual(row["failures"]["C2"], "no_eligible_swap")

    def test_unrelated_merge_not_eligible(self):
        def unrelated(text):
            return {"complete": True, "content_lemmas": [text],
                    "capitalization_valid_mid_sentence": True}
        row = transform(TEXT, unrelated)
        self.assertEqual(row["failures"]["C2"], "no_eligible_swap")

    def test_pronoun_capitalization_not_silently_lowercased(self):
        text = TEXT.replace("Eve checks", "She checks")
        row = transform(text, marked_syntax)
        self.assertNotEqual(row["selected_pair"]["merge"]["unit_index"], 0)
        self.assertEqual(word_tokens(row["conditions"]["C2"]["text"]), word_tokens(text))

    def test_invalid_first_pair_is_not_replaced(self):
        # Punkt mistakenly retains the first edited boundary; another merge
        # remains available, but the selected C2 must fail without trying it.
        def segmentation(text):
            if text.startswith("Alice repairs the boat;"):
                return sentence_spans(text.replace("boat;", "boat.", 1))
            return sentence_spans(text)
        row = transform(TEXT, marked_syntax, segment=segmentation)
        self.assertNotIn("C2", row["conditions"])
        self.assertIn("no_retry", row["failures"]["C2"])
        self.assertEqual(row["selected_pair"]["merge"]["unit_index"], 0)

    def test_same_count_wrong_boundaries_rejected(self):
        def segmentation(text):
            spans = sentence_spans(text)
            if text != TEXT and len(spans) == 5:
                spans[0][1] -= 1
                spans[1][0] = spans[0][1]
            return spans
        row = transform(TEXT, marked_syntax, segment=segmentation)
        self.assertEqual(row["failures"]["C2"], "unexpected_boundary_positions_no_retry")

    def test_quotes_lists_unicode_layout_fail_closed(self):
        for prefix in ('"Quoted." ', "“Quoted.” ", "1. A list. ", "• Item. ",
                       "A heading:\n", "(aside) ", "A—B. ", "A…B. "):
            with self.subTest(prefix=prefix):
                row = transform(prefix + TEXT, marked_syntax)
                self.assertNotIn("C2", row["conditions"])
        row = transform(TEXT.replace("Alice", "Alice's", 1), marked_syntax)
        self.assertIn("C2", row["conditions"])

    def test_abbreviation_and_named_entity_periods(self):
        text = "Dr. Alice repairs the boat. " + TEXT
        row = transform(text, marked_syntax)
        self.assertEqual(row["conditions"]["C0"]["num_detection_units"], 6)
        self.assertIn("Dr. Alice", row["conditions"]["C2"]["text"])
        ambiguous = "Alice joined Acme Inc. " + TEXT
        row = transform(ambiguous, marked_syntax)
        self.assertNotEqual(row.get("selected_pair", {}).get("merge", {}).get("position"),
                            ambiguous.index("Inc.") + 3)

    def test_tamper_prevents_restoration(self):
        row = transform(TEXT, marked_syntax)["conditions"]
        changed = {**row["C2"], "text": row["C2"]["text"].replace("Alice", "alice")}
        with self.assertRaises(ValueError):
            restore_original(changed, row["C0"]["segmentation_spans"],
                             row["C0"]["document_sha256"])
        with self.assertRaises(ValueError):
            apply_edits(TEXT, [{"position": 0, "before": "Z", "after": "A"}])

    def test_stops_at_first_crossing_even_when_too_few_units(self):
        slots = [" ".join(["word"] * 100) + "."]
        self.assertEqual(stop_document(slots), "first_100_words")
        self.assertFalse(document_structure(slots)["structural_pass"])
        self.assertIsNone(stop_document(["Short sentence."] * 23))
        self.assertEqual(stop_document(["Short sentence."] * 24), "maximum_slots")

    def test_document_structure_no_acceptance_gate(self):
        slots = [f"Alice carefully checks boat number {i} near the main gate every early morning."
                 for i in range(8)]
        self.assertTrue(document_structure(slots)["structural_pass"])
        slots[0] += " This is another sentence."
        self.assertFalse(document_structure(slots)["structural_pass"])
        self.assertFalse(document_structure(["word " * 221 + "."])["structural_pass"])


if __name__ == "__main__":
    unittest.main()
