"""Offline tests for the source-anchored splittable SWAN pilot."""
import re
import unittest

import run_swan_source_anchored_splittable as pilot


class FakeSyntax:
    def __call__(self, text):
        words = {
            token.casefold()
            for token in re.findall(r"[A-Za-z]+", text)
            if token.casefold() not in {"the", "a", "an"}
        }
        return {
            "complete": bool(text),
            "capitalization_valid_mid_sentence": bool(
                text and text[0].isupper()
            ),
            "content_lemmas": sorted(words),
        }


def source_fixture():
    anchors = [
        f"Atlas Council approved harbor project phase {index} after public review."
        for index in range(10)
    ]
    counts = [10] * 10
    counts[4] = 12
    return {
        "source_anchors": anchors,
        "anchor_word_counts": counts,
    }


class ScheduleTests(unittest.TestCase):
    def test_longest_anchor_receives_adjacent_third_slot(self):
        schedule = pilot.anchor_schedule(source_fixture())
        self.assertEqual(len(schedule), 21)
        self.assertEqual(schedule[:2], (0, 0))
        self.assertEqual(schedule.count(4), 3)
        self.assertTrue(all(schedule.count(index) == 2 for index in range(10) if index != 4))

    def test_budget_is_exact(self):
        self.assertEqual(pilot.MAX_CALLS, 633)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 1_788_098_400)
        self.assertEqual(pilot.PILOT_CAP, (633, 1_788_098_400))


class SurfaceTests(unittest.TestCase):
    def test_requires_two_anchor_lemmas(self):
        source = source_fixture()
        accepted = pilot.surface_result_with_syntax(
            "Atlas Council approved harbor repairs.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertTrue(accepted["pass"])
        self.assertGreaterEqual(
            len(accepted["anchor_content_lemma_overlap"]),
            2,
        )
        rejected = pilot.surface_result_with_syntax(
            "Writers published unrelated books today.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["at_least_two_anchor_content_lemmas"])

    def test_rejects_internal_punctuation_and_repetition(self):
        source = source_fixture()
        punctuated = pilot.surface_result_with_syntax(
            "Atlas Council approved harbor repairs, today.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(punctuated["pass"])
        prior = "Atlas Council approved harbor repairs."
        repeated = pilot.surface_result_with_syntax(
            prior,
            [prior],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(repeated["pass"])
        self.assertFalse(repeated["prior_token_jaccard_at_most_point8"])


class ConstructionTests(unittest.TestCase):
    def test_standard_c2_is_count_neutral_and_restorable(self):
        generated = [
            f"Atlas Council approved harbor phase {index}."
            for index in range(21)
        ]
        compound = generated[0][:-1] + "; " + generated[1]
        text = " ".join([compound, *generated[2:]])
        result = pilot.construct_conditions(text, FakeSyntax())
        self.assertTrue(result["constructed"])
        c0 = result["conditions"]["C0"]
        c2 = result["conditions"]["C2"]
        r4 = result["conditions"]["R4"]
        self.assertEqual(c0["num_detection_units"], 20)
        self.assertEqual(c2["num_detection_units"], 20)
        self.assertEqual(c0["word_tokens"], c2["word_tokens"])
        self.assertEqual(c0["punctuation"], c2["punctuation"])
        self.assertEqual(c0["character_length"], c2["character_length"])
        self.assertEqual(c0, r4)
        self.assertGreaterEqual(result["selected_merge"]["unit_index"], 2)


class ReviewTests(unittest.TestCase):
    def test_source_consistency_is_strict(self):
        row = {
            "coherence": 4,
            "naturalness": 3,
            "nonredundancy": 3,
            "task_consistency": 4,
            "source_consistency": 5,
            "severe_problem": False,
        }
        self.assertTrue(pilot.review_pass(row))
        row["source_consistency"] = 4
        self.assertFalse(pilot.review_pass(row))


if __name__ == "__main__":
    unittest.main()
