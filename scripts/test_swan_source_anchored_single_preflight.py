"""Offline tests for the corrected source-anchored single-slot preflight."""
import re
import unittest

import run_swan_source_anchored_single_preflight as pilot
from swan_anchor1_sources import identity


class FakeSyntax:
    def __call__(self, text):
        words = {
            token.casefold()
            for token in re.findall(r"[A-Za-z]+", text)
            if token.casefold() not in {"the", "a", "an"}
        }
        return {
            "complete": bool(text),
            "content_lemmas": sorted(words),
        }


def source_fixture():
    return {
        "source_anchors": [
            "The summit delivered a rebuke to Brussels after lengthy talks.",
            *[
                f"Atlas Council approved harbor project phase {index}."
                for index in range(1, 10)
            ],
        ],
    }


class ContractTests(unittest.TestCase):
    def test_budget_is_exact_and_scope_is_single_slot(self):
        self.assertEqual(pilot.TEMPLATE_SCHEDULE, (47,))
        self.assertEqual(pilot.MAX_SLOTS, 1)
        self.assertEqual(pilot.REVIEW_COUNT, 0)
        self.assertEqual(pilot.MAX_CALLS, 31)
        self.assertEqual(pilot.MAX_COST_NANODOLLARS, 87_568_800)
        self.assertEqual(pilot.budget()["max_cost_usd"], 0.0875688)

    def test_context_rejects_nonempty_or_later_context(self):
        with self.assertRaisesRegex(ValueError, "slot-0"):
            pilot.context_for(1, [])
        with self.assertRaisesRegex(ValueError, "slot-0"):
            pilot.context_for(0, ["Prior sentence."])


class SurfaceTests(unittest.TestCase):
    def test_the_summit_regression_passes_without_proper_noun_gate(self):
        text = "The summit delivered a rebuke to Brussels."
        result = pilot.surface_result_with_syntax(
            text,
            [],
            set(),
            source_fixture(),
            FakeSyntax(),
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["orthographic_initial_capital"])
        self.assertNotIn("capitalized_clause", result)
        self.assertGreaterEqual(
            len(result["anchor_content_lemma_overlap"]),
            2,
        )

    def test_lowercase_and_historical_text_are_rejected(self):
        source = source_fixture()
        lower = "the summit delivered a rebuke to Brussels."
        result = pilot.surface_result_with_syntax(
            lower,
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(result["orthographic_initial_capital"])
        self.assertFalse(result["pass"])

        text = "The summit delivered a rebuke to Brussels."
        result = pilot.surface_result_with_syntax(
            text,
            [],
            {identity(text)},
            source,
            FakeSyntax(),
        )
        self.assertFalse(result["not_previously_observed_text"])
        self.assertFalse(result["pass"])

    def test_requires_two_anchor_lemmas_and_five_to_twelve_words(self):
        source = source_fixture()
        unrelated = pilot.surface_result_with_syntax(
            "Writers published several unrelated books today.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(unrelated["at_least_two_anchor_content_lemmas"])
        short = pilot.surface_result_with_syntax(
            "Summit talks ended.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(short["word_count_5_12"])

    def test_rejects_internal_punctuation_and_multiple_units(self):
        source = source_fixture()
        punctuated = pilot.surface_result_with_syntax(
            "The summit delivered a rebuke, after talks.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(punctuated["simple_punctuation"])
        multiple = pilot.surface_result_with_syntax(
            "The summit ended. Brussels issued a response.",
            [],
            set(),
            source,
            FakeSyntax(),
        )
        self.assertFalse(multiple["one_punkt_unit"])


if __name__ == "__main__":
    unittest.main()
