from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.core.length_boundary_validation import (
    VARIANTS, freeze_runtime, lexical_tokens, punctuation_variant,
    select_short_cohort, summarize,
)


class LengthBoundaryValidationTests(unittest.TestCase):
    def test_all_variants_preserve_case_sensitive_lexical_order(self):
        source = 'Alice said "No." Bob left! Did Carol stay? Dan arrived.'
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                text = punctuation_variant(source, variant)
                self.assertEqual(lexical_tokens(text), lexical_tokens(source))
        self.assertIn("Bob left!", punctuation_variant(source, "semicolon_pairs"))
        self.assertNotIn("\n", punctuation_variant(source, "space_all"))
        self.assertIn("\n", punctuation_variant(source, "newline_all"))

    def test_standalone_punctuation_preserves_words_even_if_split_count_changes(self):
        source = "Alice left . Bob stayed."
        result = punctuation_variant(source, "space_all")
        self.assertEqual(lexical_tokens(result), lexical_tokens(source))

    def test_resume_reuses_timestamp_and_rejects_configuration_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime.json"
            first = freeze_runtime(path, {"version": "v1", "config": 3})
            self.assertEqual(freeze_runtime(path, {"version": "v1", "config": 3}), first)
            with self.assertRaises(ValueError):
                freeze_runtime(path, {"version": "v1", "config": 4})

    def test_selection_excludes_previously_used_sources_and_has_one_window_per_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.json"
            rows = []
            for i, words in enumerate((20, 28, 36, 44), 1):
                text = " ".join(["First"] + ["word"] * (words // 2 - 2) + ["ends."])
                text += " " + " ".join(["Second"] + ["word"] * (words - words // 2 - 2) + ["ends."])
                rows.append({
                    "ordinal": i, "allocation_id": f"id{i}", "locator": f"loc{i}",
                    "normalized_source_sha256": f"src{i}", "url_sha256": f"url{i}",
                    "text": text,
                })
            path.write_text(json.dumps(rows))
            old = [{
                "allocation_ordinal": 0, "allocation_id": "old", "locator": "old",
                "normalized_source_sha256": "old", "source_text_sha256": "old",
                "url_sha256": "old", "window_sha256": "old",
            }]
            selected = select_short_cohort(path, "calibration", 1, old)
            self.assertEqual(len(selected), 4)
            self.assertEqual(len({r["allocation_id"] for r in selected}), 4)
            self.assertEqual(len({r["length_stratum"] for r in selected}), 4)
            old[0]["url_sha256"] = "url1"
            with self.assertRaises(ValueError):
                select_short_cohort(path, "calibration", 1, old)

    def test_correlated_variant_counts_use_documents_and_conditional_baseline(self):
        def row(cohort, group, variant, detected):
            return {
                "cohort": cohort, "group_id": group, "variant_id": variant,
                "word_count": 30,
                "calibration": {"is_watermarked": detected},
                "legacy_calibration": {"is_watermarked": detected},
            }
        stress = [row("confirmation_baseline_positive", "a", "baseline", True),
                  row("confirmation_baseline_positive", "b", "baseline", False),
                  row("confirmation_boundary_attack", "a", "attack", True),
                  row("confirmation_boundary_attack", "b", "attack", False)]
        for variant in VARIANTS:
            stress += [
                row("confirmation_punctuation_stress", "a", variant, variant != "space_all"),
                row("confirmation_punctuation_stress", "b", variant, True),
                row("independent_null_punctuation_stress", "n1", variant, variant == "comma_all"),
                row("independent_null_punctuation_stress", "n2", variant, False),
            ]
        test = [{"allocation_id": "n1", "word_count": 30,
                 "calibration": {"is_watermarked": False},
                 "legacy_calibration": {"is_watermarked": False}},
                {"allocation_id": "n2", "word_count": 30,
                 "calibration": {"is_watermarked": False},
                 "legacy_calibration": {"is_watermarked": False}}]
        summary = summarize(test, stress)["calibration"]
        family = summary["any_of_five_null_false_positive_rate"]
        self.assertEqual((family["successes"], family["trials"]), (1, 2))
        worst = summary["survival_all_five_variants"]
        self.assertEqual((worst["successes"], worst["trials"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
