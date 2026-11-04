"""Tests for the v3 variable-N score-blind profile runner."""

from pathlib import Path
import tempfile
import unittest

from run_semstamp_certified_defense_v3_variable_n_profile import (
    EXPECTED_HARDWARE,
    aggregate,
    deduplicate_feature_texts,
    prepare_document_profile,
    selected_indices,
    target_counts,
)


class FakeDocument:
    atom_count = 9
    original_boundaries = (2, 5, 7)

    def render_unit_bytes(self, start, end):
        return f"{start}:{end}".encode("ascii")


class DuplicateTextDocument(FakeDocument):
    def render_unit_bytes(self, start, end):
        return f"width:{end - start}".encode("ascii")


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, texts, **_kwargs):
        rows = []
        for text in texts:
            start_text, end_text = text.split(":")
            start = int(start_text)
            end = int(end_text)
            size = 513 if (start, end) == (0, 4) else 3
            rows.append(list(range(size)))
        return {"input_ids": rows}


class DuplicateTokenizer:
    model_max_length = 512

    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": [
                list(range(int(text.split(":")[1]) + 2))
                for text in texts
            ]
        }


def profiles():
    return [
        {
            "atom_count": 5,
            "selection_slot": "small",
            "feature_seconds_per_input_token": 2.0,
            "sidecar_seconds_per_span": 3.0,
            "core_seconds_per_pair": 4.0,
            "model_load_seconds": 5.0,
        },
        {
            "atom_count": 10,
            "selection_slot": "large",
            "feature_seconds_per_input_token": 1.0,
            "sidecar_seconds_per_span": 1.5,
            "core_seconds_per_pair": 2.0,
            "model_load_seconds": 6.0,
        },
    ]


class VariableNProfileTests(unittest.TestCase):
    def test_index_parser_and_target_layers(self):
        self.assertEqual(selected_indices("0,2-4,9"), [0, 2, 3, 4, 9])
        self.assertEqual(target_counts(FakeDocument(), 0), (3,))
        self.assertEqual(target_counts(FakeDocument(), 3), tuple(range(1, 7)))
        with self.assertRaises(ValueError):
            selected_indices("2,1")

    def test_exact_text_deduplication_precedes_token_accounting(self):
        import numpy as np

        document = DuplicateTextDocument()
        base = document.atom_count + 1
        codes = np.asarray([
            0 * base + 2,
            1 * base + 3,
            0 * base + 3,
            2 * base + 5,
        ], dtype=np.uint64)
        allowed, counts = deduplicate_feature_texts(
            document, codes, DuplicateTokenizer()
        )
        self.assertEqual(counts["deduplicated_feature_text_count"], 2)
        self.assertEqual(counts["exact_duplicate_span_count"], 2)
        self.assertEqual(counts["deduplicated_legal_input_tokens"], 9)
        self.assertTrue(all(allowed))
        self.assertEqual(counts["sha256_collision_bucket_count"], 0)

    def test_all_target_layers_use_native_pruned_union_sidecars(self):
        root = Path(__file__).resolve().parents[1]
        core = (
            root
            / "build/semstamp_v3/"
            "semstamp_certified_defense_v3_variable_n_core"
        )
        if not core.is_file():
            self.skipTest("compiled v3 core is absent")
        with tempfile.TemporaryDirectory() as directory:
            radii, resources, projection = prepare_document_profile(
                FakeDocument(),
                FakeTokenizer(),
                core,
                Path(directory),
                profiles(),
            )
        self.assertEqual(
            sum(
                len(row["layers"])
                for key, row in radii.items()
                if int(key) == row["effective_radius"]
            ),
            15,
        )
        self.assertEqual(radii["3"]["target_edge_counts"], list(range(1, 7)))
        self.assertGreater(
            radii["3"]["native_forbidden_union_span_count"], 0
        )
        self.assertLessEqual(resources["maximum_layer_pair_count"], 200_000_000)
        self.assertLessEqual(
            resources["maximum_radius_union_span_count"], 2_000_000
        )
        self.assertGreater(projection["measured_pruned_core_seconds"], 0)

    def test_subset_aggregate_reports_both_two_replay_formulas(self):
        row = {
            "plan_index": 7,
            "cohort": "calibration_null",
            "refinement_count": 1,
            "zero_byte_gap_count": 2,
            "score_access_count": 0,
            "radii": {
                str(radius): {
                    "effective_radius": radius,
                    "all_allowed_union_span_count": 10,
                    "layers": [{}],
                }
                for radius in range(4)
            },
            "resource_counts": {
                "deduplicated_legal_feature_text_count": 5,
                "deduplicated_legal_input_tokens": 20,
                "sum_distinct_radius_union_spans": 40,
                "sum_distinct_layer_pruned_pairs": 80,
                "maximum_layer_pair_count": 30,
                "maximum_radius_union_span_count": 10,
                "maximum_sidecar_bytes": 200,
                "sha256_collision_bucket_count": 0,
            },
            "projection": {
                "feature_seconds": 2.0,
                "union_sidecar_rate_seconds": 3.0,
                "pair_rate_core_seconds": 4.0,
                "measured_topology_seconds": 5.0,
                "measured_union_sidecar_seconds": 1.0,
                "measured_pruned_core_seconds": 6.0,
            },
        }
        timing = {
            "plan_index": 7,
            "worker_peak_rss_bytes": 100,
            "child_peak_rss_bytes": 200,
        }
        result = aggregate(
            [row],
            [timing],
            [7],
            profiles(),
            9.0,
            EXPECTED_HARDWARE,
        )
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["full_denominator"])
        self.assertEqual(
            result["runtime"]["two_replay_measured_core_formula_seconds"],
            38.0,
        )
        self.assertEqual(
            result["runtime"]["two_replay_pair_rate_formula_seconds"],
            34.0,
        )
        self.assertEqual(
            result["runtime"][
                "conservative_two_replay_upper_bound_seconds"
            ],
            38.0,
        )


if __name__ == "__main__":
    unittest.main()
