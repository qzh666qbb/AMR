import copy
import sys
from pathlib import Path
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import swan_v4_confirmation1_sources as sources

TEXT = "The local museum received a valuable collection from a generous resident who had collected historical books for many years."


class SourceTests(unittest.TestCase):
    def test_historical_text_url_and_locator_exclusion(self):
        row = {"text": TEXT, "url": "https://Example.org/article#part"}
        result = sources.candidate(row, 400, 0, sources.empty_bags(), [True] * 4)
        self.assertIsNotNone(result)
        for key, value in (
            ("texts", sources.source.identity(TEXT)),
            ("urls", sources.source.url_identity("https://example.org/article")),
            ("ids", result["dataset_locator"]),
        ):
            bags = sources.empty_bags()
            bags[key].add(value)
            self.assertIsNone(sources.candidate(row, 400, 0, bags, [True] * 4))

    def test_one_source_cannot_enter_multiple_roles(self):
        row = {"text": TEXT, "url": "https://example.org/a"}
        bags = sources.empty_bags()
        one = sources.candidate(row, 400, 0, bags, [True] * 4)
        sources.collect(one, bags)
        self.assertIsNone(sources.candidate(row, 401, 99, bags, [True] * 4))
        self.assertIsNone(sources.candidate({**row, "url": "https://different.org/b"},
                                            401, 99, bags, [True] * 4))

    def test_exact_phase_counts_and_reproducible_assignment(self):
        pool = [{"length_bin": bucket, "text": TEXT, "dataset_locator": f"{bucket}:{i}"}
                for bucket in range(4)
                for i in range(sum(v[bucket] for v in sources.COUNTS.values()))]
        a = sources.allocations(pool)
        b = sources.allocations(copy.deepcopy(pool))
        self.assertEqual(a, b)
        self.assertEqual({k: len(v) for k, v in a.items()},
                         {"calibration": 1000, "null": 1000, "positive_source": 250})
        all_ids = [r["dataset_locator"] for rows in a.values() for r in rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        with self.assertRaises(ValueError):
            sources.allocations(pool[:-1])

    def test_observed_legacy_and_current_hashes_are_excluded(self):
        bags = sources.empty_bags()
        sources.collect({"normalized_window_sha256": sources.source.identity(TEXT)}, bags)
        self.assertIsNone(sources.candidate({"text": TEXT, "url": "https://example.org/fresh"},
                                            400, 0, bags, [True] * 4))

    def test_full_bin_is_not_overfilled(self):
        self.assertIsNone(sources.candidate({"text": TEXT, "url": "https://example.org/fresh"},
                                            400, 0, sources.empty_bags(), [False] * 4))

    def test_generation_slots_have_fixed_full_denominator(self):
        slots = sum(n * count for n, count in zip(sources.COUNTS["positive_source"], (2, 3, 4, 4)))
        self.assertEqual(slots, 811)
        self.assertEqual(slots * 50, 40550)


if __name__ == "__main__":
    unittest.main()
