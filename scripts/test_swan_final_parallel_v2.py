"""Tests for the eight-worker final-baseline scheduler amendment."""

import unittest

import parallel_swan_final_baselines_v2 as scheduler


class EightWorkerSchedulerTest(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        sources = [{"source_id": f"wm-{index:03d}"} for index in range(250)]
        groups = [
            scheduler.assigned_sources(
                sources,
                worker,
                scheduler.WORKERS,
            )
            for worker in range(scheduler.WORKERS)
        ]
        identities = [
            row["source_id"] for group in groups for row in group
        ]
        self.assertEqual(len(identities), 250)
        self.assertEqual(len(set(identities)), 250)
        self.assertEqual(
            sorted(identities),
            [row["source_id"] for row in sources],
        )
        self.assertEqual(
            [len(group) for group in groups],
            [32, 32, 31, 31, 31, 31, 31, 31],
        )


if __name__ == "__main__":
    unittest.main()
