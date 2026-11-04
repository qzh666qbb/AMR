"""Tests for the scheduler-only final-baseline parallel amendment."""

from __future__ import annotations

import unittest

import parallel_swan_final_baselines as parallel


class ParallelPartitionTest(unittest.TestCase):
    def test_partitions_are_disjoint_complete_and_ordered(self):
        sources = [{"source_id": f"wm-{index:03d}"} for index in range(17)]
        groups = [
            parallel.assigned_sources(sources, worker, 4)
            for worker in range(4)
        ]
        flattened = [row["source_id"] for group in groups for row in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["source_id"] for row in sources})
        for worker, group in enumerate(groups):
            self.assertEqual(
                [row["source_id"] for row in group],
                [
                    row["source_id"]
                    for index, row in enumerate(sources)
                    if index % 4 == worker
                ],
            )

    def test_invalid_partition_fails(self):
        with self.assertRaises(ValueError):
            parallel.assigned_sources([], 0, 0)
        with self.assertRaises(ValueError):
            parallel.assigned_sources([], 2, 2)


if __name__ == "__main__":
    unittest.main()
