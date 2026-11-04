"""Tests for the WP2-v8 exact global-reuse audit."""

import tempfile
from pathlib import Path
import unittest

from audit_certified_resegmentation_wp2_v8_global_reuse import (
    aggregate_shards,
    canonical_metrics_sha256,
    partition_records,
    read_document_header,
    sha256_file,
    summarize_sorted_records,
    write_document_records,
)


def record(primary, secondary, tokens, rate):
    return (
        bytes([primary]) * 32,
        bytes([secondary]) * 64,
        tokens,
        rate,
    )


class GlobalReuseTests(unittest.TestCase):
    def test_exact_duplicates_are_charged_once_at_slower_rate(self):
        metrics = summarize_sorted_records([
            record(1, 11, 10, 0.1),
            record(2, 22, 5, 0.3),
            record(1, 11, 10, 0.2),
        ])
        self.assertEqual(metrics["record_count"], 3)
        self.assertEqual(metrics["global_unique_feature_texts"], 2)
        self.assertEqual(metrics["global_unique_input_tokens"], 15)
        self.assertEqual(metrics["repeated_unique_feature_texts"], 1)
        self.assertEqual(metrics["duplicate_occurrences_removed"], 1)
        self.assertAlmostEqual(
            metrics["reconstructed_feature_seconds"],
            4.5,
        )
        self.assertAlmostEqual(metrics["cached_feature_seconds"], 3.5)

    def test_sha256_collision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA256 collision"):
            summarize_sorted_records([
                record(1, 11, 10, 0.1),
                record(1, 12, 10, 0.1),
            ])

    def test_token_count_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "token-count mismatch"):
            summarize_sorted_records([
                record(1, 11, 10, 0.1),
                record(1, 11, 12, 0.1),
            ])

    def test_document_record_round_trip(self):
        records = [
            (bytes([1]) * 32, bytes([11]) * 64, 10),
            (bytes([2]) * 32, bytes([22]) * 64, 5),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0007.bin"
            token_sum = write_document_records(
                path,
                7,
                "ab" * 32,
                0.125,
                records,
            )
            header = read_document_header(path)
            self.assertEqual(token_sum, 15)
            self.assertEqual(header["manifest_index"], 7)
            self.assertEqual(header["raw_text_sha256"], "ab" * 32)
            self.assertEqual(header["record_count"], 2)
            self.assertEqual(header["token_sum"], 15)
            self.assertEqual(header["feature_rate"], 0.125)
            self.assertEqual(len(sha256_file(path)), 64)

    def test_partition_and_aggregate_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, rate in enumerate((0.1, 0.2)):
                path = root / f"{index:04d}.bin"
                write_document_records(
                    path,
                    index,
                    f"{index + 1:064x}",
                    rate,
                    [
                        (bytes([1]) * 32, bytes([11]) * 64, 10),
                        (
                            bytes([index + 2]) * 32,
                            bytes([index + 22]) * 64,
                            5,
                        ),
                    ],
                )
                rows.append({"record_path": str(path)})
            shard_paths, count = partition_records(
                rows,
                root / "shards",
            )
            metrics = aggregate_shards(shard_paths)
            self.assertEqual(count, 4)
            self.assertEqual(metrics["record_count"], 4)
            self.assertEqual(metrics["global_unique_feature_texts"], 3)
            self.assertEqual(metrics["global_unique_input_tokens"], 20)
            self.assertEqual(metrics["repeated_unique_feature_texts"], 1)
            self.assertAlmostEqual(
                metrics["cached_feature_seconds"],
                3.5,
            )

    def test_canonical_metrics_hash_ignores_dict_insertion_order(self):
        left = {"a": 1, "b": 2}
        right = {"b": 2, "a": 1}
        self.assertEqual(
            canonical_metrics_sha256(left),
            canonical_metrics_sha256(right),
        )


if __name__ == "__main__":
    unittest.main()
