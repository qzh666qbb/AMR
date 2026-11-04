import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_certified_resegmentation_wp2_v12_assets as audit
import run_certified_resegmentation_wp2_v12_semstamp_smoke as smoke


class FreshAssetAuditTests(unittest.TestCase):
    def test_raw_hash_and_order_are_deterministic(self):
        expected = hashlib.sha256(b"example").hexdigest()
        self.assertEqual(audit.raw_text_sha256("example"), expected)
        self.assertEqual(
            audit.order_key(expected),
            audit.order_key(expected),
        )
        self.assertNotEqual(
            audit.order_key(expected),
            audit.order_key(hashlib.sha256(b"other").hexdigest()),
        )

    def test_source_reader_rejects_empty_rows_and_keeps_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"text": "First. Second."}) + "\n")
                handle.write(json.dumps({"text": ""}) + "\n")
            rows, invalid = audit.read_source_rows(path)
        self.assertEqual(invalid, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_row_index"], 0)
        self.assertEqual(
            rows[0]["raw_text_sha256"],
            audit.raw_text_sha256("First. Second."),
        )

    def test_jsonl_writer_is_canonical_and_atomic(self):
        rows = [{"b": 2, "a": 1}, {"value": "x"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            first = audit.write_jsonl(path, rows)
            first_bytes = path.read_bytes()
            second = audit.write_jsonl(path, rows)
        self.assertEqual(first, second)
        self.assertEqual(
            hashlib.sha256(first_bytes).hexdigest(),
            first["sha256"],
        )
        self.assertEqual(first["rows"], 2)

    def test_power_boundaries_match_protocol(self):
        result = audit.power_analysis()
        self.assertEqual(result["calibration_order_statistic"], 951)
        self.assertEqual(result["fpr_gate"]["max_exceedances"], 65)
        self.assertEqual(result["tpr_gate"]["min_successes"], 800)
        self.assertEqual(
            result["k1_certificate_gate"]["min_successes"],
            200,
        )

    def test_distribution(self):
        self.assertEqual(
            audit.distribution([4, 1, 3, 2]),
            {"count": 4, "min": 1, "median": 2.5, "max": 4},
        )
        self.assertEqual(
            audit.distribution([]),
            {"count": 0, "min": None, "median": None, "max": None},
        )

    def test_semstamp_smoke_seed_is_stable_and_bounded(self):
        first = smoke.generation_seed("0" * 64)
        self.assertEqual(first, 1575591264)
        self.assertEqual(first, smoke.generation_seed("0" * 64))
        self.assertNotEqual(first, smoke.generation_seed("1" * 64))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2 ** 31)


if __name__ == "__main__":
    unittest.main()
