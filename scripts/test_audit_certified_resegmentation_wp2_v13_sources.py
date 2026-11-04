import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_certified_resegmentation_wp2_v13_sources as audit


class SemStampV13SourceAuditTests(unittest.TestCase):
    def test_order_key_is_stable_and_salted(self):
        identity = hashlib.sha256(b"example").hexdigest()
        self.assertEqual(audit.order_key(identity), audit.order_key(identity))
        self.assertNotEqual(
            audit.order_key(identity),
            hashlib.sha256(identity.encode("ascii")).hexdigest(),
        )

    def test_source_reader_preserves_identity_and_rejects_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "text": "First sentence. Second sentence.",
                    "timestamp": "2020-01-01",
                    "url": "https://example.test",
                }) + "\n")
                handle.write(json.dumps({"text": ""}) + "\n")
            rows, invalid = audit.read_source(path)
        self.assertEqual(invalid, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_row_index"], 0)
        self.assertEqual(
            rows[0]["raw_text_sha256"],
            audit.sha256_text("First sentence. Second sentence."),
        )

    def test_jsonl_writer_is_canonical(self):
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

    def test_distribution(self):
        self.assertEqual(
            audit.distribution([4, 1, 3, 2]),
            {"count": 4, "min": 1, "median": 2.5, "max": 4},
        )
        self.assertEqual(
            audit.distribution([]),
            {"count": 0, "min": None, "median": None, "max": None},
        )


if __name__ == "__main__":
    unittest.main()
