from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from run_swan_multiparser1 import validate_documents, read_graph
import run_swan_independent_endpoint1 as endpoint
from swan_independent_calibration1_core import CONDITIONS
from swan_final1_common import freeze
import run_multiscale_transfer as rt


def documents():
    docs = [{"source_id": role + str(i), "role": role, "condition": c,
             "row_id": role + str(i) + c}
            for role in ("calibration", "null") for i in range(400) for c in CONDITIONS]
    for i in range(72):
        for condition, candidate in [(c, None) for c in CONDITIONS] + [("paraphrase", str(j)) for j in range(16)]:
            docs.append({"source_id": "positive" + str(i), "role": "pilot" if i < 12 else "main",
                         "condition": condition, "candidate_id": candidate, "arm": "a",
                         "row_id": "positive" + str(i) + condition + str(candidate)})
    return docs


class IntegrityTests(unittest.TestCase):
    def test_denominators_and_overlap(self):
        docs = documents()
        validate_documents(docs)
        for malformed in (docs[:-1], docs + [docs[0]]):
            with self.assertRaises(RuntimeError):
                validate_documents(malformed)
        copy = deepcopy(docs)
        for row in copy:
            if row["source_id"] == "positive0":
                row["source_id"] = "calibration0"
        with self.assertRaises(RuntimeError):
            validate_documents(copy)
        copy = deepcopy(docs)
        copy[-1]["candidate_id"] = "0"
        with self.assertRaises(RuntimeError):
            validate_documents(copy)

    def test_checkpoint_checksum_covers_recovery_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            freeze(out / "FREEZE.json", {"fake": True})
            identity = rt.digest("x")
            path = out / "private/raw/amrbart" / (identity + ".json")
            row = {"case_id": identity, "text_sha256": identity, "freeze_sha256": rt.sha(out / "FREEZE.json"),
                   "amr": None, "amr_sha256": None, "valid": False, "attempts": [],
                   "events": [], "decode_events": [], "clipped": False, "native_status": "BACKOFF"}
            endpoint.save_row(path, row)
            self.assertEqual(read_graph(out, "amrbart", "x"), row)
            envelope = rt.read(path)
            envelope["row"]["native_status"] = "OK"
            rt.atomic(path, envelope)
            with self.assertRaises(RuntimeError):
                read_graph(out, "amrbart", "x")


if __name__ == "__main__":
    unittest.main()
