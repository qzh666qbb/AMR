"""Synthetic integrity regressions for scoring across two independent freezes."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

import run_atomic_alignment_completion2 as base
import run_multiscale_transfer as rt
from run_atomic_alignment_scoring1 import ReadOnlyGraphs


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent, self.out = self.root / "parent", self.root / "successor"
        rows = [{"row_id": c, "group_id": "synthetic", "variant_id": c,
                 "cohort": "synthetic", "text": "A child runs.",
                 "text_sha256": rt.digest("A child runs.")} for c in base.CONDITIONS]
        self.rows, _ = base.inputs(rows)
        rt.atomic(self.parent / "private/INPUTS.json", self.rows)
        rt.atomic(self.parent / "FREEZE.json", {"files": {
            "private/INPUTS.json": rt.sha(self.parent / "private/INPUTS.json")}})
        self.graph = self.parent / "private/raw_graphs" / (rt.digest("A child runs.") + ".json")
        rt.atomic(self.graph, base.native_erasure("A child runs.",
                                                 rt.sha(self.parent / "FREEZE.json"), 1024))
        rt.atomic(self.parent / "RAW_GRAPH_SEAL.json", {
            "freeze_sha256": rt.sha(self.parent / "FREEZE.json"),
            "files": {str(self.graph.relative_to(self.parent)): rt.sha(self.graph)}})
        rt.atomic(self.parent / "HALT.json", {"type": "ModuleNotFoundError"})
        rt.atomic(self.out / "FREEZE.json", {"version": "new-scoring-freeze"})
        self.ref = {"root": str(self.parent),
                    "files": {str(p.relative_to(self.parent)): rt.sha(p)
                              for p in self.parent.rglob("*") if p.is_file()},
                    "counts": {"rows": 7, "graphs": 1, "candidate_occurrences": 7,
                               "eligible": 0, "excluded": 1}}

    def store(self):
        return ReadOnlyGraphs(self.ref, self.rows)

    def test_coverage_and_original_halt_preserved(self):
        before = copy.deepcopy(self.ref["files"])
        self.assertEqual(self.store().verify()["excluded"], 1)
        self.assertEqual(before, {n: rt.sha(self.parent / n) for n in before})

    def test_each_read_rejects_post_audit_drift(self):
        store = self.store()
        store.verify()
        self.graph.write_text(self.graph.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            store.load("A child runs.")

    def test_incomplete_or_extra_graph_set_rejected(self):
        rt.atomic(self.graph.parent / "extra.json", {})
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.store().verify()

    def test_incomplete_seal_rejected_even_if_resigned(self):
        seal = rt.read(self.parent / "RAW_GRAPH_SEAL.json")
        seal["files"] = {}
        rt.atomic(self.parent / "RAW_GRAPH_SEAL.json", seal)
        self.ref["files"]["RAW_GRAPH_SEAL.json"] = rt.sha(self.parent / "RAW_GRAPH_SEAL.json")
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.store().verify()

    def test_source_freeze_not_output_freeze(self):
        store = self.store()
        record, _ = store.load("A child runs.")
        self.assertEqual(record["freeze_sha256"], rt.sha(self.parent / "FREEZE.json"))
        self.assertNotEqual(record["freeze_sha256"], rt.sha(self.out / "FREEZE.json"))
        store.freeze_sha256 = rt.sha(self.out / "FREEZE.json")
        with self.assertRaisesRegex(ValueError, "freeze mismatch"):
            store.load("A child runs.")

    def test_cohort_substitution_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["row_id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "cohort changed"):
            ReadOnlyGraphs(self.ref, rows).verify()

    def test_reference_exclusions_match_legacy_and_never_parse(self):
        class Encoder:
            np = np
            def fits(self, text):
                return True
            def encode(self, texts):
                return np.zeros((len(texts), 3), dtype=np.float32)
            class model:
                @staticmethod
                def get_sentence_embedding_dimension():
                    return 3
        class Surface:
            def analyze_many(self, texts):
                return [{} for _ in texts]
        with patch.object(rt, "Parser", side_effect=AssertionError("Parsing forbidden")):
            legacy = base.observe(self.rows[0], self.parent, Encoder(), Surface(), None)
            successor = base.observe(self.rows[0], self.out, Encoder(), Surface(), None, self.store())
        legacy["freeze_sha256"] = rt.sha(self.out / "FREEZE.json")
        self.assertEqual(successor, legacy)
        self.assertEqual(len(successor["observations"]), 1)
        self.assertEqual(successor["A0"]["statistic"], 0)
        self.assertEqual(successor["A1"]["statistic"], 0)
        self.assertFalse((self.out / "private/raw_graphs").exists())


if __name__ == "__main__":
    unittest.main()
