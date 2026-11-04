import copy
import sys
from pathlib import Path
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import run_semantic_supervision_graph_completion1 as job


def request():
    text = "Alice hands it over."
    return {"alphabet_version": "test", "core_fact_matcher_implemented": False, "parser_calls": 0,
            "views": ["R1_full", "R2_full", "R3_core"], "documents": [
                {"key": "one", "text": text, "text_sha256": job.runtime.digest(text),
                 "candidate_status": "ok", "contract": {"role_one": "Alice", "role_two": "Bob", "object": "box"},
                 "candidates": [{"anchor_span": [6, 11], "anchor_text": "hands", "contexts": [
                     {"requested_sentences": w, "span": [0, len(text)], "text": text} for w in (1, 2, 3)]}]}]}


class GraphCompletionTests(unittest.TestCase):
    def test_duplicate_views_retained_single_parse(self):
        work = job.workload(request())
        self.assertEqual((len(work["unique_texts"]), len(work["occurrences"])), (1, 3))

    def test_target_leakage_rejected_at_each_level(self):
        for level in ("documents", "candidates", "contexts"):
            r = request()
            obj = r["documents"][0]
            if level != "documents":
                obj = obj["candidates"][0]
            if level == "contexts":
                obj = obj["contexts"][0]
            obj["planned_symbol"] = "A"
            with self.assertRaises(ValueError):
                job.workload(r)

    def test_window_and_anchor_identity(self):
        for field in ("text", "span"):
            r = request()
            c = r["documents"][0]["candidates"][0]["contexts"][0]
            c[field] = "different" if field == "text" else [0, 8]
            with self.assertRaises(ValueError):
                job.workload(r)

    def test_absent_anchor_keeps_document(self):
        r = request()
        r["documents"][0]["candidates"] = []
        self.assertEqual(job.workload(r)["documents"], 1)

    def test_native_limit_is_erasure(self):
        row = job.workload(request())["unique_texts"][0]
        record = job.native_erasure(row, "freeze", 1024)
        self.assertFalse(job.audit_raw(record, row, "freeze")["structurally_eligible"])

    def test_resume_graph_drift_rejected(self):
        row = job.workload(request())["unique_texts"][0]
        record = job.native_erasure(row, "freeze", 1024)
        record["amr"] = "(x / thing)"
        with self.assertRaises(ValueError):
            job.audit_raw(record, row, "freeze")


if __name__ == "__main__":
    unittest.main()
