import logging
from pathlib import Path
import unittest
import warnings
from unittest.mock import Mock

import penman
from swan_amrbart_audited1 import (
    SOURCE, MODEL, capture, native_tokenizer, strict_graph, unified_input, decode_record,
    replay_signature)
from swan_multiparser1_core import statistics, thresholds, null_summary
from swan_independent_calibration1_core import CONDITIONS, detected


class NativeAuditTests(unittest.TestCase):
    def test_exact_pointer_graph(self):
        graph = strict_graph(["<s>", "(", "<pointer:0>", "bark-01", ":ARG0",
                              "(", "<pointer:1>", "dog", ")", ")", "</s>"])
        self.assertEqual(len(graph.triples), 3)
        for nodes in (
            ["(", "<pointer:0>", "dog"],
            ["(", "<pointer:0>", "dog", ")", "garbage"],
            ["(", "<pointer:0>", "dog", ":ARG0", "(", "<pointer:0>", "cat", ")", ")"],
            ["(", "<pointer:0>", "dog", ":ARG0", ")"],
        ):
            with self.assertRaises(Exception):
                strict_graph(nodes)

    def test_native_repair_is_not_eligible(self):
        from semantic_fact_raw_amr import audit_raw
        from types import SimpleNamespace
        tok = Mock(pad_token_id=1, bos_token_id=0, eos_token_id=2,
                   amr_bos_token_id=53226, amr_eos_token_id=53227, unk_token_id=3)
        graph = penman.decode("(z0 / dog)")
        for status, nodes, eligible in (
            ("OK", ["(", "<pointer:0>", "dog", ")"], True),
            ("FIXED", ["(", "<pointer:0>", "dog", ")"], False),
            ("BACKOFF", None, False),
            ("OK", ["(", "<pointer:0>", "dog"], False),
        ):
            tok.decode_amr.return_value = graph, SimpleNamespace(name=status), (nodes, None)
            raw = decode_record(tok, [53226, 10, 53227], "x", "f", 5)
            self.assertEqual(audit_raw(raw, {"case_id": raw["case_id"],
                                             "input_text_sha256": raw["text_sha256"]}, "f")
                             ["structurally_eligible"], eligible)
        tok.decode_amr.return_value = graph, SimpleNamespace(name="OK"), (
            ["(", "<pointer:0>", "dog", ")"], None)
        self.assertTrue(decode_record(tok, [53226] + [10] * 1022 + [53227], "x", "f", 5)["clipped"])
        self.assertTrue(decode_record(tok, [53226, 10], "x", "f", 5)["clipped"])

    def test_all_diagnostics_bound(self):
        events = []
        with capture(events, "one_sentence"):
            print("repair")
            logging.getLogger("penman").warning("repair")
            warnings.warn("repair")
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e["phase"] == "one_sentence" for e in events))

    def test_input_is_never_sliced(self):
        tok = Mock(amr_bos_token_id=53226, amr_eos_token_id=53227, mask_token_id=50264)
        tok.return_value = {"input_ids": list(range(1024))}
        self.assertEqual(len(unified_input(tok, "x")), 1027)
        tok.assert_called_once_with("x", truncation=False, padding=False)

    def test_replay_keeps_repair_evidence(self):
        self.assertEqual(replay_signature({"seconds": 1, "native_status": "OK"}),
                         replay_signature({"seconds": 2, "native_status": "OK"}))
        self.assertNotEqual(replay_signature({"native_status": "OK"}),
                            replay_signature({"native_status": "FIXED"}))

    def test_native_tokenizer_integration(self):
        root = Path(__file__).resolve().parents[1]
        model = root / MODEL
        if not model.is_dir():
            # Remote isolated runtime points at separately verified shared assets.
            import os
            model = Path(os.environ["AMRBART_ASSETS"]) / MODEL
        tok = native_tokenizer(root, model)
        words = "( <pointer:0> bark-01 :ARG0 ( <pointer:1> dog ) )".split()
        ids = [53226] + tok.tokenize_amr(words) + [53227]
        raw = decode_record(tok, ids, "The dog barks.", "f", 10)
        self.assertEqual(raw["native_status"], "OK")
        self.assertTrue(raw["strict_syntax_matches"], raw["native_nodes"])
        self.assertFalse(raw["events"] + raw["decode_events"])


class PairedStatisticsTests(unittest.TestCase):
    def test_same_candidates_and_complementarity(self):
        row = statistics([1, None, 0], [0, 1, None])
        self.assertEqual(row["complementarity"], {
            "both_green": 0, "amrlib_only_green": 1, "amrbart_only_green": 1})
        self.assertEqual(row["erasures"], {"amrlib": 1, "amrbart": 1})
        self.assertAlmostEqual(row["scores"]["ensemble"], row["scores"]["amrlib"] * 2**.5)
        with self.assertRaises(ValueError):
            statistics([1], [])
        self.assertIsNone(statistics([1], [1])["scores"]["ensemble"])

    def test_independent_conformal_and_ties(self):
        ids = [str(i) for i in range(400)]
        rows = [{"source_id": i, "role": "calibration", "condition": c,
                 **statistics([1, 0], [1, 0])} for i in ids for c in CONDITIONS]
        fit = thresholds(rows, ids)
        self.assertFalse(detected(rows[0]["scores"]["amrlib"], fit["clean"]["amrlib"]))
        null = [{**r, "role": "null", "source_id": "n" + r["source_id"]} for r in rows]
        result = null_summary(null, ["n" + i for i in ids], fit)
        self.assertEqual(len(result), 27)
        self.assertTrue(all(r["false_positives"] == 0 for r in result))
        with self.assertRaises(ValueError):
            thresholds(null, ["n" + i for i in ids])
        with self.assertRaises(ValueError):
            null_summary(null[:-1], ["n" + i for i in ids], fit)


if __name__ == "__main__":
    unittest.main()
