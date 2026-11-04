import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import swan_independent_calibration1_core as c


class CalibrationTests(unittest.TestCase):
    def rows(self):
        return [{"source_id": str(i), "role": "calibration", "condition": "clean",
                 "scores": {flow: float(i) for flow in c.FLOWS}} for i in range(400)]

    def test_exact_source_quantile_and_tie(self):
        rows = self.rows()
        t = c.fit_threshold(rows, [str(i) for i in range(400)], "a", "clean")
        self.assertEqual(t["threshold"], 396.)
        self.assertFalse(c.detected(396., t))
        self.assertTrue(c.detected(397., t))

    def test_leakage_and_duplicate_rejected(self):
        rows = self.rows()
        rows[0]["role"] = "null"
        with self.assertRaises(ValueError):
            c.fit_threshold(rows, [str(i) for i in range(400)], "a", "clean")
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            c.fit_threshold(rows, [str(i) for i in range(400)], "a", "clean")

    def test_erasure_in_full_denominator(self):
        r = c.document_statistics([{"public": .65, "a": .64, "b": .9},
                                   {"public": None, "a": None, "b": None}])
        self.assertEqual(r["green"]["public"], 1)
        self.assertEqual(r["green"]["a"], 0)
        self.assertEqual(r["green"]["pooled"], 1)
        self.assertAlmostEqual(r["scores"]["public"], (1 - .5) / math.sqrt(.375))
        self.assertEqual(r["erased"]["a"], 1)

    def test_single_sentence_abstains(self):
        r = c.document_statistics([{"public": 1, "a": 1, "b": 1}])
        self.assertTrue(r["abstain"])
        self.assertTrue(all(v is None for v in r["scores"].values()))

    def test_official_function_parity(self):
        import ast
        import itertools
        import numpy as np
        import logging
        root = Path(__file__).resolve().parents[1]
        path = root / "third_party/SWAN_official_2ca0c207_clean/detection/detection_utils.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "detect_amr_sentences")
        ns = {"np": np, "logger": logging.getLogger("parity"),
              "parse_amr": lambda s: s, "normalize_amr_variables": lambda s: s,
              "generate_template_amr": lambda s: s,
              "compute_s2match_score": lambda s, _: float(s)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        for n in range(2, 7):
            for values in itertools.product(("", ".64", ".65"), repeat=n):
                expected = ns["detect_amr_sentences"](list(values), ["bank"], .65, .25)
                rows = [{k: float(v) if v else None for k in c.BANK_KEYS} for v in values]
                self.assertAlmostEqual(c.document_statistics(rows)["scores"]["a"], expected)

    def test_missing_calibration_not_removed(self):
        rows = self.rows()
        for row in rows[:20]:
            row["scores"]["a"] = None
        t = c.fit_threshold(rows, [str(i) for i in range(400)], "a", "clean")
        self.assertEqual(t["n"], 400)
        self.assertEqual(t["threshold"], 396.)
        self.assertEqual(t["abstentions"], 20)

    def test_empty_signal_disables_detection(self):
        rows = self.rows()
        for row in rows:
            row["scores"]["a"] = None
        t = c.fit_threshold(rows, [str(i) for i in range(400)], "a", "clean")
        self.assertTrue(t["disabled"])
        self.assertFalse(c.detected(1e9, t))

    def test_null_condition_denominators(self):
        rows = self.rows()
        ts = {cond: {flow: c.fit_threshold(
            [{**r, "condition": cond} for r in rows], [str(i) for i in range(400)], flow, cond)
                     for flow in c.FLOWS} for cond in c.CONDITIONS}
        nulls = [{**r, "role": "null", "condition": cond} for r in rows for cond in c.CONDITIONS]
        result = c.evaluate_null(nulls, [str(i) for i in range(400)], ts)
        self.assertEqual(len(result), 45)
        self.assertTrue(all(r["false_positives"] == 3 and r["sources"] == 400 for r in result))
        with self.assertRaises(ValueError):
            c.evaluate_null(nulls[:-1], [str(i) for i in range(400)], ts)

    def test_lexical_sequence_and_internal_punctuation(self):
        import nltk
        root = Path(__file__).resolve().parents[1]
        nltk.data.path.insert(0, str(root / ".nltk_data"))
        t = nltk.tokenize.PunktTokenizer("english")
        text = 'Dr. Smith bought 2.5 kilos. "Keep it," Anne said. They left.'
        variants = c.variants(text, t)
        self.assertTrue(all("Dr." in v and "2.5" in v for v in variants.values()))
        self.assertEqual(variants["clean"], text)
        self.assertEqual(len(variants), 5)


if __name__ == "__main__":
    unittest.main()
