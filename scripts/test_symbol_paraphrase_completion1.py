"""Synthetic integrity/erasure tests; these are not scientific observations."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_symbol_paraphrase_completion1 as run


class CompletionTests(unittest.TestCase):
    def test_scoring_stage_has_no_evaluator_reads(self):
        tree = ast.parse(Path(run.__file__).read_text())
        scoring = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "score")
        strings = [n.value for n in ast.walk(scoring) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertFalse(any("EVALUATOR" in s for s in strings))
        self.assertIn("PREDICTION_SEAL.json", strings)

    def test_tied_or_missing_evidence_does_not_invent_order(self):
        candidates = run.equal_composition_sequences()
        for qs in ([None] * 7, [np.ones(4) / 4] * 7):
            for mode in ("hard", "soft"):
                self.assertEqual(np.ptp(run.exact_costs(qs, candidates, mode)), 0)

    def test_invalid_candidate_preserves_erasure(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            text = "Synthetic test text."
            p = out / "private/parses" / (run.runtime.digest(text) + ".json")
            record = {"freeze_sha256": "test", "amr": None}
            run.runtime.atomic(p, record)

            class Parser:
                freeze_sha = "test"
                def parse(self, text):
                    return None

            parser = Parser()
            parser.out = out
            self.assertEqual(run.graph_scores(parser, object(), text, {}), (None, "parser_ineligible"))

    def test_full_dictionary_failure_erases_only_this_sentence(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            text = "Synthetic test text."
            run.runtime.atomic(out / "private/parses" / (run.runtime.digest(text) + ".json"), {"amr": "(a / alpha)"})

            class Parser:
                freeze_sha = "test"
                def parse(self, text):
                    return "(a / alpha)"

            class Scorer:
                bank = list(range(100))
                def normalize(self, amr, kind):
                    return amr
                def pair(self, norm, t):
                    return None if t == 23 else 0.2

            parser = Parser()
            parser.out = out
            with patch.object(run.diagnostic, "eligible_parse", return_value=True):
                result = run.graph_scores(parser, Scorer(), text, {
                    "pool": list(range(24)), "groups": [list(range(s * 6, (s + 1) * 6)) for s in range(4)]})
            self.assertEqual(result, (None, "dictionary_score_invalid"))

    def test_immutable_outputs_reject_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            path = out / "record.json"
            run.frozen(path, {"x": 1})
            run.frozen(path, {"x": 1})
            with self.assertRaises(RuntimeError):
                run.frozen(path, {"x": 2})
            run.runtime.atomic(out / "FREEZE.json", {"test": True})
            seal = {"freeze_sha256": run.runtime.sha(out / "FREEZE.json"),
                    "files": {"record.json": run.runtime.sha(path)}}
            run.verify_outputs(out, seal)
            path.write_text("{}")
            with self.assertRaises(RuntimeError):
                run.verify_outputs(out, seal)


if __name__ == "__main__":
    unittest.main()
