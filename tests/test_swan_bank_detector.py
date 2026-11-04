from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.core.amr_extractor import AMRExtractor
from src.core.bank_detector import SwanBankLatticeDetector
from src.core.bank_scorer import SwanAMRBankScorer, template_features
from src.core.lattice_builder import LatticeBuilder
from src.core.models import DetectionResult
from src.core.null_calibration import EmpiricalNullCalibrator, iter_input_texts
from src.core.sqlite_cache import AMRCache, normalize_text


class StaticBackend:
    def __init__(self, graphs):
        self.graphs = {
            normalize_text(text): graph for text, graph in graphs.items()
        }
        self.calls = 0

    @property
    def fingerprint(self):
        return "static-amr:v1"

    def parse_batch(self, texts):
        self.calls += 1
        return tuple(self.graphs[normalize_text(text)] for text in texts)


def detection(statistic):
    return DetectionResult(
        statistic=statistic,
        raw_utility=statistic,
        unit_count=1,
        green_count=0,
        units=(),
    )


class SwanBankDetectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="swan-bank-detector-"
        )
        self.folder = Path(self.temporary.name)
        self.bank_path = self.folder / "bank.json"
        self.bank_path.write_text(
            json.dumps(
                [
                    {
                        "normalized_template_amr": (
                            "(x0 / V :ARG0 (x1 / N) :ARG1 (x2 / X))"
                        ),
                        "count": 1,
                    }
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def scorer(self, graphs):
        backend = StaticBackend(graphs)
        extractor = AMRExtractor(
            backend=backend,
            cache=AMRCache(str(self.folder / "cache.db")),
        )
        return SwanAMRBankScorer(extractor, str(self.bank_path)), backend

    def test_raw_amr_templates_to_exact_bank_structure(self):
        scorer, _ = self.scorer(
            {
                "candidate": (
                    "(w / watch-01 "
                    ":ARG0 (c / cat) "
                    ":ARG1 (u / amr-unknown))"
                )
            }
        )
        match = scorer.score("candidate")
        self.assertEqual(match.template_index, 0)
        self.assertEqual(match.score, 1.0)
        self.assertEqual(match.relation_score, 1.0)

    def test_features_are_variable_invariant_and_include_two_hop_paths(self):
        left = template_features(
            "(a / V :ARG1 (b / N :mod (c / ADJ)))"
        )
        right = template_features(
            "(root / V :ARG1 (child / N :mod (leaf / ADJ)))"
        )
        self.assertEqual(left, right)
        self.assertEqual(sum(left.paths.values()), 1)

    def test_amr_abstract_relations_and_name_operands_are_canonicalized(self):
        features = template_features(
            """
            (h / have-quant-91
                :ARG1 (c / currency
                    :name (n / name :op1 "dollar"))
                :ARG2 (u / amr-unknown))
            """
        )
        self.assertIn(("N",), features.concepts)
        self.assertIn(("NE", ":name", "N"), features.relations)
        self.assertIn(("N", ":op1", "CONST_NE"), features.relations)

    def test_detector_chooses_target_free_split_path(self):
        scorer, backend = self.scorer(
            {
                "Alpha acts": (
                    "(a / act-01 :ARG0 (p / person) "
                    ":ARG1 (u / amr-unknown))"
                ),
                "and beta moves.": (
                    "(m / move-01 :ARG0 (b / ball) "
                    ":ARG1 (u / amr-unknown))"
                ),
                "Alpha acts and beta moves.": "(t / thing)",
            }
        )
        detector = SwanBankLatticeDetector(
            builder=LatticeBuilder(max_boundary_hops=2),
            scorer=scorer,
            score_center=0.65,
        )
        result = detector.detect("Alpha acts and beta moves.")
        self.assertEqual(result.unit_count, 2)
        self.assertEqual(
            tuple(unit.edge.span_text for unit in result.units),
            ("Alpha acts", "and beta moves."),
        )
        self.assertEqual(result.green_count, 2)
        self.assertGreater(result.statistic, 0.0)
        self.assertEqual(backend.calls, 1)

    def test_empirical_p_value_uses_finite_sample_correction(self):
        calibrator = EmpiricalNullCalibrator.fit(
            samples=(
                ("short text", detection(1.0)),
                ("another text", detection(2.0)),
                ("third text", detection(3.0)),
            ),
            detector_fingerprint="detector",
            bank_fingerprint="bank",
            bank_sha256="sha",
            parser_fingerprint="parser",
            min_bucket_size=4,
        )
        result = calibrator.calibrate(4.0, word_count=2, alpha=0.25)
        self.assertEqual(result.p_value, 0.25)
        self.assertTrue(result.is_watermarked)
        self.assertEqual(result.scope, "global_fallback")
        self.assertEqual(result.null_count, 3)

    def test_calibration_roundtrip_and_fingerprint_guard(self):
        calibrator = EmpiricalNullCalibrator.fit(
            samples=(("text", detection(0.5)),),
            detector_fingerprint="detector",
            bank_fingerprint="bank",
            bank_sha256="sha",
            parser_fingerprint="parser",
        )
        path = self.folder / "null.json"
        calibrator.save(path)
        loaded = EmpiricalNullCalibrator.load(path)
        loaded.assert_compatible("detector", "bank", "parser")
        with self.assertRaisesRegex(ValueError, "parser_fingerprint"):
            loaded.assert_compatible("detector", "bank", "other-parser")

    def test_manifest_source_input_is_supported(self):
        path = self.folder / "manifest.json"
        path.write_text(
            json.dumps({"machine_sources": [{"source": "A."}, {"source": "B."}]}),
            encoding="utf-8",
        )
        texts = tuple(
            iter_input_texts(
                path=path,
                text_field="source",
                collection_field="machine_sources",
                limit=1,
            )
        )
        self.assertEqual(texts, ("A.",))


if __name__ == "__main__":
    unittest.main()
