from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.core.elastic_v3 import (
    ElasticBoundaryProposerV3,
    InvalidTolerantAMRExtractor,
    SwanAMRBankScorerV3,
    SwanBankLatticeDetectorV3,
    classify_edge_evidence,
)
from src.core.elastic_v2 import ElasticBoundaryProposerV2
from src.core.models import (
    AMR_AUDIT_VERSION,
    AMRRecord,
    BankMatch,
    Boundary,
    Lattice,
    LatticeEdge,
)
from src.core.sqlite_cache import AMRCache, normalize_text, text_sha256


def match(score: float) -> BankMatch:
    return BankMatch(
        template_index=0,
        score=score,
        concept_score=score,
        relation_score=score,
        path_score=score,
    )


def small_lattice() -> Lattice:
    return Lattice(
        text="alpha beta",
        boundaries=(
            Boundary(index=0, char_offset=0, kind="terminal"),
            Boundary(index=1, char_offset=5, kind="test"),
            Boundary(index=2, char_offset=10, kind="terminal"),
        ),
        edges=(
            LatticeEdge(0, 1, "alpha", 1),
            LatticeEdge(1, 2, "beta", 1),
            LatticeEdge(0, 2, "alpha beta", 2),
        ),
    )


def record(
    text: str,
    *,
    graph_valid: bool = True,
    roundtrip_valid: bool = True,
) -> AMRRecord:
    normalized = normalize_text(text)
    roundtrip_valid = graph_valid and roundtrip_valid
    return AMRRecord(
        text=normalized,
        text_sha256=text_sha256(normalized),
        parser_fingerprint="test-parser:v1",
        amr="(x / test)",
        concepts=("test",) if graph_valid and roundtrip_valid else (),
        triples=(),
        audit_version=AMR_AUDIT_VERSION,
        penman_warnings=(),
        graph_valid=graph_valid,
        roundtrip_valid=roundtrip_valid,
    )


def recovery_records(lattice: Lattice, error_span: str = "") -> dict:
    records = {}
    for edge in lattice.edges:
        records[text_sha256(edge.span_text)] = {
            "selected_beam": 0,
            "events": (
                [{
                    "level": "ERROR",
                    "beam_index": 0,
                    "message": "synthetic selected-beam error",
                }]
                if edge.span_text == error_span
                else []
            ),
        }
    return records


class StaticBackend:
    def __init__(self, graph: str) -> None:
        self.graph = graph
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return "invalid-tolerant-static:v1"

    def parse_batch(self, texts):
        self.calls += 1
        return tuple(self.graph for _ in texts)


class StaticEvidenceScorer:
    def __init__(self, scores, records) -> None:
        self.scores = scores
        self._records = records
        self.fingerprint = "static-evidence-scorer:v1"

    def score_many(self, texts):
        return tuple(self.scores[normalize_text(text)] for text in texts)


class ElasticV3Tests(unittest.TestCase):
    def test_v3_preserves_v2_boundary_topology(self):
        texts = (
            "Alpha acts. Beta responds.",
            "Alpha acts, Beta responds, Gamma closes.",
            "Alpha acts\nBeta responds\nGamma closes.",
            " ".join(f"word{i}" for i in range(205)),
        )
        v2 = ElasticBoundaryProposerV2()
        v3 = ElasticBoundaryProposerV3()
        for text in texts:
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    v3.build_lattice(text),
                    v2.build_lattice(text),
                )

    def test_invalid_graph_is_cached_and_reused_without_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = AMRCache(str(Path(folder) / "cache.db"))
            backend = StaticBackend("(a / alpha :ARG0 (b))")
            extractor = InvalidTolerantAMRExtractor(backend, cache)
            parsed = extractor.extract("invalid fixture")

            self.assertFalse(parsed.graph_valid)
            self.assertFalse(parsed.roundtrip_valid)
            self.assertEqual(parsed.concepts, ())
            self.assertEqual(parsed.triples, ())
            self.assertEqual(cache.count(backend.fingerprint), 1)
            self.assertEqual(extractor.stats.invalid_graph_records, 1)

            bank = Path(folder) / "bank.json"
            bank.write_text(
                json.dumps([{
                    "normalized_template_amr": "(x / V :ARG0 (y / N))",
                }]),
                encoding="utf-8",
            )
            warm = InvalidTolerantAMRExtractor(backend, cache)
            scorer = SwanAMRBankScorerV3(warm, str(bank))
            self.assertIsNone(scorer.score_optional("invalid fixture"))
            self.assertEqual(backend.calls, 1)
            self.assertEqual(warm.stats.cached_invalid_records, 1)

    def test_evidence_classifies_recovery_graph_and_roundtrip_failures(self):
        lattice = small_lattice()
        records = {
            "alpha": record("alpha"),
            "beta": record("beta", roundtrip_valid=False),
            "alpha beta": record("alpha beta", graph_valid=False),
        }
        evidence = classify_edge_evidence(
            lattice,
            recovery_records(lattice, error_span="alpha"),
            records,
        )
        self.assertEqual(evidence[0].reasons, ("selected_beam_error",))
        self.assertEqual(evidence[1].reasons, ("roundtrip_invalid",))
        self.assertEqual(
            evidence[2].reasons,
            ("graph_invalid", "roundtrip_invalid"),
        )
        self.assertTrue(all(not item.valid for item in evidence))

    def test_invalid_high_score_edge_cannot_enter_dp(self):
        lattice = small_lattice()
        records = {
            "alpha": record("alpha"),
            "beta": record("beta"),
            "alpha beta": record("alpha beta", graph_valid=False),
        }
        detector = SwanBankLatticeDetectorV3(
            builder=ElasticBoundaryProposerV3(),
            scorer=StaticEvidenceScorer(
                {
                    "alpha": match(0.85),
                    "beta": match(0.85),
                    "alpha beta": match(1.0),
                },
                records,
            ),
        )
        result, evidence = detector.detect_lattice_with_evidence(
            lattice,
            recovery_records(lattice),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unit_count, 2)
        self.assertEqual(
            tuple(unit.edge.span_text for unit in result.units),
            ("alpha", "beta"),
        )
        self.assertEqual(result.invalid_edge_count, 1)
        self.assertFalse(evidence[2].valid)

    def test_no_complete_structurally_valid_path_abstains(self):
        lattice = small_lattice()
        records = {
            edge.span_text: record(edge.span_text, graph_valid=False)
            for edge in lattice.edges
        }
        detector = SwanBankLatticeDetectorV3(
            builder=ElasticBoundaryProposerV3(),
            scorer=StaticEvidenceScorer(
                {
                    edge.span_text: match(1.0)
                    for edge in lattice.edges
                },
                records,
            ),
        )
        result, evidence = detector.detect_lattice_with_evidence(
            lattice,
            recovery_records(lattice),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstain_reason, "no_complete_valid_path")
        self.assertEqual(result.invalid_edge_count, 3)
        self.assertTrue(all(not item.valid for item in evidence))

    def test_missing_recovery_or_amr_evidence_stops(self):
        lattice = small_lattice()
        complete_records = {
            edge.span_text: record(edge.span_text)
            for edge in lattice.edges
        }
        with self.assertRaisesRegex(RuntimeError, "recovery evidence"):
            classify_edge_evidence(lattice, {}, complete_records)

        missing_amr = dict(complete_records)
        missing_amr.pop("alpha beta")
        with self.assertRaisesRegex(RuntimeError, "AMR audit record"):
            classify_edge_evidence(
                lattice,
                recovery_records(lattice),
                missing_amr,
            )


if __name__ == "__main__":
    unittest.main()
