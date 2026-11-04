from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.core.amr_extractor import (
    AMRExtractor,
    HeuristicBackend,
    amr_features,
    decode_amr_with_audit,
)
from src.core.lattice_builder import LatticeBuilder
from src.core.real_scorer import RealScorer
from src.core.sqlite_cache import AMRCache, text_sha256
from src.core.viterbi_aligner import ViterbiAligner


class CountingBackend(HeuristicBackend):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return "counting:v1"

    def parse_batch(self, texts):
        self.calls += 1
        return super().parse_batch(texts)


class FailingBackend(HeuristicBackend):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return "failing:v1"

    def parse_batch(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic parser failure")
        return super().parse_batch(texts)


class StaticBackend:
    def __init__(self, graph: str, fingerprint: str = "static:v1") -> None:
        self.graph = graph
        self._fingerprint = fingerprint
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def parse_batch(self, texts):
        self.calls += 1
        return tuple(self.graph for _ in texts)


class ElasticIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="swan-elastic-tests-"
        )
        self.cache_path = str(Path(self.temporary.name) / "cache.db")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lattice_can_split_inside_attacker_merged_sentence(self) -> None:
        lattice = LatticeBuilder(max_boundary_hops=3).build_lattice(
            "Alpha finishes and beta starts. Gamma follows."
        )
        spans = {edge.span_text for edge in lattice.edges}
        self.assertIn("Alpha finishes", spans)
        self.assertIn("and beta starts.", spans)
        self.assertGreaterEqual(lattice.node_count, 4)

    def test_amr_features_extract_concepts_and_core_triples(self) -> None:
        concepts, triples = amr_features(
            """
            (w / watch-01
                :ARG0 (c / cat)
                :ARG1 (b / bird)
                :time (n / now))
            """
        )
        self.assertEqual(concepts, ("bird", "cat", "now", "watch"))
        self.assertEqual(
            triples,
            (("watch", ":ARG0", "cat"), ("watch", ":ARG1", "bird")),
        )

    def test_extractor_batches_then_reuses_persistent_cache(self) -> None:
        cache = AMRCache(self.cache_path)
        backend = CountingBackend()
        first = AMRExtractor(backend, cache, request_batch_size=8)
        records = first.extract_many(("Alpha acts.", "Beta responds."))
        self.assertEqual(len(records), 2)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(first.stats.parsed_texts, 2)

        second = AMRExtractor(backend, cache, request_batch_size=8)
        second.extract_many(("Alpha acts.", "Beta responds."))
        self.assertEqual(backend.calls, 1)
        self.assertEqual(second.stats.cache_hits, 2)
        self.assertEqual(second.stats.parser_calls, 0)

    def test_parser_failure_does_not_poison_cache(self) -> None:
        cache = AMRCache(self.cache_path)
        backend = FailingBackend()
        extractor = AMRExtractor(backend, cache)
        with self.assertRaisesRegex(RuntimeError, "synthetic parser failure"):
            extractor.extract("Alpha acts.")
        self.assertEqual(cache.count(backend.fingerprint), 0)

        record = extractor.extract("Alpha acts.")
        self.assertIn("alpha", record.concepts)
        self.assertEqual(cache.count(backend.fingerprint), 1)

    def test_penman_recovery_warning_is_captured_and_roundtrips(self) -> None:
        graph = (
            "(a / alpha :ARG0 (b / beta) "
            ":ARG0 b)"
        )
        _, warnings, graph_valid, roundtrip_valid = decode_amr_with_audit(
            graph
        )
        self.assertTrue(graph_valid)
        self.assertTrue(roundtrip_valid)
        self.assertTrue(
            any("duplicate triple" in warning for warning in warnings)
        )

        backend = StaticBackend(graph)
        record = AMRExtractor(
            backend, AMRCache(self.cache_path)
        ).extract("warning fixture")
        self.assertEqual(record.penman_warnings, warnings)
        self.assertTrue(record.graph_valid)
        self.assertTrue(record.roundtrip_valid)

    def test_invalid_penman_graph_fails_closed(self) -> None:
        backend = StaticBackend(
            "(a / alpha :ARG0 (b))",
            fingerprint="invalid-static:v1",
        )
        cache = AMRCache(self.cache_path)
        with self.assertRaisesRegex(ValueError, "structural validity"):
            AMRExtractor(backend, cache).extract("invalid fixture")
        self.assertEqual(cache.count(backend.fingerprint), 0)

    def test_legacy_cache_row_is_reparsed_for_current_audit(self) -> None:
        connection = sqlite3.connect(self.cache_path)
        connection.executescript(
            """
            CREATE TABLE cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE amr_records (
                text_sha256 TEXT NOT NULL,
                parser_fingerprint TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                amr TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                triples_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (text_sha256, parser_fingerprint)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO amr_records(
                text_sha256, parser_fingerprint, normalized_text, amr,
                concepts_json, triples_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text_sha256("legacy row"),
                "legacy:v1",
                "legacy row",
                "(x / stale)",
                '["stale"]',
                "[]",
            ),
        )
        connection.commit()
        connection.close()

        backend = StaticBackend("(a / alpha)", fingerprint="legacy:v1")
        cache = AMRCache(self.cache_path)
        first = AMRExtractor(backend, cache).extract("legacy row")
        self.assertEqual(backend.calls, 1)
        second = AMRExtractor(backend, cache).extract("legacy row")
        self.assertEqual(backend.calls, 1)
        self.assertEqual(first, second)

    def test_full_lattice_amr_cache_alignment_chain(self) -> None:
        targets = (
            "Alpha finishes.",
            "Beta starts.",
            "Gamma follows.",
            "Delta ends.",
        )
        attacked = (
            "Alpha finishes and beta starts. "
            "Gamma follows; delta ends."
        )
        lattice = LatticeBuilder(max_boundary_hops=3).build_lattice(attacked)
        backend = CountingBackend()
        cache = AMRCache(self.cache_path)
        extractor = AMRExtractor(backend, cache, request_batch_size=64)
        scorer = RealScorer(extractor)

        result = ViterbiAligner().align(lattice, targets, scorer)

        self.assertEqual(len(result.matched_steps), 4)
        self.assertGreater(result.normalized_score, 0.95)
        self.assertEqual(extractor.stats.parser_calls, 1)
        self.assertEqual(backend.calls, 1)

        warm_extractor = AMRExtractor(backend, cache, request_batch_size=64)
        warm_result = ViterbiAligner().align(
            lattice,
            targets,
            RealScorer(warm_extractor),
        )
        self.assertEqual(warm_result.raw_score, result.raw_score)
        self.assertEqual(warm_extractor.stats.parser_calls, 0)
        self.assertGreater(warm_extractor.stats.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
