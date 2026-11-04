from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .amr_extractor import (
    AMRExtractor,
    ExtractorStats,
    amr_features,
    decode_amr_with_audit,
)
from .bank_scorer import SwanAMRBankScorer
from .elastic_v2 import (
    CountStatePathOptimizer,
    ElasticBoundaryProposerV2,
    V2PathResult,
)
from .models import AMR_AUDIT_VERSION, AMRRecord, BankMatch, Lattice
from .sqlite_cache import normalize_text, text_sha256


ELASTIC_V3_VERSION = "swan-bank-lattice-detector:v3-invalid-edge"
INVALID_EDGE_POLICY = "penman-invalid-edge:v1"


class ElasticBoundaryProposerV3(ElasticBoundaryProposerV2):
    """Versioned alias: v3 intentionally preserves the frozen v2 topology."""


@dataclass
class V3ExtractorStats(ExtractorStats):
    invalid_graph_records: int = 0
    invalid_roundtrip_records: int = 0
    cached_invalid_records: int = 0


class InvalidTolerantAMRExtractor(AMRExtractor):
    """Persist audited invalid graphs so the detector can reject their edges."""

    def __init__(self, backend, cache, request_batch_size: int = 128) -> None:
        super().__init__(backend, cache, request_batch_size)
        self.stats = V3ExtractorStats()

    @staticmethod
    def _record(
        text: str,
        parser_fingerprint: str,
        amr: str,
    ) -> AMRRecord:
        (
            _,
            penman_warnings,
            graph_valid,
            roundtrip_valid,
        ) = decode_amr_with_audit(amr)
        if graph_valid and roundtrip_valid:
            concepts, triples = amr_features(amr)
        else:
            concepts, triples = (), ()
        return AMRRecord(
            text=text,
            text_sha256=text_sha256(text),
            parser_fingerprint=parser_fingerprint,
            amr=amr,
            concepts=concepts,
            triples=triples,
            audit_version=AMR_AUDIT_VERSION,
            penman_warnings=penman_warnings,
            graph_valid=graph_valid,
            roundtrip_valid=roundtrip_valid,
        )

    def extract_many(self, texts: Iterable[str]) -> Dict[str, AMRRecord]:
        normalized_texts = []
        seen = set()
        for text in texts:
            normalized = normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_texts.append(normalized)

        records: Dict[str, AMRRecord] = {}
        missing = []
        for text in normalized_texts:
            cached = self.cache.get(text, self.backend.fingerprint)
            if cached is None:
                self.stats.cache_misses += 1
                missing.append(text)
                continue
            self.stats.cache_hits += 1
            if not cached.graph_valid or not cached.roundtrip_valid:
                self.stats.cached_invalid_records += 1
            records[text] = cached

        for start in range(0, len(missing), self.request_batch_size):
            batch = missing[start:start + self.request_batch_size]
            self.stats.parser_calls += 1
            outputs = self.backend.parse_batch(batch)
            if len(outputs) != len(batch):
                raise RuntimeError("Parser backend violated the batch-size contract")
            parsed_records = []
            for text, amr in zip(batch, outputs):
                record = self._record(
                    text,
                    self.backend.fingerprint,
                    amr,
                )
                parsed_records.append(record)
                records[text] = record
                if record.penman_warnings:
                    self.stats.parsed_with_warnings += 1
                    self.stats.penman_warning_records += len(
                        record.penman_warnings
                    )
                if not record.graph_valid:
                    self.stats.invalid_graph_records += 1
                elif not record.roundtrip_valid:
                    self.stats.invalid_roundtrip_records += 1
            self.cache.put_many(parsed_records)
            self.stats.parsed_texts += len(parsed_records)

        return records


class SwanAMRBankScorerV3(SwanAMRBankScorer):
    """Return no match for an invalid audited graph."""

    def __init__(self, extractor: InvalidTolerantAMRExtractor, bank_path: str):
        super().__init__(extractor=extractor, bank_path=bank_path)
        base_fingerprint = self.fingerprint
        payload = {
            "base_scorer_fingerprint": base_fingerprint,
            "invalid_edge_policy": INVALID_EDGE_POLICY,
        }
        self.fingerprint = "swan-bank-scorer-v3:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def score_optional(self, span_text: str) -> Optional[BankMatch]:
        record = self._record(span_text)
        if not record.graph_valid or not record.roundtrip_valid:
            return None
        return self.score_amr(record.amr)

    def score_many(
        self,
        span_texts: Sequence[str],
    ) -> Tuple[Optional[BankMatch], ...]:
        self.prepare(span_texts)
        return tuple(self.score_optional(text) for text in span_texts)


@dataclass(frozen=True)
class V3EdgeEvidence:
    valid: bool
    reasons: Tuple[str, ...]
    graph_valid: bool
    roundtrip_valid: bool
    selected_beam: int


def classify_edge_evidence(
    lattice: Lattice,
    recovery_records: Mapping[str, dict],
    span_records: Mapping[str, AMRRecord],
) -> Tuple[V3EdgeEvidence, ...]:
    evidence = []
    for edge in lattice.edges:
        normalized = normalize_text(edge.span_text)
        digest = text_sha256(normalized)
        recovery = recovery_records.get(digest)
        if recovery is None:
            raise RuntimeError(
                "Lattice edge lacks bound parser recovery evidence: "
                f"{digest}"
            )
        record = span_records.get(normalized)
        if record is None:
            raise RuntimeError(
                "Lattice edge lacks bound AMR audit record: "
                f"{digest}"
            )
        if record.text_sha256 != digest:
            raise RuntimeError(f"AMR audit record identity differs: {digest}")
        selected_beam = recovery["selected_beam"]
        reasons = []
        if any(
            event["level"] in ("ERROR", "CRITICAL")
            and event["beam_index"] == selected_beam
            for event in recovery["events"]
        ):
            reasons.append("selected_beam_error")
        if not record.graph_valid:
            reasons.append("graph_invalid")
        if not record.roundtrip_valid:
            reasons.append("roundtrip_invalid")
        evidence.append(
            V3EdgeEvidence(
                valid=not reasons,
                reasons=tuple(reasons),
                graph_valid=record.graph_valid,
                roundtrip_valid=record.roundtrip_valid,
                selected_beam=selected_beam,
            )
        )
    return tuple(evidence)


class SwanBankLatticeDetectorV3:
    """Optimize only over recovery-bound, PENMAN-valid candidate edges."""

    def __init__(
        self,
        builder: ElasticBoundaryProposerV3,
        scorer: SwanAMRBankScorerV3,
        score_center: float = 0.65,
        green_threshold: float = 0.75,
    ) -> None:
        self.builder = builder
        self.scorer = scorer
        self.optimizer = CountStatePathOptimizer(
            score_center=score_center,
            green_threshold=green_threshold,
        )
        config = {
            "capitalization_gap_words": builder.capitalization_gap_words,
            "detector_version": ELASTIC_V3_VERSION,
            "green_threshold": green_threshold,
            "invalid_edge_policy": INVALID_EDGE_POLICY,
            "long_gap_stride_words": builder.long_gap_stride_words,
            "max_boundary_hops": builder.max_boundary_hops,
            "max_span_words": builder.max_span_words,
            "score_center": score_center,
            "scorer_fingerprint": scorer.fingerprint,
        }
        self.fingerprint = "swan-bank-detector:" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def detect_lattice_with_evidence(
        self,
        lattice: Lattice,
        recovery_records: Mapping[str, dict],
    ) -> Tuple[V2PathResult, Tuple[V3EdgeEvidence, ...]]:
        matches = self.scorer.score_many(
            tuple(edge.span_text for edge in lattice.edges)
        )
        evidence = classify_edge_evidence(
            lattice,
            recovery_records,
            self.scorer._records,  # pylint: disable=protected-access
        )
        result = self.optimizer.optimize(
            lattice,
            matches,
            valid_edges=tuple(item.valid for item in evidence),
        )
        return result, evidence

    def detect_lattice_with_recovery(
        self,
        lattice: Lattice,
        recovery_records: Mapping[str, dict],
    ) -> V2PathResult:
        result, _ = self.detect_lattice_with_evidence(
            lattice,
            recovery_records,
        )
        return result

    def detect_with_recovery(
        self,
        text: str,
        recovery_records: Mapping[str, dict],
    ) -> V2PathResult:
        return self.detect_lattice_with_recovery(
            self.builder.build_lattice(text),
            recovery_records,
        )
