from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    BankMatch,
    Boundary,
    DetectionUnit,
    Lattice,
    LatticeEdge,
)
from .sqlite_cache import normalize_text, text_sha256


ELASTIC_V2_VERSION = "swan-bank-lattice-detector:v2-preflight"

_HARD_BOUNDARY_RE = re.compile(r'[.!?]+(?:["\')\]]*)\s+')
_LINE_BOUNDARY_RE = re.compile(r"\r?\n[ \t]*")
_PUNCTUATION_BOUNDARY_RE = re.compile(r"[;:,]+\s+")
_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:and|but|or|however|therefore|meanwhile|then|because|"
    r"although|while|whereas|yet|so)\b)",
    re.IGNORECASE,
)
_CAPITALIZATION_RE = re.compile(r"""\s+(?=["'(]*[A-Z])""")
_CAPITALIZED_COMMA_RE = re.compile(
    r"""\s+(?=["'(]*[A-Z][A-Za-z'-]*,)"""
)
_NONSPACE_RE = re.compile(r"\S+")


class ElasticBoundaryProposerV2:
    """Build a score-blind lattice that remains usable after punctuation edits."""

    def __init__(
        self,
        max_boundary_hops: int = 3,
        max_span_words: int = 80,
        capitalization_gap_words: int = 12,
        long_gap_stride_words: int = 40,
    ) -> None:
        if max_boundary_hops < 1:
            raise ValueError("max_boundary_hops must be positive")
        if max_span_words < 1:
            raise ValueError("max_span_words must be positive")
        if capitalization_gap_words < 1:
            raise ValueError("capitalization_gap_words must be positive")
        if not 1 <= long_gap_stride_words <= max_span_words:
            raise ValueError(
                "long_gap_stride_words must be between one and max_span_words"
            )
        self.max_boundary_hops = max_boundary_hops
        self.max_span_words = max_span_words
        self.capitalization_gap_words = capitalization_gap_words
        self.long_gap_stride_words = long_gap_stride_words

    @staticmethod
    def _record(
        kinds: Dict[int, str],
        pattern: re.Pattern,
        text: str,
        kind: str,
    ) -> None:
        for match in pattern.finditer(text):
            kinds.setdefault(match.end(), kind)

    def _add_capitalization_fallbacks(
        self,
        kinds: Dict[int, str],
        text: str,
        primary_offsets: Sequence[int],
    ) -> None:
        candidates = tuple(
            match.end() for match in _CAPITALIZATION_RE.finditer(text)
        )
        for left, right in zip(primary_offsets, primary_offsets[1:]):
            if len(text[left:right].split()) <= self.capitalization_gap_words:
                continue
            for offset in candidates:
                if left < offset < right:
                    kinds.setdefault(offset, "capitalization")

        for match in _CAPITALIZED_COMMA_RE.finditer(text):
            kinds.setdefault(match.end(), "capitalized_comma")

    def _add_long_gap_fallbacks(
        self,
        kinds: Dict[int, str],
        text: str,
    ) -> None:
        offsets = sorted(kinds)
        for left, right in zip(offsets, offsets[1:]):
            tokens = tuple(_NONSPACE_RE.finditer(text, left, right))
            if len(tokens) <= self.max_span_words:
                continue
            for index in range(
                self.long_gap_stride_words,
                len(tokens),
                self.long_gap_stride_words,
            ):
                kinds.setdefault(tokens[index].start(), "length_fallback")

    def candidate_boundaries(self, text: str) -> Tuple[Boundary, ...]:
        normalized = text.strip()
        if not normalized:
            return (Boundary(index=0, char_offset=0, kind="terminal"),)

        kinds = {0: "terminal", len(normalized): "terminal"}
        self._record(kinds, _HARD_BOUNDARY_RE, normalized, "hard")
        self._record(kinds, _LINE_BOUNDARY_RE, normalized, "line")
        self._record(
            kinds,
            _PUNCTUATION_BOUNDARY_RE,
            normalized,
            "punctuation",
        )
        self._record(kinds, _CLAUSE_BOUNDARY_RE, normalized, "clause")

        primary_offsets = tuple(sorted(kinds))
        self._add_capitalization_fallbacks(
            kinds,
            normalized,
            primary_offsets,
        )
        self._add_long_gap_fallbacks(kinds, normalized)

        return tuple(
            Boundary(index=index, char_offset=offset, kind=kinds[offset])
            for index, offset in enumerate(sorted(kinds))
        )

    def build_lattice(self, text: str) -> Lattice:
        normalized = text.strip()
        boundaries = self.candidate_boundaries(normalized)
        if len(boundaries) == 1:
            return Lattice(text=normalized, boundaries=boundaries, edges=())

        edges: List[LatticeEdge] = []
        for start in range(len(boundaries) - 1):
            max_end = min(
                len(boundaries) - 1,
                start + self.max_boundary_hops,
            )
            for end in range(start + 1, max_end + 1):
                left = boundaries[start].char_offset
                right = boundaries[end].char_offset
                span = normalized[left:right].strip()
                if not span:
                    continue
                if len(span.split()) > self.max_span_words:
                    break
                edges.append(
                    LatticeEdge(
                        start=start,
                        end=end,
                        span_text=span,
                        boundary_hops=end - start,
                    )
                )
        return Lattice(
            text=normalized,
            boundaries=boundaries,
            edges=tuple(edges),
        )


@dataclass(frozen=True)
class V2PathResult:
    status: str
    statistic: Optional[float]
    raw_utility: Optional[float]
    unit_count: int
    green_count: int
    units: Tuple[DetectionUnit, ...]
    invalid_edge_count: int
    abstain_reason: Optional[str] = None

    @property
    def abstained(self) -> bool:
        return self.status == "abstain"


class CountStatePathOptimizer:
    """Exactly maximize final utility/sqrt(N) over complete valid paths."""

    def __init__(
        self,
        score_center: float = 0.65,
        green_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= score_center <= 1.0:
            raise ValueError("score_center must be between zero and one")
        if not 0.0 <= green_threshold <= 1.0:
            raise ValueError("green_threshold must be between zero and one")
        self.score_center = score_center
        self.green_threshold = green_threshold

    def optimize(
        self,
        lattice: Lattice,
        matches: Sequence[Optional[BankMatch]],
        valid_edges: Optional[Sequence[bool]] = None,
    ) -> V2PathResult:
        if len(matches) != len(lattice.edges):
            raise ValueError("matches must align one-to-one with lattice edges")
        if valid_edges is None:
            validity = tuple(match is not None for match in matches)
        else:
            if len(valid_edges) != len(lattice.edges):
                raise ValueError(
                    "valid_edges must align one-to-one with lattice edges"
                )
            validity = tuple(
                bool(valid) and match is not None
                for valid, match in zip(valid_edges, matches)
            )

        invalid_edge_count = len(validity) - sum(validity)
        if lattice.node_count == 1:
            return V2PathResult(
                status="ok",
                statistic=0.0,
                raw_utility=0.0,
                unit_count=0,
                green_count=0,
                units=(),
                invalid_edge_count=invalid_edge_count,
            )

        outgoing: List[List[Tuple[LatticeEdge, BankMatch]]] = [
            [] for _ in range(lattice.node_count)
        ]
        for edge, match, valid in zip(lattice.edges, matches, validity):
            if valid and match is not None:
                outgoing[edge.start].append((edge, match))

        states: List[Dict[int, float]] = [
            {} for _ in range(lattice.node_count)
        ]
        back: Dict[
            Tuple[int, int],
            Tuple[int, int, DetectionUnit],
        ] = {}
        states[0][0] = 0.0
        epsilon = 1e-12

        for node in range(lattice.node_count):
            for count, raw_utility in tuple(states[node].items()):
                for edge, match in outgoing[node]:
                    edge_utility = match.score - self.score_center
                    next_count = count + 1
                    candidate = raw_utility + edge_utility
                    current = states[edge.end].get(next_count)
                    if current is not None and candidate <= current + epsilon:
                        continue
                    states[edge.end][next_count] = candidate
                    back[(edge.end, next_count)] = (
                        node,
                        count,
                        DetectionUnit(
                            edge=edge,
                            match=match,
                            utility=edge_utility,
                        ),
                    )

        final_node = lattice.node_count - 1
        if not states[final_node]:
            return V2PathResult(
                status="abstain",
                statistic=None,
                raw_utility=None,
                unit_count=0,
                green_count=0,
                units=(),
                invalid_edge_count=invalid_edge_count,
                abstain_reason="no_complete_valid_path",
            )

        best_count: Optional[int] = None
        best_raw: Optional[float] = None
        best_statistic: Optional[float] = None
        for count, raw_utility in states[final_node].items():
            statistic = raw_utility / math.sqrt(count)
            improves = (
                best_statistic is None
                or statistic > best_statistic + epsilon
            )
            ties_with_better_raw = (
                best_statistic is not None
                and abs(statistic - best_statistic) <= epsilon
                and (
                    best_raw is None
                    or raw_utility > best_raw + epsilon
                )
            )
            ties_with_fewer_units = (
                best_statistic is not None
                and best_raw is not None
                and best_count is not None
                and abs(statistic - best_statistic) <= epsilon
                and abs(raw_utility - best_raw) <= epsilon
                and count < best_count
            )
            if improves or ties_with_better_raw or ties_with_fewer_units:
                best_count = count
                best_raw = raw_utility
                best_statistic = statistic

        if best_count is None or best_raw is None or best_statistic is None:
            raise RuntimeError("Complete path state selection failed")

        units: List[DetectionUnit] = []
        state = (final_node, best_count)
        while state != (0, 0):
            pointer = back.get(state)
            if pointer is None:
                raise RuntimeError("Count-state DP backtrace is incomplete")
            previous_node, previous_count, unit = pointer
            units.append(unit)
            state = (previous_node, previous_count)
        units.reverse()

        return V2PathResult(
            status="ok",
            statistic=best_statistic,
            raw_utility=best_raw,
            unit_count=best_count,
            green_count=sum(
                unit.match.score >= self.green_threshold for unit in units
            ),
            units=tuple(units),
            invalid_edge_count=invalid_edge_count,
        )


def recovery_edge_validity(
    lattice: Lattice,
    recovery_records: Dict[str, dict],
) -> Tuple[bool, ...]:
    """Bind every edge to recovery evidence and reject selected-beam errors."""
    validity = []
    for edge in lattice.edges:
        digest = text_sha256(normalize_text(edge.span_text))
        record = recovery_records.get(digest)
        if record is None:
            raise RuntimeError(
                "Lattice edge lacks bound parser recovery evidence: "
                f"{digest}"
            )
        selected_beam = record["selected_beam"]
        selected_errors = [
            event
            for event in record["events"]
            if event["level"] in ("ERROR", "CRITICAL")
            and event["beam_index"] == selected_beam
        ]
        validity.append(not selected_errors)
    return tuple(validity)


class SwanBankLatticeDetectorV2:
    """Score v2 lattices and optimize only over recovery-valid edges."""

    def __init__(
        self,
        builder: ElasticBoundaryProposerV2,
        scorer,
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
            "detector_version": ELASTIC_V2_VERSION,
            "green_threshold": green_threshold,
            "long_gap_stride_words": builder.long_gap_stride_words,
            "max_boundary_hops": builder.max_boundary_hops,
            "max_span_words": builder.max_span_words,
            "score_center": score_center,
        }
        self.fingerprint = "swan-bank-detector:" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def detect_with_recovery(
        self,
        text: str,
        recovery_records: Dict[str, dict],
    ) -> V2PathResult:
        return self.detect_lattice_with_recovery(
            self.builder.build_lattice(text),
            recovery_records,
        )

    def detect_lattice_with_recovery(
        self,
        lattice: Lattice,
        recovery_records: Dict[str, dict],
    ) -> V2PathResult:
        matches = self.scorer.score_many(
            tuple(edge.span_text for edge in lattice.edges)
        )
        validity = recovery_edge_validity(lattice, recovery_records)
        return self.optimizer.optimize(lattice, matches, validity)
