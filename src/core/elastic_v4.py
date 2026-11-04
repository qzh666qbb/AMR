from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .elastic_v2 import CountStatePathOptimizer, DetectionUnit, V2PathResult
from .elastic_v3 import (
    ElasticBoundaryProposerV3,
    SwanAMRBankScorerV3,
    V3EdgeEvidence,
    classify_edge_evidence,
)
from .models import Lattice

ELASTIC_V4_VERSION = "swan-bank-lattice-detector:v4-hop-penalty"

class HopPenalizedOptimizer(CountStatePathOptimizer):
    def __init__(self, score_center: float, green_threshold: float, p2: float = 0.18, p3: float = 0.18):
        super().__init__(score_center, green_threshold)
        self.p2 = p2
        self.p3 = p3

    def optimize(self, lattice: Lattice, matches: Sequence[Optional[object]], valid_edges: Sequence[bool]) -> V2PathResult:
        invalid_edge_count = len(valid_edges) - sum(valid_edges)
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

        outgoing = [[] for _ in range(lattice.node_count)]
        for edge, match, valid in zip(lattice.edges, matches, valid_edges):
            if valid and match is not None:
                outgoing[edge.start].append((edge, match))

        states = [{} for _ in range(lattice.node_count)]
        back = {}
        states[0][0] = 0.0
        epsilon = 1e-12

        for node in range(lattice.node_count):
            for count, raw_utility in tuple(states[node].items()):
                for edge, match in outgoing[node]:
                    pen = 0.0
                    if edge.boundary_hops == 2:
                        pen = self.p2
                    elif edge.boundary_hops >= 3:
                        pen = self.p3

                    edge_utility = match.score - self.score_center - pen
                    next_count = count + 1
                    candidate = raw_utility + edge_utility
                    current = states[edge.end].get(next_count)
                    if current is not None and candidate <= current + epsilon:
                        continue
                    states[edge.end][next_count] = candidate
                    back[(edge.end, next_count)] = (
                        node,
                        count,
                        DetectionUnit(edge=edge, match=match, utility=edge_utility),
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

        best_count, best_raw, best_statistic = None, None, None
        for count, raw_utility in states[final_node].items():
            statistic = raw_utility / math.sqrt(count)
            if best_statistic is None or statistic > best_statistic + epsilon:
                best_count, best_raw, best_statistic = count, raw_utility, statistic

        units = []
        state = (final_node, best_count)
        while state != (0, 0):
            node, count, unit = back[state]
            units.append(unit)
            state = (node, count)
        units.reverse()

        return V2PathResult(
            status="ok",
            statistic=best_statistic,
            raw_utility=best_raw,
            unit_count=best_count,
            green_count=sum(u.match.score >= self.green_threshold for u in units),
            units=tuple(units),
            invalid_edge_count=invalid_edge_count,
        )


class SwanBankLatticeDetectorV4:
    """Optimize with hop penalties over recovery-bound, PENMAN-valid candidate edges."""

    def __init__(
        self,
        builder: ElasticBoundaryProposerV3,
        scorer: SwanAMRBankScorerV3,
        score_center: float = 0.65,
        green_threshold: float = 0.75,
        p2: float = 0.18,
        p3: float = 0.18,
    ) -> None:
        self.builder = builder
        self.scorer = scorer
        self.optimizer = HopPenalizedOptimizer(
            score_center=score_center,
            green_threshold=green_threshold,
            p2=p2,
            p3=p3,
        )
        config = {
            "capitalization_gap_words": builder.capitalization_gap_words,
            "detector_version": ELASTIC_V4_VERSION,
            "green_threshold": green_threshold,
            "hop_penalty_p2": p2,
            "hop_penalty_p3": p3,
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
