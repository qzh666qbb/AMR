"""Versioned empty-path candidate; the historical v4 scorer is unchanged."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Sequence

from .bank_scorer import TemplateFeatures, _multiset_f1, template_features
from .elastic_v3 import SwanAMRBankScorerV3
from .elastic_v4 import HopPenalizedOptimizer
from .models import BankMatch


PATH0_VERSION = "swan-path-empty-empty-zero:v1"


class AllBankScorer(SwanAMRBankScorerV3):
    """Expose every template score without changing the control fingerprint."""

    empty_path_zero = False

    def all_matches(self, candidate: TemplateFeatures) -> tuple:
        matches = []
        for index, bank in enumerate(self._bank_features):
            concept = _multiset_f1(candidate.concepts, bank.concepts)
            relation = _multiset_f1(candidate.relations, bank.relations)
            path = (
                0.0
                if self.empty_path_zero and not candidate.paths and not bank.paths
                else _multiset_f1(candidate.paths, bank.paths)
            )
            score = (
                self.concept_weight * concept
                + self.relation_weight * relation
                + self.path_weight * path
            )
            if not all(math.isfinite(value) for value in (concept, relation, path, score)):
                raise ValueError("Non-finite Bank component")
            matches.append(BankMatch(index, score, concept, relation, path))
        return tuple(matches)

    def score_features(self, candidate: TemplateFeatures) -> BankMatch:
        # max retains the first template on a tie, as in the original scorer.
        return max(self.all_matches(candidate), key=lambda match: match.score)

    def audited_matches(self, amr: str) -> tuple:
        features = template_features(amr)
        matches = self.all_matches(features)
        ranked = sorted(matches, key=lambda match: (-match.score, match.template_index))
        return matches, {
            "concept_count": sum(features.concepts.values()),
            "relation_count": sum(features.relations.values()),
            "path_count": sum(features.paths.values()),
            "best": asdict(ranked[0]),
            "second": asdict(ranked[1]) if len(ranked) > 1 else None,
        }


class EmptyPathZeroScorer(AllBankScorer):
    empty_path_zero = True

    def __init__(self, extractor, bank_path: str):
        super().__init__(extractor, bank_path)
        config = {
            "base_scorer_fingerprint": self.fingerprint,
            "path_policy": PATH0_VERSION,
        }
        self.fingerprint = "swan-bank-scorer-path0:" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()


class CheckedHopOptimizer(HopPenalizedOptimizer):
    """Fail explicitly on malformed inputs before entering the existing DP."""

    def __init__(self, score_center=0.65, green_threshold=0.75, p2=0.18, p3=0.18):
        values = (score_center, green_threshold, p2, p3)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite optimizer configuration")
        if not 0 <= score_center <= 1 or not 0 <= green_threshold <= 1:
            raise ValueError("Threshold out of range")
        if p2 < 0 or p3 < 0:
            raise ValueError("Negative hop penalty")
        super().__init__(score_center, green_threshold, p2, p3)

    def optimize(self, lattice, matches: Sequence, valid_edges: Sequence):
        if len(matches) != len(lattice.edges) or len(valid_edges) != len(lattice.edges):
            raise ValueError("Edge/match/validity cardinality mismatch")
        for edge, match, valid in zip(lattice.edges, matches, valid_edges):
            if not 0 <= edge.start < edge.end < lattice.node_count:
                raise ValueError("Lattice edge is not a forward DAG edge")
            if valid and match is None:
                raise ValueError("Valid edge lacks a Bank match")
            if match is not None and not all(
                math.isfinite(value)
                for value in (match.score, match.concept_score, match.relation_score, match.path_score)
            ):
                raise ValueError("Non-finite edge score")
        result = super().optimize(lattice, matches, valid_edges)
        if not result.abstained and not all(
            math.isfinite(value) for value in (result.statistic, result.raw_utility)
        ):
            raise ValueError("Non-finite document statistic")
        return result
