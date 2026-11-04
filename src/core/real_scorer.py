from __future__ import annotations

from typing import Dict, Iterable

from .amr_extractor import AMRExtractor
from .models import AMRRecord
from .sqlite_cache import normalize_text


def _jaccard(left, right) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


class RealScorer:
    """Lightweight AMR skeleton scorer used by the lattice aligner."""

    def __init__(
        self,
        extractor: AMRExtractor,
        concept_weight: float = 0.7,
        triple_weight: float = 0.3,
    ) -> None:
        if concept_weight < 0 or triple_weight < 0:
            raise ValueError("Scorer weights must be non-negative")
        total = concept_weight + triple_weight
        if total <= 0:
            raise ValueError("At least one scorer weight must be positive")
        self.extractor = extractor
        self.concept_weight = concept_weight / total
        self.triple_weight = triple_weight / total
        self._records: Dict[str, AMRRecord] = {}

    def prepare(self, texts: Iterable[str]) -> None:
        self._records.update(self.extractor.extract_many(texts))

    def _record(self, text: str) -> AMRRecord:
        normalized = normalize_text(text)
        record = self._records.get(normalized)
        if record is None:
            record = self.extractor.extract(normalized)
            self._records[normalized] = record
        return record

    def score(self, span_text: str, target_text: str) -> float:
        span = self._record(span_text)
        target = self._record(target_text)
        concept_score = _jaccard(span.concepts, target.concepts)
        triple_score = _jaccard(span.triples, target.triples)
        return (
            self.concept_weight * concept_score
            + self.triple_weight * triple_score
        )
