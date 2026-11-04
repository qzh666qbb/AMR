from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from .models import Boundary, Lattice, LatticeEdge


_HARD_BOUNDARY_RE = re.compile(r'[.!?]+(?:["\')\]]*)\s+')
_SOFT_PUNCTUATION_RE = re.compile(r"[;:]\s+")
_COORDINATION_RE = re.compile(
    r"\s+(?=(?:and|but|or|however|therefore|meanwhile|then|because|although|while)\b)",
    re.IGNORECASE,
)


class LatticeBuilder:
    """Build a bounded DAG over plausible sentence boundaries."""

    def __init__(
        self,
        max_boundary_hops: int = 3,
        max_span_words: int = 80,
        include_soft_boundaries: bool = True,
    ) -> None:
        if max_boundary_hops < 1:
            raise ValueError("max_boundary_hops must be positive")
        if max_span_words < 1:
            raise ValueError("max_span_words must be positive")
        self.max_boundary_hops = max_boundary_hops
        self.max_span_words = max_span_words
        self.include_soft_boundaries = include_soft_boundaries

    @staticmethod
    def _offsets(pattern: re.Pattern, text: str) -> Iterable[int]:
        for match in pattern.finditer(text):
            yield match.end()

    def candidate_boundaries(self, text: str) -> Tuple[Boundary, ...]:
        normalized = text.strip()
        if not normalized:
            return (Boundary(index=0, char_offset=0, kind="terminal"),)

        kinds = {0: "terminal", len(normalized): "terminal"}
        for offset in self._offsets(_HARD_BOUNDARY_RE, normalized):
            kinds[offset] = "hard"

        if self.include_soft_boundaries:
            for offset in self._offsets(_SOFT_PUNCTUATION_RE, normalized):
                kinds.setdefault(offset, "soft")
            for offset in self._offsets(_COORDINATION_RE, normalized):
                kinds.setdefault(offset, "soft")

        offsets = sorted(kinds)
        return tuple(
            Boundary(index=index, char_offset=offset, kind=kinds[offset])
            for index, offset in enumerate(offsets)
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


def edge_texts(lattice: Lattice) -> Sequence[str]:
    return tuple(edge.span_text for edge in lattice.edges)
