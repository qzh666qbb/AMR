from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Triple = Tuple[str, str, str]
AMR_AUDIT_VERSION = "penman-audit:v1"


@dataclass(frozen=True)
class Boundary:
    index: int
    char_offset: int
    kind: str


@dataclass(frozen=True)
class LatticeEdge:
    start: int
    end: int
    span_text: str
    boundary_hops: int


@dataclass(frozen=True)
class Lattice:
    text: str
    boundaries: Tuple[Boundary, ...]
    edges: Tuple[LatticeEdge, ...]

    @property
    def node_count(self) -> int:
        return len(self.boundaries)


@dataclass(frozen=True)
class AMRRecord:
    text: str
    text_sha256: str
    parser_fingerprint: str
    amr: str
    concepts: Tuple[str, ...]
    triples: Tuple[Triple, ...]
    audit_version: str
    penman_warnings: Tuple[str, ...]
    graph_valid: bool
    roundtrip_valid: bool


@dataclass(frozen=True)
class AlignmentStep:
    operation: str
    edge: Optional[LatticeEdge]
    target_index: Optional[int]
    match_score: float


@dataclass(frozen=True)
class AlignmentResult:
    raw_score: float
    normalized_score: float
    steps: Tuple[AlignmentStep, ...]

    @property
    def matched_steps(self) -> Tuple[AlignmentStep, ...]:
        return tuple(step for step in self.steps if step.operation == "match")


@dataclass(frozen=True)
class BankMatch:
    template_index: int
    score: float
    concept_score: float
    relation_score: float
    path_score: float


@dataclass(frozen=True)
class DetectionUnit:
    edge: LatticeEdge
    match: BankMatch
    utility: float


@dataclass(frozen=True)
class DetectionResult:
    statistic: float
    raw_utility: float
    unit_count: int
    green_count: int
    units: Tuple[DetectionUnit, ...]


@dataclass(frozen=True)
class CalibrationResult:
    statistic: float
    p_value: float
    alpha: float
    is_watermarked: bool
    null_count: int
    scope: str
    bucket: str
