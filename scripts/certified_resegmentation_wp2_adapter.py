"""Model-free WP2 adapter and deterministic cohort contracts.

The adapter freezes an input's original unitization once, exposes only direct
unit-list candidates thereafter, and bridges rendered adjacent units to the
existing exact pairwise certificate implementation.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Callable, Iterable, Optional, Tuple, Union

from certified_resegmentation import (
    PairwiseCertificateProblem,
    build_pair_scores,
)


RawValue = Union[str, list[str]]
AtomSpan = Tuple[int, int]
Boundaries = Tuple[int, ...]
VALID_SYSTEM_IDS = frozenset({"semstamp", "simmark"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATOM_RE = re.compile(r"\S+")
_LIST_DOMAIN = b"\xffWP2-LIST\0"


def _validate_system_id(system_id: str) -> None:
    if system_id not in VALID_SYSTEM_IDS:
        raise ValueError(
            "system_id must be exactly 'semstamp' or 'simmark'"
        )


def _raw_field_type(value: RawValue) -> str:
    if isinstance(value, str):
        return "string"
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "list"
    raise TypeError("raw value must be a string or a list of strings")


def _raw_bytes(value: RawValue) -> bytes:
    field_type = _raw_field_type(value)
    if field_type == "string":
        return value.encode("utf-8")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _LIST_DOMAIN + encoded


def hash_raw_value(value: RawValue) -> str:
    """Hash a raw field with a collision-free type domain for list fields."""
    return hashlib.sha256(_raw_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_units(units: Tuple[str, ...]) -> None:
    if not units:
        raise ValueError("the original partition must contain at least one unit")
    for unit in units:
        if not isinstance(unit, str):
            raise TypeError("every original unit must be a string")
        if not unit or unit != unit.strip():
            raise ValueError(
                "every original unit must be nonempty and equal to unit.strip()"
            )
        if _ATOM_RE.search(unit) is None:
            raise ValueError("every original unit must contain an atom")


@dataclass(frozen=True)
class CanonicalDocument:
    """A once-segmented document represented by immutable atom spans."""

    raw_field_type: str
    raw_text_sha256: str
    original_units: Tuple[str, ...]
    canonical_tape: str
    canonical_tape_sha256: str
    atom_spans: Tuple[AtomSpan, ...]
    original_boundaries: Boundaries

    def __post_init__(self) -> None:
        if self.raw_field_type not in {"string", "list"}:
            raise ValueError("raw_field_type must be 'string' or 'list'")
        if _SHA256_RE.fullmatch(self.raw_text_sha256) is None:
            raise ValueError("raw_text_sha256 must be lowercase SHA256")
        if _SHA256_RE.fullmatch(self.canonical_tape_sha256) is None:
            raise ValueError("canonical_tape_sha256 must be lowercase SHA256")
        _validate_units(self.original_units)
        if self.canonical_tape != " ".join(self.original_units):
            raise ValueError("canonical_tape must be the ASCII-space unit join")
        if self.canonical_tape_sha256 != _sha256_text(self.canonical_tape):
            raise ValueError("canonical_tape_sha256 does not match the tape")

        expected_spans = tuple(
            (match.start(), match.end())
            for match in _ATOM_RE.finditer(self.canonical_tape)
        )
        if self.atom_spans != expected_spans:
            raise ValueError("atom_spans must be maximal non-whitespace spans")

        cumulative = 0
        expected_boundaries = []
        for unit in self.original_units[:-1]:
            cumulative += len(_ATOM_RE.findall(unit))
            expected_boundaries.append(cumulative)
        if self.original_boundaries != tuple(expected_boundaries):
            raise ValueError("original_boundaries do not match original units")
        if self.render_partition(self.original_boundaries) != self.original_units:
            raise ValueError("original units do not map exactly to canonical tape")

    @property
    def atom_count(self) -> int:
        return len(self.atom_spans)

    @property
    def unit_count(self) -> int:
        return len(self.original_units)

    def render_unit(self, start: int, end: int) -> str:
        """Render an atom half-open span as its exact tape substring."""
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise TypeError("atom endpoints must be integers")
        if not 0 <= start < end <= self.atom_count:
            raise ValueError("unit atom span must satisfy 0 <= start < end <= n")
        char_start = self.atom_spans[start][0]
        char_end = self.atom_spans[end - 1][1]
        return self.canonical_tape[char_start:char_end]

    def render_partition(
        self, boundaries: Iterable[int]
    ) -> Tuple[str, ...]:
        """Render a strictly increasing set of internal atom boundaries."""
        boundaries = tuple(boundaries)
        if any(
            isinstance(boundary, bool) or not isinstance(boundary, int)
            for boundary in boundaries
        ):
            raise TypeError("partition boundaries must be integers")
        if tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("partition boundaries must be strictly increasing")
        if any(
            boundary <= 0 or boundary >= self.atom_count
            for boundary in boundaries
        ):
            raise ValueError("partition boundaries must be internal atom gaps")

        endpoints = (0, *boundaries, self.atom_count)
        return tuple(
            self.render_unit(endpoints[index], endpoints[index + 1])
            for index in range(len(endpoints) - 1)
        )

    def preprocess_unit(self, system_id: str, start: int, end: int) -> str:
        """Apply only the frozen detector-side leading-space convention."""
        _validate_system_id(system_id)
        rendered = self.render_unit(start, end)
        if system_id == "simmark" and start > 0:
            return " " + rendered
        return rendered

    def preprocess_partition(
        self,
        system_id: str,
        boundaries: Optional[Iterable[int]] = None,
    ) -> Tuple[str, ...]:
        _validate_system_id(system_id)
        selected = (
            self.original_boundaries
            if boundaries is None
            else tuple(boundaries)
        )
        self.render_partition(selected)
        endpoints = (0, *selected, self.atom_count)
        return tuple(
            self.preprocess_unit(
                system_id,
                endpoints[index],
                endpoints[index + 1],
            )
            for index in range(len(endpoints) - 1)
        )


def canonicalize(
    value: RawValue,
    sentence_splitter: Optional[Callable[[str], list[str]]] = None,
) -> CanonicalDocument:
    """Freeze P0 once and map it exactly to a canonical atom tape."""
    field_type = _raw_field_type(value)
    if field_type == "string":
        if sentence_splitter is None or not callable(sentence_splitter):
            raise ValueError("string fields require an injected sentence splitter")
        split_units = sentence_splitter(value)
        if not isinstance(split_units, list):
            raise TypeError("sentence splitter must return a list of strings")
        units = tuple(split_units)
    else:
        units = tuple(value)
    _validate_units(units)

    tape = " ".join(units)
    atom_spans = tuple(
        (match.start(), match.end()) for match in _ATOM_RE.finditer(tape)
    )
    cumulative = 0
    original_boundaries = []
    for unit in units[:-1]:
        cumulative += len(_ATOM_RE.findall(unit))
        original_boundaries.append(cumulative)

    return CanonicalDocument(
        raw_field_type=field_type,
        raw_text_sha256=hash_raw_value(value),
        original_units=units,
        canonical_tape=tape,
        canonical_tape_sha256=_sha256_text(tape),
        atom_spans=atom_spans,
        original_boundaries=tuple(original_boundaries),
    )


def build_document_pair_scores(
    document: CanonicalDocument,
    system_id: str,
    min_unit_atoms: int,
    max_unit_atoms: int,
    scorer: Callable[[str, str], float],
    allowed_boundaries: Optional[frozenset[int]] = None,
):
    """Score every admissible adjacent rendered-unit pair."""
    _validate_system_id(system_id)
    if not callable(scorer):
        raise TypeError("scorer must be callable")

    def score_span_pair(start: int, middle: int, end: int) -> float:
        previous = document.preprocess_unit(system_id, start, middle)
        current = document.preprocess_unit(system_id, middle, end)
        return scorer(previous, current)

    return build_pair_scores(
        document.atom_count,
        min_unit_atoms,
        max_unit_atoms,
        score_span_pair,
        allowed_boundaries,
    )


def build_document_pairwise_problem(
    document: CanonicalDocument,
    system_id: str,
    min_unit_atoms: int,
    max_unit_atoms: int,
    max_boundary_moves: int,
    scorer: Callable[[str, str], float],
    threshold: float,
    normalization: Optional[float] = None,
    bias: float = 0.0,
    allowed_boundaries: Optional[frozenset[int]] = None,
) -> PairwiseCertificateProblem:
    """Bridge a rendered text scorer to the exact pairwise DP contract."""
    if document.unit_count < 2:
        raise ValueError("pairwise certificates require at least two units")
    allowed = (
        frozenset(range(1, document.atom_count))
        if allowed_boundaries is None
        else frozenset(allowed_boundaries)
    )
    edge_count = document.unit_count - 1
    return PairwiseCertificateProblem(
        token_count=document.atom_count,
        original_boundaries=document.original_boundaries,
        min_unit_words=min_unit_atoms,
        max_unit_words=max_unit_atoms,
        max_boundary_moves=max_boundary_moves,
        pair_scores=build_document_pair_scores(
            document,
            system_id,
            min_unit_atoms,
            max_unit_atoms,
            scorer,
            allowed,
        ),
        threshold=threshold,
        normalization=(
            float(edge_count) if normalization is None else normalization
        ),
        bias=bias,
        allowed_boundaries=allowed,
    )


@dataclass(frozen=True)
class SourceRecord:
    """Minimal score-blind identity needed for deterministic cohort splits."""

    system_id: str
    data_revision: str
    row_index: int
    raw_field_type: str
    raw_text_sha256: str
    source_repository: str = ""
    relative_path: str = ""

    def __post_init__(self) -> None:
        _validate_system_id(self.system_id)
        if not isinstance(self.data_revision, str) or not self.data_revision:
            raise ValueError("data_revision must be a nonempty string")
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a nonnegative integer")
        if self.raw_field_type not in {"string", "list"}:
            raise ValueError("raw_field_type must be 'string' or 'list'")
        if _SHA256_RE.fullmatch(self.raw_text_sha256) is None:
            raise ValueError("raw_text_sha256 must be lowercase SHA256")
        if not isinstance(self.source_repository, str):
            raise TypeError("source_repository must be a string")
        if not isinstance(self.relative_path, str):
            raise TypeError("relative_path must be a string")


def make_source_record(
    system_id: str,
    data_revision: str,
    row_index: int,
    raw_value: RawValue,
    source_repository: str = "",
    relative_path: str = "",
) -> SourceRecord:
    return SourceRecord(
        system_id=system_id,
        data_revision=data_revision,
        row_index=row_index,
        raw_field_type=_raw_field_type(raw_value),
        raw_text_sha256=hash_raw_value(raw_value),
        source_repository=source_repository,
        relative_path=relative_path,
    )


def cohort_sort_hash(record: SourceRecord) -> str:
    """Compute the protocol H with literal field concatenation."""
    payload = (
        record.system_id.encode("utf-8")
        + record.data_revision.encode("utf-8")
        + str(record.row_index).encode("ascii")
        + record.raw_text_sha256.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CohortAttrition:
    null_input_count: int
    null_unique_count: int
    null_duplicate_count: int
    positive_input_count: int
    positive_retained_count: int
    positive_overlap_count: int


@dataclass(frozen=True)
class CohortSplit:
    detector_fit: Tuple[SourceRecord, ...]
    certificate_calibration: Tuple[SourceRecord, ...]
    independent_test: Tuple[SourceRecord, ...]
    positive_retained: Tuple[SourceRecord, ...]
    positive_overlap_excluded: Tuple[SourceRecord, ...]
    attrition: CohortAttrition


def _identity_key(record: SourceRecord):
    return (
        record.row_index,
        record.source_repository,
        record.relative_path,
        record.raw_field_type,
        record.raw_text_sha256,
    )


def _fraction(value, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite fraction")
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite fraction") from error
    if fraction < 0 or fraction > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return fraction


def _validate_records(
    records: Iterable[SourceRecord],
    system_id: str,
    data_revision: str,
    label: str,
) -> Tuple[SourceRecord, ...]:
    materialized = tuple(records)
    for record in materialized:
        if not isinstance(record, SourceRecord):
            raise TypeError(f"{label} must contain SourceRecord values")
        if record.system_id != system_id:
            raise ValueError(f"{label} contains a mismatched system_id")
        if record.data_revision != data_revision:
            raise ValueError(f"{label} contains a mismatched data_revision")
    return materialized


def split_null_cohort(
    system_id: str,
    data_revision: str,
    null_records: Iterable[SourceRecord],
    positive_records: Iterable[SourceRecord],
    detector_fit_fraction=0.2,
    calibration_fraction=0.4,
) -> CohortSplit:
    """Deduplicate null, exclude positive overlap, then split by stable H."""
    _validate_system_id(system_id)
    if not isinstance(data_revision, str) or not data_revision:
        raise ValueError("data_revision must be a nonempty string")
    fit_fraction = _fraction(detector_fit_fraction, "detector_fit_fraction")
    calibration_fraction = _fraction(
        calibration_fraction, "calibration_fraction"
    )
    if fit_fraction + calibration_fraction > 1:
        raise ValueError("cohort fractions must sum to at most one")

    null_input = _validate_records(
        null_records, system_id, data_revision, "null_records"
    )
    positive_input = _validate_records(
        positive_records, system_id, data_revision, "positive_records"
    )

    unique_null_by_hash = {}
    for record in null_input:
        existing = unique_null_by_hash.get(record.raw_text_sha256)
        if existing is None or _identity_key(record) < _identity_key(existing):
            unique_null_by_hash[record.raw_text_sha256] = record
    unique_null = tuple(
        sorted(
            unique_null_by_hash.values(),
            key=lambda record: (
                cohort_sort_hash(record),
                record.row_index,
                _identity_key(record),
            ),
        )
    )

    null_hashes = frozenset(unique_null_by_hash)
    positive_overlap = tuple(sorted(
        (
            record
            for record in positive_input
            if record.raw_text_sha256 in null_hashes
        ),
        key=_identity_key,
    ))
    positive_retained = tuple(sorted(
        (
            record
            for record in positive_input
            if record.raw_text_sha256 not in null_hashes
        ),
        key=_identity_key,
    ))

    total = len(unique_null)
    fit_count = (total * fit_fraction.numerator) // fit_fraction.denominator
    calibration_count = (
        total * calibration_fraction.numerator
    ) // calibration_fraction.denominator
    calibration_end = fit_count + calibration_count

    return CohortSplit(
        detector_fit=unique_null[:fit_count],
        certificate_calibration=unique_null[
            fit_count:calibration_end
        ],
        independent_test=unique_null[calibration_end:],
        positive_retained=positive_retained,
        positive_overlap_excluded=positive_overlap,
        attrition=CohortAttrition(
            null_input_count=len(null_input),
            null_unique_count=total,
            null_duplicate_count=len(null_input) - total,
            positive_input_count=len(positive_input),
            positive_retained_count=len(positive_retained),
            positive_overlap_count=len(positive_overlap),
        ),
    )
