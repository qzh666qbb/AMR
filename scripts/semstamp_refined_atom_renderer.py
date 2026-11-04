"""Raw-byte SemStamp renderer with NLTK-boundary-refined lexical atoms."""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, Tuple


_LEXICAL_RE = re.compile(r"\S+")


class RefinedAtomRendererError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}")


def _fail(code: str, detail: str):
    raise RefinedAtomRendererError(code, detail)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _codepoint_to_byte_offsets(text: str) -> Tuple[int, ...]:
    offsets = [0]
    size = 0
    for character in text:
        size += len(character.encode("utf-8"))
        offsets.append(size)
    return tuple(offsets)


@dataclass(frozen=True)
class LexicalSpan:
    codepoint_start: int
    codepoint_end: int
    byte_start: int
    byte_end: int
    payload_sha256: str


@dataclass(frozen=True)
class RefinedAtom:
    codepoint_start: int
    codepoint_end: int
    byte_start: int
    byte_end: int
    lexical_span_index: int
    payload_sha256: str


@dataclass(frozen=True)
class RefinedAtomDocument:
    raw_utf8_bytes: bytes
    raw_utf8_sha256: str
    decoded_text: str
    lexical_spans: Tuple[LexicalSpan, ...]
    atoms: Tuple[RefinedAtom, ...]
    refinement_cut_codepoints: Tuple[int, ...]
    original_units: Tuple[str, ...]
    original_unit_codepoint_spans: Tuple[Tuple[int, int], ...]
    original_unit_atom_spans: Tuple[Tuple[int, int], ...]
    original_boundaries: Tuple[int, ...]

    @property
    def lexical_span_count(self) -> int:
        return len(self.lexical_spans)

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def refinement_count(self) -> int:
        return self.atom_count - self.lexical_span_count

    @property
    def unit_count(self) -> int:
        return len(self.original_units)

    @property
    def prefix_gap(self) -> bytes:
        return self.raw_utf8_bytes[: self.atoms[0].byte_start]

    @property
    def suffix_gap(self) -> bytes:
        return self.raw_utf8_bytes[self.atoms[-1].byte_end :]

    @property
    def zero_byte_gap_boundaries(self) -> Tuple[int, ...]:
        return tuple(
            boundary
            for boundary in range(1, self.atom_count)
            if not self.gap_after(boundary)
        )

    def gap_after(self, atom_end: int) -> bytes:
        if not 1 <= atom_end < self.atom_count:
            raise ValueError("gap index must be an internal atom boundary")
        return self.raw_utf8_bytes[
            self.atoms[atom_end - 1].byte_end : self.atoms[atom_end].byte_start
        ]

    def render_unit_bytes(self, start: int, end: int) -> bytes:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= self.atom_count
        ):
            raise ValueError("unit span must satisfy 0 <= start < end <= atoms")
        return self.raw_utf8_bytes[
            self.atoms[start].byte_start : self.atoms[end - 1].byte_end
        ]

    def render_unit(self, start: int, end: int) -> str:
        return self.render_unit_bytes(start, end).decode("utf-8", "strict")

    def render_partition_bytes(
        self,
        boundaries: Iterable[int],
    ) -> Tuple[bytes, ...]:
        selected = tuple(boundaries)
        if (
            selected != tuple(sorted(set(selected)))
            or any(
                isinstance(boundary, bool)
                or not isinstance(boundary, int)
                or boundary <= 0
                or boundary >= self.atom_count
                for boundary in selected
            )
        ):
            raise ValueError("boundaries must be unique increasing atom gaps")
        endpoints = (0, *selected, self.atom_count)
        return tuple(
            self.render_unit_bytes(endpoints[index], endpoints[index + 1])
            for index in range(len(endpoints) - 1)
        )

    def render_partition(
        self,
        boundaries: Iterable[int],
    ) -> Tuple[str, ...]:
        return tuple(
            unit.decode("utf-8", "strict")
            for unit in self.render_partition_bytes(boundaries)
        )

    def reconstruct(self, boundaries: Iterable[int]) -> bytes:
        selected = tuple(boundaries)
        units = self.render_partition_bytes(selected)
        rebuilt = bytearray(self.prefix_gap)
        for index, unit in enumerate(units):
            rebuilt.extend(unit)
            if index < len(selected):
                rebuilt.extend(self.gap_after(selected[index]))
        rebuilt.extend(self.suffix_gap)
        return bytes(rebuilt)

    def public_evidence(self) -> dict:
        lexical_table = [
            {
                "codepoint_span": [
                    lexical.codepoint_start,
                    lexical.codepoint_end,
                ],
                "byte_span": [lexical.byte_start, lexical.byte_end],
                "payload_sha256": lexical.payload_sha256,
            }
            for lexical in self.lexical_spans
        ]
        atom_table = [
            {
                "codepoint_span": [atom.codepoint_start, atom.codepoint_end],
                "byte_span": [atom.byte_start, atom.byte_end],
                "lexical_span_index": atom.lexical_span_index,
                "payload_sha256": atom.payload_sha256,
            }
            for atom in self.atoms
        ]
        gap_table = [
            {
                "boundary": boundary,
                "bytes": len(self.gap_after(boundary)),
                "is_original_boundary": boundary in self.original_boundaries,
                "sha256": _sha256_bytes(self.gap_after(boundary)),
            }
            for boundary in range(1, self.atom_count)
        ]
        zero_byte_boundaries = list(self.zero_byte_gap_boundaries)
        rendered_units = self.render_partition_bytes(self.original_boundaries)
        return {
            "raw_utf8_sha256": self.raw_utf8_sha256,
            "raw_utf8_bytes": len(self.raw_utf8_bytes),
            "lexical_span_count": self.lexical_span_count,
            "atom_count": self.atom_count,
            "refinement_count": self.refinement_count,
            "zero_byte_gap_count": len(zero_byte_boundaries),
            "zero_byte_gaps_all_original": set(zero_byte_boundaries).issubset(
                self.original_boundaries
            ),
            "zero_byte_gap_boundaries_sha256": _canonical_digest(
                zero_byte_boundaries
            ),
            "unit_count": self.unit_count,
            "original_boundaries": list(self.original_boundaries),
            "refinement_cuts_sha256": _canonical_digest(
                list(self.refinement_cut_codepoints)
            ),
            "original_units_sha256": _canonical_digest(
                [_sha256_bytes(unit) for unit in rendered_units]
            ),
            "prefix_gap_sha256": _sha256_bytes(self.prefix_gap),
            "suffix_gap_sha256": _sha256_bytes(self.suffix_gap),
            "gap_table_sha256": _canonical_digest(gap_table),
            "topology_sha256": _canonical_digest(
                {
                    "atoms": atom_table,
                    "boundaries": list(self.original_boundaries),
                    "gaps": gap_table,
                    "lexical_spans": lexical_table,
                    "raw_utf8_sha256": self.raw_utf8_sha256,
                    "refinement_cut_codepoints": list(
                        self.refinement_cut_codepoints
                    ),
                }
            ),
            "byte_reconstruction_sha256": _sha256_bytes(
                self.reconstruct(self.original_boundaries)
            ),
        }


def _align_original_units(
    decoded: str,
    sentence_splitter: Callable[[str], list[str]],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int], ...]]:
    units_value = sentence_splitter(decoded)
    if not isinstance(units_value, list) or any(
        not isinstance(unit, str) for unit in units_value
    ):
        raise TypeError("sentence splitter must return a list of strings")
    units = tuple(units_value)
    if len(units) < 2:
        _fail("P0_SEGMENTATION_INVALID", "unit_count_lt_2")

    cursor = 0
    spans = []
    for unit_index, unit in enumerate(units):
        if not unit or unit != unit.strip() or _LEXICAL_RE.search(unit) is None:
            _fail("P0_ALIGNMENT_FAILED", f"invalid_unit_{unit_index}")
        match_start = decoded.find(unit, cursor)
        if match_start < 0:
            _fail("P0_ALIGNMENT_FAILED", f"unit_not_found_{unit_index}")
        skipped = decoded[cursor:match_start]
        if skipped and not skipped.isspace():
            _fail(
                "P0_ALIGNMENT_FAILED",
                f"non_whitespace_before_unit_{unit_index}",
            )
        match_end = match_start + len(unit)
        spans.append((match_start, match_end))
        cursor = match_end
    trailing = decoded[cursor:]
    if trailing and not trailing.isspace():
        _fail("P0_ALIGNMENT_FAILED", "non_whitespace_trailing_text")
    return units, tuple(spans)


def canonicalize_refined_bytes(
    raw_utf8_bytes: bytes,
    sentence_splitter: Callable[[str], list[str]],
) -> RefinedAtomDocument:
    if not isinstance(raw_utf8_bytes, bytes):
        raise TypeError("raw input must be bytes")
    try:
        decoded = raw_utf8_bytes.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        _fail("INVALID_UTF8", str(error))

    offsets = _codepoint_to_byte_offsets(decoded)
    lexical_matches = tuple(_LEXICAL_RE.finditer(decoded))
    if not lexical_matches:
        _fail("INSUFFICIENT_ATOMS", "lexical_span_count_lt_1")
    lexical_spans = tuple(
        LexicalSpan(
            codepoint_start=match.start(),
            codepoint_end=match.end(),
            byte_start=offsets[match.start()],
            byte_end=offsets[match.end()],
            payload_sha256=_sha256_bytes(
                raw_utf8_bytes[offsets[match.start()] : offsets[match.end()]]
            ),
        )
        for match in lexical_matches
    )
    units, unit_codepoint_spans = _align_original_units(
        decoded,
        sentence_splitter,
    )

    unit_endpoints = {
        endpoint
        for span in unit_codepoint_spans
        for endpoint in span
    }
    atoms = []
    refinement_cuts = []
    for lexical_index, lexical in enumerate(lexical_spans):
        cuts = sorted({
            lexical.codepoint_start,
            lexical.codepoint_end,
            *(
                endpoint
                for endpoint in unit_endpoints
                if lexical.codepoint_start < endpoint < lexical.codepoint_end
            ),
        })
        refinement_cuts.extend(cuts[1:-1])
        for start, end in zip(cuts, cuts[1:]):
            payload = raw_utf8_bytes[offsets[start] : offsets[end]]
            if (
                start >= end
                or not payload
                or _LEXICAL_RE.fullmatch(decoded[start:end]) is None
            ):
                _fail("REFINEMENT_INVALID", f"lexical_span_{lexical_index}")
            atoms.append(
                RefinedAtom(
                    codepoint_start=start,
                    codepoint_end=end,
                    byte_start=offsets[start],
                    byte_end=offsets[end],
                    lexical_span_index=lexical_index,
                    payload_sha256=_sha256_bytes(payload),
                )
            )
    if len(atoms) < 2:
        _fail("INSUFFICIENT_ATOMS", "refined_atom_count_lt_2")

    atom_start_to_index = {
        atom.codepoint_start: index for index, atom in enumerate(atoms)
    }
    atom_end_to_index = {
        atom.codepoint_end: index + 1 for index, atom in enumerate(atoms)
    }
    unit_atom_spans = []
    for unit_index, ((start_cp, end_cp), unit) in enumerate(
        zip(unit_codepoint_spans, units)
    ):
        if start_cp not in atom_start_to_index or end_cp not in atom_end_to_index:
            _fail("P0_ALIGNMENT_FAILED", f"refinement_missed_unit_{unit_index}")
        start = atom_start_to_index[start_cp]
        end = atom_end_to_index[end_cp]
        rendered = raw_utf8_bytes[
            atoms[start].byte_start : atoms[end - 1].byte_end
        ].decode("utf-8", "strict")
        if rendered != unit:
            _fail("P0_ALIGNMENT_FAILED", f"unit_bytes_drift_{unit_index}")
        unit_atom_spans.append((start, end))
    if (
        unit_atom_spans[0][0] != 0
        or unit_atom_spans[-1][1] != len(atoms)
        or any(
            left[1] != right[0]
            for left, right in zip(unit_atom_spans, unit_atom_spans[1:])
        )
    ):
        _fail("P0_ALIGNMENT_FAILED", "non_contiguous_unit_atom_spans")

    original_boundaries = tuple(
        span[1] for span in unit_atom_spans[:-1]
    )
    document = RefinedAtomDocument(
        raw_utf8_bytes=raw_utf8_bytes,
        raw_utf8_sha256=_sha256_bytes(raw_utf8_bytes),
        decoded_text=decoded,
        lexical_spans=lexical_spans,
        atoms=tuple(atoms),
        refinement_cut_codepoints=tuple(sorted(set(refinement_cuts))),
        original_units=units,
        original_unit_codepoint_spans=unit_codepoint_spans,
        original_unit_atom_spans=tuple(unit_atom_spans),
        original_boundaries=original_boundaries,
    )

    original_boundary_set = set(original_boundaries)
    for boundary in range(1, document.atom_count):
        gap = document.gap_after(boundary)
        if not gap:
            if boundary not in original_boundary_set:
                _fail(
                    "UNAUTHORIZED_ZERO_BYTE_GAP",
                    f"boundary_{boundary}",
                )
        elif not gap.decode("utf-8", "strict").isspace():
            _fail("NON_WHITESPACE_ATOM_GAP", f"boundary_{boundary}")
    if (
        document.render_partition(document.original_boundaries) != units
        or document.reconstruct(document.original_boundaries) != raw_utf8_bytes
    ):
        _fail("P0_RENDER_MISMATCH", "round_trip_failed")
    return document
