"""Fixed-byte detector-unit renderer for SemStamp certification experiments."""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, Tuple


_ATOM_RE = re.compile(r"\S+")


class FixedByteRendererError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}")


def _fail(code: str, detail: str):
    raise FixedByteRendererError(code, detail)


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
class Atom:
    codepoint_start: int
    codepoint_end: int
    byte_start: int
    byte_end: int
    payload_sha256: str


@dataclass(frozen=True)
class FixedByteDocument:
    raw_utf8_bytes: bytes
    raw_utf8_sha256: str
    decoded_text: str
    atoms: Tuple[Atom, ...]
    original_units: Tuple[str, ...]
    original_unit_atom_spans: Tuple[Tuple[int, int], ...]
    original_boundaries: Tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def unit_count(self) -> int:
        return len(self.original_units)

    @property
    def prefix_gap(self) -> bytes:
        return self.raw_utf8_bytes[: self.atoms[0].byte_start]

    @property
    def suffix_gap(self) -> bytes:
        return self.raw_utf8_bytes[self.atoms[-1].byte_end :]

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
        atom_table = [
            {
                "codepoint_span": [atom.codepoint_start, atom.codepoint_end],
                "byte_span": [atom.byte_start, atom.byte_end],
                "payload_sha256": atom.payload_sha256,
            }
            for atom in self.atoms
        ]
        gap_table = [
            {
                "boundary": boundary,
                "bytes": len(self.gap_after(boundary)),
                "sha256": _sha256_bytes(self.gap_after(boundary)),
            }
            for boundary in range(1, self.atom_count)
        ]
        rendered_units = self.render_partition_bytes(self.original_boundaries)
        return {
            "raw_utf8_sha256": self.raw_utf8_sha256,
            "raw_utf8_bytes": len(self.raw_utf8_bytes),
            "atom_count": self.atom_count,
            "unit_count": self.unit_count,
            "original_boundaries": list(self.original_boundaries),
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
                    "raw_utf8_sha256": self.raw_utf8_sha256,
                }
            ),
            "byte_reconstruction_sha256": _sha256_bytes(
                self.reconstruct(self.original_boundaries)
            ),
        }


def canonicalize_fixed_bytes(
    raw_utf8_bytes: bytes,
    sentence_splitter: Callable[[str], list[str]],
) -> FixedByteDocument:
    if not isinstance(raw_utf8_bytes, bytes):
        raise TypeError("raw input must be bytes")
    decoded = raw_utf8_bytes.decode("utf-8", "strict")
    offsets = _codepoint_to_byte_offsets(decoded)
    matches = tuple(_ATOM_RE.finditer(decoded))
    if len(matches) < 2:
        _fail("INSUFFICIENT_ATOMS", "atom_count_lt_2")
    atoms = tuple(
        Atom(
            codepoint_start=match.start(),
            codepoint_end=match.end(),
            byte_start=offsets[match.start()],
            byte_end=offsets[match.end()],
            payload_sha256=_sha256_bytes(
                raw_utf8_bytes[offsets[match.start()] : offsets[match.end()]]
            ),
        )
        for match in matches
    )
    for boundary in range(1, len(atoms)):
        gap = raw_utf8_bytes[atoms[boundary - 1].byte_end : atoms[boundary].byte_start]
        if not gap or not gap.decode("utf-8", "strict").isspace():
            _fail("NON_WHITESPACE_ATOM_GAP", f"boundary_{boundary}")

    units_value = sentence_splitter(decoded)
    if not isinstance(units_value, list) or any(
        not isinstance(unit, str) for unit in units_value
    ):
        raise TypeError("sentence splitter must return a list of strings")
    units = tuple(units_value)
    if len(units) < 2:
        _fail("P0_SEGMENTATION_INVALID", "unit_count_lt_2")

    atom_start_to_index = {
        atom.codepoint_start: index for index, atom in enumerate(atoms)
    }
    atom_end_to_index = {
        atom.codepoint_end: index + 1 for index, atom in enumerate(atoms)
    }
    cursor = 0
    unit_spans = []
    for unit_index, unit in enumerate(units):
        if not unit or unit != unit.strip() or _ATOM_RE.search(unit) is None:
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
        start_aligned = match_start in atom_start_to_index
        end_aligned = match_end in atom_end_to_index
        if not start_aligned and not end_aligned:
            _fail(
                "P0_ALIGNMENT_FAILED",
                f"both_endpoints_inside_atoms_at_unit_{unit_index}",
            )
        if not start_aligned:
            _fail(
                "P0_ALIGNMENT_FAILED",
                f"start_inside_atom_at_unit_{unit_index}",
            )
        if not end_aligned:
            _fail(
                "P0_ALIGNMENT_FAILED",
                f"end_inside_atom_at_unit_{unit_index}",
            )
        start = atom_start_to_index[match_start]
        end = atom_end_to_index[match_end]
        if unit != raw_utf8_bytes[
            atoms[start].byte_start : atoms[end - 1].byte_end
        ].decode("utf-8", "strict"):
            _fail("P0_ALIGNMENT_FAILED", f"unit_bytes_drift_{unit_index}")
        unit_spans.append((start, end))
        cursor = match_end
    trailing = decoded[cursor:]
    if trailing and not trailing.isspace():
        _fail("P0_ALIGNMENT_FAILED", "non_whitespace_trailing_text")
    if (
        unit_spans[0][0] != 0
        or unit_spans[-1][1] != len(atoms)
        or any(
            left[1] != right[0]
            for left, right in zip(unit_spans, unit_spans[1:])
        )
    ):
        _fail("P0_ALIGNMENT_FAILED", "non_contiguous_unit_atom_spans")

    document = FixedByteDocument(
        raw_utf8_bytes=raw_utf8_bytes,
        raw_utf8_sha256=_sha256_bytes(raw_utf8_bytes),
        decoded_text=decoded,
        atoms=atoms,
        original_units=units,
        original_unit_atom_spans=tuple(unit_spans),
        original_boundaries=tuple(span[1] for span in unit_spans[:-1]),
    )
    if (
        document.render_partition(document.original_boundaries) != units
        or document.reconstruct(document.original_boundaries) != raw_utf8_bytes
    ):
        _fail("P0_RENDER_MISMATCH", "round_trip_failed")
    return document
