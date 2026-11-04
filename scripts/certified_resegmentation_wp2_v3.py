"""Typed-list adapter and sparse certificates for WP2-v3.

The structured adapter preserves every original unit exactly at P0. A virtual
ASCII-space separator represents each list boundary only when that boundary is
removed by a candidate partition.
"""

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import math
import re
from typing import Callable, Dict, Iterable, Optional, Tuple

from certified_resegmentation_wp2_adapter import (
    Boundaries,
    RawValue,
    _raw_field_type,
    hash_raw_value,
)


AtomSpan = Tuple[int, int]
BoundaryCut = Tuple[int, int]
_ATOM_RE = re.compile(r"\S+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_units(units: Tuple[str, ...]) -> None:
    if not units:
        raise ValueError("the original partition must contain at least one unit")
    for unit in units:
        if not isinstance(unit, str):
            raise TypeError("every original unit must be a string")
        if _ATOM_RE.search(unit) is None:
            raise ValueError("every original unit must contain an atom")


def _validate_boundaries(
    boundaries: Boundaries,
    atom_count: int,
) -> None:
    if any(
        isinstance(boundary, bool) or not isinstance(boundary, int)
        for boundary in boundaries
    ):
        raise TypeError("partition boundaries must be integers")
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("partition boundaries must be strictly increasing")
    if any(
        boundary <= 0 or boundary >= atom_count
        for boundary in boundaries
    ):
        raise ValueError("partition boundaries must be internal atom gaps")


@dataclass(frozen=True)
class StructuredDocument:
    """A typed unit list represented by atoms and virtual separators."""

    raw_field_type: str
    raw_text_sha256: str
    original_units: Tuple[str, ...]
    structural_tape: str
    structural_tape_sha256: str
    atom_spans: Tuple[AtomSpan, ...]
    original_boundaries: Boundaries
    original_boundary_cuts: Tuple[BoundaryCut, ...]

    def __post_init__(self) -> None:
        if self.raw_field_type not in {"string", "list"}:
            raise ValueError("raw_field_type must be 'string' or 'list'")
        if _SHA256_RE.fullmatch(self.raw_text_sha256) is None:
            raise ValueError("raw_text_sha256 must be lowercase SHA256")
        if _SHA256_RE.fullmatch(self.structural_tape_sha256) is None:
            raise ValueError(
                "structural_tape_sha256 must be lowercase SHA256"
            )
        _validate_units(self.original_units)
        if self.structural_tape != " ".join(self.original_units):
            raise ValueError(
                "structural_tape must be the ASCII-space unit join"
            )
        if self.structural_tape_sha256 != _sha256_text(
            self.structural_tape
        ):
            raise ValueError("structural_tape_sha256 does not match the tape")

        expected_spans = tuple(
            (match.start(), match.end())
            for match in _ATOM_RE.finditer(self.structural_tape)
        )
        if self.atom_spans != expected_spans:
            raise ValueError("atom_spans must be maximal non-whitespace spans")

        expected_boundaries = []
        cumulative_atoms = 0
        expected_cuts = []
        offset = 0
        for index, unit in enumerate(self.original_units):
            cumulative_atoms += len(_ATOM_RE.findall(unit))
            offset += len(unit)
            if index < len(self.original_units) - 1:
                expected_boundaries.append(cumulative_atoms)
                expected_cuts.append((offset, offset + 1))
                offset += 1
        if self.original_boundaries != tuple(expected_boundaries):
            raise ValueError("original_boundaries do not match original units")
        if self.original_boundary_cuts != tuple(expected_cuts):
            raise ValueError(
                "original_boundary_cuts do not match virtual separators"
            )
        if self.render_partition(
            self.original_boundaries
        ) != self.original_units:
            raise ValueError("original units do not round-trip exactly")
        if self.partition_atoms(
            self.original_boundaries
        ) != self.atom_payloads:
            raise ValueError("original atom payloads do not round-trip")

    @property
    def atom_count(self) -> int:
        return len(self.atom_spans)

    @property
    def unit_count(self) -> int:
        return len(self.original_units)

    @property
    def atom_payloads(self) -> Tuple[str, ...]:
        return tuple(
            self.structural_tape[start:end]
            for start, end in self.atom_spans
        )

    @property
    def boundary_cut_map(self) -> dict[int, BoundaryCut]:
        return dict(zip(
            self.original_boundaries,
            self.original_boundary_cuts,
        ))

    def _left_cut(self, boundary: int) -> int:
        if boundary == self.atom_count:
            return len(self.structural_tape)
        original_cut = self.boundary_cut_map.get(boundary)
        if original_cut is not None:
            return original_cut[0]
        return self.atom_spans[boundary - 1][1]

    def _right_cut(self, boundary: int) -> int:
        if boundary == 0:
            return 0
        original_cut = self.boundary_cut_map.get(boundary)
        if original_cut is not None:
            return original_cut[1]
        return self.atom_spans[boundary - 1][1]

    def render_unit(self, start: int, end: int) -> str:
        """Render one atom span under the structural-separator contract."""
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise TypeError("atom endpoints must be integers")
        if not 0 <= start < end <= self.atom_count:
            raise ValueError("unit atom span must satisfy 0 <= start < end <= n")
        rendered = self.structural_tape[
            self._right_cut(start):self._left_cut(end)
        ]
        if tuple(_ATOM_RE.findall(rendered)) != self.atom_payloads[start:end]:
            raise AssertionError("rendered unit changed its atom payload")
        return rendered

    def render_partition(
        self,
        boundaries: Iterable[int],
    ) -> Tuple[str, ...]:
        boundaries = tuple(boundaries)
        _validate_boundaries(boundaries, self.atom_count)
        endpoints = (0, *boundaries, self.atom_count)
        rendered = tuple(
            self.render_unit(endpoints[index], endpoints[index + 1])
            for index in range(len(endpoints) - 1)
        )
        if tuple(
            atom for unit in rendered for atom in _ATOM_RE.findall(unit)
        ) != self.atom_payloads:
            raise AssertionError("partition rendering changed atom order")
        return rendered

    def partition_atoms(
        self,
        boundaries: Iterable[int],
    ) -> Tuple[str, ...]:
        return tuple(
            atom
            for unit in self.render_partition(boundaries)
            for atom in _ATOM_RE.findall(unit)
        )

    def preprocess_unit(
        self,
        system_id: str,
        start: int,
        end: int,
    ) -> str:
        if system_id not in {"semstamp", "simmark"}:
            raise ValueError("unknown system_id")
        rendered = self.render_unit(start, end)
        if system_id == "simmark" and start > 0:
            return " " + rendered
        return rendered

    def preprocess_partition(
        self,
        system_id: str,
        boundaries: Optional[Iterable[int]] = None,
    ) -> Tuple[str, ...]:
        selected = (
            self.original_boundaries
            if boundaries is None
            else tuple(boundaries)
        )
        _validate_boundaries(selected, self.atom_count)
        endpoints = (0, *selected, self.atom_count)
        return tuple(
            self.preprocess_unit(
                system_id,
                endpoints[index],
                endpoints[index + 1],
            )
            for index in range(len(endpoints) - 1)
        )


def canonicalize_structured(
    value: RawValue,
    sentence_splitter: Optional[Callable[[str], list[str]]] = None,
) -> StructuredDocument:
    """Freeze a typed P0 while preserving every original unit at k=0."""
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
    boundaries = []
    cuts = []
    cumulative_atoms = 0
    offset = 0
    for index, unit in enumerate(units):
        cumulative_atoms += len(_ATOM_RE.findall(unit))
        offset += len(unit)
        if index < len(units) - 1:
            boundaries.append(cumulative_atoms)
            cuts.append((offset, offset + 1))
            offset += 1

    return StructuredDocument(
        raw_field_type=field_type,
        raw_text_sha256=hash_raw_value(value),
        original_units=units,
        structural_tape=tape,
        structural_tape_sha256=_sha256_text(tape),
        atom_spans=atom_spans,
        original_boundaries=tuple(boundaries),
        original_boundary_cuts=tuple(cuts),
    )


@dataclass(frozen=True)
class SparsePairwiseProblem:
    """A fixed-count pairwise problem with content-dependent legal spans."""

    token_count: int
    original_boundaries: Boundaries
    max_boundary_moves: int
    span_allowed: Callable[[int, int], bool]
    pair_scorer: Callable[[int, int, int], float]
    threshold: float = 0.0
    normalization: Optional[float] = None
    bias: float = 0.0


@dataclass(frozen=True)
class SparsePairwiseResult:
    algorithm: str
    token_count: int
    unit_count: int
    requested_radius: int
    original_boundaries: Boundaries
    worst_boundaries: Boundaries
    original_score: float
    certified_score: float
    threshold: float
    normalization: float
    threat_set_size: int
    exact_move_histogram: Tuple[Tuple[int, int], ...]
    reachable_state_count: int
    transitions_evaluated: int
    peak_layer_states: int
    candidate_boundary_positions: int
    unique_span_checks: int
    unique_pair_scores: int

    def to_dict(self) -> dict:
        output = asdict(self)
        output["original_boundaries"] = list(self.original_boundaries)
        output["worst_boundaries"] = list(self.worst_boundaries)
        output["exact_move_histogram"] = {
            str(radius): count for radius, count in self.exact_move_histogram
        }
        return output


@dataclass
class _SparseCell:
    best_sum: float
    count: int
    predecessor: Optional[Tuple[int, int, int]]
    predecessor_rank: int
    path_rank: int = -1


def _validate_sparse_problem(problem: SparsePairwiseProblem):
    if (
        isinstance(problem.token_count, bool)
        or not isinstance(problem.token_count, int)
        or problem.token_count < 2
    ):
        raise ValueError("token_count must be an integer >= 2")
    boundaries = problem.original_boundaries
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("original_boundaries must be strictly increasing")
    if not boundaries:
        raise ValueError("pairwise certificates require at least two units")
    if any(
        boundary <= 0 or boundary >= problem.token_count
        for boundary in boundaries
    ):
        raise ValueError("original boundaries must be internal atom gaps")
    if (
        isinstance(problem.max_boundary_moves, bool)
        or not isinstance(problem.max_boundary_moves, int)
        or not 0 <= problem.max_boundary_moves <= len(boundaries)
    ):
        raise ValueError("max_boundary_moves is outside the valid range")
    if not callable(problem.span_allowed):
        raise TypeError("span_allowed must be callable")
    if not callable(problem.pair_scorer):
        raise TypeError("pair_scorer must be callable")
    if not math.isfinite(problem.threshold) or not math.isfinite(problem.bias):
        raise ValueError("threshold and bias must be finite")
    normalization = (
        float(len(boundaries))
        if problem.normalization is None
        else float(problem.normalization)
    )
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("normalization must be finite and positive")
    return normalization


def rank_candidate_boundaries(
    token_count: int,
    original_boundaries: Boundaries,
    max_boundary_moves: int,
) -> Tuple[Boundaries, ...]:
    """Return every boundary value feasible at each one-indexed rank."""
    boundary_count = len(original_boundaries)
    original_set = frozenset(original_boundaries)
    candidates = []
    for rank in range(1, boundary_count + 1):
        minimum = rank
        maximum = token_count - boundary_count + rank - 1
        rank_values = []
        for value in range(minimum, maximum + 1):
            before = bisect_left(original_boundaries, value)
            is_original = int(value in original_set)
            after = boundary_count - before - is_original
            minimum_new_before = max(0, rank - 1 - before)
            minimum_new_after = max(
                0, boundary_count - rank - after
            )
            minimum_new = (
                minimum_new_before
                + (1 - is_original)
                + minimum_new_after
            )
            if minimum_new <= max_boundary_moves:
                rank_values.append(value)
        candidates.append(tuple(rank_values))
    return tuple(candidates)


def exact_sparse_pairwise_certificate(
    problem: SparsePairwiseProblem,
) -> SparsePairwiseResult:
    """Compute an exact pairwise certificate with lazy score evaluation."""
    normalization = _validate_sparse_problem(problem)
    n = problem.token_count
    original = problem.original_boundaries
    original_set = frozenset(original)
    boundary_count = len(original)
    k = problem.max_boundary_moves
    candidates = rank_candidate_boundaries(n, original, k)

    span_cache: Dict[Tuple[int, int], bool] = {}
    pair_cache: Dict[Tuple[int, int, int], float] = {}

    def span_is_allowed(start: int, end: int) -> bool:
        key = (start, end)
        if key not in span_cache:
            value = problem.span_allowed(start, end)
            if not isinstance(value, bool):
                raise TypeError("span_allowed must return bool")
            span_cache[key] = value
        return span_cache[key]

    def pair_score(left: int, middle: int, right: int) -> float:
        key = (left, middle, right)
        if key not in pair_cache:
            value = float(problem.pair_scorer(left, middle, right))
            if not math.isfinite(value):
                raise ValueError(f"non-finite pair score for {key}")
            pair_cache[key] = value
        return pair_cache[key]

    endpoints = (0, *original, n)
    if any(
        not span_is_allowed(endpoints[index], endpoints[index + 1])
        for index in range(len(endpoints) - 1)
    ):
        raise ValueError("the original partition contains an illegal span")

    def can_complete(
        current: int,
        selected_count: int,
        new_count: int,
    ) -> bool:
        remaining = boundary_count - selected_count
        original_after = boundary_count - bisect_right(original, current)
        minimum_future_new = max(0, remaining - original_after)
        return new_count + minimum_future_new <= k

    layers: list[Dict[Tuple[int, int, int], _SparseCell]] = [
        {} for _ in range(boundary_count + 1)
    ]
    first_layer = {}
    for end in candidates[0]:
        moves = int(end not in original_set)
        if (
            moves <= k
            and can_complete(end, 1, moves)
            and span_is_allowed(0, end)
        ):
            first_layer[(0, end, moves)] = _SparseCell(
                best_sum=0.0,
                count=1,
                predecessor=None,
                predecessor_rank=-1,
            )
    for rank, key in enumerate(sorted(first_layer, key=lambda item: item[1])):
        first_layer[key].path_rank = rank
    layers[1] = first_layer

    reachable_state_count = len(first_layer)
    peak_layer_states = len(first_layer)
    transitions_evaluated = 0
    for selected_count in range(1, boundary_count):
        previous = layers[selected_count]
        current: Dict[Tuple[int, int, int], _SparseCell] = {}
        next_values = candidates[selected_count]
        for key, previous_cell in sorted(previous.items()):
            previous_start, start, moves = key
            for end in next_values[bisect_right(next_values, start):]:
                new_moves = moves + int(end not in original_set)
                if (
                    new_moves > k
                    or not can_complete(
                        end, selected_count + 1, new_moves
                    )
                    or not span_is_allowed(start, end)
                ):
                    continue
                transitions_evaluated += 1
                candidate_sum = (
                    previous_cell.best_sum
                    + pair_score(previous_start, start, end)
                )
                next_key = (start, end, new_moves)
                cell = current.get(next_key)
                if cell is None:
                    current[next_key] = _SparseCell(
                        best_sum=candidate_sum,
                        count=previous_cell.count,
                        predecessor=key,
                        predecessor_rank=previous_cell.path_rank,
                    )
                    continue
                cell.count += previous_cell.count
                if (
                    candidate_sum < cell.best_sum
                    or (
                        candidate_sum == cell.best_sum
                        and previous_cell.path_rank
                        < cell.predecessor_rank
                    )
                ):
                    cell.best_sum = candidate_sum
                    cell.predecessor = key
                    cell.predecessor_rank = previous_cell.path_rank

        ranked = sorted(
            current.items(),
            key=lambda item: (
                item[1].predecessor_rank,
                item[0][1],
            ),
        )
        for rank, (_, cell) in enumerate(ranked):
            cell.path_rank = rank
        layers[selected_count + 1] = current
        reachable_state_count += len(current)
        peak_layer_states = max(peak_layer_states, len(current))

    finals = []
    histogram: Dict[int, int] = {}
    for key, cell in layers[boundary_count].items():
        previous_start, start, moves = key
        if not span_is_allowed(start, n):
            continue
        transitions_evaluated += 1
        final_sum = (
            cell.best_sum + pair_score(previous_start, start, n)
        )
        finals.append((key, cell, final_sum))
        histogram[moves] = histogram.get(moves, 0) + cell.count
    if not finals:
        raise ValueError("the declared threat set is empty")

    worst_key, _, worst_sum = min(
        finals,
        key=lambda item: (
            item[2],
            item[0][2],
            item[1].path_rank,
        ),
    )
    reversed_boundaries = [worst_key[1]]
    key = worst_key
    for selected_count in range(boundary_count, 1, -1):
        predecessor = layers[selected_count][key].predecessor
        if predecessor is None:
            raise AssertionError("broken sparse DP predecessor chain")
        key = predecessor
        reversed_boundaries.append(key[1])
    worst_boundaries = tuple(reversed(reversed_boundaries))

    original_sum = sum(
        pair_score(
            endpoints[index - 1],
            endpoints[index],
            endpoints[index + 1],
        )
        for index in range(1, len(endpoints) - 1)
    )
    return SparsePairwiseResult(
        algorithm="exact_radius_aware_sparse_pairwise_dp",
        token_count=n,
        unit_count=boundary_count + 1,
        requested_radius=k,
        original_boundaries=original,
        worst_boundaries=worst_boundaries,
        original_score=problem.bias + original_sum / normalization,
        certified_score=problem.bias + worst_sum / normalization,
        threshold=problem.threshold,
        normalization=normalization,
        threat_set_size=sum(histogram.values()),
        exact_move_histogram=tuple(sorted(histogram.items())),
        reachable_state_count=reachable_state_count,
        transitions_evaluated=transitions_evaluated,
        peak_layer_states=peak_layer_states,
        candidate_boundary_positions=sum(map(len, candidates)),
        unique_span_checks=len(span_cache),
        unique_pair_scores=len(pair_cache),
    )


def brute_force_sparse_pairwise_certificate(
    problem: SparsePairwiseProblem,
    combination_limit: int = 1_000_000,
) -> SparsePairwiseResult:
    """Independent exhaustive oracle for small sparse problems."""
    normalization = _validate_sparse_problem(problem)
    n = problem.token_count
    original = problem.original_boundaries
    boundary_count = len(original)
    search_size = math.comb(n - 1, boundary_count)
    if search_size > combination_limit:
        raise ValueError(
            f"oracle search has {search_size} combinations, above "
            f"limit {combination_limit}"
        )

    span_cache = {}
    pair_cache = {}

    def span_is_allowed(start, end):
        key = (start, end)
        if key not in span_cache:
            value = problem.span_allowed(start, end)
            if not isinstance(value, bool):
                raise TypeError("span_allowed must return bool")
            span_cache[key] = value
        return span_cache[key]

    def pair_score(left, middle, right):
        key = (left, middle, right)
        if key not in pair_cache:
            value = float(problem.pair_scorer(left, middle, right))
            if not math.isfinite(value):
                raise ValueError(f"non-finite pair score for {key}")
            pair_cache[key] = value
        return pair_cache[key]

    histogram = {}
    best_sum = math.inf
    best_moves = None
    best_boundaries = None
    for candidate in combinations(range(1, n), boundary_count):
        moves = len(set(candidate) - set(original))
        if moves > problem.max_boundary_moves:
            continue
        endpoints = (0, *candidate, n)
        if any(
            not span_is_allowed(endpoints[index], endpoints[index + 1])
            for index in range(len(endpoints) - 1)
        ):
            continue
        histogram[moves] = histogram.get(moves, 0) + 1
        candidate_sum = sum(
            pair_score(
                endpoints[index - 1],
                endpoints[index],
                endpoints[index + 1],
            )
            for index in range(1, len(endpoints) - 1)
        )
        if (
            candidate_sum < best_sum
            or (
                candidate_sum == best_sum
                and (
                    best_moves is None
                    or moves < best_moves
                    or (
                        moves == best_moves
                        and (
                            best_boundaries is None
                            or candidate < best_boundaries
                        )
                    )
                )
            )
        ):
            best_sum = candidate_sum
            best_moves = moves
            best_boundaries = candidate
    if best_boundaries is None:
        raise ValueError("the declared threat set is empty")

    original_endpoints = (0, *original, n)
    if any(
        not span_is_allowed(
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(len(original_endpoints) - 1)
    ):
        raise ValueError("the original partition contains an illegal span")
    original_sum = sum(
        pair_score(
            original_endpoints[index - 1],
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(1, len(original_endpoints) - 1)
    )
    return SparsePairwiseResult(
        algorithm="brute_force_sparse_pairwise_oracle",
        token_count=n,
        unit_count=boundary_count + 1,
        requested_radius=problem.max_boundary_moves,
        original_boundaries=original,
        worst_boundaries=best_boundaries,
        original_score=problem.bias + original_sum / normalization,
        certified_score=problem.bias + best_sum / normalization,
        threshold=problem.threshold,
        normalization=normalization,
        threat_set_size=sum(histogram.values()),
        exact_move_histogram=tuple(sorted(histogram.items())),
        reachable_state_count=0,
        transitions_evaluated=0,
        peak_layer_states=0,
        candidate_boundary_positions=0,
        unique_span_checks=len(span_cache),
        unique_pair_scores=len(pair_cache),
    )
