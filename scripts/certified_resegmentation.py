"""Exact certificates for fixed-count, word-preserving resegmentation.

This module is deliberately detector-agnostic. Callers provide deterministic
scores for admissible token spans or adjacent-span pairs; no parser, model,
API, or watermark implementation is imported here.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Callable, Dict, FrozenSet, Iterator, Mapping, Optional, Tuple


Span = Tuple[int, int]
PairScoreKey = Tuple[int, int, int]
Boundaries = Tuple[int, ...]


@dataclass(frozen=True)
class CertificateProblem:
    """A fixed-unit-count additive certificate problem.

    Token spans use half-open indices. ``original_boundaries`` and
    ``allowed_boundaries`` contain internal gaps in ``1..token_count - 1``.
    """

    token_count: int
    original_boundaries: Boundaries
    min_unit_words: int
    max_unit_words: int
    max_boundary_moves: int
    span_scores: Mapping[Span, float]
    threshold: float
    bias: float = 0.0
    allowed_boundaries: Optional[FrozenSet[int]] = None


@dataclass(frozen=True)
class CertificateResult:
    algorithm: str
    token_count: int
    unit_count: int
    requested_radius: int
    original_boundaries: Boundaries
    worst_boundaries: Boundaries
    original_score: float
    certified_score: float
    threshold: float
    original_positive: bool
    certified_positive: bool
    certificate_margin: float
    worst_case_drop: float
    threat_set_size: int
    exact_move_histogram: Tuple[Tuple[int, int], ...]
    max_realized_moves: int
    singleton_threat_set: bool
    budget_reaches_requested_radius: bool
    original_is_worst_case: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["original_boundaries"] = list(self.original_boundaries)
        payload["worst_boundaries"] = list(self.worst_boundaries)
        payload["exact_move_histogram"] = {
            str(radius): count for radius, count in self.exact_move_histogram
        }
        return payload


@dataclass(frozen=True)
class PairwiseCertificateProblem:
    """A fixed-unit-count adjacent-span certificate problem.

    ``pair_scores[(h, i, j)]`` scores adjacent spans ``[h, i)`` and
    ``[i, j)``. The first unit is unscored. Document scores are the pair-score
    sum divided by the fixed positive ``normalization``, plus ``bias``.
    """

    token_count: int
    original_boundaries: Boundaries
    min_unit_words: int
    max_unit_words: int
    max_boundary_moves: int
    pair_scores: Mapping[PairScoreKey, float]
    threshold: float
    normalization: float
    bias: float = 0.0
    allowed_boundaries: Optional[FrozenSet[int]] = None


@dataclass(frozen=True)
class PairwiseCertificateResult:
    algorithm: str
    token_count: int
    unit_count: int
    edge_count: int
    requested_radius: int
    original_boundaries: Boundaries
    worst_boundaries: Boundaries
    original_score: float
    certified_score: float
    threshold: float
    normalization: float
    original_positive: bool
    certified_positive: bool
    certificate_margin: float
    worst_case_drop: float
    threat_set_size: int
    exact_move_histogram: Tuple[Tuple[int, int], ...]
    max_realized_moves: int
    singleton_threat_set: bool
    budget_reaches_requested_radius: bool
    original_is_worst_case: bool
    reachable_state_count: Optional[int]
    transitions_evaluated: Optional[int]
    peak_layer_states: Optional[int]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["original_boundaries"] = list(self.original_boundaries)
        payload["worst_boundaries"] = list(self.worst_boundaries)
        payload["exact_move_histogram"] = {
            str(radius): count for radius, count in self.exact_move_histogram
        }
        return payload


@dataclass
class _Cell:
    best_sum: float
    count: int
    predecessor: Optional[Tuple[int, int]]


@dataclass
class _PairCell:
    best_sum: float
    count: int
    predecessor: Optional[Tuple[int, int, int]]


@dataclass(frozen=True)
class _Validated:
    allowed: FrozenSet[int]
    unit_count: int


def build_span_scores(
    token_count: int,
    min_unit_words: int,
    max_unit_words: int,
    scorer: Callable[[int, int], float],
    allowed_boundaries: Optional[FrozenSet[int]] = None,
) -> Dict[Span, float]:
    """Evaluate ``scorer`` on every local edge the certificate may need."""
    allowed = (
        frozenset(range(1, token_count))
        if allowed_boundaries is None
        else frozenset(allowed_boundaries)
    )
    starts = (0, *sorted(allowed))
    ends = (*sorted(allowed), token_count)
    scores: Dict[Span, float] = {}
    for start in starts:
        for end in ends:
            length = end - start
            if min_unit_words <= length <= max_unit_words:
                value = float(scorer(start, end))
                if not math.isfinite(value):
                    raise ValueError(f"non-finite span score for {(start, end)}")
                scores[(start, end)] = value
    return scores


def build_pair_scores(
    token_count: int,
    min_unit_words: int,
    max_unit_words: int,
    scorer: Callable[[int, int, int], float],
    allowed_boundaries: Optional[FrozenSet[int]] = None,
) -> Dict[PairScoreKey, float]:
    """Evaluate ``scorer`` on every locally admissible adjacent-span pair."""
    allowed = (
        frozenset(range(1, token_count))
        if allowed_boundaries is None
        else frozenset(allowed_boundaries)
    )
    starts = (0, *sorted(allowed))
    middles = sorted(allowed)
    ends = (*sorted(allowed), token_count)
    scores: Dict[PairScoreKey, float] = {}
    for middle in middles:
        for start in starts:
            previous_length = middle - start
            if not min_unit_words <= previous_length <= max_unit_words:
                continue
            for end in ends:
                current_length = end - middle
                if not min_unit_words <= current_length <= max_unit_words:
                    continue
                value = float(scorer(start, middle, end))
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite pair score for {(start, middle, end)}"
                    )
                scores[(start, middle, end)] = value
    return scores


def boundary_moves(boundaries: Boundaries, original: Boundaries) -> int:
    """Count one-for-one boundary replacements for equal-size partitions."""
    if len(boundaries) != len(original):
        raise ValueError("boundary relocation requires equal boundary counts")
    return len(set(boundaries) - set(original))


def _spans(boundaries: Boundaries, token_count: int) -> Iterator[Span]:
    start = 0
    for end in (*boundaries, token_count):
        yield start, end
        start = end


def _validate_structure(problem) -> _Validated:
    if problem.token_count < 1:
        raise ValueError("token_count must be positive")
    if problem.min_unit_words < 1:
        raise ValueError("min_unit_words must be positive")
    if problem.max_unit_words < problem.min_unit_words:
        raise ValueError("max_unit_words must be >= min_unit_words")
    if tuple(sorted(set(problem.original_boundaries))) != problem.original_boundaries:
        raise ValueError("original_boundaries must be strictly increasing")
    if any(
        boundary <= 0 or boundary >= problem.token_count
        for boundary in problem.original_boundaries
    ):
        raise ValueError("original boundaries must be internal token gaps")

    allowed = (
        frozenset(range(1, problem.token_count))
        if problem.allowed_boundaries is None
        else frozenset(problem.allowed_boundaries)
    )
    if any(boundary <= 0 or boundary >= problem.token_count for boundary in allowed):
        raise ValueError("allowed boundaries must be internal token gaps")
    if not set(problem.original_boundaries).issubset(allowed):
        raise ValueError("allowed_boundaries must include every original boundary")

    unit_count = len(problem.original_boundaries) + 1
    if not 0 <= problem.max_boundary_moves <= unit_count - 1:
        raise ValueError("max_boundary_moves must be in [0, unit_count - 1]")
    if not math.isfinite(problem.threshold) or not math.isfinite(problem.bias):
        raise ValueError("threshold and bias must be finite")

    for start, end in _spans(problem.original_boundaries, problem.token_count):
        length = end - start
        if not problem.min_unit_words <= length <= problem.max_unit_words:
            raise ValueError("the original partition violates unit-length bounds")

    return _Validated(allowed=allowed, unit_count=unit_count)


def _validate(problem: CertificateProblem) -> _Validated:
    validated = _validate_structure(problem)
    allowed = validated.allowed
    required_starts = (0, *sorted(allowed))
    required_ends = (*sorted(allowed), problem.token_count)
    for start in required_starts:
        for end in required_ends:
            length = end - start
            if not problem.min_unit_words <= length <= problem.max_unit_words:
                continue
            if (start, end) not in problem.span_scores:
                raise ValueError(f"missing score for admissible span {(start, end)}")
            if not math.isfinite(float(problem.span_scores[(start, end)])):
                raise ValueError(f"non-finite score for span {(start, end)}")

    return validated


def _validate_pairwise(
    problem: PairwiseCertificateProblem,
) -> _Validated:
    validated = _validate_structure(problem)
    if validated.unit_count < 2:
        raise ValueError("pairwise certificates require at least two units")
    if (
        not math.isfinite(problem.normalization)
        or problem.normalization <= 0
    ):
        raise ValueError("normalization must be finite and positive")

    allowed = validated.allowed
    required_starts = (0, *sorted(allowed))
    required_ends = (*sorted(allowed), problem.token_count)
    for middle in sorted(allowed):
        for start in required_starts:
            previous_length = middle - start
            if not (
                problem.min_unit_words
                <= previous_length
                <= problem.max_unit_words
            ):
                continue
            for end in required_ends:
                current_length = end - middle
                if not (
                    problem.min_unit_words
                    <= current_length
                    <= problem.max_unit_words
                ):
                    continue
                key = (start, middle, end)
                if key not in problem.pair_scores:
                    raise ValueError(f"missing score for admissible pair {key}")
                if not math.isfinite(float(problem.pair_scores[key])):
                    raise ValueError(f"non-finite score for pair {key}")

    return validated


def _partition_sum(problem: CertificateProblem, boundaries: Boundaries) -> float:
    return sum(
        float(problem.span_scores[span])
        for span in _spans(boundaries, problem.token_count)
    )


def _partition_pair_sum(
    problem: PairwiseCertificateProblem,
    boundaries: Boundaries,
) -> float:
    endpoints = (0, *boundaries, problem.token_count)
    return sum(
        float(
            problem.pair_scores[
                (endpoints[index - 1], endpoints[index], endpoints[index + 1])
            ]
        )
        for index in range(1, len(endpoints) - 1)
    )


def _result(
    problem: CertificateProblem,
    validated: _Validated,
    algorithm: str,
    worst_boundaries: Boundaries,
    worst_sum: float,
    histogram: Mapping[int, int],
) -> CertificateResult:
    unit_count = validated.unit_count
    original_score = problem.bias + (
        _partition_sum(problem, problem.original_boundaries) / unit_count
    )
    certified_score = problem.bias + worst_sum / unit_count
    threat_set_size = sum(histogram.values())
    max_realized = max(radius for radius, count in histogram.items() if count)
    return CertificateResult(
        algorithm=algorithm,
        token_count=problem.token_count,
        unit_count=unit_count,
        requested_radius=problem.max_boundary_moves,
        original_boundaries=problem.original_boundaries,
        worst_boundaries=worst_boundaries,
        original_score=original_score,
        certified_score=certified_score,
        threshold=problem.threshold,
        original_positive=original_score > problem.threshold,
        certified_positive=certified_score > problem.threshold,
        certificate_margin=certified_score - problem.threshold,
        worst_case_drop=original_score - certified_score,
        threat_set_size=threat_set_size,
        exact_move_histogram=tuple(sorted(histogram.items())),
        max_realized_moves=max_realized,
        singleton_threat_set=threat_set_size == 1,
        budget_reaches_requested_radius=histogram.get(
            problem.max_boundary_moves, 0
        ) > 0,
        original_is_worst_case=math.isclose(
            original_score, certified_score, rel_tol=1e-12, abs_tol=1e-12
        ),
    )


def _pairwise_result(
    problem: PairwiseCertificateProblem,
    validated: _Validated,
    algorithm: str,
    worst_boundaries: Boundaries,
    worst_sum: float,
    histogram: Mapping[int, int],
    reachable_state_count: Optional[int] = None,
    transitions_evaluated: Optional[int] = None,
    peak_layer_states: Optional[int] = None,
) -> PairwiseCertificateResult:
    original_score = problem.bias + (
        _partition_pair_sum(problem, problem.original_boundaries)
        / problem.normalization
    )
    certified_score = problem.bias + worst_sum / problem.normalization
    threat_set_size = sum(histogram.values())
    max_realized = max(radius for radius, count in histogram.items() if count)
    return PairwiseCertificateResult(
        algorithm=algorithm,
        token_count=problem.token_count,
        unit_count=validated.unit_count,
        edge_count=validated.unit_count - 1,
        requested_radius=problem.max_boundary_moves,
        original_boundaries=problem.original_boundaries,
        worst_boundaries=worst_boundaries,
        original_score=original_score,
        certified_score=certified_score,
        threshold=problem.threshold,
        normalization=problem.normalization,
        original_positive=original_score > problem.threshold,
        certified_positive=certified_score > problem.threshold,
        certificate_margin=certified_score - problem.threshold,
        worst_case_drop=original_score - certified_score,
        threat_set_size=threat_set_size,
        exact_move_histogram=tuple(sorted(histogram.items())),
        max_realized_moves=max_realized,
        singleton_threat_set=threat_set_size == 1,
        budget_reaches_requested_radius=histogram.get(
            problem.max_boundary_moves, 0
        ) > 0,
        original_is_worst_case=math.isclose(
            original_score, certified_score, rel_tol=1e-12, abs_tol=1e-12
        ),
        reachable_state_count=reachable_state_count,
        transitions_evaluated=transitions_evaluated,
        peak_layer_states=peak_layer_states,
    )


def exact_certificate(problem: CertificateProblem) -> CertificateResult:
    """Compute the exact worst additive score by dynamic programming."""
    validated = _validate(problem)
    n = problem.token_count
    m = validated.unit_count
    k = problem.max_boundary_moves
    original = set(problem.original_boundaries)

    layers = [{(0, 0): _Cell(best_sum=0.0, count=1, predecessor=None)}]
    for units_used in range(1, m + 1):
        previous = layers[-1]
        current: Dict[Tuple[int, int], _Cell] = {}
        final_unit = units_used == m
        for (start, moves), previous_cell in sorted(previous.items()):
            minimum_end = start + problem.min_unit_words
            maximum_end = min(n, start + problem.max_unit_words)
            for end in range(minimum_end, maximum_end + 1):
                if final_unit:
                    if end != n:
                        continue
                elif end == n or end not in validated.allowed:
                    continue

                remaining_units = m - units_used
                remaining_words = n - end
                if not (
                    remaining_units * problem.min_unit_words
                    <= remaining_words
                    <= remaining_units * problem.max_unit_words
                ):
                    continue

                added_move = int(end < n and end not in original)
                new_moves = moves + added_move
                if new_moves > k:
                    continue
                candidate_sum = (
                    previous_cell.best_sum
                    + float(problem.span_scores[(start, end)])
                )
                key = (end, new_moves)
                cell = current.get(key)
                if cell is None:
                    current[key] = _Cell(
                        best_sum=candidate_sum,
                        count=previous_cell.count,
                        predecessor=(start, moves),
                    )
                    continue

                cell.count += previous_cell.count
                if candidate_sum < cell.best_sum:
                    cell.best_sum = candidate_sum
                    cell.predecessor = (start, moves)
        layers.append(current)

    finals = {
        moves: cell
        for (end, moves), cell in layers[m].items()
        if end == n and moves <= k
    }
    if not finals:
        raise ValueError("the declared threat set is empty")
    worst_moves, worst_cell = min(
        finals.items(), key=lambda item: (item[1].best_sum, item[0])
    )

    reversed_boundaries = []
    key = (n, worst_moves)
    for units_used in range(m, 0, -1):
        predecessor = layers[units_used][key].predecessor
        if predecessor is None:
            raise AssertionError("broken DP predecessor chain")
        previous_end, previous_moves = predecessor
        if previous_end > 0:
            reversed_boundaries.append(previous_end)
        key = (previous_end, previous_moves)
    worst_boundaries = tuple(reversed(reversed_boundaries))

    return _result(
        problem=problem,
        validated=validated,
        algorithm="exact_dynamic_program",
        worst_boundaries=worst_boundaries,
        worst_sum=worst_cell.best_sum,
        histogram={moves: cell.count for moves, cell in finals.items()},
    )


def exact_pairwise_certificate(
    problem: PairwiseCertificateProblem,
) -> PairwiseCertificateResult:
    """Compute the exact worst adjacent-pair score by edge-state DP."""
    validated = _validate_pairwise(problem)
    n = problem.token_count
    m = validated.unit_count
    k = problem.max_boundary_moves
    original = set(problem.original_boundaries)

    layers: list = [{} for _ in range(m + 1)]
    first_layer: Dict[Tuple[int, int, int], _PairCell] = {}
    for end in range(
        problem.min_unit_words,
        min(n - 1, problem.max_unit_words) + 1,
    ):
        if end not in validated.allowed:
            continue
        remaining_units = m - 1
        remaining_words = n - end
        if not (
            remaining_units * problem.min_unit_words
            <= remaining_words
            <= remaining_units * problem.max_unit_words
        ):
            continue
        moves = int(end not in original)
        if moves <= k:
            first_layer[(0, end, moves)] = _PairCell(
                best_sum=0.0,
                count=1,
                predecessor=None,
            )
    layers[1] = first_layer

    reachable_state_count = len(first_layer)
    peak_layer_states = len(first_layer)
    transitions_evaluated = 0
    for units_used in range(1, m):
        previous = layers[units_used]
        current: Dict[Tuple[int, int, int], _PairCell] = {}
        final_unit = units_used + 1 == m
        for (previous_start, start, moves), previous_cell in sorted(
            previous.items()
        ):
            minimum_end = start + problem.min_unit_words
            maximum_end = min(n, start + problem.max_unit_words)
            for end in range(minimum_end, maximum_end + 1):
                if final_unit:
                    if end != n:
                        continue
                elif end == n or end not in validated.allowed:
                    continue

                remaining_units = m - units_used - 1
                remaining_words = n - end
                if not (
                    remaining_units * problem.min_unit_words
                    <= remaining_words
                    <= remaining_units * problem.max_unit_words
                ):
                    continue

                added_move = int(end < n and end not in original)
                new_moves = moves + added_move
                if new_moves > k:
                    continue

                transitions_evaluated += 1
                candidate_sum = (
                    previous_cell.best_sum
                    + float(
                        problem.pair_scores[
                            (previous_start, start, end)
                        ]
                    )
                )
                key = (start, end, new_moves)
                predecessor = (previous_start, start, moves)
                cell = current.get(key)
                if cell is None:
                    current[key] = _PairCell(
                        best_sum=candidate_sum,
                        count=previous_cell.count,
                        predecessor=predecessor,
                    )
                    continue

                cell.count += previous_cell.count
                if candidate_sum < cell.best_sum:
                    cell.best_sum = candidate_sum
                    cell.predecessor = predecessor
        layers[units_used + 1] = current
        reachable_state_count += len(current)
        peak_layer_states = max(peak_layer_states, len(current))

    final_items = [
        (key, cell)
        for key, cell in layers[m].items()
        if key[1] == n and key[2] <= k
    ]
    if not final_items:
        raise ValueError("the declared threat set is empty")

    histogram: Dict[int, int] = {}
    for (_, _, moves), cell in final_items:
        histogram[moves] = histogram.get(moves, 0) + cell.count
    worst_key, worst_cell = min(
        final_items,
        key=lambda item: (
            item[1].best_sum,
            item[0][2],
            item[0][0],
        ),
    )

    reversed_boundaries = []
    key = worst_key
    for units_used in range(m, 1, -1):
        reversed_boundaries.append(key[0])
        predecessor = layers[units_used][key].predecessor
        if predecessor is None:
            raise AssertionError("broken pairwise DP predecessor chain")
        key = predecessor
    if layers[1][key].predecessor is not None:
        raise AssertionError("invalid first-layer predecessor")
    worst_boundaries = tuple(reversed(reversed_boundaries))

    return _pairwise_result(
        problem=problem,
        validated=validated,
        algorithm="exact_pairwise_edge_state_dynamic_program",
        worst_boundaries=worst_boundaries,
        worst_sum=worst_cell.best_sum,
        histogram=histogram,
        reachable_state_count=reachable_state_count,
        transitions_evaluated=transitions_evaluated,
        peak_layer_states=peak_layer_states,
    )


def brute_force_certificate(
    problem: CertificateProblem,
    combination_limit: int = 1_000_000,
) -> CertificateResult:
    """Independent combinatorial oracle for small instances."""
    validated = _validate(problem)
    boundary_count = validated.unit_count - 1
    search_size = math.comb(len(validated.allowed), boundary_count)
    if search_size > combination_limit:
        raise ValueError(
            f"oracle search has {search_size} combinations, above "
            f"limit {combination_limit}"
        )

    histogram: Dict[int, int] = {}
    best_sum = math.inf
    best_boundaries: Optional[Boundaries] = None
    for candidate in combinations(sorted(validated.allowed), boundary_count):
        lengths = [
            end - start
            for start, end in _spans(candidate, problem.token_count)
        ]
        if any(
            length < problem.min_unit_words
            or length > problem.max_unit_words
            for length in lengths
        ):
            continue
        moves = boundary_moves(candidate, problem.original_boundaries)
        if moves > problem.max_boundary_moves:
            continue
        histogram[moves] = histogram.get(moves, 0) + 1
        candidate_sum = _partition_sum(problem, candidate)
        if (
            candidate_sum < best_sum
            or (
                candidate_sum == best_sum
                and (best_boundaries is None or candidate < best_boundaries)
            )
        ):
            best_sum = candidate_sum
            best_boundaries = candidate

    if best_boundaries is None:
        raise ValueError("the declared threat set is empty")
    return _result(
        problem=problem,
        validated=validated,
        algorithm="brute_force_oracle",
        worst_boundaries=best_boundaries,
        worst_sum=best_sum,
        histogram=histogram,
    )


def brute_force_pairwise_certificate(
    problem: PairwiseCertificateProblem,
    combination_limit: int = 1_000_000,
) -> PairwiseCertificateResult:
    """Independent combinatorial oracle for small pairwise instances."""
    validated = _validate_pairwise(problem)
    boundary_count = validated.unit_count - 1
    search_size = math.comb(len(validated.allowed), boundary_count)
    if search_size > combination_limit:
        raise ValueError(
            f"oracle search has {search_size} combinations, above "
            f"limit {combination_limit}"
        )

    histogram: Dict[int, int] = {}
    best_sum = math.inf
    best_boundaries: Optional[Boundaries] = None
    for candidate in combinations(sorted(validated.allowed), boundary_count):
        lengths = [
            end - start
            for start, end in _spans(candidate, problem.token_count)
        ]
        if any(
            length < problem.min_unit_words
            or length > problem.max_unit_words
            for length in lengths
        ):
            continue
        moves = boundary_moves(candidate, problem.original_boundaries)
        if moves > problem.max_boundary_moves:
            continue
        histogram[moves] = histogram.get(moves, 0) + 1
        candidate_sum = _partition_pair_sum(problem, candidate)
        if (
            candidate_sum < best_sum
            or (
                candidate_sum == best_sum
                and (best_boundaries is None or candidate < best_boundaries)
            )
        ):
            best_sum = candidate_sum
            best_boundaries = candidate

    if best_boundaries is None:
        raise ValueError("the declared threat set is empty")
    return _pairwise_result(
        problem=problem,
        validated=validated,
        algorithm="brute_force_pairwise_oracle",
        worst_boundaries=best_boundaries,
        worst_sum=best_sum,
        histogram=histogram,
    )
