"""Exact variable-N SemStamp certificate for synthetic CPU verification.

The implementation follows the WP1 theorem contract.  Threat topology is
fully constructed from span legality before the first pair-hit access.  The
public variable-N solver always covers the complete radius-induced edge-count
range; a separate fixed-count entrypoint exists only for the reduction test.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
from itertools import combinations
from numbers import Integral
from typing import Callable, Dict, Iterable, Optional, Tuple


Boundaries = Tuple[int, ...]
State = Tuple[int, int, int]
SpanAllowed = Callable[[int, int], bool]
PairHit = Callable[[int, int, int], int]

CERTIFIED_WATERMARKED = "CERTIFIED_WATERMARKED"
NOT_CERTIFIED = "NOT_CERTIFIED"


class InvalidOrIncompleteError(RuntimeError):
    """Fail-closed error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ExactScore:
    """The exact symbolic representation of D(H, m)."""

    hits: int
    edge_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.hits, bool)
            or not isinstance(self.hits, int)
            or self.hits < 0
        ):
            raise ValueError("hits must be a nonnegative integer")
        if (
            isinstance(self.edge_count, bool)
            or not isinstance(self.edge_count, int)
            or self.edge_count < 1
        ):
            raise ValueError("edge_count must be a positive integer")
        if self.hits > self.edge_count:
            raise ValueError("hits cannot exceed edge_count")

    @property
    def numerator(self) -> int:
        return 4 * self.hits - self.edge_count

    @property
    def value(self) -> float:
        return self.numerator / math.sqrt(3 * self.edge_count)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "edge_count": self.edge_count,
            "numerator": self.numerator,
            "value": self.value,
        }


def compare_exact_scores(left: ExactScore, right: ExactScore) -> int:
    """Return -1, 0, or 1 without floating-point comparison."""
    left_numerator = left.numerator
    right_numerator = right.numerator
    left_sign = (left_numerator > 0) - (left_numerator < 0)
    right_sign = (right_numerator > 0) - (right_numerator < 0)
    if left_sign != right_sign:
        return -1 if left_sign < right_sign else 1
    if left_sign == 0:
        return 0

    left_product = (
        left_numerator * left_numerator * right.edge_count
    )
    right_product = (
        right_numerator * right_numerator * left.edge_count
    )
    if left_product == right_product:
        return 0
    if left_sign > 0:
        return -1 if left_product < right_product else 1
    return -1 if left_product > right_product else 1


@dataclass(frozen=True)
class VariableNPairwiseProblem:
    token_count: int
    original_boundaries: Boundaries
    radius: int
    span_allowed: SpanAllowed
    pair_hit: PairHit
    threshold: Optional[ExactScore] = None


@dataclass(frozen=True)
class VariableNPairwiseResult:
    algorithm: str
    token_count: int
    original_boundaries: Boundaries
    requested_radius: int
    worst_boundaries: Boundaries
    exact_score: ExactScore
    certified_score: float
    original_exact_score: ExactScore
    original_score: float
    insertions: int
    deletions: int
    operation_cost: int
    threshold: Optional[ExactScore]
    decision: Optional[str]
    threat_set_size: int
    histogram_by_insertions_deletions: Tuple[Tuple[int, int, int], ...]
    histogram_by_edge_count: Tuple[Tuple[int, int], ...]
    histogram_by_operation_cost: Tuple[Tuple[int, int], ...]
    reachable_state_count: int
    transitions_evaluated: int
    peak_layer_states: int
    unique_span_checks: int
    unique_pair_hits: int
    score_access_count_during_construction: int

    def semantic_payload(self) -> dict:
        return {
            "token_count": self.token_count,
            "original_boundaries": list(self.original_boundaries),
            "requested_radius": self.requested_radius,
            "worst_boundaries": list(self.worst_boundaries),
            "exact_score": self.exact_score.to_dict(),
            "certified_score": self.certified_score,
            "original_exact_score": self.original_exact_score.to_dict(),
            "original_score": self.original_score,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "operation_cost": self.operation_cost,
            "threshold": (
                None if self.threshold is None else self.threshold.to_dict()
            ),
            "decision": self.decision,
            "threat_set_size": self.threat_set_size,
            "histogram_by_insertions_deletions": [
                {
                    "insertions": insertions,
                    "deletions": deletions,
                    "count": count,
                }
                for insertions, deletions, count
                in self.histogram_by_insertions_deletions
            ],
            "histogram_by_edge_count": {
                str(edge_count): count
                for edge_count, count in self.histogram_by_edge_count
            },
            "histogram_by_operation_cost": {
                str(cost): count
                for cost, count in self.histogram_by_operation_cost
            },
        }

    def to_dict(self) -> dict:
        output = self.semantic_payload()
        output.update({
            "algorithm": self.algorithm,
            "reachable_state_count": self.reachable_state_count,
            "transitions_evaluated": self.transitions_evaluated,
            "peak_layer_states": self.peak_layer_states,
            "unique_span_checks": self.unique_span_checks,
            "unique_pair_hits": self.unique_pair_hits,
            "score_access_count_during_construction": (
                self.score_access_count_during_construction
            ),
        })
        return output


@dataclass(frozen=True)
class _Topology:
    edge_count: int
    layers: Tuple[Dict[State, int], ...]
    transitions: Tuple[Tuple[Tuple[State, State], ...], ...]
    terminals: Tuple[State, ...]


@dataclass
class _ScoreCell:
    best_hits: int
    predecessor: Optional[State]
    predecessor_rank: int
    path_rank: int = -1


@dataclass(frozen=True)
class _Candidate:
    score: ExactScore
    boundaries: Boundaries
    insertions: int
    deletions: int
    operation_cost: int


class _CallbackAccess:
    def __init__(self, problem: VariableNPairwiseProblem):
        self._problem = problem
        self._span_cache: Dict[Tuple[int, int], bool] = {}
        self._pair_cache: Dict[Tuple[int, int, int], int] = {}
        self.scoring_started = False
        self.score_access_count = 0

    @property
    def unique_span_checks(self) -> int:
        return len(self._span_cache)

    @property
    def unique_pair_hits(self) -> int:
        return len(self._pair_cache)

    def span_allowed(self, start: int, end: int) -> bool:
        key = (start, end)
        if key not in self._span_cache:
            try:
                value = self._problem.span_allowed(start, end)
            except Exception as exc:
                raise InvalidOrIncompleteError(
                    "SPAN_LEGALITY_FAILURE",
                    f"span_allowed failed for {key}",
                ) from exc
            if type(value) is not bool:
                raise InvalidOrIncompleteError(
                    "NONBOOLEAN_SPAN_LEGALITY",
                    f"span_allowed returned {type(value).__name__} for {key}",
                )
            self._span_cache[key] = value
        return self._span_cache[key]

    def pair_hit(self, left: int, middle: int, right: int) -> int:
        if not self.scoring_started:
            raise InvalidOrIncompleteError(
                "SCORE_ACCESS_DURING_CONSTRUCTION",
                "pair_hit was read before topology was frozen",
            )
        key = (left, middle, right)
        if key not in self._pair_cache:
            self.score_access_count += 1
            try:
                value = self._problem.pair_hit(left, middle, right)
            except Exception as exc:
                raise InvalidOrIncompleteError(
                    "MISSING_PAIR_HIT",
                    f"pair_hit failed for {key}",
                ) from exc
            if not isinstance(value, Integral) or int(value) not in (0, 1):
                raise InvalidOrIncompleteError(
                    "NONBINARY_PAIR_HIT",
                    f"pair_hit returned {value!r} for {key}",
                )
            self._pair_cache[key] = int(value)
        return self._pair_cache[key]


def _validate_problem(problem: VariableNPairwiseProblem) -> None:
    if not isinstance(problem, VariableNPairwiseProblem):
        raise TypeError("problem must be VariableNPairwiseProblem")
    if (
        isinstance(problem.token_count, bool)
        or not isinstance(problem.token_count, int)
        or problem.token_count < 2
    ):
        raise ValueError("token_count must be an integer >= 2")
    if not isinstance(problem.original_boundaries, tuple):
        raise TypeError("original_boundaries must be a tuple")
    if not problem.original_boundaries:
        raise ValueError("the original partition must have at least one edge")
    if any(
        isinstance(boundary, bool) or not isinstance(boundary, int)
        for boundary in problem.original_boundaries
    ):
        raise TypeError("original boundaries must be integers")
    if tuple(sorted(set(problem.original_boundaries))) != (
        problem.original_boundaries
    ):
        raise ValueError("original boundaries must be strictly increasing")
    if any(
        boundary <= 0 or boundary >= problem.token_count
        for boundary in problem.original_boundaries
    ):
        raise ValueError("original boundaries must be internal atom gaps")
    if (
        isinstance(problem.radius, bool)
        or not isinstance(problem.radius, int)
        or problem.radius < 0
    ):
        raise ValueError("radius must be a nonnegative integer")
    if not callable(problem.span_allowed):
        raise TypeError("span_allowed must be callable")
    if not callable(problem.pair_hit):
        raise TypeError("pair_hit must be callable")
    if problem.threshold is not None and not isinstance(
        problem.threshold, ExactScore
    ):
        raise TypeError("threshold must be ExactScore or None")


def _edge_counts(problem: VariableNPairwiseProblem) -> Tuple[int, ...]:
    original_count = len(problem.original_boundaries)
    lower = max(1, original_count - problem.radius)
    upper = min(
        problem.token_count - 1,
        original_count + problem.radius,
    )
    return tuple(range(lower, upper + 1))


def _allowed_final_insertions(
    edge_count: int,
    original_count: int,
    radius: int,
) -> Tuple[int, ...]:
    allowed = []
    for insertions in range(edge_count + 1):
        deletions = original_count - edge_count + insertions
        if (
            deletions >= 0
            and max(insertions, deletions) <= radius
        ):
            allowed.append(insertions)
    return tuple(allowed)


def _can_complete(
    current: int,
    selected_count: int,
    insertions: int,
    edge_count: int,
    token_count: int,
    original: Boundaries,
    allowed_final_insertions: Tuple[int, ...],
) -> bool:
    remaining = edge_count - selected_count
    available_after = token_count - 1 - current
    if available_after < remaining:
        return False
    original_after = len(original) - bisect_right(original, current)
    nonoriginal_after = available_after - original_after
    minimum_new = max(0, remaining - original_after)
    maximum_new = min(remaining, nonoriginal_after)
    minimum_final = insertions + minimum_new
    maximum_final = insertions + maximum_new
    return any(
        minimum_final <= value <= maximum_final
        for value in allowed_final_insertions
    )


def _build_topology(
    problem: VariableNPairwiseProblem,
    callbacks: _CallbackAccess,
    edge_count: int,
) -> _Topology:
    token_count = problem.token_count
    original = problem.original_boundaries
    original_set = frozenset(original)
    allowed_final = _allowed_final_insertions(
        edge_count,
        len(original),
        problem.radius,
    )
    layers: list[Dict[State, int]] = [
        {} for _ in range(edge_count + 1)
    ]
    transitions: list[list[Tuple[State, State]]] = [
        [] for _ in range(edge_count + 1)
    ]
    if not allowed_final:
        return _Topology(
            edge_count=edge_count,
            layers=tuple(layers),
            transitions=tuple(tuple(items) for items in transitions),
            terminals=(),
        )

    first_layer: Dict[State, int] = {}
    for end in range(1, token_count):
        insertions = int(end not in original_set)
        if (
            _can_complete(
                end,
                1,
                insertions,
                edge_count,
                token_count,
                original,
                allowed_final,
            )
            and callbacks.span_allowed(0, end)
        ):
            first_layer[(0, end, insertions)] = 1
    layers[1] = first_layer

    for selected_count in range(1, edge_count):
        current: Dict[State, int] = {}
        for previous_state, path_count in sorted(
            layers[selected_count].items()
        ):
            _, start, insertions = previous_state
            for end in range(start + 1, token_count):
                new_insertions = (
                    insertions + int(end not in original_set)
                )
                if (
                    not _can_complete(
                        end,
                        selected_count + 1,
                        new_insertions,
                        edge_count,
                        token_count,
                        original,
                        allowed_final,
                    )
                    or not callbacks.span_allowed(start, end)
                ):
                    continue
                next_state = (start, end, new_insertions)
                current[next_state] = (
                    current.get(next_state, 0) + path_count
                )
                transitions[selected_count + 1].append(
                    (previous_state, next_state)
                )
        layers[selected_count + 1] = current

    terminals = tuple(
        state
        for state in sorted(layers[edge_count])
        if (
            state[2] in allowed_final
            and callbacks.span_allowed(state[1], token_count)
        )
    )

    viable: list[set[State]] = [
        set() for _ in range(edge_count + 1)
    ]
    viable[edge_count].update(terminals)
    for next_layer in range(edge_count, 1, -1):
        for previous_state, next_state in transitions[next_layer]:
            if next_state in viable[next_layer]:
                viable[next_layer - 1].add(previous_state)

    filtered_layers = []
    filtered_transitions = []
    for layer_index in range(edge_count + 1):
        filtered_layers.append({
            state: count
            for state, count in layers[layer_index].items()
            if state in viable[layer_index]
        })
        if layer_index < 2:
            filtered_transitions.append(())
        else:
            filtered_transitions.append(tuple(
                (previous_state, next_state)
                for previous_state, next_state
                in transitions[layer_index]
                if (
                    previous_state in viable[layer_index - 1]
                    and next_state in viable[layer_index]
                )
            ))
    return _Topology(
        edge_count=edge_count,
        layers=tuple(filtered_layers),
        transitions=tuple(filtered_transitions),
        terminals=terminals,
    )


def _candidate_is_better(
    candidate: _Candidate,
    incumbent: Optional[_Candidate],
) -> bool:
    if incumbent is None:
        return True
    score_comparison = compare_exact_scores(
        candidate.score, incumbent.score
    )
    if score_comparison != 0:
        return score_comparison < 0
    return (
        candidate.operation_cost,
        candidate.insertions,
        candidate.deletions,
        candidate.score.edge_count,
        candidate.boundaries,
    ) < (
        incumbent.operation_cost,
        incumbent.insertions,
        incumbent.deletions,
        incumbent.score.edge_count,
        incumbent.boundaries,
    )


def _reconstruct_boundaries(
    score_layers: Tuple[Dict[State, _ScoreCell], ...],
    edge_count: int,
    terminal: State,
) -> Boundaries:
    reversed_boundaries = [terminal[1]]
    state = terminal
    for selected_count in range(edge_count, 1, -1):
        predecessor = score_layers[selected_count][state].predecessor
        if predecessor is None:
            raise AssertionError("broken DP predecessor chain")
        state = predecessor
        reversed_boundaries.append(state[1])
    return tuple(reversed(reversed_boundaries))


def _score_topology(
    problem: VariableNPairwiseProblem,
    callbacks: _CallbackAccess,
    topology: _Topology,
) -> Tuple[Optional[_Candidate], int]:
    edge_count = topology.edge_count
    if not topology.terminals:
        return None, 0

    score_layers: list[Dict[State, _ScoreCell]] = [
        {} for _ in range(edge_count + 1)
    ]
    first_layer = {
        state: _ScoreCell(
            best_hits=0,
            predecessor=None,
            predecessor_rank=-1,
        )
        for state in topology.layers[1]
    }
    for rank, state in enumerate(
        sorted(first_layer, key=lambda item: item[1])
    ):
        first_layer[state].path_rank = rank
    score_layers[1] = first_layer

    transitions_evaluated = 0
    for selected_count in range(2, edge_count + 1):
        current: Dict[State, _ScoreCell] = {}
        for previous_state, next_state in topology.transitions[
            selected_count
        ]:
            previous_cell = score_layers[
                selected_count - 1
            ][previous_state]
            candidate_hits = (
                previous_cell.best_hits
                + callbacks.pair_hit(
                    previous_state[0],
                    previous_state[1],
                    next_state[1],
                )
            )
            transitions_evaluated += 1
            cell = current.get(next_state)
            if cell is None:
                current[next_state] = _ScoreCell(
                    best_hits=candidate_hits,
                    predecessor=previous_state,
                    predecessor_rank=previous_cell.path_rank,
                )
            elif (
                candidate_hits < cell.best_hits
                or (
                    candidate_hits == cell.best_hits
                    and previous_cell.path_rank
                    < cell.predecessor_rank
                )
            ):
                cell.best_hits = candidate_hits
                cell.predecessor = previous_state
                cell.predecessor_rank = previous_cell.path_rank

        if set(current) != set(topology.layers[selected_count]):
            raise AssertionError("scoring DP diverged from frozen topology")
        ranked = sorted(
            current.items(),
            key=lambda item: (
                item[1].predecessor_rank,
                item[0][1],
            ),
        )
        for rank, (_, cell) in enumerate(ranked):
            cell.path_rank = rank
        score_layers[selected_count] = current

    best: Optional[_Candidate] = None
    original_set = frozenset(problem.original_boundaries)
    for terminal in topology.terminals:
        cell = score_layers[edge_count][terminal]
        hits = (
            cell.best_hits
            + callbacks.pair_hit(
                terminal[0],
                terminal[1],
                problem.token_count,
            )
        )
        transitions_evaluated += 1
        boundaries = _reconstruct_boundaries(
            tuple(score_layers), edge_count, terminal
        )
        insertions = len(set(boundaries) - original_set)
        deletions = len(original_set - set(boundaries))
        candidate = _Candidate(
            score=ExactScore(hits, edge_count),
            boundaries=boundaries,
            insertions=insertions,
            deletions=deletions,
            operation_cost=max(insertions, deletions),
        )
        if _candidate_is_better(candidate, best):
            best = candidate
    return best, transitions_evaluated


def _direct_score(
    boundaries: Boundaries,
    token_count: int,
    callbacks: _CallbackAccess,
) -> ExactScore:
    endpoints = (0, *boundaries, token_count)
    hits = sum(
        callbacks.pair_hit(
            endpoints[index - 1],
            endpoints[index],
            endpoints[index + 1],
        )
        for index in range(1, len(endpoints) - 1)
    )
    return ExactScore(hits, len(boundaries))


def _decision(
    score: ExactScore,
    threshold: Optional[ExactScore],
) -> Optional[str]:
    if threshold is None:
        return None
    if compare_exact_scores(score, threshold) > 0:
        return CERTIFIED_WATERMARKED
    return NOT_CERTIFIED


def _solve_dp(
    problem: VariableNPairwiseProblem,
    edge_counts: Iterable[int],
    algorithm: str,
) -> VariableNPairwiseResult:
    _validate_problem(problem)
    callbacks = _CallbackAccess(problem)
    original_endpoints = (
        0,
        *problem.original_boundaries,
        problem.token_count,
    )
    if any(
        not callbacks.span_allowed(
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(len(original_endpoints) - 1)
    ):
        raise InvalidOrIncompleteError(
            "P0_NATIVE_ILLEGAL",
            "the original partition contains an illegal span",
        )

    topologies = tuple(
        _build_topology(problem, callbacks, edge_count)
        for edge_count in edge_counts
    )
    construction_score_accesses = callbacks.score_access_count
    if construction_score_accesses != 0:
        raise InvalidOrIncompleteError(
            "SCORE_ACCESS_DURING_CONSTRUCTION",
            "topology construction accessed pair_hit",
        )

    insertion_deletion_histogram: Dict[Tuple[int, int], int] = {}
    edge_count_histogram: Dict[int, int] = {}
    operation_cost_histogram: Dict[int, int] = {}
    threat_set_size = 0
    original_count = len(problem.original_boundaries)
    for topology in topologies:
        topology_count = 0
        for terminal in topology.terminals:
            count = topology.layers[topology.edge_count][terminal]
            insertions = terminal[2]
            deletions = original_count - topology.edge_count + insertions
            cost = max(insertions, deletions)
            insertion_deletion_histogram[(insertions, deletions)] = (
                insertion_deletion_histogram.get(
                    (insertions, deletions), 0
                )
                + count
            )
            operation_cost_histogram[cost] = (
                operation_cost_histogram.get(cost, 0) + count
            )
            topology_count += count
        if topology_count:
            edge_count_histogram[topology.edge_count] = topology_count
            threat_set_size += topology_count
    if threat_set_size == 0:
        raise InvalidOrIncompleteError(
            "EMPTY_THREAT_SET",
            "the declared radius has no legal partition",
        )

    callbacks.scoring_started = True
    best: Optional[_Candidate] = None
    transitions_evaluated = 0
    for topology in topologies:
        candidate, transition_count = _score_topology(
            problem, callbacks, topology
        )
        transitions_evaluated += transition_count
        if candidate is not None and _candidate_is_better(candidate, best):
            best = candidate
    if best is None:
        raise AssertionError("nonempty topology produced no score candidate")

    original_score = _direct_score(
        problem.original_boundaries,
        problem.token_count,
        callbacks,
    )
    reachable_state_count = sum(
        len(layer)
        for topology in topologies
        for layer in topology.layers
    )
    peak_layer_states = max(
        (
            len(layer)
            for topology in topologies
            for layer in topology.layers
        ),
        default=0,
    )
    return VariableNPairwiseResult(
        algorithm=algorithm,
        token_count=problem.token_count,
        original_boundaries=problem.original_boundaries,
        requested_radius=problem.radius,
        worst_boundaries=best.boundaries,
        exact_score=best.score,
        certified_score=best.score.value,
        original_exact_score=original_score,
        original_score=original_score.value,
        insertions=best.insertions,
        deletions=best.deletions,
        operation_cost=best.operation_cost,
        threshold=problem.threshold,
        decision=_decision(best.score, problem.threshold),
        threat_set_size=threat_set_size,
        histogram_by_insertions_deletions=tuple(
            (insertions, deletions, count)
            for (insertions, deletions), count
            in sorted(insertion_deletion_histogram.items())
        ),
        histogram_by_edge_count=tuple(
            sorted(edge_count_histogram.items())
        ),
        histogram_by_operation_cost=tuple(
            sorted(operation_cost_histogram.items())
        ),
        reachable_state_count=reachable_state_count,
        transitions_evaluated=transitions_evaluated,
        peak_layer_states=peak_layer_states,
        unique_span_checks=callbacks.unique_span_checks,
        unique_pair_hits=callbacks.unique_pair_hits,
        score_access_count_during_construction=(
            construction_score_accesses
        ),
    )


def exact_variable_n_pairwise_certificate(
    problem: VariableNPairwiseProblem,
) -> VariableNPairwiseResult:
    """Solve the complete variable-N threat set induced by the radius."""
    return _solve_dp(
        problem,
        _edge_counts(problem),
        "exact_variable_n_pairwise_dp",
    )


def exact_fixed_count_reduction_certificate(
    problem: VariableNPairwiseProblem,
) -> VariableNPairwiseResult:
    """Force m=m0 only for the WP2 fixed-N reduction hard gate."""
    return _solve_dp(
        problem,
        (len(problem.original_boundaries),),
        "exact_fixed_count_reduction_dp",
    )


def brute_force_variable_n_pairwise_certificate(
    problem: VariableNPairwiseProblem,
    combination_limit: int = 1_000_000,
) -> VariableNPairwiseResult:
    """Independent exhaustive combinations oracle for small instances."""
    _validate_problem(problem)
    if (
        isinstance(combination_limit, bool)
        or not isinstance(combination_limit, int)
        or combination_limit < 1
    ):
        raise ValueError("combination_limit must be a positive integer")

    callbacks = _CallbackAccess(problem)
    original = problem.original_boundaries
    original_set = frozenset(original)
    original_endpoints = (0, *original, problem.token_count)
    if any(
        not callbacks.span_allowed(
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(len(original_endpoints) - 1)
    ):
        raise InvalidOrIncompleteError(
            "P0_NATIVE_ILLEGAL",
            "the original partition contains an illegal span",
        )

    candidates = []
    insertion_deletion_histogram: Dict[Tuple[int, int], int] = {}
    edge_count_histogram: Dict[int, int] = {}
    operation_cost_histogram: Dict[int, int] = {}
    search_size = sum(
        math.comb(problem.token_count - 1, edge_count)
        for edge_count in _edge_counts(problem)
    )
    if search_size > combination_limit:
        raise ValueError(
            f"oracle search has {search_size} combinations, above "
            f"limit {combination_limit}"
        )

    for edge_count in _edge_counts(problem):
        for boundaries in combinations(
            range(1, problem.token_count), edge_count
        ):
            insertions = len(set(boundaries) - original_set)
            deletions = len(original_set - set(boundaries))
            cost = max(insertions, deletions)
            if cost > problem.radius:
                continue
            endpoints = (0, *boundaries, problem.token_count)
            if any(
                not callbacks.span_allowed(
                    endpoints[index], endpoints[index + 1]
                )
                for index in range(len(endpoints) - 1)
            ):
                continue
            candidates.append((
                boundaries,
                insertions,
                deletions,
                cost,
            ))
            insertion_deletion_histogram[(insertions, deletions)] = (
                insertion_deletion_histogram.get(
                    (insertions, deletions), 0
                )
                + 1
            )
            edge_count_histogram[edge_count] = (
                edge_count_histogram.get(edge_count, 0) + 1
            )
            operation_cost_histogram[cost] = (
                operation_cost_histogram.get(cost, 0) + 1
            )
    if not candidates:
        raise InvalidOrIncompleteError(
            "EMPTY_THREAT_SET",
            "the declared radius has no legal partition",
        )
    construction_score_accesses = callbacks.score_access_count
    if construction_score_accesses != 0:
        raise InvalidOrIncompleteError(
            "SCORE_ACCESS_DURING_CONSTRUCTION",
            "oracle topology construction accessed pair_hit",
        )

    callbacks.scoring_started = True
    best: Optional[_Candidate] = None
    transitions_evaluated = 0
    for boundaries, insertions, deletions, cost in candidates:
        score = _direct_score(
            boundaries, problem.token_count, callbacks
        )
        transitions_evaluated += len(boundaries)
        candidate = _Candidate(
            score=score,
            boundaries=boundaries,
            insertions=insertions,
            deletions=deletions,
            operation_cost=cost,
        )
        if _candidate_is_better(candidate, best):
            best = candidate
    if best is None:
        raise AssertionError("nonempty oracle topology produced no candidate")

    original_score = _direct_score(
        original, problem.token_count, callbacks
    )
    return VariableNPairwiseResult(
        algorithm="brute_force_variable_n_combinations_oracle",
        token_count=problem.token_count,
        original_boundaries=original,
        requested_radius=problem.radius,
        worst_boundaries=best.boundaries,
        exact_score=best.score,
        certified_score=best.score.value,
        original_exact_score=original_score,
        original_score=original_score.value,
        insertions=best.insertions,
        deletions=best.deletions,
        operation_cost=best.operation_cost,
        threshold=problem.threshold,
        decision=_decision(best.score, problem.threshold),
        threat_set_size=len(candidates),
        histogram_by_insertions_deletions=tuple(
            (insertions, deletions, count)
            for (insertions, deletions), count
            in sorted(insertion_deletion_histogram.items())
        ),
        histogram_by_edge_count=tuple(
            sorted(edge_count_histogram.items())
        ),
        histogram_by_operation_cost=tuple(
            sorted(operation_cost_histogram.items())
        ),
        reachable_state_count=0,
        transitions_evaluated=transitions_evaluated,
        peak_layer_states=0,
        unique_span_checks=callbacks.unique_span_checks,
        unique_pair_hits=callbacks.unique_pair_hits,
        score_access_count_during_construction=(
            construction_score_accesses
        ),
    )


def verify_result_witness(
    problem: VariableNPairwiseProblem,
    result: VariableNPairwiseResult,
) -> bool:
    """Independently recompute witness membership, legality, and score."""
    _validate_problem(problem)
    boundaries = result.worst_boundaries
    if tuple(sorted(set(boundaries))) != boundaries:
        return False
    if not boundaries or any(
        boundary <= 0 or boundary >= problem.token_count
        for boundary in boundaries
    ):
        return False
    original_set = set(problem.original_boundaries)
    candidate_set = set(boundaries)
    insertions = len(candidate_set - original_set)
    deletions = len(original_set - candidate_set)
    if max(insertions, deletions) > problem.radius:
        return False

    callbacks = _CallbackAccess(problem)
    endpoints = (0, *boundaries, problem.token_count)
    if any(
        not callbacks.span_allowed(
            endpoints[index], endpoints[index + 1]
        )
        for index in range(len(endpoints) - 1)
    ):
        return False
    callbacks.scoring_started = True
    score = _direct_score(
        boundaries, problem.token_count, callbacks
    )
    return (
        score == result.exact_score
        and score.value == result.certified_score
        and insertions == result.insertions
        and deletions == result.deletions
        and max(insertions, deletions) == result.operation_cost
        and _decision(score, problem.threshold) == result.decision
    )
