"""Grouped sparse pairwise certificates for WP2-v4.

The v4 implementation preserves the v3 problem and result contracts. It groups
states by their final two boundaries, bounds pair-score caching to the only
layers where a triple can recur, and stores predecessor indices in packed
arrays for deterministic witness replay.
"""

from array import array
from bisect import bisect_right
import math
from typing import Dict, Tuple

from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    SparsePairwiseResult,
    _validate_sparse_problem,
)


_MISSING = object()


def rank_candidate_boundaries_grouped(
    token_count: int,
    original_boundaries: Tuple[int, ...],
    max_boundary_moves: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Return the v3 rank projection using its closed-form intervals."""
    boundary_count = len(original_boundaries)
    candidates = []
    for rank in range(1, boundary_count + 1):
        lower_original_rank = rank - max_boundary_moves
        upper_original_rank = rank + max_boundary_moves
        lower = (
            0
            if lower_original_rank <= 0
            else original_boundaries[lower_original_rank - 1]
        )
        upper = (
            token_count
            if upper_original_rank > boundary_count
            else original_boundaries[upper_original_rank - 1]
        )
        lower = max(rank, lower)
        upper = min(
            token_count - boundary_count + rank - 1,
            upper,
        )
        candidates.append(tuple(range(lower, upper + 1)))
    return tuple(candidates)


def exact_grouped_sparse_pairwise_certificate(
    problem: SparsePairwiseProblem,
    *,
    progress_callback=None,
) -> SparsePairwiseResult:
    """Compute the v3 certificate with grouped, layer-local sparse state."""
    normalization = _validate_sparse_problem(problem)
    n = problem.token_count
    original = problem.original_boundaries
    original_set = frozenset(original)
    boundary_count = len(original)
    k = problem.max_boundary_moves
    move_width = k + 1
    candidates = rank_candidate_boundaries_grouped(n, original, k)

    allowed_spans = set()
    denied_spans = set()

    def span_is_allowed(start: int, end: int) -> bool:
        key = start * (n + 1) + end
        if key in allowed_spans:
            return True
        if key in denied_spans:
            return False
        value = problem.span_allowed(start, end)
        if not isinstance(value, bool):
            raise TypeError("span_allowed must return bool")
        (allowed_spans if value else denied_spans).add(key)
        return value

    endpoints = (0, *original, n)
    if any(
        not span_is_allowed(endpoints[index], endpoints[index + 1])
        for index in range(len(endpoints) - 1)
    ):
        raise ValueError("the original partition contains an illegal span")

    triple_base = n + 1
    pair_cache: Dict[int, float] = {}
    pair_keys_by_layer = [[], []]
    unique_pair_scores = 0
    original_pair_scores: Dict[int, float] = {}

    def pair_score(
        left: int,
        middle: int,
        right: int,
        layer: int,
    ) -> float:
        nonlocal unique_pair_scores
        key = (left * triple_base + middle) * triple_base + right
        value = pair_cache.get(key, _MISSING)
        if value is _MISSING:
            value = original_pair_scores.get(key, _MISSING)
            if value is _MISSING:
                value = float(problem.pair_scorer(left, middle, right))
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite pair score for "
                        f"{(left, middle, right)}"
                    )
                unique_pair_scores += 1
            pair_cache[key] = value
            pair_keys_by_layer[layer].append(key)
        return value

    def expire_pair_scores(layer: int) -> None:
        expired_layer = layer - (2 * k + 1)
        if expired_layer < 2:
            return
        for key in pair_keys_by_layer[expired_layer]:
            del pair_cache[key]

    # A boundary can shift by at most k ranks in either direction. Therefore a
    # fixed triple can recur only within 2k layers, which makes this cache exact.
    for index in range(1, len(endpoints) - 1):
        left = endpoints[index - 1]
        middle = endpoints[index]
        right = endpoints[index + 1]
        key = (left * triple_base + middle) * triple_base + right
        value = float(problem.pair_scorer(left, middle, right))
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite pair score for {(left, middle, right)}"
            )
        original_pair_scores[key] = value
    unique_pair_scores = len(original_pair_scores)
    original_sum = sum(original_pair_scores.values())

    predecessor_history = [None]
    boundary_history = [None]

    previous_groups = {}
    previous_best = []
    previous_counts = []
    previous_path_ranks = []
    previous_pair_codes = []
    previous_moves = bytearray()

    remaining_after_first = boundary_count - 1
    for end in candidates[0]:
        moves = int(end not in original_set)
        original_after = boundary_count - bisect_right(original, end)
        minimum_future_new = max(
            0, remaining_after_first - original_after
        )
        if (
            moves + minimum_future_new <= k
            and span_is_allowed(0, end)
        ):
            state_index = len(previous_best)
            slots = [-1] * move_width
            slots[moves] = state_index
            previous_groups[end] = slots
            previous_best.append(0.0)
            previous_counts.append(1)
            previous_path_ranks.append(state_index)
            previous_pair_codes.append(end)
            previous_moves.append(moves)

    predecessor_history.append(
        array("I", [0] * len(previous_best))
    )
    boundary_history.append(array("I", (
        code % triple_base for code in previous_pair_codes
    )))
    reachable_state_count = len(previous_best)
    peak_layer_states = len(previous_best)
    transitions_evaluated = 0

    for selected_count in range(1, boundary_count):
        layer = selected_count + 1
        pair_keys_by_layer.append([])
        next_values = candidates[selected_count]
        next_is_new = bytearray(
            int(value not in original_set) for value in next_values
        )
        remaining = boundary_count - layer
        next_move_caps = bytearray(
            k - max(
                0,
                remaining
                - (boundary_count - bisect_right(original, value)),
            )
            for value in next_values
        )

        current_groups = {}
        current_best = []
        current_counts = []
        current_predecessors = []
        current_predecessor_ranks = []
        current_pair_codes = []
        current_moves = bytearray()

        for pair_code, slots in previous_groups.items():
            left, middle = divmod(pair_code, triple_base)
            active = tuple(
                (moves, state_index)
                for moves, state_index in enumerate(slots)
                if state_index >= 0
            )
            for value_index in range(
                bisect_right(next_values, middle),
                len(next_values),
            ):
                end = next_values[value_index]
                added_move = next_is_new[value_index]
                move_cap = next_move_caps[value_index]
                eligible_limit = move_cap - added_move
                eligible_count = 0
                for moves, _ in active:
                    if moves > eligible_limit:
                        break
                    eligible_count += 1
                if not eligible_count or not span_is_allowed(middle, end):
                    continue

                score = pair_score(left, middle, end, layer)
                destination_code = middle * triple_base + end
                destination_slots = current_groups.get(destination_code)
                if destination_slots is None:
                    destination_slots = [-1] * move_width
                    current_groups[destination_code] = destination_slots

                transitions_evaluated += eligible_count
                for active_index in range(eligible_count):
                    moves, state_index = active[active_index]
                    new_moves = moves + added_move
                    candidate_sum = previous_best[state_index] + score
                    destination_index = destination_slots[new_moves]
                    predecessor_rank = previous_path_ranks[state_index]
                    if destination_index < 0:
                        destination_index = len(current_best)
                        destination_slots[new_moves] = destination_index
                        current_best.append(candidate_sum)
                        current_counts.append(
                            previous_counts[state_index]
                        )
                        current_predecessors.append(state_index)
                        current_predecessor_ranks.append(
                            predecessor_rank
                        )
                        current_pair_codes.append(destination_code)
                        current_moves.append(new_moves)
                        continue

                    current_counts[destination_index] += (
                        previous_counts[state_index]
                    )
                    if (
                        candidate_sum
                        < current_best[destination_index]
                        or (
                            candidate_sum
                            == current_best[destination_index]
                            and predecessor_rank
                            < current_predecessor_ranks[
                                destination_index
                            ]
                        )
                    ):
                        current_best[destination_index] = candidate_sum
                        current_predecessors[
                            destination_index
                        ] = state_index
                        current_predecessor_ranks[
                            destination_index
                        ] = predecessor_rank

        order = sorted(
            range(len(current_best)),
            key=lambda index: (
                current_predecessor_ranks[index],
                current_pair_codes[index] % triple_base,
            ),
        )
        current_path_ranks = [0] * len(current_best)
        for path_rank, state_index in enumerate(order):
            current_path_ranks[state_index] = path_rank

        predecessor_history.append(array("I", current_predecessors))
        boundary_history.append(array("I", (
            code % triple_base for code in current_pair_codes
        )))
        reachable_state_count += len(current_best)
        peak_layer_states = max(peak_layer_states, len(current_best))

        previous_groups = current_groups
        previous_best = current_best
        previous_counts = current_counts
        previous_path_ranks = current_path_ranks
        previous_pair_codes = current_pair_codes
        previous_moves = current_moves
        expire_pair_scores(layer)
        if progress_callback is not None:
            progress_callback({
                "layer": layer,
                "layer_states": len(current_best),
                "reachable_states": reachable_state_count,
                "transitions": transitions_evaluated,
                "unique_pair_scores": unique_pair_scores,
                "unique_span_checks": (
                    len(allowed_spans) + len(denied_spans)
                ),
            })

    pair_keys_by_layer.append([])
    final_layer = boundary_count + 1
    histogram = {}
    worst_sum = math.inf
    worst_moves = None
    worst_path_rank = None
    worst_state_index = None
    for pair_code, slots in previous_groups.items():
        left, middle = divmod(pair_code, triple_base)
        active = tuple(
            (moves, state_index)
            for moves, state_index in enumerate(slots)
            if state_index >= 0
        )
        if not active or not span_is_allowed(middle, n):
            continue
        score = pair_score(left, middle, n, final_layer)
        transitions_evaluated += len(active)
        for moves, state_index in active:
            final_sum = previous_best[state_index] + score
            histogram[moves] = (
                histogram.get(moves, 0)
                + previous_counts[state_index]
            )
            path_rank = previous_path_ranks[state_index]
            if (
                final_sum < worst_sum
                or (
                    final_sum == worst_sum
                    and (
                        worst_moves is None
                        or moves < worst_moves
                        or (
                            moves == worst_moves
                            and path_rank < worst_path_rank
                        )
                    )
                )
            ):
                worst_sum = final_sum
                worst_moves = moves
                worst_path_rank = path_rank
                worst_state_index = state_index
    if worst_state_index is None:
        raise ValueError("the declared threat set is empty")

    reversed_boundaries = []
    state_index = worst_state_index
    for layer in range(boundary_count, 0, -1):
        reversed_boundaries.append(boundary_history[layer][state_index])
        if layer > 1:
            state_index = predecessor_history[layer][state_index]
    worst_boundaries: Tuple[int, ...] = tuple(
        reversed(reversed_boundaries)
    )

    return SparsePairwiseResult(
        algorithm="exact_grouped_layer_local_sparse_pairwise_dp",
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
        unique_span_checks=len(allowed_spans) + len(denied_spans),
        unique_pair_scores=unique_pair_scores,
    )
