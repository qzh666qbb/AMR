"""Exact variable-N wrapper for SemStamp v2 refined-atom topology."""

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Optional, Tuple

from semstamp_certified_defense_v1_variable_n import (
    ExactScore,
    InvalidOrIncompleteError,
    PairHit,
    SpanAllowed,
    VariableNPairwiseProblem,
    VariableNPairwiseResult,
    brute_force_variable_n_pairwise_certificate,
    exact_fixed_count_reduction_certificate,
    exact_variable_n_pairwise_certificate,
    verify_result_witness,
)
from semstamp_refined_atom_renderer import RefinedAtomDocument


Boundaries = Tuple[int, ...]
GapByteLengths = Tuple[int, ...]


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RefinedVariableNPairwiseProblem:
    atom_count: int
    original_boundaries: Boundaries
    gap_byte_lengths: GapByteLengths
    radius: int
    span_allowed: SpanAllowed
    pair_hit: PairHit
    threshold: Optional[ExactScore] = None


@dataclass(frozen=True)
class RefinedVariableNPairwiseResult:
    core_result: VariableNPairwiseResult
    gap_byte_lengths: GapByteLengths
    zero_byte_gap_boundaries: Boundaries
    gap_metadata_sha256: str

    def __getattr__(self, name):
        return getattr(self.core_result, name)

    def semantic_payload(self) -> dict:
        output = self.core_result.semantic_payload()
        output["refined_topology"] = {
            "gap_byte_lengths": list(self.gap_byte_lengths),
            "zero_byte_gap_boundaries": list(
                self.zero_byte_gap_boundaries
            ),
            "gap_metadata_sha256": self.gap_metadata_sha256,
        }
        return output

    def to_dict(self) -> dict:
        output = self.core_result.to_dict()
        output["refined_topology"] = {
            "gap_byte_lengths": list(self.gap_byte_lengths),
            "zero_byte_gap_boundaries": list(
                self.zero_byte_gap_boundaries
            ),
            "gap_metadata_sha256": self.gap_metadata_sha256,
        }
        return output


def _validate_refined_metadata(
    problem: RefinedVariableNPairwiseProblem,
) -> None:
    if not isinstance(problem, RefinedVariableNPairwiseProblem):
        raise TypeError(
            "problem must be RefinedVariableNPairwiseProblem"
        )
    if (
        isinstance(problem.atom_count, bool)
        or not isinstance(problem.atom_count, int)
        or problem.atom_count < 2
    ):
        raise ValueError("atom_count must be an integer >= 2")
    if not isinstance(problem.gap_byte_lengths, tuple):
        raise TypeError("gap_byte_lengths must be a tuple")
    if len(problem.gap_byte_lengths) != problem.atom_count - 1:
        raise InvalidOrIncompleteError(
            "REFINED_GAP_COUNT_MISMATCH",
            "gap_byte_lengths must contain atom_count - 1 entries",
        )
    for boundary, byte_length in enumerate(
        problem.gap_byte_lengths, start=1
    ):
        if (
            isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 0
        ):
            raise InvalidOrIncompleteError(
                "INVALID_GAP_BYTE_LENGTH",
                f"gap {boundary} has invalid byte length {byte_length!r}",
            )
    original_set = frozenset(problem.original_boundaries)
    for boundary, byte_length in enumerate(
        problem.gap_byte_lengths, start=1
    ):
        if byte_length == 0 and boundary not in original_set:
            raise InvalidOrIncompleteError(
                "UNAUTHORIZED_ZERO_BYTE_GAP",
                f"zero-byte gap {boundary} is not in B0",
            )


def _gap_metadata(
    problem: RefinedVariableNPairwiseProblem,
) -> Tuple[Boundaries, str]:
    zero_boundaries = tuple(
        boundary
        for boundary, byte_length in enumerate(
            problem.gap_byte_lengths, start=1
        )
        if byte_length == 0
    )
    digest = _canonical_sha256({
        "atom_count": problem.atom_count,
        "gap_byte_lengths": list(problem.gap_byte_lengths),
        "zero_byte_gap_boundaries": list(zero_boundaries),
    })
    return zero_boundaries, digest


def _to_core_problem(
    problem: RefinedVariableNPairwiseProblem,
) -> VariableNPairwiseProblem:
    _validate_refined_metadata(problem)
    return VariableNPairwiseProblem(
        token_count=problem.atom_count,
        original_boundaries=problem.original_boundaries,
        radius=problem.radius,
        span_allowed=problem.span_allowed,
        pair_hit=problem.pair_hit,
        threshold=problem.threshold,
    )


def _wrap_result(
    problem: RefinedVariableNPairwiseProblem,
    result: VariableNPairwiseResult,
    algorithm: str,
) -> RefinedVariableNPairwiseResult:
    zero_boundaries, digest = _gap_metadata(problem)
    return RefinedVariableNPairwiseResult(
        core_result=replace(result, algorithm=algorithm),
        gap_byte_lengths=problem.gap_byte_lengths,
        zero_byte_gap_boundaries=zero_boundaries,
        gap_metadata_sha256=digest,
    )


def exact_refined_variable_n_pairwise_certificate(
    problem: RefinedVariableNPairwiseProblem,
) -> RefinedVariableNPairwiseResult:
    core_problem = _to_core_problem(problem)
    return _wrap_result(
        problem,
        exact_variable_n_pairwise_certificate(core_problem),
        "exact_refined_atom_variable_n_pairwise_dp",
    )


def brute_force_refined_variable_n_pairwise_certificate(
    problem: RefinedVariableNPairwiseProblem,
    combination_limit: int = 1_000_000,
) -> RefinedVariableNPairwiseResult:
    core_problem = _to_core_problem(problem)
    return _wrap_result(
        problem,
        brute_force_variable_n_pairwise_certificate(
            core_problem,
            combination_limit=combination_limit,
        ),
        "brute_force_refined_atom_combinations_oracle",
    )


def exact_refined_fixed_count_reduction_certificate(
    problem: RefinedVariableNPairwiseProblem,
) -> RefinedVariableNPairwiseResult:
    core_problem = _to_core_problem(problem)
    return _wrap_result(
        problem,
        exact_fixed_count_reduction_certificate(core_problem),
        "exact_refined_atom_fixed_count_reduction_dp",
    )


def verify_refined_result_witness(
    problem: RefinedVariableNPairwiseProblem,
    result: RefinedVariableNPairwiseResult,
) -> bool:
    _validate_refined_metadata(problem)
    if not isinstance(result, RefinedVariableNPairwiseResult):
        return False
    zero_boundaries, digest = _gap_metadata(problem)
    if (
        result.gap_byte_lengths != problem.gap_byte_lengths
        or result.zero_byte_gap_boundaries != zero_boundaries
        or result.gap_metadata_sha256 != digest
    ):
        return False
    return verify_result_witness(
        _to_core_problem(problem), result.core_result
    )


def refined_problem_from_document(
    document: RefinedAtomDocument,
    radius: int,
    span_allowed: SpanAllowed,
    pair_hit: PairHit,
    threshold: Optional[ExactScore] = None,
) -> RefinedVariableNPairwiseProblem:
    """Bind a validated refined renderer document to the exact solver."""
    if not isinstance(document, RefinedAtomDocument):
        raise TypeError("document must be RefinedAtomDocument")

    gap_lengths = []
    original_set = frozenset(document.original_boundaries)
    for boundary in range(1, document.atom_count):
        gap = document.gap_after(boundary)
        if not gap:
            if boundary not in original_set:
                raise InvalidOrIncompleteError(
                    "UNAUTHORIZED_ZERO_BYTE_GAP",
                    f"renderer zero-byte gap {boundary} is not in B0",
                )
        else:
            try:
                decoded = gap.decode("utf-8", "strict")
            except UnicodeDecodeError as exc:
                raise InvalidOrIncompleteError(
                    "NONWHITESPACE_REFINED_GAP",
                    f"renderer gap {boundary} is not valid UTF-8",
                ) from exc
            if not decoded or any(
                not character.isspace() for character in decoded
            ):
                raise InvalidOrIncompleteError(
                    "NONWHITESPACE_REFINED_GAP",
                    f"renderer gap {boundary} contains non-whitespace",
                )
        gap_lengths.append(len(gap))

    problem = RefinedVariableNPairwiseProblem(
        atom_count=document.atom_count,
        original_boundaries=document.original_boundaries,
        gap_byte_lengths=tuple(gap_lengths),
        radius=radius,
        span_allowed=span_allowed,
        pair_hit=pair_hit,
        threshold=threshold,
    )
    _validate_refined_metadata(problem)
    return problem
