"""Finite-sample calibration helpers for certified detector scores."""

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Tuple


@dataclass(frozen=True)
class CalibrationResult:
    calibration_size: int
    target_fpr: float
    order_rank: int
    threshold: float
    maximum_calibration_exceedances: int
    strict_exceedances: int
    empirical_fpr: float
    marginal_fpr_bound: float

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_scores(scores: Iterable[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in scores)
    if not values:
        raise ValueError("calibration scores must be nonempty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("calibration scores must be finite")
    return values


def split_conformal_calibrate(
    scores: Iterable[float],
    target_fpr: float,
) -> CalibrationResult:
    """Freeze an upper-tail threshold with a finite-sample marginal bound.

    Detection uses strict ``score > threshold``. Ties are therefore retained
    on the null side and can only make the realized exceedance rate smaller.
    """
    values = _finite_scores(scores)
    if not math.isfinite(target_fpr) or not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be finite and in (0, 1)")

    sample_size = len(values)
    rank = math.ceil((sample_size + 1) * (1 - target_fpr))
    threshold = (
        math.inf
        if rank == sample_size + 1
        else sorted(values)[rank - 1]
    )
    maximum_exceedances = max(0, sample_size - rank)
    exceedances = sum(value > threshold for value in values)
    marginal_bound = (sample_size + 1 - rank) / (sample_size + 1)
    return CalibrationResult(
        calibration_size=sample_size,
        target_fpr=target_fpr,
        order_rank=rank,
        threshold=threshold,
        maximum_calibration_exceedances=maximum_exceedances,
        strict_exceedances=exceedances,
        empirical_fpr=exceedances / sample_size,
        marginal_fpr_bound=marginal_bound,
    )


def strict_exceedance_count(
    scores: Iterable[float],
    threshold: float,
) -> int:
    values = tuple(float(value) for value in scores)
    if not math.isfinite(threshold) and threshold != math.inf:
        raise ValueError("threshold must be finite or positive infinity")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("test scores must be finite")
    return sum(value > threshold for value in values)


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Compute P[X <= successes] for X~Binomial(trials, probability)."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 0
    ):
        raise ValueError("successes and trials must be integer counts")
    if not -1 <= successes <= trials:
        raise ValueError("successes must lie in [-1, trials]")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if successes < 0:
        return 0.0
    if successes == trials:
        return 1.0
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0

    log_p = math.log(probability)
    log_one_minus_p = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_p
        + (trials - index) * log_one_minus_p
        for index in range(successes + 1)
    ]
    maximum = max(log_terms)
    return min(
        1.0,
        math.exp(maximum)
        * sum(math.exp(value - maximum) for value in log_terms),
    )


def clopper_pearson_upper(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    """One-sided exact Clopper--Pearson upper confidence limit."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 1
        or not 0 <= successes <= trials
    ):
        raise ValueError("require integer counts with 0 <= successes <= trials")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    if successes == trials:
        return 1.0

    tail_probability = 1 - confidence
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if binomial_cdf(successes, trials, midpoint) > tail_probability:
            lower = midpoint
        else:
            upper = midpoint
    return upper
