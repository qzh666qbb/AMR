"""Audit post-certificate null calibration on frozen synthetic families."""

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from certified_resegmentation import (
    PairwiseCertificateProblem,
    build_pair_scores,
    exact_pairwise_certificate,
)
from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
    strict_exceedance_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "certified_resegmentation_calibration_preflight1"
    / "audit_results.json"
)
TOKEN_COUNT = 24
ORIGINAL_BOUNDARIES = (6, 12, 18)
MIN_UNIT_WORDS = 4
MAX_UNIT_WORDS = 8
RADII = (0, 1, 2, 3)
EXPECTED_THREAT_COUNTS = (1, 13, 49, 85)
FAMILIES = ("continuous_gaussian", "tied_nine_point")
REPLICATES = 10
CALIBRATION_SIZE = 999
TEST_SIZE = 4000
TARGET_FPR = 0.05
CONFIDENCE = 0.95
TEST_UPPER_LIMIT = 0.08
DP_CHECKS_PER_FAMILY = 25


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(*parts):
    material = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("ascii")).hexdigest()[:16], 16)


def boundary_moves(boundaries):
    return len(set(boundaries) - set(ORIGINAL_BOUNDARIES))


def compile_topology():
    allowed = frozenset(range(1, TOKEN_COUNT))
    pair_keys = tuple(sorted(build_pair_scores(
        TOKEN_COUNT,
        MIN_UNIT_WORDS,
        MAX_UNIT_WORDS,
        lambda *_: 0,
        allowed,
    )))
    pair_indices = {key: index for index, key in enumerate(pair_keys)}
    candidates = []
    exact_histogram = {radius: 0 for radius in RADII}
    for boundaries in combinations(
        sorted(allowed), len(ORIGINAL_BOUNDARIES)
    ):
        endpoints = (0, *boundaries, TOKEN_COUNT)
        lengths = tuple(
            endpoints[index + 1] - endpoints[index]
            for index in range(len(endpoints) - 1)
        )
        if not all(
            MIN_UNIT_WORDS <= length <= MAX_UNIT_WORDS
            for length in lengths
        ):
            continue
        moves = boundary_moves(boundaries)
        edges = tuple(
            pair_indices[
                (
                    endpoints[index - 1],
                    endpoints[index],
                    endpoints[index + 1],
                )
            ]
            for index in range(1, len(endpoints) - 1)
        )
        candidates.append({
            "boundaries": boundaries,
            "moves": moves,
            "edge_indices": edges,
        })
        exact_histogram[moves] += 1

    cumulative_counts = tuple(
        sum(count for radius, count in exact_histogram.items() if radius <= k)
        for k in RADII
    )
    if cumulative_counts != EXPECTED_THREAT_COUNTS:
        raise AssertionError(
            f"unexpected threat counts: {cumulative_counts}"
        )
    return {
        "allowed": allowed,
        "pair_keys": pair_keys,
        "candidates": tuple(candidates),
        "exact_histogram": exact_histogram,
        "cumulative_counts": cumulative_counts,
    }


def draw_score_vector(family, rng, size):
    if family == "continuous_gaussian":
        return tuple(rng.gauss(0, 1) for _ in range(size))
    if family == "tied_nine_point":
        return tuple(float(rng.randint(-4, 4)) for _ in range(size))
    raise ValueError(f"unknown family: {family}")


def exact_scores_from_compiled(score_vector, topology):
    best_by_exact_moves = [math.inf] * len(RADII)
    for candidate in topology["candidates"]:
        score = sum(
            score_vector[index] for index in candidate["edge_indices"]
        ) / len(ORIGINAL_BOUNDARIES)
        moves = candidate["moves"]
        if score < best_by_exact_moves[moves]:
            best_by_exact_moves[moves] = score

    cumulative = []
    best = math.inf
    for radius in RADII:
        best = min(best, best_by_exact_moves[radius])
        cumulative.append(best)
    if any(not math.isfinite(value) for value in cumulative):
        raise AssertionError("compiled threat set has an empty radius")
    return tuple(cumulative)


def generate_cohort(family, seed, size, topology):
    rng = random.Random(seed)
    scores_by_radius = [[] for _ in RADII]
    monotonic_checks = 0
    for _ in range(size):
        score_vector = draw_score_vector(
            family, rng, len(topology["pair_keys"])
        )
        scores = exact_scores_from_compiled(score_vector, topology)
        if any(
            scores[index + 1] > scores[index] + 1e-12
            for index in range(len(scores) - 1)
        ):
            raise AssertionError("certified scores are not radius-monotone")
        monotonic_checks += len(scores) - 1
        for radius, score in enumerate(scores):
            scores_by_radius[radius].append(score)
    return tuple(tuple(values) for values in scores_by_radius), monotonic_checks


def dp_cross_checks(topology):
    records = []
    comparisons = 0
    for family in FAMILIES:
        seed = derived_seed("wp1c", "dp-verification", family)
        rng = random.Random(seed)
        maximum_error = 0.0
        for _ in range(DP_CHECKS_PER_FAMILY):
            score_vector = draw_score_vector(
                family, rng, len(topology["pair_keys"])
            )
            compiled_scores = exact_scores_from_compiled(
                score_vector, topology
            )
            pair_scores = dict(zip(topology["pair_keys"], score_vector))
            for radius in RADII:
                result = exact_pairwise_certificate(
                    PairwiseCertificateProblem(
                        token_count=TOKEN_COUNT,
                        original_boundaries=ORIGINAL_BOUNDARIES,
                        min_unit_words=MIN_UNIT_WORDS,
                        max_unit_words=MAX_UNIT_WORDS,
                        max_boundary_moves=radius,
                        pair_scores=pair_scores,
                        threshold=0.0,
                        normalization=len(ORIGINAL_BOUNDARIES),
                        allowed_boundaries=topology["allowed"],
                    )
                )
                error = abs(
                    result.certified_score - compiled_scores[radius]
                )
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    raise AssertionError(
                        "compiled enumerator and edge-state DP differ"
                    )
                if result.threat_set_size != EXPECTED_THREAT_COUNTS[radius]:
                    raise AssertionError("DP threat-set count differs")
                comparisons += 1
        records.append({
            "family": family,
            "seed": seed,
            "documents": DP_CHECKS_PER_FAMILY,
            "radius_comparisons": DP_CHECKS_PER_FAMILY * len(RADII),
            "maximum_absolute_error": maximum_error,
            "status": "pass",
        })
    return {
        "records": records,
        "total_radius_comparisons": comparisons,
        "status": "pass",
    }


def replicate_result(family, replicate, topology):
    calibration_seed = derived_seed(
        "wp1c", family, replicate, "calibration"
    )
    test_seed = derived_seed("wp1c", family, replicate, "test")
    if calibration_seed == test_seed:
        raise AssertionError("calibration and test seeds collide")

    calibration_scores, calibration_monotonic_checks = generate_cohort(
        family, calibration_seed, CALIBRATION_SIZE, topology
    )
    test_scores, test_monotonic_checks = generate_cohort(
        family, test_seed, TEST_SIZE, topology
    )
    calibrations = tuple(
        split_conformal_calibrate(scores, TARGET_FPR)
        for scores in calibration_scores
    )
    thresholds = tuple(result.threshold for result in calibrations)
    if any(
        thresholds[index + 1] > thresholds[index] + 1e-12
        for index in range(len(thresholds) - 1)
    ):
        raise AssertionError("post-certificate thresholds are not monotone")

    raw_threshold = thresholds[0]
    radius_records = []
    raw_counts = []
    for radius in RADII:
        post_count = strict_exceedance_count(
            test_scores[radius], thresholds[radius]
        )
        raw_count = strict_exceedance_count(
            test_scores[radius], raw_threshold
        )
        raw_counts.append(raw_count)
        post_upper = clopper_pearson_upper(
            post_count, TEST_SIZE, CONFIDENCE
        )
        calibration = calibrations[radius]
        gates = {
            "calibration_exceedance_within_bound": (
                calibration.strict_exceedances
                <= calibration.maximum_calibration_exceedances
            ),
            "post_threshold_not_above_raw": (
                thresholds[radius] <= raw_threshold + 1e-12
            ),
            "post_test_cp_upper_at_most_0_08": (
                post_upper <= TEST_UPPER_LIMIT
            ),
            "continuous_post_test_nonzero": (
                family != "continuous_gaussian" or post_count > 0
            ),
        }
        radius_records.append({
            "radius": radius,
            "calibration": calibration.to_dict(),
            "post_certificate_test": {
                "strict_exceedances": post_count,
                "sample_size": TEST_SIZE,
                "empirical_fpr": post_count / TEST_SIZE,
                "one_sided_cp_upper": post_upper,
                "confidence": CONFIDENCE,
            },
            "raw_threshold_test": {
                "threshold": raw_threshold,
                "strict_exceedances": raw_count,
                "sample_size": TEST_SIZE,
                "empirical_fpr": raw_count / TEST_SIZE,
            },
            "gates": gates,
        })

    raw_monotone = all(
        raw_counts[index + 1] <= raw_counts[index]
        for index in range(len(raw_counts) - 1)
    )
    all_radius_gates = all(
        all(record["gates"].values()) for record in radius_records
    )
    return {
        "replicate": replicate,
        "calibration_seed": calibration_seed,
        "test_seed": test_seed,
        "calibration_test_seeds_distinct": True,
        "calibration_score_monotonic_checks": calibration_monotonic_checks,
        "test_score_monotonic_checks": test_monotonic_checks,
        "thresholds_radius_monotone": True,
        "raw_threshold_test_fpr_radius_monotone": raw_monotone,
        "radii": radius_records,
        "status": (
            "pass" if raw_monotone and all_radius_gates else "fail"
        ),
    }


def aggregate_family(family, replicates):
    radius_summaries = []
    for radius in RADII:
        post_rates = [
            record["radii"][radius]["post_certificate_test"][
                "empirical_fpr"
            ]
            for record in replicates
        ]
        raw_rates = [
            record["radii"][radius]["raw_threshold_test"]["empirical_fpr"]
            for record in replicates
        ]
        cp_uppers = [
            record["radii"][radius]["post_certificate_test"][
                "one_sided_cp_upper"
            ]
            for record in replicates
        ]
        thresholds = [
            record["radii"][radius]["calibration"]["threshold"]
            for record in replicates
        ]
        radius_summaries.append({
            "radius": radius,
            "post_certificate_test_fpr": {
                "mean": statistics.fmean(post_rates),
                "minimum": min(post_rates),
                "maximum": max(post_rates),
            },
            "raw_threshold_test_fpr": {
                "mean": statistics.fmean(raw_rates),
                "minimum": min(raw_rates),
                "maximum": max(raw_rates),
            },
            "post_certificate_cp_upper_maximum": max(cp_uppers),
            "post_certificate_threshold": {
                "mean": statistics.fmean(thresholds),
                "minimum": min(thresholds),
                "maximum": max(thresholds),
            },
        })
    return {
        "family": family,
        "replicates": replicates,
        "radius_summaries": radius_summaries,
        "all_replicates_pass": all(
            record["status"] == "pass" for record in replicates
        ),
    }


def run_simulation(topology):
    families = []
    for family in FAMILIES:
        replicates = [
            replicate_result(family, replicate, topology)
            for replicate in range(REPLICATES)
        ]
        families.append(aggregate_family(family, replicates))
    return families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    topology = compile_topology()
    cross_checks = dp_cross_checks(topology)
    families = run_simulation(topology)
    all_families_pass = all(
        family["all_replicates_pass"] for family in families
    )
    status = (
        "pass"
        if cross_checks["status"] == "pass" and all_families_pass
        else "stop_gate_failed"
    )
    elapsed = time.perf_counter() - started

    source_paths = [
        ROOT / "scripts" / "certified_resegmentation.py",
        ROOT / "scripts" / "certified_resegmentation_calibration.py",
        ROOT / "scripts" / "test_certified_resegmentation_calibration.py",
        ROOT / "scripts" / "audit_certified_resegmentation_calibration.py",
        ROOT
        / "experiments"
        / "certified_resegmentation_calibration_preflight1"
        / "PROTOCOL.md",
    ]
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "WP1c synthetic post-certificate null calibration and "
            "independent-test implementation preflight only"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "protocol": {
            "families": list(FAMILIES),
            "replicates_per_family": REPLICATES,
            "calibration_size_per_replicate": CALIBRATION_SIZE,
            "test_size_per_replicate": TEST_SIZE,
            "target_fpr": TARGET_FPR,
            "test_confidence": CONFIDENCE,
            "test_cp_upper_stop_limit": TEST_UPPER_LIMIT,
            "dp_checks_per_family": DP_CHECKS_PER_FAMILY,
        },
        "topology": {
            "token_count": TOKEN_COUNT,
            "original_boundaries": list(ORIGINAL_BOUNDARIES),
            "unit_count": len(ORIGINAL_BOUNDARIES) + 1,
            "min_unit_words": MIN_UNIT_WORDS,
            "max_unit_words": MAX_UNIT_WORDS,
            "radii": list(RADII),
            "pair_score_count": len(topology["pair_keys"]),
            "exact_move_histogram": {
                str(radius): topology["exact_histogram"][radius]
                for radius in RADII
            },
            "cumulative_threat_set_sizes": {
                str(radius): topology["cumulative_counts"][radius]
                for radius in RADII
            },
        },
        "theorem_checks": {
            "score_direction": "C[k+1] <= C[k] <= C[0]",
            "fixed_raw_threshold_event_inclusion": (
                "{C[k] > tau} subset_of {C[0] > tau}"
            ),
            "split_conformal_rank": "ceil((n+1)*(1-alpha))",
            "strict_threshold": True,
            "ties_are_conservative": True,
        },
        "compiled_enumerator_dp_cross_checks": cross_checks,
        "families": families,
        "gates": {
            "compiled_enumerator_matches_edge_state_dp": (
                cross_checks["status"] == "pass"
            ),
            "all_document_scores_radius_monotone": True,
            "all_thresholds_radius_monotone": all(
                all(
                    replicate["thresholds_radius_monotone"]
                    for replicate in family["replicates"]
                )
                for family in families
            ),
            "all_raw_threshold_test_fpr_radius_monotone": all(
                all(
                    replicate["raw_threshold_test_fpr_radius_monotone"]
                    for replicate in family["replicates"]
                )
                for family in families
            ),
            "all_calibration_exceedance_bounds_pass": all(
                all(
                    all(
                        radius["gates"][
                            "calibration_exceedance_within_bound"
                        ]
                        for radius in replicate["radii"]
                    )
                    for replicate in family["replicates"]
                )
                for family in families
            ),
            "all_test_cp_upper_stop_gates_pass": all(
                all(
                    all(
                        radius["gates"][
                            "post_test_cp_upper_at_most_0_08"
                        ]
                        for radius in replicate["radii"]
                    )
                    for replicate in family["replicates"]
                )
                for family in families
            ),
            "all_continuous_post_test_counts_nonzero": all(
                all(
                    all(
                        radius["gates"]["continuous_post_test_nonzero"]
                        for radius in replicate["radii"]
                    )
                    for replicate in family["replicates"]
                )
                for family in families
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in source_paths
        },
        "total_elapsed_seconds": elapsed,
        "interpretation": {
            "raw_threshold_fpr_inflation_possible_for_minimum_certificate": False,
            "raw_threshold_can_be_overly_conservative": True,
            "post_certificate_calibration_required_for_equal_fpr_comparison": True,
            "real_detector_fpr_tested": False,
            "real_positive_nonvacuity_tested": False,
            "wp2_authorized": False,
        },
        "execution_attestation": {
            "provider_api_calls": 0,
            "watermark_generation_calls": 0,
            "watermark_detector_calls": 0,
            "llm_judge_calls": 0,
            "amr_parser_calls": 0,
            "s2match_calls": 0,
            "model_downloads": 0,
            "dataset_payload_downloads": 0,
            "external_gpu_jobs": 0,
        },
    }
    if not all(payload["gates"].values()):
        payload["status"] = "stop_gate_failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "dp_cross_checks": cross_checks["total_radius_comparisons"],
        "synthetic_documents": (
            len(FAMILIES)
            * REPLICATES
            * (CALIBRATION_SIZE + TEST_SIZE)
        ),
        "total_elapsed_seconds": elapsed,
    }, indent=2))
    if payload["status"] != "pass":
        raise SystemExit("WP1c calibration preflight gate failed")


if __name__ == "__main__":
    main()
