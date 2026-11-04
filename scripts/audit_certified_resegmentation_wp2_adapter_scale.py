"""Benchmark the model-free WP2 adapter at the frozen threat-set scale."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

from certified_resegmentation import exact_pairwise_certificate
from certified_resegmentation_wp2_adapter import (
    build_document_pairwise_problem,
    canonicalize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "certified_resegmentation_wp2_adapter_scale1"
    / "audit_results.json"
)
TOKEN_COUNTS = (100, 200, 300)
SYSTEM_IDS = ("semstamp", "simmark")
MIN_UNIT_ATOMS = 5
MAX_UNIT_ATOMS = 35
UNIT_ATOMS = 20
RADII = (0, 1, 2, 3)
PAIR_PRECOMPUTE_LIMIT_SECONDS = 5.0
DP_LIMIT_SECONDS = 10.0
EXPECTED_LARGEST_PAIR_COUNT = 250_821
EXPECTED_LARGEST_THREAT_SET = 14_606_509


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_pair_score(previous, current):
    payload = previous.encode("utf-8") + b"\0" + current.encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value % 2001 - 1000) / 100.0


def make_document(token_count):
    if token_count % UNIT_ATOMS:
        raise ValueError("synthetic token count must be divisible by UNIT_ATOMS")
    atoms = [f"atom{index:03d}" for index in range(token_count)]
    units = [
        " ".join(atoms[start:start + UNIT_ATOMS])
        for start in range(0, token_count, UNIT_ATOMS)
    ]
    return canonicalize(units)


def pair_score_digest(pair_scores):
    digest = hashlib.sha256()
    for key, value in sorted(pair_scores.items()):
        digest.update(
            (
                f"{key[0]},{key[1]},{key[2]}:{float(value).hex()}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def score_partition(document, system_id, boundaries):
    units = document.preprocess_partition(system_id, boundaries)
    return sum(
        text_pair_score(units[index - 1], units[index])
        for index in range(1, len(units))
    ) / (len(units) - 1)


def stable_result(result):
    return {
        "radius": result.requested_radius,
        "original_boundaries": list(result.original_boundaries),
        "worst_boundaries": list(result.worst_boundaries),
        "original_score": result.original_score,
        "certified_score": result.certified_score,
        "certified_positive": result.certified_positive,
        "threat_set_size": result.threat_set_size,
        "exact_move_histogram": {
            str(radius): count
            for radius, count in result.exact_move_histogram
        },
        "reachable_state_count": result.reachable_state_count,
        "transitions_evaluated": result.transitions_evaluated,
        "peak_layer_states": result.peak_layer_states,
    }


def run_configuration(token_count, system_id):
    document = make_document(token_count)
    scorer_calls = 0

    def counted_scorer(previous, current):
        nonlocal scorer_calls
        scorer_calls += 1
        return text_pair_score(previous, current)

    precompute_started = time.perf_counter()
    base_problem = build_document_pairwise_problem(
        document,
        system_id=system_id,
        min_unit_atoms=MIN_UNIT_ATOMS,
        max_unit_atoms=MAX_UNIT_ATOMS,
        max_boundary_moves=0,
        scorer=counted_scorer,
        threshold=0.0,
    )
    precompute_seconds = time.perf_counter() - precompute_started
    pair_count = len(base_problem.pair_scores)
    if scorer_calls != pair_count:
        raise AssertionError("scorer call count differs from pair-score count")
    if token_count == 300 and pair_count != EXPECTED_LARGEST_PAIR_COUNT:
        raise AssertionError("300-atom pair-score count changed")

    records = []
    previous_score = None
    previous_size = None
    maximum_witness_error = 0.0
    direct_p0_score = score_partition(
        document, system_id, document.original_boundaries
    )
    for radius in RADII:
        problem = replace(base_problem, max_boundary_moves=radius)
        started = time.perf_counter()
        result = exact_pairwise_certificate(problem)
        elapsed = time.perf_counter() - started

        witness_score = score_partition(
            document, system_id, result.worst_boundaries
        )
        witness_error = abs(witness_score - result.certified_score)
        maximum_witness_error = max(maximum_witness_error, witness_error)
        if witness_error > 1e-12:
            raise AssertionError("rendered witness score differs")
        if abs(result.original_score - direct_p0_score) > 1e-12:
            raise AssertionError("direct P0 and original scores differ")
        if radius == 0:
            if result.threat_set_size != 1:
                raise AssertionError("k=0 threat set is not a singleton")
            if result.certified_score != result.original_score:
                raise AssertionError("k=0 score differs from original score")
        if (
            previous_score is not None
            and result.certified_score > previous_score + 1e-12
        ):
            raise AssertionError("certified score increased with radius")
        if (
            previous_size is not None
            and result.threat_set_size < previous_size
        ):
            raise AssertionError("threat set shrank with radius")
        previous_score = result.certified_score
        previous_size = result.threat_set_size

        record = stable_result(result)
        record["elapsed_seconds"] = elapsed
        record["performance_gate_seconds"] = (
            DP_LIMIT_SECONDS if token_count == 300 else None
        )
        record["performance_gate_passed"] = (
            elapsed < DP_LIMIT_SECONDS if token_count == 300 else None
        )
        records.append(record)

    if (
        token_count == 300
        and records[-1]["threat_set_size"] != EXPECTED_LARGEST_THREAT_SET
    ):
        raise AssertionError("300-atom k=3 threat-set size changed")

    return {
        "token_count": token_count,
        "unit_count": document.unit_count,
        "system_id": system_id,
        "canonical_tape_sha256": document.canonical_tape_sha256,
        "pair_score_count": pair_count,
        "scorer_call_count": scorer_calls,
        "pair_score_sha256": pair_score_digest(base_problem.pair_scores),
        "pair_score_precompute_seconds": precompute_seconds,
        "pair_precompute_gate_seconds": (
            PAIR_PRECOMPUTE_LIMIT_SECONDS if token_count == 300 else None
        ),
        "pair_precompute_gate_passed": (
            precompute_seconds < PAIR_PRECOMPUTE_LIMIT_SECONDS
            if token_count == 300
            else None
        ),
        "direct_p0_score": direct_p0_score,
        "maximum_witness_error": maximum_witness_error,
        "radii": records,
    }


def stable_configuration(record):
    return {
        "token_count": record["token_count"],
        "unit_count": record["unit_count"],
        "system_id": record["system_id"],
        "canonical_tape_sha256": record["canonical_tape_sha256"],
        "pair_score_count": record["pair_score_count"],
        "scorer_call_count": record["scorer_call_count"],
        "pair_score_sha256": record["pair_score_sha256"],
        "direct_p0_score": record["direct_p0_score"],
        "maximum_witness_error": record["maximum_witness_error"],
        "radii": [
            {
                key: value
                for key, value in radius.items()
                if key not in {
                    "elapsed_seconds",
                    "performance_gate_seconds",
                    "performance_gate_passed",
                }
            }
            for radius in record["radii"]
        ],
    }


def build_audit():
    started = time.perf_counter()
    records = [
        run_configuration(token_count, system_id)
        for token_count in TOKEN_COUNTS
        for system_id in SYSTEM_IDS
    ]
    repeat_records = [
        run_configuration(300, system_id) for system_id in SYSTEM_IDS
    ]
    primary_300 = {
        record["system_id"]: record
        for record in records
        if record["token_count"] == 300
    }
    for repeated in repeat_records:
        original = primary_300[repeated["system_id"]]
        if stable_configuration(original) != stable_configuration(repeated):
            raise AssertionError("300-atom repeat changed deterministic outputs")

    for token_count in TOKEN_COUNTS:
        by_system = {
            record["system_id"]: record
            for record in records
            if record["token_count"] == token_count
        }
        semstamp = by_system["semstamp"]
        simmark = by_system["simmark"]
        if semstamp["pair_score_count"] != simmark["pair_score_count"]:
            raise AssertionError("system pair-score lattices differ")
        for sem_radius, sim_radius in zip(
            semstamp["radii"], simmark["radii"]
        ):
            if (
                sem_radius["threat_set_size"]
                != sim_radius["threat_set_size"]
                or sem_radius["exact_move_histogram"]
                != sim_radius["exact_move_histogram"]
            ):
                raise AssertionError("system threat-set combinatorics differ")
        if semstamp["pair_score_sha256"] == simmark["pair_score_sha256"]:
            raise AssertionError("SimMark prefix did not change pair scores")

    maximum_300_pair_seconds = max(
        record["pair_score_precompute_seconds"]
        for record in records + repeat_records
        if record["token_count"] == 300
    )
    maximum_300_dp_seconds = max(
        radius["elapsed_seconds"]
        for record in records + repeat_records
        if record["token_count"] == 300
        for radius in record["radii"]
    )
    if maximum_300_pair_seconds >= PAIR_PRECOMPUTE_LIMIT_SECONDS:
        raise AssertionError("300-atom pair precompute exceeded its gate")
    if maximum_300_dp_seconds >= DP_LIMIT_SECONDS:
        raise AssertionError("300-atom DP exceeded its gate")

    source_paths = (
        ROOT / "scripts" / "certified_resegmentation.py",
        ROOT / "scripts" / "certified_resegmentation_wp2_adapter.py",
        ROOT / "scripts" / "audit_certified_resegmentation_wp2_adapter_scale.py",
    )
    return {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "offline_model_free_wp2a_adapter_scale",
        "network_access_by_script": False,
        "frozen_configuration": {
            "token_counts": list(TOKEN_COUNTS),
            "systems": list(SYSTEM_IDS),
            "unit_atoms": UNIT_ATOMS,
            "min_unit_atoms": MIN_UNIT_ATOMS,
            "max_unit_atoms": MAX_UNIT_ATOMS,
            "radii": list(RADII),
            "pair_precompute_limit_seconds": (
                PAIR_PRECOMPUTE_LIMIT_SECONDS
            ),
            "dp_limit_seconds": DP_LIMIT_SECONDS,
        },
        "records": records,
        "determinism_repeat_300": {
            "systems_repeated": list(SYSTEM_IDS),
            "stable_output_match": True,
            "repeat_timings": [
                {
                    "system_id": record["system_id"],
                    "pair_score_precompute_seconds": (
                        record["pair_score_precompute_seconds"]
                    ),
                    "dp_seconds": [
                        radius["elapsed_seconds"]
                        for radius in record["radii"]
                    ],
                }
                for record in repeat_records
            ],
        },
        "summary": {
            "configuration_count": len(records),
            "certificate_runs": len(records) * len(RADII),
            "direct_p0_checks": len(records),
            "rendered_witness_checks": len(records) * len(RADII),
            "largest_pair_score_count": max(
                record["pair_score_count"] for record in records
            ),
            "largest_threat_set": max(
                radius["threat_set_size"]
                for record in records
                for radius in record["radii"]
            ),
            "maximum_witness_error": max(
                record["maximum_witness_error"] for record in records
            ),
            "maximum_300_pair_precompute_seconds": (
                maximum_300_pair_seconds
            ),
            "maximum_300_dp_seconds": maximum_300_dp_seconds,
            "pair_precompute_gate_passed": True,
            "dp_gate_passed": True,
            "determinism_gate_passed": True,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "pass",
        },
        "decisions": {
            "model_free_adapter_scale_gate": "pass",
            "official_detector_resource_gate": "not_tested",
            "wp2_current_host": "no_go",
            "wp2_authorized": False,
            "overall_certified_resegmentation_project": "conditional_go",
        },
        "local_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in source_paths
        },
        "execution_attestation": {
            "provider_api_calls": 0,
            "watermark_generation_calls": 0,
            "watermark_detector_calls": 0,
            "llm_judge_calls": 0,
            "amr_parser_calls": 0,
            "s2match_calls": 0,
            "model_payload_downloads": 0,
            "dataset_payload_downloads": 0,
            "external_gpu_jobs": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit()
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
