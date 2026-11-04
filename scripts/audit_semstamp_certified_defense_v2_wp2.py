"""Independently audit v2 WP2 replays without importing solver code."""

import argparse
import ast
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path


EXPECTED_V1_CORE_SHA256 = (
    "cacde2fb0d238d6c1ae7469d49dccc0a62453dcb4c03fe4b5a81a46a269be19e"
)
EXPECTED_PARAMETERS = {
    "seed": 20261008,
    "systematic_cases": 5000,
    "random_cases": 2000,
    "fixed_n_reduction_cases": 1000,
    "maximum_synthetic_atom_count": 10,
    "oracle_combination_limit": 1000000,
    "independent_disclosed_sample": 64,
}
EXPECTED_FAILURE_CODES = {
    "gap_count": "REFINED_GAP_COUNT_MISMATCH",
    "negative_gap": "INVALID_GAP_BYTE_LENGTH",
    "boolean_gap": "INVALID_GAP_BYTE_LENGTH",
    "unauthorized_zero_gap": "UNAUTHORIZED_ZERO_BYTE_GAP",
    "nonwhitespace_gap": "NONWHITESPACE_REFINED_GAP",
    "bridge_unauthorized_zero_gap": "UNAUTHORIZED_ZERO_BYTE_GAP",
}
EXPECTED_ATTESTATION_KEYS = (
    "model_loads",
    "gpu_calls",
    "remote_calls",
    "network_downloads",
    "sealed_examples_read",
    "detector_score_reads",
    "generated_text_examples",
)
FORBIDDEN_IMPORT_ROOTS = {
    "httpx",
    "requests",
    "sentence_transformers",
    "socket",
    "torch",
    "transformers",
    "urllib",
}


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_value(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def imported_roots(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def exact_score_dict(hits, edge_count):
    numerator = 4 * hits - edge_count
    return {
        "hits": hits,
        "edge_count": edge_count,
        "numerator": numerator,
        "value": numerator / math.sqrt(3 * edge_count),
    }


def compare_scores(left, right):
    left_numerator = 4 * left[0] - left[1]
    right_numerator = 4 * right[0] - right[1]
    left_sign = (left_numerator > 0) - (left_numerator < 0)
    right_sign = (right_numerator > 0) - (right_numerator < 0)
    if left_sign != right_sign:
        return -1 if left_sign < right_sign else 1
    if left_sign == 0:
        return 0
    left_product = left_numerator * left_numerator * right[1]
    right_product = right_numerator * right_numerator * left[1]
    if left_product == right_product:
        return 0
    if left_sign > 0:
        return -1 if left_product < right_product else 1
    return -1 if left_product > right_product else 1


def candidate_is_better(candidate, incumbent):
    if incumbent is None:
        return True
    comparison = compare_scores(
        candidate["score"], incumbent["score"]
    )
    if comparison != 0:
        return comparison < 0
    return (
        candidate["cost"],
        candidate["insertions"],
        candidate["deletions"],
        candidate["score"][1],
        candidate["boundaries"],
    ) < (
        incumbent["cost"],
        incumbent["insertions"],
        incumbent["deletions"],
        incumbent["score"][1],
        incumbent["boundaries"],
    )


def score_boundaries(boundaries, atom_count, hit_triples):
    endpoints = (0, *boundaries, atom_count)
    return sum(
        (endpoints[index - 1], endpoints[index], endpoints[index + 1])
        in hit_triples
        for index in range(1, len(endpoints) - 1)
    )


def independently_solve_descriptor(descriptor):
    atom_count = descriptor["atom_count"]
    original = tuple(descriptor["original_boundaries"])
    gap_lengths = tuple(descriptor["gap_byte_lengths"])
    radius = descriptor["radius"]
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 2
    ):
        raise AssertionError("invalid disclosed atom count")
    if len(gap_lengths) != atom_count - 1:
        raise AssertionError("disclosed gap count mismatch")
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or length < 0
        for length in gap_lengths
    ):
        raise AssertionError("invalid disclosed gap byte length")
    if tuple(sorted(set(original))) != original or not original:
        raise AssertionError("invalid disclosed B0")
    zero_boundaries = tuple(
        boundary
        for boundary, length in enumerate(gap_lengths, start=1)
        if length == 0
    )
    if not set(zero_boundaries).issubset(original):
        raise AssertionError("disclosed unauthorized zero gap")

    allowed_spans = {
        tuple(span) for span in descriptor["allowed_spans"]
    }
    hit_triples = {
        tuple(triple) for triple in descriptor["hit_triples"]
    }
    original_set = frozenset(original)
    original_count = len(original)
    lower = max(1, original_count - radius)
    upper = min(atom_count - 1, original_count + radius)
    histogram_by_id = {}
    histogram_by_edges = {}
    histogram_by_cost = {}
    candidates = []

    for edge_count in range(lower, upper + 1):
        for boundaries in combinations(
            range(1, atom_count), edge_count
        ):
            candidate_set = frozenset(boundaries)
            insertions = len(candidate_set - original_set)
            deletions = len(original_set - candidate_set)
            cost = max(insertions, deletions)
            if cost > radius:
                continue
            endpoints = (0, *boundaries, atom_count)
            if any(
                (endpoints[index], endpoints[index + 1])
                not in allowed_spans
                for index in range(len(endpoints) - 1)
            ):
                continue
            hits = score_boundaries(
                boundaries, atom_count, hit_triples
            )
            candidate = {
                "boundaries": boundaries,
                "insertions": insertions,
                "deletions": deletions,
                "cost": cost,
                "score": (hits, edge_count),
            }
            candidates.append(candidate)
            histogram_by_id[(insertions, deletions)] = (
                histogram_by_id.get((insertions, deletions), 0) + 1
            )
            histogram_by_edges[edge_count] = (
                histogram_by_edges.get(edge_count, 0) + 1
            )
            histogram_by_cost[cost] = (
                histogram_by_cost.get(cost, 0) + 1
            )
    if not candidates:
        raise AssertionError("disclosed threat set is empty")

    best = None
    for candidate in candidates:
        if candidate_is_better(candidate, best):
            best = candidate
    if best is None:
        raise AssertionError("independent minimizer is missing")

    original_hits = score_boundaries(
        original, atom_count, hit_triples
    )
    exact_score = exact_score_dict(*best["score"])
    original_score = exact_score_dict(original_hits, original_count)
    threshold = descriptor["threshold"]
    decision = None
    if threshold is not None:
        threshold_pair = (
            threshold["hits"], threshold["edge_count"]
        )
        decision = (
            "CERTIFIED_WATERMARKED"
            if compare_scores(best["score"], threshold_pair) > 0
            else "NOT_CERTIFIED"
        )
    gap_digest = sha256_value({
        "atom_count": atom_count,
        "gap_byte_lengths": list(gap_lengths),
        "zero_byte_gap_boundaries": list(zero_boundaries),
    })
    return {
        "token_count": atom_count,
        "original_boundaries": list(original),
        "requested_radius": radius,
        "worst_boundaries": list(best["boundaries"]),
        "exact_score": exact_score,
        "certified_score": exact_score["value"],
        "original_exact_score": original_score,
        "original_score": original_score["value"],
        "insertions": best["insertions"],
        "deletions": best["deletions"],
        "operation_cost": best["cost"],
        "threshold": threshold,
        "decision": decision,
        "threat_set_size": len(candidates),
        "histogram_by_insertions_deletions": [
            {
                "insertions": insertions,
                "deletions": deletions,
                "count": count,
            }
            for (insertions, deletions), count
            in sorted(histogram_by_id.items())
        ],
        "histogram_by_edge_count": {
            str(edge_count): count
            for edge_count, count in sorted(histogram_by_edges.items())
        },
        "histogram_by_operation_cost": {
            str(cost): count
            for cost, count in sorted(histogram_by_cost.items())
        },
        "refined_topology": {
            "gap_byte_lengths": list(gap_lengths),
            "zero_byte_gap_boundaries": list(zero_boundaries),
            "gap_metadata_sha256": gap_digest,
        },
    }


def validate_replay(replay, expected_replay_id):
    if replay.get("schema_version") != 1:
        raise AssertionError("unexpected replay schema")
    if replay.get("project_id") != "semstamp_certified_defense_v2":
        raise AssertionError("unexpected replay project")
    if replay.get("stage") != "WP2" or replay.get("status") != "pass":
        raise AssertionError("replay did not pass WP2")
    if replay.get("replay_id") != expected_replay_id:
        raise AssertionError("replay id mismatch")
    if replay.get("parameters") != EXPECTED_PARAMETERS:
        raise AssertionError("frozen replay parameters changed")
    if replay.get("wp3_authorized") is not False:
        raise AssertionError("replay improperly authorized WP3")
    attestation = replay.get("zero_model_attestation", {})
    if any(attestation.get(key) != 0 for key in EXPECTED_ATTESTATION_KEYS):
        raise AssertionError("zero-model attestation failed")

    semantic = replay.get("semantic")
    if replay.get("semantic_sha256") != sha256_value(semantic):
        raise AssertionError("replay semantic digest mismatch")
    correctness = semantic["correctness"]
    fixed = semantic["fixed_n_reduction"]
    specialized = semantic["specialized_gates"]
    sample = semantic["independent_audit_sample"]
    if (
        correctness["systematic_cases"] != 5000
        or correctness["random_cases"] != 2000
        or correctness["total_cases"] != 7000
        or correctness["zero_gap_cases"] <= 0
    ):
        raise AssertionError("correctness counts are incomplete")
    if fixed["cases"] != 1000 or fixed["zero_gap_cases"] <= 0:
        raise AssertionError("fixed-N counts are incomplete")
    required_true = (
        "score_match",
        "exact_score_match",
        "decision_match",
        "witness_match",
        "threat_set_size_match",
        "all_histograms_match",
        "gap_metadata_match",
        "witness_verification",
        "score_blind_topology",
    )
    if any(correctness.get(key) is not True for key in required_true):
        raise AssertionError("correctness gate is false")
    fixed_true = (
        "minimum_hits_match",
        "original_hits_match",
        "witness_match",
        "threat_set_size_match",
        "move_histogram_match",
        "strict_decision_match",
    )
    if any(fixed.get(key) is not True for key in fixed_true):
        raise AssertionError("fixed-N gate is false")
    specialized_true = (
        "cross_m_exact_tie",
        "strict_threshold_tie",
        "radius_zero_equals_clean",
        "radius_monotonicity",
        "operation_classes",
        "two_unit_case",
        "illegal_span_filtering",
        "single_problem_replay",
        "zero_gap_identity",
        "zero_gap_deletion",
        "zero_gap_relocation",
        "metadata_validation_before_callbacks",
        "renderer_to_solver_bridge",
    )
    if any(specialized.get(key) is not True for key in specialized_true):
        raise AssertionError("specialized gate is false")
    if specialized.get("failure_codes") != EXPECTED_FAILURE_CODES:
        raise AssertionError("fail-closed code set changed")
    if sample.get("case_count") != 64:
        raise AssertionError("independent sample count changed")
    if len(sample.get("cases", ())) != 64:
        raise AssertionError("independent sample rows are incomplete")
    if sample.get("cases_sha256") != sha256_value(sample["cases"]):
        raise AssertionError("independent sample digest mismatch")


def audit_disclosed_samples(cases):
    zero_gap_cases = 0
    total_threat_members = 0
    digest = hashlib.sha256()
    for descriptor in cases:
        independently_computed = independently_solve_descriptor(
            descriptor
        )
        if independently_computed != descriptor["expected"]:
            raise AssertionError(
                f"independent enumeration differs for "
                f"{descriptor['case_id']}"
            )
        zero_gap_cases += int(
            bool(
                independently_computed["refined_topology"][
                    "zero_byte_gap_boundaries"
                ]
            )
        )
        total_threat_members += independently_computed["threat_set_size"]
        digest.update(
            canonical_bytes(independently_computed) + b"\n"
        )
    if zero_gap_cases == 0:
        raise AssertionError("independent sample has no zero-gap cases")
    return {
        "cases": len(cases),
        "zero_gap_cases": zero_gap_cases,
        "total_threat_members": total_threat_members,
        "independent_semantic_sha256": digest.hexdigest(),
        "all_exact_scores_match": True,
        "all_decisions_match": True,
        "all_witnesses_match": True,
        "all_histograms_match": True,
        "all_gap_metadata_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay0", type=Path, required=True)
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--theorem", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--auditor-test", type=Path, required=True)
    parser.add_argument("--v1-core", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()

    replay0 = load_json(args.replay0)
    replay1 = load_json(args.replay1)
    validate_replay(replay0, 0)
    validate_replay(replay1, 1)
    if replay0["semantic"] != replay1["semantic"]:
        raise AssertionError("full semantic replays differ")
    if replay0["semantic_sha256"] != replay1["semantic_sha256"]:
        raise AssertionError("full semantic replay digests differ")
    if sha256_file(args.v1_core) != EXPECTED_V1_CORE_SHA256:
        raise AssertionError("frozen v1 exact core changed")
    execution_imports = (
        imported_roots(args.implementation)
        | imported_roots(args.runner)
    )
    forbidden_imports = sorted(
        execution_imports & FORBIDDEN_IMPORT_ROOTS
    )
    if forbidden_imports:
        raise AssertionError(
            f"forbidden execution imports: {forbidden_imports}"
        )

    samples = replay0["semantic"][
        "independent_audit_sample"
    ]["cases"]
    independent = audit_disclosed_samples(samples)
    paths = {
        "protocol": args.protocol,
        "theorem_delta": args.theorem,
        "contract": args.contract,
        "implementation": args.implementation,
        "runner": args.runner,
        "test": args.test,
        "auditor": Path(__file__).resolve(),
        "auditor_test": args.auditor_test,
        "v1_core": args.v1_core,
        "renderer": args.renderer,
        "replay0": args.replay0,
        "replay1": args.replay1,
    }
    bindings = {
        name: sha256_file(path) for name, path in paths.items()
    }
    checks = {
        "frozen_parameters_exact": True,
        "v1_core_sha_frozen": True,
        "systematic_5000_complete": True,
        "random_2000_complete": True,
        "fixed_n_1000_complete": True,
        "zero_gap_cases_present": True,
        "zero_gap_specialized_gates_pass": True,
        "metadata_prevalidation_pass": True,
        "renderer_bridge_pass": True,
        "two_semantic_replays_exact": True,
        "independent_64_case_enumeration_pass": True,
        "zero_model_attestation_pass": True,
        "forbidden_execution_imports_absent": True,
        "wp3_not_started": True,
    }
    audit = {
        "schema_version": 1,
        "project_id": "semstamp_certified_defense_v2",
        "stage": "WP2",
        "scope": "independent_refined_topology_combinations_audit",
        "status": "go",
        "bindings": bindings,
        "checks": checks,
        "independent_sample": independent,
        "replay": {
            "semantic_sha256": replay0["semantic_sha256"],
            "byte_identical_semantic_payloads": True,
            "replay0_file_sha256": bindings["replay0"],
            "replay1_file_sha256": bindings["replay1"],
        },
        "execution_attestation": {
            key: 0 for key in EXPECTED_ATTESTATION_KEYS
        },
        "decision": {
            "wp2": "algorithm_go_for_v2_wp3_design",
            "wp3_authorized_to_freeze_design": True,
            "wp3_started": False,
        },
        "terminal_reason": "WP2_ALGORITHM_GO_FOR_V2_WP3_DESIGN",
    }
    atomic_write_json(args.output, audit)
    audit_sha256 = sha256_file(args.output)

    semantic = replay0["semantic"]
    result = {
        "schema_version": 1,
        "project_id": "semstamp_certified_defense_v2",
        "stage": "WP2",
        "audit": "refined_atom_algorithm1_pure_cpu_correctness",
        "created_on": "2026-10-08",
        "status": "pass",
        "contract_sha256": bindings["contract"],
        "protocol_sha256": bindings["protocol"],
        "theorem_delta_sha256": bindings["theorem_delta"],
        "implementation_sha256": bindings["implementation"],
        "v1_core_sha256": bindings["v1_core"],
        "audit_sha256": audit_sha256,
        "parameters": replay0["parameters"],
        "correctness": semantic["correctness"],
        "fixed_n_reduction": semantic["fixed_n_reduction"],
        "specialized_gates": semantic["specialized_gates"],
        "independent_audit": independent,
        "replay": audit["replay"],
        "runtime_seconds_total": (
            replay0["runtime_seconds"] + replay1["runtime_seconds"]
        ),
        "zero_model_attestation": {
            key: 0 for key in EXPECTED_ATTESTATION_KEYS
        },
        "gates": checks,
        "terminal": "WP2_ALGORITHM_GO_FOR_V2_WP3_DESIGN",
        "wp3_authorized_to_freeze_design": True,
        "wp3_started": False,
    }
    atomic_write_json(args.result_output, result)
    print(json.dumps({
        "audit_output": str(args.output),
        "result_output": str(args.result_output),
        "status": audit["status"],
        "semantic_sha256": replay0["semantic_sha256"],
        "independent_cases": independent["cases"],
        "independent_zero_gap_cases": independent["zero_gap_cases"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
