"""Run the zero-download WP2a adapter and cohort preflight."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from certified_resegmentation import (
    brute_force_pairwise_certificate,
    exact_pairwise_certificate,
)
from certified_resegmentation_wp2_adapter import (
    build_document_pair_scores,
    build_document_pairwise_problem,
    canonicalize,
    cohort_sort_hash,
    hash_raw_value,
    make_source_record,
    split_null_cohort,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "certified_resegmentation_wp2_adapter_preflight1"
    / "audit_results.json"
)


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


def score_rendered_partition(document, system_id, boundaries):
    units = document.preprocess_partition(system_id, boundaries)
    return sum(
        text_pair_score(units[index - 1], units[index])
        for index in range(1, len(units))
    ) / (len(units) - 1)


def canonical_contract_checks():
    calls = []
    raw = "Alpha\tbeta. Gamma \u4e16\u754c!"

    def splitter(value):
        calls.append(value)
        return ["Alpha\tbeta.", "Gamma \u4e16\u754c!"]

    string_document = canonicalize(raw, splitter)
    list_document = canonicalize(
        ["Alpha\tbeta.", "Gamma \u4e16\u754c!", "Last unit."]
    )
    if calls != [raw]:
        raise AssertionError("string splitter was not called exactly once")
    if (
        string_document.render_partition(
            string_document.original_boundaries
        )
        != string_document.original_units
    ):
        raise AssertionError("string P0 did not round-trip")
    if (
        list_document.render_partition(list_document.original_boundaries)
        != list_document.original_units
    ):
        raise AssertionError("list P0 did not round-trip")
    candidate = list_document.render_partition((1, 3))
    if candidate != ("Alpha", "beta. Gamma", "\u4e16\u754c! Last unit."):
        raise AssertionError("candidate rendering changed a tape substring")
    if list_document.preprocess_partition("simmark") != (
        "Alpha\tbeta.",
        " Gamma \u4e16\u754c!",
        " Last unit.",
    ):
        raise AssertionError("SimMark leading-space preprocessing differs")
    if list_document.preprocess_partition("semstamp") != (
        "Alpha\tbeta.",
        "Gamma \u4e16\u754c!",
        "Last unit.",
    ):
        raise AssertionError("SemStamp preprocessing differs")
    if hash_raw_value(["x"]) == hash_raw_value('["x"]'):
        raise AssertionError("string and list raw hash domains collided")
    return {
        "fixtures_checked": 2,
        "string_splitter_call_count": len(calls),
        "p0_exact_round_trips": 2,
        "arbitrary_partition_checks": 1,
        "unicode_and_internal_whitespace_checked": True,
        "simmark_nonfirst_ascii_space_checked": True,
        "semstamp_no_prefix_checked": True,
        "raw_string_list_type_domain_checked": True,
        "status": "pass",
    }


def cohort_contract_checks():
    def record(row_index, text):
        return make_source_record(
            "semstamp",
            "synthetic-data-revision",
            row_index,
            text,
            source_repository="offline/synthetic",
            relative_path="rows.jsonl",
        )

    unique_null = [record(index, f"null-{index}") for index in range(103)]
    duplicates = [
        record(1000 + index, f"null-{index}") for index in range(7)
    ]
    positive = [
        record(2000, "positive-0"),
        record(2001, "positive-1"),
        record(2002, "positive-2"),
        record(2003, "positive-3"),
        record(2004, "null-3"),
        record(2005, "null-77"),
    ]
    null_input = unique_null + duplicates
    forward = split_null_cohort(
        "semstamp",
        "synthetic-data-revision",
        null_input,
        positive,
    )

    rng = random.Random(20260922)
    shuffled_null = list(null_input)
    shuffled_positive = list(positive)
    rng.shuffle(shuffled_null)
    rng.shuffle(shuffled_positive)
    shuffled = split_null_cohort(
        "semstamp",
        "synthetic-data-revision",
        shuffled_null,
        shuffled_positive,
    )
    if forward != shuffled:
        raise AssertionError("cohort split changed after input permutation")
    if tuple(map(len, (
        forward.detector_fit,
        forward.certificate_calibration,
        forward.independent_test,
    ))) != (20, 41, 42):
        raise AssertionError("20/40/remainder cohort sizes differ")
    combined = (
        forward.detector_fit
        + forward.certificate_calibration
        + forward.independent_test
    )
    if tuple(cohort_sort_hash(item) for item in combined) != tuple(sorted(
        cohort_sort_hash(item) for item in combined
    )):
        raise AssertionError("null cohort is not ordered by stable H")
    if forward.attrition.null_duplicate_count != 7:
        raise AssertionError("null duplicate attrition differs")
    if forward.attrition.positive_overlap_count != 2:
        raise AssertionError("positive/null overlap attrition differs")
    return {
        "seed": 20260922,
        "null_input_count": len(null_input),
        "null_unique_count": forward.attrition.null_unique_count,
        "null_duplicate_count": forward.attrition.null_duplicate_count,
        "detector_fit_count": len(forward.detector_fit),
        "certificate_calibration_count": len(
            forward.certificate_calibration
        ),
        "independent_test_count": len(forward.independent_test),
        "positive_input_count": len(positive),
        "positive_retained_count": (
            forward.attrition.positive_retained_count
        ),
        "positive_overlap_excluded_count": (
            forward.attrition.positive_overlap_count
        ),
        "input_permutation_invariance": True,
        "stable_hash_order_checked": True,
        "status": "pass",
    }


def pairwise_integration_checks(case_count=500):
    rng = random.Random(22092026)
    comparisons = 0
    p0_checks = 0
    threat_members = 0
    system_counts = {"semstamp": 0, "simmark": 0}
    maximum_score_error = 0.0
    maximum_witness_error = 0.0
    started = time.perf_counter()

    for case_index in range(case_count):
        system_id = "semstamp" if case_index % 2 == 0 else "simmark"
        unit_count = rng.randint(2, 4)
        lengths = [rng.randint(1, 3) for _ in range(unit_count)]
        tokens = [
            f"case{case_index}_atom{atom_index}"
            for atom_index in range(sum(lengths))
        ]
        units = []
        cursor = 0
        for length in lengths:
            units.append(" ".join(tokens[cursor:cursor + length]))
            cursor += length
        document = canonicalize(units)
        radius = rng.randint(0, unit_count - 1)
        problem = build_document_pairwise_problem(
            document,
            system_id=system_id,
            min_unit_atoms=1,
            max_unit_atoms=4,
            max_boundary_moves=radius,
            scorer=text_pair_score,
            threshold=(rng.randint(-8, 8) / 4),
        )
        dynamic = exact_pairwise_certificate(problem)
        oracle = brute_force_pairwise_certificate(problem)
        score_error = abs(
            dynamic.certified_score - oracle.certified_score
        )
        maximum_score_error = max(maximum_score_error, score_error)
        if score_error > 1e-12:
            raise AssertionError("adapter DP and oracle scores differ")
        if dynamic.exact_move_histogram != oracle.exact_move_histogram:
            raise AssertionError("adapter DP and oracle histograms differ")
        if dynamic.threat_set_size != oracle.threat_set_size:
            raise AssertionError("adapter DP and oracle threat sizes differ")
        witness_score = score_rendered_partition(
            document, system_id, dynamic.worst_boundaries
        )
        witness_error = abs(witness_score - dynamic.certified_score)
        maximum_witness_error = max(maximum_witness_error, witness_error)
        if witness_error > 1e-12:
            raise AssertionError("rendered witness did not reproduce score")

        zero = exact_pairwise_certificate(
            replace(problem, max_boundary_moves=0)
        )
        direct_p0_score = score_rendered_partition(
            document, system_id, document.original_boundaries
        )
        if abs(zero.original_score - direct_p0_score) > 1e-12:
            raise AssertionError("k=0 adapter score differs from direct P0")
        if zero.certified_score != zero.original_score:
            raise AssertionError("k=0 threat set was not a singleton")

        comparisons += 1
        p0_checks += 1
        threat_members += dynamic.threat_set_size
        system_counts[system_id] += 1

    return {
        "seed": 22092026,
        "case_count": comparisons,
        "system_case_counts": system_counts,
        "dp_oracle_comparisons": comparisons,
        "oracle_threat_members_checked": threat_members,
        "direct_p0_k0_score_checks": p0_checks,
        "maximum_dp_oracle_score_error": maximum_score_error,
        "maximum_rendered_witness_error": maximum_witness_error,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
    }


def fail_closed_checks():
    document = canonicalize(["aa bb", "cc dd"])
    cases = []

    def expect(error_type, action):
        try:
            action()
        except error_type:
            cases.append("pass")
            return
        raise AssertionError(f"expected {error_type.__name__}")

    expect(ValueError, lambda: canonicalize("raw"))
    expect(TypeError, lambda: canonicalize("raw", lambda _: ("raw",)))
    expect(ValueError, lambda: canonicalize([" aa", "bb"]))
    expect(ValueError, lambda: document.render_partition((0,)))
    expect(ValueError, lambda: document.preprocess_partition("unknown"))
    expect(
        ValueError,
        lambda: build_document_pair_scores(
            document, "semstamp", 1, 2, lambda *_: float("nan")
        ),
    )
    expect(
        ValueError,
        lambda: split_null_cohort(
            "semstamp",
            "revision",
            [],
            [],
            detector_fit_fraction=0.7,
            calibration_fraction=0.4,
        ),
    )
    mismatched = make_source_record("simmark", "revision", 0, "raw")
    expect(
        ValueError,
        lambda: split_null_cohort(
            "semstamp", "revision", [mismatched], []
        ),
    )
    return {
        "expected_failures_checked": len(cases),
        "all_failed_closed": True,
        "status": "pass",
    }


def build_audit():
    canonical = canonical_contract_checks()
    cohort = cohort_contract_checks()
    integration = pairwise_integration_checks()
    fail_closed = fail_closed_checks()
    source_paths = (
        ROOT / "scripts" / "certified_resegmentation.py",
        ROOT / "scripts" / "certified_resegmentation_wp2_adapter.py",
        ROOT / "scripts" / "test_certified_resegmentation_wp2_adapter.py",
        ROOT / "scripts" / "audit_certified_resegmentation_wp2_adapter.py",
    )
    return {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "offline_model_free_wp2a_preflight",
        "network_access_by_script": False,
        "scope": {
            "systems": ["semstamp", "simmark"],
            "input": "synthetic strings and unit lists only",
            "atomization": "maximal non-whitespace tape spans",
            "candidate_transport": "direct unit lists only",
            "model_or_dataset_payload_download": "not_authorized",
            "watermark_detector_execution": "not_authorized",
        },
        "canonical_adapter": canonical,
        "cohort_split": cohort,
        "synthetic_pairwise_integration": integration,
        "fail_closed_contract": fail_closed,
        "gates": {
            "canonical_unit_adapter_implemented": True,
            "synthetic_k0_direct_list_equivalence": True,
            "stable_null_cohort_split_implemented": True,
            "synthetic_pairwise_dp_integration": True,
            "official_detector_k0_equivalence": False,
            "real_arrow_schema_row_overlap_audit": False,
            "real_cohorts_materialized": False,
            "wp2_execution_authorized": False,
        },
        "decisions": {
            "wp2a_model_free_adapter_preflight": "pass",
            "wp2_current_host": "no_go",
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
    encoded = json.dumps(
        audit, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
