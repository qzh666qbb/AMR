"""Freeze and audit terminal WP2-v13 G1 positive identities."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


PROTOCOL_SHA256 = (
    "e3fa05e532986077840651ae37a1ff1b9a43a9a52706d2337a2435420b274768"
)
CONTRACT_SHA256 = (
    "1114df7d706d6cca2354b4d1cdddbfc6850bbefd1427238004bd251ab34a4b26"
)
G0_SHA256 = (
    "937e0388da8aba10bde503f80e13df831a007f96369a76cf4e2c5ceed4ca7953"
)
V13_SOURCE_SHA256 = (
    "27e9967972acff5118a5bb9c112e0724463644139a8aed321ac0c2e8d965bddf"
)
SEALED_SOURCE_SHA256 = (
    "5b172d040257f53ee843ee4ca50d1a05c38ba7f75b26d9f27f7576192c7e35b7"
)
V12_SOURCE_SHA256 = (
    "ebc1b60759008bc97e0701eef7ba35baa5df35ba8fb1d7c4d8141fcf7ff9cceb"
)
V12_A1_SHA256 = (
    "86cae250106c7de5fc7d7f1025a696737e572d109c350f7e0dab18c270bf8faf"
)
V12_SMOKE_RUNNER_SHA256 = (
    "c266f6d8e14acccdc6360454c384a405c285dfbc52717c771339aef4823864df"
)
GENERATION_RUNNER_SHA256 = (
    "9a2785b302a4a66b2a432eb148b08d2856fa4308693d9117854e5ef8ceedb87b"
)
HISTORICAL_SEMSTAMP_SHA256 = (
    "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218"
)
HISTORICAL_SIMMARK_SHA256 = (
    "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
)
TARGET_ELIGIBLE = 1000
MAX_ATTEMPTS = 1600
FORBIDDEN_PUBLIC_KEYS = {
    "source_text",
    "prompt",
    "generated_text",
    "raw_text",
    "text",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def file_metadata(path):
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def atomic_write_json(path, value):
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_file(path, expected, label):
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} identity mismatch")
    return {
        "relative_path": None,
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def relative_to_root(path, root):
    return str(path.resolve().relative_to(root.resolve()))


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(
                f"raw public key at {location}: {sorted(overlap)!r}"
            )
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def identities(rows, field):
    output = []
    for index, row in enumerate(rows):
        identity = row.get(field)
        if not is_sha256(identity):
            raise ValueError(
                f"invalid {field} at row {index}: {identity!r}"
            )
        output.append(identity)
    return output


def validate_v13_sources(rows):
    if len(rows) != MAX_ATTEMPTS:
        raise ValueError("v13 source count mismatch")
    if [row.get("attempt_index") for row in rows] != list(
        range(MAX_ATTEMPTS)
    ):
        raise ValueError("v13 source attempt order mismatch")
    source_identities = identities(rows, "raw_text_sha256")
    if len(set(source_identities)) != MAX_ATTEMPTS:
        raise ValueError("v13 source identities are not unique")
    return source_identities


def validate_v12_sources(rows):
    if len(rows) != 3500:
        raise ValueError("v12 source count mismatch")
    if [row.get("allocation_index") for row in rows] != list(
        range(3500)
    ):
        raise ValueError("v12 source allocation order mismatch")
    source_identities = identities(rows, "raw_text_sha256")
    if len(set(source_identities)) != 3500:
        raise ValueError("v12 source identities are not unique")
    cohorts = Counter(row.get("cohort") for row in rows)
    expected = {
        "calibration_null": 1000,
        "generation_primary": 1000,
        "generation_reserve": 500,
        "independent_test_null": 1000,
    }
    if dict(cohorts) != expected:
        raise ValueError("v12 source cohort counts mismatch")
    return source_identities


def validate_v12_smoke(result):
    if (
        result.get("status") != "stop"
        or result.get("contains_raw_text") is not False
        or set(result.get("replays", {})) != {"0", "1"}
    ):
        raise ValueError("v12 smoke terminal evidence mismatch")
    replay_zero = result["replays"]["0"]
    replay_one = result["replays"]["1"]
    if len(replay_zero) != 20 or len(replay_one) != 20:
        raise ValueError("v12 smoke replay count mismatch")
    zero_identities = [
        row["output"]["raw_text_sha256"] for row in replay_zero
    ]
    one_identities = [
        row["output"]["raw_text_sha256"] for row in replay_one
    ]
    if zero_identities != one_identities:
        raise ValueError("v12 smoke replay identity mismatch")
    if not all(is_sha256(value) for value in zero_identities):
        raise ValueError("v12 smoke output identity is invalid")
    if len(set(zero_identities)) != 20:
        raise ValueError("v12 smoke output identities are not unique")
    return zero_identities


def validate_execution_attestation(attestation):
    expected_zero = {
        "detector_imports",
        "detector_model_loads",
        "detector_score_reads",
        "detector_scores_computed",
        "resegmentation_calls",
        "certificate_calls",
    }
    if any(attestation.get(key) != 0 for key in expected_zero):
        raise ValueError("G1 score-blind execution attestation failed")
    if attestation.get("raw_text_emitted_to_public_artifacts") is not False:
        raise ValueError("G1 public raw-text attestation failed")


def validate_g1_result(
    result,
    positive_rows,
    positive_metadata,
    positive_relative_path,
    v13_sources,
    expected_sealed_sha256,
    sealed_metadata,
    sealed_relative_path,
):
    reject_raw_public_keys(result, "G1_RESULT")
    reject_raw_public_keys(positive_rows, "POSITIVE_IDENTITIES")
    if (
        result.get("schema_version") != 1
        or result.get("stage")
        != "WP2-v13-G1-SemStamp-generation"
        or result.get("status") != "go"
        or result.get("terminal_reason") != "target_eligible_reached"
        or result.get("contains_raw_text") is not False
        or result.get("scope")
        != "formal_generation_only_no_detector_scores"
        or result.get("next_gate") != "run_H0_identity_freeze"
    ):
        raise ValueError("G1 is not an authorized Go terminal result")
    expected_parents = {
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": G0_SHA256,
        "identity_manifest_sha256": V13_SOURCE_SHA256,
        "sealed_source_sha256": SEALED_SOURCE_SHA256,
        "runner_sha256": GENERATION_RUNNER_SHA256,
        "v12_smoke_runner_sha256": V12_SMOKE_RUNNER_SHA256,
    }
    if any(result.get(key) != value for key, value in expected_parents.items()):
        raise ValueError("G1 parent identity mismatch")
    validate_execution_attestation(result.get("execution_attestation", {}))

    counts = result.get("counts", {})
    attempt_count = counts.get("attempt_count")
    if (
        not isinstance(attempt_count, int)
        or not TARGET_ELIGIBLE <= attempt_count <= MAX_ATTEMPTS
        or counts.get("eligible_count") != TARGET_ELIGIBLE
        or counts.get("target_eligible") != TARGET_ELIGIBLE
        or counts.get("max_attempts") != MAX_ATTEMPTS
        or counts.get("failure_count")
        != attempt_count - TARGET_ELIGIBLE
    ):
        raise ValueError("G1 terminal counts mismatch")

    attempts = result.get("attempts", [])
    if len(attempts) != attempt_count:
        raise ValueError("G1 public attempt count mismatch")
    if [row.get("attempt_index") for row in attempts] != list(
        range(attempt_count)
    ):
        raise ValueError("G1 public attempt order mismatch")

    failure_counts = Counter()
    eligible_attempts = []
    all_output_identities = []
    failed_output_identities = []
    for attempt_index, row in enumerate(attempts):
        source_identity = row.get("source_raw_text_sha256")
        if source_identity != v13_sources[attempt_index]:
            raise ValueError("G1 source order does not match G0")
        elapsed = row.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("G1 attempt runtime is invalid")
        codes = row.get("exclusion_codes")
        if (
            not isinstance(codes, list)
            or codes != sorted(set(codes))
            or not all(isinstance(code, str) and code for code in codes)
        ):
            raise ValueError("G1 exclusion codes are invalid")
        if row.get("eligible") is not (not codes):
            raise ValueError("G1 eligible flag mismatch")
        failure_counts.update(codes)
        output_identity = row.get("output", {}).get("raw_text_sha256")
        if output_identity is not None:
            if not is_sha256(output_identity):
                raise ValueError("G1 output identity is invalid")
            all_output_identities.append(output_identity)
        if codes:
            if output_identity is not None:
                failed_output_identities.append(output_identity)
        else:
            if output_identity is None:
                raise ValueError("eligible G1 output has no identity")
            if (
                row["output"].get("unit_count", 0) < 2
                or not 0 < row["output"].get(
                    "max_native_tokens", 0
                ) <= 512
                or row["output"].get("output_equals_prompt") is not False
            ):
                raise ValueError("eligible G1 output fails P0 structure")
            eligible_attempts.append(row)
    if len(set(all_output_identities)) != len(all_output_identities):
        raise ValueError("G1 outputs are not identity-unique")
    if len(eligible_attempts) != TARGET_ELIGIBLE:
        raise ValueError("G1 eligible attempt count mismatch")
    if dict(sorted(failure_counts.items())) != counts.get("failure_codes"):
        raise ValueError("G1 failure-code summary mismatch")

    manifest = result.get("positive_manifest", {})
    if (
        manifest.get("sha256") != positive_metadata["sha256"]
        or manifest.get("bytes") != positive_metadata["bytes"]
        or manifest.get("rows") != TARGET_ELIGIBLE
        or manifest.get("contains_raw_text") is not False
        or manifest.get("relative_path") != positive_relative_path
        or len(positive_rows) != TARGET_ELIGIBLE
    ):
        raise ValueError("G1 positive manifest metadata mismatch")
    if [row.get("positive_index") for row in positive_rows] != list(
        range(TARGET_ELIGIBLE)
    ):
        raise ValueError("positive index order mismatch")

    positive_identities = identities(
        positive_rows, "output_raw_text_sha256"
    )
    if len(set(positive_identities)) != TARGET_ELIGIBLE:
        raise ValueError("positive output identities are not unique")
    for positive_row, attempt in zip(positive_rows, eligible_attempts):
        expected = {
            "attempt_index": attempt["attempt_index"],
            "source_raw_text_sha256": (
                attempt["source_raw_text_sha256"]
            ),
            "output_raw_text_sha256": (
                attempt["output"]["raw_text_sha256"]
            ),
            "seed": attempt["seed"],
            "unit_count": attempt["output"]["unit_count"],
            "max_native_tokens": (
                attempt["output"]["max_native_tokens"]
            ),
            "contains_raw_text": False,
        }
        observed = {
            key: positive_row.get(key) for key in expected
        }
        if observed != expected:
            raise ValueError("positive row does not match eligible attempt")

    sealed = result.get("sealed_checkpoint", {})
    if (
        not is_sha256(expected_sealed_sha256)
        or sealed.get("sha256") != expected_sealed_sha256
        or sealed_metadata["sha256"] != expected_sealed_sha256
        or sealed.get("bytes") != sealed_metadata["bytes"]
        or sealed.get("relative_path") != sealed_relative_path
        or sealed.get("contains_raw_text") is not True
        or sealed.get("public_copy_allowed") is not False
    ):
        raise ValueError("sealed checkpoint attestation mismatch")

    runtime = result.get("runtime", {})
    peak = runtime.get("max_cuda_memory_allocated_bytes")
    total = runtime.get("total_cuda_memory_bytes")
    if (
        not isinstance(peak, int)
        or not isinstance(total, int)
        or peak < 0
        or total <= 0
        or peak / total >= 0.80
        or not isinstance(runtime.get("process_loads"), list)
        or not runtime["process_loads"]
    ):
        raise ValueError("G1 runtime resource gate failed")
    return {
        "attempt_count": attempt_count,
        "failure_count": attempt_count - TARGET_ELIGIBLE,
        "positive_identities": positive_identities,
        "failed_output_identities": failed_output_identities,
        "peak_vram_fraction": peak / total,
    }


def overlap_summary(left, right):
    overlap = sorted(set(left) & set(right))
    payload = "\n".join(overlap).encode("ascii")
    return {
        "count": len(overlap),
        "identities_sha256": hashlib.sha256(payload).hexdigest(),
    }


def compute_overlaps(
    positive_identities,
    v13_sources,
    v12_rows,
    historical_rows,
    v12_smoke_outputs,
    failed_outputs,
):
    v12_all = [row["raw_text_sha256"] for row in v12_rows]
    calibration = [
        row["raw_text_sha256"]
        for row in v12_rows
        if row["cohort"] == "calibration_null"
    ]
    test = [
        row["raw_text_sha256"]
        for row in v12_rows
        if row["cohort"] == "independent_test_null"
    ]
    historical_all = [
        row["raw_text_sha256"] for row in historical_rows
    ]
    historical_positive = [
        row["raw_text_sha256"]
        for row in historical_rows
        if row.get("role") == "positive"
    ]
    comparison_sets = {
        "v13_attempt_sources": v13_sources,
        "v12_all_sources": v12_all,
        "v12_calibration_null": calibration,
        "v12_independent_test_null": test,
        "historical_all": historical_all,
        "historical_positive": historical_positive,
        "v12_smoke_outputs": v12_smoke_outputs,
        "g1_failed_outputs": failed_outputs,
    }
    return {
        label: overlap_summary(positive_identities, values)
        for label, values in comparison_sets.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument(
        "--v13-source-identities", type=Path, required=True
    )
    parser.add_argument(
        "--v12-source-identities", type=Path, required=True
    )
    parser.add_argument("--v12-a1-result", type=Path, required=True)
    parser.add_argument(
        "--historical-semstamp", type=Path, required=True
    )
    parser.add_argument(
        "--historical-simmark", type=Path, required=True
    )
    parser.add_argument("--generation-runner", type=Path, required=True)
    parser.add_argument("--g1-result", type=Path, required=True)
    parser.add_argument("--positive-identities", type=Path, required=True)
    parser.add_argument("--sealed-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-g1-result-sha256", required=True
    )
    parser.add_argument(
        "--expected-positive-identities-sha256", required=True
    )
    parser.add_argument(
        "--expected-sealed-checkpoint-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    fixed_files = {
        "protocol": (
            args.protocol,
            PROTOCOL_SHA256,
        ),
        "g1_contract": (
            args.contract,
            CONTRACT_SHA256,
        ),
        "g0_readiness": (
            args.g0,
            G0_SHA256,
        ),
        "v13_source_identities": (
            args.v13_source_identities,
            V13_SOURCE_SHA256,
        ),
        "v12_source_identities": (
            args.v12_source_identities,
            V12_SOURCE_SHA256,
        ),
        "v12_a1_result": (
            args.v12_a1_result,
            V12_A1_SHA256,
        ),
        "historical_semstamp": (
            args.historical_semstamp,
            HISTORICAL_SEMSTAMP_SHA256,
        ),
        "historical_simmark": (
            args.historical_simmark,
            HISTORICAL_SIMMARK_SHA256,
        ),
        "generation_runner": (
            args.generation_runner,
            GENERATION_RUNNER_SHA256,
        ),
    }
    inputs = {}
    for label, (path, expected) in fixed_files.items():
        metadata = verify_file(path, expected, label)
        metadata["relative_path"] = relative_to_root(path, root)
        inputs[label] = metadata

    for value, label in (
        (args.expected_g1_result_sha256, "expected G1 result"),
        (
            args.expected_positive_identities_sha256,
            "expected positive identities",
        ),
        (
            args.expected_sealed_checkpoint_sha256,
            "expected sealed checkpoint",
        ),
    ):
        if not is_sha256(value):
            raise ValueError(f"{label} is not a canonical SHA256")
    g1_metadata = verify_file(
        args.g1_result,
        args.expected_g1_result_sha256,
        "terminal G1 result",
    )
    positive_metadata = verify_file(
        args.positive_identities,
        args.expected_positive_identities_sha256,
        "terminal positive identities",
    )
    sealed_metadata = verify_file(
        args.sealed_checkpoint,
        args.expected_sealed_checkpoint_sha256,
        "terminal sealed checkpoint",
    )
    for metadata, path in (
        (g1_metadata, args.g1_result),
        (positive_metadata, args.positive_identities),
        (sealed_metadata, args.sealed_checkpoint),
    ):
        metadata["relative_path"] = relative_to_root(path, root)

    v13_rows = read_jsonl(args.v13_source_identities)
    v13_sources = validate_v13_sources(v13_rows)
    v12_rows = read_jsonl(args.v12_source_identities)
    validate_v12_sources(v12_rows)
    historical_semstamp = read_jsonl(args.historical_semstamp)
    historical_simmark = read_jsonl(args.historical_simmark)
    identities(historical_semstamp, "raw_text_sha256")
    identities(historical_simmark, "raw_text_sha256")
    historical_rows = historical_semstamp + historical_simmark
    v12_smoke_outputs = validate_v12_smoke(
        json.loads(args.v12_a1_result.read_text(encoding="utf-8"))
    )
    g1_result = json.loads(
        args.g1_result.read_text(encoding="utf-8")
    )
    positive_rows = read_jsonl(args.positive_identities)
    g1_audit = validate_g1_result(
        g1_result,
        positive_rows,
        positive_metadata,
        positive_metadata["relative_path"],
        v13_sources,
        args.expected_sealed_checkpoint_sha256,
        sealed_metadata,
        sealed_metadata["relative_path"],
    )
    overlaps = compute_overlaps(
        g1_audit["positive_identities"],
        v13_sources,
        v12_rows,
        historical_rows,
        v12_smoke_outputs,
        g1_audit["failed_output_identities"],
    )
    status = (
        "go"
        if all(item["count"] == 0 for item in overlaps.values())
        else "stop"
    )
    runner_path = Path(__file__).resolve()
    output = {
        "schema_version": 1,
        "stage": "WP2-v13-H0-identity-freeze",
        "status": status,
        "contains_raw_text": False,
        "scope": "identity_only_no_detector_scores",
        "runner_sha256": sha256_file(runner_path),
        "inputs": {
            **inputs,
            "g1_result": g1_metadata,
            "positive_identities": positive_metadata,
            "sealed_checkpoint": {
                **sealed_metadata,
                "contains_raw_text": True,
                "public_copy_allowed": False,
            },
        },
        "counts": {
            "positive": TARGET_ELIGIBLE,
            "positive_unique": len(
                set(g1_audit["positive_identities"])
            ),
            "g1_attempts": g1_audit["attempt_count"],
            "g1_failures": g1_audit["failure_count"],
            "v13_attempt_sources": len(v13_sources),
            "v12_sources": len(v12_rows),
            "v12_calibration_null": sum(
                row["cohort"] == "calibration_null"
                for row in v12_rows
            ),
            "v12_independent_test_null": sum(
                row["cohort"] == "independent_test_null"
                for row in v12_rows
            ),
            "historical_rows": len(historical_rows),
            "v12_smoke_outputs": len(v12_smoke_outputs),
        },
        "overlaps": overlaps,
        "resource_gate": {
            "peak_vram_fraction": g1_audit["peak_vram_fraction"],
            "below_0_80": g1_audit["peak_vram_fraction"] < 0.80,
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "decision": {
            "h0_identity_freeze": status,
            "b0_detector_health": (
                "authorized_to_freeze_protocol"
                if status == "go"
                else "stop"
            ),
            "b1_exact_certificate": "stop",
            "next_action": (
                "freeze a separate B0 detector-health protocol"
                if status == "go"
                else "stop without detector calls"
            ),
        },
    }
    atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
