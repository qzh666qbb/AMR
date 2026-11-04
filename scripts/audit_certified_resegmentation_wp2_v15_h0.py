"""Freeze and audit terminal WP2-v15 G1 positive identities."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


PROTOCOL_SHA256 = (
    "3cbbdb1569577ff0d104c9c9eaeeea54e3fbcae4423707db6b1023c01d095f9b"
)
CONTRACT_SHA256 = (
    "eb680c6b646b1712c043c2c2e3ca3b853a08a5b37f23f5547a643303eb39840f"
)
G0_SHA256 = (
    "937e0388da8aba10bde503f80e13df831a007f96369a76cf4e2c5ceed4ca7953"
)
V15_SOURCE_SHA256 = (
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
    "1d91a61f24d82f483652244a0c526dbda1371ad940d99fec20ce1998d2a2230f"
)
PARENT_GENERATION_RUNNER_SHA256 = (
    "732ba2bb107a67d2e71a39ad7b257ab50f6df51c8a464b1162fc908b4528bb06"
)
BOUNDED_SAMPLER_SHA256 = (
    "f00c6ddbad960b64aa386e06e519ed9c3011a70b8726fabeb9b1643828ef93cb"
)
V13_RUNTIME_STOP_SHA256 = (
    "387eb8c2c1f3226c85f7bd6fb8bbf851d5b36f6d9bf99e0bedd849607abbdfac"
)
V13_PREFIX_SHA256 = (
    "607c7bc63c7271419c063d4d31cfc2b1fbf8b0e91a533c130cf20ec4851cce2e"
)
V14_RUNTIME_STOP_SHA256 = (
    "93bb0533e805c9a09b450bee5a18249a0363a8e49e656e9e3d331b8f09874d38"
)
V14_G1_RESULT_SHA256 = (
    "12f19815fd72c8b2a2b402d4e65928419f621739a7241d1b200fd55ccbd19a4c"
)
V14_POSITIVE_SHA256 = (
    "d482a4ebd0169a726ac91769e489c31f67cabb65855f568d1509df82c7f6f8b4"
)
HISTORICAL_SEMSTAMP_SHA256 = (
    "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218"
)
HISTORICAL_SIMMARK_SHA256 = (
    "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
)
TARGET_ELIGIBLE = 1000
MAX_ATTEMPTS = 1600
PREFIX_ATTEMPTS = 372
PATCH_ID = "semstamp-max-trials-control-flow-v1"
PREFIX_CLASSIFICATIONS = {
    "exact",
    "cumulative_shift_only",
    "patch_explained_divergence",
    "unexplained_divergence",
}
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


def sha256_value(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def validate_v15_sources(rows):
    if len(rows) != MAX_ATTEMPTS:
        raise ValueError("v15 source count mismatch")
    if [row.get("attempt_index") for row in rows] != list(
        range(MAX_ATTEMPTS)
    ):
        raise ValueError("v15 source attempt order mismatch")
    source_identities = identities(rows, "raw_text_sha256")
    if len(set(source_identities)) != MAX_ATTEMPTS:
        raise ValueError("v15 source identities are not unique")
    return source_identities


def validate_v13_prefix(rows):
    if len(rows) != PREFIX_ATTEMPTS:
        raise ValueError("v13 prefix count mismatch")
    eligible_count = 0
    for index, row in enumerate(rows):
        codes = row.get("exclusion_codes")
        eligible = row.get("eligible")
        if (
            row.get("attempt_index") != index
            or not is_sha256(row.get("output_raw_text_sha256"))
            or type(eligible) is not bool
            or not isinstance(codes, list)
            or codes != sorted(set(codes))
            or eligible is not (not codes)
        ):
            raise ValueError("v13 prefix row mismatch")
        eligible_count += int(eligible)
        if row.get("eligible_count") != eligible_count:
            raise ValueError("v13 prefix cumulative count mismatch")
    if eligible_count != 326:
        raise ValueError("v13 prefix eligible total mismatch")
    return rows


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
    for key in (
        "v13_sealed_checkpoint_content_parsed",
        "v14_sealed_checkpoint_content_parsed",
    ):
        if attestation.get(key) is not False:
            raise ValueError("G1 predecessor checkpoint attestation failed")


def validate_sampling_metrics(metrics):
    expected_keys = {
        "patch_id",
        "max_trials",
        "total_trials",
        "accepted_segments",
        "max_trials_reached_segments",
        "fallback_acceptances",
        "max_current_trials_observed",
        "terminated_with_empty_generation",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected_keys:
        raise ValueError("G1 sampling metrics schema mismatch")
    if (
        metrics["patch_id"] != PATCH_ID
        or metrics["max_trials"] != 100
        or type(metrics["terminated_with_empty_generation"]) is not bool
    ):
        raise ValueError("G1 sampling metrics contract mismatch")
    for key in (
        "total_trials",
        "accepted_segments",
        "max_trials_reached_segments",
        "fallback_acceptances",
        "max_current_trials_observed",
    ):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError(f"invalid G1 sampling metric {key}")
    if (
        metrics["max_current_trials_observed"] > 100
        or metrics["max_trials_reached_segments"]
        > metrics["accepted_segments"]
        or metrics["fallback_acceptances"]
        > metrics["max_trials_reached_segments"]
        or metrics["total_trials"]
        > (metrics["accepted_segments"] + 1) * 100
    ):
        raise ValueError("G1 bounded sampling invariant failed")
    return metrics


def validate_prefix_evidence(evidence, row, expected, eligible_count):
    expected_keys = {
        "expected_sha256",
        "actual_sha256",
        "expected_semantic_sha256",
        "actual_semantic_sha256",
        "index_match",
        "output_identity_match",
        "eligibility_match",
        "semantic_match",
        "cumulative_eligible_match",
        "fallback_activated",
        "classification",
        "acceptable",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ValueError("G1 causal-prefix evidence schema mismatch")
    expected_semantic = {
        "output_raw_text_sha256": expected["output_raw_text_sha256"],
        "eligible": expected["eligible"],
        "exclusion_codes": expected["exclusion_codes"],
    }
    actual_semantic = {
        "output_raw_text_sha256": row["output"]["raw_text_sha256"],
        "eligible": row["eligible"],
        "exclusion_codes": row["exclusion_codes"],
    }
    actual = {
        "attempt_index": row["attempt_index"],
        **actual_semantic,
        "eligible_count": eligible_count,
    }
    index_match = actual["attempt_index"] == expected["attempt_index"]
    output_identity_match = (
        actual_semantic["output_raw_text_sha256"]
        == expected_semantic["output_raw_text_sha256"]
    )
    eligibility_match = (
        actual_semantic["eligible"] == expected_semantic["eligible"]
        and actual_semantic["exclusion_codes"]
        == expected_semantic["exclusion_codes"]
    )
    semantic_match = output_identity_match and eligibility_match
    cumulative_match = eligible_count == expected["eligible_count"]
    fallback_activated = row["sampling"]["fallback_acceptances"] > 0
    if not index_match:
        classification = "unexplained_divergence"
    elif semantic_match and cumulative_match:
        classification = "exact"
    elif semantic_match:
        classification = "cumulative_shift_only"
    elif fallback_activated and not output_identity_match:
        classification = "patch_explained_divergence"
    else:
        classification = "unexplained_divergence"
    calculated = {
        "expected_sha256": sha256_value(expected),
        "actual_sha256": sha256_value(actual),
        "expected_semantic_sha256": sha256_value(expected_semantic),
        "actual_semantic_sha256": sha256_value(actual_semantic),
        "index_match": index_match,
        "output_identity_match": output_identity_match,
        "eligibility_match": eligibility_match,
        "semantic_match": semantic_match,
        "cumulative_eligible_match": cumulative_match,
        "fallback_activated": fallback_activated,
        "classification": classification,
        "acceptable": classification != "unexplained_divergence",
    }
    if evidence != calculated:
        raise ValueError("G1 causal-prefix evidence mismatch")
    return classification


def validate_g1_result(
    result,
    positive_rows,
    positive_metadata,
    positive_relative_path,
    v15_sources,
    v13_prefix,
    expected_sealed_sha256,
    sealed_metadata,
    sealed_relative_path,
):
    reject_raw_public_keys(result, "G1_RESULT")
    reject_raw_public_keys(positive_rows, "POSITIVE_IDENTITIES")
    if (
        result.get("schema_version") != 1
        or result.get("stage")
        != "WP2-v15-G1-SemStamp-causal-prefix-generation"
        or result.get("status") != "go"
        or result.get("terminal_reason") != "target_eligible_reached"
        or result.get("contains_raw_text") is not False
        or result.get("scope")
        != "formal_causal_prefix_bounded_generation_only_no_detector_scores"
        or result.get("next_gate")
        != "freeze_and_run_v15_H0_identity_audit"
    ):
        raise ValueError("G1 is not an authorized Go terminal result")
    expected_parents = {
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "g0_sha256": G0_SHA256,
        "identity_manifest_sha256": V15_SOURCE_SHA256,
        "sealed_source_sha256": SEALED_SOURCE_SHA256,
        "runner_sha256": GENERATION_RUNNER_SHA256,
        "parent_runner_sha256": PARENT_GENERATION_RUNNER_SHA256,
        "bounded_sampler_sha256": BOUNDED_SAMPLER_SHA256,
        "v13_runtime_stop_sha256": V13_RUNTIME_STOP_SHA256,
        "v13_prefix_sha256": V13_PREFIX_SHA256,
        "v14_runtime_stop_sha256": V14_RUNTIME_STOP_SHA256,
        "v14_g1_result_sha256": V14_G1_RESULT_SHA256,
        "v14_positive_identities_sha256": V14_POSITIVE_SHA256,
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
    prefix_classifications = Counter()
    eligible_attempts = []
    all_output_identities = []
    failed_output_identities = []
    eligible_count = 0
    for attempt_index, row in enumerate(attempts):
        source_identity = row.get("source_raw_text_sha256")
        if source_identity != v15_sources[attempt_index]:
            raise ValueError("G1 source order does not match G0")
        sampling = validate_sampling_metrics(row.get("sampling"))
        eligible_count += int(row.get("eligible") is True)
        prefix_evidence = row.get("v13_prefix_replay")
        if attempt_index < PREFIX_ATTEMPTS:
            classification = validate_prefix_evidence(
                prefix_evidence,
                row,
                v13_prefix[attempt_index],
                eligible_count,
            )
            prefix_classifications[classification] += 1
            if (
                prefix_evidence["fallback_activated"]
                is not (sampling["fallback_acceptances"] > 0)
            ):
                raise ValueError("G1 fallback evidence mismatch")
        elif prefix_evidence is not None:
            raise ValueError("unexpected G1 causal-prefix evidence")
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
    prefix_summary = result.get("prefix_replay", {})
    expected_prefix_counts = {
        label: prefix_classifications[label]
        for label in sorted(PREFIX_CLASSIFICATIONS)
    }
    if (
        prefix_summary.get("expected") != PREFIX_ATTEMPTS
        or prefix_summary.get("checked") != PREFIX_ATTEMPTS
        or prefix_summary.get("acceptable") != PREFIX_ATTEMPTS
        or prefix_summary.get("classifications")
        != expected_prefix_counts
        or prefix_classifications["unexplained_divergence"] != 0
    ):
        raise ValueError("G1 causal-prefix terminal summary mismatch")
    bounded_summary = result.get("bounded_sampling", {})
    if (
        bounded_summary.get("patch_id") != PATCH_ID
        or bounded_summary.get("maximum_trials_per_segment") != 100
        or bounded_summary.get("total_trials")
        != sum(row["sampling"]["total_trials"] for row in attempts)
        or bounded_summary.get("fallback_attempts")
        != sum(
            row["sampling"]["fallback_acceptances"] > 0
            for row in attempts
        )
        or bounded_summary.get("fallback_acceptances")
        != sum(
            row["sampling"]["fallback_acceptances"] for row in attempts
        )
    ):
        raise ValueError("G1 bounded sampling summary mismatch")

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
        "prefix_classifications": expected_prefix_counts,
        "fallback_attempts": bounded_summary["fallback_attempts"],
        "fallback_acceptances": bounded_summary["fallback_acceptances"],
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
    v15_sources,
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
        "v15_attempt_sources": v15_sources,
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
        "--v15-source-identities", type=Path, required=True
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
    parser.add_argument(
        "--parent-generation-runner", type=Path, required=True
    )
    parser.add_argument("--bounded-sampler", type=Path, required=True)
    parser.add_argument("--v13-runtime-stop", type=Path, required=True)
    parser.add_argument("--v13-prefix", type=Path, required=True)
    parser.add_argument("--v14-runtime-stop", type=Path, required=True)
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
        "v15_source_identities": (
            args.v15_source_identities,
            V15_SOURCE_SHA256,
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
        "parent_generation_runner": (
            args.parent_generation_runner,
            PARENT_GENERATION_RUNNER_SHA256,
        ),
        "bounded_sampler": (
            args.bounded_sampler,
            BOUNDED_SAMPLER_SHA256,
        ),
        "v13_runtime_stop": (
            args.v13_runtime_stop,
            V13_RUNTIME_STOP_SHA256,
        ),
        "v13_prefix": (
            args.v13_prefix,
            V13_PREFIX_SHA256,
        ),
        "v14_runtime_stop": (
            args.v14_runtime_stop,
            V14_RUNTIME_STOP_SHA256,
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

    v15_rows = read_jsonl(args.v15_source_identities)
    v15_sources = validate_v15_sources(v15_rows)
    v13_prefix = validate_v13_prefix(read_jsonl(args.v13_prefix))
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
        v15_sources,
        v13_prefix,
        args.expected_sealed_checkpoint_sha256,
        sealed_metadata,
        sealed_metadata["relative_path"],
    )
    overlaps = compute_overlaps(
        g1_audit["positive_identities"],
        v15_sources,
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
        "stage": "WP2-v15-H0-identity-freeze",
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
            "v15_attempt_sources": len(v15_sources),
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
        "causal_prefix": {
            "expected": PREFIX_ATTEMPTS,
            "classifications": g1_audit["prefix_classifications"],
            "fallback_attempts": g1_audit["fallback_attempts"],
            "fallback_acceptances": g1_audit[
                "fallback_acceptances"
            ],
            "unexplained_divergence": 0,
        },
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
            "v13_sealed_checkpoint_content_parsed": False,
            "v14_sealed_checkpoint_content_parsed": False,
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
