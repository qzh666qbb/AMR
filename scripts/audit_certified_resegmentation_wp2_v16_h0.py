"""Audit WP2-v16 rescue identities and freeze the combined positive set."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import audit_certified_resegmentation_wp2_v15_h0 as common


STAGE = "WP2-v16-H0-rescue-identity-freeze"
PROTOCOL_SHA256 = (
    "0449b363abff408007de8e6024c2d147b5aa586c00ad8b19a8b519e080c41cfe"
)
CONTRACT_SHA256 = (
    "c8344700cf6c420bfb4cbe0fe6fb4870febf70aa772a5c328d5ecfba5dfe3fe9"
)
GENERATION_RUNNER_SHA256 = (
    "080c847c0cf1168ba05e2b4ebe2a12206cce9b09a98d74c20cebeaf5a48ef927"
)
V15_H0_RUNNER_SHA256 = (
    "01ec7d4208dda55550b575db57cc0f521f6afb7f72c026756ea31f0306c7ba3f"
)
V15_G1_RESULT_SHA256 = (
    "c9d8bc4d1c56d39efbc01993e563403a299a66d55ea9de27a522b0ecdf1a2a3f"
)
V15_POSITIVE_SHA256 = (
    "fde40f2ecb8fcfe5653111739ea42bd2a7a8b5ae05da0b66c2b654fc6d5d6d38"
)
V15_H0_STOP_SHA256 = (
    "c832cea4ac029111a3a4239d441cd3618f578690dc1fe8155bf4cc1262c1b681"
)
V15_SEALED_SHA256 = (
    "b60bfdb8d84e358dca049f73a681609e23991c05e1f2327f1038c5d5e11d9404"
)
V15_SEALED_BYTES = 5_329_796
BOUNDED_SAMPLER_SHA256 = common.BOUNDED_SAMPLER_SHA256
START_ATTEMPT = 1111
MAX_ATTEMPTS = 1600
RETAINED_PARENT = 995
TARGET_SUPPLEMENT = 5
TARGET_COMBINED = 1000
PROMPT_ONLY_ATTEMPTS = (8, 561, 823, 834, 912)
SEED_SALT = "WP2-v12-semstamp-generation-seed-v1"
GENERATION_PARAMETERS = {
    "do_sample": True,
    "len_prompt": 32,
    "lmbd": 0.25,
    "margin": 0.02,
    "max_new_tokens": 205,
    "max_trials": 100,
    "min_new_tokens": 195,
    "repetition_penalty": 1.05,
    "sp_dim": 3,
    "temperature": 0.7,
    "top_k": 0,
}
RUNTIME_IDENTITY_KEYS = (
    "python",
    "torch",
    "transformers",
    "sentence-transformers",
    "numpy",
    "nltk",
    "openai",
    "tenacity",
    "NearPy",
    "torch_cuda",
    "device_name",
    "total_cuda_memory_bytes",
)


def atomic_write_jsonl(path, rows):
    encoded = (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
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
    return {
        "sha256": common.sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def validate_zero_attestation(attestation):
    for key in (
        "detector_imports",
        "detector_model_loads",
        "detector_score_reads",
        "detector_scores_computed",
        "resegmentation_calls",
        "certificate_calls",
    ):
        if attestation.get(key) != 0:
            raise ValueError("score-blind execution attestation failed")
    if attestation.get("raw_text_emitted_to_public_artifacts") is not False:
        raise ValueError("public raw-text attestation failed")
    if (
        attestation.get("v15_sealed_checkpoint_content_parsed")
        is not False
    ):
        raise ValueError("v15 sealed checkpoint parsing attestation failed")


def generation_seed(source_identity):
    payload = f"{SEED_SALT}\0{source_identity}".encode("ascii")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    ) % (2 ** 31)


def validate_parent(result, positive_rows, h0_stop):
    common.reject_raw_public_keys(result, "V15_G1_RESULT")
    common.reject_raw_public_keys(
        positive_rows, "V15_POSITIVE_IDENTITIES"
    )
    counts = result.get("counts", {})
    if (
        result.get("stage")
        != "WP2-v15-G1-SemStamp-causal-prefix-generation"
        or result.get("status") != "go"
        or result.get("terminal_reason") != "target_eligible_reached"
        or counts.get("attempt_count") != START_ATTEMPT
        or counts.get("eligible_count") != TARGET_COMBINED
        or len(positive_rows) != TARGET_COMBINED
    ):
        raise ValueError("v15 terminal evidence mismatch")
    attempts = result.get("attempts", [])
    if (
        len(attempts) != START_ATTEMPT
        or [row.get("attempt_index") for row in attempts]
        != list(range(START_ATTEMPT))
    ):
        raise ValueError("v15 attempt sequence mismatch")
    claimed = [row for row in attempts if row.get("eligible") is True]
    if len(claimed) != TARGET_COMBINED:
        raise ValueError("v15 claimed positive count mismatch")
    retained = []
    for positive_index, (attempt, positive) in enumerate(
        zip(claimed, positive_rows)
    ):
        expected = {
            "positive_index": positive_index,
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
        if positive != expected:
            raise ValueError("v15 positive row mismatch")
        if attempt["output"].get("output_equals_prompt") is False:
            if (
                attempt["output"].get("unit_count", 0) < 2
                or not 0 < attempt["output"].get(
                    "max_native_tokens", 0
                ) <= 512
            ):
                raise ValueError("retained v15 P0 metadata mismatch")
            retained.append(positive)
    prompt_only = tuple(
        row["attempt_index"]
        for row in claimed
        if row["output"].get("output_equals_prompt") is True
    )
    violation = h0_stop.get("protocol_violation", {})
    if (
        prompt_only != PROMPT_ONLY_ATTEMPTS
        or h0_stop.get("status") != "stop"
        or h0_stop.get("stop_reason")
        != "prompt_only_positive_contract_violation"
        or tuple(violation.get("attempt_indices", []))
        != PROMPT_ONLY_ATTEMPTS
    ):
        raise ValueError("v15 prompt-only Stop evidence mismatch")
    if len(retained) != RETAINED_PARENT:
        raise ValueError("retained v15 positive count mismatch")
    all_outputs = {
        row["output"]["raw_text_sha256"]
        for row in attempts
        if row.get("output", {}).get("raw_text_sha256") is not None
    }
    failed_outputs = [
        row["output"]["raw_text_sha256"]
        for row in attempts
        if (
            row.get("output", {}).get("raw_text_sha256") is not None
            and row not in claimed
        )
    ]
    return retained, all_outputs, failed_outputs


def validate_supplement(
    result,
    positive_rows,
    positive_metadata,
    positive_relative_path,
    source_identities,
    v15_output_identities,
    expected_sealed_sha256,
    sealed_metadata,
    sealed_relative_path,
):
    common.reject_raw_public_keys(result, "G1R_RESULT")
    common.reject_raw_public_keys(
        positive_rows, "SUPPLEMENT_POSITIVE_IDENTITIES"
    )
    if (
        result.get("schema_version") != 1
        or result.get("stage") != "WP2-v16-G1R-prompt-only-rescue"
        or result.get("status") != "go"
        or result.get("terminal_reason") != "target_supplement_reached"
        or result.get("contains_raw_text") is not False
        or result.get("scope")
        != "fixed_queue_prompt_only_rescue_no_detector_scores"
        or result.get("protocol_sha256") != PROTOCOL_SHA256
        or result.get("contract_sha256") != CONTRACT_SHA256
        or result.get("runner_sha256") != GENERATION_RUNNER_SHA256
        or result.get("bounded_sampler_sha256")
        != BOUNDED_SAMPLER_SHA256
        or result.get("generation_parameters") != GENERATION_PARAMETERS
        or result.get("next_gate")
        != "run_independent_v16_H0_identity_audit"
    ):
        raise ValueError("G1R is not an authorized Go terminal result")
    expected_parent = {
        "protocol_sha256": common.PROTOCOL_SHA256,
        "contract_sha256": common.CONTRACT_SHA256,
        "runner_sha256": common.GENERATION_RUNNER_SHA256,
        "g1_result_sha256": V15_G1_RESULT_SHA256,
        "positive_identities_sha256": V15_POSITIVE_SHA256,
        "h0_stop_sha256": V15_H0_STOP_SHA256,
        "sealed_checkpoint_sha256": V15_SEALED_SHA256,
        "sealed_checkpoint_bytes": V15_SEALED_BYTES,
        "sealed_checkpoint_content_parsed": False,
    }
    if result.get("v15_parent") != expected_parent:
        raise ValueError("G1R parent identity mismatch")
    reclassification = result.get("parent_reclassification", {})
    if (
        reclassification.get("claimed_positive") != 1000
        or reclassification.get("excluded_prompt_only") != 5
        or tuple(reclassification.get("prompt_only_attempts", []))
        != PROMPT_ONLY_ATTEMPTS
        or reclassification.get("retained_positive")
        != RETAINED_PARENT
        or reclassification.get("rule")
        != "output_equals_prompt_true_to_prompt_only_output"
        or reclassification.get("score_conditioned") is not False
    ):
        raise ValueError("G1R parent reclassification mismatch")
    validate_zero_attestation(result.get("execution_attestation", {}))

    counts = result.get("counts", {})
    attempt_count = counts.get("supplement_attempt_count")
    if (
        type(attempt_count) is not int
        or not TARGET_SUPPLEMENT <= attempt_count
        <= MAX_ATTEMPTS - START_ATTEMPT
        or counts.get("retained_parent_count") != RETAINED_PARENT
        or counts.get("supplement_eligible_count")
        != TARGET_SUPPLEMENT
        or counts.get("supplement_failure_count")
        != attempt_count - TARGET_SUPPLEMENT
        or counts.get("combined_eligible_count") != TARGET_COMBINED
        or counts.get("target_supplement") != TARGET_SUPPLEMENT
        or counts.get("target_combined") != TARGET_COMBINED
    ):
        raise ValueError("G1R terminal counts mismatch")
    attempts = result.get("attempts", [])
    expected_indices = list(
        range(START_ATTEMPT, START_ATTEMPT + attempt_count)
    )
    if (
        len(attempts) != attempt_count
        or [row.get("attempt_index") for row in attempts]
        != expected_indices
    ):
        raise ValueError("G1R continuation sequence mismatch")
    continuation = result.get("continuation", {})
    if (
        continuation.get("start_attempt") != START_ATTEMPT
        or continuation.get("end_attempt_inclusive")
        != expected_indices[-1]
        or continuation.get("strictly_sequential") is not True
        or continuation.get("v15_output_identity_exclusion_count")
        != len(v15_output_identities)
    ):
        raise ValueError("G1R continuation summary mismatch")

    eligible_attempts = []
    failure_counts = Counter()
    all_output_identities = []
    failed_output_identities = []
    seen_supplement = set()
    for row in attempts:
        attempt_index = row["attempt_index"]
        if (
            row.get("source_raw_text_sha256")
            != source_identities[attempt_index]
            or row.get("seed") != generation_seed(
                source_identities[attempt_index]
            )
            or not common.is_sha256(row.get("prompt_sha256"))
            or not common.is_sha256(row.get("hidden_log_sha256"))
        ):
            raise ValueError("G1R source/seed identity mismatch")
        common.validate_sampling_metrics(row.get("sampling"))
        codes = row.get("exclusion_codes")
        if (
            not isinstance(codes, list)
            or codes != sorted(set(codes))
            or row.get("eligible") is not (not codes)
        ):
            raise ValueError("G1R eligibility evidence mismatch")
        elapsed = row.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("G1R runtime evidence mismatch")
        output = row.get("output", {})
        identity = output.get("raw_text_sha256")
        prompt_only = output.get("output_equals_prompt")
        if type(prompt_only) is not bool:
            raise ValueError("G1R prompt equality evidence mismatch")
        if ("prompt_only_output" in codes) is not prompt_only:
            raise ValueError("G1R prompt-only exclusion mismatch")
        if identity is not None and not common.is_sha256(identity):
            raise ValueError("G1R output identity mismatch")
        if identity in v15_output_identities:
            if "duplicate_v15_output" not in codes:
                raise ValueError("G1R v15 duplicate exclusion mismatch")
        elif "duplicate_v15_output" in codes:
            raise ValueError("G1R spurious v15 duplicate exclusion")
        if identity in seen_supplement:
            if "duplicate_supplement_output" not in codes:
                raise ValueError(
                    "G1R supplement duplicate exclusion mismatch"
                )
        elif "duplicate_supplement_output" in codes:
            raise ValueError("G1R spurious supplement duplicate exclusion")
        if identity is not None:
            all_output_identities.append(identity)
            seen_supplement.add(identity)
        failure_counts.update(codes)
        if codes:
            if identity is not None:
                failed_output_identities.append(identity)
            continue
        if (
            identity is None
            or output.get("unit_count", 0) < 2
            or not 0 < output.get("max_native_tokens", 0) <= 512
            or prompt_only
        ):
            raise ValueError("eligible G1R output fails P0 structure")
        eligible_attempts.append(row)
    if len(eligible_attempts) != TARGET_SUPPLEMENT:
        raise ValueError("G1R eligible attempt count mismatch")
    if dict(sorted(failure_counts.items())) != counts.get("failure_codes"):
        raise ValueError("G1R failure-code summary mismatch")

    bounded = result.get("bounded_sampling", {})
    if (
        bounded.get("patch_id") != common.PATCH_ID
        or bounded.get("maximum_trials_per_segment") != 100
        or bounded.get("total_trials")
        != sum(row["sampling"]["total_trials"] for row in attempts)
        or bounded.get("fallback_attempts")
        != sum(
            row["sampling"]["fallback_acceptances"] > 0
            for row in attempts
        )
        or bounded.get("fallback_acceptances")
        != sum(
            row["sampling"]["fallback_acceptances"]
            for row in attempts
        )
    ):
        raise ValueError("G1R bounded sampling summary mismatch")

    manifest = result.get("supplement_manifest", {})
    if (
        manifest.get("sha256") != positive_metadata["sha256"]
        or manifest.get("bytes") != positive_metadata["bytes"]
        or manifest.get("rows") != TARGET_SUPPLEMENT
        or manifest.get("relative_path") != positive_relative_path
        or manifest.get("contains_raw_text") is not False
        or len(positive_rows) != TARGET_SUPPLEMENT
    ):
        raise ValueError("G1R supplement manifest metadata mismatch")
    for index, (positive, attempt) in enumerate(
        zip(positive_rows, eligible_attempts)
    ):
        expected = {
            "supplement_index": index,
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
        if positive != expected:
            raise ValueError("G1R supplement row mismatch")

    sealed = result.get("sealed_checkpoint", {})
    if (
        not common.is_sha256(expected_sealed_sha256)
        or sealed.get("sha256") != expected_sealed_sha256
        or sealed_metadata["sha256"] != expected_sealed_sha256
        or sealed.get("bytes") != sealed_metadata["bytes"]
        or sealed.get("relative_path") != sealed_relative_path
        or sealed.get("contains_raw_text") is not True
        or sealed.get("public_copy_allowed") is not False
    ):
        raise ValueError("G1R sealed checkpoint attestation mismatch")
    runtime = result.get("runtime", {})
    peak = runtime.get("max_cuda_memory_allocated_bytes")
    total = runtime.get("total_cuda_memory_bytes")
    if (
        type(peak) is not int
        or type(total) is not int
        or peak < 0
        or total <= 0
        or peak / total >= 0.80
        or not result.get("runtime", {}).get("process_loads")
    ):
        raise ValueError("G1R resource gate failed")
    return {
        "attempt_count": attempt_count,
        "eligible_attempts": eligible_attempts,
        "failed_output_identities": failed_output_identities,
        "peak_vram_fraction": peak / total,
    }


def build_combined(retained, supplement):
    rows = []
    for origin, source_rows in (
        ("v15_retained", retained),
        ("v16_supplement", supplement),
    ):
        for origin_index, row in enumerate(source_rows):
            rows.append({
                "positive_index": len(rows),
                "origin": origin,
                "origin_index": origin_index,
                "attempt_index": row["attempt_index"],
                "source_raw_text_sha256": (
                    row["source_raw_text_sha256"]
                ),
                "output_raw_text_sha256": (
                    row["output_raw_text_sha256"]
                ),
                "seed": row["seed"],
                "unit_count": row["unit_count"],
                "max_native_tokens": row["max_native_tokens"],
                "contains_raw_text": False,
            })
    if len(rows) != TARGET_COMBINED:
        raise ValueError("combined positive count mismatch")
    outputs = common.identities(rows, "output_raw_text_sha256")
    sources = common.identities(rows, "source_raw_text_sha256")
    if (
        len(set(outputs)) != TARGET_COMBINED
        or len(set(sources)) != TARGET_COMBINED
    ):
        raise ValueError("combined positive identities are not unique")
    return rows


def overlap_summary(combined, comparison):
    overlap = sorted(set(combined) & set(comparison))
    return {
        "count": len(overlap),
        "identities_sha256": common.sha256_value(overlap),
    }


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--generation-runner", type=Path, required=True)
    value.add_argument("--v15-h0-runner", type=Path, required=True)
    value.add_argument("--bounded-sampler", type=Path, required=True)
    value.add_argument("--source-identities", type=Path, required=True)
    value.add_argument("--v12-source-identities", type=Path, required=True)
    value.add_argument("--v12-a1-result", type=Path, required=True)
    value.add_argument("--historical-semstamp", type=Path, required=True)
    value.add_argument("--historical-simmark", type=Path, required=True)
    value.add_argument("--v15-g1-result", type=Path, required=True)
    value.add_argument(
        "--v15-positive-identities", type=Path, required=True
    )
    value.add_argument("--v15-h0-stop", type=Path, required=True)
    value.add_argument(
        "--v15-sealed-checkpoint", type=Path, required=True
    )
    value.add_argument("--g1r-result", type=Path, required=True)
    value.add_argument(
        "--supplement-positive-identities", type=Path, required=True
    )
    value.add_argument(
        "--supplement-sealed-checkpoint", type=Path, required=True
    )
    value.add_argument("--expected-g1r-result-sha256", required=True)
    value.add_argument(
        "--expected-supplement-positive-sha256", required=True
    )
    value.add_argument(
        "--expected-supplement-sealed-sha256", required=True
    )
    value.add_argument("--combined-output", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main():
    args = parser().parse_args()
    root = args.root.resolve()
    fixed = {
        "protocol": (args.protocol, PROTOCOL_SHA256),
        "g1r_contract": (args.contract, CONTRACT_SHA256),
        "generation_runner": (
            args.generation_runner, GENERATION_RUNNER_SHA256
        ),
        "v15_h0_runner": (
            args.v15_h0_runner, V15_H0_RUNNER_SHA256
        ),
        "bounded_sampler": (
            args.bounded_sampler, BOUNDED_SAMPLER_SHA256
        ),
        "source_identities": (
            args.source_identities, common.V15_SOURCE_SHA256
        ),
        "v12_source_identities": (
            args.v12_source_identities, common.V12_SOURCE_SHA256
        ),
        "v12_a1_result": (
            args.v12_a1_result, common.V12_A1_SHA256
        ),
        "historical_semstamp": (
            args.historical_semstamp, common.HISTORICAL_SEMSTAMP_SHA256
        ),
        "historical_simmark": (
            args.historical_simmark, common.HISTORICAL_SIMMARK_SHA256
        ),
        "v15_g1_result": (
            args.v15_g1_result, V15_G1_RESULT_SHA256
        ),
        "v15_positive_identities": (
            args.v15_positive_identities, V15_POSITIVE_SHA256
        ),
        "v15_h0_stop": (args.v15_h0_stop, V15_H0_STOP_SHA256),
        "v15_sealed_checkpoint": (
            args.v15_sealed_checkpoint, V15_SEALED_SHA256
        ),
    }
    inputs = {}
    for label, (path, expected) in fixed.items():
        metadata = common.verify_file(path, expected, label)
        metadata["relative_path"] = common.relative_to_root(path, root)
        inputs[label] = metadata
    if (
        inputs["v15_sealed_checkpoint"]["bytes"] != V15_SEALED_BYTES
        or stat.S_IMODE(args.v15_sealed_checkpoint.stat().st_mode)
        != 0o600
    ):
        raise ValueError("v15 sealed checkpoint metadata mismatch")

    for expected, label in (
        (args.expected_g1r_result_sha256, "G1R result"),
        (
            args.expected_supplement_positive_sha256,
            "supplement positive",
        ),
        (
            args.expected_supplement_sealed_sha256,
            "supplement sealed",
        ),
    ):
        if not common.is_sha256(expected):
            raise ValueError(f"expected {label} SHA is invalid")
    terminal_specs = (
        (
            "g1r_result",
            args.g1r_result,
            args.expected_g1r_result_sha256,
        ),
        (
            "supplement_positive_identities",
            args.supplement_positive_identities,
            args.expected_supplement_positive_sha256,
        ),
        (
            "supplement_sealed_checkpoint",
            args.supplement_sealed_checkpoint,
            args.expected_supplement_sealed_sha256,
        ),
    )
    for label, path, expected in terminal_specs:
        metadata = common.verify_file(path, expected, label)
        metadata["relative_path"] = common.relative_to_root(path, root)
        inputs[label] = metadata
    if (
        stat.S_IMODE(
            args.supplement_sealed_checkpoint.stat().st_mode
        ) != 0o600
    ):
        raise ValueError("supplement sealed checkpoint is not private")

    source_rows = common.read_jsonl(args.source_identities)
    source_identities = common.validate_v15_sources(source_rows)
    v12_rows = common.read_jsonl(args.v12_source_identities)
    common.validate_v12_sources(v12_rows)
    v12_smoke = common.validate_v12_smoke(json.loads(
        args.v12_a1_result.read_text(encoding="utf-8")
    ))
    historical_rows = (
        common.read_jsonl(args.historical_semstamp)
        + common.read_jsonl(args.historical_simmark)
    )
    common.identities(historical_rows, "raw_text_sha256")

    v15_result = json.loads(
        args.v15_g1_result.read_text(encoding="utf-8")
    )
    retained, v15_outputs, v15_failed = validate_parent(
        v15_result,
        common.read_jsonl(args.v15_positive_identities),
        json.loads(args.v15_h0_stop.read_text(encoding="utf-8")),
    )
    supplement_rows = common.read_jsonl(
        args.supplement_positive_identities
    )
    g1r_result = json.loads(
        args.g1r_result.read_text(encoding="utf-8")
    )
    if (
        g1r_result.get("generator_model")
        != v15_result.get("generator_model")
        or {
            key: g1r_result.get("runtime", {}).get(key)
            for key in RUNTIME_IDENTITY_KEYS
        }
        != {
            key: v15_result.get("runtime", {}).get(key)
            for key in RUNTIME_IDENTITY_KEYS
        }
    ):
        raise ValueError("G1R model/runtime identity mismatch")
    supplement = validate_supplement(
        g1r_result,
        supplement_rows,
        inputs["supplement_positive_identities"],
        inputs["supplement_positive_identities"]["relative_path"],
        source_identities,
        v15_outputs,
        args.expected_supplement_sealed_sha256,
        inputs["supplement_sealed_checkpoint"],
        inputs["supplement_sealed_checkpoint"]["relative_path"],
    )
    combined = build_combined(retained, supplement_rows)
    combined_outputs = [
        row["output_raw_text_sha256"] for row in combined
    ]
    comparisons = {
        "all_attempt_sources": source_identities,
        "v12_all_sources": [
            row["raw_text_sha256"] for row in v12_rows
        ],
        "v12_calibration_null": [
            row["raw_text_sha256"] for row in v12_rows
            if row["cohort"] == "calibration_null"
        ],
        "v12_independent_test_null": [
            row["raw_text_sha256"] for row in v12_rows
            if row["cohort"] == "independent_test_null"
        ],
        "historical_all": [
            row["raw_text_sha256"] for row in historical_rows
        ],
        "v12_smoke_outputs": v12_smoke,
        "v15_failed_outputs": v15_failed,
        "v15_nonretained_outputs": (
            v15_outputs
            - {
                row["output_raw_text_sha256"] for row in retained
            }
        ),
        "supplement_failed_outputs": supplement[
            "failed_output_identities"
        ],
    }
    overlaps = {
        label: overlap_summary(combined_outputs, values)
        for label, values in comparisons.items()
    }
    if any(item["count"] for item in overlaps.values()):
        raise ValueError("combined identity overlap gate failed")
    combined_metadata = atomic_write_jsonl(
        args.combined_output, combined
    )
    combined_metadata["relative_path"] = common.relative_to_root(
        args.combined_output, root
    )
    combined_metadata["contains_raw_text"] = False

    output = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "go",
        "contains_raw_text": False,
        "scope": "identity_only_no_detector_scores",
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "inputs": inputs,
        "counts": {
            "retained_parent": RETAINED_PARENT,
            "supplement_positive": TARGET_SUPPLEMENT,
            "combined_positive": TARGET_COMBINED,
            "combined_output_unique": len(set(combined_outputs)),
            "combined_source_unique": len({
                row["source_raw_text_sha256"] for row in combined
            }),
            "supplement_attempts": supplement["attempt_count"],
        },
        "combined_manifest": combined_metadata,
        "overlaps": overlaps,
        "resource_gate": {
            "peak_vram_fraction": supplement["peak_vram_fraction"],
            "below_0_80": supplement["peak_vram_fraction"] < 0.80,
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "v15_sealed_checkpoint_content_parsed": False,
            "supplement_sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "decision": {
            "h0_identity_freeze": "go",
            "b0_detector_health": "authorized_to_freeze_protocol",
            "b1_exact_certificate": "stop",
            "next_action": "freeze a separate B0 detector-health protocol",
        },
    }
    common.atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
