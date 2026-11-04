"""Independently audit the v3 SemStamp throughput sweep without a model."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


EXPECTED_BATCHES = (64, 128, 256, 512)
PROBE_REFERENCE_SHA = (
    "218d1ed9f5683e0053953a3d429f57c0cf4d282e78775a326ddd3808871684dc"
)
MAXIMUM_REFERENCE_SHA = (
    "704a4724b48f6d49faaad8e21592524568c619c85cda93abaf2b5f32f5518f43"
)
MAXIMUM_OBSERVED_SHA = (
    "00d3ebbf5328b576913b5d584df6722c4beeb2a5a0e28d8842e62bf015a8552a"
)
RATE_TIE_FRACTION = 0.005


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_winner(rows):
    minimum_rate = min(
        row["provider"]["feature_seconds_per_input_token"]
        for row in rows
    )
    tied = [
        row for row in rows
        if row["provider"]["feature_seconds_per_input_token"]
        <= minimum_rate * (1 + RATE_TIE_FRACTION)
    ]
    return min(tied, key=lambda row: row["batch_size"])


def all_zero_attestation(attestation, allowed_nonzero=()):
    return all(
        value == 0 or key in allowed_nonzero
        for key, value in attestation.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    experiment = args.experiment_dir.resolve()
    output = args.output.resolve()
    if (
        not experiment.is_relative_to(root)
        or not output.is_relative_to(experiment)
        or output.exists()
    ):
        raise ValueError("audit paths must be new and inside the experiment")

    contract_path = experiment / "CONTRACT.json"
    result_path = experiment / "THROUGHPUT_RESULT.json"
    contract = load_json(contract_path)
    result = load_json(result_path)
    probes = [
        load_json(experiment / f"nearest_300_batch_{batch}.json")
        for batch in EXPECTED_BATCHES
    ]
    maximum = load_json(experiment / "maximum_remaining_batch_128.json")
    maximum_replays = maximum["provider"]["replays"]

    result_probe_by_batch = {
        row["batch_size"]: row for row in result["probe_results"]
    }
    winner = select_winner(probes)
    probe_seconds = sum(
        row["provider"]["total_feature_seconds"] for row in probes
    )
    maximum_seconds = maximum["provider"]["total_feature_seconds"]
    corrected_total = probe_seconds + maximum_seconds
    baseline_rate = probes[0]["provider"][
        "feature_seconds_per_input_token"
    ]
    speedup = baseline_rate / winner["provider"][
        "feature_seconds_per_input_token"
    ]

    probe_hashes = {
        row["provider"]["replays"][0]["feature_sha256"]
        for row in probes
    }
    query_hashes = {
        row["topology"]["query_sequence_sha256"] for row in probes
    }
    artifact_hashes_exact = all(
        sha256_file(
            experiment / f"nearest_300_batch_{batch}.json"
        ) == result_probe_by_batch[batch]["artifact_sha256"]
        for batch in EXPECTED_BATCHES
    ) and (
        sha256_file(experiment / "maximum_remaining_batch_128.json")
        == result["maximum_confirmation"]["artifact_sha256"]
    )

    checks = {
        "artifact_hashes_exact": artifact_hashes_exact,
        "batch_set_exact": tuple(
            row["batch_size"] for row in probes
        ) == EXPECTED_BATCHES,
        "contract_frozen_and_authorized": (
            contract["status"] == "frozen"
            and contract["execution_authorized"] is True
            and contract["parameters"]["batch_sizes"]
            == list(EXPECTED_BATCHES)
        ),
        "corrected_feature_seconds_le_3600": corrected_total <= 3600,
        "detector_certificate_generation_calls_zero": (
            all(
                all_zero_attestation(
                    row["execution_attestation"],
                    allowed_nonzero=(
                        "model_loads",
                        "provider_feature_replays",
                    ),
                )
                for row in (*probes, maximum)
            )
            and all_zero_attestation(
                result["execution_attestation"],
                allowed_nonzero=("model_loads_upper_bound",),
            )
        ),
        "exitcode_is_protocol_stop": (
            (experiment / "SWEEP_RUN.exitcode")
            .read_text(encoding="ascii")
            .strip()
            == "3"
        ),
        "maximum_artifact_is_stop": maximum["status"] == "stop",
        "maximum_exactly_one_failed_gate": (
            {
                key for key, value in maximum["gates"].items()
                if not value
            }
            == {"reference_region_sha256_exact"}
        ),
        "maximum_identity_and_topology_exact": (
            maximum["gates"]["input_identity_exact"]
            and maximum["gates"]["topology_counts_exact"]
            and maximum["topology"]["deduplicated_feature_texts"]
            == contract["references"]["maximum_remaining"][
                "deduplicated_feature_texts"
            ]
            and maximum["topology"]["deduplicated_input_tokens"]
            == contract["references"]["maximum_remaining"][
                "deduplicated_input_tokens"
            ]
        ),
        "maximum_reference_sha_bound": (
            maximum["provider"]["reference_feature_sha256"]
            == MAXIMUM_REFERENCE_SHA
            == contract["references"]["maximum_remaining"][
                "feature_sha256"
            ]
        ),
        "maximum_replays_exact_each_other": (
            len(maximum_replays) == 2
            and maximum_replays[0]["feature_sha256"]
            == maximum_replays[1]["feature_sha256"]
            == MAXIMUM_OBSERVED_SHA
        ),
        "maximum_resource_gates_pass": (
            maximum["gates"]["cuda_reserved_le_80_percent"]
            and maximum["gates"]["process_rss_le_16_gib"]
        ),
        "maximum_semantic_drift_observed": (
            MAXIMUM_OBSERVED_SHA != MAXIMUM_REFERENCE_SHA
        ),
        "probe_all_gates_pass": all(
            row["status"] == "pass"
            and all(row["gates"].values())
            for row in probes
        ),
        "probe_query_sequence_exact": len(query_hashes) == 1,
        "probe_region_sha_exact": probe_hashes == {PROBE_REFERENCE_SHA},
        "public_artifacts_claim_no_raw_text": (
            all(row["contains_raw_text"] is False for row in probes)
            and maximum["contains_raw_text"] is False
            and result["contains_raw_text"] is False
        ),
        "result_stop_exact": (
            result["status"] == "stop"
            and result["winner_batch_size"] == 128
            and result["gates"]["confirm_maximum_remaining_passed"]
            is False
        ),
        "summary_failed_attempt_omission_identified": (
            math.isclose(
                result["total_feature_gpu_seconds"],
                probe_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and maximum_seconds > 0
            and corrected_total > result["total_feature_gpu_seconds"]
        ),
        "winner_rule_recomputed_exact": (
            winner["batch_size"] == 128
            and math.isclose(
                result["winner_probe_speedup_over_batch_64"],
                speedup,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
    }
    audit = {
        "audit_mode": "independent_model_free_throughput_correction",
        "checks": checks,
        "contains_raw_text": False,
        "corrected_accounting": {
            "maximum_failed_confirmation_feature_seconds": (
                maximum_seconds
            ),
            "probe_feature_seconds": probe_seconds,
            "reported_feature_seconds": result[
                "total_feature_gpu_seconds"
            ],
            "total_feature_seconds": corrected_total,
        },
        "decision": "THROUGHPUT_STOP_BATCH_DEPENDENT_REGION",
        "evidence": {
            "maximum_observed_region_sha256": MAXIMUM_OBSERVED_SHA,
            "maximum_reference_region_sha256": MAXIMUM_REFERENCE_SHA,
            "probe_region_sha256": PROBE_REFERENCE_SHA,
            "winner_batch_size": winner["batch_size"],
            "winner_probe_speedup_over_batch_64": speedup,
        },
        "inputs": {
            "contract_sha256": sha256_file(contract_path),
            "maximum_artifact_sha256": sha256_file(
                experiment / "maximum_remaining_batch_128.json"
            ),
            "result_sha256": sha256_file(result_path),
        },
        "next_authorized_stage": (
            "score_blind_variable_n_profile_with_canonical_batch_64"
        ),
        "schema_version": 1,
        "stage": "semstamp_certified_defense_v3_wp2_throughput_audit",
        "status": "pass" if all(checks.values()) else "fail",
    }
    atomic_write_json(output, audit)
    print(json.dumps({
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "decision": audit["decision"],
        "status": audit["status"],
    }, sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
