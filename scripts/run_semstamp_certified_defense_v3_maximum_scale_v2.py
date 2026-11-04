"""Run the frozen v3 deterministic amplified maximum-scale protocol."""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import run_semstamp_certified_defense_v3_maximum_scale as base


AMPLIFIED_ATOM_COUNT = 6_500
OBSERVATION_SHA256 = (
    "7390f0e621279891703f287c7faf1d9aa428e9b53546c04f55be1dd15f3fde68"
)
CORRECTNESS_FILES = (
    "CORRECTNESS_MACOS_REPLAY0.json",
    "CORRECTNESS_MACOS_REPLAY1.json",
    "CORRECTNESS_LINUX_REPLAY0.json",
    "CORRECTNESS_LINUX_REPLAY1.json",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_correctness(directory):
    directory = Path(directory)
    artifacts = {}
    for name in CORRECTNESS_FILES:
        path = directory / name
        value = read_json(path)
        if (
            value.get("status") != "pass"
            or value.get("observation_stream_sha256")
            != OBSERVATION_SHA256
            or not all(value.get("gates", {}).values())
        ):
            raise ValueError(f"invalid correctness artifact: {name}")
        artifacts[name] = base.sha256_file(path)
    return artifacts


def amplified_row(source):
    output = copy.deepcopy(source)
    output["atom_count"] = AMPLIFIED_ATOM_COUNT
    output["original_boundaries"] = [
        boundary * AMPLIFIED_ATOM_COUNT // base.EXPECTED_ATOM_COUNT
        for boundary in source["original_boundaries"]
    ]
    boundaries = output["original_boundaries"]
    if (
        len(boundaries) != base.EXPECTED_ORIGINAL_EDGES
        or boundaries != sorted(set(boundaries))
        or boundaries[0] <= 0
        or boundaries[-1] >= AMPLIFIED_ATOM_COUNT
    ):
        raise AssertionError("amplified boundaries are invalid")
    return output


def profile(core, row, target_count, work_dir, label):
    timed = base.timed_core(
        core,
        (),
        base.payload(row, target_count, 3, 1),
        work_dir / f"{label}.time",
    )
    return {
        "elapsed_seconds": timed["elapsed_seconds"],
        "peak_rss_bytes": timed["peak_rss_bytes"],
        "result": timed["result"],
        "rss_measurement": timed["rss_measurement"],
        "stdout_sha256": timed["stdout_sha256"],
        "target_edge_count": target_count,
    }


def run(args):
    core = Path(args.core).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(work_dir, 0o700)
    correctness_hashes = validate_correctness(args.correctness_dir)
    source = base.read_jsonl_row(args.v2_rows, base.PLAN_INDEX)
    base.validate_identity(source)

    real_profiles = [
        profile(
            core,
            source,
            target_count,
            work_dir,
            f"real_m{target_count}",
        )
        for target_count in base.TARGET_COUNTS
    ]
    fixed_profile = next(
        item for item in real_profiles
        if item["target_edge_count"] == base.EXPECTED_ORIGINAL_EDGES
    )
    real_fixed_exact = (
        fixed_profile["result"]["unique_pair_scores"]
        == base.HISTORICAL_PAIR_COUNT
        and fixed_profile["result"]["unique_span_checks"]
        == base.HISTORICAL_SPAN_COUNT
    )
    real_qualifying = [
        item for item in real_profiles
        if (
            base.HISTORICAL_PAIR_COUNT
            < item["result"]["unique_pair_scores"]
            <= base.PAIR_CAP
        )
    ]

    amplified = amplified_row(source)
    amplified_profile = profile(
        core,
        amplified,
        base.EXPECTED_ORIGINAL_EDGES,
        work_dir,
        "amplified_topology",
    )
    sidecar = work_dir / "amplified_mode5.bin"
    dumped = base.timed_core(
        core,
        ("--dump-synthetic-sidecar", sidecar, "5"),
        base.payload(
            amplified,
            base.EXPECTED_ORIGINAL_EDGES,
            3,
            1,
        ),
        work_dir / "amplified_dump.time",
    )
    if dumped["stdout_sha256"] != amplified_profile["stdout_sha256"]:
        raise AssertionError("amplified sidecar dump changed topology")
    sidecar_bytes = sidecar.stat().st_size
    sidecar_sha256 = base.sha256_file(sidecar)
    scored = base.timed_core(
        core,
        (),
        base.payload(
            amplified,
            base.EXPECTED_ORIGINAL_EDGES,
            5,
            2,
            suffix=sidecar,
        ),
        work_dir / "amplified_mode5.time",
    )
    sidecar.unlink()
    if base.topology_fields(scored["result"]) != base.topology_fields(
        amplified_profile["result"]
    ):
        raise AssertionError("amplified mode-5 topology changed")

    timed_items = [
        *real_profiles,
        amplified_profile,
        dumped,
        scored,
    ]
    maximum_rss = max(item["peak_rss_bytes"] for item in timed_items)
    maximum_elapsed = max(
        item["elapsed_seconds"] for item in timed_items
    )
    amplified_pairs = amplified_profile[
        "result"
    ]["unique_pair_scores"]
    amplified_spans = amplified_profile[
        "result"
    ]["unique_span_checks"]
    gates = {
        "amplified_pair_count_gt_historical": (
            amplified_pairs > base.HISTORICAL_PAIR_COUNT
        ),
        "amplified_pair_count_le_200m": (
            amplified_pairs <= base.PAIR_CAP
        ),
        "amplified_span_count_le_2m": (
            amplified_spans <= base.SPAN_CAP
        ),
        "correctness_artifacts_exact": True,
        "maximum_call_le_3600_seconds": (
            maximum_elapsed <= base.TIMEOUT_SECONDS
        ),
        "maximum_rss_le_16_gib": (
            maximum_rss <= base.RSS_CAP_BYTES
        ),
        "mode5_topology_exact": True,
        "real_fixed_n_historical_counts_exact": real_fixed_exact,
        "real_selector_stop_reconstructed": not real_qualifying,
        "sidecar_dump_stdout_exact": True,
    }
    if not all(gates.values()):
        raise AssertionError(f"scale2 gate failure: {gates}")
    if not all(
        math.isfinite(item["elapsed_seconds"]) for item in timed_items
    ):
        raise AssertionError("non-finite scale2 timing")

    result = {
        "amplification": {
            "amplified_atom_count": AMPLIFIED_ATOM_COUNT,
            "amplified_boundaries_sha256": base.sha256_bytes(
                base.canonical_bytes(amplified["original_boundaries"])
            ),
            "formula": "floor(6500 * boundary / 5554)",
            "source_atom_count": base.EXPECTED_ATOM_COUNT,
        },
        "amplified_mode5": {
            "elapsed_seconds": scored["elapsed_seconds"],
            "peak_rss_bytes": scored["peak_rss_bytes"],
            "result": scored["result"],
            "rss_measurement": scored["rss_measurement"],
            "sidecar_bytes": sidecar_bytes,
            "sidecar_sha256": sidecar_sha256,
            "stdout_sha256": scored["stdout_sha256"],
        },
        "amplified_sidecar_dump": {
            "elapsed_seconds": dumped["elapsed_seconds"],
            "peak_rss_bytes": dumped["peak_rss_bytes"],
            "rss_measurement": dumped["rss_measurement"],
            "stdout_sha256": dumped["stdout_sha256"],
        },
        "amplified_topology": amplified_profile,
        "caps": {
            "historical_pair_count": base.HISTORICAL_PAIR_COUNT,
            "pair_cap": base.PAIR_CAP,
            "replay_timeout_seconds": base.TIMEOUT_SECONDS,
            "rss_cap_bytes": base.RSS_CAP_BYTES,
            "span_cap": base.SPAN_CAP,
        },
        "execution_attestation": {
            "certificate_computations": 0,
            "cuda_initializations": 0,
            "detector_score_computations": 0,
            "generation_calls": 0,
            "model_loads": 0,
            "network_provider_calls": 0,
        },
        "gates": gates,
        "inputs": {
            "core_sha256": base.sha256_file(core),
            "correctness_artifacts": correctness_hashes,
            "protocol_sha256": base.sha256_file(args.protocol),
            "runner_sha256": base.sha256_file(Path(__file__)),
            "v2_rows_sha256": base.sha256_file(args.v2_rows),
        },
        "maximum_observed_elapsed_seconds": maximum_elapsed,
        "maximum_observed_rss_bytes": maximum_rss,
        "real_target_profiles": real_profiles,
        "replay_index": args.replay_index,
        "schema_version": 1,
        "stage": "semstamp_certified_defense_v3_maximum_scale2",
        "status": "pass",
    }
    base.atomic_write_json(args.output, result)
    print(json.dumps({
        "amplified_pairs": amplified_pairs,
        "amplified_spans": amplified_spans,
        "mode5_stdout_sha256": scored["stdout_sha256"],
        "status": "pass",
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--v2-rows", required=True)
    parser.add_argument("--correctness-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--replay-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.replay_index not in (0, 1):
        parser.error("--replay-index must be 0 or 1")
    run(args)


if __name__ == "__main__":
    main()
