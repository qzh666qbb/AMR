"""Independently audit the terminal v2 WP3 detector-smoke evidence."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile


STAGE = "semstamp_certified_defense_v2_wp3_detector_smoke"
AUDIT_STAGE = f"{STAGE}_terminal_audit"
PROTOCOL_SHA256 = (
    "49cc738c67bc41f28e05095a983a966c52150e5c3c7a77d9706e49601f33d627"
)
CONTRACT_SHA256 = (
    "d6af481ae2c27a2015c2c46296e839264c04be0c30a7d9f00c920780b411b203"
)
MANIFEST_SHA256 = (
    "326f36d14e8dcb7c12ce019a2aa3668d67b30ad3f5a9284bbf4e4451fde69697"
)
RUNNER_SHA256 = (
    "d6f90f713c676b7daa9f04dc6fad1f55e44dabfe4fb9618f78c7bc600b16edda"
)
AUTHORIZATION_SHA256 = (
    "3f39cffa92b3a34aae0f955c260f2d8a13e0ab2a2563f258921984525bcb8e79"
)
SMOKE_PLAN_SHA256 = (
    "f4354914023a62927931e1423fa4db119cc202b6c002e765b3a79a1bd80509bf"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V17_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
SMOKE_RESULT_SHA256 = (
    "b604c5ce22be68a1fb08d0774b39f37aa19007c5d5d9b9bc998594aa696b0fe8"
)
REPLAY_SHA256 = (
    "a4b7403bae9095be1a5235b447a57bfc722f19c6c20556c6aae0df867dc2cc7b"
)
CHECKPOINT_SHA256 = (
    "999f919009ca1de4f8dfe059fee951edaef611c49d2564dbcaae553a61dfceda"
)
ATTEMPT0_RESULT_SHA256 = (
    "21c84fcb93bde6884c7abb93e8775ffa39330207531eccd61c6fd7163a4e3cf8"
)
ATTEMPT0_FAILURE_SHA256 = (
    "b84f6efaf8958f8a8840e197b6d1ab75ae8405891b6640b86ff138f3682edd69"
)
ATTEMPT0_HEARTBEAT_SHA256 = (
    "5bf4a2fe934633be6b65415517b6c8fdd2268420ae44f71d4cc4a3f31f336080"
)
ATTEMPT0_CHECKPOINT_SHA256 = (
    "c868c6f7c9299ab66ace90451822b637edf193e4f9b1c8fdb812529e6c17b3e9"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
RELATION_HEX = (
    "0000010000010000000001000000010000000100000100000100000000000100"
    "0000010000000001000100000100000001000001000000000000000100000100"
)
RELATION = bytes.fromhex(RELATION_HEX)
COHORTS = ("calibration_null", "independent_test_null", "positive")
SMOKE_PER_COHORT = 32
SMOKE_ROWS = 96
REPLAY_COUNT = 2
EXPECTED_RISK_ROWS = 42
EXPECTED_ZERO_GAPS = 59
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) for row in rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


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
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw public key at {location}: {overlap}")
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def parse_manifest(path):
    entries = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in entries
        ):
            raise ValueError(f"invalid manifest line {line_number}")
        entries[relative] = digest
    return entries


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_file(path, expected_sha, expected_mode=None):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"file SHA mismatch: {path}")
    if (
        expected_mode is not None
        and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
    ):
        raise ValueError(f"file mode mismatch: {path}")


def score_from_counts(hits, edges):
    return (hits - 0.25 * edges) / math.sqrt(edges * 0.25 * 0.75)


def paths_for(root, output_dir, attempt0_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    attempt0_dir = Path(attempt0_dir).resolve()
    output = Path(output).resolve()
    for path in (output_dir, attempt0_dir, output):
        if not path.is_relative_to(root):
            raise ValueError("audit paths must remain inside root")
    p0_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
    )
    v17_dir = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "runner": root / "scripts/run_semstamp_certified_defense_v2_wp3_smoke.py",
        "auditor": Path(__file__).resolve(),
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "result": output_dir / "SMOKE_RESULT.json",
        "rows0": output_dir / "SMOKE_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "SMOKE_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/SMOKE_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "audit": output,
        "smoke_plan": p0_dir / "SMOKE_PLAN.jsonl",
        "p0_rows": p0_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "v17_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "attempt0_result": attempt0_dir / "ATTEMPT_RESULT.json",
        "attempt0_failure": attempt0_dir / "LAST_FAILURE.json",
        "attempt0_heartbeat": attempt0_dir / "HEARTBEAT.json",
        "attempt0_checkpoint": attempt0_dir / "SMOKE_CHECKPOINT.json",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


def verify_frozen_inputs(paths, contract):
    manifest = parse_manifest(paths["manifest"])
    if len(manifest) != 46 or set(manifest) != set(contract.get("inputs", {})):
        raise ValueError("input manifest/contract set mismatch")
    for relative, digest in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        metadata = contract["inputs"][relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or metadata.get("sha256") != digest
            or metadata.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"input metadata mismatch: {relative}")
        verify_file(path, digest, metadata.get("mode"))
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    return manifest


def verify_attempt0(paths):
    expected = {
        "attempt0_result": ATTEMPT0_RESULT_SHA256,
        "attempt0_failure": ATTEMPT0_FAILURE_SHA256,
        "attempt0_heartbeat": ATTEMPT0_HEARTBEAT_SHA256,
        "attempt0_checkpoint": ATTEMPT0_CHECKPOINT_SHA256,
    }
    for label, digest in expected.items():
        verify_file(paths[label], digest, "0600")
    result = read_json(paths["attempt0_result"])
    failure = read_json(paths["attempt0_failure"])
    heartbeat = read_json(paths["attempt0_heartbeat"])
    checkpoint = read_json(paths["attempt0_checkpoint"])
    if (
        result.get("terminal") != "WP3_SMOKE_INVALID_OR_INCOMPLETE"
        or result.get("completed_rows") != 0
        or result.get("detector_model_loads") != 1
        or result.get("detector_scores_computed") != 0
        or failure.get("terminal") != "WP3_SMOKE_INVALID_OR_INCOMPLETE"
        or heartbeat.get("completed_total") != 0
        or heartbeat.get("target_total") != 192
        or any(checkpoint["replays"][key] for key in ("0", "1"))
    ):
        raise ValueError("attempt0 was not an empty pre-score failure")
    if "CUBLAS_WORKSPACE_CONFIG" not in failure.get("error", ""):
        raise ValueError("attempt0 failure cause mismatch")
    reject_raw_public_keys([result, failure, heartbeat, checkpoint])
    return expected


def verify_rows(paths):
    if hashlib.sha256(RELATION).hexdigest() != RELATION_SHA256:
        raise ValueError("independent relation bytes mismatch")
    plan = read_jsonl(paths["smoke_plan"])
    p0_rows = read_jsonl(paths["p0_rows"])
    references = read_jsonl(paths["v17_rows"])
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    checkpoint = read_json(paths["checkpoint"])
    reject_raw_public_keys([plan, p0_rows, references, rows0, rows1, checkpoint])
    if (
        len(plan) != SMOKE_ROWS
        or len(rows0) != SMOKE_ROWS
        or rows0 != rows1
        or jsonl_bytes(rows0) != paths["rows0"].read_bytes()
        or paths["rows0"].read_bytes() != paths["rows1"].read_bytes()
        or len(p0_rows) != 3000
        or len(references) != 3000
    ):
        raise ValueError("smoke public rows are incomplete or noncanonical")
    indices = [row["plan_index"] for row in plan]
    if (
        len(set(indices)) != SMOKE_ROWS
        or checkpoint.get("indices") != indices
        or checkpoint.get("contract_sha256") != CONTRACT_SHA256
        or checkpoint.get("authorization_sha256") != AUTHORIZATION_SHA256
        or checkpoint.get("smoke_plan_sha256") != SMOKE_PLAN_SHA256
        or set(checkpoint.get("replays", {})) != {"0", "1"}
        or any(
            len(checkpoint["replays"][str(replay)]) != SMOKE_ROWS
            for replay in range(REPLAY_COUNT)
        )
        or checkpoint["replays"]["0"] != checkpoint["replays"]["1"]
    ):
        raise ValueError("sealed checkpoint is incomplete or drifted")

    for offset, (selected, public) in enumerate(zip(plan, rows0)):
        entry = checkpoint["replays"]["0"][offset]
        if entry.get("public") != public:
            raise ValueError(f"checkpoint/public mismatch at {offset}")
        regions = entry.get("regions")
        if (
            not isinstance(regions, list)
            or len(regions) != public["unit_count"]
            or any(type(value) is not int or value < 0 or value > 7 for value in regions)
        ):
            raise ValueError(f"invalid regions at {offset}")
        hits = [
            RELATION[previous * 8 + current]
            for previous, current in zip(regions, regions[1:])
        ]
        expected_score = score_from_counts(sum(hits), len(hits))
        if (
            public["regions_sha256"] != hashlib.sha256(bytes(regions)).hexdigest()
            or public["hits_sha256"] != hashlib.sha256(bytes(hits)).hexdigest()
            or public["edge_count"] != len(hits)
            or public["hit_count"] != sum(hits)
            or public["direct_score"] != expected_score
            or public["official_score"] != expected_score
            or public["score_abs_error"] != 0.0
            or not math.isfinite(expected_score)
        ):
            raise ValueError(f"independent score mismatch at {offset}")

        reference = references[public["plan_index"]]
        if any(public.get(key) != value for key, value in reference.items()):
            raise ValueError(f"v17 reference mismatch at {offset}")
        parent = p0_rows[public["plan_index"]]
        structural = {
            "atom_count": parent["atom_count"],
            "byte_reconstruction_sha256": parent["byte_reconstruction_sha256"],
            "cohort": selected["cohort"],
            "cohort_index": selected["cohort_index"],
            "legality_sha256": selected["legality_sha256"],
            "plan_index": selected["plan_index"],
            "raw_text_sha256": selected["raw_text_sha256"],
            "refinement_count": selected["refinement_count"],
            "selection_class": selected["selection_class"],
            "smoke_index": selected["smoke_index"],
            "structural_rank": selected["structural_rank"],
            "topology_sha256": selected["topology_sha256"],
            "unit_count": selected["unit_count"],
            "zero_byte_gap_count": selected["zero_byte_gap_count"],
        }
        if (
            any(public.get(key) != value for key, value in structural.items())
            or public["p0_row_sha256"]
            != hashlib.sha256(canonical_bytes(parent)).hexdigest()
            or public["renderer_contract"] != "refined_nltk_boundary_atoms_v2"
            or public["stage"] != STAGE
            or public["v17_reference_exact"] is not True
        ):
            raise ValueError(f"P0/smoke structure mismatch at {offset}")

    if (
        Counter(row["cohort"] for row in plan)
        != {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        or sum(
            row["selection_class"] == "mandatory_refinement"
            for row in plan
        )
        != EXPECTED_RISK_ROWS
        or sum(row["zero_byte_gap_count"] for row in plan)
        != EXPECTED_ZERO_GAPS
    ):
        raise ValueError("smoke coverage mismatch")
    return plan, rows0


def verify_result(paths, contract, authorization):
    result = read_json(paths["result"])
    if canonical_bytes(result) != paths["result"].read_bytes():
        raise ValueError("smoke result is not canonical")
    expected_counts = {
        "direct_score_computations": 192,
        "mismatches": 0,
        "official_detect_lsh_calls": 192,
        "replays": 2,
        "rows_per_replay": 96,
        "sample_exclusions": 0,
        "sample_supplements": 0,
        "v17_reference_comparisons": 192,
    }
    outputs = result.get("outputs", {})
    if (
        result.get("schema_version") != 1
        or result.get("stage") != STAGE
        or result.get("status") != "go"
        or result.get("terminal") != "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN"
        or result.get("protocol_sha256") != PROTOCOL_SHA256
        or result.get("contract_sha256") != CONTRACT_SHA256
        or result.get("authorization_sha256") != AUTHORIZATION_SHA256
        or result.get("runner_sha256") != RUNNER_SHA256
        or result.get("smoke_plan_sha256") != SMOKE_PLAN_SHA256
        or not result.get("gates")
        or not all(result["gates"].values())
        or result.get("counts") != expected_counts
        or result.get("artifacts", {}).get("authorization") != authorization
        or result.get("artifacts", {}).get("inputs") != contract["inputs"]
        or result.get("contains_raw_text") is not False
    ):
        raise ValueError("smoke result terminal contract mismatch")
    expected_outputs = {
        "replay0": (paths["rows0"], REPLAY_SHA256, 118533),
        "replay1": (paths["rows1"], REPLAY_SHA256, 118533),
        "checkpoint": (paths["checkpoint"], CHECKPOINT_SHA256, 250467),
    }
    for label, (path, digest, size) in expected_outputs.items():
        value = outputs.get(label, {})
        if (
            value.get("sha256") != digest
            or value.get("bytes") != size
            or sha256_file(path) != digest
            or path.stat().st_size != size
        ):
            raise ValueError(f"smoke output binding mismatch: {label}")
    runtime = result.get("runtime", {})
    attestation = result.get("execution_attestation", {})
    decision = result.get("decision", {})
    if (
        runtime.get("device_name") != "NVIDIA GeForce RTX 3090"
        or runtime.get("nltk") != "3.8.1"
        or runtime.get("peak_vram_fraction", 1.0) >= 0.8
        or runtime.get("detector_elapsed_seconds", 901) > 900
        or attestation.get("detector_imports") != 1
        or attestation.get("detector_model_loads") != 1
        or attestation.get("detector_scores_computed") != 192
        or attestation.get("gpu_calls") != 192
        or attestation.get("detector_score_reads") != 0
        or attestation.get("certificate_calls") != 0
        or attestation.get("generation_calls") != 0
        or attestation.get("full_replay_started") is not False
        or attestation.get("wp4_started") is not False
        or decision.get("full_replay") != "authorized_to_freeze_design_only"
        or decision.get("full_replay_execution_authorized") is not False
        or decision.get("wp4_started") is not False
    ):
        raise ValueError("runtime or authorization boundary mismatch")
    reject_raw_public_keys(result)
    return result


def audit(root, output_dir, attempt0_dir, output):
    paths = paths_for(root, output_dir, attempt0_dir, output)
    fixed = {
        "protocol": (PROTOCOL_SHA256, "0644"),
        "contract": (CONTRACT_SHA256, "0644"),
        "manifest": (MANIFEST_SHA256, "0644"),
        "runner": (RUNNER_SHA256, "0644"),
        "authorization": (AUTHORIZATION_SHA256, "0600"),
        "result": (SMOKE_RESULT_SHA256, "0600"),
        "rows0": (REPLAY_SHA256, "0600"),
        "rows1": (REPLAY_SHA256, "0600"),
        "checkpoint": (CHECKPOINT_SHA256, "0600"),
        "smoke_plan": (SMOKE_PLAN_SHA256, "0600"),
        "p0_rows": (P0_ROWS_SHA256, "0600"),
        "v17_rows": (V17_ROWS_SHA256, "0600"),
    }
    for label, (digest, mode) in fixed.items():
        verify_file(paths[label], digest, mode)
    if paths["heartbeat"].exists() or paths["last_failure"].exists():
        raise ValueError("active execution residue exists after smoke Go")

    contract = read_json(paths["contract"])
    authorization = read_json(paths["authorization"])
    if (
        contract.get("status") != "frozen_design"
        or contract.get("execution_authorized") is not False
        or authorization.get("status") != "SMOKE_EXECUTION_AUTHORIZED"
        or authorization.get("scope") != "96_row_detector_smoke_only"
        or authorization.get("full_replay_authorized") is not False
        or authorization.get("wp4_authorized") is not False
        or authorization.get("certificate_authorized") is not False
    ):
        raise ValueError("design or authorization boundary mismatch")
    manifest = verify_frozen_inputs(paths, contract)
    attempt0_bindings = verify_attempt0(paths)
    _, rows = verify_rows(paths)
    result = verify_result(paths, contract, authorization)

    checks = {
        "frozen_design_and_authorization_exact": True,
        "all_46_input_bytes_sha_modes_exact": True,
        "repository_revisions_exact": True,
        "attempt0_failure_evidence_exact": True,
        "attempt0_completed_zero_rows_and_scores": True,
        "attempt0_checkpoint_empty": True,
        "successful_result_terminal_go": True,
        "all_runner_gates_pass": True,
        "smoke_96_unique_and_balanced": True,
        "all_42_risk_rows_retained": True,
        "all_59_zero_gaps_retained": True,
        "two_public_replays_byte_exact": True,
        "two_sealed_region_replays_exact": True,
        "all_192_region_hashes_recomputed": True,
        "all_192_hit_hashes_recomputed": True,
        "all_192_direct_scores_recomputed": True,
        "all_192_official_direct_scores_exact": True,
        "all_192_v17_references_exact": True,
        "all_192_p0_structures_exact": True,
        "no_nonfinite_scores": True,
        "no_sample_exclusion_or_supplement": True,
        "successful_transaction_single_model_load": True,
        "runtime_and_vram_below_limits": True,
        "active_failure_and_heartbeat_absent": True,
        "full_replay_wp4_certificate_not_started": True,
        "raw_text_absent_from_public_artifacts": True,
        "auditor_model_gpu_score_calls_zero": True,
    }
    audit_result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go" if all(checks.values()) else "stop",
        "terminal": (
            "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN_CONFIRMED"
        ),
        "scope": "independent_structural_and_score_recalculation_no_model_load",
        "bindings": {
            **{label: value[0] for label, value in fixed.items()},
            **attempt0_bindings,
            "auditor": sha256_file(paths["auditor"]),
        },
        "checks": checks,
        "counts": {
            "input_manifest_entries": len(manifest),
            "process_attempts": 2,
            "pre_score_invalid_attempts": 1,
            "pre_score_invalid_completed_rows": 0,
            "pre_score_invalid_score_computations": 0,
            "cumulative_model_loads": 2,
            "successful_transaction_model_loads": 1,
            "rows_per_replay": len(rows),
            "replays": REPLAY_COUNT,
            "independently_recomputed_scores": REPLAY_COUNT * len(rows),
            "v17_reference_comparisons": REPLAY_COUNT * len(rows),
            "p0_structure_comparisons": REPLAY_COUNT * len(rows),
            "mandatory_risk_rows": EXPECTED_RISK_ROWS,
            "zero_byte_gaps": EXPECTED_ZERO_GAPS,
        },
        "runtime": result["runtime"],
        "decision": {
            "smoke": "confirmed_go",
            "pre_score_restart": "disclosed_and_non_selective",
            "full_replay": "authorized_to_freeze_design_only",
            "full_replay_execution_authorized": False,
            "wp4_authorized": False,
            "certificate_authorized": False,
        },
        "audit_execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
            "gpu_calls": 0,
            "certificate_calls": 0,
            "generation_calls": 0,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(audit_result)
    atomic_write_json(paths["audit"], audit_result)
    print(json.dumps(audit_result, sort_keys=True))
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt0-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.attempt0_dir, args.output)


if __name__ == "__main__":
    main()
