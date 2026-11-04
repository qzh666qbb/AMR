"""Independently audit terminal v2 WP3 full-replay evidence without a model."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile


STAGE = "semstamp_certified_defense_v2_wp3_detector_full_replay"
AUDIT_STAGE = f"{STAGE}_terminal_audit"
PROTOCOL_SHA256 = (
    "c4102127400bf1b20c7aa9a9849252bc7f737e8b628e67699bb4269213fb0d91"
)
CONTRACT_SHA256 = (
    "d62eeca5efdfb5b86554bf01060509b1211a0c7e52d3662a185b28b911ef0451"
)
MANIFEST_SHA256 = (
    "04b7a5d357d86b7a2adb7a369599336a74d0683648bd5f0096a6404147b4513a"
)
RUNNER_SHA256 = (
    "0c5d53c033b3d2f0605da0e51b84d807eb471fdbe9306502776f601e37fea93c"
)
AUTHORIZATION_SHA256 = (
    "9f4a76d655d7ae11af2249fe7908e423b9639e311f1ac6b0a5806caa91cd7b3b"
)
RESULT_SHA256 = (
    "5892ab27bd9bde618ee44612a0e047ac258a51c02f06f4311c26442822dba0ab"
)
REPLAY_SHA256 = (
    "50b37b7f716accc4728f5b9ca7f0d7377cb9b578c5c07d78b4eb776d6f20b21e"
)
CHECKPOINT_SHA256 = (
    "0b455ec7ec4aec1acd7797a00ace1e3bcea1f647224b2319b4a4cfd349e8e80d"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V17_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
V17_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
SMOKE_FINAL_SHA256 = (
    "64e509330781183057c7c479dacc3ae5862a39923f18dbfa3401c53236baa637"
)
SMOKE_AUDIT_SHA256 = (
    "163933e86d171fe5f289441580606d7bce66cfe6867baf59553438579f9badf5"
)
SMOKE_MANIFEST_SHA256 = (
    "326f36d14e8dcb7c12ce019a2aa3668d67b30ad3f5a9284bbf4e4451fde69697"
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
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
REPLAY_COUNT = 2
EXPECTED_RISK_ROWS = 42
EXPECTED_ZERO_GAPS = 59
EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
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
    if not entries:
        raise ValueError("empty manifest")
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


def verify_manifest(root, manifest_path, contract):
    manifest = parse_manifest(manifest_path)
    inputs = contract.get("inputs", {})
    if set(manifest) != set(inputs):
        raise ValueError("manifest/contract path set mismatch")
    for relative, digest in manifest.items():
        relative_path = Path(relative)
        path = (root / relative_path).resolve()
        value = inputs[relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(root)
            or value.get("sha256") != digest
            or value.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"input metadata mismatch: {relative}")
        verify_file(path, digest, value.get("mode"))
    return manifest


def score_from_counts(hits, edges):
    return (hits - 0.25 * edges) / math.sqrt(edges * 0.25 * 0.75)


def binomial_cdf(successes, trials, probability):
    if successes < 0:
        return 0.0
    if successes == trials:
        return 1.0
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_p
        + (trials - index) * log_q
        for index in range(successes + 1)
    ]
    maximum = max(terms)
    return min(
        1.0,
        math.exp(maximum)
        * sum(math.exp(value - maximum) for value in terms),
    )


def cp_upper(successes, trials, confidence=0.95):
    if successes == trials:
        return 1.0
    lower = 0.0
    upper = 1.0
    tail = 1 - confidence
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if binomial_cdf(successes, trials, midpoint) > tail:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def independently_summarize(rows):
    groups = {
        cohort: [
            row["direct_score"] for row in rows if row["cohort"] == cohort
        ]
        for cohort in COHORTS
    }
    if any(len(values) != COHORT_SIZE for values in groups.values()):
        raise ValueError("statistics cohort count mismatch")
    calibration = sorted(groups["calibration_null"])
    rank = math.ceil((COHORT_SIZE + 1) * 0.95)
    threshold = calibration[rank - 1]
    calibration_hits = sum(value > threshold for value in calibration)
    result = {
        "calibration": {
            "calibration_size": COHORT_SIZE,
            "target_fpr": 0.05,
            "order_rank": rank,
            "threshold": threshold,
            "maximum_calibration_exceedances": COHORT_SIZE - rank,
            "strict_exceedances": calibration_hits,
            "empirical_fpr": calibration_hits / COHORT_SIZE,
            "marginal_fpr_bound": (
                COHORT_SIZE + 1 - rank
            ) / (COHORT_SIZE + 1),
        }
    }
    for label, cohort in (
        ("test", "independent_test_null"),
        ("positive", "positive"),
    ):
        hits = sum(value > threshold for value in groups[cohort])
        result[label] = {
            "trials": COHORT_SIZE,
            "strict_exceedances": hits,
            "rate": hits / COHORT_SIZE,
            "one_sided_cp_confidence": 0.95,
            "one_sided_cp_upper": cp_upper(hits, COHORT_SIZE),
            "one_sided_cp_lower": (
                1 - cp_upper(COHORT_SIZE - hits, COHORT_SIZE)
            ),
        }
    return result


def paths_for(root, output_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output = Path(output).resolve()
    if (
        not output_dir.is_relative_to(root)
        or not output.is_relative_to(output_dir)
    ):
        raise ValueError("audit paths must remain inside experiment root")
    p0_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
    )
    v17_dir = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    smoke_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_smoke_design1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "runner": (
            root
            / "scripts/run_semstamp_certified_defense_v2_wp3_full_replay.py"
        ),
        "auditor": Path(__file__).resolve(),
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "resume_authorization": output_dir / "RESUME_AUTHORIZATION.json",
        "result": output_dir / "FULL_REPLAY_RESULT.json",
        "rows0": output_dir / "FULL_REPLAY_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "FULL_REPLAY_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/FULL_REPLAY_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "lock": output_dir / "EXECUTION.lock",
        "audit": output,
        "p0_rows": p0_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "v17_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "v17_result": v17_dir / "B0_RESULT.json",
        "smoke_final": smoke_dir / "FINAL_RESULT.json",
        "smoke_audit": smoke_dir / "EXECUTION_AUDIT.json",
        "smoke_contract": smoke_dir / "CONTRACT.json",
        "smoke_manifest": smoke_dir / "INPUT_MANIFEST.sha256",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


def verify_execution_lock(path):
    path = Path(path)
    if path.stat().st_size != 0:
        raise ValueError("execution lock file is not empty")
    if f"{stat.S_IMODE(path.stat().st_mode):04o}" != "0600":
        raise ValueError("execution lock mode mismatch")
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("full-replay process still holds lock") from error
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_rows(paths):
    if hashlib.sha256(RELATION).hexdigest() != RELATION_SHA256:
        raise ValueError("independent relation bytes mismatch")
    p0_rows = read_jsonl(paths["p0_rows"])
    references = read_jsonl(paths["v17_rows"])
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    checkpoint = read_json(paths["checkpoint"])
    reject_raw_public_keys([
        p0_rows, references, rows0, rows1, checkpoint
    ])
    if (
        len(p0_rows) != TOTAL_ROWS
        or len(references) != TOTAL_ROWS
        or len(rows0) != TOTAL_ROWS
        or rows0 != rows1
        or jsonl_bytes(rows0) != paths["rows0"].read_bytes()
        or paths["rows0"].read_bytes() != paths["rows1"].read_bytes()
    ):
        raise ValueError("full-replay public rows incomplete or noncanonical")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != STAGE
        or checkpoint.get("phase") != "full_replay"
        or checkpoint.get("protocol_sha256") != PROTOCOL_SHA256
        or checkpoint.get("contract_sha256") != CONTRACT_SHA256
        or checkpoint.get("authorization_sha256") != AUTHORIZATION_SHA256
        or checkpoint.get("p0_rows_sha256") != P0_ROWS_SHA256
        or checkpoint.get("v17_rows_sha256") != V17_ROWS_SHA256
        or set(checkpoint.get("replays", {})) != {"0", "1"}
        or checkpoint["replays"]["0"] != checkpoint["replays"]["1"]
        or any(
            len(checkpoint["replays"][str(replay)]) != TOTAL_ROWS
            for replay in range(REPLAY_COUNT)
        )
    ):
        raise ValueError("sealed checkpoint incomplete or drifted")
    attempts = checkpoint.get("attempts")
    expected_attempt = {
        "attempt_index": 0,
        "completed_after_replay": [TOTAL_ROWS, TOTAL_ROWS],
        "completed_before_model_load": [0, 0],
        "model_load_attempted": True,
        "model_load_completed": True,
        "outcome": "completed",
        "resume_authorization_sha256": None,
        "resumed": False,
    }
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or any(
            attempts[0].get(key) != value
            for key, value in expected_attempt.items()
        )
        or "started_at" not in attempts[0]
        or "completed_at" not in attempts[0]
    ):
        raise ValueError("attempt ledger mismatch")

    for index, (public, parent, reference) in enumerate(
        zip(rows0, p0_rows, references)
    ):
        entry = checkpoint["replays"]["0"][index]
        if entry.get("public") != public:
            raise ValueError(f"checkpoint/public mismatch at {index}")
        regions = entry.get("regions")
        if (
            not isinstance(regions, list)
            or len(regions) != public["unit_count"]
            or any(
                type(value) is not int or not 0 <= value <= 7
                for value in regions
            )
        ):
            raise ValueError(f"invalid regions at {index}")
        hits = [
            RELATION[previous * 8 + current]
            for previous, current in zip(regions, regions[1:])
        ]
        expected_score = score_from_counts(sum(hits), len(hits))
        if (
            public["plan_index"] != index
            or public["full_replay_index"] != index
            or public["cohort"] != COHORTS[index // COHORT_SIZE]
            or public["cohort_index"] != index % COHORT_SIZE
            or public["regions_sha256"]
            != hashlib.sha256(bytes(regions)).hexdigest()
            or public["hits_sha256"]
            != hashlib.sha256(bytes(hits)).hexdigest()
            or public["edge_count"] != len(hits)
            or public["hit_count"] != sum(hits)
            or public["direct_score"] != expected_score
            or public["official_score"] != expected_score
            or public["score_abs_error"] != 0.0
            or not math.isfinite(expected_score)
        ):
            raise ValueError(f"independent score mismatch at {index}")
        if any(public.get(key) != value for key, value in reference.items()):
            raise ValueError(f"v17 reference mismatch at {index}")
        structural = {
            "atom_count": parent["atom_count"],
            "byte_reconstruction_sha256": (
                parent["byte_reconstruction_sha256"]
            ),
            "legality_sha256": parent["legality_sha256"],
            "refinement_count": parent["refinement_count"],
            "topology_sha256": parent["topology_sha256"],
            "zero_byte_gap_count": parent["zero_byte_gap_count"],
        }
        if (
            any(public.get(key) != value for key, value in structural.items())
            or public["p0_row_sha256"]
            != hashlib.sha256(canonical_bytes(parent)).hexdigest()
            or public["renderer_contract"]
            != "refined_nltk_boundary_atoms_v2"
            or public["stage"] != STAGE
            or public["v17_reference_exact"] is not True
        ):
            raise ValueError(f"P0 structure mismatch at {index}")
    risk_rows = sum(
        bool(row["refinement_count"] or row["zero_byte_gap_count"])
        for row in rows0
    )
    zero_gaps = sum(row["zero_byte_gap_count"] for row in rows0)
    if (
        Counter(row["cohort"] for row in rows0)
        != {cohort: COHORT_SIZE for cohort in COHORTS}
        or risk_rows != EXPECTED_RISK_ROWS
        or zero_gaps != EXPECTED_ZERO_GAPS
    ):
        raise ValueError("full denominator coverage mismatch")
    return rows0, checkpoint


def verify_result(paths, contract, authorization, rows):
    result = read_json(paths["result"])
    if canonical_bytes(result) != paths["result"].read_bytes():
        raise ValueError("full-replay result is not canonical")
    statistics = independently_summarize(rows)
    v17_result = read_json(paths["v17_result"])
    expected_counts = {
        "completed_model_loads": 1,
        "direct_input_bindings": 11,
        "direct_score_computations": 6000,
        "inherited_input_bindings": 46,
        "mismatches": 0,
        "model_load_attempts": 1,
        "official_detect_lsh_calls": 6000,
        "p0_structure_comparisons": 3000,
        "process_attempts": 1,
        "replays": 2,
        "rows_per_replay": 3000,
        "sample_exclusions": 0,
        "sample_supplements": 0,
        "v17_reference_comparisons": 6000,
    }
    if (
        result.get("schema_version") != 1
        or result.get("stage") != STAGE
        or result.get("status") != "go"
        or result.get("terminal") != "WP3_FULL_REPLAY_GO_FOR_WP4_DESIGN"
        or result.get("protocol_sha256") != PROTOCOL_SHA256
        or result.get("contract_sha256") != CONTRACT_SHA256
        or result.get("authorization_sha256") != AUTHORIZATION_SHA256
        or result.get("runner_sha256") != RUNNER_SHA256
        or result.get("p0_rows_sha256") != P0_ROWS_SHA256
        or result.get("v17_rows_sha256") != V17_ROWS_SHA256
        or result.get("smoke_final_sha256") != SMOKE_FINAL_SHA256
        or not result.get("gates")
        or not all(result["gates"].values())
        or result.get("counts") != expected_counts
        or result.get("statistics") != statistics
        or statistics != v17_result["statistics"]
        or result.get("artifacts", {}).get("authorization") != authorization
        or result.get("artifacts", {}).get("inputs") != contract["inputs"]
        or result.get("artifacts", {}).get("deterministic_environment")
        != EXPECTED_ENVIRONMENT
        or result.get("contains_raw_text") is not False
    ):
        raise ValueError("full-replay result terminal mismatch")
    outputs = result.get("outputs", {})
    expected_outputs = {
        "replay0": (paths["rows0"], REPLAY_SHA256, 3554450),
        "replay1": (paths["rows1"], REPLAY_SHA256, 3554450),
        "checkpoint": (paths["checkpoint"], CHECKPOINT_SHA256, 7468550),
    }
    for label, (path, digest, size) in expected_outputs.items():
        value = outputs.get(label, {})
        if (
            value.get("sha256") != digest
            or value.get("bytes") != size
            or sha256_file(path) != digest
            or path.stat().st_size != size
        ):
            raise ValueError(f"output binding mismatch: {label}")
    runtime = result.get("runtime", {})
    attestation = result.get("execution_attestation", {})
    decision = result.get("decision", {})
    if (
        runtime.get("device_name") != "NVIDIA GeForce RTX 3090"
        or runtime.get("nltk") != "3.8.1"
        or runtime.get("peak_vram_fraction", 1.0) >= 0.8
        or runtime.get("detector_elapsed_seconds", 1801) > 1800
        or attestation.get("detector_imports_in_terminal_invocation") != 1
        or attestation.get("detector_model_loads_in_terminal_invocation") != 1
        or attestation.get("detector_scores_computed") != 6000
        or attestation.get("gpu_calls") != 6000
        or attestation.get("detector_score_reads") != 0
        or attestation.get("certificate_calls") != 0
        or attestation.get("generation_calls") != 0
        or attestation.get("wp4_started") is not False
        or decision.get("detector_health") != "go"
        or decision.get("wp4") != "authorized_to_freeze_design_only"
        or decision.get("wp4_execution_authorized") is not False
        or decision.get("certificate_authorized") is not False
    ):
        raise ValueError("runtime or authorization boundary mismatch")
    reject_raw_public_keys(result)
    return result, statistics


def audit(root, output_dir, output):
    paths = paths_for(root, output_dir, output)
    fixed = {
        "protocol": (PROTOCOL_SHA256, "0644"),
        "contract": (CONTRACT_SHA256, "0644"),
        "manifest": (MANIFEST_SHA256, "0644"),
        "runner": (RUNNER_SHA256, "0644"),
        "authorization": (AUTHORIZATION_SHA256, "0600"),
        "result": (RESULT_SHA256, "0600"),
        "rows0": (REPLAY_SHA256, "0600"),
        "rows1": (REPLAY_SHA256, "0600"),
        "checkpoint": (CHECKPOINT_SHA256, "0600"),
        "p0_rows": (P0_ROWS_SHA256, "0600"),
        "v17_rows": (V17_ROWS_SHA256, "0600"),
        "v17_result": (V17_RESULT_SHA256, "0600"),
        "smoke_final": (SMOKE_FINAL_SHA256, "0600"),
        "smoke_audit": (SMOKE_AUDIT_SHA256, "0600"),
        "smoke_manifest": (SMOKE_MANIFEST_SHA256, "0644"),
    }
    for label, (digest, mode) in fixed.items():
        verify_file(paths[label], digest, mode)
    if (
        paths["heartbeat"].exists()
        or paths["last_failure"].exists()
        or paths["resume_authorization"].exists()
    ):
        raise ValueError("active or unauthorized resume residue exists")
    verify_execution_lock(paths["lock"])

    contract = read_json(paths["contract"])
    authorization = read_json(paths["authorization"])
    if (
        contract.get("status") != "frozen_design"
        or contract.get("execution_authorized") is not False
        or authorization.get("status")
        != "FULL_REPLAY_EXECUTION_AUTHORIZED"
        or authorization.get("scope")
        != "3000_row_detector_full_replay_only"
        or authorization.get("authorized_replays") != REPLAY_COUNT
        or authorization.get("authorized_rows_per_replay") != TOTAL_ROWS
        or authorization.get("deterministic_environment")
        != EXPECTED_ENVIRONMENT
        or authorization.get("wp4_authorized") is not False
        or authorization.get("certificate_authorized") is not False
    ):
        raise ValueError("design or authorization scope mismatch")
    direct_manifest = verify_manifest(
        paths["root"], paths["manifest"], contract
    )
    smoke_contract = read_json(paths["smoke_contract"])
    inherited_manifest = verify_manifest(
        paths["root"], paths["smoke_manifest"], smoke_contract
    )
    if len(direct_manifest) != 11 or len(inherited_manifest) != 46:
        raise ValueError("input manifest counts mismatch")
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")

    rows, checkpoint = verify_rows(paths)
    result, statistics = verify_result(
        paths, contract, authorization, rows
    )
    checks = {
        "frozen_design_and_authorization_exact": True,
        "all_57_direct_and_inherited_inputs_exact": True,
        "repository_revisions_exact": True,
        "successful_result_terminal_go": True,
        "all_runner_gates_pass": True,
        "full_3000_row_denominator_exact": True,
        "balanced_1000_per_cohort": True,
        "all_42_risk_rows_retained": True,
        "all_59_zero_gaps_retained": True,
        "two_public_replays_byte_exact": True,
        "two_sealed_region_replays_exact": True,
        "all_6000_region_hashes_recomputed": True,
        "all_6000_hit_hashes_recomputed": True,
        "all_6000_direct_scores_recomputed": True,
        "all_6000_official_direct_scores_exact": True,
        "all_6000_v17_references_exact": True,
        "all_3000_p0_structures_exact": True,
        "calibration_threshold_independently_recomputed": True,
        "test_fpr_and_positive_tpr_independently_recomputed": True,
        "v17_statistics_exact": True,
        "no_nonfinite_scores": True,
        "no_sample_exclusion_or_supplement": True,
        "single_process_attempt_and_model_load": True,
        "no_resume_authorization_or_restart": True,
        "runtime_and_vram_below_limits": True,
        "active_failure_heartbeat_and_process_lock_absent": True,
        "wp4_certificate_generation_not_started": True,
        "raw_text_absent_from_public_artifacts": True,
        "auditor_model_gpu_score_calls_zero": True,
    }
    audit_result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go" if all(checks.values()) else "stop",
        "terminal": "WP3_FULL_REPLAY_GO_FOR_WP4_DESIGN_CONFIRMED",
        "scope": (
            "independent_structural_score_and_statistics_recalculation_"
            "without_model_load"
        ),
        "bindings": {
            **{label: value[0] for label, value in fixed.items()},
            "auditor": sha256_file(paths["auditor"]),
        },
        "checks": checks,
        "counts": {
            "direct_input_manifest_entries": len(direct_manifest),
            "inherited_input_manifest_entries": len(inherited_manifest),
            "process_attempts": len(checkpoint["attempts"]),
            "resume_attempts": sum(
                attempt["resumed"] for attempt in checkpoint["attempts"]
            ),
            "model_load_attempts": sum(
                attempt["model_load_attempted"]
                for attempt in checkpoint["attempts"]
            ),
            "completed_model_loads": sum(
                attempt["model_load_completed"]
                for attempt in checkpoint["attempts"]
            ),
            "rows_per_replay": len(rows),
            "replays": REPLAY_COUNT,
            "independently_recomputed_scores": REPLAY_COUNT * len(rows),
            "v17_reference_comparisons": REPLAY_COUNT * len(rows),
            "p0_structure_comparisons": len(rows),
            "mandatory_risk_rows": EXPECTED_RISK_ROWS,
            "zero_byte_gaps": EXPECTED_ZERO_GAPS,
        },
        "statistics": statistics,
        "runtime": result["runtime"],
        "decision": {
            "detector_health": "confirmed_go",
            "wp4": "authorized_to_freeze_design_only",
            "wp4_execution_authorized": False,
            "certificate_authorized": False,
        },
        "audit_execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
            "gpu_calls": 0,
            "certificate_calls": 0,
            "generation_calls": 0,
            "wp4_calls": 0,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.output)


if __name__ == "__main__":
    main()
