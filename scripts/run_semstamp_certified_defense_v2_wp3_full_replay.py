"""Run the separately authorized v2 WP3 3,000-row detector replay."""

import argparse
from collections import Counter
import fcntl
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import time
from datetime import datetime, timezone

import run_semstamp_certified_defense_v2_wp3_smoke as smoke


STAGE = "semstamp_certified_defense_v2_wp3_detector_full_replay"
PROTOCOL_SHA256 = (
    "c4102127400bf1b20c7aa9a9849252bc7f737e8b628e67699bb4269213fb0d91"
)
SMOKE_FINAL_SHA256 = (
    "64e509330781183057c7c479dacc3ae5862a39923f18dbfa3401c53236baa637"
)
SMOKE_RESULT_SHA256 = (
    "b604c5ce22be68a1fb08d0774b39f37aa19007c5d5d9b9bc998594aa696b0fe8"
)
SMOKE_AUDIT_SHA256 = (
    "163933e86d171fe5f289441580606d7bce66cfe6867baf59553438579f9badf5"
)
SMOKE_REPLAY_SHA256 = (
    "a4b7403bae9095be1a5235b447a57bfc722f19c6c20556c6aae0df867dc2cc7b"
)
SMOKE_INPUT_MANIFEST_SHA256 = (
    "326f36d14e8dcb7c12ce019a2aa3668d67b30ad3f5a9284bbf4e4451fde69697"
)
V17_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
V17_ROWS_SHA256 = smoke.V17_ROWS_SHA256
P0_ROWS_SHA256 = smoke.P0_ROWS_SHA256
COHORTS = smoke.COHORTS
COHORT_SIZE = smoke.COHORT_SIZE
TOTAL_ROWS = smoke.TOTAL_ROWS
REPLAY_COUNT = smoke.REPLAY_COUNT
BATCH_SIZE = smoke.BATCH_SIZE
NATIVE_TOKEN_LIMIT = smoke.NATIVE_TOKEN_LIMIT
MIN_FREE_GIB = 5.0
MAX_VRAM_FRACTION = 0.80
MAX_GPU_SECONDS = 1800
EXPECTED_MANDATORY_ROWS = smoke.EXPECTED_MANDATORY_ROWS
EXPECTED_ZERO_GAPS = smoke.EXPECTED_ZERO_GAPS
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
FORBIDDEN_PUBLIC_KEYS = smoke.FORBIDDEN_PUBLIC_KEYS


class ReplayStop(RuntimeError):
    """A deterministic detector/reference/replay failure."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside root")
    base = smoke.paths_for(root, output_dir)
    smoke_dir = (
        root
        / "experiments/semstamp_certified_defense_v2_wp3_detector_smoke_design1"
    )
    base.update({
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "resume_authorization": output_dir / "RESUME_AUTHORIZATION.json",
        "rows0": output_dir / "FULL_REPLAY_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "FULL_REPLAY_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/FULL_REPLAY_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "result": output_dir / "FULL_REPLAY_RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "lock": output_dir / "EXECUTION.lock",
        "runner": Path(__file__).resolve(),
        "test": (
            root
            / "scripts/test_semstamp_certified_defense_v2_wp3_full_replay.py"
        ),
        "auditor": (
            root
            / "scripts/audit_semstamp_certified_defense_v2_wp3_full_replay_design.py"
        ),
        "smoke_runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_smoke.py"
        ),
        "smoke_dir": smoke_dir,
        "smoke_contract": smoke_dir / "CONTRACT.json",
        "smoke_manifest": smoke_dir / "INPUT_MANIFEST.sha256",
        "smoke_final": smoke_dir / "FINAL_RESULT.json",
        "smoke_result": smoke_dir / "SMOKE_RESULT.json",
        "smoke_audit": smoke_dir / "EXECUTION_AUDIT.json",
        "v17_result": base["v17_dir"] / "B0_RESULT.json",
    })
    return base


def validate_contract(paths):
    contract = smoke.read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen_design",
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "rows_per_replay": TOTAL_ROWS,
        "replay_count": REPLAY_COUNT,
        "batch_size": BATCH_SIZE,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "maximum_vram_fraction": MAX_VRAM_FRACTION,
        "maximum_gpu_seconds": MAX_GPU_SECONDS,
        "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
        "semstamp_revision": smoke.SEMSTAMP_REVISION,
        "semstamp_data_revision": smoke.SEMSTAMP_DATA_REVISION,
        "model_revision": smoke.MODEL_REVISION,
        "relation_sha256": smoke.RELATION_SHA256,
        "punkt_payload_sha256": smoke.PUNKT_SHA256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "v17_rows_sha256": V17_ROWS_SHA256,
        "v17_result_sha256": V17_RESULT_SHA256,
        "smoke_final_sha256": SMOKE_FINAL_SHA256,
        "smoke_result_sha256": SMOKE_RESULT_SHA256,
        "smoke_audit_sha256": SMOKE_AUDIT_SHA256,
        "smoke_replay_sha256": SMOKE_REPLAY_SHA256,
        "inherited_input_manifest_sha256": SMOKE_INPUT_MANIFEST_SHA256,
        "inherited_input_count": 46,
        "execution_authorized": False,
        "resume_authorized": False,
        "wp4_authorized": False,
        "certificate_authorized": False,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    artifact_paths = {
        "runner_sha256": paths["runner"],
        "smoke_runner_sha256": paths["smoke_runner"],
        "renderer_sha256": paths["renderer"],
        "p0_runner_sha256": paths["p0_runner"],
        "test_sha256": paths["test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["manifest"],
    }
    for field, path in artifact_paths.items():
        if contract.get(field) != smoke.sha256_file(path):
            raise ValueError(f"contract artifact mismatch: {field}")
    manifest = smoke.parse_manifest(paths["manifest"])
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(manifest):
        raise ValueError("contract/input manifest path set mismatch")
    for relative, expected_sha in manifest.items():
        value = inputs[relative]
        if (
            not isinstance(value, dict)
            or value.get("sha256") != expected_sha
            or type(value.get("bytes")) is not int
            or value["bytes"] < 1
        ):
            raise ValueError(f"contract input metadata mismatch: {relative}")
    return contract, manifest


def validate_authorization(paths, contract):
    if not paths["authorization"].is_file():
        raise PermissionError(
            "full replay is not authorized: "
            "EXECUTION_AUTHORIZATION.json is absent"
        )
    authorization = smoke.read_json(paths["authorization"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "FULL_REPLAY_EXECUTION_AUTHORIZED",
        "scope": "3000_row_detector_full_replay_only",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": smoke.sha256_file(paths["contract"]),
        "runner_sha256": smoke.sha256_file(paths["runner"]),
        "smoke_final_sha256": SMOKE_FINAL_SHA256,
        "authorized_replays": REPLAY_COUNT,
        "authorized_rows_per_replay": TOTAL_ROWS,
        "deterministic_environment": DETERMINISTIC_ENVIRONMENT,
        "wp4_authorized": False,
        "certificate_authorized": False,
    }
    if set(authorization) != set(required) or any(
        authorization.get(key) != value
        for key, value in required.items()
    ):
        raise PermissionError("full-replay execution authorization is invalid")
    if (
        contract["execution_authorized"] is not False
        or contract["resume_authorized"] is not False
    ):
        raise ValueError("design contract authorization state drift")
    return authorization


def validate_pre_model_environment():
    observed = {
        key: os.environ.get(key) for key in DETERMINISTIC_ENVIRONMENT
    }
    if observed != DETERMINISTIC_ENVIRONMENT:
        raise RuntimeError(
            "deterministic/offline environment mismatch before model import"
        )
    return observed


def verify_inputs(paths, contract, manifest):
    observed = {}
    for relative, expected_sha in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        value = contract["inputs"][relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
            or smoke.sha256_file(path) != expected_sha
            or path.stat().st_size != value["bytes"]
        ):
            raise ValueError(f"input bytes/SHA mismatch: {relative}")
        expected_mode = value.get("mode")
        if (
            expected_mode is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
        ):
            raise ValueError(f"input mode mismatch: {relative}")
        observed[relative] = {
            "bytes": value["bytes"],
            "sha256": expected_sha,
            **({"mode": expected_mode} if expected_mode else {}),
        }
    if smoke.git_head(paths["repo"]) != smoke.SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if smoke.git_head(paths["data_repo"]) != smoke.SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    free_gib = shutil.disk_usage(paths["output_dir"]).free / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        raise ValueError("insufficient free disk for full replay")
    return observed, free_gib


def verify_inherited_smoke_inputs(paths):
    smoke_paths = smoke.paths_for(paths["root"], paths["smoke_dir"])
    inherited_contract, inherited_manifest = smoke.validate_contract(
        smoke_paths
    )
    observed, _ = smoke.verify_inputs(
        smoke_paths,
        inherited_contract,
        inherited_manifest,
    )
    if (
        smoke.sha256_file(paths["smoke_manifest"])
        != SMOKE_INPUT_MANIFEST_SHA256
        or len(inherited_manifest) != 46
    ):
        raise ValueError("inherited smoke input manifest drift")
    return observed


def validate_smoke_parent(paths):
    expected = {
        paths["smoke_final"]: SMOKE_FINAL_SHA256,
        paths["smoke_result"]: SMOKE_RESULT_SHA256,
        paths["smoke_audit"]: SMOKE_AUDIT_SHA256,
        paths["p0_rows"]: P0_ROWS_SHA256,
        paths["v17_rows"]: V17_ROWS_SHA256,
        paths["v17_result"]: V17_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if smoke.sha256_file(path) != digest:
            raise ValueError(f"frozen parent drift: {path.name}")
    final = smoke.read_json(paths["smoke_final"])
    result = smoke.read_json(paths["smoke_result"])
    audit = smoke.read_json(paths["smoke_audit"])
    v17_result = smoke.read_json(paths["v17_result"])
    if (
        final.get("status") != "go"
        or final.get("terminal") != "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN"
        or final.get("decision", {}).get("full_replay")
        != "authorized_to_freeze_design_only"
        or final.get("decision", {}).get("full_replay_execution_authorized")
        is not False
        or result.get("status") != "go"
        or result.get("terminal") != "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN"
        or not all(result.get("gates", {}).values())
        or audit.get("status") != "go"
        or audit.get("terminal")
        != "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN_CONFIRMED"
        or not all(audit.get("checks", {}).values())
        or v17_result.get("status") != "go"
        or v17_result.get("terminal_reason") != "b0_detector_health_go"
    ):
        raise ValueError("smoke/v17 parent is not the frozen audited Go")
    return v17_result


def validate_full_denominator(paths):
    renderer_rows = smoke.read_jsonl(paths["p0_rows"])
    reference_rows = smoke.read_jsonl(paths["v17_rows"])
    frozen_plan = smoke.read_jsonl(paths["v17_plan"])
    if (
        len(renderer_rows) != TOTAL_ROWS
        or len(reference_rows) != TOTAL_ROWS
        or len(frozen_plan) != TOTAL_ROWS
        or Counter(row["cohort"] for row in renderer_rows)
        != {cohort: COHORT_SIZE for cohort in COHORTS}
    ):
        raise ValueError("full replay denominator count mismatch")
    for index, (renderer, reference, planned) in enumerate(
        zip(renderer_rows, reference_rows, frozen_plan)
    ):
        identity = {
            "plan_index": index,
            "cohort": COHORTS[index // COHORT_SIZE],
            "cohort_index": index % COHORT_SIZE,
        }
        if any(renderer.get(key) != value for key, value in identity.items()):
            raise ValueError(f"P0 denominator order drift at {index}")
        if any(reference.get(key) != value for key, value in identity.items()):
            raise ValueError(f"v17 denominator order drift at {index}")
        if any(planned.get(key) != value for key, value in identity.items()):
            raise ValueError(f"v17 plan order drift at {index}")
        if (
            renderer.get("raw_text_sha256") != planned["raw_text_sha256"]
            or reference.get("raw_text_sha256") != planned["raw_text_sha256"]
            or renderer.get("unit_count") != planned["unit_count"]
            or reference.get("unit_count") != planned["unit_count"]
        ):
            raise ValueError(f"full denominator identity drift at {index}")
    risk_rows = sum(
        bool(row["refinement_count"] or row["zero_byte_gap_count"])
        for row in renderer_rows
    )
    zero_gaps = sum(row["zero_byte_gap_count"] for row in renderer_rows)
    if (
        risk_rows != EXPECTED_MANDATORY_ROWS
        or zero_gaps != EXPECTED_ZERO_GAPS
    ):
        raise ValueError("full denominator risk coverage drift")
    return renderer_rows, reference_rows, frozen_plan


def prepare_documents(paths, renderer_rows, frozen_plan, nltk_data):
    b0_paths = smoke.v17.paths_for(paths["root"], paths["v17_dir"])
    _, plan, texts = smoke.v17.materialize_inputs(b0_paths)
    if (
        len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
        or smoke.v17.jsonl_bytes(plan) != paths["v17_plan"].read_bytes()
        or smoke.canonical_bytes(plan) != smoke.canonical_bytes(frozen_plan)
    ):
        raise ValueError("materialized inputs differ from frozen v17 plan")
    parent_rows = smoke.read_jsonl(paths["wp1_rows"])
    legacy_indices = tuple(
        smoke.read_json(paths["v1_readiness"])["renderer"][
            "failed_plan_indices"
        ]
    )
    if len(parent_rows) != TOTAL_ROWS or len(legacy_indices) != 42:
        raise ValueError("frozen WP1 parent evidence mismatch")
    sent_tokenize = smoke.load_nltk(nltk_data)
    documents = []
    for index, text in enumerate(texts):
        units = sent_tokenize(text)
        document = smoke.canonicalize_refined_bytes(
            text.encode("utf-8"),
            lambda _, frozen=units: frozen,
        )
        observed = smoke.p0.p0_row(
            index,
            plan[index],
            document,
            legacy_indices,
            parent_rows[index],
        )
        if smoke.canonical_bytes(observed) != smoke.canonical_bytes(
            renderer_rows[index]
        ):
            raise ValueError(f"refined P0 row drift at {index}")
        documents.append(document)
    return plan, documents


def acquire_execution_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    os.chmod(path, 0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("another full-replay process holds the lock")
    return handle


def checkpoint_completed(checkpoint):
    return [
        len(checkpoint["replays"]["0"]),
        len(checkpoint["replays"]["1"]),
    ]


def build_public_row(planned, renderer_row, regions, relation, official_score):
    base = smoke.support.make_public_score(
        planned, regions, relation, official_score
    )
    row = {
        **base,
        "stage": STAGE,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "full_replay_index": planned["plan_index"],
        "atom_count": renderer_row["atom_count"],
        "refinement_count": renderer_row["refinement_count"],
        "zero_byte_gap_count": renderer_row["zero_byte_gap_count"],
        "topology_sha256": renderer_row["topology_sha256"],
        "legality_sha256": renderer_row["legality_sha256"],
        "byte_reconstruction_sha256": (
            renderer_row["byte_reconstruction_sha256"]
        ),
        "p0_row_sha256": smoke.sha256_bytes(
            smoke.canonical_bytes(renderer_row)
        ),
        "v17_reference_exact": True,
        "contains_raw_text": False,
    }
    smoke.reject_raw_public_keys(row)
    return base, row


def validate_checkpoint(
    checkpoint,
    contract_sha256,
    authorization_sha256,
    plan,
    renderer_rows,
    reference_rows,
    relation=None,
):
    required_header = {
        "schema_version": 1,
        "stage": STAGE,
        "phase": "full_replay",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": contract_sha256,
        "authorization_sha256": authorization_sha256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "v17_rows_sha256": V17_ROWS_SHA256,
    }
    if any(checkpoint.get(key) != value for key, value in required_header.items()):
        raise ValueError("invalid full-replay checkpoint header")
    if (
        set(checkpoint.get("replays", {})) != {"0", "1"}
        or not isinstance(checkpoint.get("attempts"), list)
    ):
        raise ValueError("invalid full-replay checkpoint structure")
    zero = checkpoint["replays"]["0"]
    one = checkpoint["replays"]["1"]
    if (
        not isinstance(zero, list)
        or not isinstance(one, list)
        or len(zero) > TOTAL_ROWS
        or len(one) > TOTAL_ROWS
        or (one and len(zero) != TOTAL_ROWS)
    ):
        raise ValueError("invalid full-replay checkpoint prefix")
    for attempt_index, attempt in enumerate(checkpoint["attempts"]):
        if (
            attempt.get("attempt_index") != attempt_index
            or type(attempt.get("model_load_attempted")) is not bool
            or type(attempt.get("model_load_completed")) is not bool
            or (
                attempt["model_load_completed"]
                and not attempt["model_load_attempted"]
            )
        ):
            raise ValueError("invalid checkpoint attempt ledger")
    for replay, entries in enumerate((zero, one)):
        for index, entry in enumerate(entries):
            public = entry.get("public", {})
            regions = entry.get("regions")
            if (
                set(entry) != {"public", "regions"}
                or public.get("plan_index") != index
                or public.get("full_replay_index") != index
                or public.get("v17_reference_exact") is not True
                or not isinstance(regions, list)
                or len(regions) != plan[index]["unit_count"]
                or any(type(value) is not int or not 0 <= value <= 7
                       for value in regions)
            ):
                raise ValueError("invalid full-replay checkpoint entry")
            if relation is not None:
                base, rebuilt = build_public_row(
                    plan[index],
                    renderer_rows[index],
                    regions,
                    relation,
                    public.get("official_score"),
                )
                if (
                    smoke.canonical_bytes(base)
                    != smoke.canonical_bytes(reference_rows[index])
                    or smoke.canonical_bytes(rebuilt)
                    != smoke.canonical_bytes(public)
                ):
                    raise ReplayStop(
                        f"checkpoint/reference drift at row {index}"
                    )
            if replay and smoke.canonical_bytes(entry) != smoke.canonical_bytes(
                zero[index]
            ):
                raise ReplayStop(f"checkpoint replay drift at row {index}")
    return {
        "completed": [len(zero), len(one)],
        "complete": len(one) == TOTAL_ROWS,
    }


def checkpoint_template(contract_sha256, authorization_sha256):
    return {
        "schema_version": 1,
        "stage": STAGE,
        "phase": "full_replay",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": contract_sha256,
        "authorization_sha256": authorization_sha256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "v17_rows_sha256": V17_ROWS_SHA256,
        "attempts": [],
        "replays": {"0": [], "1": []},
    }


def validate_resume_authorization(paths, checkpoint):
    attempts = checkpoint["attempts"]
    completed = checkpoint_completed(checkpoint)
    has_prior_attempt = bool(attempts)
    if not has_prior_attempt:
        if paths["resume_authorization"].exists():
            raise PermissionError(
                "resume authorization is forbidden for initial execution"
            )
        return None
    if completed == [TOTAL_ROWS, TOTAL_ROWS]:
        raise PermissionError("a completed full-replay checkpoint cannot resume")
    if not paths["resume_authorization"].is_file():
        raise PermissionError(
            "restart requires a separate RESUME_AUTHORIZATION.json"
        )
    authorization = smoke.read_json(paths["resume_authorization"])
    evidence_relative = authorization.get("evidence_relative_path")
    if evidence_relative not in {"HEARTBEAT.json", "LAST_FAILURE.json"}:
        raise PermissionError("resume evidence path is invalid")
    evidence_path = paths["output_dir"] / evidence_relative
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "FULL_REPLAY_RESUME_AUTHORIZED",
        "scope": "exact_checkpoint_prefix_only",
        "contract_sha256": smoke.sha256_file(paths["contract"]),
        "execution_authorization_sha256": smoke.sha256_file(
            paths["authorization"]
        ),
        "checkpoint_sha256": smoke.sha256_file(paths["checkpoint"]),
        "completed_prefix_lengths": completed,
        "next_attempt_index": len(attempts),
        "evidence_relative_path": evidence_relative,
        "evidence_sha256": (
            smoke.sha256_file(evidence_path)
            if evidence_path.is_file() else None
        ),
        "no_score_based_retry": True,
        "wp4_authorized": False,
        "certificate_authorized": False,
    }
    if (
        not evidence_path.is_file()
        or set(authorization) != set(required)
        or any(authorization.get(key) != value
               for key, value in required.items())
    ):
        raise PermissionError("full-replay resume authorization is invalid")
    return authorization


def load_or_create_checkpoint(paths, plan, renderer_rows, reference_rows):
    contract_sha256 = smoke.sha256_file(paths["contract"])
    authorization_sha256 = smoke.sha256_file(paths["authorization"])
    if paths["checkpoint"].is_file():
        checkpoint = smoke.read_json(paths["checkpoint"])
    else:
        checkpoint = checkpoint_template(
            contract_sha256, authorization_sha256
        )
        smoke.atomic_write_json(paths["checkpoint"], checkpoint)
    validate_checkpoint(
        checkpoint,
        contract_sha256,
        authorization_sha256,
        plan,
        renderer_rows,
        reference_rows,
    )
    resume = validate_resume_authorization(paths, checkpoint)
    if checkpoint["attempts"]:
        previous = checkpoint["attempts"][-1]
        if previous.get("outcome") == "running":
            previous["outcome"] = "interrupted_or_failed"
    attempt = {
        "attempt_index": len(checkpoint["attempts"]),
        "started_at": utc_now(),
        "resumed": resume is not None,
        "completed_before_model_load": checkpoint_completed(checkpoint),
        "resume_authorization_sha256": (
            smoke.sha256_file(paths["resume_authorization"])
            if resume is not None else None
        ),
        "model_load_attempted": True,
        "model_load_completed": False,
        "outcome": "running",
    }
    checkpoint["attempts"].append(attempt)
    smoke.atomic_write_json(paths["checkpoint"], checkpoint)
    return checkpoint, attempt


def write_heartbeat(
    paths,
    attempt_index,
    replay,
    completed,
    started_at,
    model_load_seconds,
):
    smoke.atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "phase": "full_replay",
        "attempt_index": attempt_index,
        "replay": replay,
        "completed_in_replay": completed,
        "completed_total": replay * TOTAL_ROWS + completed,
        "target_total": REPLAY_COUNT * TOTAL_ROWS,
        "started_at": started_at,
        "updated_at": utc_now(),
        "model_load_seconds": model_load_seconds,
        "contains_raw_text": False,
    })


def make_scored_row(
    planned,
    document,
    renderer_row,
    regions,
    relation,
    official_detect,
    reference,
):
    replay_model = smoke.ReplayLSHModel(regions)
    official_score = float(
        official_detect(
            list(document.original_units),
            replay_model,
            lmbd=0.25,
            lsh_dim=3,
        )
    )
    replay_model.assert_exhausted()
    base, row = build_public_row(
        planned, renderer_row, regions, relation, official_score
    )
    if smoke.canonical_bytes(base) != smoke.canonical_bytes(reference):
        raise ReplayStop(
            f"v17 reference mismatch at row {planned['plan_index']}"
        )
    return row


def execute_replays(
    paths,
    checkpoint,
    plan,
    documents,
    renderer_rows,
    reference_rows,
    detector,
):
    contract_sha256 = smoke.sha256_file(paths["contract"])
    authorization_sha256 = smoke.sha256_file(paths["authorization"])
    validate_checkpoint(
        checkpoint,
        contract_sha256,
        authorization_sha256,
        plan,
        renderer_rows,
        reference_rows,
        detector["relation"],
    )
    started_at = utc_now()
    started = time.monotonic()
    torch = detector["torch"]
    attempt_index = checkpoint["attempts"][-1]["attempt_index"]
    with torch.inference_mode():
        for replay in range(REPLAY_COUNT):
            entries = checkpoint["replays"][str(replay)]
            write_heartbeat(
                paths,
                attempt_index,
                replay,
                len(entries),
                started_at,
                detector["model_load_seconds"],
            )
            for index in range(len(entries), TOTAL_ROWS):
                observed = detector["model"].get_hash(
                    list(documents[index].original_units)
                )
                regions = [int(value) for value in observed]
                if (
                    len(regions) != plan[index]["unit_count"]
                    or any(value < 0 or value > 7 for value in regions)
                ):
                    raise ReplayStop(f"invalid regions at row {index}")
                public = make_scored_row(
                    plan[index],
                    documents[index],
                    renderer_rows[index],
                    regions,
                    detector["relation"],
                    detector["official_detect"],
                    reference_rows[index],
                )
                entry = {"public": public, "regions": regions}
                if replay and smoke.canonical_bytes(
                    entry
                ) != smoke.canonical_bytes(
                    checkpoint["replays"]["0"][index]
                ):
                    raise ReplayStop(f"full replay drift at row {index}")
                entries.append(entry)
                smoke.atomic_write_json(paths["checkpoint"], checkpoint)
                write_heartbeat(
                    paths,
                    attempt_index,
                    replay,
                    index + 1,
                    started_at,
                    detector["model_load_seconds"],
                )
    state = validate_checkpoint(
        checkpoint,
        contract_sha256,
        authorization_sha256,
        plan,
        renderer_rows,
        reference_rows,
        detector["relation"],
    )
    if not state["complete"]:
        raise AssertionError("both full replays did not complete")
    rows = [
        [entry["public"] for entry in checkpoint["replays"][str(replay)]]
        for replay in range(REPLAY_COUNT)
    ]
    encoded = [smoke.jsonl_bytes(value) for value in rows]
    if encoded[0] != encoded[1]:
        raise ReplayStop("public full replays differ")
    torch.cuda.synchronize()
    return {
        "rows": rows,
        "encoded": encoded,
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "total_cuda_bytes": torch.cuda.get_device_properties(0).total_memory,
        "device_name": torch.cuda.get_device_name(0),
    }


def summarize_rows(rows):
    return smoke.support.summarize_scores(*[
        [row["direct_score"] for row in rows if row["cohort"] == cohort]
        for cohort in COHORTS
    ])


def run_full_replay(args):
    paths = paths_for(args.root, args.output_dir)
    if paths["result"].exists():
        raise PermissionError("a terminal full-replay result already exists")
    contract, manifest = validate_contract(paths)
    authorization = validate_authorization(paths, contract)
    environment = validate_pre_model_environment()
    observed_inputs, free_gib = verify_inputs(paths, contract, manifest)
    inherited_inputs = verify_inherited_smoke_inputs(paths)
    v17_result = validate_smoke_parent(paths)
    renderer_rows, reference_rows, frozen_plan = validate_full_denominator(
        paths
    )
    plan, documents = prepare_documents(
        paths, renderer_rows, frozen_plan, args.nltk_data
    )
    lock_handle = acquire_execution_lock(paths["lock"])
    try:
        checkpoint, attempt = load_or_create_checkpoint(
            paths, plan, renderer_rows, reference_rows
        )
        detector = smoke.initialize_detector(paths, args.embedder_dir)
        attempt["model_load_completed"] = True
        smoke.atomic_write_json(paths["checkpoint"], checkpoint)
        executed = execute_replays(
            paths,
            checkpoint,
            plan,
            documents,
            renderer_rows,
            reference_rows,
            detector,
        )
        attempt["outcome"] = "completed"
        attempt["completed_at"] = utc_now()
        attempt["completed_after_replay"] = checkpoint_completed(checkpoint)
        smoke.atomic_write_json(paths["checkpoint"], checkpoint)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    smoke.atomic_write_bytes(paths["rows0"], executed["encoded"][0])
    smoke.atomic_write_bytes(paths["rows1"], executed["encoded"][1])
    statistics = summarize_rows(executed["rows"][0])
    peak_fraction = (
        executed["peak_cuda_bytes"] / executed["total_cuda_bytes"]
    )
    total_seconds = (
        executed["elapsed_seconds"] + detector["model_load_seconds"]
    )
    v17_statistics_exact = (
        smoke.canonical_bytes(statistics)
        == smoke.canonical_bytes(v17_result["statistics"])
    )
    gates = {
        "authorization_exact": True,
        "deterministic_offline_environment_exact": True,
        "contract_and_inputs_exact": True,
        "smoke_go_terminal_exact": True,
        "fixed_3000_ordered_rows": len(executed["rows"][0]) == TOTAL_ROWS,
        "balanced_1000_per_cohort": Counter(
            row["cohort"] for row in executed["rows"][0]
        ) == {cohort: COHORT_SIZE for cohort in COHORTS},
        "all_risk_rows_and_zero_gaps_retained": (
            sum(
                bool(row["refinement_count"] or row["zero_byte_gap_count"])
                for row in executed["rows"][0]
            ) == EXPECTED_MANDATORY_ROWS
            and sum(
                row["zero_byte_gap_count"] for row in executed["rows"][0]
            ) == EXPECTED_ZERO_GAPS
        ),
        "refined_renderer_exact": True,
        "official_direct_error_zero": all(
            row["score_abs_error"] == 0.0
            for row in executed["rows"][0]
        ),
        "v17_reference_exact": all(
            row["v17_reference_exact"] is True
            for row in executed["rows"][0]
        ),
        "v17_statistics_exact": v17_statistics_exact,
        "v17_health_gates_pass": (
            statistics["test"]["one_sided_cp_upper"] <= 0.08
            and statistics["positive"]["rate"] >= 0.80
            and statistics["positive"]["one_sided_cp_lower"] > 0.75
        ),
        "two_public_replays_exact": (
            executed["encoded"][0] == executed["encoded"][1]
        ),
        "two_sealed_region_replays_exact": True,
        "no_nonfinite_scores": all(
            math.isfinite(row["direct_score"])
            for row in executed["rows"][0]
        ),
        "no_exclusion_or_supplementation": True,
        "terminal_invocation_single_model_load": (
            attempt["model_load_attempted"] is True
            and attempt["model_load_completed"] is True
        ),
        "all_attempts_disclosed": all(
            item["attempt_index"] == index
            for index, item in enumerate(checkpoint["attempts"])
        ),
        "peak_vram_below_0_80": peak_fraction < MAX_VRAM_FRACTION,
        "runtime_below_1800_seconds": total_seconds <= MAX_GPU_SECONDS,
        "public_artifacts_contain_no_raw_text": True,
        "certificate_generation_wp4_not_started": True,
    }
    status = "go" if all(gates.values()) else "stop"
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "terminal": (
            "WP3_FULL_REPLAY_GO_FOR_WP4_DESIGN"
            if status == "go" else "WP3_FULL_REPLAY_STOP"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": smoke.sha256_file(paths["contract"]),
        "authorization_sha256": smoke.sha256_file(paths["authorization"]),
        "runner_sha256": smoke.sha256_file(paths["runner"]),
        "smoke_final_sha256": SMOKE_FINAL_SHA256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "v17_rows_sha256": V17_ROWS_SHA256,
        "gates": gates,
        "counts": {
            "rows_per_replay": TOTAL_ROWS,
            "replays": REPLAY_COUNT,
            "official_detect_lsh_calls": REPLAY_COUNT * TOTAL_ROWS,
            "direct_score_computations": REPLAY_COUNT * TOTAL_ROWS,
            "v17_reference_comparisons": REPLAY_COUNT * TOTAL_ROWS,
            "p0_structure_comparisons": TOTAL_ROWS,
            "process_attempts": len(checkpoint["attempts"]),
            "model_load_attempts": sum(
                item["model_load_attempted"]
                for item in checkpoint["attempts"]
            ),
            "completed_model_loads": sum(
                item["model_load_completed"]
                for item in checkpoint["attempts"]
            ),
            "sample_exclusions": 0,
            "sample_supplements": 0,
            "mismatches": 0,
            "direct_input_bindings": len(observed_inputs),
            "inherited_input_bindings": len(inherited_inputs),
        },
        "statistics": statistics,
        "outputs": {
            "replay0": {
                **smoke.file_metadata(paths["rows0"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "replay1": {
                **smoke.file_metadata(paths["rows1"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "checkpoint": {
                **smoke.file_metadata(paths["checkpoint"], paths["root"]),
                "mode": "0600",
                "contains_raw_text": False,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "nltk": metadata.version("nltk"),
            "started_at": executed["started_at"],
            "finished_at": utc_now(),
            "detector_elapsed_seconds": total_seconds,
            "replay_elapsed_seconds": executed["elapsed_seconds"],
            "model_load_seconds": detector["model_load_seconds"],
            "device_name": executed["device_name"],
            "peak_cuda_bytes": executed["peak_cuda_bytes"],
            "total_cuda_bytes": executed["total_cuda_bytes"],
            "peak_vram_fraction": peak_fraction,
            "free_disk_gib_before_execution": free_gib,
        },
        "artifacts": {
            "inputs": observed_inputs,
            "inherited_inputs": inherited_inputs,
            "authorization": authorization,
            "deterministic_environment": environment,
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports_in_terminal_invocation": 1,
            "detector_model_loads_in_terminal_invocation": 1,
            "detector_score_reads": 0,
            "detector_scores_computed": REPLAY_COUNT * TOTAL_ROWS,
            "generation_calls": 0,
            "gpu_calls": REPLAY_COUNT * TOTAL_ROWS,
            "raw_text_emitted_to_public_artifacts": False,
            "wp4_started": False,
        },
        "decision": {
            "detector_health": status,
            "wp4": (
                "authorized_to_freeze_design_only"
                if status == "go" else "not_authorized"
            ),
            "wp4_execution_authorized": False,
            "certificate_authorized": False,
        },
        "contains_raw_text": False,
    }
    smoke.reject_raw_public_keys(result)
    smoke.atomic_write_json(paths["result"], result)
    paths["heartbeat"].unlink(missing_ok=True)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))
    if status != "go":
        raise ReplayStop("WP3 full detector replay stopped")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("full_replay",), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    try:
        run_full_replay(args)
    except Exception as error:
        if not paths["result"].is_file():
            smoke.atomic_write_json(paths["last_failure"], {
                "schema_version": 1,
                "stage": STAGE,
                "status": "invalid_or_incomplete",
                "terminal": "WP3_FULL_REPLAY_INVALID_OR_INCOMPLETE",
                "phase": args.phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "contains_raw_text": False,
            })
        raise


if __name__ == "__main__":
    main()
