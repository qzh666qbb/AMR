"""Run the separately authorized v2 WP3 96-row detector smoke."""

import argparse
from collections import Counter
from importlib import metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import certified_resegmentation_wp2_v13_b0_support as support
import run_certified_resegmentation_wp2_v17_b0 as v17
import run_semstamp_certified_defense_v2_wp3_p0 as p0
from semstamp_refined_atom_renderer import canonicalize_refined_bytes


STAGE = "semstamp_certified_defense_v2_wp3_detector_smoke"
PROTOCOL_SHA256 = (
    "49cc738c67bc41f28e05095a983a966c52150e5c3c7a77d9706e49601f33d627"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
MODEL_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
PUNKT_SHA256 = (
    "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c"
)
P0_FINAL_SHA256 = (
    "6a03448655d0084fe5e55505490c9018dedf79c66609dfd504183eeea9c3d6fe"
)
P0_READINESS_SHA256 = (
    "eb4da58d35be349a53aa4580884d23e6ecbb9d534a34c4f19d09769378b7a8f9"
)
P0_AUDIT_SHA256 = (
    "af21e2c408bbb5936c1c646b84ac8ddc2dafba8cb53f3f47ac592755e0041d78"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
SMOKE_PLAN_SHA256 = (
    "f4354914023a62927931e1423fa4db119cc202b6c002e765b3a79a1bd80509bf"
)
V17_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
SMOKE_PER_COHORT = 32
SMOKE_ROWS = 96
REPLAY_COUNT = 2
BATCH_SIZE = 64
NATIVE_TOKEN_LIMIT = 512
MIN_FREE_GIB = 5.0
MAX_VRAM_FRACTION = 0.80
MAX_GPU_SECONDS = 900
EXPECTED_MANDATORY_ROWS = 42
EXPECTED_ZERO_GAPS = 59
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


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
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def atomic_write_bytes(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, mode)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value):
    atomic_write_bytes(path, canonical_bytes(value))


def file_metadata(path, root=None):
    path = Path(path)
    result = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if root is not None:
        result["relative_path"] = str(
            path.resolve().relative_to(Path(root).resolve())
        )
    return result


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw text key at {location}: {overlap}")
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside root")
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
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "rows0": output_dir / "SMOKE_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "SMOKE_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/SMOKE_CHECKPOINT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "result": output_dir / "SMOKE_RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "runner": Path(__file__).resolve(),
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "p0_runner": (
            root / "scripts/run_semstamp_certified_defense_v2_wp3_p0.py"
        ),
        "test": (
            root / "scripts/test_semstamp_certified_defense_v2_wp3_smoke.py"
        ),
        "auditor": (
            root
            / "scripts/audit_semstamp_certified_defense_v2_wp3_smoke_design.py"
        ),
        "p0_dir": p0_dir,
        "p0_final": p0_dir / "FINAL_RESULT.json",
        "p0_readiness": p0_dir / "P0_READINESS.json",
        "p0_audit": p0_dir / "AUDIT.json",
        "p0_rows": p0_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "smoke_plan": p0_dir / "SMOKE_PLAN.jsonl",
        "wp1_rows": (
            root
            / "experiments/semstamp_certified_defense_v2_wp1_threat_model1"
            / "P0_RENDERER_ROWS_REPLAY0.jsonl"
        ),
        "v1_readiness": (
            root
            / "experiments/semstamp_certified_defense_v1_wp3_detector_health1"
            / "P0_READINESS.json"
        ),
        "v17_dir": v17_dir,
        "v17_plan": v17_dir / "B0_PLAN.jsonl",
        "v17_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


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
        raise ValueError("input manifest is empty")
    return entries


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen_design",
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "total_parent_rows": TOTAL_ROWS,
        "smoke_per_cohort": SMOKE_PER_COHORT,
        "smoke_rows": SMOKE_ROWS,
        "replay_count": REPLAY_COUNT,
        "batch_size": BATCH_SIZE,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "maximum_vram_fraction": MAX_VRAM_FRACTION,
        "maximum_gpu_seconds": MAX_GPU_SECONDS,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
        "punkt_payload_sha256": PUNKT_SHA256,
        "p0_final_sha256": P0_FINAL_SHA256,
        "p0_readiness_sha256": P0_READINESS_SHA256,
        "p0_audit_sha256": P0_AUDIT_SHA256,
        "p0_rows_sha256": P0_ROWS_SHA256,
        "smoke_plan_sha256": SMOKE_PLAN_SHA256,
        "execution_authorized": False,
        "full_replay_authorized": False,
        "wp4_authorized": False,
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
        "renderer_sha256": paths["renderer"],
        "p0_runner_sha256": paths["p0_runner"],
        "test_sha256": paths["test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["manifest"],
    }
    for field, path in artifact_paths.items():
        if contract.get(field) != sha256_file(path):
            raise ValueError(f"contract artifact mismatch: {field}")
    manifest = parse_manifest(paths["manifest"])
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
            "smoke execution is not authorized: "
            "EXECUTION_AUTHORIZATION.json is absent"
        )
    authorization = read_json(paths["authorization"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "SMOKE_EXECUTION_AUTHORIZED",
        "scope": "96_row_detector_smoke_only",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "runner_sha256": sha256_file(paths["runner"]),
        "smoke_plan_sha256": SMOKE_PLAN_SHA256,
        "authorized_replays": REPLAY_COUNT,
        "authorized_rows_per_replay": SMOKE_ROWS,
        "full_replay_authorized": False,
        "wp4_authorized": False,
        "certificate_authorized": False,
    }
    if set(authorization) != set(required) or any(
        authorization.get(key) != value
        for key, value in required.items()
    ):
        raise PermissionError("execution authorization is invalid")
    if contract["execution_authorized"] is not False:
        raise ValueError("design contract authorization state drift")
    return authorization


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
            or sha256_file(path) != expected_sha
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
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    free_gib = shutil.disk_usage(paths["output_dir"]).free / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        raise ValueError("insufficient free disk for smoke")
    return observed, free_gib


def validate_parent_p0(paths):
    expected = {
        paths["p0_final"]: P0_FINAL_SHA256,
        paths["p0_readiness"]: P0_READINESS_SHA256,
        paths["p0_audit"]: P0_AUDIT_SHA256,
        paths["p0_rows"]: P0_ROWS_SHA256,
        paths["smoke_plan"]: SMOKE_PLAN_SHA256,
        paths["v17_rows"]: V17_ROWS_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen parent drift: {path.name}")
    final = read_json(paths["p0_final"])
    readiness = read_json(paths["p0_readiness"])
    audit = read_json(paths["p0_audit"])
    if (
        final.get("terminal") != "WP3_P0_GO_FOR_SMOKE_DESIGN"
        or final.get("status") != "go"
        or readiness.get("status") != "go"
        or not all(readiness.get("gates", {}).values())
        or audit.get("status") != "go"
        or not all(audit.get("checks", {}).values())
        or final["decision"].get("smoke_execution_authorized") is not False
    ):
        raise ValueError("P0 is not the frozen audited Go")
    renderer_rows = read_jsonl(paths["p0_rows"])
    smoke_plan = read_jsonl(paths["smoke_plan"])
    rebuilt = p0.select_smoke_rows(renderer_rows)
    if (
        len(renderer_rows) != TOTAL_ROWS
        or jsonl_bytes(rebuilt) != paths["smoke_plan"].read_bytes()
        or smoke_plan != rebuilt
        or len(smoke_plan) != SMOKE_ROWS
        or len({row["plan_index"] for row in smoke_plan}) != SMOKE_ROWS
        or Counter(row["cohort"] for row in smoke_plan)
        != {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        or sum(
            row["selection_class"] == "mandatory_refinement"
            for row in smoke_plan
        ) != EXPECTED_MANDATORY_ROWS
        or sum(row["zero_byte_gap_count"] for row in smoke_plan)
        != EXPECTED_ZERO_GAPS
    ):
        raise ValueError("frozen smoke plan is invalid")
    return renderer_rows, smoke_plan


def load_nltk(nltk_data):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version must be 3.8.1")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    resource = nltk.data.find("tokenizers/punkt/english.pickle")
    with resource.open() as handle:
        if sha256_bytes(handle.read()) != PUNKT_SHA256:
            raise ValueError("Punkt payload SHA mismatch")
    return sent_tokenize


def prepare_documents(paths, smoke_plan, renderer_rows, nltk_data):
    b0_paths = v17.paths_for(paths["root"], paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    if (
        len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
        or v17.jsonl_bytes(plan) != paths["v17_plan"].read_bytes()
    ):
        raise ValueError("materialized inputs differ from frozen v17 plan")
    parent_rows = read_jsonl(paths["wp1_rows"])
    legacy_indices = tuple(
        read_json(paths["v1_readiness"])["renderer"]["failed_plan_indices"]
    )
    if len(parent_rows) != TOTAL_ROWS or len(legacy_indices) != 42:
        raise ValueError("frozen WP1 parent evidence mismatch")
    sent_tokenize = load_nltk(nltk_data)
    documents = {}
    for selected in smoke_plan:
        index = selected["plan_index"]
        text = texts[index]
        units = sent_tokenize(text)
        document = canonicalize_refined_bytes(
            text.encode("utf-8"),
            lambda _raw, frozen=units: frozen,
        )
        observed = p0.p0_row(
            index,
            plan[index],
            document,
            legacy_indices,
            parent_rows[index],
        )
        if canonical_bytes(observed) != canonical_bytes(
            renderer_rows[index]
        ):
            raise ValueError(f"refined P0 row drift at {index}")
        selected_fields = {
            "plan_index": observed["plan_index"],
            "cohort": observed["cohort"],
            "cohort_index": observed["cohort_index"],
            "raw_text_sha256": observed["raw_text_sha256"],
            "atom_count": observed["atom_count"],
            "unit_count": observed["unit_count"],
            "refinement_count": observed["refinement_count"],
            "zero_byte_gap_count": observed["zero_byte_gap_count"],
            "topology_sha256": observed["topology_sha256"],
            "legality_sha256": observed["legality_sha256"],
            "wp1_reference_row_sha256": (
                observed["wp1_reference_row_sha256"]
            ),
        }
        if any(
            selected.get(key) != value
            for key, value in selected_fields.items()
        ):
            raise ValueError(f"smoke/P0 binding drift at {index}")
        documents[index] = document
    return plan, documents


class ReplayLSHModel:
    def __init__(self, regions):
        self._regions = tuple(regions)
        self._index = 0

    def get_hash(self, texts):
        if len(texts) != 1:
            raise ValueError("official detect_lsh requested non-scalar input")
        if self._index >= len(self._regions):
            raise AssertionError("official detect_lsh over-consumed regions")
        value = self._regions[self._index]
        self._index += 1
        return [value]

    def assert_exhausted(self):
        if self._index != len(self._regions):
            raise AssertionError("official detect_lsh under-consumed regions")


def initialize_detector(paths, embedder_dir):
    import torch
    from run_certified_resegmentation_wp2_detector_equivalence import (
        semstamp_core,
    )

    expected_embedder = (
        paths["root"] / "models/semstamp-c4-sbert-a77dce8530e1"
    ).resolve()
    if Path(embedder_dir).resolve() != expected_embedder:
        raise ValueError("embedder directory is not the frozen model")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, get_mask, official_detect = semstamp_core(
        paths["repo"], expected_embedder
    )
    model.batch_size = BATCH_SIZE
    relation = bytes(
        int(
            current
            in {
                int(value)
                for value in get_mask(3, 0.25, previous)
                .detach()
                .cpu()
                .tolist()
            }
        )
        for previous in range(8)
        for current in range(8)
    )
    relation = support.validate_relation(relation)
    if sha256_bytes(relation) != RELATION_SHA256:
        raise ValueError("detector relation mismatch")
    torch.cuda.synchronize()
    return {
        "torch": torch,
        "model": model,
        "official_detect": official_detect,
        "relation": relation,
        "model_load_seconds": time.monotonic() - started,
    }


def make_scored_row(
    selected,
    planned,
    document,
    renderer_row,
    regions,
    relation,
    official_detect,
    reference,
):
    replay_model = ReplayLSHModel(regions)
    official_score = float(
        official_detect(
            list(document.original_units),
            replay_model,
            lmbd=0.25,
            lsh_dim=3,
        )
    )
    replay_model.assert_exhausted()
    base = support.make_public_score(
        planned,
        regions,
        relation,
        official_score,
    )
    if canonical_bytes(base) != canonical_bytes(reference):
        raise ValueError(
            f"v17 reference mismatch at row {planned['plan_index']}"
        )
    row = {
        **base,
        "stage": STAGE,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "smoke_index": selected["smoke_index"],
        "selection_class": selected["selection_class"],
        "structural_rank": selected["structural_rank"],
        "atom_count": renderer_row["atom_count"],
        "refinement_count": renderer_row["refinement_count"],
        "zero_byte_gap_count": renderer_row["zero_byte_gap_count"],
        "topology_sha256": renderer_row["topology_sha256"],
        "legality_sha256": renderer_row["legality_sha256"],
        "byte_reconstruction_sha256": (
            renderer_row["byte_reconstruction_sha256"]
        ),
        "p0_row_sha256": sha256_bytes(canonical_bytes(renderer_row)),
        "v17_reference_exact": True,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row


def validate_checkpoint(
    checkpoint,
    indices,
    contract_sha256,
    authorization_sha256,
):
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != STAGE
        or checkpoint.get("phase") != "smoke"
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint.get("authorization_sha256")
        != authorization_sha256
        or checkpoint.get("smoke_plan_sha256") != SMOKE_PLAN_SHA256
        or checkpoint.get("indices") != indices
        or set(checkpoint.get("replays", {})) != {"0", "1"}
    ):
        raise ValueError("invalid smoke checkpoint header")
    replays = checkpoint["replays"]
    if (
        len(replays["0"]) > len(indices)
        or len(replays["1"]) > len(replays["0"])
    ):
        raise ValueError("invalid smoke checkpoint lengths")
    for replay in range(REPLAY_COUNT):
        for offset, entry in enumerate(replays[str(replay)]):
            if (
                entry.get("public", {}).get("plan_index") != indices[offset]
                or entry.get("public", {}).get("v17_reference_exact")
                is not True
                or not isinstance(entry.get("regions"), list)
            ):
                raise ValueError("invalid smoke checkpoint entry")
            if replay and canonical_bytes(entry) != canonical_bytes(
                replays["0"][offset]
            ):
                raise ValueError("smoke checkpoint replay mismatch")
    return checkpoint


def write_heartbeat(paths, replay, completed, started, model_load_seconds):
    atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "phase": "smoke",
        "replay": replay,
        "completed_in_replay": completed,
        "completed_total": replay * SMOKE_ROWS + completed,
        "target_total": REPLAY_COUNT * SMOKE_ROWS,
        "started_at": started,
        "updated_at": utc_now(),
        "model_load_seconds": model_load_seconds,
        "contains_raw_text": False,
    })


def execute_replays(
    paths,
    plan,
    documents,
    renderer_rows,
    smoke_plan,
    reference_rows,
    detector,
):
    indices = [row["plan_index"] for row in smoke_plan]
    contract_sha256 = sha256_file(paths["contract"])
    authorization_sha256 = sha256_file(paths["authorization"])
    if paths["checkpoint"].is_file():
        checkpoint = validate_checkpoint(
            read_json(paths["checkpoint"]),
            indices,
            contract_sha256,
            authorization_sha256,
        )
    else:
        checkpoint = {
            "schema_version": 1,
            "stage": STAGE,
            "phase": "smoke",
            "contract_sha256": contract_sha256,
            "authorization_sha256": authorization_sha256,
            "smoke_plan_sha256": SMOKE_PLAN_SHA256,
            "indices": indices,
            "replays": {"0": [], "1": []},
        }
        atomic_write_json(paths["checkpoint"], checkpoint)
    started_at = utc_now()
    started = time.monotonic()
    torch = detector["torch"]
    with torch.inference_mode():
        for replay in range(REPLAY_COUNT):
            entries = checkpoint["replays"][str(replay)]
            write_heartbeat(
                paths,
                replay,
                len(entries),
                started_at,
                detector["model_load_seconds"],
            )
            for offset in range(len(entries), SMOKE_ROWS):
                selected = smoke_plan[offset]
                index = selected["plan_index"]
                document = documents[index]
                observed = detector["model"].get_hash(
                    list(document.original_units)
                )
                regions = [int(value) for value in observed]
                if (
                    len(regions) != plan[index]["unit_count"]
                    or any(value < 0 or value > 7 for value in regions)
                ):
                    raise ValueError(f"invalid regions at row {index}")
                public = make_scored_row(
                    selected,
                    plan[index],
                    document,
                    renderer_rows[index],
                    regions,
                    detector["relation"],
                    detector["official_detect"],
                    reference_rows[index],
                )
                entry = {"public": public, "regions": regions}
                if replay and canonical_bytes(entry) != canonical_bytes(
                    checkpoint["replays"]["0"][offset]
                ):
                    raise ValueError(f"smoke replay drift at row {index}")
                checkpoint["replays"][str(replay)].append(entry)
                atomic_write_json(paths["checkpoint"], checkpoint)
                write_heartbeat(
                    paths,
                    replay,
                    offset + 1,
                    started_at,
                    detector["model_load_seconds"],
                )
    validate_checkpoint(
        checkpoint,
        indices,
        contract_sha256,
        authorization_sha256,
    )
    if any(
        len(checkpoint["replays"][str(replay)]) != SMOKE_ROWS
        for replay in range(REPLAY_COUNT)
    ):
        raise AssertionError("smoke replay did not complete")
    rows = [
        [entry["public"] for entry in checkpoint["replays"][str(replay)]]
        for replay in range(REPLAY_COUNT)
    ]
    encoded = [jsonl_bytes(value) for value in rows]
    if encoded[0] != encoded[1]:
        raise AssertionError("public smoke replays differ")
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


def run_smoke(args):
    paths = paths_for(args.root, args.output_dir)
    contract, manifest = validate_contract(paths)
    authorization = validate_authorization(paths, contract)
    observed_inputs, free_gib = verify_inputs(paths, contract, manifest)
    renderer_rows, smoke_plan = validate_parent_p0(paths)
    plan, documents = prepare_documents(
        paths,
        smoke_plan,
        renderer_rows,
        args.nltk_data,
    )
    reference_rows = read_jsonl(paths["v17_rows"])
    if len(reference_rows) != TOTAL_ROWS:
        raise ValueError("v17 reference row count mismatch")

    detector = initialize_detector(paths, args.embedder_dir)
    executed = execute_replays(
        paths,
        plan,
        documents,
        renderer_rows,
        smoke_plan,
        reference_rows,
        detector,
    )
    atomic_write_bytes(paths["rows0"], executed["encoded"][0])
    atomic_write_bytes(paths["rows1"], executed["encoded"][1])
    peak_fraction = (
        executed["peak_cuda_bytes"] / executed["total_cuda_bytes"]
    )
    total_seconds = (
        executed["elapsed_seconds"] + detector["model_load_seconds"]
    )
    gates = {
        "authorization_exact": True,
        "contract_and_inputs_exact": True,
        "p0_terminal_exact": True,
        "smoke_plan_exact": True,
        "fixed_96_unique_rows": (
            len(smoke_plan) == SMOKE_ROWS
            and len({row["plan_index"] for row in smoke_plan}) == SMOKE_ROWS
        ),
        "balanced_32_per_cohort": (
            Counter(row["cohort"] for row in smoke_plan)
            == {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        ),
        "all_risk_rows_retained": (
            sum(
                row["selection_class"] == "mandatory_refinement"
                for row in smoke_plan
            ) == EXPECTED_MANDATORY_ROWS
            and sum(row["zero_byte_gap_count"] for row in smoke_plan)
            == EXPECTED_ZERO_GAPS
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
        "two_public_replays_exact": (
            executed["encoded"][0] == executed["encoded"][1]
        ),
        "two_sealed_region_replays_exact": True,
        "no_nonfinite_scores": all(
            math.isfinite(row["direct_score"])
            for row in executed["rows"][0]
        ),
        "no_exclusion_or_supplementation": True,
        "peak_vram_below_0_80": peak_fraction < MAX_VRAM_FRACTION,
        "runtime_below_900_seconds": total_seconds <= MAX_GPU_SECONDS,
        "public_artifacts_contain_no_raw_text": True,
        "certificate_generation_full_replay_wp4_not_started": True,
    }
    status = "go" if all(gates.values()) else "stop"
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "terminal": (
            "WP3_SMOKE_GO_FOR_FULL_REPLAY_DESIGN"
            if status == "go" else "WP3_SMOKE_STOP"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "authorization_sha256": sha256_file(paths["authorization"]),
        "smoke_plan_sha256": SMOKE_PLAN_SHA256,
        "runner_sha256": sha256_file(paths["runner"]),
        "gates": gates,
        "counts": {
            "rows_per_replay": SMOKE_ROWS,
            "replays": REPLAY_COUNT,
            "official_detect_lsh_calls": REPLAY_COUNT * SMOKE_ROWS,
            "direct_score_computations": REPLAY_COUNT * SMOKE_ROWS,
            "v17_reference_comparisons": REPLAY_COUNT * SMOKE_ROWS,
            "sample_exclusions": 0,
            "sample_supplements": 0,
            "mismatches": 0,
        },
        "outputs": {
            "replay0": {
                **file_metadata(paths["rows0"], paths["root"]),
                "rows": SMOKE_ROWS,
                "contains_raw_text": False,
            },
            "replay1": {
                **file_metadata(paths["rows1"], paths["root"]),
                "rows": SMOKE_ROWS,
                "contains_raw_text": False,
            },
            "checkpoint": {
                **file_metadata(paths["checkpoint"], paths["root"]),
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
            "authorization": authorization,
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 1,
            "detector_model_loads": 1,
            "detector_score_reads": 0,
            "detector_scores_computed": REPLAY_COUNT * SMOKE_ROWS,
            "generation_calls": 0,
            "gpu_calls": REPLAY_COUNT * SMOKE_ROWS,
            "raw_text_emitted_to_public_artifacts": False,
            "full_replay_started": False,
            "wp4_started": False,
        },
        "decision": {
            "smoke": status,
            "full_replay": (
                "authorized_to_freeze_design_only"
                if status == "go" else "not_authorized"
            ),
            "full_replay_execution_authorized": False,
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["result"], result)
    paths["heartbeat"].unlink(missing_ok=True)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))
    if status != "go":
        raise RuntimeError("WP3 detector smoke stopped")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke",), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    try:
        run_smoke(args)
    except Exception as error:
        atomic_write_json(paths["last_failure"], {
            "schema_version": 1,
            "stage": STAGE,
            "status": "invalid_or_incomplete",
            "terminal": "WP3_SMOKE_INVALID_OR_INCOMPLETE",
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "contains_raw_text": False,
        })
        raise


if __name__ == "__main__":
    main()
