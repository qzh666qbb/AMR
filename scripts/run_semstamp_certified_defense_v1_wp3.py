"""Run SemStamp certified-defense WP3 detector-health gates."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
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

import certified_resegmentation_wp2_v13_b0_support as support
import run_certified_resegmentation_wp2_v17_b0 as v17
from semstamp_fixed_byte_renderer import (
    FixedByteRendererError,
    canonicalize_fixed_bytes,
)


STAGE = "semstamp_certified_defense_v1_wp3_detector_health"
PROTOCOL_SHA256 = (
    "394368de4f1ec863e142d0f477c96e0053dd0ad03cd32312df8e1f2ff88a6684"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
MODEL_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
V17_THRESHOLD = 1.6666666666666667
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
MAX_GPU_SECONDS = 2 * 60 * 60
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


def canonical_digest(value):
    return sha256_bytes(canonical_bytes(value))


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


def atomic_write_json(path, value, mode=0o600):
    atomic_write_bytes(path, canonical_bytes(value), mode)


def file_metadata(path, root=None):
    path = Path(path)
    result = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if root is not None:
        result["relative_path"] = str(
            path.resolve().relative_to(Path(root).resolve())
        )
    return result


def package_versions():
    packages = (
        "nltk",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in packages},
    }


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(
                f"raw text key in public artifact at {location}: {overlap}"
            )
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside root")
    v17_dir = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "input_manifest": output_dir / "INPUT_MANIFEST.sha256",
        "renderer_rows": output_dir / "P0_RENDERER_ROWS.jsonl",
        "smoke_plan": output_dir / "SMOKE_PLAN.jsonl",
        "readiness": output_dir / "P0_READINESS.json",
        "smoke_rows0": output_dir / "SMOKE_ROWS_REPLAY0.jsonl",
        "smoke_rows1": output_dir / "SMOKE_ROWS_REPLAY1.jsonl",
        "smoke_checkpoint": output_dir / "sealed/smoke_checkpoint.json",
        "smoke_result": output_dir / "SMOKE_RESULT.json",
        "rows0": output_dir / "ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/full_checkpoint.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "result": output_dir / "RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "runner": Path(__file__).resolve(),
        "renderer": (
            root / "scripts/semstamp_fixed_byte_renderer.py"
        ),
        "test": (
            root / "scripts/test_semstamp_certified_defense_v1_wp3.py"
        ),
        "auditor": (
            root / "scripts/audit_semstamp_certified_defense_v1_wp3.py"
        ),
        "v17_dir": v17_dir,
        "v17_plan": v17_dir / "B0_PLAN.jsonl",
        "v17_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "v17_result": v17_dir / "B0_RESULT.json",
        "v17_audit": v17_dir / "B0_AUDIT.json",
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
    }


def parse_input_manifest(path):
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
            raise ValueError(f"invalid input manifest line {line_number}")
        entries[relative] = digest
    if not entries:
        raise ValueError("input manifest is empty")
    return entries


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen",
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "fixed_raw_utf8_bytes_v1",
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "total_rows": TOTAL_ROWS,
        "smoke_per_cohort": SMOKE_PER_COHORT,
        "smoke_rows": SMOKE_ROWS,
        "replay_count": REPLAY_COUNT,
        "batch_size": BATCH_SIZE,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "maximum_vram_fraction": MAX_VRAM_FRACTION,
        "maximum_gpu_seconds": MAX_GPU_SECONDS,
        "v17_threshold": V17_THRESHOLD,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
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
        "test_sha256": paths["test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["input_manifest"],
    }
    for field, path in artifact_paths.items():
        if contract.get(field) != sha256_file(path):
            raise ValueError(f"contract artifact mismatch: {field}")
    manifest = parse_input_manifest(paths["input_manifest"])
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(manifest):
        raise ValueError("contract/input manifest path set mismatch")
    for relative, expected_sha in manifest.items():
        meta = inputs[relative]
        if (
            not isinstance(meta, dict)
            or meta.get("sha256") != expected_sha
            or type(meta.get("bytes")) is not int
            or meta["bytes"] < 1
        ):
            raise ValueError(f"contract input metadata mismatch: {relative}")
    return contract, manifest


def verify_inputs(paths, contract, manifest):
    root = paths["root"]
    observed = {}
    for relative, expected_sha in manifest.items():
        relative_path = Path(relative)
        path = (root / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError(f"input is missing or escapes root: {relative}")
        meta = contract["inputs"][relative]
        if (
            sha256_file(path) != expected_sha
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"input bytes/SHA mismatch: {relative}")
        expected_mode = meta.get("mode")
        if (
            expected_mode is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
        ):
            raise ValueError(f"input mode mismatch: {relative}")
        observed[relative] = {
            "bytes": meta["bytes"],
            "sha256": expected_sha,
            **({"mode": expected_mode} if expected_mode else {}),
        }
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    return observed


def validate_v17_terminal(paths):
    result = read_json(paths["v17_result"])
    audit = read_json(paths["v17_audit"])
    if (
        result.get("status") != "go"
        or result.get("statistics", {}).get("calibration", {}).get("threshold")
        != V17_THRESHOLD
        or not all(result.get("gates", {}).values())
        or audit.get("status") != "go"
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("v17 B0 is not the frozen audited Go")
    reference_rows = read_jsonl(paths["v17_rows"])
    if len(reference_rows) != TOTAL_ROWS:
        raise ValueError("v17 reference row count mismatch")
    return result, reference_rows


def materialize_inputs(paths):
    b0_paths = v17.paths_for(paths["root"], paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    if v17.jsonl_bytes(plan) != paths["v17_plan"].read_bytes():
        raise ValueError("materialized plan differs from v17")
    if len(plan) != TOTAL_ROWS or len(texts) != TOTAL_ROWS:
        raise ValueError("materialized input count mismatch")
    return plan, texts


def load_nltk(nltk_data):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version must be 3.8.1")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    resource = nltk.data.find("tokenizers/punkt/english.pickle")
    with resource.open() as handle:
        resource_sha256 = sha256_bytes(handle.read())
    return sent_tokenize, resource_sha256


def renderer_row(index, planned, document):
    evidence = document.public_evidence()
    if (
        document.raw_utf8_sha256 != planned["raw_text_sha256"]
        or document.unit_count != planned["unit_count"]
        or evidence["byte_reconstruction_sha256"]
        != document.raw_utf8_sha256
    ):
        raise ValueError(f"renderer identity mismatch at row {index}")
    legality_sha256 = canonical_digest(
        {
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "original_unit_atom_spans": [
                list(span) for span in document.original_unit_atom_spans
            ],
            "v17_max_native_tokens": planned["max_native_tokens"],
        }
    )
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "pass",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "max_native_tokens": planned["max_native_tokens"],
        **evidence,
        "legality_sha256": legality_sha256,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row


def renderer_failure_row(index, planned, raw_utf8_bytes, units, error):
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "invalid_or_incomplete",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": sha256_bytes(raw_utf8_bytes),
        "raw_utf8_bytes": len(raw_utf8_bytes),
        "unit_count": planned["unit_count"],
        "max_native_tokens": planned["max_native_tokens"],
        "observed_nltk_unit_count": len(units),
        "failure_code": error.code,
        "failure_detail": error.detail,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }
    if (
        row["raw_text_sha256"] != planned["raw_text_sha256"]
        or row["observed_nltk_unit_count"] != planned["unit_count"]
    ):
        raise ValueError(f"failure-row identity drift at row {index}")
    reject_raw_public_keys(row)
    return row


def select_smoke_rows(renderer_rows):
    selected = []
    for cohort in COHORTS:
        cohort_rows = [
            row for row in renderer_rows if row["cohort"] == cohort
        ]
        if len(cohort_rows) != COHORT_SIZE:
            raise ValueError(f"renderer cohort count mismatch: {cohort}")
        ordered = sorted(
            cohort_rows,
            key=lambda row: (
                row["atom_count"],
                row["unit_count"],
                row["raw_text_sha256"],
                row["plan_index"],
            ),
        )
        ranks = [
            ((2 * index + 1) * COHORT_SIZE) // (2 * SMOKE_PER_COHORT)
            for index in range(SMOKE_PER_COHORT)
        ]
        if len(set(ranks)) != SMOKE_PER_COHORT:
            raise AssertionError("smoke quantile ranks are not unique")
        for selection_index, rank in enumerate(ranks):
            row = ordered[rank]
            selected.append({
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "smoke_index": len(selected),
                "cohort": cohort,
                "cohort_selection_index": selection_index,
                "structural_rank": rank,
                "plan_index": row["plan_index"],
                "cohort_index": row["cohort_index"],
                "raw_text_sha256": row["raw_text_sha256"],
                "atom_count": row["atom_count"],
                "unit_count": row["unit_count"],
                "topology_sha256": row["topology_sha256"],
                "legality_sha256": row["legality_sha256"],
                "score_access_count_during_construction": 0,
                "contains_raw_text": False,
            })
    if (
        len(selected) != SMOKE_ROWS
        or len({row["plan_index"] for row in selected}) != SMOKE_ROWS
        or Counter(row["cohort"] for row in selected)
        != {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
    ):
        raise AssertionError("invalid smoke selection")
    return selected


def build_p0(args):
    if args.batch_size != BATCH_SIZE:
        raise ValueError("batch size is frozen at 64")
    if args.min_free_gib != MIN_FREE_GIB:
        raise ValueError("minimum free disk is frozen at 5 GiB")
    paths = paths_for(args.root, args.output_dir)
    contract, manifest = validate_contract(paths)
    observed_inputs = verify_inputs(paths, contract, manifest)
    required_free = int(MIN_FREE_GIB * 1024**3)
    if shutil.disk_usage(paths["root"]).free < required_free:
        raise OSError("free disk is below 5 GiB")
    plan, texts = materialize_inputs(paths)
    sent_tokenize, punkt_sha256 = load_nltk(args.nltk_data)

    documents = []
    rows = []
    for index, (planned, text) in enumerate(zip(plan, texts)):
        raw_utf8_bytes = text.encode("utf-8")
        units = sent_tokenize(text)
        try:
            document = canonicalize_fixed_bytes(
                raw_utf8_bytes,
                lambda _text, frozen=units: frozen,
            )
        except FixedByteRendererError as error:
            documents.append(None)
            rows.append(
                renderer_failure_row(
                    index,
                    planned,
                    raw_utf8_bytes,
                    units,
                    error,
                )
            )
        else:
            documents.append(document)
            rows.append(renderer_row(index, planned, document))
    failures = [row for row in rows if row["status"] != "pass"]
    valid_rows = [row for row in rows if row["status"] == "pass"]
    status = "stop" if failures else "pass"
    smoke_plan = [] if failures else select_smoke_rows(rows)
    renderer_encoded = jsonl_bytes(rows)
    smoke_encoded = jsonl_bytes(smoke_plan)
    contract_sha256 = sha256_file(paths["contract"])
    readiness = {
        "schema_version": 1,
        "stage": f"{STAGE}_p0",
        "status": status,
        "terminal_reason": (
            "renderer_contract_stop"
            if failures
            else "p0_renderer_and_readiness_pass"
        ),
        "scope": "fixed_byte_renderer_all_3000_model_free",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": contract_sha256,
        "input_manifest_sha256": sha256_file(paths["input_manifest"]),
        "artifacts": {
            "runner": file_metadata(paths["runner"], paths["root"]),
            "renderer": file_metadata(paths["renderer"], paths["root"]),
            "test": file_metadata(paths["test"], paths["root"]),
            "auditor": file_metadata(paths["auditor"], paths["root"]),
            "inputs": observed_inputs,
        },
        "cohorts": {
            "rows": TOTAL_ROWS,
            "counts": dict(Counter(row["cohort"] for row in rows)),
            "raw_identity_unique": (
                len({row["raw_text_sha256"] for row in rows}) == TOTAL_ROWS
            ),
        },
        "renderer": {
            "contract": "fixed_raw_utf8_bytes_v1",
            "rows": TOTAL_ROWS,
            "valid_rows": len(valid_rows),
            "invalid_rows": len(failures),
            "failure_code_counts": dict(Counter(
                row["failure_code"] for row in failures
            )),
            "failure_detail_counts": dict(Counter(
                row["failure_detail"].split("_at_unit_")[0]
                for row in failures
            )),
            "failure_cohort_counts": dict(Counter(
                row["cohort"] for row in failures
            )),
            "failed_plan_indices": [
                row["plan_index"] for row in failures
            ],
            "failed_raw_text_sha256": [
                row["raw_text_sha256"] for row in failures
            ],
            "renderer_rows_sha256": sha256_bytes(renderer_encoded),
            "topology_sha256": canonical_digest(
                [row["topology_sha256"] for row in valid_rows]
            ),
            "legality_sha256": canonical_digest(
                [row["legality_sha256"] for row in valid_rows]
            ),
            "reconstruction_sha256": canonical_digest(
                [row["byte_reconstruction_sha256"] for row in valid_rows]
            ),
            "all_original_partitions_exact": not failures,
            "all_raw_bytes_reconstructed": not failures,
            "all_v17_unit_counts_exact": True,
        },
        "smoke_selection": {
            "algorithm": "per_cohort_structural_midpoint_quantiles_v1",
            "status": "not_started" if failures else "frozen",
            "rows": 0 if failures else SMOKE_ROWS,
            "per_cohort": SMOKE_PER_COHORT,
            "smoke_plan_sha256": (
                None if failures else sha256_bytes(smoke_encoded)
            ),
            "score_blind": True,
        },
        "environment": package_versions(),
        "nltk": {
            "version": "3.8.1",
            "english_punkt_payload_sha256": punkt_sha256,
        },
        "resource_gate": {
            "minimum_free_gib": MIN_FREE_GIB,
            "free_disk_pass": True,
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "score_access_count_during_construction": 0,
            "sealed_inputs_parsed": True,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys([rows, smoke_plan, readiness])
    return (
        paths,
        contract,
        plan,
        documents,
        rows,
        smoke_plan,
        renderer_encoded,
        smoke_encoded,
        readiness,
    )


def run_p0(args):
    (
        paths,
        _contract,
        _plan,
        _documents,
        _rows,
        _smoke_plan,
        renderer_encoded,
        smoke_encoded,
        readiness,
    ) = build_p0(args)
    atomic_write_bytes(paths["renderer_rows"], renderer_encoded)
    if readiness["status"] == "pass":
        atomic_write_bytes(paths["smoke_plan"], smoke_encoded)
    else:
        paths["smoke_plan"].unlink(missing_ok=True)
    atomic_write_json(paths["readiness"], readiness)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(readiness, sort_keys=True))
    return readiness


def require_frozen_p0(args):
    built = build_p0(args)
    paths = built[0]
    renderer_encoded = built[6]
    smoke_encoded = built[7]
    readiness = built[8]
    if readiness["status"] != "pass":
        raise ValueError("P0 renderer gate did not pass")
    if (
        not paths["renderer_rows"].is_file()
        or paths["renderer_rows"].read_bytes() != renderer_encoded
        or not paths["smoke_plan"].is_file()
        or paths["smoke_plan"].read_bytes() != smoke_encoded
        or not paths["readiness"].is_file()
        or paths["readiness"].read_bytes() != canonical_bytes(readiness)
    ):
        raise ValueError("current P0 does not match frozen P0 artifacts")
    return built


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
    load_started = time.monotonic()
    model, get_mask, official_detect = semstamp_core(
        paths["repo"],
        Path(embedder_dir).resolve(),
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
    if hashlib.sha256(relation).hexdigest() != RELATION_SHA256:
        raise ValueError("detector relation mismatch")
    torch.cuda.synchronize()
    return {
        "torch": torch,
        "model": model,
        "official_detect": official_detect,
        "relation": relation,
        "model_load_seconds": time.monotonic() - load_started,
    }


def make_scored_row(
    plan_index,
    planned,
    document,
    renderer,
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
        raise ValueError(f"v17 reference mismatch at row {plan_index}")
    row = {
        **base,
        "renderer_contract": "fixed_raw_utf8_bytes_v1",
        "atom_count": renderer["atom_count"],
        "original_boundaries": renderer["original_boundaries"],
        "original_units_sha256": renderer["original_units_sha256"],
        "topology_sha256": renderer["topology_sha256"],
        "legality_sha256": renderer["legality_sha256"],
        "byte_reconstruction_sha256": (
            renderer["byte_reconstruction_sha256"]
        ),
        "v17_reference_exact": True,
    }
    reject_raw_public_keys(row)
    return row


def validate_checkpoint(checkpoint, phase, indices, contract_sha256, p0_sha256):
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != STAGE
        or checkpoint.get("phase") != phase
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint.get("p0_readiness_sha256") != p0_sha256
        or checkpoint.get("indices") != indices
        or set(checkpoint.get("replays", {})) != {"0", "1"}
    ):
        raise ValueError(f"invalid {phase} checkpoint header")
    replays = checkpoint["replays"]
    if (
        len(replays["0"]) > len(indices)
        or len(replays["1"]) > len(replays["0"])
    ):
        raise ValueError(f"invalid {phase} checkpoint lengths")
    for replay in range(REPLAY_COUNT):
        for offset, entry in enumerate(replays[str(replay)]):
            if (
                not isinstance(entry, dict)
                or entry.get("public", {}).get("plan_index")
                != indices[offset]
                or entry.get("public", {}).get("v17_reference_exact") is not True
                or not isinstance(entry.get("regions"), list)
            ):
                raise ValueError(f"invalid {phase} checkpoint entry")
            if replay and canonical_bytes(entry) != canonical_bytes(
                replays["0"][offset]
            ):
                raise ValueError(f"{phase} checkpoint replay mismatch")
    return checkpoint


def write_heartbeat(paths, phase, replay, completed, target, started, **extra):
    atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "phase": phase,
        "replay": replay,
        "completed_in_replay": completed,
        "completed_total": replay * target + completed,
        "target_total": REPLAY_COUNT * target,
        "started_at": started,
        "updated_at": utc_now(),
        "contains_raw_text": False,
        **extra,
    })


def execute_replays(
    paths,
    phase,
    indices,
    plan,
    documents,
    renderer_rows,
    reference_rows,
    detector,
    checkpoint_path,
):
    torch = detector["torch"]
    contract_sha256 = sha256_file(paths["contract"])
    p0_sha256 = sha256_file(paths["readiness"])
    if checkpoint_path.is_file():
        checkpoint = validate_checkpoint(
            read_json(checkpoint_path),
            phase,
            indices,
            contract_sha256,
            p0_sha256,
        )
    else:
        checkpoint = {
            "schema_version": 1,
            "stage": STAGE,
            "phase": phase,
            "contract_sha256": contract_sha256,
            "p0_readiness_sha256": p0_sha256,
            "indices": indices,
            "replays": {"0": [], "1": []},
        }
        atomic_write_json(checkpoint_path, checkpoint)

    started_at = utc_now()
    replay_started = time.monotonic()
    with torch.inference_mode():
        for replay in range(REPLAY_COUNT):
            entries = checkpoint["replays"][str(replay)]
            write_heartbeat(
                paths,
                phase,
                replay,
                len(entries),
                len(indices),
                started_at,
                model_load_seconds=detector["model_load_seconds"],
            )
            for offset in range(len(entries), len(indices)):
                plan_index = indices[offset]
                document = documents[plan_index]
                observed = detector["model"].get_hash(
                    list(document.original_units)
                )
                regions = [int(value) for value in observed]
                if (
                    len(regions) != plan[plan_index]["unit_count"]
                    or any(value < 0 or value > 7 for value in regions)
                ):
                    raise ValueError(
                        f"invalid region output at row {plan_index}"
                    )
                public = make_scored_row(
                    plan_index,
                    plan[plan_index],
                    document,
                    renderer_rows[plan_index],
                    regions,
                    detector["relation"],
                    detector["official_detect"],
                    reference_rows[plan_index],
                )
                entry = {"public": public, "regions": regions}
                if replay and canonical_bytes(entry) != canonical_bytes(
                    checkpoint["replays"]["0"][offset]
                ):
                    raise ValueError(
                        f"{phase} replay mismatch at row {plan_index}"
                    )
                checkpoint["replays"][str(replay)].append(entry)
                atomic_write_json(checkpoint_path, checkpoint)
                write_heartbeat(
                    paths,
                    phase,
                    replay,
                    offset + 1,
                    len(indices),
                    started_at,
                    model_load_seconds=detector["model_load_seconds"],
                )
    validate_checkpoint(
        checkpoint,
        phase,
        indices,
        contract_sha256,
        p0_sha256,
    )
    if any(
        len(checkpoint["replays"][str(replay)]) != len(indices)
        for replay in range(REPLAY_COUNT)
    ):
        raise AssertionError(f"{phase} replay did not complete")
    public_rows = [
        [entry["public"] for entry in checkpoint["replays"][str(replay)]]
        for replay in range(REPLAY_COUNT)
    ]
    encoded = [jsonl_bytes(rows) for rows in public_rows]
    if encoded[0] != encoded[1]:
        raise AssertionError(f"{phase} public replay mismatch")
    torch.cuda.synchronize()
    return {
        "checkpoint": checkpoint,
        "rows": public_rows,
        "encoded": encoded,
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - replay_started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "total_cuda_bytes": torch.cuda.get_device_properties(0).total_memory,
        "device_name": torch.cuda.get_device_name(0),
    }


def run_smoke(args):
    (
        paths,
        _contract,
        plan,
        documents,
        renderer_rows,
        smoke_plan,
        _renderer_encoded,
        _smoke_encoded,
        _readiness,
    ) = require_frozen_p0(args)
    _v17_result, reference_rows = validate_v17_terminal(paths)
    indices = [row["plan_index"] for row in smoke_plan]
    detector = initialize_detector(paths, args.embedder_dir)
    executed = execute_replays(
        paths,
        "smoke",
        indices,
        plan,
        documents,
        renderer_rows,
        reference_rows,
        detector,
        paths["smoke_checkpoint"],
    )
    atomic_write_bytes(paths["smoke_rows0"], executed["encoded"][0])
    atomic_write_bytes(paths["smoke_rows1"], executed["encoded"][1])
    peak_fraction = (
        executed["peak_cuda_bytes"] / executed["total_cuda_bytes"]
    )
    gates = {
        "p0_exact": True,
        "fixed_96_rows": len(indices) == SMOKE_ROWS,
        "score_blind_selection": all(
            row["score_access_count_during_construction"] == 0
            for row in smoke_plan
        ),
        "renderer_exact": True,
        "official_direct_error_zero": True,
        "v17_reference_exact": True,
        "two_replays_exact": executed["encoded"][0] == executed["encoded"][1],
        "no_nonfinite_scores": all(
            math.isfinite(row["direct_score"])
            for row in executed["rows"][0]
        ),
        "peak_vram_below_0_80": peak_fraction < MAX_VRAM_FRACTION,
        "runtime_below_2_gpu_hours": (
            executed["elapsed_seconds"] + detector["model_load_seconds"]
            <= MAX_GPU_SECONDS
        ),
    }
    status = "pass" if all(gates.values()) else "stop"
    result = {
        "schema_version": 1,
        "stage": f"{STAGE}_smoke",
        "status": status,
        "terminal_reason": (
            "wp3_smoke_pass" if status == "pass" else "wp3_health_stop"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "p0_readiness_sha256": sha256_file(paths["readiness"]),
        "smoke_plan_sha256": sha256_file(paths["smoke_plan"]),
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
                **file_metadata(paths["smoke_rows0"], paths["root"]),
                "rows": SMOKE_ROWS,
                "contains_raw_text": False,
            },
            "replay1": {
                **file_metadata(paths["smoke_rows1"], paths["root"]),
                "rows": SMOKE_ROWS,
                "contains_raw_text": False,
            },
            "checkpoint": {
                **file_metadata(paths["smoke_checkpoint"], paths["root"]),
                "mode": "0600",
                "contains_raw_text": False,
            },
        },
        "runtime": {
            "started_at": executed["started_at"],
            "finished_at": utc_now(),
            "detector_elapsed_seconds": (
                executed["elapsed_seconds"] + detector["model_load_seconds"]
            ),
            "replay_elapsed_seconds": executed["elapsed_seconds"],
            "model_load_seconds": detector["model_load_seconds"],
            "device_name": executed["device_name"],
            "peak_cuda_bytes": executed["peak_cuda_bytes"],
            "total_cuda_bytes": executed["total_cuda_bytes"],
            "peak_vram_fraction": peak_fraction,
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 1,
            "detector_model_loads": 1,
            "detector_scores_computed": REPLAY_COUNT * SMOKE_ROWS,
            "generation_calls": 0,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["smoke_result"], result)
    print(json.dumps(result, sort_keys=True))
    if status != "pass":
        raise RuntimeError("WP3 smoke failed")
    return result


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": ordered[len(ordered) // 4],
        "median": ordered[len(ordered) // 2],
        "q75": ordered[(3 * len(ordered)) // 4],
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def run_full(args):
    (
        paths,
        _contract,
        plan,
        documents,
        renderer_rows,
        _smoke_plan,
        _renderer_encoded,
        _smoke_encoded,
        _readiness,
    ) = require_frozen_p0(args)
    smoke = read_json(paths["smoke_result"])
    if (
        smoke.get("status") != "pass"
        or not all(smoke.get("gates", {}).values())
        or smoke.get("contract_sha256") != sha256_file(paths["contract"])
        or smoke.get("p0_readiness_sha256")
        != sha256_file(paths["readiness"])
        or smoke.get("runner_sha256") != sha256_file(paths["runner"])
    ):
        raise ValueError("full replay requires the exact passing smoke")
    v17_result, reference_rows = validate_v17_terminal(paths)
    indices = list(range(TOTAL_ROWS))
    detector = initialize_detector(paths, args.embedder_dir)
    executed = execute_replays(
        paths,
        "full",
        indices,
        plan,
        documents,
        renderer_rows,
        reference_rows,
        detector,
        paths["checkpoint"],
    )
    atomic_write_bytes(paths["rows0"], executed["encoded"][0])
    atomic_write_bytes(paths["rows1"], executed["encoded"][1])

    score_groups = {
        cohort: [
            row["direct_score"]
            for row in executed["rows"][0]
            if row["cohort"] == cohort
        ]
        for cohort in COHORTS
    }
    statistics = support.summarize_scores(
        score_groups["calibration_null"],
        score_groups["independent_test_null"],
        score_groups["positive"],
    )
    statistics_exact = canonical_bytes(statistics) == canonical_bytes(
        v17_result["statistics"]
    )
    peak_fraction = (
        executed["peak_cuda_bytes"] / executed["total_cuda_bytes"]
    )
    total_detector_seconds = (
        smoke["runtime"]["detector_elapsed_seconds"]
        + detector["model_load_seconds"]
        + executed["elapsed_seconds"]
    )
    gates = {
        "p0_exact": True,
        "smoke_pass": True,
        "all_3000_rows_complete": all(
            len(values) == COHORT_SIZE for values in score_groups.values()
        ),
        "renderer_exact": True,
        "official_direct_error_zero": True,
        "v17_reference_exact": True,
        "v17_statistics_exact": statistics_exact,
        "threshold_exact": (
            statistics["calibration"]["threshold"] == V17_THRESHOLD
        ),
        "fpr_cp95_upper_le_0_08": (
            statistics["test"]["one_sided_cp_upper"] <= 0.08
        ),
        "tpr_ge_0_80": statistics["positive"]["rate"] >= 0.80,
        "tpr_cp95_lower_gt_0_75": (
            statistics["positive"]["one_sided_cp_lower"] > 0.75
        ),
        "two_full_replays_exact": (
            executed["encoded"][0] == executed["encoded"][1]
        ),
        "no_nonfinite_scores": all(
            math.isfinite(score)
            for values in score_groups.values()
            for score in values
        ),
        "no_sample_exclusion_or_supplement": True,
        "peak_vram_below_0_80": peak_fraction < MAX_VRAM_FRACTION,
        "total_runtime_below_2_gpu_hours": (
            total_detector_seconds <= MAX_GPU_SECONDS
        ),
    }
    status = "go" if all(gates.values()) else "stop"
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "terminal_reason": (
            "wp3_detector_health_go_for_wp4_design"
            if status == "go"
            else "wp3_health_stop"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["input_manifest"]),
        "runner_sha256": sha256_file(paths["runner"]),
        "renderer_sha256": sha256_file(paths["renderer"]),
        "p0_readiness_sha256": sha256_file(paths["readiness"]),
        "smoke_result_sha256": sha256_file(paths["smoke_result"]),
        "detector": {
            "semstamp_revision": SEMSTAMP_REVISION,
            "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
            "model_revision": MODEL_REVISION,
            "relation_sha256": RELATION_SHA256,
            "device": "cuda:0",
            "arithmetic": "FP32",
            "batch_size": BATCH_SIZE,
        },
        "statistics": statistics,
        "diagnostics": {
            "score_sha256": {
                cohort: canonical_digest(values)
                for cohort, values in score_groups.items()
            },
            "score_distribution": {
                cohort: distribution(values)
                for cohort, values in score_groups.items()
            },
            "threshold_ties": sum(
                score == statistics["calibration"]["threshold"]
                for score in score_groups["calibration_null"]
            ),
        },
        "gates": gates,
        "counts": {
            "cohort_size": COHORT_SIZE,
            "rows_per_replay": TOTAL_ROWS,
            "replays": REPLAY_COUNT,
            "official_detect_lsh_calls": REPLAY_COUNT * TOTAL_ROWS,
            "direct_score_computations": REPLAY_COUNT * TOTAL_ROWS,
            "v17_reference_comparisons": REPLAY_COUNT * TOTAL_ROWS,
            "sample_exclusions": 0,
            "sample_supplements": 0,
            "nonfinite_scores": 0,
            "official_direct_mismatches": 0,
            "v17_reference_mismatches": 0,
            "replay_mismatches": 0,
        },
        "outputs": {
            "replay0": {
                **file_metadata(paths["rows0"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "replay1": {
                **file_metadata(paths["rows1"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "checkpoint": {
                **file_metadata(paths["checkpoint"], paths["root"]),
                "mode": "0600",
                "contains_raw_text": False,
            },
        },
        "runtime": {
            "started_at": executed["started_at"],
            "finished_at": utc_now(),
            "full_detector_elapsed_seconds": (
                detector["model_load_seconds"] + executed["elapsed_seconds"]
            ),
            "full_replay_elapsed_seconds": executed["elapsed_seconds"],
            "smoke_detector_elapsed_seconds": (
                smoke["runtime"]["detector_elapsed_seconds"]
            ),
            "total_detector_elapsed_seconds": total_detector_seconds,
            "model_load_seconds": detector["model_load_seconds"],
            "device_name": executed["device_name"],
            "peak_cuda_bytes": executed["peak_cuda_bytes"],
            "total_cuda_bytes": executed["total_cuda_bytes"],
            "peak_vram_fraction": peak_fraction,
            "environment": package_versions(),
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 1,
            "detector_model_loads": 1,
            "detector_scores_computed": REPLAY_COUNT * TOTAL_ROWS,
            "generation_calls": 0,
            "raw_text_emitted_to_public_artifacts": False,
            "wp4_started": False,
        },
        "decision": {
            "wp3_detector_health": status,
            "wp4": (
                "authorized_to_freeze_design" if status == "go" else "stop"
            ),
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["result"], result)
    atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "phase": "terminal",
        "updated_at": utc_now(),
        "result_sha256": sha256_file(paths["result"]),
        "contains_raw_text": False,
    })
    print(json.dumps(result, sort_keys=True))
    if status != "go":
        raise RuntimeError("WP3 detector health failed")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--phase", choices=("p0", "smoke", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    try:
        if args.phase == "p0":
            run_p0(args)
        elif args.phase == "smoke":
            run_smoke(args)
        else:
            run_full(args)
    except Exception as error:
        atomic_write_json(paths["last_failure"], {
            "schema_version": 1,
            "stage": STAGE,
            "status": "invalid_or_incomplete",
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at": utc_now(),
            "contains_raw_text": False,
        })
        raise


if __name__ == "__main__":
    main()
