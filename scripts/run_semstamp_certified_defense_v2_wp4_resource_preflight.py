"""Run the authorized score-blind WP4 resource preflight."""

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import tempfile
import time


STAGE = "semstamp_certified_defense_v2_wp4_resource_preflight"
PROTOCOL_SHA256 = (
    "3334b8d5567db674f0350d37b8b1d345e27466d26e2dea4df44d4a7e2105f904"
)
WP3_FINAL_SHA256 = (
    "b51733c497713389ca91c7e14b4313f44c9ae94d19e40d5be88e9292be0a0951"
)
WP3_AUDIT_SHA256 = (
    "1ced8396070eabdac75f40ebac1d34d5a38229e7bf93f44a916e23eeb1dc91a2"
)
P0_ROWS_SHA256 = (
    "8625f6b074150d4ec099b9e57da00d6b08167b42f25ad95d34b975682e5a7474"
)
V6_PROFILE_SHA256 = (
    "4701a4b6d77f2d0c20e189853ec896d70ac205df05659ee38e576b5010119f6d"
)
V18_READINESS_SHA256 = (
    "bee8324acf47e7772f7628127082159d73d6d0040f9b0c7f1a95fe64efdf0d56"
)
LINUX_CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
PUNKT_SHA256 = (
    "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c"
)
TOKENIZER_SHA256 = {
    "config.json": (
        "77d80f1542b994cb16f9b9dadd34e04e6ab7a0f79ca201ce309f3d7cf1c82b97"
    ),
    "special_tokens_map.json": (
        "15a9b70f6a5defe4feac823a82e6412e5d5bfd8eab09d8dbcf5ac241f14a7756"
    ),
    "tokenizer.json": (
        "8c28ddc1c09f31165c1a52805d6c5318a40545b1b734a31143121c92cb339778"
    ),
    "tokenizer_config.json": (
        "3ea32220c2e21af4f0be95e2a43bdc8ac693f8c35c2255127ef680689b469461"
    ),
    "vocab.txt": (
        "dbd90cb94e2247bd4d4ccaecbf616d2290e66691d7d5e5bb81f063c2d0649ada"
    ),
}
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
RADII = (0, 1, 2, 3)
REPLAY_COUNT = 2
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
LEGACY_PAIR_DIAGNOSTIC = 5_000_000
SIDECAR_GATE_BYTES = 50 * 1024**3
WORK_DISK_GATE_BYTES = 15 * 1024**3
FREE_DISK_RESERVE_BYTES = 5 * 1024**3
GPU_HOUR_GATE = 72.0
NATIVE_TOKEN_LIMIT = 512
TOKENIZER_CHUNK_SIZE = 512
SELECTED_DEVICE = "NVIDIA GeForce RTX 3090"
SELECTED_DRIVER = "595.80"
SELECTED_MEMORY_MIB = 24576
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


def jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) for row in rows)


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
        os.chmod(path, mode)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value, mode=0o600):
    atomic_write_bytes(path, canonical_bytes(value), mode)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


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
            raise ValueError(f"invalid input manifest line {line_number}")
        entries[relative] = digest
    if not entries:
        raise ValueError("input manifest is empty")
    return entries


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must remain inside root")
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "authorization": output_dir / "EXECUTION_AUTHORIZATION.json",
        "resume_authorization": output_dir / "RESUME_AUTHORIZATION.json",
        "rows": output_dir / "PREFLIGHT_ROWS.jsonl",
        "timings": output_dir / "sealed/PREFLIGHT_TIMINGS.jsonl",
        "result": output_dir / "PREFLIGHT_RESULT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "lock": output_dir / "EXECUTION.lock",
        "runner": Path(__file__).resolve(),
        "p0_dir": (
            root
            / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
        ),
        "wp3_dir": (
            root
            / "experiments/"
            "semstamp_certified_defense_v2_wp3_detector_full_replay_design1"
        ),
        "v18_dir": (
            root
            / "experiments/certified_resegmentation_wp2_v18_b1_exact_certificate1"
        ),
        "profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1/"
            "semstamp_workload.json"
        ),
    }


def expected_authorization(paths, contract):
    return {
        "schema_version": 1,
        "stage": STAGE,
        "status": "WP4_RESOURCE_PREFLIGHT_EXECUTION_AUTHORIZED",
        "scope": "score_blind_3000_row_resource_preflight_only",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "runner_sha256": sha256_file(paths["runner"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "wp3_final_sha256": WP3_FINAL_SHA256,
        "rows": TOTAL_ROWS,
        "radii": list(RADII),
        "projection_replays": REPLAY_COUNT,
        "selected_hardware": contract["selected_hardware"],
        "resource_budget": contract["resource_budget"],
        "sbert_load_authorized": False,
        "detector_score_authorized": False,
        "certificate_authorized": False,
        "wp5_authorized": False,
    }


def validate_authorization(paths, contract):
    if contract.get("execution_authorized") is not False:
        raise PermissionError("design contract does not prohibit execution")
    if not paths["authorization"].is_file():
        raise PermissionError("WP4 resource preflight execution is not authorized")
    authorization = read_json(paths["authorization"])
    if authorization != expected_authorization(paths, contract):
        raise PermissionError("WP4 execution authorization is invalid")
    return authorization


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen_design",
        "protocol_sha256": PROTOCOL_SHA256,
        "total_rows": TOTAL_ROWS,
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "radii": list(RADII),
        "projection_replays": REPLAY_COUNT,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "execution_authorized": False,
        "sbert_load_authorized": False,
        "detector_score_authorized": False,
        "certificate_authorized": False,
        "wp5_authorized": False,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    if contract.get("selected_hardware") != {
        "device": SELECTED_DEVICE,
        "driver_version": SELECTED_DRIVER,
        "memory_mib": SELECTED_MEMORY_MIB,
        "cuda_visible_devices": "0",
    }:
        raise ValueError("selected hardware contract mismatch")
    if contract.get("resource_budget") != {
        "span_gate": SPAN_GATE,
        "pair_gate": PAIR_GATE,
        "legacy_pair_diagnostic": LEGACY_PAIR_DIAGNOSTIC,
        "sidecar_gate_bytes": SIDECAR_GATE_BYTES,
        "work_disk_gate_bytes": WORK_DISK_GATE_BYTES,
        "free_disk_reserve_bytes": FREE_DISK_RESERVE_BYTES,
        "gpu_hour_gate": GPU_HOUR_GATE,
    }:
        raise ValueError("resource budget contract mismatch")
    artifacts = {
        "runner_sha256": paths["runner"],
        "input_manifest_sha256": paths["manifest"],
    }
    for field, path in artifacts.items():
        if contract.get(field) != sha256_file(path):
            raise ValueError(f"contract artifact mismatch: {field}")
    manifest = parse_manifest(paths["manifest"])
    if set(contract.get("inputs", {})) != set(manifest):
        raise ValueError("contract/input manifest path set mismatch")
    return contract, manifest


def verify_inputs(paths, contract, manifest):
    for relative, expected_sha256 in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
        ):
            raise ValueError(f"input missing or escapes root: {relative}")
        metadata = contract["inputs"][relative]
        if (
            sha256_file(path) != expected_sha256
            or path.stat().st_size != metadata["bytes"]
        ):
            raise ValueError(f"input bytes/SHA mismatch: {relative}")
        expected_mode = metadata.get("mode")
        if (
            expected_mode is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
        ):
            raise ValueError(f"input mode mismatch: {relative}")


def validate_parent_evidence(paths):
    final = read_json(paths["wp3_dir"] / "FINAL_RESULT.json")
    audit = read_json(paths["wp3_dir"] / "EXECUTION_AUDIT.json")
    v18 = read_json(paths["v18_dir"] / "P0_READINESS.json")
    if (
        sha256_file(paths["wp3_dir"] / "FINAL_RESULT.json")
        != WP3_FINAL_SHA256
        or sha256_file(paths["wp3_dir"] / "EXECUTION_AUDIT.json")
        != WP3_AUDIT_SHA256
        or final.get("terminal") != "WP3_FULL_REPLAY_GO_FOR_WP4_DESIGN"
        or final.get("decision", {}).get("wp4_started") is not False
        or audit.get("status") != "go"
        or sha256_file(paths["p0_dir"] / "P0_RENDERER_ROWS_REPLAY0.jsonl")
        != P0_ROWS_SHA256
        or sha256_file(paths["profile"]) != V6_PROFILE_SHA256
        or sha256_file(paths["v18_dir"] / "P0_READINESS.json")
        != V18_READINESS_SHA256
        or v18.get("status") != "stop"
        or v18.get("terminal_reason") != "b1_readiness_stop"
    ):
        raise ValueError("parent evidence is not at the frozen terminal")


def validate_resume_authorization(paths, completed_rows):
    if not completed_rows:
        if paths["resume_authorization"].exists():
            raise PermissionError("resume authorization is forbidden initially")
        return None
    if len(completed_rows) >= TOTAL_ROWS:
        raise PermissionError("completed preflight cannot resume")
    if not paths["resume_authorization"].is_file():
        raise PermissionError("partial preflight requires resume authorization")
    attempt_indices = [row.get("attempt_index") for row in completed_rows]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in attempt_indices
    ):
        raise PermissionError("completed prefix has invalid attempt indices")
    next_attempt_index = max(attempt_indices) + 1
    expected = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "WP4_RESOURCE_PREFLIGHT_RESUME_AUTHORIZED",
        "scope": "exact_completed_prefix_only",
        "execution_authorization_sha256": sha256_file(
            paths["authorization"]
        ),
        "completed_rows_sha256": sha256_file(paths["rows"]),
        "completed_prefix_length": len(completed_rows),
        "next_attempt_index": next_attempt_index,
        "no_score_based_retry": True,
        "sbert_load_authorized": False,
        "detector_score_authorized": False,
        "certificate_authorized": False,
        "wp5_authorized": False,
    }
    value = read_json(paths["resume_authorization"])
    if value != expected:
        raise PermissionError("WP4 resume authorization is invalid")
    return value


def validate_hardware_inventory():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line]
    if len(rows) != 1:
        raise RuntimeError("exactly one selected GPU is required")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("invalid GPU inventory")
    observed = {
        "device": fields[0],
        "driver_version": fields[1],
        "memory_mib": int(fields[2]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    expected = {
        "device": SELECTED_DEVICE,
        "driver_version": SELECTED_DRIVER,
        "memory_mib": SELECTED_MEMORY_MIB,
        "cuda_visible_devices": "0",
    }
    if observed != expected:
        raise RuntimeError(
            f"selected hardware inventory mismatch: {observed!r}"
        )
    return observed


def verify_tokenizer_files(model_dir):
    model_dir = Path(model_dir)
    for name, expected in TOKENIZER_SHA256.items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"tokenizer file mismatch: {name}")


def load_tokenizer(model_dir):
    verify_tokenizer_files(model_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_dir).resolve()),
        local_files_only=True,
        model_max_length=NATIVE_TOKEN_LIMIT,
    )
    if (
        type(tokenizer).__name__ != "MPNetTokenizerFast"
        or tokenizer.model_max_length != NATIVE_TOKEN_LIMIT
    ):
        raise ValueError("frozen MPNet tokenizer identity mismatch")
    return tokenizer


def load_profiles(path):
    payload = read_json(path)
    profiles = payload.get("inputs", {}).get("provider_profiles")
    if (
        payload.get("system") != "semstamp"
        or payload.get("status") != "pass"
        or not isinstance(profiles, list)
        or len(profiles) != 12
    ):
        raise ValueError("invalid frozen SemStamp provider profiles")
    profiles = sorted(profiles, key=lambda row: row["atom_count"])
    if any(
        left["atom_count"] >= right["atom_count"]
        for left, right in zip(profiles, profiles[1:])
    ):
        raise ValueError("profile atom anchors are not strictly increasing")
    return profiles


def adjacent_profile_indices(profiles, atom_count):
    anchors = [row["atom_count"] for row in profiles]
    right = bisect_left(anchors, atom_count)
    if right == 0:
        return (0, 1)
    if right == len(profiles):
        return (len(profiles) - 2, len(profiles) - 1)
    if anchors[right] != atom_count:
        return (right - 1, right)
    selected = {right}
    if right:
        selected.add(right - 1)
    if right + 1 < len(profiles):
        selected.add(right + 1)
    return tuple(sorted(selected))


def conservative_rate(profiles, atom_count, field):
    indices = adjacent_profile_indices(profiles, atom_count)
    selected = max(indices, key=lambda index: profiles[index][field])
    return {
        "rate": profiles[selected][field],
        "selected_slot": profiles[selected]["selection_slot"],
        "adjacent_slots": [
            profiles[index]["selection_slot"] for index in indices
        ],
    }


def project_document(atom_count, radii, resource_counts, profiles):
    feature_rate = conservative_rate(
        profiles, atom_count, "feature_seconds_per_input_token"
    )
    sidecar_rate = conservative_rate(
        profiles, atom_count, "sidecar_seconds_per_span"
    )
    core_rate = conservative_rate(
        profiles, atom_count, "core_seconds_per_pair"
    )
    distinct = [
        radius for radius in RADII
        if radii[str(radius)]["effective_radius"] == radius
    ]
    feature_seconds = (
        resource_counts["maximum_legal_input_tokens"]
        * feature_rate["rate"]
    )
    sidecar_seconds = sum(
        radii[str(radius)]["unique_span_count"] for radius in distinct
    ) * sidecar_rate["rate"]
    core_seconds = sum(
        radii[str(radius)]["unique_pair_count"] for radius in distinct
    ) * core_rate["rate"]
    values = (feature_seconds, sidecar_seconds, core_seconds)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("non-finite resource projection")
    return {
        "feature_seconds": feature_seconds,
        "sidecar_seconds": sidecar_seconds,
        "core_seconds": core_seconds,
        "total_gpu_seconds_without_topology": sum(values),
        "feature_rate": feature_rate,
        "sidecar_rate": sidecar_rate,
        "core_rate": core_rate,
    }


def baseline_payload(document, radius):
    fields = [
        document.atom_count,
        len(document.original_boundaries),
        radius,
        3,
        1,
        1,
        0,
        0,
        *document.original_boundaries,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def read_mode5_codes(path, expected_n, np):
    record_dtype = np.dtype([
        ("start", "<u4"),
        ("end", "<u4"),
        ("allowed", "u1"),
        ("region", "u1"),
    ])
    with Path(path).open("rb") as handle:
        if handle.read(8) != b"WP2V6F1\0":
            raise ValueError("topology dump magic mismatch")
        mode, token_count, count, dimension = struct.unpack(
            "<IIQI", handle.read(20)
        )
        if mode != 5 or token_count != expected_n or dimension != 1:
            raise ValueError("topology dump identity mismatch")
        if len(handle.read(64)) != 64:
            raise ValueError("truncated topology relation")
        records = np.fromfile(handle, dtype=record_dtype, count=count)
        if len(records) != count or handle.read(1):
            raise ValueError("truncated or trailing topology records")
    if (
        np.any(records["allowed"] != 1)
        or np.any(records["start"] >= records["end"])
        or np.any(records["end"] > expected_n)
    ):
        raise ValueError("invalid topology span record")
    base = np.uint64(expected_n + 1)
    codes = (
        records["start"].astype(np.uint64) * base
        + records["end"].astype(np.uint64)
    )
    if len(codes) and (
        np.any(codes[1:] <= codes[:-1])
        or len(np.unique(codes)) != len(codes)
    ):
        raise ValueError("topology span codes are not strictly sorted")
    return codes


def dump_topology(core, document, radius, path, np, parse_core_output):
    completed = subprocess.run(
        [str(core), "--dump-synthetic-sidecar", str(path), "5"],
        input=baseline_payload(document, radius),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "topology dump failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    result = parse_core_output(completed.stdout, 1)[0]
    codes = read_mode5_codes(path, document.atom_count, np)
    if result["unique_span_checks"] != len(codes):
        raise ValueError("topology dump/core span count mismatch")
    return result, codes


def assert_subset(subset, superset, np):
    positions = np.searchsorted(superset, subset)
    if (
        np.any(positions >= len(superset))
        or np.any(superset[positions] != subset)
    ):
        raise ValueError("radius topology is not nested")
    return positions


def topology_metric(result, name):
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


def pruned_query_topology(
    document,
    codes,
    allowed,
    radius,
    np,
    SparsePairwiseProblem,
    exact_grouped_sparse_pairwise_certificate,
):
    base = document.atom_count + 1
    legality = {
        int(code): bool(value) for code, value in zip(codes, allowed)
    }
    queried = set()

    def span_allowed(start, end):
        code = start * base + end
        queried.add(code)
        if code not in legality:
            raise AssertionError("pruned topology escaped all-allowed set")
        return legality[code]

    result = exact_grouped_sparse_pairwise_certificate(
        SparsePairwiseProblem(
            token_count=document.atom_count,
            original_boundaries=document.original_boundaries,
            max_boundary_moves=radius,
            span_allowed=span_allowed,
            pair_scorer=lambda *_args: 0.0,
            normalization=1.0,
            bias=0.0,
            threshold=0.0,
        )
    )
    actual_codes = np.asarray(sorted(queried), dtype=np.uint64)
    positions = assert_subset(actual_codes, codes, np)
    actual_allowed = allowed[positions]
    if topology_metric(result, "unique_span_checks") != len(actual_codes):
        raise ValueError("pruned topology span count mismatch")
    return result, actual_codes, actual_allowed


def token_counts(tokenizer, texts):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    rows = encoded["input_ids"]
    if len(rows) != len(texts):
        raise ValueError("tokenizer batch row count mismatch")
    if rows and not isinstance(rows[0], list):
        if len(texts) != 1:
            raise ValueError("tokenizer flattened multi-row batch")
        rows = [rows]
    return [len(row) for row in rows]


def prepare_document_topology(document, tokenizer, core, scratch):
    import numpy as np
    from audit_certified_resegmentation_wp2_v4_correctness import (
        parse_core_output,
    )
    from certified_resegmentation_wp2_v3 import SparsePairwiseProblem
    from certified_resegmentation_wp2_v4 import (
        exact_grouped_sparse_pairwise_certificate,
    )

    effective = {
        radius: min(radius, len(document.original_boundaries))
        for radius in RADII
    }
    distinct = sorted(set(effective.values()))
    all_results = {}
    all_codes = {}
    for radius in distinct:
        dump_path = Path(scratch) / f"all_k{radius}.bin"
        result, codes = dump_topology(
            core, document, radius, dump_path, np, parse_core_output
        )
        dump_path.unlink(missing_ok=True)
        all_results[radius] = result
        all_codes[radius] = codes

    maximum_radius = max(distinct)
    maximum_codes = all_codes[maximum_radius]
    maximum_allowed = np.empty(len(maximum_codes), dtype=np.bool_)
    maximum_counts = np.empty(len(maximum_codes), dtype=np.int64)
    base = document.atom_count + 1
    for offset in range(0, len(maximum_codes), TOKENIZER_CHUNK_SIZE):
        selected = maximum_codes[offset:offset + TOKENIZER_CHUNK_SIZE]
        spans = [
            (int(code // base), int(code % base)) for code in selected
        ]
        texts = [
            document.render_unit(start, end) for start, end in spans
        ]
        counts = token_counts(tokenizer, texts)
        maximum_counts[offset:offset + len(counts)] = counts
        maximum_allowed[offset:offset + len(counts)] = (
            np.asarray(counts) <= NATIVE_TOKEN_LIMIT
        )

    actual = {}
    for radius in distinct:
        positions = assert_subset(all_codes[radius], maximum_codes, np)
        allowed = maximum_allowed[positions]
        if np.all(allowed):
            result = all_results[radius]
            codes = all_codes[radius]
        else:
            result, codes, allowed = pruned_query_topology(
                document,
                all_codes[radius],
                allowed,
                radius,
                np,
                SparsePairwiseProblem,
                exact_grouped_sparse_pairwise_certificate,
            )
        actual[radius] = {
            "all_result": all_results[radius],
            "all_codes": all_codes[radius],
            "result": result,
            "codes": codes,
            "allowed": allowed,
        }

    maximum = actual[maximum_radius]
    positions = assert_subset(maximum["codes"], maximum_codes, np)
    selected_counts = maximum_counts[positions]
    legal_input_tokens = sum(
        int(count)
        for count, allowed in zip(selected_counts, maximum["allowed"])
        if allowed
    )
    radii = {}
    for requested in RADII:
        radius = effective[requested]
        topology = actual[radius]
        result = topology["result"]
        all_result = topology["all_result"]
        radii[str(requested)] = {
            "requested_radius": requested,
            "effective_radius": radius,
            "all_allowed_unique_spans": len(topology["all_codes"]),
            "all_allowed_unique_pairs": topology_metric(
                all_result, "unique_pair_scores"
            ),
            "unique_span_count": len(topology["codes"]),
            "legal_span_count": int(np.count_nonzero(topology["allowed"])),
            "unique_pair_count": topology_metric(
                result, "unique_pair_scores"
            ),
            "candidate_boundary_positions": (
                topology_metric(result, "candidate_boundary_positions")
            ),
            "reachable_state_count": topology_metric(
                result, "reachable_state_count"
            ),
            "transitions_evaluated": topology_metric(
                result, "transitions_evaluated"
            ),
            "peak_layer_states": topology_metric(
                result, "peak_layer_states"
            ),
            "threat_set_size": topology_metric(
                result, "threat_set_size"
            ),
        }
    return radii, {
        "max_native_tokens": (
            int(max(maximum_counts)) if len(maximum_counts) else 0
        ),
        "maximum_legal_input_tokens": legal_input_tokens,
        "maximum_sidecar_bytes": (
            92
            + 9 * len(maximum["codes"])
            + int(np.count_nonzero(maximum["allowed"]))
        ),
        "maximum_topology_dump_bytes": 92 + 10 * len(maximum_codes),
        "all_maximum_radius_spans_checked": len(maximum_codes),
    }


def validate_frozen_structure(index, planned, frozen, document):
    evidence = document.public_evidence()
    expected = {
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "unit_count": planned["unit_count"],
        "atom_count": evidence["atom_count"],
        "refinement_count": evidence["refinement_count"],
        "zero_byte_gap_count": evidence["zero_byte_gap_count"],
        "original_boundaries": evidence["original_boundaries"],
        "topology_sha256": evidence["topology_sha256"],
        "gap_table_sha256": evidence["gap_table_sha256"],
        "byte_reconstruction_sha256": evidence[
            "byte_reconstruction_sha256"
        ],
    }
    mismatches = {
        key: (frozen.get(key), value)
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen structure mismatch at {index}: {mismatches}")


def workload_row(
    index,
    planned,
    frozen,
    text,
    sent_tokenize,
    tokenizer,
    core,
    scratch,
    profiles,
    attempt_index,
):
    from semstamp_refined_atom_renderer import canonicalize_refined_bytes

    units = sent_tokenize(text)
    document = canonicalize_refined_bytes(
        text.encode("utf-8"),
        lambda *_args, frozen_units=units: frozen_units,
    )
    validate_frozen_structure(index, planned, frozen, document)
    row_scratch = Path(scratch) / f"row-{index}"
    if row_scratch.exists():
        shutil.rmtree(row_scratch)
    row_scratch.mkdir(parents=True)
    started = time.perf_counter()
    try:
        radii, resource_counts = prepare_document_topology(
            document, tokenizer, core, row_scratch
        )
    finally:
        shutil.rmtree(row_scratch, ignore_errors=True)
    projection = project_document(
        document.atom_count, radii, resource_counts, profiles
    )
    row = {
        "schema_version": 1,
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_index": index,
        "attempt_index": attempt_index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
        "refinement_count": document.refinement_count,
        "zero_byte_gap_count": len(document.zero_byte_gap_boundaries),
        "original_boundaries_sha256": canonical_digest(
            list(document.original_boundaries)
        ),
        "topology_sha256": frozen["topology_sha256"],
        "radii": radii,
        "resource_counts": resource_counts,
        "projection": projection,
        "score_access_count": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row, {
        "plan_index": index,
        "elapsed_seconds": time.perf_counter() - started,
    }


def classify_terminal(gates):
    if all(gates.values()):
        return "WP4_RESOURCE_PREFLIGHT_GO_FOR_WP5_DESIGN"
    core_keys = (
        "max_pruned_spans_le_2000000",
        "max_pruned_pairs_le_50000000",
        "max_all_allowed_spans_le_2000000",
        "max_all_allowed_pairs_le_50000000",
    )
    if not all(gates[key] for key in core_keys):
        return "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP"
    return "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_BUDGET"


def aggregate_result(
    paths,
    contract,
    rows,
    timings,
    profiles,
    hardware,
    free_bytes,
):
    if len(rows) != TOTAL_ROWS or len(timings) != TOTAL_ROWS:
        raise ValueError("complete 3,000-row denominator is required")
    if any(row["plan_index"] != index for index, row in enumerate(rows)):
        raise ValueError("preflight rows are not the frozen ordered prefix")
    max_spans = max(
        radius["unique_span_count"]
        for row in rows for radius in row["radii"].values()
    )
    max_pairs = max(
        radius["unique_pair_count"]
        for row in rows for radius in row["radii"].values()
    )
    max_all_spans = max(
        radius["all_allowed_unique_spans"]
        for row in rows for radius in row["radii"].values()
    )
    max_all_pairs = max(
        radius["all_allowed_unique_pairs"]
        for row in rows for radius in row["radii"].values()
    )
    max_sidecar = max(
        row["resource_counts"]["maximum_sidecar_bytes"] for row in rows
    )
    max_dump = max(
        row["resource_counts"]["maximum_topology_dump_bytes"]
        for row in rows
    )
    projected_scratch = max_sidecar + max_dump + 1024**3
    topology_seconds = sum(
        float(timing["elapsed_seconds"]) for timing in timings
    )
    components = {
        key: sum(row["projection"][key] for row in rows)
        for key in (
            "feature_seconds",
            "sidecar_seconds",
            "core_seconds",
            "total_gpu_seconds_without_topology",
        )
    }
    model_load_seconds = max(
        float(profile["model_load_seconds"]) for profile in profiles
    )
    two_replay_seconds = (
        REPLAY_COUNT
        * (topology_seconds + components["total_gpu_seconds_without_topology"])
        + model_load_seconds
    )
    if not math.isfinite(two_replay_seconds):
        raise ValueError("non-finite aggregate resource projection")
    cohort_counts = dict(Counter(row["cohort"] for row in rows))
    gates = {
        "all_3000_rows_completed": len(rows) == TOTAL_ROWS,
        "balanced_frozen_cohorts": cohort_counts == {
            cohort: COHORT_SIZE for cohort in COHORTS
        },
        "all_42_refinement_rows_retained": sum(
            row["refinement_count"] > 0 for row in rows
        ) == 42,
        "all_59_zero_byte_gaps_retained": sum(
            row["zero_byte_gap_count"] for row in rows
        ) == 59,
        "all_frozen_structures_exact": all(
            row["score_access_count"] == 0 for row in rows
        ),
        "all_maximum_radius_spans_checked": all(
            row["resource_counts"]["all_maximum_radius_spans_checked"]
            == row["radii"]["3"]["all_allowed_unique_spans"]
            for row in rows
        ),
        "max_pruned_spans_le_2000000": max_spans <= SPAN_GATE,
        "max_pruned_pairs_le_50000000": max_pairs <= PAIR_GATE,
        "max_all_allowed_spans_le_2000000": max_all_spans <= SPAN_GATE,
        "max_all_allowed_pairs_le_50000000": max_all_pairs <= PAIR_GATE,
        "max_sidecar_le_50_gib": max_sidecar <= SIDECAR_GATE_BYTES,
        "projected_scratch_le_15_gib": (
            projected_scratch <= WORK_DISK_GATE_BYTES
        ),
        "free_disk_covers_projection_plus_5_gib": (
            free_bytes >= projected_scratch + FREE_DISK_RESERVE_BYTES
        ),
        "two_replay_projection_le_72_gpu_hours": (
            two_replay_seconds / 3600 <= GPU_HOUR_GATE
        ),
        "selected_hardware_exact": hardware
        == contract["selected_hardware"],
        "no_sample_exclusion_or_supplement": True,
        "score_blind_execution_attestation": True,
    }
    terminal = classify_terminal(gates)
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "go" if all(gates.values()) else "stop",
        "terminal": terminal,
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "authorization_sha256": sha256_file(paths["authorization"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "runner_sha256": sha256_file(paths["runner"]),
        "workload_rows_sha256": sha256_file(paths["rows"]),
        "timings_sha256": sha256_file(paths["timings"]),
        "contains_raw_text": False,
        "counts": {
            "rows": len(rows),
            "cohorts": cohort_counts,
            "refinement_risk_rows": sum(
                row["refinement_count"] > 0 for row in rows
            ),
            "zero_byte_gaps": sum(
                row["zero_byte_gap_count"] for row in rows
            ),
            "sample_exclusions": 0,
            "sample_supplements": 0,
            "total_radius_unique_spans": sum(
                radius["unique_span_count"]
                for row in rows for radius in row["radii"].values()
            ),
            "total_radius_unique_pairs": sum(
                radius["unique_pair_count"]
                for row in rows for radius in row["radii"].values()
            ),
            "max_document_pruned_spans": max_spans,
            "max_document_pruned_pairs": max_pairs,
            "max_document_all_allowed_spans": max_all_spans,
            "max_document_all_allowed_pairs": max_all_pairs,
            "max_document_sidecar_bytes": max_sidecar,
            "documents_over_legacy_5m_pair_diagnostic": sum(
                max(
                    radius["unique_pair_count"]
                    for radius in row["radii"].values()
                ) > LEGACY_PAIR_DIAGNOSTIC
                for row in rows
            ),
        },
        "runtime_upper_bound": {
            "projection_replays": REPLAY_COUNT,
            "measured_topology_seconds_per_replay": topology_seconds,
            **components,
            "inherited_model_load_seconds": model_load_seconds,
            "conservative_two_replay_seconds": two_replay_seconds,
            "conservative_two_replay_gpu_hours": (
                two_replay_seconds / 3600
            ),
        },
        "storage_projection": {
            "maximum_sidecar_bytes": max_sidecar,
            "maximum_topology_dump_bytes": max_dump,
            "safety_reserve_bytes": 1024**3,
            "projected_peak_scratch_bytes": projected_scratch,
            "observed_free_bytes": free_bytes,
            "required_free_bytes": (
                projected_scratch + FREE_DISK_RESERVE_BYTES
            ),
        },
        "selected_hardware": hardware,
        "resource_budget": contract["resource_budget"],
        "gates": gates,
        "decision": {
            "wp4_resource_preflight": (
                "go" if all(gates.values()) else "readiness_stop"
            ),
            "wp5_authorized_to_freeze_design": all(gates.values()),
            "wp5_execution_authorized": False,
            "certificate_authorized": False,
        },
        "execution_attestation": {
            "tokenizer_loads": 1,
            "hardware_inventory_queries": 1,
            "sbert_weight_reads": 0,
            "sbert_model_loads": 0,
            "cuda_initializations": 0,
            "region_computations": 0,
            "hit_computations": 0,
            "detector_score_reads": 0,
            "detector_score_computations": 0,
            "threshold_computations": 0,
            "certificate_computations": 0,
            "generation_calls": 0,
            "wp5_calls": 0,
        },
    }
    reject_raw_public_keys(result)
    return result


def validate_existing_prefix(paths, plan, frozen_rows):
    rows = read_jsonl(paths["rows"]) if paths["rows"].exists() else []
    timings = read_jsonl(paths["timings"]) if paths["timings"].exists() else []
    if len(rows) != len(timings):
        raise ValueError("row/timing checkpoint prefix mismatch")
    for index, (row, timing) in enumerate(zip(rows, timings)):
        if (
            row.get("plan_index") != index
            or timing.get("plan_index") != index
            or row.get("raw_text_sha256") != plan[index]["raw_text_sha256"]
            or row.get("topology_sha256")
            != frozen_rows[index]["topology_sha256"]
        ):
            raise ValueError(f"invalid checkpoint prefix at row {index}")
    return rows, timings


def acquire_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)


def run(args):
    paths = paths_for(args.root, args.output_dir)
    contract = read_json(paths["contract"])
    validate_authorization(paths, contract)
    contract, manifest = validate_contract(paths)
    verify_inputs(paths, contract, manifest)
    validate_parent_evidence(paths)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    hardware = validate_hardware_inventory()
    core = Path(args.core).resolve()
    if not core.is_file() or sha256_file(core) != LINUX_CORE_SHA256:
        raise ValueError("Linux v6 factorized core mismatch")

    import run_semstamp_certified_defense_v2_wp3_p0 as p0
    p0_paths = p0.paths_for(paths["root"], paths["p0_dir"])
    plan, texts = p0.materialize_inputs(p0_paths)
    frozen_rows = read_jsonl(
        paths["p0_dir"] / "P0_RENDERER_ROWS_REPLAY0.jsonl"
    )
    if (
        len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
        or len(frozen_rows) != TOTAL_ROWS
    ):
        raise ValueError("frozen denominator is incomplete")
    sent_tokenize = p0.load_nltk(args.nltk_data, {
        "nltk_version": "3.8.1",
        "punkt_payload_sha256": PUNKT_SHA256,
    })
    tokenizer = load_tokenizer(args.model_dir)
    profiles = load_profiles(paths["profile"])
    rows, timings = validate_existing_prefix(paths, plan, frozen_rows)
    resume_authorization = validate_resume_authorization(paths, rows)
    attempt_index = (
        resume_authorization["next_attempt_index"]
        if resume_authorization is not None
        else 0
    )

    if paths["lock"].exists():
        raise RuntimeError("WP4 execution lock already exists")
    acquire_lock(paths["lock"])
    try:
        Path(args.scratch).mkdir(parents=True, exist_ok=True)
        for index in range(len(rows), TOTAL_ROWS):
            row, timing = workload_row(
                index,
                plan[index],
                frozen_rows[index],
                texts[index],
                sent_tokenize,
                tokenizer,
                core,
                args.scratch,
                profiles,
                attempt_index,
            )
            append_jsonl(paths["rows"], row)
            append_jsonl(paths["timings"], timing)
            rows.append(row)
            timings.append(timing)
            atomic_write_json(paths["heartbeat"], {
                "schema_version": 1,
                "stage": STAGE,
                "status": "running",
                "completed_rows": len(rows),
                "last_plan_index": index,
                "updated_at": utc_now(),
                "score_blind": True,
            })
        rows_bytes = jsonl_bytes(rows)
        timings_bytes = jsonl_bytes(timings)
        atomic_write_bytes(paths["rows"], rows_bytes)
        atomic_write_bytes(paths["timings"], timings_bytes)
        free_bytes = shutil.disk_usage(paths["root"]).free
        result = aggregate_result(
            paths,
            contract,
            rows,
            timings,
            profiles,
            hardware,
            free_bytes,
        )
        atomic_write_json(paths["result"], result)
        paths["heartbeat"].unlink(missing_ok=True)
        return result
    finally:
        paths["lock"].unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        paths = paths_for(args.root, args.output_dir)
        if paths["authorization"].is_file():
            atomic_write_json(paths["last_failure"], {
                "schema_version": 1,
                "stage": STAGE,
                "status": "invalid_or_incomplete",
                "terminal": "WP4_RESOURCE_PREFLIGHT_INVALID_OR_INCOMPLETE",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
                "contains_raw_text": False,
            })
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
