"""Run the frozen WP2-v18 SemStamp B1 exact-certificate experiment."""

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import stat
import tempfile
import time

import numpy as np

from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
)
from certified_resegmentation_wp2_v3 import canonicalize_structured
import run_certified_resegmentation_wp2_v10_semstamp as v10
import run_certified_resegmentation_wp2_v17_b0 as b0


STAGE = "WP2-v18-B1-SemStamp-exact-certificate"
PROTOCOL_SHA256 = (
    "0da27bc53e06e76ba5f1d25aefe3bf787893c0fa4421113dddb1a995892ab82d"
)
B0_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
B0_AUDIT_SHA256 = (
    "bb152276a01c1b7a78671061a30ac6219a3905fcafebb454ba2a270782a8f342"
)
B0_TERMINAL_SHA256 = (
    "58f28e7088100548b95e2ab0e880e8333f30695ca24bc5bbbf44d2f2d52677d1"
)
B0_PLAN_SHA256 = (
    "a1427a9978751c756ad5072aeb45bb6cb393713d8339d62ff6ef4397591fc4e2"
)
B0_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
B0_CHECKPOINT_SHA256 = (
    "b4fbd2d9d610fe170f1a4700ac35c2b1c405b1a145e8bc1d32cd7d7ec0f8d909"
)
PROFILE_SHA256 = (
    "4701a4b6d77f2d0c20e189853ec896d70ac205df05659ee38e576b5010119f6d"
)
CORE_SOURCE_SHA256 = (
    "d77d23c0e44a343c154e93d9897ee3e40c1a05194d9200108ff7e619f52e1360"
)
LINUX_CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
DEPENDENCY_SHA256 = {
    "scripts/audit_certified_resegmentation_wp2_v4_correctness.py": (
        "e6e3cc41c4c10844f16932652223674f3999739356e37c8c0511d2891eb703d2"
    ),
    "scripts/certified_resegmentation_wp2_v3.py": (
        "31e89a3be64780ae3d5537fc93ac4715b2ed3a88427fac20b0fe2007831a39d3"
    ),
    "scripts/certified_resegmentation_wp2_v4.py": (
        "fb6ee4aa1ce944ec9c6a6a96ca103bcdf68ad2d8f4e67444c5337786c83853ec"
    ),
    "scripts/certified_resegmentation_wp2_v6_features.py": (
        "16ba74b614780e7d64f0bc51d14b0fd554b4ed5cc51d4a3af4f54f2cf2484ae3"
    ),
    "scripts/certified_resegmentation_wp2_v13_b0_support.py": (
        "692f26230b7f335f715b23a05d2984d617d064e89db0d7ad0fa23fc4c2025166"
    ),
    "scripts/run_certified_resegmentation_wp2_v10_semstamp.py": (
        "f003aff6aa8245a7fdb619c7a5a7af5544fb1fb51de2df0de2ab6774b6d86583"
    ),
    "scripts/run_certified_resegmentation_wp2_v17_b0.py": (
        "a15c7bfc65b6b7bea62bbeb012484532e389972e7d6d5bc45833e2a98d0f5c72"
    ),
}
B0_THRESHOLD = 1.6666666666666667
TOTAL_ROWS = 3000
COHORT_SIZE = 1000
RADII = tuple(range(4))
TARGET_FPR = 0.05
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
LEGACY_PAIR_DIAGNOSTIC = 5_000_000
CACHE_GATE_BYTES = 50 * 1024**3
GPU_HOUR_GATE = 72.0
MIN_FREE_GIB = 5.0
BATCH_SIZE = 64
TOKENIZER_CHUNK_SIZE = 512
TEXT_CHUNK_SIZE = 4096
REPLAY_COUNT = 2
_WORKER = {}


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


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def file_metadata(path, root=None):
    path = Path(path)
    output = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if root is not None:
        output["relative_path"] = str(
            path.resolve().relative_to(Path(root).resolve())
        )
    return output


def verify_file(path, expected_sha256, label, mode=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise ValueError(f"{label} mode mismatch")


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside experiment root")
    v17_dir = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "B1_CONTRACT.json",
        "workload_rows": output_dir / "P0_WORKLOAD_ROWS.jsonl",
        "workload_timings": output_dir / "sealed/p0_workload_timings.jsonl",
        "readiness": output_dir / "P0_READINESS.json",
        "heartbeat": output_dir / "B1_HEARTBEAT.json",
        "runtime": output_dir / "B1_RUNTIME.json",
        "result": output_dir / "B1_RESULT.json",
        "rows0": output_dir / "B1_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "B1_ROWS_REPLAY1.jsonl",
        "v17_dir": v17_dir,
        "b0_result": v17_dir / "B0_RESULT.json",
        "b0_audit": v17_dir / "B0_AUDIT.json",
        "b0_terminal": v17_dir / "TERMINAL.sha256",
        "b0_plan": v17_dir / "B0_PLAN.jsonl",
        "b0_rows": v17_dir / "B0_ROWS_REPLAY0.jsonl",
        "b0_checkpoint": v17_dir / "sealed/b0_checkpoint.json",
        "profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1"
            / "semstamp_workload.json"
        ),
        "core_source": (
            root / "scripts/certified_resegmentation_wp2_v6_core.cpp"
        ),
        "runner": Path(__file__).resolve(),
    }


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen",
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(paths["runner"]),
        "core_source_sha256": CORE_SOURCE_SHA256,
        "b0_result_sha256": B0_RESULT_SHA256,
        "b0_audit_sha256": B0_AUDIT_SHA256,
        "b0_plan_sha256": B0_PLAN_SHA256,
        "b0_rows_sha256": B0_ROWS_SHA256,
        "b0_checkpoint_sha256": B0_CHECKPOINT_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "formal_linux_core_sha256": LINUX_CORE_SHA256,
        "radii": list(RADII),
        "replay_count": REPLAY_COUNT,
        "span_gate": SPAN_GATE,
        "pair_gate": PAIR_GATE,
        "legacy_pair_diagnostic": LEGACY_PAIR_DIAGNOSTIC,
        "gpu_hour_gate": GPU_HOUR_GATE,
        "minimum_free_gib": MIN_FREE_GIB,
        "embedding_batch_size": BATCH_SIZE,
        "tokenizer_chunk_size": TOKENIZER_CHUNK_SIZE,
        "text_chunk_size": TEXT_CHUNK_SIZE,
        "dependency_sha256": DEPENDENCY_SHA256,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in required.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B1 contract mismatch: {mismatches}")
    for relative_path, expected in DEPENDENCY_SHA256.items():
        if sha256_file(paths["root"] / relative_path) != expected:
            raise ValueError(f"B1 dependency changed: {relative_path}")
    for key in ("auditor_sha256", "test_sha256"):
        value = contract.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"B1 contract lacks {key}")
    return contract


def validate_b0(paths):
    for path, expected, label, mode in (
        (paths["protocol"], PROTOCOL_SHA256, "B1 protocol", None),
        (paths["b0_result"], B0_RESULT_SHA256, "B0 result", None),
        (paths["b0_audit"], B0_AUDIT_SHA256, "B0 audit", None),
        (paths["b0_terminal"], B0_TERMINAL_SHA256, "B0 terminal", None),
        (paths["b0_plan"], B0_PLAN_SHA256, "B0 plan", None),
        (paths["b0_rows"], B0_ROWS_SHA256, "B0 rows", None),
        (
            paths["b0_checkpoint"],
            B0_CHECKPOINT_SHA256,
            "B0 checkpoint",
            0o600,
        ),
        (paths["profile"], PROFILE_SHA256, "v6 profile", None),
        (
            paths["core_source"],
            CORE_SOURCE_SHA256,
            "v6 core source",
            None,
        ),
    ):
        verify_file(path, expected, label, mode)
    result = read_json(paths["b0_result"])
    audit = read_json(paths["b0_audit"])
    if (
        result.get("status") != "go"
        or result.get("decision", {}).get("b1_exact_certificate")
        != "authorized_to_freeze_separate_protocol"
        or result.get("decision", {}).get("b1_started") is not False
        or result.get("statistics", {}).get("calibration", {}).get("threshold")
        != B0_THRESHOLD
        or not all(result.get("gates", {}).values())
        or audit.get("status") != "go"
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("v17 B0 is not the frozen audited Go")
    plan = read_jsonl(paths["b0_plan"])
    b0_rows = read_jsonl(paths["b0_rows"])
    if len(plan) != TOTAL_ROWS or len(b0_rows) != TOTAL_ROWS:
        raise ValueError("v17 B0 row count mismatch")
    for index, (planned, scored) in enumerate(zip(plan, b0_rows)):
        if (
            planned.get("plan_index") != index
            or scored.get("plan_index") != index
            or planned.get("cohort") != scored.get("cohort")
            or planned.get("cohort_index") != scored.get("cohort_index")
            or planned.get("raw_text_sha256")
            != scored.get("raw_text_sha256")
            or planned.get("unit_count") != scored.get("unit_count")
        ):
            raise ValueError(f"v17 B0 row identity mismatch at {index}")
    return result, plan, b0_rows


def materialize_documents(paths, model_dir, nltk_data):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    b0_paths = b0.paths_for(paths["root"], paths["v17_dir"])
    b0.verified_metadata(b0_paths, model_dir)
    _, rebuilt_plan, texts = b0.materialize_inputs(b0_paths)
    if canonical_digest(rebuilt_plan) != canonical_digest(read_jsonl(
        paths["b0_plan"]
    )):
        raise ValueError("rebuilt B0 plan mismatch")
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    nltk.data.find("tokenizers/punkt/english.pickle")
    documents = [
        canonicalize_structured(text, sentence_splitter=sent_tokenize)
        for text in texts
    ]
    return rebuilt_plan, texts, documents


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_dir).resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if (
        type(tokenizer).__name__ != "MPNetTokenizerFast"
        or tokenizer.model_max_length != 512
    ):
        raise ValueError("SemStamp tokenizer identity mismatch")
    return tokenizer


def batched_token_counts(tokenizer, texts):
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
            raise ValueError("tokenizer flattened a multi-row batch")
        rows = [rows]
    return np.asarray([len(row) for row in rows], dtype=np.int64)


def native_legality(document, codes, tokenizer, chunk_size):
    allowed = np.empty(len(codes), dtype=np.bool_)
    counts = np.empty(len(codes), dtype=np.int64)
    base = document.atom_count + 1
    for offset in range(0, len(codes), chunk_size):
        selected = codes[offset:offset + chunk_size]
        spans = [
            (int(code // base), int(code % base))
            for code in selected
        ]
        texts = [
            document.preprocess_unit("semstamp", start, end)
            for start, end in spans
        ]
        values = batched_token_counts(tokenizer, texts)
        counts[offset:offset + len(values)] = values
        allowed[offset:offset + len(values)] = values <= 512
    return allowed, counts


def renderer_checks(document):
    endpoints = (0, *document.original_boundaries, document.atom_count)
    rendered = tuple(
        document.render_unit(endpoints[index], endpoints[index + 1])
        for index in range(document.unit_count)
    )
    return {
        "original_partition_exact": rendered == document.original_units,
        "original_units_sha256": canonical_digest(list(rendered)),
    }


def prepare_document_topologies(document, tokenizer, core, scratch, chunk_size):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    effective = {
        radius: min(radius, len(document.original_boundaries))
        for radius in RADII
    }
    distinct = sorted(set(effective.values()))
    all_results = {}
    all_codes = {}
    for radius in distinct:
        path = scratch / f"all_k{radius}.bin"
        result, codes = v10.dump_topology(core, document, radius, path)
        path.unlink(missing_ok=True)
        all_results[radius] = result
        all_codes[radius] = codes

    maximum_radius = max(distinct)
    maximum_all_codes = all_codes[maximum_radius]
    maximum_allowed, maximum_counts = native_legality(
        document,
        maximum_all_codes,
        tokenizer,
        chunk_size,
    )
    actual = {}
    for radius in distinct:
        positions = v10.assert_subset(
            all_codes[radius],
            maximum_all_codes,
        )
        allowed = maximum_allowed[positions]
        if np.all(allowed):
            result = all_results[radius]
            codes = all_codes[radius]
        else:
            result, codes, allowed = v10.pruned_query_topology(
                document,
                all_codes[radius],
                allowed,
                radius,
            )
        actual[radius] = {
            "all_result": all_results[radius],
            "all_codes": all_codes[radius],
            "result": result,
            "codes": codes,
            "allowed": allowed,
        }

    maximum = actual[maximum_radius]
    maximum_positions = v10.assert_subset(
        maximum["codes"],
        maximum_all_codes,
    )
    maximum_token_counts = maximum_counts[maximum_positions]
    legal_token_total = int(sum(
        int(value)
        for value, allowed in zip(
            maximum_token_counts,
            maximum["allowed"],
        )
        if allowed
    ))
    rows = {}
    for requested in RADII:
        radius = effective[requested]
        topology = actual[radius]
        rows[str(requested)] = {
            "requested_radius": requested,
            "effective_radius": radius,
            "all_allowed_unique_spans": len(topology["all_codes"]),
            "all_allowed_unique_pairs": v10.topology_metric(
                topology["all_result"], "unique_pair_scores"
            ),
            "unique_span_count": len(topology["codes"]),
            "legal_span_count": int(np.count_nonzero(topology["allowed"])),
            "unique_pair_count": v10.topology_metric(
                topology["result"], "unique_pair_scores"
            ),
            "candidate_boundary_positions": v10.topology_metric(
                topology["result"], "candidate_boundary_positions"
            ),
            "reachable_state_count": v10.topology_metric(
                topology["result"], "reachable_state_count"
            ),
            "transitions_evaluated": v10.topology_metric(
                topology["result"], "transitions_evaluated"
            ),
            "peak_layer_states": v10.topology_metric(
                topology["result"], "peak_layer_states"
            ),
            "threat_set_size": v10.topology_metric(
                topology["result"], "threat_set_size"
            ),
        }
    return effective, distinct, actual, rows, {
        "max_native_tokens": (
            int(max(maximum_counts)) if len(maximum_counts) else 0
        ),
        "maximum_legal_input_tokens": legal_token_total,
        "maximum_sidecar_bytes": (
            92 + 9 * len(maximum["codes"])
            + int(np.count_nonzero(maximum["allowed"]))
        ),
    }


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
        raise ValueError("provider profile anchors are not strictly ordered")
    return profiles


def adjacent_profile_indices(profiles, atom_count):
    atoms = [row["atom_count"] for row in profiles]
    right = bisect_left(atoms, atom_count)
    if right == 0:
        return (0, 1)
    if right == len(profiles):
        return (len(profiles) - 2, len(profiles) - 1)
    if atoms[right] != atom_count:
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


def workload_estimate(atom_count, radii, resource_counts, profiles):
    feature_rate = conservative_rate(
        profiles, atom_count, "feature_seconds_per_input_token"
    )
    sidecar_rate = conservative_rate(
        profiles, atom_count, "sidecar_seconds_per_span"
    )
    core_rate = conservative_rate(
        profiles, atom_count, "core_seconds_per_pair"
    )
    feature_seconds = (
        resource_counts["maximum_legal_input_tokens"]
        * feature_rate["rate"]
    )
    sidecar_units = sum(
        radii[str(radius)]["unique_span_count"]
        for radius in RADII
        if radii[str(radius)]["effective_radius"] == radius
    )
    core_units = sum(
        radii[str(radius)]["unique_pair_count"]
        for radius in RADII
        if radii[str(radius)]["effective_radius"] == radius
    )
    sidecar_seconds = sidecar_units * sidecar_rate["rate"]
    core_seconds = core_units * core_rate["rate"]
    return {
        "feature_seconds": feature_seconds,
        "sidecar_seconds": sidecar_seconds,
        "core_seconds": core_seconds,
        "total_gpu_seconds_without_topology": (
            feature_seconds + sidecar_seconds + core_seconds
        ),
        "feature_rate": feature_rate,
        "sidecar_rate": sidecar_rate,
        "core_rate": core_rate,
    }


def initialize_readiness_worker(model_dir, core, scratch, chunk_size, profiles):
    _WORKER.update({
        "tokenizer": load_tokenizer(model_dir),
        "core": Path(core),
        "scratch": Path(scratch),
        "chunk_size": chunk_size,
        "profiles": profiles,
    })


def workload_task(task):
    index, planned, text = task
    from nltk.tokenize import sent_tokenize

    started = time.perf_counter()
    document = canonicalize_structured(text, sentence_splitter=sent_tokenize)
    if (
        document.raw_text_sha256 != planned["raw_text_sha256"]
        or document.unit_count != planned["unit_count"]
    ):
        raise ValueError(f"document identity mismatch at row {index}")
    renderer = renderer_checks(document)
    if not renderer["original_partition_exact"]:
        raise ValueError(f"renderer mismatch at row {index}")
    scratch = _WORKER["scratch"] / f"{os.getpid()}-{index}"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        (
            _effective,
            _distinct,
            _actual,
            radii,
            resource_counts,
        ) = prepare_document_topologies(
            document,
            _WORKER["tokenizer"],
            _WORKER["core"],
            scratch,
            _WORKER["chunk_size"],
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    estimate = workload_estimate(
        document.atom_count,
        radii,
        resource_counts,
        _WORKER["profiles"],
    )
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": document.raw_text_sha256,
        "tape_sha256": document.structural_tape_sha256,
        "original_units_sha256": renderer["original_units_sha256"],
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
        "edge_count": document.unit_count - 1,
        "original_boundaries": list(document.original_boundaries),
        "radii": radii,
        "resource_counts": resource_counts,
        "estimate": estimate,
        "contains_raw_text": False,
    }
    timing = {
        "plan_index": index,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return row, timing


def validate_workload_row(row, index, planned):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("plan_index") != index
        or row.get("cohort") != planned["cohort"]
        or row.get("cohort_index") != planned["cohort_index"]
        or row.get("raw_text_sha256") != planned["raw_text_sha256"]
        or row.get("unit_count") != planned["unit_count"]
        or row.get("edge_count") != planned["unit_count"] - 1
        or len(row.get("original_boundaries", [])) != row.get("edge_count")
        or row.get("original_boundaries")
        != sorted(set(row.get("original_boundaries", [])))
        or row.get("contains_raw_text") is not False
        or set(row.get("radii", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError(f"invalid workload row at {index}")
    for radius in RADII:
        current = row["radii"][str(radius)]
        if (
            current.get("requested_radius") != radius
            or current.get("effective_radius")
            != min(radius, row["edge_count"])
            or current.get("unique_span_count", 0) < 1
            or current.get("legal_span_count", 0) < 1
            or current["legal_span_count"] > current["unique_span_count"]
            or current["unique_span_count"]
            > current["all_allowed_unique_spans"]
            or current["unique_pair_count"]
            > current["all_allowed_unique_pairs"]
            or current.get("threat_set_size", 0) < 1
        ):
            raise ValueError(
                f"invalid workload topology at {index}, k={radius}"
            )
    return row


def collect_append_checkpoint(path, plan):
    path = Path(path)
    if not path.exists():
        return {}
    output = {}
    for row in read_jsonl(path):
        index = row.get("plan_index")
        if (
            not isinstance(index, int)
            or not 0 <= index < len(plan)
            or index in output
        ):
            raise ValueError("invalid workload checkpoint index")
        output[index] = validate_workload_row(row, index, plan[index])
    return output


def aggregate_readiness(
    paths,
    contract,
    plan,
    rows,
    timings,
    profiles,
    free_bytes,
):
    if len(rows) != TOTAL_ROWS or set(rows) != set(range(TOTAL_ROWS)):
        raise ValueError("readiness requires all 3,000 workload rows")
    ordered = [rows[index] for index in range(TOTAL_ROWS)]
    max_spans = max(
        radius["unique_span_count"]
        for row in ordered
        for radius in row["radii"].values()
    )
    max_pairs = max(
        radius["unique_pair_count"]
        for row in ordered
        for radius in row["radii"].values()
    )
    max_all_spans = max(
        radius["all_allowed_unique_spans"]
        for row in ordered
        for radius in row["radii"].values()
    )
    max_all_pairs = max(
        radius["all_allowed_unique_pairs"]
        for row in ordered
        for radius in row["radii"].values()
    )
    max_sidecar = max(
        row["resource_counts"]["maximum_sidecar_bytes"] for row in ordered
    )
    topology_seconds = sum(
        float(timings[index]["elapsed_seconds"]) for index in range(TOTAL_ROWS)
    )
    component = {
        key: sum(row["estimate"][key] for row in ordered)
        for key in (
            "feature_seconds",
            "sidecar_seconds",
            "core_seconds",
            "total_gpu_seconds_without_topology",
        )
    }
    model_load_seconds = max(
        float(row["model_load_seconds"]) for row in profiles
    )
    two_replay_seconds = (
        2
        * (
            topology_seconds
            + component["total_gpu_seconds_without_topology"]
        )
        + model_load_seconds
    )
    legacy_over = sum(
        max(radius["unique_pair_count"] for radius in row["radii"].values())
        > LEGACY_PAIR_DIAGNOSTIC
        for row in ordered
    )
    gates = {
        "all_3000_rows_completed": len(ordered) == TOTAL_ROWS,
        "all_identity_and_structure_exact": all(
            row["raw_text_sha256"] == plan[index]["raw_text_sha256"]
            and row["unit_count"] == plan[index]["unit_count"]
            for index, row in enumerate(ordered)
        ),
        "all_renderers_restore_original_units": True,
        "max_pruned_spans_le_2000000": max_spans <= SPAN_GATE,
        "max_pruned_pairs_le_50000000": max_pairs <= PAIR_GATE,
        "max_all_allowed_spans_le_2000000": max_all_spans <= SPAN_GATE,
        "max_all_allowed_pairs_le_50000000": max_all_pairs <= PAIR_GATE,
        "max_sidecar_le_50_gib": max_sidecar <= CACHE_GATE_BYTES,
        "free_disk_ge_5_gib": free_bytes >= MIN_FREE_GIB * 1024**3,
        "two_replay_estimate_le_72_gpu_hours": (
            two_replay_seconds / 3600 <= GPU_HOUR_GATE
        ),
    }
    rows_bytes = jsonl_bytes(ordered)
    return {
        "schema_version": 1,
        "stage": f"{STAGE}-P0",
        "status": "go" if all(gates.values()) else "stop",
        "terminal_reason": (
            "b1_readiness_go"
            if all(gates.values())
            else "b1_readiness_stop"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "runner_sha256": sha256_file(paths["runner"]),
        "core_source_sha256": CORE_SOURCE_SHA256,
        "core_executable_sha256": sha256_file(_WORKER.get(
            "core_path", paths["root"] / "build/wp2_v6/certified_resegmentation_wp2_v6_core"
        )),
        "workload_rows": {
            "rows": len(ordered),
            "bytes": len(rows_bytes),
            "sha256": sha256_bytes(rows_bytes),
            "contains_raw_text": False,
        },
        "counts": {
            "cohorts": dict(Counter(row["cohort"] for row in ordered)),
            "total_maximum_legal_input_tokens": sum(
                row["resource_counts"]["maximum_legal_input_tokens"]
                for row in ordered
            ),
            "total_radius_unique_spans": sum(
                radius["unique_span_count"]
                for row in ordered
                for radius in row["radii"].values()
            ),
            "total_radius_unique_pairs": sum(
                radius["unique_pair_count"]
                for row in ordered
                for radius in row["radii"].values()
            ),
            "max_document_pruned_spans": max_spans,
            "max_document_pruned_pairs": max_pairs,
            "max_document_all_allowed_spans": max_all_spans,
            "max_document_all_allowed_pairs": max_all_pairs,
            "max_document_sidecar_bytes": max_sidecar,
            "documents_over_legacy_5m_pair_diagnostic": legacy_over,
        },
        "runtime_upper_bound": {
            "replay_count": REPLAY_COUNT,
            "measured_topology_seconds_per_replay": topology_seconds,
            **component,
            "model_load_seconds": model_load_seconds,
            "conservative_two_replay_seconds": two_replay_seconds,
            "conservative_two_replay_gpu_hours": two_replay_seconds / 3600,
        },
        "parameters": {
            "radii": list(RADII),
            "native_token_limit": 512,
            "span_gate": SPAN_GATE,
            "factorized_pair_gate": PAIR_GATE,
            "legacy_pair_diagnostic": LEGACY_PAIR_DIAGNOSTIC,
            "cache_gate_bytes": CACHE_GATE_BYTES,
            "gpu_hour_gate": GPU_HOUR_GATE,
            "embedding_batch_size": BATCH_SIZE,
            "tokenizer_chunk_size": TOKENIZER_CHUNK_SIZE,
        },
        "inputs": {
            "b0_result_sha256": B0_RESULT_SHA256,
            "b0_audit_sha256": B0_AUDIT_SHA256,
            "b0_plan_sha256": B0_PLAN_SHA256,
            "b0_rows_sha256": B0_ROWS_SHA256,
            "b0_checkpoint_sha256": B0_CHECKPOINT_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "provider_profiles": profiles,
        },
        "resource_gate": {
            "minimum_free_gib": MIN_FREE_GIB,
            "observed_free_bytes": free_bytes,
        },
        "gates": gates,
        "execution_attestation": {
            "sealed_inputs_parsed": True,
            "detector_source_imports": 0,
            "detector_model_loads": 0,
            "region_computations": 0,
            "certificate_score_computations": 0,
            "threshold_computations": 0,
            "generation_calls": 0,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "workers": contract["readiness_workers"],
        },
        "completed_at": utc_now(),
        "contains_raw_text": False,
    }, rows_bytes


def run_readiness(args, paths, contract):
    if args.workers != contract["readiness_workers"]:
        raise ValueError("readiness worker count differs from contract")
    _b0_result, plan, _b0_rows = validate_b0(paths)
    rebuilt_plan, texts, _documents = materialize_documents(
        paths, args.model_dir, args.nltk_data
    )
    if rebuilt_plan != plan:
        raise ValueError("materialized plan differs from frozen B0 plan")
    if sha256_file(args.core) != (
        LINUX_CORE_SHA256 if platform.system() == "Linux"
        else contract["development_core_sha256"]
    ):
        raise ValueError("v6 core executable SHA mismatch")
    profiles = load_profiles(paths["profile"])
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["workload_timings"].parent.mkdir(parents=True, exist_ok=True)
    existing = collect_append_checkpoint(paths["workload_rows"], plan)
    timing_rows = {
        row["plan_index"]: row
        for row in (
            read_jsonl(paths["workload_timings"])
            if paths["workload_timings"].exists()
            else []
        )
    }
    if set(existing) != set(timing_rows):
        raise ValueError("workload/timing checkpoint prefix mismatch")
    tasks = [
        (index, plan[index], texts[index])
        for index in range(TOTAL_ROWS)
        if index not in existing
    ]
    heartbeat_started = time.monotonic()
    if tasks:
        scratch = Path(args.work_dir) / "wp2_v18_readiness"
        scratch.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=initialize_readiness_worker,
            initargs=(
                args.model_dir,
                args.core,
                scratch,
                args.tokenizer_chunk_size,
                profiles,
            ),
        ) as pool:
            for row, timing in pool.imap_unordered(workload_task, tasks):
                index = row["plan_index"]
                validate_workload_row(row, index, plan[index])
                append_jsonl(paths["workload_rows"], row)
                append_jsonl(paths["workload_timings"], timing)
                existing[index] = row
                timing_rows[index] = timing
                atomic_write_json(paths["heartbeat"], {
                    "schema_version": 1,
                    "stage": f"{STAGE}-P0",
                    "status": "running",
                    "completed": len(existing),
                    "total": TOTAL_ROWS,
                    "elapsed_seconds": (
                        time.monotonic() - heartbeat_started
                    ),
                    "updated_at": utc_now(),
                    "contains_raw_text": False,
                })
                print(json.dumps({
                    "stage": "readiness",
                    "completed": len(existing),
                    "total": TOTAL_ROWS,
                    "plan_index": index,
                }, sort_keys=True), flush=True)
        shutil.rmtree(scratch, ignore_errors=True)

    ordered = [existing[index] for index in range(TOTAL_ROWS)]
    ordered_timings = [timing_rows[index] for index in range(TOTAL_ROWS)]
    atomic_write_bytes(paths["workload_rows"], jsonl_bytes(ordered))
    atomic_write_bytes(
        paths["workload_timings"],
        jsonl_bytes(ordered_timings),
    )
    _WORKER["core_path"] = Path(args.core)
    readiness, rows_bytes = aggregate_readiness(
        paths,
        contract,
        plan,
        existing,
        timing_rows,
        profiles,
        shutil.disk_usage(paths["root"]).free,
    )
    if readiness["workload_rows"]["sha256"] != sha256_bytes(rows_bytes):
        raise AssertionError("workload row digest drift")
    atomic_write_json(paths["readiness"], readiness)
    atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": f"{STAGE}-P0",
        "status": readiness["status"],
        "completed": TOTAL_ROWS,
        "total": TOTAL_ROWS,
        "updated_at": utc_now(),
        "contains_raw_text": False,
    })
    print(json.dumps({
        "stage": "readiness",
        "status": readiness["status"],
        "gates": readiness["gates"],
        "gpu_hours": readiness["runtime_upper_bound"][
            "conservative_two_replay_gpu_hours"
        ],
    }, sort_keys=True), flush=True)


def validate_readiness(paths, contract, plan):
    readiness = read_json(paths["readiness"])
    rows = read_jsonl(paths["workload_rows"])
    if (
        readiness.get("schema_version") != 1
        or readiness.get("stage") != f"{STAGE}-P0"
        or readiness.get("status") != "go"
        or readiness.get("protocol_sha256") != PROTOCOL_SHA256
        or readiness.get("contract_sha256") != sha256_file(paths["contract"])
        or readiness.get("runner_sha256") != sha256_file(paths["runner"])
        or not all(readiness.get("gates", {}).values())
        or len(rows) != TOTAL_ROWS
        or readiness.get("workload_rows", {}).get("sha256")
        != sha256_file(paths["workload_rows"])
    ):
        raise ValueError("B1 readiness is not a frozen Go")
    for index, row in enumerate(rows):
        validate_workload_row(row, index, plan[index])
    return readiness, rows


def feature_digest(codes, allowed, regions):
    digest = hashlib.sha256()
    digest.update(np.asarray(codes, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(allowed, dtype="u1").tobytes(order="C"))
    digest.update(np.asarray(regions, dtype="u1").tobytes(order="C"))
    return digest.hexdigest()


def certificate_digest(value):
    selected = {
        key: value[key]
        for key in (
            "requested_radius",
            "effective_radius",
            "exact_min_hits",
            "exact_score",
            "cheap_min_hits",
            "cheap_score",
            "threat_set_size",
            "reachable_state_count",
            "unique_span_count",
            "unique_pair_count",
            "witness_boundaries",
            "witness_score",
            "feature_cache_sha256",
        )
    }
    return canonical_digest(selected)


def valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_witness(boundaries, atom_count, edge_count, original, radius):
    if (
        not isinstance(boundaries, list)
        or len(boundaries) != edge_count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in boundaries
        )
        or boundaries != sorted(set(boundaries))
        or any(value <= 0 or value >= atom_count for value in boundaries)
        or len(set(original) - set(boundaries)) > radius
    ):
        raise ValueError("invalid certificate witness")


def process_certificate_document(
    args,
    plan_index,
    planned,
    text,
    workload,
    b0_row,
    model,
    tokenizer,
    relation,
    scratch,
):
    from nltk.tokenize import sent_tokenize

    started = time.perf_counter()
    document = canonicalize_structured(text, sentence_splitter=sent_tokenize)
    if (
        document.raw_text_sha256 != planned["raw_text_sha256"]
        or document.structural_tape_sha256 != workload["tape_sha256"]
        or document.unit_count != planned["unit_count"]
        or document.atom_count != workload["atom_count"]
    ):
        raise ValueError(f"execution identity drift at row {plan_index}")
    (
        _effective,
        distinct,
        topologies,
        radii,
        resource_counts,
    ) = prepare_document_topologies(
        document,
        tokenizer,
        args.core,
        scratch,
        args.tokenizer_chunk_size,
    )
    if radii != workload["radii"]:
        raise ValueError(f"execution topology drift at row {plan_index}")
    maximum_radius = max(distinct)
    maximum = topologies[maximum_radius]
    memmap_path = scratch / "regions.u8"
    regions, _provider = v10.encode_region_memmap(
        document,
        maximum["codes"],
        maximum["allowed"],
        model,
        args.text_chunk_size,
        memmap_path,
        lambda _completed, _total: None,
    )
    cache_sha = feature_digest(
        maximum["codes"], maximum["allowed"], regions
    )
    original_endpoints = (
        0, *document.original_boundaries, document.atom_count
    )
    original_codes = np.asarray([
        original_endpoints[index] * (document.atom_count + 1)
        + original_endpoints[index + 1]
        for index in range(document.unit_count)
    ], dtype=np.uint64)
    original_positions = np.searchsorted(maximum["codes"], original_codes)
    if (
        np.any(original_positions >= len(maximum["codes"]))
        or np.any(maximum["codes"][original_positions] != original_codes)
    ):
        raise ValueError(f"original spans absent at row {plan_index}")
    original_regions = np.asarray(
        regions[original_positions], dtype=np.uint8
    )
    original_regions_sha = sha256_bytes(
        original_regions.tobytes(order="C")
    )
    if original_regions_sha != b0_row["regions_sha256"]:
        raise ValueError(f"original region/B0 mismatch at row {plan_index}")
    certificates_by_effective = {}
    core_seconds = 0.0
    sidecar_seconds = 0.0
    max_sidecar_bytes = 0
    for radius in distinct:
        topology = topologies[radius]
        positions = v10.assert_subset(
            topology["codes"], maximum["codes"]
        )
        selected_regions = regions[positions]
        sidecar = scratch / f"features_k{radius}.bin"
        sidecar_started = time.perf_counter()
        sidecar_bytes = v10.write_mode5_sidecar(
            sidecar,
            document.atom_count,
            topology["codes"],
            topology["allowed"],
            selected_regions,
            relation,
        )
        sidecar_seconds += time.perf_counter() - sidecar_started
        max_sidecar_bytes = max(max_sidecar_bytes, sidecar_bytes)
        core_started = time.perf_counter()
        observed, _stdout_sha = v10.run_factorized_core(
            args.core, document, radius, sidecar
        )
        core_seconds += time.perf_counter() - core_started
        sidecar.unlink(missing_ok=True)
        original_hits = v10.raw_sum_for_boundaries(
            document,
            document.original_boundaries,
            maximum["codes"],
            regions,
            relation,
        )
        witness_hits = v10.raw_sum_for_boundaries(
            document,
            observed["worst_boundaries"],
            maximum["codes"],
            regions,
            relation,
        )
        if (
            observed["original_score"] != original_hits
            or observed["certified_score"] != witness_hits
        ):
            raise ValueError(
                f"witness replay mismatch at row {plan_index}, k={radius}"
            )
        edge_count = document.unit_count - 1
        cheap_hits = max(0, original_hits - 5 * radius)
        value = {
            "requested_radius": radius,
            "effective_radius": radius,
            "exact_min_hits": int(witness_hits),
            "exact_score": v10.score_from_raw(witness_hits, edge_count),
            "cheap_min_hits": int(cheap_hits),
            "cheap_score": v10.score_from_raw(cheap_hits, edge_count),
            "threat_set_size": observed["threat_set_size"],
            "reachable_state_count": observed["reachable_state_count"],
            "unique_span_count": observed["unique_span_checks"],
            "unique_pair_count": observed["unique_pair_scores"],
            "witness_boundaries": list(observed["worst_boundaries"]),
            "witness_score": v10.score_from_raw(witness_hits, edge_count),
            "feature_cache_sha256": cache_sha,
        }
        value["certificate_sha256"] = certificate_digest(value)
        certificates_by_effective[radius] = value

    certificates = {}
    for requested in RADII:
        effective = min(requested, document.unit_count - 1)
        value = dict(certificates_by_effective[effective])
        value["requested_radius"] = requested
        value["certificate_sha256"] = certificate_digest(value)
        certificates[str(requested)] = value
    exact_hits = [
        certificates[str(radius)]["exact_min_hits"] for radius in RADII
    ]
    if any(
        current > previous
        for previous, current in zip(exact_hits, exact_hits[1:])
    ):
        raise ValueError(f"non-monotone certificate at row {plan_index}")
    for radius in RADII:
        value = certificates[str(radius)]
        if (
            value["cheap_min_hits"] > value["exact_min_hits"]
            or value["exact_min_hits"] > b0_row["hit_count"]
        ):
            raise ValueError(
                f"bound ordering failed at row {plan_index}, k={radius}"
            )
    if (
        certificates["0"]["exact_min_hits"] != b0_row["hit_count"]
        or certificates["0"]["exact_score"] != b0_row["direct_score"]
        or certificates["0"]["witness_boundaries"]
        != list(document.original_boundaries)
    ):
        raise ValueError(f"k=0/B0 mismatch at row {plan_index}")
    del regions
    memmap_path.unlink(missing_ok=True)
    deterministic = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "plan_index": plan_index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": document.raw_text_sha256,
        "tape_sha256": document.structural_tape_sha256,
        "original_units_sha256": workload["original_units_sha256"],
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
        "edge_count": document.unit_count - 1,
        "original_hits": b0_row["hit_count"],
        "original_score": b0_row["direct_score"],
        "original_regions_sha256": original_regions_sha,
        "certificates": certificates,
        "contains_raw_text": False,
    }
    metrics = {
        "plan_index": plan_index,
        "elapsed_seconds": time.perf_counter() - started,
        "core_seconds": core_seconds,
        "sidecar_seconds": sidecar_seconds,
        "max_sidecar_bytes": max_sidecar_bytes,
        "maximum_legal_input_tokens": resource_counts[
            "maximum_legal_input_tokens"
        ],
    }
    return deterministic, metrics


def validate_certificate_row(row, index, planned, workload, b0_row):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("plan_index") != index
        or row.get("cohort") != planned["cohort"]
        or row.get("cohort_index") != planned["cohort_index"]
        or row.get("raw_text_sha256") != planned["raw_text_sha256"]
        or row.get("tape_sha256") != workload["tape_sha256"]
        or row.get("unit_count") != planned["unit_count"]
        or row.get("original_hits") != b0_row["hit_count"]
        or row.get("original_score") != b0_row["direct_score"]
        or row.get("original_regions_sha256") != b0_row["regions_sha256"]
        or row.get("contains_raw_text") is not False
        or set(row.get("certificates", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError(f"invalid certificate row at {index}")
    previous = math.inf
    for radius in RADII:
        value = row["certificates"][str(radius)]
        validate_witness(
            value.get("witness_boundaries"),
            row["atom_count"],
            row["edge_count"],
            workload["original_boundaries"],
            value.get("effective_radius"),
        )
        if (
            value.get("requested_radius") != radius
            or value.get("effective_radius")
            != min(radius, row["edge_count"])
            or value.get("unique_span_count")
            != workload["radii"][str(radius)]["unique_span_count"]
            or value.get("unique_pair_count")
            != workload["radii"][str(radius)]["unique_pair_count"]
            or len(value.get("witness_boundaries", [])) != row["edge_count"]
            or value.get("cheap_min_hits")
            != max(
                0,
                row["original_hits"] - 5 * value["effective_radius"],
            )
            or not valid_sha256(value.get("feature_cache_sha256"))
            or value.get("certificate_sha256")
            != certificate_digest(value)
        ):
            raise ValueError(
                f"invalid certificate payload at {index}, k={radius}"
            )
        exact_hits = value["exact_min_hits"]
        exact_score = value["exact_score"]
        if (
            exact_hits > previous
            or value["cheap_min_hits"] > exact_hits
            or exact_hits > row["original_hits"]
            or exact_score != v10.score_from_raw(
                exact_hits, row["edge_count"]
            )
            or value["cheap_score"] != v10.score_from_raw(
                value["cheap_min_hits"], row["edge_count"]
            )
            or value["witness_score"] != exact_score
        ):
            raise ValueError(
                f"certificate invariant failed at {index}, k={radius}"
            )
        previous = exact_hits
    if (
        row["certificates"]["0"]["exact_min_hits"] != row["original_hits"]
        or row["certificates"]["0"]["exact_score"] != row["original_score"]
        or row["certificates"]["0"]["witness_boundaries"]
        != workload["original_boundaries"]
    ):
        raise ValueError(f"k=0/B0 row mismatch at {index}")
    return row


def clopper_pearson_lower(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("require 0 <= successes <= trials")
    if successes == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(
        trials - successes, trials, confidence
    )


def classification(scores, threshold, confidence, lower=False):
    successes = sum(float(score) > threshold for score in scores)
    output = {
        "threshold": threshold,
        "strict_exceedances": successes,
        "trials": len(scores),
        "rate": successes / len(scores),
        "one_sided_cp_confidence": confidence,
    }
    if lower:
        output["one_sided_cp_lower"] = clopper_pearson_lower(
            successes, len(scores), confidence
        )
    else:
        output["one_sided_cp_upper"] = clopper_pearson_upper(
            successes, len(scores), confidence
        )
    return output


def paired_counts(exact_scores, cheap_scores, threshold):
    output = Counter()
    for exact, cheap in zip(exact_scores, cheap_scores):
        exact_pass = exact > threshold
        cheap_pass = cheap > threshold
        if exact_pass and cheap_pass:
            output["both_certify"] += 1
        elif exact_pass:
            output["exact_only"] += 1
        elif cheap_pass:
            output["cheap_only"] += 1
        else:
            output["neither"] += 1
    return {
        key: output[key]
        for key in (
            "both_certify",
            "exact_only",
            "cheap_only",
            "neither",
        )
    }


def summarize_results(rows, runtime):
    if len(rows) != TOTAL_ROWS:
        raise ValueError("summary requires all 3,000 rows")
    groups = {
        cohort: [
            row for row in rows if row["cohort"] == cohort
        ]
        for cohort in (
            "calibration_null",
            "independent_test_null",
            "positive",
        )
    }
    if any(len(group) != COHORT_SIZE for group in groups.values()):
        raise ValueError("summary cohort size mismatch")
    thresholds = {}
    calibrations = {}
    for radius in RADII:
        scores = [
            row["certificates"][str(radius)]["exact_score"]
            for row in groups["calibration_null"]
        ]
        calibrated = split_conformal_calibrate(scores, TARGET_FPR)
        thresholds[radius] = calibrated.threshold
        calibrations[str(radius)] = calibrated.to_dict()

    analyses = {}
    paired = {}
    for radius in RADII:
        key = str(radius)
        exact_test = [
            row["certificates"][key]["exact_score"]
            for row in groups["independent_test_null"]
        ]
        exact_positive = [
            row["certificates"][key]["exact_score"]
            for row in groups["positive"]
        ]
        cheap_test = [
            row["certificates"][key]["cheap_score"]
            for row in groups["independent_test_null"]
        ]
        cheap_positive = [
            row["certificates"][key]["cheap_score"]
            for row in groups["positive"]
        ]
        analyses[key] = {
            "exact_recalibrated": {
                "test": classification(
                    exact_test,
                    thresholds[radius],
                    0.975 if radius == 1 else 0.95,
                ),
                "positive": classification(
                    exact_positive, thresholds[radius], 0.95, lower=True
                ),
            },
            "exact_fixed_b0": {
                "test": classification(
                    exact_test, B0_THRESHOLD, 0.95
                ),
                "positive": classification(
                    exact_positive, B0_THRESHOLD, 0.95, lower=True
                ),
            },
            "cheap_fixed_b0": {
                "test": classification(
                    cheap_test, B0_THRESHOLD, 0.95
                ),
                "positive": classification(
                    cheap_positive, B0_THRESHOLD, 0.95, lower=True
                ),
            },
            "cheap_at_exact_threshold": {
                "test": classification(
                    cheap_test, thresholds[radius], 0.95
                ),
                "positive": classification(
                    cheap_positive, thresholds[radius], 0.95, lower=True
                ),
            },
        }
        paired[key] = {
            "fixed_b0_test": paired_counts(
                exact_test, cheap_test, B0_THRESHOLD
            ),
            "fixed_b0_positive": paired_counts(
                exact_positive, cheap_positive, B0_THRESHOLD
            ),
            "exact_threshold_test": paired_counts(
                exact_test, cheap_test, thresholds[radius]
            ),
            "exact_threshold_positive": paired_counts(
                exact_positive, cheap_positive, thresholds[radius]
            ),
        }

    main = analyses["1"]["exact_recalibrated"]
    gates = {
        "k1_test_cp975_upper_le_0_08": (
            main["test"]["one_sided_cp_upper"] <= 0.08
        ),
        "k1_positive_rate_ge_0_20": (
            main["positive"]["rate"] >= 0.20
        ),
        "k1_positive_cp95_lower_gt_0_10": (
            main["positive"]["one_sided_cp_lower"] > 0.10
        ),
        "clean_k0_positive_rate_ge_0_80": (
            analyses["0"]["exact_recalibrated"]["positive"]["rate"]
            >= 0.80
        ),
        "k0_threshold_equals_b0": thresholds[0] == B0_THRESHOLD,
        "two_full_replays_exact": runtime["replay_rows_sha256"]["0"]
        == runtime["replay_rows_sha256"]["1"],
        "all_cheap_only_counts_zero": all(
            table["cheap_only"] == 0
            for radius in paired.values()
            for table in radius.values()
        ),
        "peak_vram_below_0_80": runtime["peak_vram_fraction"] < 0.80,
        "all_3000_rows_completed": len(rows) == TOTAL_ROWS,
    }
    return {
        "schema_version": 1,
        "stage": STAGE,
        "status": "go" if all(gates.values()) else "stop",
        "terminal_reason": (
            "b1_exact_certificate_go"
            if all(gates.values())
            else "b1_exact_certificate_stop"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "b0_threshold": B0_THRESHOLD,
        "calibration": calibrations,
        "analyses": analyses,
        "paired": paired,
        "gates": gates,
        "runtime": runtime,
        "selection_disclosure": (
            "B1 was run only after B0 passed on the same test and positive "
            "cohorts; interval coverage is not claimed conditional on B0."
        ),
        "scope": "semstamp_only_simmark_remains_stopped",
        "completed_at": utc_now(),
        "contains_raw_text": False,
    }


def collect_certificate_checkpoint(
    directory, plan, workload, b0_rows
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdigit():
            raise ValueError(f"unexpected checkpoint file: {path.name}")
        index = int(path.stem)
        if not 0 <= index < TOTAL_ROWS or index in output:
            raise ValueError(f"invalid checkpoint index: {index}")
        output[index] = validate_certificate_row(
            read_json(path),
            index,
            plan[index],
            workload[index],
            b0_rows[index],
        )
    return output


def collect_runtime_metrics(output_dir):
    rows = []
    for replay in range(REPLAY_COUNT):
        directory = Path(output_dir) / "sealed" / f"metrics{replay}"
        paths = sorted(directory.glob("*.json"))
        if len(paths) != TOTAL_ROWS:
            raise ValueError(
                f"runtime metric count mismatch for replay {replay}"
            )
        for index, path in enumerate(paths):
            if path.name != f"{index:04d}.json":
                raise ValueError("runtime metric filename mismatch")
            row = read_json(path)
            if (
                row.get("replay") != replay
                or row.get("plan_index") != index
                or not math.isfinite(float(row.get("elapsed_seconds", -1)))
                or row["elapsed_seconds"] < 0
            ):
                raise ValueError("invalid runtime metric row")
            rows.append(row)
    return rows


def model_batch_invariance(model, documents):
    selections = []
    for document in documents[:3]:
        units = list(document.preprocess_partition("semstamp"))
        selections.extend(units[:2])
    if len(selections) < 4:
        raise ValueError("insufficient units for batch invariance check")
    import torch
    from run_certified_resegmentation_wp2_v6_provider import semstamp_features

    with torch.inference_mode():
        together = v10.validate_regions(
            semstamp_features(model, selections), len(selections)
        )
        separate = np.concatenate([
            v10.validate_regions(semstamp_features(model, [text]), 1)
            for text in selections
        ])
        reversed_values = v10.validate_regions(
            semstamp_features(model, list(reversed(selections))),
            len(selections),
        )[::-1]
    if (
        not np.array_equal(together, separate)
        or not np.array_equal(together, reversed_values)
    ):
        raise ValueError("SemStamp region changed with batch composition")
    return {
        "tested_units": len(selections),
        "together_sha256": sha256_bytes(together.tobytes(order="C")),
        "separate_sha256": sha256_bytes(separate.tobytes(order="C")),
        "reversed_sha256": sha256_bytes(
            reversed_values.tobytes(order="C")
        ),
        "exact": True,
    }


def write_heartbeat(path, replay, completed, started, **fields):
    atomic_write_json(path, {
        "schema_version": 1,
        "stage": STAGE,
        "status": "running",
        "replay": replay,
        "completed_in_replay": completed,
        "completed_total": replay * TOTAL_ROWS + completed,
        "target_total": REPLAY_COUNT * TOTAL_ROWS,
        "elapsed_seconds": time.monotonic() - started,
        "updated_at": utc_now(),
        "contains_raw_text": False,
        **fields,
    })


def run_certificates(args, paths, contract):
    import torch
    from run_certified_resegmentation_wp2_v6_provider import load_semstamp

    if platform.system() != "Linux":
        raise ValueError("formal B1 execution requires Linux")
    if sha256_file(args.core) != LINUX_CORE_SHA256:
        raise ValueError("formal Linux core SHA mismatch")
    if args.batch_size != BATCH_SIZE:
        raise ValueError("embedding batch size differs from protocol")
    if args.text_chunk_size != TEXT_CHUNK_SIZE:
        raise ValueError("text chunk size differs from contract")
    if shutil.disk_usage(paths["root"]).free < MIN_FREE_GIB * 1024**3:
        raise OSError("free disk is below the 5 GiB gate")
    _b0_result, plan, b0_rows = validate_b0(paths)
    readiness, workload = validate_readiness(paths, contract, plan)
    rebuilt_plan, texts, documents = materialize_documents(
        paths, args.model_dir, args.nltk_data
    )
    if rebuilt_plan != plan:
        raise ValueError("execution plan differs from frozen B0 plan")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.reset_peak_memory_stats()
    started_at = utc_now()
    monotonic_started = time.monotonic()
    write_heartbeat(paths["heartbeat"], 0, 0, monotonic_started, phase="load")
    model_load_started = time.perf_counter()
    model, relation = load_semstamp(
        paths["root"], args.model_dir, args.batch_size
    )
    relation = v10.validate_relation(relation)
    tokenizer = load_tokenizer(args.model_dir)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    batch_invariance = model_batch_invariance(model, documents)
    replay_sha = {}
    resumed = {}
    for replay in range(REPLAY_COUNT):
        checkpoint_dir = paths["output_dir"] / "sealed" / f"replay{replay}"
        metric_dir = paths["output_dir"] / "sealed" / f"metrics{replay}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        existing = collect_certificate_checkpoint(
            checkpoint_dir, plan, workload, b0_rows
        )
        resumed[str(replay)] = len(existing)
        write_heartbeat(
            paths["heartbeat"],
            replay,
            len(existing),
            monotonic_started,
            phase="certificates",
        )
        for index in range(TOTAL_ROWS):
            if index in existing:
                continue
            scratch = (
                Path(args.work_dir)
                / f"wp2_v18_replay{replay}_row{index}"
            )
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True)
            write_heartbeat(
                paths["heartbeat"],
                replay,
                len(existing),
                monotonic_started,
                phase="certificate_document",
                plan_index=index,
            )
            try:
                row, metrics = process_certificate_document(
                    args,
                    index,
                    plan[index],
                    texts[index],
                    workload[index],
                    b0_rows[index],
                    model,
                    tokenizer,
                    relation,
                    scratch,
                )
                path = checkpoint_dir / f"{index:04d}.json"
                atomic_write_json(path, row)
                existing[index] = validate_certificate_row(
                    read_json(path),
                    index,
                    plan[index],
                    workload[index],
                    b0_rows[index],
                )
                metrics.update({
                    "replay": replay,
                    "peak_cuda_allocated_bytes": (
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_cuda_reserved_bytes": (
                        torch.cuda.max_memory_reserved()
                    ),
                    "process_peak_rss_bytes": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * 1024
                    ),
                })
                atomic_write_json(
                    metric_dir / f"{index:04d}.json", metrics
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            write_heartbeat(
                paths["heartbeat"],
                replay,
                len(existing),
                monotonic_started,
                phase="certificates",
                plan_index=index,
            )
            print(json.dumps({
                "stage": "certificates",
                "replay": replay,
                "completed": len(existing),
                "total": TOTAL_ROWS,
                "plan_index": index,
            }, sort_keys=True), flush=True)
        ordered = [existing[index] for index in range(TOTAL_ROWS)]
        target = paths[f"rows{replay}"]
        atomic_write_bytes(target, jsonl_bytes(ordered))
        replay_sha[str(replay)] = sha256_file(target)
        if replay and replay_sha[str(replay)] != replay_sha["0"]:
            raise ValueError("full certificate replay mismatch")

    metric_rows = collect_runtime_metrics(paths["output_dir"])
    total_cuda = torch.cuda.get_device_properties(0).total_memory
    peak_cuda = max(
        [torch.cuda.max_memory_reserved()]
        + [row["peak_cuda_reserved_bytes"] for row in metric_rows]
    )
    peak_allocated = max(
        [torch.cuda.max_memory_allocated()]
        + [row["peak_cuda_allocated_bytes"] for row in metric_rows]
    )
    runtime = {
        "started_at": started_at,
        "finished_at": utc_now(),
        "current_process_elapsed_seconds": (
            time.monotonic() - monotonic_started
        ),
        "summed_document_seconds": sum(
            row["elapsed_seconds"] for row in metric_rows
        ),
        "model_load_seconds": model_load_seconds,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_total_bytes": total_cuda,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_cuda,
        "peak_vram_fraction": peak_cuda / total_cuda,
        "batch_size": args.batch_size,
        "text_chunk_size": args.text_chunk_size,
        "batch_invariance": batch_invariance,
        "resumed_documents": resumed,
        "replay_rows_sha256": replay_sha,
        "readiness_sha256": sha256_file(paths["readiness"]),
        "workload_rows_sha256": sha256_file(paths["workload_rows"]),
        "runtime_metric_rows": len(metric_rows),
        "platform": platform.platform(),
    }
    atomic_write_json(paths["runtime"], runtime)
    rows = read_jsonl(paths["rows0"])
    result = summarize_results(rows, runtime)
    result["contract_sha256"] = sha256_file(paths["contract"])
    result["runner_sha256"] = sha256_file(paths["runner"])
    result["readiness"] = {
        "sha256": sha256_file(paths["readiness"]),
        "status": readiness["status"],
    }
    atomic_write_json(paths["result"], result)
    atomic_write_json(paths["heartbeat"], {
        "schema_version": 1,
        "stage": STAGE,
        "status": result["status"],
        "completed_total": REPLAY_COUNT * TOTAL_ROWS,
        "target_total": REPLAY_COUNT * TOTAL_ROWS,
        "elapsed_seconds": runtime["current_process_elapsed_seconds"],
        "updated_at": utc_now(),
        "contains_raw_text": False,
    })
    print(json.dumps({
        "stage": "complete",
        "status": result["status"],
        "gates": result["gates"],
    }, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("readiness", "run"), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--tokenizer-chunk-size", type=int, default=TOKENIZER_CHUNK_SIZE
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--text-chunk-size", type=int, default=TEXT_CHUNK_SIZE
    )
    return parser.parse_args()


def run(args):
    paths = paths_for(args.root, args.output_dir)
    contract = validate_contract(paths)
    if args.tokenizer_chunk_size != TOKENIZER_CHUNK_SIZE:
        raise ValueError("tokenizer chunk size differs from protocol")
    if args.stage == "readiness":
        run_readiness(args, paths, contract)
    else:
        run_certificates(args, paths, contract)


def main():
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        try:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_dir / "LAST_FAILURE.json", {
                "schema_version": 1,
                "stage": STAGE,
                "phase": args.stage,
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "contains_raw_text": False,
            })
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
