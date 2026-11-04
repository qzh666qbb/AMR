"""Run the resumable WP2-v9 exploratory SimMark certificate experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from audit_certified_resegmentation_wp2_v2_design import (
    NATIVE_TOKEN_LIMIT,
    SYSTEMS as DESIGN_SYSTEMS,
    frozen_identities,
    load_tokenizer,
    official_tokenizer_text,
)
from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)
from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
)
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    canonicalize_structured,
)
from certified_resegmentation_wp2_v6_features import (
    MAGIC,
    SIMMARK_DIMENSION,
    SIMMARK_MODE,
    serialize_core_problem,
)
from run_certified_resegmentation_wp2_detector_equivalence import (
    package_versions,
    verify_inputs,
)
from run_certified_resegmentation_wp2_v6_provider import (
    load_simmark,
    simmark_features,
)


PROTOCOL_SHA256 = (
    "221a57b207507571d434ac242530a70e5cfdd51ed047adb708f711e335bf81af"
)
IDENTITY_MANIFEST_SHA256 = (
    "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
)
WORKLOAD_MANIFEST_SHA256 = (
    "04f17eb7c73f6ca9a8e379615e1062eaee37a63a3794ae39ff81b835929de6eb"
)
PARENT_PROTOCOL_SHA256 = (
    "9e3017c1d6645452d232d191ec95e3b127c1ba7d31d1082a7793d9347b08d2ec"
)
CORE_SOURCE_SHA256 = (
    "d77d23c0e44a343c154e93d9897ee3e40c1a05194d9200108ff7e619f52e1360"
)
FIT_RANGE = range(0, 200)
CALIBRATION_RANGE = range(200, 600)
TEST_RANGE = range(600, 1000)
POSITIVE_RANGE = range(1000, 2000)
CERTIFICATE_RANGE = range(200, 2000)
TARGET_FPR = 0.05
WITNESS_TOLERANCE = 1e-8
INSTRUCTION = "Represent the sentence for cosine similarity:"
SIDECAR_RECORD_DTYPE = np.dtype([
    ("start", "<u4"),
    ("end", "<u4"),
    ("allowed", "u1"),
    ("embedding", "<f4", (SIMMARK_DIMENSION,)),
])
EXPECTED_COHORTS = (
    ("detector_fit", "null", FIT_RANGE),
    ("certificate_calibration", "null", CALIBRATION_RANGE),
    ("independent_test", "null", TEST_RANGE),
    ("positive", "positive", POSITIVE_RANGE),
)


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


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def atomic_write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path, value):
    atomic_write_bytes(path, canonical_json_bytes(value))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def identity_fields(row):
    return {
        key: row[key]
        for key in (
            "cohort",
            "role",
            "row_index",
            "raw_text_sha256",
            "structural_tape_sha256",
            "atom_count",
            "unit_count",
        )
    }


def validate_document(document, source, manifest_index):
    observed = {
        "raw_text_sha256": document.raw_text_sha256,
        "structural_tape_sha256": document.structural_tape_sha256,
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
    }
    expected = {
        key: source[key]
        for key in observed
    }
    if observed != expected:
        raise ValueError(
            f"structural identity mismatch at manifest index {manifest_index}"
        )


def validate_frozen_inputs(args):
    paths_and_hashes = (
        (args.protocol, PROTOCOL_SHA256, "WP2-v9 protocol"),
        (
            args.identity_manifest,
            IDENTITY_MANIFEST_SHA256,
            "identity manifest",
        ),
        (
            args.workload_manifest,
            WORKLOAD_MANIFEST_SHA256,
            "workload manifest",
        ),
        (
            args.parent_protocol,
            PARENT_PROTOCOL_SHA256,
            "parent protocol",
        ),
        (
            args.core_source,
            CORE_SOURCE_SHA256,
            "v6 core source",
        ),
    )
    for path, expected, label in paths_and_hashes:
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA256 mismatch")
    if not args.core.is_file() or not os.access(args.core, os.X_OK):
        raise ValueError("compiled v6 core is absent or not executable")

    identities, _split, revision = frozen_identities(args.root, "simmark")
    identity_rows = load_jsonl(args.identity_manifest)
    workload_rows = load_jsonl(args.workload_manifest)
    if not (
        len(identities) == len(identity_rows) == len(workload_rows) == 2000
    ):
        raise ValueError("frozen SimMark manifests must contain 2,000 rows")

    expected_by_index = {}
    for cohort, role, indices in EXPECTED_COHORTS:
        for index in indices:
            expected_by_index[index] = (cohort, role)

    for index, (item, row, workload) in enumerate(
        zip(identities, identity_rows, workload_rows)
    ):
        record = item["record"]
        expected_cohort, expected_role = expected_by_index[index]
        expected_identity = (
            expected_cohort,
            expected_role,
            record.row_index,
            record.raw_text_sha256,
        )
        observed_identity = (
            row.get("cohort"),
            row.get("role"),
            row.get("row_index"),
            row.get("raw_text_sha256"),
        )
        workload_identity = (
            workload.get("cohort"),
            workload.get("role"),
            workload.get("row_index"),
            workload.get("raw_text_sha256"),
        )
        if (
            observed_identity != expected_identity
            or workload_identity != expected_identity
            or workload.get("manifest_index") != index
            or workload.get("system") != "simmark"
        ):
            raise ValueError(f"frozen identity mismatch at row {index}")
        for key in ("structural_tape_sha256", "atom_count", "unit_count"):
            if row.get(key) != workload.get(key):
                raise ValueError(
                    f"manifest structural mismatch at row {index}: {key}"
                )
        if not row["candidate_eligible"][
            "STRUCTURAL_SEPARATOR_NATIVE_512"
        ]:
            raise ValueError(f"frozen row {index} is not eligible")
        if not workload.get("eligible"):
            raise ValueError(f"workload row {index} is not eligible")
        if index in CERTIFICATE_RANGE:
            topology = workload["topology"]
            if (
                topology["legal_query_spans"]
                != topology["unique_query_spans"]
                or topology["actual_unique_pairs"]
                != topology["baseline_unique_pairs"]
            ):
                raise ValueError(
                    f"certificate row {index} has a pruned frozen topology"
                )
        materialize_document(item, row, index)
    return identities, identity_rows, workload_rows, revision


def materialize_document(item, source, manifest_index):
    from nltk.tokenize import sent_tokenize

    raw_value = item["raw_value"]
    document = canonicalize_structured(
        raw_value,
        sentence_splitter=sent_tokenize if isinstance(raw_value, str) else None,
    )
    validate_document(document, source, manifest_index)
    return document


def validate_embeddings(features, expected_rows):
    array = np.asarray(features, dtype=np.float32)
    if array.shape != (expected_rows, SIMMARK_DIMENSION):
        raise ValueError(
            f"invalid embedding shape {array.shape}; "
            f"expected {(expected_rows, SIMMARK_DIMENSION)}"
        )
    converted = array.astype(np.float64, copy=False)
    norms = np.linalg.norm(converted, axis=1)
    if (
        not np.all(np.isfinite(converted))
        or not np.all(np.isfinite(norms))
        or np.any(norms <= 0)
    ):
        raise ValueError("SimMark returned non-finite or zero-norm embeddings")
    return array


def adjacent_cosines(features):
    values = []
    for index in range(len(features) - 1):
        values.append(float(cosine_similarity(
            features[index].reshape(1, -1),
            features[index + 1].reshape(1, -1),
        )[0, 0]))
    return values


def fit_gamma(distances):
    values = np.asarray(tuple(distances), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("gamma fit distances must be finite and nonempty")
    counts, bins = np.histogram(values, bins=1000)
    centers = (bins[:-1] + bins[1:]) / 2
    selected = (centers >= 0.68) & (centers <= 0.76)
    gamma = float(np.sum(counts[selected]) / len(values))
    if not 0 <= gamma <= 1:
        raise ValueError("fitted SimMark gamma is outside [0, 1]")
    return {
        "gamma": gamma,
        "pair_count": int(values.size),
        "distance_min": float(values.min()),
        "distance_max": float(values.max()),
        "distance_bytes_sha256": sha256_bytes(
            values.astype("<f8", copy=False).tobytes()
        ),
        "histogram_counts_sha256": sha256_bytes(
            counts.astype("<u8", copy=False).tobytes()
        ),
        "histogram_bin_edges_sha256": sha256_bytes(
            bins.astype("<f8", copy=False).tobytes()
        ),
    }


def score_from_raw(raw_sum, edge_count, gamma):
    if edge_count < 1 or not 0 <= gamma <= 1:
        raise ValueError("invalid SimMark score parameters")
    return float(
        (float(raw_sum) - gamma * edge_count)
        / math.sqrt(edge_count * gamma * (1 - gamma) + 1e-12)
    )


def soft_count(left, right):
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    if not denominator > 0:
        raise ValueError("cannot score a zero-norm embedding")
    cosine = float(np.dot(left64, right64) / denominator)
    distance = (
        0.0
        if 0.68 <= cosine <= 0.76
        else min(abs(cosine - 0.68), abs(cosine - 0.76))
    )
    return math.exp(-250.0 * distance)


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


def read_mode5_codes(path, expected_n):
    record_dtype = np.dtype([
        ("start", "<u4"),
        ("end", "<u4"),
        ("allowed", "u1"),
        ("region", "u1"),
    ])
    with Path(path).open("rb") as handle:
        if handle.read(8) != MAGIC:
            raise ValueError("topology dump magic mismatch")
        mode, token_count, count, dimension = struct.unpack(
            "<IIQI", handle.read(20)
        )
        if mode != 5 or token_count != expected_n or dimension != 1:
            raise ValueError("topology dump identity mismatch")
        relation = handle.read(64)
        if len(relation) != 64:
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


def dump_topology(core, document, radius, path):
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
    codes = read_mode5_codes(path, document.atom_count)
    if result["unique_span_checks"] != len(codes):
        raise ValueError("topology dump/core span count mismatch")
    return result, codes


def assert_subset(subset, superset):
    positions = np.searchsorted(superset, subset)
    if (
        np.any(positions >= len(superset))
        or np.any(superset[positions] != subset)
    ):
        raise ValueError("radius topology is not nested")
    return positions


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
    return [len(row) for row in rows]


def encode_span_memmap(
    document,
    codes,
    model,
    tokenizer,
    batch_size,
    text_chunk_size,
    path,
    progress,
):
    count = len(codes)
    features = np.memmap(
        path,
        dtype="<f4",
        mode="w+",
        shape=(count, SIMMARK_DIMENSION),
    )
    base = document.atom_count + 1
    started = time.perf_counter()
    max_native_tokens = 0
    for offset in range(0, count, text_chunk_size):
        selected = codes[offset:offset + text_chunk_size]
        spans = [
            (int(code // base), int(code % base))
            for code in selected
        ]
        texts = [
            document.preprocess_unit("simmark", start, end)
            for start, end in spans
        ]
        tokenizer_texts = [
            official_tokenizer_text(
                "simmark",
                text,
                int(start > 0),
            )
            for text, (start, _end) in zip(texts, spans)
        ]
        token_counts = batched_token_counts(tokenizer, tokenizer_texts)
        max_native_tokens = max(max_native_tokens, max(token_counts))
        if any(value > NATIVE_TOKEN_LIMIT for value in token_counts):
            raise ValueError("certificate topology contains an illegal span")
        with torch.inference_mode():
            batch_features = simmark_features(model, texts, batch_size)
        batch_features = validate_embeddings(batch_features, len(texts))
        features[offset:offset + len(texts)] = batch_features
        progress(offset + len(texts), count)
    features.flush()
    torch.cuda.synchronize()
    return features, {
        "span_count": count,
        "max_native_tokens": max_native_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": path.stat().st_size,
    }


def write_mode6_sidecar(
    path,
    token_count,
    codes,
    positions,
    all_features,
    chunk_size=2048,
):
    with Path(path).open("wb", buffering=8 * 1024 * 1024) as handle:
        handle.write(MAGIC)
        handle.write(struct.pack(
            "<IIQI",
            SIMMARK_MODE,
            token_count,
            len(codes),
            SIMMARK_DIMENSION,
        ))
        base = token_count + 1
        for offset in range(0, len(codes), chunk_size):
            selected_codes = codes[offset:offset + chunk_size]
            selected_positions = positions[offset:offset + chunk_size]
            records = np.empty(len(selected_codes), dtype=SIDECAR_RECORD_DTYPE)
            records["start"] = selected_codes // base
            records["end"] = selected_codes % base
            records["allowed"] = 1
            records["embedding"] = all_features[selected_positions]
            handle.write(records.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    return Path(path).stat().st_size


def run_factorized_core(core, document, radius, sidecar):
    problem = SparsePairwiseProblem(
        token_count=document.atom_count,
        original_boundaries=document.original_boundaries,
        max_boundary_moves=radius,
        span_allowed=lambda _start, _end: True,
        pair_scorer=lambda _left, _middle, _right: 0.0,
        normalization=1.0,
        bias=0.0,
        threshold=0.0,
    )
    completed = subprocess.run(
        [str(core)],
        input=serialize_core_problem(problem, SIMMARK_MODE, sidecar),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "factorized core failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return parse_core_output(completed.stdout, 1)[0], sha256_bytes(
        completed.stdout
    )


def raw_sum_for_boundaries(document, boundaries, codes, features):
    endpoints = (0, *boundaries, document.atom_count)
    span_codes = np.asarray([
        endpoints[index] * (document.atom_count + 1) + endpoints[index + 1]
        for index in range(len(endpoints) - 1)
    ], dtype=np.uint64)
    positions = np.searchsorted(codes, span_codes)
    if (
        np.any(positions >= len(codes))
        or np.any(codes[positions] != span_codes)
    ):
        raise ValueError("witness span is absent from the feature table")
    selected = features[positions]
    return float(sum(
        soft_count(selected[index], selected[index + 1])
        for index in range(len(selected) - 1)
    ))


def process_fit_document(
    index,
    item,
    source,
    model,
    batch_size,
):
    started = time.perf_counter()
    document = materialize_document(item, source, index)
    units = list(document.preprocess_partition("simmark"))
    with torch.inference_mode():
        features = simmark_features(model, units, batch_size)
    features = validate_embeddings(features, len(units))
    distances = adjacent_cosines(features)
    if len(distances) != len(document.original_boundaries):
        raise AssertionError("fit pair count does not match P0 boundaries")
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "stage": "gamma_fit",
        "manifest_index": index,
        **identity_fields(source),
        "pair_count": len(distances),
        "distances": distances,
        "contains_raw_text": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_fit_checkpoint(row, index, source):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("stage") != "gamma_fit"
        or row.get("manifest_index") != index
    ):
        raise ValueError(f"invalid fit checkpoint header at row {index}")
    for key, value in identity_fields(source).items():
        if row.get(key) != value:
            raise ValueError(f"fit checkpoint identity mismatch: {index}")
    distances = row.get("distances")
    if (
        not isinstance(distances, list)
        or len(distances) != source["unit_count"] - 1
        or row.get("pair_count") != len(distances)
        or any(not math.isfinite(float(value)) for value in distances)
    ):
        raise ValueError(f"invalid fit distances at row {index}")
    return row


def checkpoint_manifest_sha(paths):
    lines = [
        f"{sha256_file(path)}  {path.name}\n"
        for path in sorted(paths)
    ]
    return sha256_bytes("".join(lines).encode("ascii"))


def gamma_artifact_from_rows(rows, checkpoint_paths):
    distances = [
        distance
        for row in rows
        for distance in row["distances"]
    ]
    output = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "stage": "gamma_fit_complete",
        "fit_document_count": len(rows),
        "fit_checkpoint_manifest_sha256": checkpoint_manifest_sha(
            checkpoint_paths
        ),
        "estimator": {
            "histogram_bins": 1000,
            "bin_center_interval": [0.68, 0.76],
        },
        **fit_gamma(distances),
        "contains_raw_text": False,
    }
    return output


def validate_gamma_artifact(observed, expected):
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("saved gamma artifact does not match fit checkpoints")
    return observed


def process_certificate_document(
    args,
    index,
    item,
    source,
    workload,
    model,
    tokenizer,
    gamma,
    gamma_sha256,
    scratch,
    heartbeat,
):
    started = time.perf_counter()
    document = materialize_document(item, source, index)
    boundary_count = len(document.original_boundaries)
    effective = {radius: min(radius, boundary_count) for radius in range(4)}
    distinct_radii = sorted(set(effective.values()))
    topology_results = {}
    topology_codes = {}
    topology_seconds = 0.0

    for radius in distinct_radii:
        path = scratch / f"topology_k{radius}.bin"
        stage_started = time.perf_counter()
        result, codes = dump_topology(args.core, document, radius, path)
        topology_seconds += time.perf_counter() - stage_started
        path.unlink(missing_ok=True)
        topology_results[radius] = result
        topology_codes[radius] = codes

    maximum_radius = max(distinct_radii)
    maximum_codes = topology_codes[maximum_radius]
    maximum_topology = topology_results[maximum_radius]
    expected_topology = workload["topology"]
    topology_checks = {
        "unique_span_checks": "baseline_unique_spans",
        "unique_pair_scores": "baseline_unique_pairs",
        "candidate_boundary_positions": "candidate_boundary_positions",
        "reachable_state_count": "reachable_state_count",
        "transitions_evaluated": "transitions_evaluated",
        "peak_layer_states": "peak_layer_states",
        "threat_set_size": "threat_set_size",
    }
    for observed_key, expected_key in topology_checks.items():
        if maximum_topology[observed_key] != expected_topology[expected_key]:
            raise ValueError(
                f"frozen topology mismatch at row {index}: {observed_key}"
            )

    positions_by_radius = {}
    for radius in distinct_radii:
        positions_by_radius[radius] = assert_subset(
            topology_codes[radius],
            maximum_codes,
        )

    estimated_scratch = (
        len(maximum_codes)
        * (SIMMARK_DIMENSION * 8 + SIDECAR_RECORD_DTYPE.itemsize)
        + 2 * 1024 ** 3
    )
    if shutil.disk_usage(scratch).free < estimated_scratch:
        raise OSError(
            f"insufficient scratch space for manifest index {index}"
        )

    memmap_path = scratch / "embeddings.f32"
    features, provider = encode_span_memmap(
        document,
        maximum_codes,
        model,
        tokenizer,
        args.batch_size,
        args.text_chunk_size,
        memmap_path,
        lambda completed, total: heartbeat(
            stage="certificate_embedding",
            manifest_index=index,
            span_progress=[completed, total],
        ),
    )

    radius_outputs = {}
    core_seconds = 0.0
    sidecar_seconds = 0.0
    max_sidecar_bytes = 0
    original_raw = None
    for radius in distinct_radii:
        codes = topology_codes[radius]
        positions = positions_by_radius[radius]
        sidecar = scratch / f"features_k{radius}.bin"
        stage_started = time.perf_counter()
        sidecar_bytes = write_mode6_sidecar(
            sidecar,
            document.atom_count,
            codes,
            positions,
            features,
        )
        sidecar_seconds += time.perf_counter() - stage_started
        max_sidecar_bytes = max(max_sidecar_bytes, sidecar_bytes)

        stage_started = time.perf_counter()
        observed, stdout_sha = run_factorized_core(
            args.core,
            document,
            radius,
            sidecar,
        )
        core_seconds += time.perf_counter() - stage_started
        sidecar.unlink(missing_ok=True)

        observed_original = raw_sum_for_boundaries(
            document,
            document.original_boundaries,
            maximum_codes,
            features,
        )
        observed_witness = raw_sum_for_boundaries(
            document,
            observed["worst_boundaries"],
            maximum_codes,
            features,
        )
        original_error = abs(observed_original - observed["original_score"])
        witness_error = abs(observed_witness - observed["certified_score"])
        if (
            original_error > WITNESS_TOLERANCE
            or witness_error > WITNESS_TOLERANCE
        ):
            raise ValueError(
                f"core witness replay mismatch at row {index}, k={radius}"
            )
        if original_raw is None:
            original_raw = observed["original_score"]
        elif abs(original_raw - observed["original_score"]) > WITNESS_TOLERANCE:
            raise ValueError(
                f"core P0 score changed across radii at row {index}"
            )
        radius_outputs[radius] = {
            "effective_radius": radius,
            "raw_score": observed["certified_score"],
            "z_score": score_from_raw(
                observed["certified_score"],
                boundary_count,
                gamma,
            ),
            "original_raw_score": observed["original_score"],
            "worst_boundaries": list(observed["worst_boundaries"]),
            "threat_set_size": observed["threat_set_size"],
            "exact_move_histogram": {
                str(key): value
                for key, value in observed["exact_move_histogram"]
            },
            "reachable_state_count": observed["reachable_state_count"],
            "transitions_evaluated": observed["transitions_evaluated"],
            "peak_layer_states": observed["peak_layer_states"],
            "candidate_boundary_positions": observed[
                "candidate_boundary_positions"
            ],
            "unique_span_checks": observed["unique_span_checks"],
            "unique_pair_scores": observed["unique_pair_scores"],
            "original_replay_abs_error": original_error,
            "witness_replay_abs_error": witness_error,
            "core_stdout_sha256": stdout_sha,
        }
        heartbeat(
            stage="certificate_core",
            manifest_index=index,
            effective_radius=radius,
        )

    certificates = {}
    for requested_radius in range(4):
        selected = dict(radius_outputs[effective[requested_radius]])
        selected["requested_radius"] = requested_radius
        certificates[str(requested_radius)] = selected
    raw_scores = [
        certificates[str(radius)]["raw_score"] for radius in range(4)
    ]
    if any(
        current > previous + WITNESS_TOLERANCE
        for previous, current in zip(raw_scores, raw_scores[1:])
    ):
        raise ValueError(f"certificate scores are not monotone at row {index}")
    if (
        certificates["0"]["threat_set_size"] != 1
        or certificates["0"]["worst_boundaries"]
        != list(document.original_boundaries)
    ):
        raise ValueError(f"k=0 certificate is not P0 at row {index}")

    del features
    memmap_path.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "gamma_artifact_sha256": gamma_sha256,
        "stage": "exact_certificate",
        "manifest_index": index,
        **identity_fields(source),
        "gamma": gamma,
        "edge_count": boundary_count,
        "certificates": certificates,
        "provider": provider,
        "topology_seconds": topology_seconds,
        "sidecar_seconds": sidecar_seconds,
        "core_seconds": core_seconds,
        "max_sidecar_bytes": max_sidecar_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "contains_raw_text": False,
    }


def validate_certificate_checkpoint(
    row,
    index,
    source,
    gamma,
    gamma_sha256,
):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("gamma_artifact_sha256") != gamma_sha256
        or row.get("stage") != "exact_certificate"
        or row.get("manifest_index") != index
        or row.get("gamma") != gamma
    ):
        raise ValueError(f"invalid certificate checkpoint header: {index}")
    for key, value in identity_fields(source).items():
        if row.get(key) != value:
            raise ValueError(
                f"certificate checkpoint identity mismatch: {index}"
            )
    certificates = row.get("certificates")
    if not isinstance(certificates, dict) or set(certificates) != {
        "0", "1", "2", "3"
    }:
        raise ValueError(f"invalid certificate radii at row {index}")
    previous = math.inf
    boundary_count = source["unit_count"] - 1
    for radius in range(4):
        result = certificates[str(radius)]
        expected_effective = min(radius, boundary_count)
        if (
            result.get("requested_radius") != radius
            or result.get("effective_radius") != expected_effective
            or len(result.get("worst_boundaries", [])) != boundary_count
            or result.get("original_replay_abs_error", math.inf)
            > WITNESS_TOLERANCE
            or result.get("witness_replay_abs_error", math.inf)
            > WITNESS_TOLERANCE
        ):
            raise ValueError(
                f"invalid certificate payload at row {index}, k={radius}"
            )
        raw_score = float(result["raw_score"])
        z_score = float(result["z_score"])
        if not math.isfinite(raw_score) or not math.isfinite(z_score):
            raise ValueError(
                f"non-finite certificate at row {index}, k={radius}"
            )
        if raw_score > previous + WITNESS_TOLERANCE:
            raise ValueError(
                f"non-monotone checkpoint at row {index}, k={radius}"
            )
        if abs(score_from_raw(raw_score, boundary_count, gamma) - z_score) > 1e-12:
            raise ValueError(
                f"z-score mismatch at row {index}, k={radius}"
            )
        previous = raw_score
    return row


def clopper_pearson_lower(successes, trials, confidence=0.95):
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if successes == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(
        trials - successes,
        trials,
        confidence,
    )


def nearest_rank(values, quantile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def distribution(values):
    selected = list(values)
    if not selected or any(not math.isfinite(float(value)) for value in selected):
        raise ValueError("distribution values must be finite and nonempty")
    return {
        "min": min(selected),
        "p50": nearest_rank(selected, 0.50),
        "p95": nearest_rank(selected, 0.95),
        "max": max(selected),
        "mean": float(sum(selected) / len(selected)),
    }


def summarize_results(rows, gamma_artifact):
    by_index = {row["manifest_index"]: row for row in rows}
    if set(by_index) != set(CERTIFICATE_RANGE):
        raise ValueError("final summary requires all 1,800 certificate rows")
    calibration_rows = [by_index[index] for index in CALIBRATION_RANGE]
    test_rows = [by_index[index] for index in TEST_RANGE]
    positive_rows = [by_index[index] for index in POSITIVE_RANGE]

    calibrations = {}
    thresholds = {}
    for radius in range(4):
        scores = [
            row["certificates"][str(radius)]["z_score"]
            for row in calibration_rows
        ]
        calibration = split_conformal_calibrate(scores, TARGET_FPR)
        calibrations[str(radius)] = calibration.to_dict()
        thresholds[radius] = calibration.threshold

    test = {}
    positive = {}
    baseline_positive = [
        row
        for row in positive_rows
        if row["certificates"]["0"]["z_score"] > thresholds[0]
    ]
    for radius in range(4):
        key = str(radius)
        test_successes = sum(
            row["certificates"][key]["z_score"] > thresholds[radius]
            for row in test_rows
        )
        test_confidence = 0.975 if radius == 1 else 0.95
        test[key] = {
            "trials": len(test_rows),
            "strict_exceedances": test_successes,
            "rate": test_successes / len(test_rows),
            "one_sided_cp_confidence": test_confidence,
            "one_sided_cp_upper": clopper_pearson_upper(
                test_successes,
                len(test_rows),
                test_confidence,
            ),
        }

        positive_successes = sum(
            row["certificates"][key]["z_score"] > thresholds[radius]
            for row in positive_rows
        )
        retained = sum(
            row["certificates"][key]["z_score"] > thresholds[radius]
            for row in baseline_positive
        )
        score_drops = [
            row["certificates"]["0"]["z_score"]
            - row["certificates"][key]["z_score"]
            for row in positive_rows
        ]
        threat_sizes = [
            row["certificates"][key]["threat_set_size"]
            for row in positive_rows
        ]
        positive[key] = {
            "trials": len(positive_rows),
            "strict_exceedances": positive_successes,
            "rate": positive_successes / len(positive_rows),
            "one_sided_cp_confidence": 0.95,
            "one_sided_cp_lower": clopper_pearson_lower(
                positive_successes,
                len(positive_rows),
                0.95,
            ),
            "c0_minus_ck": distribution(score_drops),
            "raw_positive_count": len(baseline_positive),
            "raw_positive_retained": retained,
            "raw_positive_conditional_retention": (
                retained / len(baseline_positive)
                if baseline_positive
                else None
            ),
            "threat_set_size": distribution(threat_sizes),
        }

    gates = {
        "simmark_k1_test_cp_upper_le_0_08": (
            test["1"]["one_sided_cp_upper"] <= 0.08
        ),
        "simmark_clean_baseline_tpr_ge_0_80": positive["0"]["rate"] >= 0.80,
        "simmark_k1_certified_rate_ge_0_20": positive["1"]["rate"] >= 0.20,
        "simmark_k1_cp_lower_gt_0_10": (
            positive["1"]["one_sided_cp_lower"] > 0.10
        ),
    }
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment": "certified_resegmentation_wp2_v9_simmark_long1",
        "status": (
            "simmark_stage_pass" if all(gates.values())
            else "simmark_stage_stop"
        ),
        "scope": "simmark_only_semstamp_pending",
        "gamma": gamma_artifact,
        "calibration": calibrations,
        "test": test,
        "positive": positive,
        "gates": gates,
        "contains_raw_text": False,
        "completed_at": utc_now(),
    }


class Heartbeat:
    def __init__(self, path, started_at, completed_fit, completed_certificates):
        self.path = Path(path)
        self.started_at = started_at
        self.completed_fit = completed_fit
        self.completed_certificates = completed_certificates
        self.last_write = 0.0

    def __call__(self, force=False, **fields):
        now = time.monotonic()
        if not force and now - self.last_write < 30:
            return
        payload = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "pid": os.getpid(),
            "process_started_at": self.started_at,
            "updated_at": utc_now(),
            "completed_fit_documents": self.completed_fit,
            "completed_certificate_documents": self.completed_certificates,
            **fields,
        }
        atomic_write_json(self.path, payload)
        self.last_write = now


def clean_scratch(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def collect_existing(directory, allowed_indices, validator):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    allowed = set(allowed_indices)
    output = {}
    for path in sorted(directory.glob("*.json")):
        if not path.stem.isdigit():
            raise ValueError(f"unexpected checkpoint filename: {path.name}")
        index = int(path.stem)
        if index not in allowed or index in output:
            raise ValueError(f"unexpected checkpoint index: {index}")
        output[index] = validator(load_json(path), index)
    return output


def write_results_manifest(args, output_dir):
    output_dir = Path(output_dir)
    paths = [
        args.protocol,
        args.parent_protocol,
        args.identity_manifest,
        args.workload_manifest,
        args.core_source,
        args.core,
        Path(__file__).resolve(),
        output_dir / "gamma.json",
        output_dir / "final_summary.json",
        *sorted((output_dir / "fit").glob("*.json")),
        *sorted((output_dir / "certificates").glob("*.json")),
    ]
    lines = []
    for path in paths:
        try:
            display = path.relative_to(args.root)
        except ValueError:
            display = path
        lines.append(f"{sha256_file(path)}  {display}\n")
    atomic_write_bytes(
        output_dir / "RESULTS.sha256",
        "".join(lines).encode("ascii"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-chunk-size", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def run(args):
    import nltk

    if args.batch_size != 128:
        raise ValueError("the frozen WP2-v9 batch size is 128")
    if args.text_chunk_size < args.batch_size:
        raise ValueError("text chunk size must be at least the batch size")
    if args.min_free_gib < 10:
        raise ValueError("the frozen minimum free-space gate is 10 GiB")
    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(args.nltk_data))
    if nltk.__version__ != DESIGN_SYSTEMS["simmark"]["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.find("tokenizers/punkt_tab/english/")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.work_dir).free < args.min_free_gib * 1024 ** 3:
        raise OSError("free scratch space is below the frozen 10 GiB gate")

    identities, identity_rows, workload_rows, revision = (
        validate_frozen_inputs(args)
    )
    official_identity = verify_inputs(
        args.root,
        args.model_dir,
        "simmark",
    )
    tokenizer = load_tokenizer(args.model_dir, "simmark")
    preflight = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "pass",
        "created_at": utc_now(),
        "data_revision": revision,
        "identity_manifest_sha256": sha256_file(args.identity_manifest),
        "workload_manifest_sha256": sha256_file(args.workload_manifest),
        "parent_protocol_sha256": sha256_file(args.parent_protocol),
        "core_source_sha256": sha256_file(args.core_source),
        "core_executable_sha256": sha256_file(args.core),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "official_identity": official_identity,
        "row_count": len(identity_rows),
        "certificate_row_count": len(CERTIFICATE_RANGE),
        "cuda_available": torch.cuda.is_available(),
        "free_scratch_bytes": shutil.disk_usage(args.work_dir).free,
        "environment": package_versions(),
        "contains_raw_text": False,
    }
    atomic_write_json(args.output_dir / "preflight.json", preflight)
    if args.validate_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen detector contract")

    scratch = args.work_dir / "wp2_v9_simmark_active"
    clean_scratch(scratch)
    fit_dir = args.output_dir / "fit"
    certificate_dir = args.output_dir / "certificates"
    started_at = utc_now()

    fit_rows = collect_existing(
        fit_dir,
        FIT_RANGE,
        lambda row, index: validate_fit_checkpoint(
            row,
            index,
            identity_rows[index],
        ),
    )
    heartbeat = Heartbeat(
        args.output_dir / "heartbeat.json",
        started_at,
        len(fit_rows),
        0,
    )
    heartbeat(force=True, stage="loading_model")

    load_started = time.perf_counter()
    model = load_simmark(args.model_dir)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    atomic_write_json(args.output_dir / "runtime.json", {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "pid": os.getpid(),
        "process_started_at": started_at,
        "model_load_seconds": model_load_seconds,
        "batch_size": args.batch_size,
        "text_chunk_size": args.text_chunk_size,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "platform": platform.platform(),
        "environment": package_versions(),
    })

    for index in FIT_RANGE:
        if index in fit_rows:
            continue
        heartbeat(force=True, stage="gamma_fit", manifest_index=index)
        output = process_fit_document(
            index,
            identities[index],
            identity_rows[index],
            model,
            args.batch_size,
        )
        path = fit_dir / f"{index:04d}.json"
        atomic_write_json(path, output)
        fit_rows[index] = validate_fit_checkpoint(
            load_json(path),
            index,
            identity_rows[index],
        )
        heartbeat.completed_fit = len(fit_rows)
        heartbeat(
            force=True,
            stage="gamma_fit",
            manifest_index=index,
        )
        print(json.dumps({
            "stage": "gamma_fit",
            "completed": len(fit_rows),
            "total": len(FIT_RANGE),
            "manifest_index": index,
        }, sort_keys=True), flush=True)

    ordered_fit_rows = [fit_rows[index] for index in FIT_RANGE]
    fit_paths = [fit_dir / f"{index:04d}.json" for index in FIT_RANGE]
    expected_gamma = gamma_artifact_from_rows(ordered_fit_rows, fit_paths)
    gamma_path = args.output_dir / "gamma.json"
    if gamma_path.exists():
        gamma_artifact = validate_gamma_artifact(
            load_json(gamma_path),
            expected_gamma,
        )
    else:
        atomic_write_json(gamma_path, expected_gamma)
        gamma_artifact = validate_gamma_artifact(
            load_json(gamma_path),
            expected_gamma,
        )
    gamma = gamma_artifact["gamma"]
    gamma_sha256 = sha256_file(gamma_path)

    certificate_rows = collect_existing(
        certificate_dir,
        CERTIFICATE_RANGE,
        lambda row, index: validate_certificate_checkpoint(
            row,
            index,
            identity_rows[index],
            gamma,
            gamma_sha256,
        ),
    )
    heartbeat.completed_certificates = len(certificate_rows)
    heartbeat(force=True, stage="exact_certificates", gamma=gamma)

    for index in CERTIFICATE_RANGE:
        if index in certificate_rows:
            continue
        clean_scratch(scratch)
        heartbeat(
            force=True,
            stage="certificate_topology",
            manifest_index=index,
        )
        try:
            output = process_certificate_document(
                args,
                index,
                identities[index],
                identity_rows[index],
                workload_rows[index],
                model,
                tokenizer,
                gamma,
                gamma_sha256,
                scratch,
                heartbeat,
            )
            path = certificate_dir / f"{index:04d}.json"
            atomic_write_json(path, output)
            certificate_rows[index] = validate_certificate_checkpoint(
                load_json(path),
                index,
                identity_rows[index],
                gamma,
                gamma_sha256,
            )
        finally:
            clean_scratch(scratch)
        heartbeat.completed_certificates = len(certificate_rows)
        heartbeat(
            force=True,
            stage="exact_certificates",
            manifest_index=index,
        )
        print(json.dumps({
            "stage": "exact_certificates",
            "completed": len(certificate_rows),
            "total": len(CERTIFICATE_RANGE),
            "manifest_index": index,
            "elapsed_seconds": output["elapsed_seconds"],
        }, sort_keys=True), flush=True)

    ordered_certificates = [
        certificate_rows[index] for index in CERTIFICATE_RANGE
    ]
    summary = summarize_results(ordered_certificates, gamma_artifact)
    atomic_write_json(args.output_dir / "final_summary.json", summary)
    write_results_manifest(args, args.output_dir)
    heartbeat.completed_certificates = len(certificate_rows)
    heartbeat(
        force=True,
        stage="complete",
        status=summary["status"],
        gamma=gamma,
    )
    print(json.dumps({
        "stage": "complete",
        "status": summary["status"],
        "gates": summary["gates"],
    }, sort_keys=True), flush=True)


def main():
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        try:
            atomic_write_json(args.output_dir / "last_failure.json", {
                "schema_version": 1,
                "protocol_sha256": PROTOCOL_SHA256,
                "pid": os.getpid(),
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            })
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
