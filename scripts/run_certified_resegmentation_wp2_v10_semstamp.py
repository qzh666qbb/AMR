"""Run the resumable WP2-v10 conditional SemStamp experiment."""

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
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)
from certified_resegmentation_wp2_v6_features import (
    MAGIC,
    SEMSTAMP_MODE,
    serialize_core_problem,
)
from run_certified_resegmentation_wp2_detector_equivalence import (
    package_versions,
    verify_inputs,
)
from run_certified_resegmentation_wp2_v6_provider import (
    load_semstamp,
    semstamp_features,
)


PROTOCOL_SHA256 = (
    "cfded9faee1d4d9bf617a05393b185ffbe054d9bb0800bebfa7fdaba8ebe663b"
)
IDENTITY_MANIFEST_SHA256 = (
    "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218"
)
WORKLOAD_MANIFEST_SHA256 = (
    "783916556e80d8900b5e64c922c5c33e6c39c9d8c1c2f35d54ef81750d38f23e"
)
PARENT_V3_PROTOCOL_SHA256 = (
    "377f205d3e053cdbed046b1c07721a57732dc6cb5122f60ec11d225586d0f253"
)
PARENT_V6_PROTOCOL_SHA256 = (
    "ddab1e0f8d3714dda22ae383395ea91e14ff10ac55d59e63b0708806d533a350"
)
CORE_SOURCE_SHA256 = (
    "d77d23c0e44a343c154e93d9897ee3e40c1a05194d9200108ff7e619f52e1360"
)
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
FIT_RANGE = range(0, 200)
CALIBRATION_RANGE = range(200, 600)
TEST_RANGE = range(600, 1000)
POSITIVE_RANGE = range(1000, 2000)
EXCLUDED_MANIFEST_INDEX = 1432
FORMAL_INDICES = tuple(
    index
    for index in range(200, 2000)
    if index != EXCLUDED_MANIFEST_INDEX
)
TARGET_FPR = 0.05
WITNESS_TOLERANCE = 0.0
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
    expected = {key: source[key] for key in observed}
    if observed != expected:
        raise ValueError(
            f"structural identity mismatch at manifest index {manifest_index}"
        )


def materialize_document(item, source, manifest_index):
    from nltk.tokenize import sent_tokenize

    raw_value = item["raw_value"]
    document = canonicalize_structured(
        raw_value,
        sentence_splitter=sent_tokenize if isinstance(raw_value, str) else None,
    )
    validate_document(document, source, manifest_index)
    return document


def validate_frozen_inputs(args):
    paths_and_hashes = (
        (args.protocol, PROTOCOL_SHA256, "WP2-v10 protocol"),
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
            args.parent_v3_protocol,
            PARENT_V3_PROTOCOL_SHA256,
            "WP2-v3 protocol",
        ),
        (
            args.parent_v6_protocol,
            PARENT_V6_PROTOCOL_SHA256,
            "WP2-v6 protocol",
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

    identities, _split, revision = frozen_identities(args.root, "semstamp")
    identity_rows = load_jsonl(args.identity_manifest)
    workload_rows = load_jsonl(args.workload_manifest)
    if not (
        len(identities) == len(identity_rows) == len(workload_rows) == 2000
    ):
        raise ValueError("frozen SemStamp manifests must contain 2,000 rows")

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
            or workload.get("system") != "semstamp"
        ):
            raise ValueError(f"frozen identity mismatch at row {index}")
        for key in ("structural_tape_sha256", "atom_count", "unit_count"):
            if row.get(key) != workload.get(key):
                raise ValueError(
                    f"manifest structural mismatch at row {index}: {key}"
                )

        eligible = bool(row["candidate_eligible"][
            "STRUCTURAL_SEPARATOR_NATIVE_512"
        ])
        if index == EXCLUDED_MANIFEST_INDEX:
            if (
                eligible
                or workload.get("eligible")
                or row.get("candidate_exclusion_codes", {}).get(
                    "STRUCTURAL_SEPARATOR_NATIVE_512"
                ) != ["invalid_p0"]
            ):
                raise ValueError("frozen exclusion at row 1432 changed")
            continue
        if not eligible or not workload.get("eligible"):
            raise ValueError(f"unexpected ineligible frozen row {index}")
        materialize_document(item, row, index)

    observed_formal = tuple(
        index
        for index, row in enumerate(workload_rows)
        if index >= 200 and row["eligible"]
    )
    if observed_formal != FORMAL_INDICES:
        raise ValueError("formal SemStamp cohort indices changed")
    return identities, identity_rows, workload_rows, revision


def validate_relation(relation):
    relation = bytes(relation)
    if (
        len(relation) != 64
        or any(value not in {0, 1} for value in relation)
        or sha256_bytes(relation) != RELATION_SHA256
    ):
        raise ValueError("official SemStamp relation mismatch")
    if any(sum(relation[offset:offset + 8]) != 2 for offset in range(0, 64, 8)):
        raise ValueError("SemStamp relation rows must each contain two hits")
    return relation


def validate_regions(values, expected_rows):
    array = np.asarray(values)
    if array.shape != (expected_rows,):
        raise ValueError(
            f"invalid region shape {array.shape}; expected {(expected_rows,)}"
        )
    if (
        not np.issubdtype(array.dtype, np.integer)
        or np.any(array < 0)
        or np.any(array > 7)
    ):
        raise ValueError("SemStamp returned an invalid region")
    return array.astype(np.uint8, copy=False)


def raw_sum_from_regions(regions, relation):
    values = np.asarray(regions)
    if len(values) < 2:
        raise ValueError("SemStamp requires at least two units")
    return int(sum(
        relation[int(previous) * 8 + int(current)]
        for previous, current in zip(values[:-1], values[1:])
    ))


def score_from_raw(raw_sum, edge_count):
    if edge_count < 1:
        raise ValueError("SemStamp requires at least one edge")
    value = float(raw_sum)
    if not math.isfinite(value):
        raise ValueError("SemStamp raw score must be finite")
    return float(
        (value - 0.25 * edge_count)
        / math.sqrt(edge_count * 0.25 * 0.75)
    )


def process_p0_document(index, item, source, model, relation):
    started = time.perf_counter()
    document = materialize_document(item, source, index)
    units = list(document.preprocess_partition("semstamp"))
    if len(units) != source["unit_count"]:
        raise ValueError(f"P0 unit count mismatch at row {index}")
    with torch.inference_mode():
        regions = validate_regions(
            semstamp_features(model, units),
            len(units),
        )
    raw_score = raw_sum_from_regions(regions, relation)
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "relation_sha256": RELATION_SHA256,
        "stage": "p0_detector_health",
        "manifest_index": index,
        **identity_fields(source),
        "edge_count": len(units) - 1,
        "regions": [int(value) for value in regions],
        "regions_sha256": sha256_bytes(regions.tobytes(order="C")),
        "raw_score": raw_score,
        "z_score": score_from_raw(raw_score, len(units) - 1),
        "elapsed_seconds": time.perf_counter() - started,
        "contains_raw_text": False,
    }


def validate_p0_checkpoint(row, index, source, relation):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("relation_sha256") != RELATION_SHA256
        or row.get("stage") != "p0_detector_health"
        or row.get("manifest_index") != index
    ):
        raise ValueError(f"invalid P0 checkpoint header at row {index}")
    for key, value in identity_fields(source).items():
        if row.get(key) != value:
            raise ValueError(f"P0 checkpoint identity mismatch at row {index}")
    regions = validate_regions(row.get("regions"), source["unit_count"])
    if sha256_bytes(regions.tobytes(order="C")) != row.get("regions_sha256"):
        raise ValueError(f"P0 region digest mismatch at row {index}")
    edge_count = source["unit_count"] - 1
    raw_score = raw_sum_from_regions(regions, relation)
    if (
        row.get("edge_count") != edge_count
        or row.get("raw_score") != raw_score
        or abs(row.get("z_score") - score_from_raw(raw_score, edge_count))
        > 1e-12
    ):
        raise ValueError(f"P0 score mismatch at row {index}")
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


def summarize_p0(rows):
    by_index = {row["manifest_index"]: row for row in rows}
    if set(by_index) != set(FORMAL_INDICES):
        raise ValueError("P0 summary requires all 1,799 formal rows")
    calibration_rows = [by_index[index] for index in CALIBRATION_RANGE]
    test_rows = [by_index[index] for index in TEST_RANGE]
    positive_rows = [
        by_index[index]
        for index in POSITIVE_RANGE
        if index != EXCLUDED_MANIFEST_INDEX
    ]
    calibration = split_conformal_calibrate(
        [row["z_score"] for row in calibration_rows],
        TARGET_FPR,
    )
    threshold = calibration.threshold
    test_successes = sum(
        row["z_score"] > threshold for row in test_rows
    )
    positive_successes = sum(
        row["z_score"] > threshold for row in positive_rows
    )
    test_upper = clopper_pearson_upper(
        test_successes,
        len(test_rows),
        0.95,
    )
    positive_lower = clopper_pearson_lower(
        positive_successes,
        len(positive_rows),
        0.95,
    )
    gates = {
        "semstamp_p0_test_cp95_upper_le_0_08": test_upper <= 0.08,
        "semstamp_clean_baseline_tpr_ge_0_80": (
            positive_successes / len(positive_rows) >= 0.80
        ),
    }
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment": "certified_resegmentation_wp2_v10_semstamp1",
        "stage": "p0_health_complete",
        "status": (
            "semstamp_p0_pass" if all(gates.values())
            else "semstamp_p0_stop"
        ),
        "calibration": calibration.to_dict(),
        "test": {
            "trials": len(test_rows),
            "strict_exceedances": test_successes,
            "rate": test_successes / len(test_rows),
            "one_sided_cp_confidence": 0.95,
            "one_sided_cp_upper": test_upper,
        },
        "positive": {
            "trials": len(positive_rows),
            "strict_exceedances": positive_successes,
            "rate": positive_successes / len(positive_rows),
            "one_sided_cp_confidence": 0.95,
            "one_sided_cp_lower": positive_lower,
        },
        "gates": gates,
        "excluded_manifest_indices": [EXCLUDED_MANIFEST_INDEX],
        "contains_raw_text": False,
        "completed_at": utc_now(),
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
        if (
            mode != SEMSTAMP_MODE
            or token_count != expected_n
            or dimension != 1
        ):
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


def topology_metric(result, name):
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


def compute_legality(document, codes, tokenizer, chunk_size):
    allowed = np.empty(len(codes), dtype=np.bool_)
    base = document.atom_count + 1
    max_native_tokens = 0
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
        tokenizer_texts = [
            official_tokenizer_text("semstamp", text, int(start > 0))
            for text, (start, _end) in zip(texts, spans)
        ]
        counts = batched_token_counts(tokenizer, tokenizer_texts)
        if counts:
            max_native_tokens = max(max_native_tokens, max(counts))
        allowed[offset:offset + len(counts)] = (
            np.asarray(counts) <= NATIVE_TOKEN_LIMIT
        )
    return allowed, max_native_tokens


def pruned_query_topology(document, codes, allowed, radius):
    base = document.atom_count + 1
    legality = {
        int(code): bool(value)
        for code, value in zip(codes, allowed)
    }
    queried = set()

    def span_allowed(start, end):
        code = start * base + end
        queried.add(code)
        try:
            return legality[code]
        except KeyError as error:
            raise AssertionError(
                "pruned topology escaped all-allowed superset"
            ) from error

    result = exact_grouped_sparse_pairwise_certificate(
        SparsePairwiseProblem(
            token_count=document.atom_count,
            original_boundaries=document.original_boundaries,
            max_boundary_moves=radius,
            span_allowed=span_allowed,
            pair_scorer=lambda _left, _middle, _right: 0.0,
            normalization=1.0,
            bias=0.0,
            threshold=0.0,
        )
    )
    actual_codes = np.asarray(sorted(queried), dtype=np.uint64)
    positions = assert_subset(actual_codes, codes)
    actual_allowed = allowed[positions]
    if result.unique_span_checks != len(actual_codes):
        raise ValueError("pruned topology span count mismatch")
    return result, actual_codes, actual_allowed


def prepare_topologies(
    args,
    document,
    tokenizer,
    workload,
    scratch,
):
    boundary_count = len(document.original_boundaries)
    effective = {
        radius: min(radius, boundary_count)
        for radius in range(4)
    }
    distinct_radii = sorted(set(effective.values()))
    all_results = {}
    all_codes = {}
    topology_seconds = 0.0
    for radius in distinct_radii:
        path = scratch / f"topology_k{radius}.bin"
        started = time.perf_counter()
        result, codes = dump_topology(args.core, document, radius, path)
        topology_seconds += time.perf_counter() - started
        path.unlink(missing_ok=True)
        all_results[radius] = result
        all_codes[radius] = codes

    maximum_radius = max(distinct_radii)
    maximum_all_codes = all_codes[maximum_radius]
    started = time.perf_counter()
    maximum_allowed, max_native_tokens = compute_legality(
        document,
        maximum_all_codes,
        tokenizer,
        args.text_chunk_size,
    )
    topology_seconds += time.perf_counter() - started

    actual = {}
    for radius in distinct_radii:
        positions = assert_subset(all_codes[radius], maximum_all_codes)
        radius_allowed = maximum_allowed[positions]
        if np.all(radius_allowed):
            result = all_results[radius]
            codes = all_codes[radius]
            allowed = radius_allowed
        else:
            started = time.perf_counter()
            result, codes, allowed = pruned_query_topology(
                document,
                all_codes[radius],
                radius_allowed,
                radius,
            )
            topology_seconds += time.perf_counter() - started
        actual[radius] = {
            "result": result,
            "codes": codes,
            "allowed": allowed,
            "all_result": all_results[radius],
            "all_codes": all_codes[radius],
        }

    maximum = actual[maximum_radius]
    for radius in distinct_radii:
        assert_subset(actual[radius]["codes"], maximum["codes"])

    expected = workload["topology"]
    checks = {
        "baseline_unique_spans": len(maximum["all_codes"]),
        "baseline_unique_pairs": topology_metric(
            maximum["all_result"],
            "unique_pair_scores",
        ),
        "unique_query_spans": len(maximum["codes"]),
        "legal_query_spans": int(np.count_nonzero(maximum["allowed"])),
        "actual_unique_pairs": topology_metric(
            maximum["result"],
            "unique_pair_scores",
        ),
        "candidate_boundary_positions": topology_metric(
            maximum["result"],
            "candidate_boundary_positions",
        ),
        "reachable_state_count": topology_metric(
            maximum["result"],
            "reachable_state_count",
        ),
        "transitions_evaluated": topology_metric(
            maximum["result"],
            "transitions_evaluated",
        ),
        "peak_layer_states": topology_metric(
            maximum["result"],
            "peak_layer_states",
        ),
        "threat_set_size": topology_metric(
            maximum["result"],
            "threat_set_size",
        ),
    }
    mismatches = {
        key: (value, expected.get(key))
        for key, value in checks.items()
        if value != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"frozen topology mismatch: {mismatches}")
    return effective, distinct_radii, actual, {
        "elapsed_seconds": topology_seconds,
        "max_native_tokens": max_native_tokens,
    }


def encode_region_memmap(
    document,
    codes,
    allowed,
    model,
    chunk_size,
    path,
    progress,
):
    regions = np.memmap(
        path,
        dtype="u1",
        mode="w+",
        shape=(len(codes),),
    )
    regions[:] = 255
    base = document.atom_count + 1
    encoded_spans = 0
    started = time.perf_counter()
    for offset in range(0, len(codes), chunk_size):
        selected_codes = codes[offset:offset + chunk_size]
        selected_allowed = allowed[offset:offset + chunk_size]
        local_positions = np.flatnonzero(selected_allowed)
        if len(local_positions):
            legal_codes = selected_codes[local_positions]
            texts = [
                document.preprocess_unit(
                    "semstamp",
                    int(code // base),
                    int(code % base),
                )
                for code in legal_codes
            ]
            with torch.inference_mode():
                values = validate_regions(
                    semstamp_features(model, texts),
                    len(texts),
                )
            regions[offset + local_positions] = values
            encoded_spans += len(values)
        progress(offset + len(selected_codes), len(codes))
    regions.flush()
    torch.cuda.synchronize()
    if (
        np.any(regions[allowed] > 7)
        or np.any(regions[~allowed] != 255)
    ):
        raise ValueError("region memmap legality mismatch")
    return regions, {
        "span_count": len(codes),
        "legal_span_count": int(np.count_nonzero(allowed)),
        "encoded_span_count": encoded_spans,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": Path(path).stat().st_size,
    }


def write_mode5_sidecar(
    path,
    token_count,
    codes,
    allowed,
    regions,
    relation,
    chunk_size=8192,
):
    if not (len(codes) == len(allowed) == len(regions)):
        raise ValueError("sidecar arrays have inconsistent lengths")
    with Path(path).open("wb", buffering=8 * 1024 * 1024) as handle:
        handle.write(MAGIC)
        handle.write(struct.pack(
            "<IIQI",
            SEMSTAMP_MODE,
            token_count,
            len(codes),
            1,
        ))
        handle.write(relation)
        base = token_count + 1
        for offset in range(0, len(codes), chunk_size):
            payload = bytearray()
            stop = min(offset + chunk_size, len(codes))
            for position in range(offset, stop):
                code = int(codes[position])
                is_allowed = bool(allowed[position])
                payload.extend(struct.pack(
                    "<IIB",
                    code // base,
                    code % base,
                    int(is_allowed),
                ))
                if is_allowed:
                    region = int(regions[position])
                    if not 0 <= region <= 7:
                        raise ValueError("invalid legal span region")
                    payload.extend(struct.pack("<B", region))
                elif int(regions[position]) != 255:
                    raise ValueError("illegal span unexpectedly has a region")
            handle.write(payload)
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
        input=serialize_core_problem(problem, SEMSTAMP_MODE, sidecar),
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


def raw_sum_for_boundaries(
    document,
    boundaries,
    codes,
    regions,
    relation,
):
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
    selected = np.asarray(regions[positions], dtype=np.uint8)
    if np.any(selected > 7):
        raise ValueError("witness includes an illegal span")
    return raw_sum_from_regions(selected, relation)


def process_certificate_document(
    args,
    index,
    item,
    source,
    workload,
    p0_row,
    model,
    tokenizer,
    relation,
    scratch,
    heartbeat,
):
    started = time.perf_counter()
    document = materialize_document(item, source, index)
    effective, distinct_radii, topologies, topology_stats = (
        prepare_topologies(
            args,
            document,
            tokenizer,
            workload,
            scratch,
        )
    )
    maximum_radius = max(distinct_radii)
    maximum = topologies[maximum_radius]

    estimated_scratch = len(maximum["codes"]) * 32 + 2 * 1024 ** 3
    if shutil.disk_usage(scratch).free < estimated_scratch:
        raise OSError(
            f"insufficient scratch space for manifest index {index}"
        )

    memmap_path = scratch / "regions.u8"
    regions, provider = encode_region_memmap(
        document,
        maximum["codes"],
        maximum["allowed"],
        model,
        args.text_chunk_size,
        memmap_path,
        lambda completed, total: heartbeat(
            stage="certificate_regions",
            manifest_index=index,
            span_progress=[completed, total],
        ),
    )
    provider["max_native_tokens"] = topology_stats["max_native_tokens"]

    radius_outputs = {}
    core_seconds = 0.0
    sidecar_seconds = 0.0
    max_sidecar_bytes = 0
    original_raw = None
    for radius in distinct_radii:
        topology = topologies[radius]
        positions = assert_subset(topology["codes"], maximum["codes"])
        selected_regions = regions[positions]
        sidecar = scratch / f"features_k{radius}.bin"
        stage_started = time.perf_counter()
        sidecar_bytes = write_mode5_sidecar(
            sidecar,
            document.atom_count,
            topology["codes"],
            topology["allowed"],
            selected_regions,
            relation,
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
            maximum["codes"],
            regions,
            relation,
        )
        observed_witness = raw_sum_for_boundaries(
            document,
            observed["worst_boundaries"],
            maximum["codes"],
            regions,
            relation,
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
        elif original_raw != observed["original_score"]:
            raise ValueError(
                f"core P0 score changed across radii at row {index}"
            )
        result = topology["result"]
        radius_outputs[radius] = {
            "effective_radius": radius,
            "raw_score": observed["certified_score"],
            "z_score": score_from_raw(
                observed["certified_score"],
                len(document.original_boundaries),
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
            "legal_span_checks": int(np.count_nonzero(topology["allowed"])),
            "unique_pair_scores": observed["unique_pair_scores"],
            "topology_reference_unique_pairs": topology_metric(
                result,
                "unique_pair_scores",
            ),
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
        current > previous
        for previous, current in zip(raw_scores, raw_scores[1:])
    ):
        raise ValueError(f"certificate scores are not monotone at row {index}")
    if (
        certificates["0"]["threat_set_size"] != 1
        or certificates["0"]["worst_boundaries"]
        != list(document.original_boundaries)
        or certificates["0"]["raw_score"] != p0_row["raw_score"]
        or certificates["0"]["z_score"] != p0_row["z_score"]
    ):
        raise ValueError(f"full/P0 mismatch at row {index}")

    del regions
    memmap_path.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "relation_sha256": RELATION_SHA256,
        "stage": "exact_certificate",
        "manifest_index": index,
        **identity_fields(source),
        "edge_count": len(document.original_boundaries),
        "certificates": certificates,
        "provider": provider,
        "topology_seconds": topology_stats["elapsed_seconds"],
        "sidecar_seconds": sidecar_seconds,
        "core_seconds": core_seconds,
        "max_sidecar_bytes": max_sidecar_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "contains_raw_text": False,
    }


def validate_certificate_checkpoint(row, index, source, p0_row):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("relation_sha256") != RELATION_SHA256
        or row.get("stage") != "exact_certificate"
        or row.get("manifest_index") != index
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
            or result.get("original_replay_abs_error") != 0
            or result.get("witness_replay_abs_error") != 0
        ):
            raise ValueError(
                f"invalid certificate payload at row {index}, k={radius}"
            )
        raw_score = float(result["raw_score"])
        z_score = float(result["z_score"])
        if (
            not math.isfinite(raw_score)
            or not math.isfinite(z_score)
            or raw_score > previous
            or abs(score_from_raw(raw_score, boundary_count) - z_score)
            > 1e-12
        ):
            raise ValueError(
                f"invalid certificate score at row {index}, k={radius}"
            )
        previous = raw_score
    if (
        certificates["0"]["raw_score"] != p0_row["raw_score"]
        or certificates["0"]["z_score"] != p0_row["z_score"]
    ):
        raise ValueError(f"saved full/P0 mismatch at row {index}")
    return row


def nearest_rank(values, quantile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def distribution(values):
    selected = list(values)
    if not selected or any(
        not math.isfinite(float(value)) for value in selected
    ):
        raise ValueError("distribution values must be finite and nonempty")
    return {
        "min": min(selected),
        "p50": nearest_rank(selected, 0.50),
        "p95": nearest_rank(selected, 0.95),
        "max": max(selected),
        "mean": float(sum(selected) / len(selected)),
    }


def summarize_results(rows, p0_summary):
    by_index = {row["manifest_index"]: row for row in rows}
    if set(by_index) != set(FORMAL_INDICES):
        raise ValueError("final summary requires all 1,799 certificate rows")
    calibration_rows = [by_index[index] for index in CALIBRATION_RANGE]
    test_rows = [by_index[index] for index in TEST_RANGE]
    positive_rows = [
        by_index[index]
        for index in POSITIVE_RANGE
        if index != EXCLUDED_MANIFEST_INDEX
    ]

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
            "c0_minus_ck": distribution([
                row["certificates"]["0"]["z_score"]
                - row["certificates"][key]["z_score"]
                for row in positive_rows
            ]),
            "raw_positive_count": len(baseline_positive),
            "raw_positive_retained": retained,
            "raw_positive_conditional_retention": (
                retained / len(baseline_positive)
                if baseline_positive
                else None
            ),
            "threat_set_size": distribution([
                row["certificates"][key]["threat_set_size"]
                for row in positive_rows
            ]),
        }

    if (
        calibrations["0"] != p0_summary["calibration"]
        or test["0"]["strict_exceedances"]
        != p0_summary["test"]["strict_exceedances"]
        or positive["0"]["strict_exceedances"]
        != p0_summary["positive"]["strict_exceedances"]
    ):
        raise ValueError("full summary does not reproduce the P0 health stage")

    gates = {
        "semstamp_k1_test_cp_upper_le_0_08": (
            test["1"]["one_sided_cp_upper"] <= 0.08
        ),
        "semstamp_clean_baseline_tpr_ge_0_80": (
            positive["0"]["rate"] >= 0.80
        ),
        "semstamp_k1_certified_rate_ge_0_20": (
            positive["1"]["rate"] >= 0.20
        ),
        "semstamp_k1_cp_lower_gt_0_10": (
            positive["1"]["one_sided_cp_lower"] > 0.10
        ),
    }
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment": "certified_resegmentation_wp2_v10_semstamp1",
        "status": (
            "semstamp_stage_pass" if all(gates.values())
            else "semstamp_certificate_stop"
        ),
        "scope": "semstamp_only_simmark_v9_stopped",
        "p0_health": p0_summary,
        "calibration": calibrations,
        "test": test,
        "positive": positive,
        "gates": gates,
        "excluded_manifest_indices": [EXCLUDED_MANIFEST_INDEX],
        "contains_raw_text": False,
        "completed_at": utc_now(),
    }


class Heartbeat:
    def __init__(self, path, started_at, completed_p0, completed_certificates):
        self.path = Path(path)
        self.started_at = started_at
        self.completed_p0 = completed_p0
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
            "completed_p0_documents": self.completed_p0,
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


def checkpoint_manifest_sha(paths):
    lines = [
        f"{sha256_file(path)}  {path.name}\n"
        for path in sorted(paths)
    ]
    return sha256_bytes("".join(lines).encode("ascii"))


def write_results_manifest(args, output_dir):
    output_dir = Path(output_dir)
    paths = [
        args.protocol,
        args.parent_v3_protocol,
        args.parent_v6_protocol,
        args.identity_manifest,
        args.workload_manifest,
        args.core_source,
        args.core,
        Path(__file__).resolve(),
        output_dir / "p0_summary.json",
        output_dir / "final_summary.json",
        *sorted((output_dir / "p0").glob("*.json")),
        *sorted((output_dir / "certificates").glob("*.json")),
    ]
    lines = []
    for path in paths:
        if not path.exists():
            continue
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
    parser.add_argument("--parent-v3-protocol", type=Path, required=True)
    parser.add_argument("--parent-v6-protocol", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-chunk-size", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def run(args):
    import nltk

    if args.batch_size != 64:
        raise ValueError("the frozen WP2-v10 batch size is 64")
    if args.text_chunk_size < args.batch_size:
        raise ValueError("text chunk size must be at least the batch size")
    if args.min_free_gib < 10:
        raise ValueError("the frozen minimum free-space gate is 10 GiB")
    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(args.nltk_data))
    if nltk.__version__ != DESIGN_SYSTEMS["semstamp"]["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.find("tokenizers/punkt/english.pickle")

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
        "semstamp",
    )
    tokenizer = load_tokenizer(args.model_dir, "semstamp")
    preflight = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "pass",
        "created_at": utc_now(),
        "data_revision": revision,
        "identity_manifest_sha256": sha256_file(args.identity_manifest),
        "workload_manifest_sha256": sha256_file(args.workload_manifest),
        "parent_v3_protocol_sha256": sha256_file(
            args.parent_v3_protocol
        ),
        "parent_v6_protocol_sha256": sha256_file(
            args.parent_v6_protocol
        ),
        "core_source_sha256": sha256_file(args.core_source),
        "core_executable_sha256": sha256_file(args.core),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "official_identity": official_identity,
        "row_count": len(identity_rows),
        "formal_row_count": len(FORMAL_INDICES),
        "excluded_manifest_indices": [EXCLUDED_MANIFEST_INDEX],
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

    scratch = args.work_dir / "wp2_v10_semstamp_active"
    clean_scratch(scratch)
    p0_dir = args.output_dir / "p0"
    certificate_dir = args.output_dir / "certificates"
    started_at = utc_now()

    heartbeat = Heartbeat(
        args.output_dir / "heartbeat.json",
        started_at,
        0,
        0,
    )
    heartbeat(force=True, stage="loading_model")
    load_started = time.perf_counter()
    model, relation = load_semstamp(
        args.root,
        args.model_dir,
        args.batch_size,
    )
    relation = validate_relation(relation)
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
        "relation_sha256": sha256_bytes(relation),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "platform": platform.platform(),
        "environment": package_versions(),
    })

    p0_rows = collect_existing(
        p0_dir,
        FORMAL_INDICES,
        lambda row, index: validate_p0_checkpoint(
            row,
            index,
            identity_rows[index],
            relation,
        ),
    )
    heartbeat.completed_p0 = len(p0_rows)
    heartbeat(force=True, stage="p0_detector_health")
    for index in FORMAL_INDICES:
        if index in p0_rows:
            continue
        heartbeat(
            force=True,
            stage="p0_detector_health",
            manifest_index=index,
        )
        output = process_p0_document(
            index,
            identities[index],
            identity_rows[index],
            model,
            relation,
        )
        path = p0_dir / f"{index:04d}.json"
        atomic_write_json(path, output)
        p0_rows[index] = validate_p0_checkpoint(
            load_json(path),
            index,
            identity_rows[index],
            relation,
        )
        heartbeat.completed_p0 = len(p0_rows)
        heartbeat(
            force=True,
            stage="p0_detector_health",
            manifest_index=index,
        )
        print(json.dumps({
            "stage": "p0_detector_health",
            "completed": len(p0_rows),
            "total": len(FORMAL_INDICES),
            "manifest_index": index,
        }, sort_keys=True), flush=True)

    ordered_p0 = [p0_rows[index] for index in FORMAL_INDICES]
    p0_summary = summarize_p0(ordered_p0)
    p0_summary["checkpoint_manifest_sha256"] = checkpoint_manifest_sha(
        [p0_dir / f"{index:04d}.json" for index in FORMAL_INDICES]
    )
    atomic_write_json(args.output_dir / "p0_summary.json", p0_summary)

    if p0_summary["status"] != "semstamp_p0_pass":
        final_summary = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "experiment": "certified_resegmentation_wp2_v10_semstamp1",
            "status": "semstamp_p0_stop",
            "scope": "semstamp_only_simmark_v9_stopped",
            "p0_health": p0_summary,
            "certificate_documents_completed": 0,
            "excluded_manifest_indices": [EXCLUDED_MANIFEST_INDEX],
            "contains_raw_text": False,
            "completed_at": utc_now(),
        }
        atomic_write_json(
            args.output_dir / "final_summary.json",
            final_summary,
        )
        write_results_manifest(args, args.output_dir)
        heartbeat(
            force=True,
            stage="complete",
            status="semstamp_p0_stop",
        )
        print(json.dumps({
            "stage": "complete",
            "status": "semstamp_p0_stop",
            "gates": p0_summary["gates"],
        }, sort_keys=True), flush=True)
        return

    certificate_rows = collect_existing(
        certificate_dir,
        FORMAL_INDICES,
        lambda row, index: validate_certificate_checkpoint(
            row,
            index,
            identity_rows[index],
            p0_rows[index],
        ),
    )
    heartbeat.completed_certificates = len(certificate_rows)
    heartbeat(force=True, stage="exact_certificates")
    for index in FORMAL_INDICES:
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
                p0_rows[index],
                model,
                tokenizer,
                relation,
                scratch,
                heartbeat,
            )
            path = certificate_dir / f"{index:04d}.json"
            atomic_write_json(path, output)
            certificate_rows[index] = validate_certificate_checkpoint(
                load_json(path),
                index,
                identity_rows[index],
                p0_rows[index],
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
            "total": len(FORMAL_INDICES),
            "manifest_index": index,
            "elapsed_seconds": output["elapsed_seconds"],
        }, sort_keys=True), flush=True)

    ordered_certificates = [
        certificate_rows[index] for index in FORMAL_INDICES
    ]
    summary = summarize_results(ordered_certificates, p0_summary)
    atomic_write_json(args.output_dir / "final_summary.json", summary)
    write_results_manifest(args, args.output_dir)
    heartbeat.completed_certificates = len(certificate_rows)
    heartbeat(
        force=True,
        stage="complete",
        status=summary["status"],
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
