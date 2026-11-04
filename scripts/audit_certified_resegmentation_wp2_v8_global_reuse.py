"""Audit exact cross-document SimMark feature reuse without model access."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import time

from audit_certified_resegmentation_wp2_v2_design import (
    NATIVE_TOKEN_LIMIT,
    SYSTEMS,
    frozen_identities,
    load_tokenizer,
    official_tokenizer_text,
    tokenizer_manifest,
)
from audit_certified_resegmentation_wp2_v3_system import PRIMARY_CANDIDATE
from audit_certified_resegmentation_wp2_v6_workload import (
    batched_token_counts,
    dump_all_allowed_topology,
    load_json,
    load_jsonl,
    verify_manifest_rows,
)
from certified_resegmentation_wp2_adapter import hash_raw_value
from certified_resegmentation_wp2_v3 import (
    SparsePairwiseProblem,
    canonicalize_structured,
)
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)


PROTOCOL_SHA256 = (
    "296aa9c3a7c955ecf46c76f76e001b7611976fcb92e5666404d30a09d0134842"
)
V3_AUDIT_SHA256 = (
    "c2a1374e2f7bb1caf809b6cf2497c9bb18fe6c1d17219399e0e09c7dc3df778f"
)
V3_ROWS_SHA256 = (
    "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
)
V6_ROWS_SHA256 = (
    "04f17eb7c73f6ca9a8e379615e1062eaee37a63a3794ae39ff81b835929de6eb"
)
V6_RESULT_SHA256 = (
    "a027577ae2b8e0adfb64fcddf71965be31a1927b705dfebf8ef6a8a4cd075d15"
)
CORE_SHA256 = (
    "d724a038f72d766b477cb1bd95c8318bc0032c3718e067005bd05921944847fc"
)
TOKENIZER_SHA256 = {
    "special_tokens_map.json": (
        "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"
    ),
    "spiece.model": (
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"
    ),
    "tokenizer.json": (
        "72fb264ef4324ec23bc4e8b94167134eaf031b2048fd51a2c2f5a329c7695847"
    ),
    "tokenizer_config.json": (
        "c31a902ec8f6335e196f500d15c9f974e5c69715b7d375642555c32113b3fdf3"
    ),
}
EXPECTED_ROWS = 2_000
EXPECTED_FEATURE_TEXTS = 36_090_669
EXPECTED_INPUT_TOKENS = 2_438_253_276
V6_FEATURE_SECONDS = 237_283.27002381114
V6_TOTAL_SECONDS = 258_525.67780724863
FIXED_SECONDS = V6_TOTAL_SECONDS - V6_FEATURE_SECONDS
TOKEN_REDUCTION_GATE = 0.361279907338794
GPU_HOUR_GATE = 48.0
CACHE_GATE_BYTES = 50 * 1024 ** 3
WORK_DISK_GATE_BYTES = 15 * 1024 ** 3
RSS_GATE_BYTES = 16 * 1024 ** 3
WALL_GATE_SECONDS = 4 * 3600
MAX_SIDECAR_BYTES = 2_962_165_858
CACHE_RECORD_BYTES = 3_200
SHARD_COUNT = 256
SYSTEM_ID = "simmark"
DOC_MAGIC = b"WP2V8R1\x00"
DOC_HEADER = struct.Struct("<8sI32sQQd")
DOC_RECORD = struct.Struct("<32s64sH")
SHARD_RECORD = struct.Struct("<32s64sHd")
_WORKER = {}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha(path, expected, label):
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {observed}"
        )


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def collect_document_records(
    document,
    tokenizer,
    tokenizer_batch_size,
    core,
    work_dir,
):
    started = time.perf_counter()
    n = document.atom_count
    base = n + 1
    baseline, ordered_codes = dump_all_allowed_topology(
        core,
        document,
        work_dir,
    )
    if baseline["unique_span_checks"] != len(ordered_codes):
        raise AssertionError("all-allowed span count mismatch")

    legality = {}
    token_count_by_code = {}
    identity_by_code = {}
    for offset in range(0, len(ordered_codes), tokenizer_batch_size):
        codes = ordered_codes[offset:offset + tokenizer_batch_size]
        spans = [divmod(code, base) for code in codes]
        rendered = [
            document.preprocess_unit(SYSTEM_ID, start, end)
            for start, end in spans
        ]
        tokenizer_texts = [
            official_tokenizer_text(
                SYSTEM_ID,
                text,
                int(span[0] > 0),
            )
            for text, span in zip(rendered, spans)
        ]
        counts = batched_token_counts(tokenizer, tokenizer_texts)
        for code, text, token_count in zip(codes, rendered, counts):
            allowed = token_count <= NATIVE_TOKEN_LIMIT
            legality[code] = allowed
            if allowed:
                encoded = text.encode("utf-8")
                token_count_by_code[code] = token_count
                identity_by_code[code] = (
                    hashlib.sha256(encoded).digest(),
                    hashlib.sha512(encoded).digest(),
                )

    actual_span_codes = set()
    pair_calls = 0

    def span_allowed(start, end):
        code = start * base + end
        actual_span_codes.add(code)
        try:
            return legality[code]
        except KeyError as error:
            raise AssertionError(
                "actual query escaped all-allowed superset"
            ) from error

    def count_pair(*coordinates):
        nonlocal pair_calls
        if len(coordinates) != 3:
            raise AssertionError("pair scorer arity mismatch")
        pair_calls += 1
        return 0.0

    if all(legality.values()):
        result = baseline
        actual_span_codes = set(ordered_codes)
        pair_calls = baseline["unique_pair_scores"]
    else:
        result = exact_grouped_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=n,
                original_boundaries=document.original_boundaries,
                max_boundary_moves=min(
                    3,
                    len(document.original_boundaries),
                ),
                span_allowed=span_allowed,
                pair_scorer=count_pair,
            )
        )
    result_span_count = (
        result["unique_span_checks"]
        if isinstance(result, dict)
        else result.unique_span_checks
    )
    result_pair_count = (
        result["unique_pair_scores"]
        if isinstance(result, dict)
        else result.unique_pair_scores
    )
    if result_span_count != len(actual_span_codes):
        raise AssertionError("actual span count mismatch")
    if result_pair_count != pair_calls:
        raise AssertionError("actual pair count mismatch")

    feature_by_primary = {}
    legal_span_count = 0
    for code in sorted(actual_span_codes):
        if not legality[code]:
            continue
        legal_span_count += 1
        primary, secondary = identity_by_code[code]
        token_count = token_count_by_code[code]
        previous = feature_by_primary.setdefault(
            primary,
            (secondary, token_count),
        )
        if previous != (secondary, token_count):
            raise AssertionError(
                "within-document SHA256 collision or token mismatch"
            )
    records = [
        (primary, secondary, token_count)
        for primary, (secondary, token_count)
        in sorted(feature_by_primary.items())
    ]
    metrics = {
        "baseline_unique_spans": baseline["unique_span_checks"],
        "baseline_unique_pairs": baseline["unique_pair_scores"],
        "unique_query_spans": len(actual_span_codes),
        "legal_query_spans": legal_span_count,
        "actual_unique_pairs": pair_calls,
        "deduplicated_feature_texts": len(records),
        "deduplicated_input_tokens": sum(
            record[2] for record in records
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return records, metrics


def write_document_records(
    path,
    manifest_index,
    raw_text_sha256,
    feature_rate,
    records,
):
    if path.exists():
        raise ValueError(f"record file already exists: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    token_sum = sum(record[2] for record in records)
    with temporary.open("xb") as handle:
        handle.write(DOC_HEADER.pack(
            DOC_MAGIC,
            manifest_index,
            bytes.fromhex(raw_text_sha256),
            len(records),
            token_sum,
            feature_rate,
        ))
        for record in records:
            handle.write(DOC_RECORD.pack(*record))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return token_sum


def read_document_header(path):
    with path.open("rb") as handle:
        payload = handle.read(DOC_HEADER.size)
    if len(payload) != DOC_HEADER.size:
        raise ValueError(f"truncated document header: {path}")
    magic, index, raw_sha, count, tokens, rate = DOC_HEADER.unpack(payload)
    if magic != DOC_MAGIC:
        raise ValueError(f"document record magic mismatch: {path}")
    expected_size = DOC_HEADER.size + count * DOC_RECORD.size
    if path.stat().st_size != expected_size:
        raise ValueError(f"document record size mismatch: {path}")
    return {
        "manifest_index": index,
        "raw_text_sha256": raw_sha.hex(),
        "record_count": count,
        "token_sum": tokens,
        "feature_rate": rate,
    }


def initialize_worker(
    model_dir,
    nltk_data,
    tokenizer_batch_size,
    core,
    work_dir,
    record_dir,
):
    import nltk

    os.environ["NLTK_DATA"] = str(nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(nltk_data))
    _WORKER.update({
        "tokenizer": load_tokenizer(model_dir, SYSTEM_ID),
        "tokenizer_batch_size": tokenizer_batch_size,
        "core": core,
        "work_dir": work_dir,
        "record_dir": record_dir,
    })


def extract_row_task(task):
    index, source, raw_value, v6_row = task
    started = time.perf_counter()
    if not source["candidate_eligible"][PRIMARY_CANDIDATE]:
        raise ValueError(f"unexpected ineligible SimMark row: {index}")
    identity_fields = (
        "manifest_index",
        "cohort",
        "role",
        "row_index",
        "raw_text_sha256",
        "structural_tape_sha256",
        "atom_count",
        "unit_count",
    )
    expected_identity = {
        "manifest_index": index,
        "cohort": source["cohort"],
        "role": source["role"],
        "row_index": source["row_index"],
        "raw_text_sha256": source["raw_text_sha256"],
        "structural_tape_sha256": source["structural_tape_sha256"],
        "atom_count": source["atom_count"],
        "unit_count": source["unit_count"],
    }
    for key in identity_fields:
        if v6_row.get(key) != expected_identity[key]:
            raise ValueError(f"v6 row identity mismatch at {index}: {key}")
    if not v6_row.get("eligible"):
        raise ValueError(f"v6 row unexpectedly ineligible: {index}")
    if hash_raw_value(raw_value) != source["raw_text_sha256"]:
        raise ValueError(f"raw-value SHA256 mismatch at row {index}")

    from nltk.tokenize import sent_tokenize

    document = canonicalize_structured(
        raw_value,
        sentence_splitter=(
            sent_tokenize if isinstance(raw_value, str) else None
        ),
    )
    if (
        document.structural_tape_sha256
        != source["structural_tape_sha256"]
        or document.atom_count != source["atom_count"]
        or document.unit_count != source["unit_count"]
    ):
        raise ValueError(f"structural identity mismatch at row {index}")
    records, metrics = collect_document_records(
        document,
        _WORKER["tokenizer"],
        _WORKER["tokenizer_batch_size"],
        _WORKER["core"],
        _WORKER["work_dir"],
    )
    frozen_topology = v6_row["topology"]
    for key in (
        "baseline_unique_spans",
        "baseline_unique_pairs",
        "unique_query_spans",
        "legal_query_spans",
        "actual_unique_pairs",
        "deduplicated_feature_texts",
        "deduplicated_input_tokens",
    ):
        if metrics[key] != frozen_topology[key]:
            raise ValueError(
                f"v6 topology mismatch at row {index}: {key}"
            )
    feature_rate = v6_row["estimate"]["feature_rate"]["rate"]
    record_path = _WORKER["record_dir"] / f"{index:04d}.bin"
    token_sum = write_document_records(
        record_path,
        index,
        source["raw_text_sha256"],
        feature_rate,
        records,
    )
    if token_sum != metrics["deduplicated_input_tokens"]:
        raise AssertionError("record token sum mismatch")
    return {
        **expected_identity,
        **metrics,
        "feature_rate": feature_rate,
        "feature_seconds": token_sum * feature_rate,
        "record_path": str(record_path),
        "record_bytes": record_path.stat().st_size,
        "record_sha256": sha256_file(record_path),
        "worker_peak_rss_bytes": peak_rss_bytes(),
        "total_elapsed_seconds": time.perf_counter() - started,
        "contains_raw_text": False,
    }


def validate_completed_rows(rows, manifest_rows, v6_rows):
    if len(rows) > len(manifest_rows):
        raise ValueError("checkpoint has too many rows")
    for index, row in enumerate(rows):
        source = manifest_rows[index]
        v6_row = v6_rows[index]
        if (
            row.get("manifest_index") != index
            or row.get("raw_text_sha256") != source["raw_text_sha256"]
            or row.get("record_count")
            not in (None, v6_row["topology"]["deduplicated_feature_texts"])
            or row.get("deduplicated_feature_texts")
            != v6_row["topology"]["deduplicated_feature_texts"]
            or row.get("deduplicated_input_tokens")
            != v6_row["topology"]["deduplicated_input_tokens"]
        ):
            raise ValueError(f"checkpoint mismatch at row {index}")
        record_path = Path(row["record_path"])
        header = read_document_header(record_path)
        if (
            header["manifest_index"] != index
            or header["raw_text_sha256"] != source["raw_text_sha256"]
            or header["record_count"]
            != row["deduplicated_feature_texts"]
            or header["token_sum"] != row["deduplicated_input_tokens"]
            or header["feature_rate"] != row["feature_rate"]
            or sha256_file(record_path) != row["record_sha256"]
        ):
            raise ValueError(f"record checkpoint mismatch at row {index}")


def extract_rows(args, manifest_rows, identities, v6_rows):
    completed = []
    if args.rows_output.exists():
        if not args.resume:
            raise ValueError(
                "rows output exists; pass --resume to validate it"
            )
        completed = load_jsonl(args.rows_output)
        validate_completed_rows(completed, manifest_rows, v6_rows)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    tasks = (
        (
            index,
            manifest_rows[index],
            identities[index]["raw_value"],
            v6_rows[index],
        )
        for index in range(len(completed), len(manifest_rows))
    )
    worker_args = (
        args.model_dir,
        args.nltk_data,
        args.tokenizer_batch_size,
        args.core,
        args.work_dir,
        args.record_dir,
    )
    if args.workers == 1:
        initialize_worker(*worker_args)
        results = map(extract_row_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=worker_args,
        )
        results = executor.map(extract_row_task, tasks, chunksize=1)
    try:
        with args.rows_output.open(mode, encoding="ascii") as handle:
            for output in results:
                handle.write(json.dumps(
                    output,
                    sort_keys=True,
                    ensure_ascii=True,
                ) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed.append(output)
                index = output["manifest_index"]
                if (
                    (index + 1) % args.progress_every == 0
                    or index + 1 == len(manifest_rows)
                    or output["total_elapsed_seconds"] >= 10
                ):
                    print(json.dumps({
                        "stage": "extract",
                        "completed_rows": index + 1,
                        "total_rows": len(manifest_rows),
                        "feature_texts": output[
                            "deduplicated_feature_texts"
                        ],
                        "input_tokens": output[
                            "deduplicated_input_tokens"
                        ],
                        "elapsed_seconds": output[
                            "total_elapsed_seconds"
                        ],
                    }, sort_keys=True), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return completed


def write_records_manifest(path, rows):
    payload = "".join(
        f"{row['record_sha256']}  {row['record_path']}\n"
        for row in rows
    )
    path.write_text(payload, encoding="ascii")


def partition_records(rows, shard_dir):
    shard_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        shard_dir / f"shard_{index:03d}.bin"
        for index in range(SHARD_COUNT)
    ]
    handles = [
        path.open("xb", buffering=1024 * 1024)
        for path in paths
    ]
    total_records = 0
    try:
        for row in rows:
            path = Path(row["record_path"])
            header = read_document_header(path)
            rate = header["feature_rate"]
            buffers = {}
            with path.open("rb") as handle:
                handle.seek(DOC_HEADER.size)
                for _ in range(header["record_count"]):
                    payload = handle.read(DOC_RECORD.size)
                    if len(payload) != DOC_RECORD.size:
                        raise ValueError(f"truncated record file: {path}")
                    primary, secondary, token_count = DOC_RECORD.unpack(
                        payload
                    )
                    buffers.setdefault(primary[0], bytearray()).extend(
                        SHARD_RECORD.pack(
                            primary,
                            secondary,
                            token_count,
                            rate,
                        )
                    )
                    total_records += 1
                if handle.read(1):
                    raise ValueError(f"trailing document bytes: {path}")
            for shard_index, payload in buffers.items():
                handles[shard_index].write(payload)
    finally:
        for handle in handles:
            handle.close()
    return paths, total_records


def summarize_sorted_records(records):
    records.sort(key=lambda row: row[0])
    metrics = {
        "record_count": len(records),
        "global_unique_feature_texts": 0,
        "global_unique_input_tokens": 0,
        "repeated_unique_feature_texts": 0,
        "duplicate_occurrences_removed": 0,
        "cached_feature_seconds": 0.0,
        "reconstructed_feature_seconds": 0.0,
    }
    cached_costs = []
    original_costs = []
    index = 0
    while index < len(records):
        primary, secondary, token_count, rate = records[index]
        next_index = index + 1
        rates = [rate]
        while (
            next_index < len(records)
            and records[next_index][0] == primary
        ):
            _, candidate_secondary, candidate_tokens, candidate_rate = (
                records[next_index]
            )
            if (
                candidate_secondary != secondary
                or candidate_tokens != token_count
            ):
                raise ValueError(
                    "global SHA256 collision or token-count mismatch"
                )
            rates.append(candidate_rate)
            next_index += 1
        occurrences = next_index - index
        metrics["global_unique_feature_texts"] += 1
        metrics["global_unique_input_tokens"] += token_count
        if occurrences > 1:
            metrics["repeated_unique_feature_texts"] += 1
            metrics["duplicate_occurrences_removed"] += occurrences - 1
        cached_costs.append(token_count * max(rates))
        original_costs.extend(token_count * value for value in rates)
        index = next_index
    metrics["cached_feature_seconds"] = math.fsum(cached_costs)
    metrics["reconstructed_feature_seconds"] = math.fsum(original_costs)
    return metrics


def aggregate_shards(shard_paths):
    partials = []
    for path in shard_paths:
        payload = path.read_bytes()
        if len(payload) % SHARD_RECORD.size:
            raise ValueError(f"truncated shard file: {path}")
        records = [
            SHARD_RECORD.unpack_from(payload, offset)
            for offset in range(0, len(payload), SHARD_RECORD.size)
        ]
        partials.append(summarize_sorted_records(records))
    integer_fields = (
        "record_count",
        "global_unique_feature_texts",
        "global_unique_input_tokens",
        "repeated_unique_feature_texts",
        "duplicate_occurrences_removed",
    )
    output = {
        key: sum(row[key] for row in partials)
        for key in integer_fields
    }
    for key in (
        "cached_feature_seconds",
        "reconstructed_feature_seconds",
    ):
        output[key] = math.fsum(row[key] for row in partials)
    return output


def canonical_metrics_sha256(metrics):
    payload = json.dumps(
        metrics,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def directory_bytes(path):
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def build_output(args, rows, metrics, replay_sha, elapsed_seconds):
    total_feature_texts = sum(
        row["deduplicated_feature_texts"] for row in rows
    )
    total_input_tokens = sum(
        row["deduplicated_input_tokens"] for row in rows
    )
    row_feature_seconds = math.fsum(
        row["feature_seconds"] for row in rows
    )
    token_reduction = (
        1
        - metrics["global_unique_input_tokens"] / total_input_tokens
    )
    cached_total_seconds = (
        FIXED_SECONDS + metrics["cached_feature_seconds"]
    )
    cache_bytes = (
        metrics["repeated_unique_feature_texts"]
        * CACHE_RECORD_BYTES
    )
    cache_plus_sidecar_bytes = cache_bytes + MAX_SIDECAR_BYTES
    work_disk_bytes = (
        directory_bytes(args.record_dir)
        + directory_bytes(args.shard_dir)
    )
    peak_rss = max(
        peak_rss_bytes(),
        max(row["worker_peak_rss_bytes"] for row in rows),
    )
    gates = {
        "all_2000_rows_completed": len(rows) == EXPECTED_ROWS,
        "feature_text_count_matches_v6": (
            total_feature_texts == EXPECTED_FEATURE_TEXTS
            and metrics["record_count"] == EXPECTED_FEATURE_TEXTS
        ),
        "input_token_count_matches_v6": (
            total_input_tokens == EXPECTED_INPUT_TOKENS
        ),
        "row_feature_seconds_matches_v6": math.isclose(
            row_feature_seconds,
            V6_FEATURE_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "aggregate_feature_seconds_matches_rows": math.isclose(
            metrics["reconstructed_feature_seconds"],
            row_feature_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "aggregate_replays_identical": replay_sha[0] == replay_sha[1],
        "token_reduction_ge_required": (
            token_reduction >= TOKEN_REDUCTION_GATE
        ),
        "cached_end_to_end_hours_le_48": (
            cached_total_seconds / 3600 <= GPU_HOUR_GATE
        ),
        "cache_plus_max_sidecar_le_50_gib": (
            cache_plus_sidecar_bytes <= CACHE_GATE_BYTES
        ),
        "work_disk_le_15_gib": (
            work_disk_bytes <= WORK_DISK_GATE_BYTES
        ),
        "peak_rss_le_16_gib": peak_rss <= RSS_GATE_BYTES,
        "wall_seconds_le_4_hours": elapsed_seconds <= WALL_GATE_SECONDS,
    }
    return {
        "schema_version": 1,
        "created_on": "2026-09-23",
        "audit_mode": "wp2_v8_simmark_global_reuse_opportunity",
        "status": (
            "global-reuse-opportunity-ready"
            if all(gates.values())
            else "stop"
        ),
        "inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "v3_audit_sha256": V3_AUDIT_SHA256,
            "v3_rows_sha256": V3_ROWS_SHA256,
            "v6_rows_sha256": V6_ROWS_SHA256,
            "v6_result_sha256": V6_RESULT_SHA256,
            "core_sha256": CORE_SHA256,
            "tokenizer_files": TOKENIZER_SHA256,
            "rows_output_sha256": sha256_file(args.rows_output),
            "records_manifest_sha256": sha256_file(
                args.records_manifest
            ),
        },
        "parameters": {
            "system": SYSTEM_ID,
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "max_boundary_moves": 3,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "workers": args.workers,
            "sha256_shards": SHARD_COUNT,
            "cache_record_bytes": CACHE_RECORD_BYTES,
            "cache_gate_bytes": CACHE_GATE_BYTES,
            "max_sidecar_bytes": MAX_SIDECAR_BYTES,
            "gpu_hour_gate": GPU_HOUR_GATE,
            "token_reduction_gate": TOKEN_REDUCTION_GATE,
            "work_disk_gate_bytes": WORK_DISK_GATE_BYTES,
            "rss_gate_bytes": RSS_GATE_BYTES,
            "wall_gate_seconds": WALL_GATE_SECONDS,
        },
        "counts": {
            "rows": len(rows),
            "document_deduplicated_feature_texts": total_feature_texts,
            "document_deduplicated_input_tokens": total_input_tokens,
            "global_unique_feature_texts": metrics[
                "global_unique_feature_texts"
            ],
            "global_unique_input_tokens": metrics[
                "global_unique_input_tokens"
            ],
            "repeated_unique_feature_texts": metrics[
                "repeated_unique_feature_texts"
            ],
            "duplicate_occurrences_removed": metrics[
                "duplicate_occurrences_removed"
            ],
        },
        "runtime_upper_bound": {
            "v6_feature_seconds": V6_FEATURE_SECONDS,
            "reconstructed_feature_seconds": metrics[
                "reconstructed_feature_seconds"
            ],
            "cached_feature_seconds": metrics[
                "cached_feature_seconds"
            ],
            "fixed_seconds": FIXED_SECONDS,
            "cached_end_to_end_seconds": cached_total_seconds,
            "cached_end_to_end_gpu_hours": (
                cached_total_seconds / 3600
            ),
            "feature_speedup": (
                V6_FEATURE_SECONDS
                / metrics["cached_feature_seconds"]
            ),
            "token_workload_reduction": token_reduction,
        },
        "cache": {
            "repeated_identity_cache_bytes": cache_bytes,
            "cache_plus_max_sidecar_bytes": cache_plus_sidecar_bytes,
        },
        "aggregation_replays": [
            {"canonical_metrics_sha256": value}
            for value in replay_sha
        ],
        "gates": gates,
        "resources": {
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_bytes": peak_rss,
            "work_disk_bytes": work_disk_bytes,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "execution_attestation": {
            "gpu_model_loads": 0,
            "model_forward_calls": 0,
            "detector_document_score_calls": 0,
            "simmark_gamma_reads_or_fits": 0,
            "calibration_model_feature_reads": 0,
            "test_model_feature_reads": 0,
            "positive_model_feature_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }


def main():
    import nltk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v3-audit", type=Path, required=True)
    parser.add_argument("--v3-rows", type=Path, required=True)
    parser.add_argument("--v6-rows", type=Path, required=True)
    parser.add_argument("--v6-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--records-manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(
        args.workers,
        args.tokenizer_batch_size,
        args.progress_every,
    ) < 1:
        raise ValueError("worker, batch, and progress values must be positive")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    if args.shard_dir.exists():
        raise ValueError(f"shard directory already exists: {args.shard_dir}")

    for path, expected, label in (
        (args.protocol, PROTOCOL_SHA256, "v8 protocol"),
        (args.v3_audit, V3_AUDIT_SHA256, "v3 audit"),
        (args.v3_rows, V3_ROWS_SHA256, "v3 rows"),
        (args.v6_rows, V6_ROWS_SHA256, "v6 workload rows"),
        (args.v6_result, V6_RESULT_SHA256, "v6 workload result"),
        (args.core, CORE_SHA256, "v6 core"),
    ):
        verify_sha(path, expected, label)
    observed_tokenizer = tokenizer_manifest(
        args.model_dir,
        SYSTEM_ID,
    )
    if observed_tokenizer != TOKENIZER_SHA256:
        raise ValueError("tokenizer manifest mismatch")

    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(args.nltk_data))
    if nltk.__version__ != SYSTEMS[SYSTEM_ID]["nltk_version"]:
        raise ValueError("NLTK version mismatch")

    started = time.perf_counter()
    v3_audit = load_json(args.v3_audit)
    if (
        v3_audit.get("audit_mode") != "wp2_v3_structural_readiness"
        or v3_audit.get("system_id") != SYSTEM_ID
        or v3_audit.get("primary_candidate") != PRIMARY_CANDIDATE
    ):
        raise ValueError("v3 audit identity mismatch")
    identities, _, revision = frozen_identities(args.root, SYSTEM_ID)
    if revision != v3_audit["data_revision"]:
        raise ValueError("v3 data revision mismatch")
    manifest_rows = verify_manifest_rows(
        v3_audit,
        args.v3_rows,
        identities,
    )
    v6_rows = load_jsonl(args.v6_rows)
    if len(v6_rows) != EXPECTED_ROWS:
        raise ValueError("v6 workload row count mismatch")
    v6_result = load_json(args.v6_result)
    if (
        v6_result.get("audit_mode")
        != "wp2_v6_score_blind_full_workload"
        or v6_result.get("system") != SYSTEM_ID
        or v6_result["counts"]["total_deduplicated_feature_texts"]
        != EXPECTED_FEATURE_TEXTS
        or v6_result["counts"]["total_deduplicated_input_tokens"]
        != EXPECTED_INPUT_TOKENS
        or v6_result["runtime_upper_bound"]["feature_seconds"]
        != V6_FEATURE_SECONDS
        or v6_result["runtime_upper_bound"][
            "conservative_end_to_end_seconds"
        ] != V6_TOTAL_SECONDS
    ):
        raise ValueError("v6 workload aggregate mismatch")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = extract_rows(args, manifest_rows, identities, v6_rows)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("incomplete row extraction")
    write_records_manifest(args.records_manifest, rows)
    shard_paths, partition_count = partition_records(
        rows,
        args.shard_dir,
    )
    if partition_count != EXPECTED_FEATURE_TEXTS:
        raise ValueError("partition record count mismatch")

    first = aggregate_shards(shard_paths)
    second = aggregate_shards(shard_paths)
    replay_sha = (
        canonical_metrics_sha256(first),
        canonical_metrics_sha256(second),
    )
    if first != second:
        raise ValueError("aggregation replay mismatch")
    elapsed_seconds = time.perf_counter() - started
    output = build_output(
        args,
        rows,
        first,
        replay_sha,
        elapsed_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "status": output["status"],
        "global_unique_feature_texts": output["counts"][
            "global_unique_feature_texts"
        ],
        "token_workload_reduction": output["runtime_upper_bound"][
            "token_workload_reduction"
        ],
        "cached_end_to_end_gpu_hours": output["runtime_upper_bound"][
            "cached_end_to_end_gpu_hours"
        ],
        "gates": output["gates"],
    }, sort_keys=True))
    if output["status"] != "global-reuse-opportunity-ready":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
