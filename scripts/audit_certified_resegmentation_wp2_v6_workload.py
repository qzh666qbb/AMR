"""Audit the full score-blind WP2-v6 workload and runtime upper bound."""

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import tempfile
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
from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
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
    "ddab1e0f8d3714dda22ae383395ea91e14ff10ac55d59e63b0708806d533a350"
)
SELECTION_SHA256 = (
    "413bdd37780b9c3e9e41807fcf80b2e8843951fb26af4570d81312544649e399"
)
EXPECTED_SLOTS = {
    "nearest_32",
    "nearest_64",
    "nearest_96",
    "nearest_128",
    "nearest_192",
    "nearest_256",
    "nearest_300",
    "nearest_384",
    "nearest_512",
    "nearest_768",
    "nearest_1024",
    "maximum_remaining",
}
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
CACHE_GATE_BYTES = 50 * 1024 ** 3
GPU_HOUR_GATE = 48.0
SIMMARK_DIMENSION = 768
EXPECTED_MANIFEST_ROWS = 2_000
EXPECTED_ELIGIBLE_ROWS = {
    "semstamp": 1_999,
    "simmark": 2_000,
}
FEATURE_MAGIC = b"WP2V6F1\x00"
STRUCTURAL_DUMP_MODE = 5
_WORKER = {}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def sidecar_bytes(system_id, span_count, legal_span_count):
    header = 8 + 4 + 4 + 8 + 4
    if system_id == "semstamp":
        return header + 64 + 9 * span_count + legal_span_count
    return (
        header
        + 9 * span_count
        + 4 * SIMMARK_DIMENSION * legal_span_count
    )


def batched_token_counts(tokenizer, texts):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(texts):
        raise ValueError("tokenizer batch row count mismatch")
    if input_ids and not isinstance(input_ids[0], list):
        if len(texts) != 1:
            raise ValueError("tokenizer flattened a multi-row batch")
        input_ids = [input_ids]
    return [len(row) for row in input_ids]


def baseline_payload(document):
    fields = [
        document.atom_count,
        len(document.original_boundaries),
        min(3, len(document.original_boundaries)),
        3,
        1,
        len(document.original_boundaries),
        0,
        0,
        *document.original_boundaries,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def dump_all_allowed_topology(core, document, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=work_dir,
        prefix="v6_workload_",
        suffix=".bin",
        delete=False,
    ) as handle:
        path = Path(handle.name)
    path.unlink()
    try:
        completed = subprocess.run(
            [
                str(core),
                "--dump-synthetic-sidecar",
                str(path),
                str(STRUCTURAL_DUMP_MODE),
            ],
            input=baseline_payload(document),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "all-allowed topology dump failed: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        baseline = parse_core_output(completed.stdout, 1)[0]
        with path.open("rb") as handle:
            if handle.read(8) != FEATURE_MAGIC:
                raise ValueError("topology dump magic mismatch")
            mode, n, count, dimension = struct.unpack(
                "<IIQI",
                handle.read(20),
            )
            if (
                mode != STRUCTURAL_DUMP_MODE
                or n != document.atom_count
                or dimension != 1
            ):
                raise ValueError("topology dump identity mismatch")
            relation = handle.read(64)
            if len(relation) != 64:
                raise ValueError("truncated topology relation")
            codes = []
            base = document.atom_count + 1
            for _ in range(count):
                record = handle.read(10)
                if len(record) != 10:
                    raise ValueError("truncated topology span")
                start, end, allowed, region = struct.unpack(
                    "<IIBB",
                    record,
                )
                if (
                    not (start < end <= document.atom_count)
                    or allowed != 1
                    or region > 7
                ):
                    raise ValueError("invalid topology span")
                codes.append(start * base + end)
            if handle.read(1):
                raise ValueError("trailing topology dump bytes")
        if len(codes) != count or len(set(codes)) != count:
            raise ValueError("duplicate topology dump span")
        if baseline["unique_span_checks"] != count:
            raise ValueError("topology dump/core span count mismatch")
        return baseline, codes
    finally:
        path.unlink(missing_ok=True)


def audit_document(
    document,
    system_id,
    tokenizer,
    tokenizer_batch_size=512,
    core=None,
    work_dir=None,
):
    audit_started = time.perf_counter()
    n = document.atom_count
    base = n + 1

    started = time.perf_counter()
    if core is None:
        all_span_codes = set()

        def collect_span(start, end):
            all_span_codes.add(start * base + end)
            return True

        def zero_pair(*coordinates):
            if len(coordinates) != 3:
                raise AssertionError("pair scorer arity mismatch")
            return 0.0

        baseline_result = exact_grouped_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=n,
                original_boundaries=document.original_boundaries,
                max_boundary_moves=min(
                    3,
                    len(document.original_boundaries),
                ),
                span_allowed=collect_span,
                pair_scorer=zero_pair,
            )
        )
        baseline = {
            "unique_span_checks": baseline_result.unique_span_checks,
            "unique_pair_scores": baseline_result.unique_pair_scores,
            "candidate_boundary_positions": (
                baseline_result.candidate_boundary_positions
            ),
            "reachable_state_count": (
                baseline_result.reachable_state_count
            ),
            "transitions_evaluated": (
                baseline_result.transitions_evaluated
            ),
            "peak_layer_states": baseline_result.peak_layer_states,
            "threat_set_size": baseline_result.threat_set_size,
        }
        ordered_codes = sorted(all_span_codes)
    else:
        if work_dir is None:
            raise ValueError("core topology dump requires a work directory")
        baseline, ordered_codes = dump_all_allowed_topology(
            core,
            document,
            work_dir,
        )
    baseline_seconds = time.perf_counter() - started
    if baseline["unique_span_checks"] != len(ordered_codes):
        raise AssertionError("all-allowed span callback count mismatch")

    started = time.perf_counter()
    legality = {}
    token_count_by_code = {}
    identity_by_code = {}
    for offset in range(0, len(ordered_codes), tokenizer_batch_size):
        codes = ordered_codes[offset:offset + tokenizer_batch_size]
        spans = [divmod(code, base) for code in codes]
        rendered = [
            document.preprocess_unit(system_id, start, end)
            for start, end in spans
        ]
        tokenizer_texts = [
            official_tokenizer_text(
                system_id,
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
    native_tokenization_seconds = time.perf_counter() - started

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

    started = time.perf_counter()
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
    actual_replay_seconds = time.perf_counter() - started
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
        raise AssertionError("span callback count mismatch")
    if result_pair_count != pair_calls:
        raise AssertionError("pair callback count mismatch")

    legal_actual_codes = [
        code for code in actual_span_codes if legality[code]
    ]
    feature_token_counts = {}
    feature_secondary_hashes = {}
    for code in legal_actual_codes:
        identity, secondary = identity_by_code[code]
        token_count = token_count_by_code[code]
        previous_count = feature_token_counts.setdefault(
            identity,
            token_count,
        )
        previous_secondary = feature_secondary_hashes.setdefault(
            identity,
            secondary,
        )
        if (
            previous_count != token_count
            or previous_secondary != secondary
        ):
            raise AssertionError(
                "deduplicated text SHA256 collision or token mismatch"
            )
    return {
        "elapsed_seconds": time.perf_counter() - audit_started,
        "stages_seconds": {
            "all_allowed_topology": baseline_seconds,
            "batched_native_tokenization": (
                native_tokenization_seconds
            ),
            "actual_topology_replay": actual_replay_seconds,
        },
        "all_allowed_topology_source": (
            "python_reference"
            if core is None
            else "v6_cpp_mode5_dump"
        ),
        "baseline_unique_spans": baseline["unique_span_checks"],
        "baseline_unique_pairs": baseline["unique_pair_scores"],
        "unique_query_spans": len(actual_span_codes),
        "legal_query_spans": len(legal_actual_codes),
        "legal_span_input_tokens": sum(
            token_count_by_code[code] for code in legal_actual_codes
        ),
        "deduplicated_feature_texts": len(feature_token_counts),
        "deduplicated_input_tokens": sum(feature_token_counts.values()),
        "actual_unique_pairs": pair_calls,
        "candidate_boundary_positions": (
            result["candidate_boundary_positions"]
            if isinstance(result, dict)
            else result.candidate_boundary_positions
        ),
        "reachable_state_count": (
            result["reachable_state_count"]
            if isinstance(result, dict)
            else result.reachable_state_count
        ),
        "transitions_evaluated": (
            result["transitions_evaluated"]
            if isinstance(result, dict)
            else result.transitions_evaluated
        ),
        "peak_layer_states": (
            result["peak_layer_states"]
            if isinstance(result, dict)
            else result.peak_layer_states
        ),
        "threat_set_size": (
            result["threat_set_size"]
            if isinstance(result, dict)
            else result.threat_set_size
        ),
        "sidecar_bytes": sidecar_bytes(
            system_id,
            len(actual_span_codes),
            len(legal_actual_codes),
        ),
    }


def verify_manifest_rows(v3_audit, row_path, identities):
    row_manifest = v3_audit["row_manifest"]
    if row_manifest["path"] != row_path.name:
        raise ValueError("v3 row-manifest path mismatch")
    if sha256_file(row_path) != row_manifest["sha256"]:
        raise ValueError("v3 row-manifest SHA256 mismatch")
    rows = load_jsonl(row_path)
    if len(rows) != row_manifest["rows"] or len(rows) != len(identities):
        raise ValueError("v3 row-manifest count mismatch")
    for index, (row, item) in enumerate(zip(rows, identities)):
        record = item["record"]
        expected = (
            item["cohort"],
            item["role"],
            record.row_index,
            record.raw_text_sha256,
        )
        observed = (
            row["cohort"],
            row["role"],
            row["row_index"],
            row["raw_text_sha256"],
        )
        if observed != expected:
            raise ValueError(
                f"v3 row identity mismatch at manifest index {index}"
            )
        if hash_raw_value(item["raw_value"]) != row["raw_text_sha256"]:
            raise ValueError(
                f"raw-value SHA256 mismatch at manifest index {index}"
            )
    return rows


def selected_by_slot(selection, system_id):
    documents = selection["systems"][system_id]["documents"]
    output = {row["selection_slot"]: row for row in documents}
    if set(output) != EXPECTED_SLOTS or len(documents) != len(output):
        raise ValueError("selection does not contain the frozen 12 slots")
    return output


def load_provider_profiles(paths, system_id, selection):
    expected = selected_by_slot(selection, system_id)
    profiles = []
    seen = set()
    for path in paths:
        artifact = load_json(path)
        slot = artifact.get("selection_slot")
        if slot in seen:
            raise ValueError(f"duplicate provider slot: {slot}")
        seen.add(slot)
        if (
            artifact.get("audit_mode")
            != "wp2_v6_real_feature_provider"
            or artifact.get("system") != system_id
            or artifact.get("status") != "pass"
            or not all(artifact.get("gates", {}).values())
        ):
            raise ValueError(f"provider artifact did not pass: {path}")
        if (
            artifact["inputs"]["protocol_sha256"] != PROTOCOL_SHA256
            or artifact["inputs"]["selection_sha256"] != SELECTION_SHA256
        ):
            raise ValueError(f"provider input SHA256 mismatch: {path}")
        frozen = expected.get(slot)
        if frozen is None:
            raise ValueError(f"unexpected provider slot: {slot}")
        identity = artifact["identity"]
        for key in (
            "row_index",
            "raw_text_sha256",
            "structural_tape_sha256",
            "atom_count",
            "unit_count",
        ):
            if identity[key] != frozen[key]:
                raise ValueError(
                    f"provider identity mismatch for {slot}: {key}"
                )
        timing = artifact["timing"]["stages_seconds"]
        provider = artifact["provider"]
        topology = artifact["topology"]
        input_tokens = provider["deduplicated_input_tokens"]
        spans = topology["actual_unique_spans"]
        pairs = topology["actual_unique_pairs"]
        if min(input_tokens, spans, pairs) <= 0:
            raise ValueError(f"invalid provider counts for {slot}")

        # Charge both validation replays as one future pass. This is an upper
        # bound without assuming that the two replay durations were equal.
        profiles.append({
            "selection_slot": slot,
            "atom_count": identity["atom_count"],
            "raw_text_sha256": identity["raw_text_sha256"],
            "deduplicated_input_tokens": input_tokens,
            "deduplicated_feature_texts": provider[
                "deduplicated_feature_texts"
            ],
            "baseline_unique_spans": topology[
                "baseline_unique_spans"
            ],
            "baseline_unique_pairs": topology[
                "baseline_unique_pairs"
            ],
            "unique_query_spans": spans,
            "actual_unique_pairs": pairs,
            "feature_seconds_per_input_token": (
                timing["two_batch_provider_replays"] / input_tokens
            ),
            "sidecar_seconds_per_span": (
                timing["sidecar_materialization"] / spans
            ),
            "core_seconds_per_pair": (
                timing["two_core_replays"] / pairs
            ),
            "model_load_seconds": provider["model_load_seconds"],
            "artifact_path": str(path),
            "artifact_sha256": sha256_file(path),
        })
    if seen != EXPECTED_SLOTS:
        missing = sorted(EXPECTED_SLOTS - seen)
        raise ValueError(f"missing provider slots: {missing}")
    profiles.sort(key=lambda row: (
        row["atom_count"],
        row["selection_slot"],
    ))
    if any(
        left["atom_count"] == right["atom_count"]
        for left, right in zip(profiles, profiles[1:])
    ):
        raise ValueError("provider atom-count anchors are not unique")
    return profiles


def adjacent_profile_indices(profiles, atom_count):
    atoms = [profile["atom_count"] for profile in profiles]
    right = bisect_left(atoms, atom_count)
    if right == 0:
        return (0, 1)
    if right == len(profiles):
        return (len(profiles) - 2, len(profiles) - 1)
    if atoms[right] != atom_count:
        return (right - 1, right)
    indices = {right}
    if right > 0:
        indices.add(right - 1)
    if right + 1 < len(profiles):
        indices.add(right + 1)
    return tuple(sorted(indices))


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


def validate_completed_prefix(completed, manifest_rows, system_id):
    if len(completed) > len(manifest_rows):
        raise ValueError("checkpoint contains too many rows")
    for index, row in enumerate(completed):
        source = manifest_rows[index]
        if (
            row.get("system") != system_id
            or row.get("manifest_index") != index
            or row.get("cohort") != source["cohort"]
            or row.get("role") != source["role"]
            or row.get("row_index") != source["row_index"]
            or row.get("raw_text_sha256")
            != source["raw_text_sha256"]
        ):
            raise ValueError(
                f"checkpoint identity mismatch at row {index}"
            )


def initialize_worker(
    system_id,
    model_dir,
    nltk_data,
    tokenizer_batch_size,
    core,
    work_dir,
    profiles,
):
    import nltk

    os.environ["NLTK_DATA"] = str(nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(nltk_data))
    _WORKER.update({
        "system": system_id,
        "tokenizer": load_tokenizer(model_dir, system_id),
        "tokenizer_batch_size": tokenizer_batch_size,
        "core": core,
        "work_dir": work_dir,
        "profiles": profiles,
    })


def audit_row_task(task):
    index, source, raw_value = task
    system_id = _WORKER["system"]
    eligible = bool(source["candidate_eligible"][PRIMARY_CANDIDATE])
    output = {
        "system": system_id,
        "manifest_index": index,
        "cohort": source["cohort"],
        "role": source["role"],
        "row_index": source["row_index"],
        "raw_text_sha256": source["raw_text_sha256"],
        "structural_tape_sha256": source["structural_tape_sha256"],
        "atom_count": source["atom_count"],
        "unit_count": source["unit_count"],
        "eligible": eligible,
        "contains_raw_text": False,
    }
    if not eligible:
        return output

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
    topology = audit_document(
        document,
        system_id,
        _WORKER["tokenizer"],
        _WORKER["tokenizer_batch_size"],
        _WORKER["core"],
        _WORKER["work_dir"],
    )
    profiles = _WORKER["profiles"]
    feature_rate = conservative_rate(
        profiles,
        document.atom_count,
        "feature_seconds_per_input_token",
    )
    sidecar_rate = conservative_rate(
        profiles,
        document.atom_count,
        "sidecar_seconds_per_span",
    )
    core_rate = conservative_rate(
        profiles,
        document.atom_count,
        "core_seconds_per_pair",
    )
    estimate = {
        "topology_seconds": topology["elapsed_seconds"],
        "feature_seconds": (
            topology["deduplicated_input_tokens"]
            * feature_rate["rate"]
        ),
        "sidecar_seconds": (
            topology["unique_query_spans"]
            * sidecar_rate["rate"]
        ),
        "core_seconds": (
            topology["actual_unique_pairs"] * core_rate["rate"]
        ),
        "feature_rate": feature_rate,
        "sidecar_rate": sidecar_rate,
        "core_rate": core_rate,
    }
    estimate["total_seconds_without_model_load"] = sum(
        estimate[key]
        for key in (
            "topology_seconds",
            "feature_seconds",
            "sidecar_seconds",
            "core_seconds",
        )
    )
    output["topology"] = topology
    output["estimate"] = estimate
    return output


def audit_rows(
    args,
    manifest_rows,
    identities,
    profiles,
):
    completed = []
    if args.rows_output.exists():
        if not args.resume:
            raise ValueError(
                "rows output already exists; pass --resume to validate it"
            )
        completed = load_jsonl(args.rows_output)
        validate_completed_prefix(
            completed,
            manifest_rows,
            args.system,
        )
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    tasks = (
        (
            index,
            manifest_rows[index],
            identities[index]["raw_value"],
        )
        for index in range(len(completed), len(manifest_rows))
    )
    worker_args = (
        args.system,
        args.model_dir,
        args.nltk_data,
        args.tokenizer_batch_size,
        args.core,
        args.work_dir,
        profiles,
    )
    if args.workers == 1:
        initialize_worker(*worker_args)
        results = map(audit_row_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=worker_args,
        )
        results = executor.map(audit_row_task, tasks, chunksize=1)
    try:
        with args.rows_output.open(mode, encoding="ascii") as handle:
            for output in results:
                index = output["manifest_index"]
                eligible = output["eligible"]
                handle.write(json.dumps(
                    output,
                    sort_keys=True,
                    ensure_ascii=True,
                ) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed.append(output)
                if (
                    (index + 1) % args.progress_every == 0
                    or index + 1 == len(manifest_rows)
                    or (
                        eligible
                        and output["topology"]["elapsed_seconds"] >= 10
                    )
                ):
                    print(json.dumps({
                        "system": args.system,
                        "completed_rows": index + 1,
                        "total_rows": len(manifest_rows),
                        "eligible": eligible,
                        "atom_count": output["atom_count"],
                        "unit_count": output["unit_count"],
                        "topology_elapsed_seconds": (
                            output.get("topology", {}).get(
                                "elapsed_seconds",
                            )
                        ),
                        "unique_spans": output.get("topology", {}).get(
                            "unique_query_spans"
                        ),
                        "unique_pairs": output.get("topology", {}).get(
                            "actual_unique_pairs"
                        ),
                    }, sort_keys=True), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return completed


def validate_provider_crosscheck(rows, profiles):
    by_sha = {
        row["raw_text_sha256"]: row
        for row in rows
        if row["eligible"]
    }
    checks = []
    for profile in profiles:
        row = by_sha.get(profile["raw_text_sha256"])
        if row is None:
            raise ValueError(
                "provider identity absent from eligible workload"
            )
        topology = row["topology"]
        fields = {
            "baseline_unique_spans": (
                topology["baseline_unique_spans"],
                profile["baseline_unique_spans"],
            ),
            "baseline_unique_pairs": (
                topology["baseline_unique_pairs"],
                profile["baseline_unique_pairs"],
            ),
            "deduplicated_input_tokens": (
                topology["deduplicated_input_tokens"],
                profile["deduplicated_input_tokens"],
            ),
            "deduplicated_feature_texts": (
                topology["deduplicated_feature_texts"],
                profile["deduplicated_feature_texts"],
            ),
            "unique_query_spans": (
                topology["unique_query_spans"],
                profile["unique_query_spans"],
            ),
            "actual_unique_pairs": (
                topology["actual_unique_pairs"],
                profile["actual_unique_pairs"],
            ),
        }
        mismatches = [
            key for key, values in fields.items()
            if values[0] != values[1]
        ]
        checks.append({
            "selection_slot": profile["selection_slot"],
            "raw_text_sha256": profile["raw_text_sha256"],
            "mismatches": mismatches,
        })
    return checks


def aggregate(args, rows, profiles, input_manifest):
    eligible = [row for row in rows if row["eligible"]]
    crosschecks = validate_provider_crosscheck(rows, profiles)
    model_load_seconds = max(
        profile["model_load_seconds"] for profile in profiles
    )
    component_totals = {
        key: sum(row["estimate"][key] for row in eligible)
        for key in (
            "topology_seconds",
            "feature_seconds",
            "sidecar_seconds",
            "core_seconds",
            "total_seconds_without_model_load",
        )
    }
    total_seconds = (
        component_totals["total_seconds_without_model_load"]
        + model_load_seconds
    )
    max_spans = max(
        row["topology"]["unique_query_spans"] for row in eligible
    )
    max_pairs = max(
        row["topology"]["actual_unique_pairs"] for row in eligible
    )
    max_baseline_spans = max(
        row["topology"]["baseline_unique_spans"] for row in eligible
    )
    max_baseline_pairs = max(
        row["topology"]["baseline_unique_pairs"] for row in eligible
    )
    max_sidecar = max(
        row["topology"]["sidecar_bytes"] for row in eligible
    )
    gates = {
        "all_2000_manifest_rows_completed": (
            len(rows) == len(input_manifest) == EXPECTED_MANIFEST_ROWS
        ),
        "eligible_count_matches_frozen_manifest": (
            len(eligible)
            == sum(
                row["candidate_eligible"][PRIMARY_CANDIDATE]
                for row in input_manifest
            )
        ),
        "eligible_count_matches_frozen_system_total": (
            len(eligible) == EXPECTED_ELIGIBLE_ROWS[args.system]
        ),
        "all_provider_topology_crosschecks_exact": all(
            not row["mismatches"] for row in crosschecks
        ),
        "max_spans_le_2000000": max_spans <= SPAN_GATE,
        "max_pairs_le_50000000": max_pairs <= PAIR_GATE,
        "max_all_allowed_spans_le_2000000": (
            max_baseline_spans <= SPAN_GATE
        ),
        "max_all_allowed_pairs_le_50000000": (
            max_baseline_pairs <= PAIR_GATE
        ),
        "max_sidecar_le_50_gib": max_sidecar <= CACHE_GATE_BYTES,
        "conservative_end_to_end_hours_le_48": (
            total_seconds / 3600 <= GPU_HOUR_GATE
        ),
    }
    return {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v6_score_blind_full_workload",
        "system": args.system,
        "status": "pass" if all(gates.values()) else "stop",
        "method": {
            "eligible_workload": (
                "all primary-candidate eligible rows in the frozen "
                "WP2-v3 row manifest"
            ),
            "topology": (
                "exact native-tokenizer v4 query topology per document"
            ),
            "rate_assignment": (
                "maximum observed seconds-per-unit among adjacent "
                "detector-fit atom-count anchors"
            ),
            "outside_anchor_range": (
                "slower rate of the two nearest endpoint anchors"
            ),
            "replay_charge": (
                "the complete two-replay provider/core duration is "
                "charged as one future run without dividing by two"
            ),
            "time_charge": (
                "observed score-blind topology plus estimated feature, "
                "sidecar, core, and one maximum observed model load"
            ),
        },
        "parameters": {
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "max_boundary_moves": 3,
            "span_gate": SPAN_GATE,
            "pair_gate": PAIR_GATE,
            "cache_gate_bytes": CACHE_GATE_BYTES,
            "gpu_hour_gate": GPU_HOUR_GATE,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "workers": args.workers,
            "all_allowed_topology_source": "v6_cpp_mode5_dump",
        },
        "inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "selection_sha256": SELECTION_SHA256,
            "v3_audit_path": str(args.v3_audit),
            "v3_audit_sha256": sha256_file(args.v3_audit),
            "v3_rows_path": str(args.v3_rows),
            "v3_rows_sha256": sha256_file(args.v3_rows),
            "rows_output_path": str(args.rows_output),
            "rows_output_sha256": sha256_file(args.rows_output),
            "core_path": str(args.core),
            "core_sha256": sha256_file(args.core),
            "tokenizer_files": tokenizer_manifest(
                args.model_dir,
                args.system,
            ),
            "provider_profiles": profiles,
        },
        "counts": {
            "manifest_rows": len(rows),
            "eligible_rows": len(eligible),
            "ineligible_rows": len(rows) - len(eligible),
            "total_unique_query_spans": sum(
                row["topology"]["unique_query_spans"]
                for row in eligible
            ),
            "total_legal_query_spans": sum(
                row["topology"]["legal_query_spans"]
                for row in eligible
            ),
            "total_deduplicated_feature_texts": sum(
                row["topology"]["deduplicated_feature_texts"]
                for row in eligible
            ),
            "total_deduplicated_input_tokens": sum(
                row["topology"]["deduplicated_input_tokens"]
                for row in eligible
            ),
            "total_actual_unique_pairs": sum(
                row["topology"]["actual_unique_pairs"]
                for row in eligible
            ),
            "max_document_unique_query_spans": max_spans,
            "max_document_actual_unique_pairs": max_pairs,
            "max_document_all_allowed_spans": max_baseline_spans,
            "max_document_all_allowed_pairs": max_baseline_pairs,
            "max_document_sidecar_bytes": max_sidecar,
        },
        "runtime_upper_bound": {
            **component_totals,
            "model_load_seconds": model_load_seconds,
            "conservative_end_to_end_seconds": total_seconds,
            "conservative_end_to_end_gpu_hours": total_seconds / 3600,
            "feature_only_gpu_hours": (
                component_totals["feature_seconds"] / 3600
            ),
        },
        "provider_topology_crosschecks": crosschecks,
        "gates": gates,
        "resources": {
            "process_peak_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "execution_attestation": {
            "model_loads": 0,
            "tokenizer_loads": args.workers,
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
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v3-audit", type=Path, required=True)
    parser.add_argument("--v3-rows", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument(
        "--provider-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(
        args.progress_every,
        args.tokenizer_batch_size,
        args.workers,
    ) < 1:
        raise ValueError("progress, tokenizer batch, and workers must be positive")
    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(args.nltk_data))
    if nltk.__version__ != SYSTEMS[args.system]["nltk_version"]:
        raise ValueError("NLTK version mismatch")

    protocol_bytes = args.protocol.read_bytes()
    selection_bytes = args.selection.read_bytes()
    if sha256_bytes(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("WP2-v6 protocol SHA256 mismatch")
    if sha256_bytes(selection_bytes) != SELECTION_SHA256:
        raise ValueError("WP2-v5 selection SHA256 mismatch")
    selection = json.loads(selection_bytes)
    v3_audit = load_json(args.v3_audit)
    if (
        v3_audit["audit_mode"] != "wp2_v3_structural_readiness"
        or v3_audit["system_id"] != args.system
        or v3_audit["primary_candidate"] != PRIMARY_CANDIDATE
    ):
        raise ValueError("v3 audit identity mismatch")

    identities, _, revision = frozen_identities(
        args.root,
        args.system,
    )
    if revision != v3_audit["data_revision"]:
        raise ValueError("v3 data revision mismatch")
    manifest_rows = verify_manifest_rows(
        v3_audit,
        args.v3_rows,
        identities,
    )
    profiles = load_provider_profiles(
        args.provider_artifact,
        args.system,
        selection,
    )
    rows = audit_rows(
        args,
        manifest_rows,
        identities,
        profiles,
    )
    output = aggregate(args, rows, profiles, manifest_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "system": args.system,
        "status": output["status"],
        "eligible_rows": output["counts"]["eligible_rows"],
        "gpu_hours": output["runtime_upper_bound"][
            "conservative_end_to_end_gpu_hours"
        ],
        "gates": output["gates"],
    }, sort_keys=True))
    if output["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
