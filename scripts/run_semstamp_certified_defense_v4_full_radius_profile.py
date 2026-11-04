"""Profile the original uncapped max(insertions,deletions) radius workload."""

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import struct
import subprocess
import tempfile
import time


STAGE = "semstamp_certified_defense_v4_full_radius_profile1"
TOTAL_ROWS = 3000
COHORTS = ("calibration_null", "independent_test_null", "positive")
RADII = (0, 1, 2, 3)
NATIVE_TOKEN_LIMIT = 512
FEATURE_BATCH_SIZE = 64
PAIR_CAP = 200_000_000
SPAN_CAP = 2_000_000
RSS_CAP_BYTES = 16 * 1024**3
SIDECAR_CAP_BYTES = 50 * 1024**3
GPU_HOUR_CAP = 72.0
CORE_TIMEOUT_SECONDS = 3600
EXPECTED_HARDWARE = {
    "device": "NVIDIA GeForce RTX 3090",
    "driver_version": "595.80",
    "memory_mib": 24576,
}
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}
_WORKER = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
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
        os.chmod(path, mode)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value):
    atomic_write_bytes(path, canonical_bytes(value))


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


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must remain inside root")
    return {
        "root": root,
        "output": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "rows": output_dir / "PROFILE_ROWS.jsonl",
        "timings": output_dir / "sealed/PROFILE_TIMINGS.jsonl",
        "checkpoints": output_dir / "sealed/checkpoints",
        "result": output_dir / "PROFILE_RESULT.json",
        "heartbeat": output_dir / "HEARTBEAT.json",
        "failure": output_dir / "LAST_FAILURE.json",
        "readiness": output_dir / "RESOURCE_AUDIT.json",
        "lock": output_dir / "EXECUTION.lock",
        "p0_dir": (
            root
            / "experiments/semstamp_certified_defense_v2_wp3_detector_health1"
        ),
        "profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1/"
            "semstamp_workload.json"
        ),
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
        raise ValueError("empty input manifest")
    return entries


def validate_contract(paths, args):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen_and_authorized",
        "total_rows": TOTAL_ROWS,
        "radii": list(RADII),
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "pair_cap_per_layer": PAIR_CAP,
        "span_cap_per_radius_union": SPAN_CAP,
        "rss_cap_bytes": RSS_CAP_BYTES,
        "gpu_hour_cap": GPU_HOUR_CAP,
        "score_blind": True,
        "certificate_authorized": False,
        "radius_rule": "requested_b_without_original_edge_cap",
        "operation_cost": "max(insertions,deletions)",
        "sidecar_cap_bytes": SIDECAR_CAP_BYTES,
        "successor_execution_authorized": False,
        "full_profile_requires_readiness_go": True,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    if sha256_file(paths["protocol"]) != contract["protocol_sha256"]:
        raise ValueError("protocol SHA mismatch")
    if sha256_file(Path(__file__).resolve()) != contract["runner_sha256"]:
        raise ValueError("runner SHA mismatch")
    if sha256_file(args.core) != contract["linux_core_sha256"]:
        raise ValueError("Linux profile core SHA mismatch")
    manifest = parse_manifest(paths["manifest"])
    if sha256_file(paths["manifest"]) != contract["manifest_sha256"]:
        raise ValueError("manifest SHA mismatch")
    if manifest != contract["inputs"]:
        raise ValueError("contract/manifest entries differ")
    for relative, expected in manifest.items():
        path = (paths["root"] / relative).resolve()
        if (
            Path(relative).is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"frozen input mismatch: {relative}")
    return contract


def load_profiles(path):
    payload = read_json(path)
    profiles = payload.get("inputs", {}).get("provider_profiles")
    if (
        payload.get("system") != "semstamp"
        or payload.get("status") != "pass"
        or not isinstance(profiles, list)
        or len(profiles) != 12
    ):
        raise ValueError("invalid frozen SemStamp profiles")
    profiles = sorted(profiles, key=lambda row: row["atom_count"])
    if any(
        left["atom_count"] >= right["atom_count"]
        for left, right in zip(profiles, profiles[1:])
    ):
        raise ValueError("profile atom anchors are not strictly increasing")
    if any(row.get("batch_size") not in (None, FEATURE_BATCH_SIZE)
           for row in profiles):
        raise ValueError("provider profile batch mismatch")
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


def parse_histogram(value):
    return {
        int(key): int(count)
        for item in value.split(",")
        for key, count in (item.split(":", 1),)
    }


def parse_core_output(payload):
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 1:
        raise ValueError("core returned the wrong row count")
    fields = lines[0].split("\t")
    if len(fields) != 12 or fields[0] != "0":
        raise ValueError("core output schema mismatch")
    witness = [] if not fields[5] else list(map(int, fields[5].split(",")))
    return {
        "candidate_boundary_positions": int(fields[9]),
        "insertion_histogram": parse_histogram(fields[4]),
        "peak_layer_states": int(fields[8]),
        "reachable_state_count": int(fields[6]),
        "threat_set_size": int(fields[3]),
        "transitions_evaluated": int(fields[7]),
        "unique_pair_scores": int(fields[11]),
        "unique_span_checks": int(fields[10]),
        "witness_edge_count": len(witness),
    }


def core_payload(document, target_m, radius, score_mode, span_mode, sidecar=None):
    fields = [
        document.atom_count,
        len(document.original_boundaries),
        target_m,
        radius,
        score_mode,
        span_mode,
        1,
        0,
        0,
        *document.original_boundaries,
    ]
    if sidecar is not None:
        fields.append(sidecar)
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(value)
    return int(value) * 1024


def peak_child_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if platform.system() == "Darwin":
        return int(value)
    return int(value) * 1024


def run_core(core, arguments, payload):
    started = time.perf_counter()
    completed = subprocess.run(
        [str(core), *map(str, arguments)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=CORE_TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return parse_core_output(completed.stdout), elapsed


def read_mode5_codes(path, expected_n):
    import numpy as np

    dtype = np.dtype([
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
        relation = handle.read(64)
        records = np.fromfile(handle, dtype=dtype, count=count)
        trailing = handle.read(1)
    if (
        mode != 5
        or token_count != expected_n
        or dimension != 1
        or len(relation) != 64
        or len(records) != count
        or trailing
        or np.any(records["allowed"] != 1)
        or np.any(records["start"] >= records["end"])
        or np.any(records["end"] > expected_n)
    ):
        raise ValueError("invalid topology dump")
    base = np.uint64(expected_n + 1)
    codes = (
        records["start"].astype(np.uint64) * base
        + records["end"].astype(np.uint64)
    )
    if len(codes) and np.any(codes[1:] <= codes[:-1]):
        raise ValueError("topology codes are not strictly increasing")
    return codes


def target_counts(document, radius):
    original_m = len(document.original_boundaries)
    return tuple(range(
        max(1, original_m - radius),
        min(document.atom_count - 1, original_m + radius) + 1,
    ))


def dump_layer(core, document, radius, target_m, path):
    observed, elapsed = run_core(
        core,
        ("--dump-synthetic-sidecar", path, "5"),
        core_payload(document, target_m, radius, 3, 1),
    )
    codes = read_mode5_codes(path, document.atom_count)
    path.unlink(missing_ok=True)
    if observed["unique_span_checks"] != len(codes):
        raise ValueError("dump/core span count mismatch")
    if (
        observed["unique_pair_scores"] > PAIR_CAP
        or len(codes) > SPAN_CAP
    ):
        raise RuntimeError("all-allowed layer exceeds frozen cap")
    return observed, codes, elapsed


def synthetic_relation():
    return bytes(
        int(((previous + 1) * 5 + (current + 3) * 3) % 4 == 0)
        for previous in range(8)
        for current in range(8)
    )


def write_mode5_union_sidecar(path, atom_count, codes, allowed):
    if len(codes) == 0 or len(codes) != len(allowed):
        raise ValueError("invalid union sidecar vectors")
    base = atom_count + 1
    started = time.perf_counter()
    with Path(path).open("wb", buffering=1024 * 1024) as handle:
        handle.write(b"WP2V6F1\0")
        handle.write(struct.pack("<IIQI", 5, atom_count, len(codes), 1))
        handle.write(synthetic_relation())
        for code, is_allowed in zip(codes, allowed):
            value = int(code)
            start, end = divmod(value, base)
            handle.write(struct.pack("<IIB", start, end, int(is_allowed)))
            if is_allowed:
                region = ((start + 3) * 5 + (end + 7) * 3) % 8
                handle.write(bytes((region,)))
        handle.flush()
        os.fsync(handle.fileno())
    elapsed = time.perf_counter() - started
    size = Path(path).stat().st_size
    expected = 92 + 9 * len(codes) + sum(bool(value) for value in allowed)
    if size != expected or size > SIDECAR_CAP_BYTES:
        raise ValueError("union sidecar size mismatch")
    return size, elapsed


def assert_subset(subset, superset):
    import numpy as np

    positions = np.searchsorted(superset, subset)
    if (
        np.any(positions >= len(superset))
        or np.any(superset[positions] != subset)
    ):
        raise ValueError("radius union is not nested in maximum union")
    return positions


def deduplicate_feature_texts(document, codes, tokenizer):
    import numpy as np

    base = document.atom_count + 1
    buckets = {}
    unique_payloads = []
    unique_sha256 = []
    span_to_unique = np.empty(len(codes), dtype=np.int64)
    collision_bucket_count = 0
    for index, code in enumerate(codes):
        start, end = divmod(int(code), base)
        payload = document.render_unit_bytes(start, end)
        digest = hashlib.sha256(payload).digest()
        bucket = buckets.setdefault(digest, [])
        selected = None
        for existing_payload, unique_index in bucket:
            if existing_payload == payload:
                selected = unique_index
                break
        if selected is None:
            if bucket:
                collision_bucket_count += 1
            selected = len(unique_payloads)
            bucket.append((payload, selected))
            unique_payloads.append(payload)
            unique_sha256.append(digest.hex())
        span_to_unique[index] = selected

    unique_counts = np.empty(len(unique_payloads), dtype=np.int64)
    for offset in range(0, len(unique_payloads), FEATURE_BATCH_SIZE):
        payloads = unique_payloads[offset:offset + FEATURE_BATCH_SIZE]
        values = tokenizer(
            [value.decode("utf-8", "strict") for value in payloads],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if values and not isinstance(values[0], list):
            if len(payloads) != 1:
                raise ValueError("tokenizer flattened a feature batch")
            values = [values]
        if len(values) != len(payloads):
            raise ValueError("tokenizer feature batch row mismatch")
        unique_counts[offset:offset + len(values)] = [
            len(value) for value in values
        ]
    span_counts = unique_counts[span_to_unique]
    allowed = span_counts <= NATIVE_TOKEN_LIMIT
    legal_unique = unique_counts <= NATIVE_TOKEN_LIMIT
    feature_digest = canonical_digest([
        {
            "sha256": digest,
            "native_tokens": int(count),
            "allowed": bool(count <= NATIVE_TOKEN_LIMIT),
        }
        for digest, count in zip(unique_sha256, unique_counts)
    ])
    return allowed, {
        "all_union_span_count": len(codes),
        "deduplicated_feature_text_count": len(unique_payloads),
        "deduplicated_legal_feature_text_count": int(
            np.count_nonzero(legal_unique)
        ),
        "deduplicated_legal_input_tokens": int(
            unique_counts[legal_unique].sum()
        ),
        "exact_duplicate_span_count": (
            len(codes) - len(unique_payloads)
        ),
        "maximum_native_tokens": int(unique_counts.max()),
        "sha256_collision_bucket_count": collision_bucket_count,
        "feature_text_table_sha256": feature_digest,
        "span_legality_sha256": hashlib.sha256(
            allowed.astype(np.uint8).tobytes(order="C")
        ).hexdigest(),
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


def prepare_document_profile(document, tokenizer, core, scratch, profiles):
    import numpy as np

    effective = {radius: radius for radius in RADII}
    distinct = sorted(set(effective.values()))
    all_layers = {}
    all_unions = {}
    topology_seconds = 0.0
    for radius in distinct:
        layers = []
        layer_codes = []
        for target_m in target_counts(document, radius):
            dump_path = scratch / f"dump-k{radius}-m{target_m}.bin"
            observed, codes, elapsed = dump_layer(
                core, document, radius, target_m, dump_path
            )
            topology_seconds += elapsed
            layers.append({
                "target_edge_count": target_m,
                "all_allowed": observed,
                "topology_seconds": elapsed,
            })
            layer_codes.append(codes)
        union = np.unique(np.concatenate(layer_codes))
        if len(union) > SPAN_CAP:
            raise RuntimeError("radius union exceeds frozen span cap")
        all_layers[radius] = (layers, layer_codes)
        all_unions[radius] = union

    maximum_radius = max(distinct)
    maximum_union = all_unions[maximum_radius]
    maximum_allowed, feature_counts = deduplicate_feature_texts(
        document, maximum_union, tokenizer
    )
    profiles_by_radius = {}
    total_sidecar_seconds = 0.0
    total_core_seconds = 0.0
    max_sidecar_bytes = 0
    max_layer_pairs = 0
    max_layer_spans = 0
    for radius in distinct:
        layers, layer_codes = all_layers[radius]
        union = all_unions[radius]
        union_positions = assert_subset(union, maximum_union)
        union_allowed = maximum_allowed[union_positions]
        sidecar = scratch / f"union-k{radius}.bin"
        sidecar_bytes, sidecar_seconds = write_mode5_union_sidecar(
            sidecar,
            document.atom_count,
            union,
            union_allowed,
        )
        total_sidecar_seconds += sidecar_seconds
        max_sidecar_bytes = max(max_sidecar_bytes, sidecar_bytes)
        layer_rows = []
        try:
            for layer, codes in zip(layers, layer_codes):
                target_m = layer["target_edge_count"]
                pruned, core_seconds = run_core(
                    core,
                    ("--allow-unused-feature-records",),
                    core_payload(
                        document,
                        target_m,
                        radius,
                        5,
                        2,
                        sidecar,
                    ),
                )
                total_core_seconds += core_seconds
                max_layer_pairs = max(
                    max_layer_pairs, pruned["unique_pair_scores"]
                )
                max_layer_spans = max(
                    max_layer_spans, pruned["unique_span_checks"]
                )
                if (
                    pruned["candidate_boundary_positions"]
                    != layer["all_allowed"]["candidate_boundary_positions"]
                    or pruned["unique_pair_scores"] > PAIR_CAP
                    or pruned["unique_span_checks"] > len(codes)
                    or pruned["witness_edge_count"] != target_m
                ):
                    raise ValueError("invalid native-pruned layer result")
                layer_rows.append({
                    **layer,
                    "native_pruned": pruned,
                    "pruned_core_seconds": core_seconds,
                })
        finally:
            sidecar.unlink(missing_ok=True)
        profiles_by_radius[radius] = {
            "effective_radius": radius,
            "target_edge_counts": list(target_counts(document, radius)),
            "all_allowed_union_span_count": len(union),
            "native_legal_union_span_count": int(
                np.count_nonzero(union_allowed)
            ),
            "native_forbidden_union_span_count": int(
                len(union_allowed) - np.count_nonzero(union_allowed)
            ),
            "union_sidecar_bytes": sidecar_bytes,
            "union_sidecar_seconds": sidecar_seconds,
            "all_allowed_pair_sum": sum(
                layer["all_allowed"]["unique_pair_scores"]
                for layer in layer_rows
            ),
            "native_pruned_queried_span_sum": sum(
                layer["native_pruned"]["unique_span_checks"]
                for layer in layer_rows
            ),
            "native_pruned_pair_sum": sum(
                layer["native_pruned"]["unique_pair_scores"]
                for layer in layer_rows
            ),
            "layers": layer_rows,
        }

    radii = {
        str(requested): {
            "requested_radius": requested,
            **profiles_by_radius[effective[requested]],
        }
        for requested in RADII
    }
    feature_rate = conservative_rate(
        profiles, document.atom_count, "feature_seconds_per_input_token"
    )
    sidecar_rate = conservative_rate(
        profiles, document.atom_count, "sidecar_seconds_per_span"
    )
    core_rate = conservative_rate(
        profiles, document.atom_count, "core_seconds_per_pair"
    )
    distinct_rows = [profiles_by_radius[radius] for radius in distinct]
    union_span_total = sum(
        row["all_allowed_union_span_count"] for row in distinct_rows
    )
    pruned_pair_total = sum(
        row["native_pruned_pair_sum"] for row in distinct_rows
    )
    projection = {
        "feature_seconds": (
            feature_counts["deduplicated_legal_input_tokens"]
            * feature_rate["rate"]
        ),
        "union_sidecar_rate_seconds": (
            union_span_total * sidecar_rate["rate"]
        ),
        "pair_rate_core_seconds": (
            pruned_pair_total * core_rate["rate"]
        ),
        "measured_topology_seconds": topology_seconds,
        "measured_union_sidecar_seconds": total_sidecar_seconds,
        "measured_pruned_core_seconds": total_core_seconds,
        "feature_rate": feature_rate,
        "sidecar_rate": sidecar_rate,
        "core_rate": core_rate,
    }
    return radii, {
        **feature_counts,
        "distinct_effective_radii": distinct,
        "maximum_radius_union_span_count": len(maximum_union),
        "maximum_sidecar_bytes": max_sidecar_bytes,
        "maximum_layer_pair_count": max_layer_pairs,
        "maximum_layer_queried_span_count": max_layer_spans,
        "sum_distinct_radius_union_spans": union_span_total,
        "sum_distinct_layer_pruned_pairs": pruned_pair_total,
    }, projection


def initialize_worker(model_dir, nltk_data, core, scratch, profiles):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import nltk
    from transformers import AutoTokenizer

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    nltk.data.find("tokenizers/punkt/english.pickle")
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
    _WORKER.update({
        "tokenizer": tokenizer,
        "core": Path(core),
        "scratch": Path(scratch),
        "profiles": profiles,
    })


def profile_task(task):
    index, planned, frozen, text, protocol_sha256 = task
    from nltk.tokenize import sent_tokenize
    from semstamp_refined_atom_renderer import canonicalize_refined_bytes

    started = time.perf_counter()
    units = sent_tokenize(text)

    def frozen_splitter(value):
        if value != text:
            raise ValueError("renderer splitter received different text")
        return units

    document = canonicalize_refined_bytes(
        text.encode("utf-8"),
        frozen_splitter,
    )
    validate_frozen_structure(index, planned, frozen, document)
    scratch = _WORKER["scratch"] / f"{os.getpid()}-{index}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        radii, resource_counts, projection = prepare_document_profile(
            document,
            _WORKER["tokenizer"],
            _WORKER["core"],
            scratch,
            _WORKER["profiles"],
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    row = {
        "schema_version": 1,
        "stage": STAGE,
        "protocol_sha256": protocol_sha256,
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
        "original_edge_count": len(document.original_boundaries),
        "refinement_count": document.refinement_count,
        "zero_byte_gap_count": len(document.zero_byte_gap_boundaries),
        "topology_sha256": frozen["topology_sha256"],
        "radii": radii,
        "resource_counts": resource_counts,
        "projection": projection,
        "score_access_count": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    timing = {
        "plan_index": index,
        "elapsed_seconds": time.perf_counter() - started,
        "worker_peak_rss_bytes": peak_rss_bytes(),
        "child_peak_rss_bytes": peak_child_rss_bytes(),
    }
    return row, timing


def validate_hardware():
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
    lines = [line.strip() for line in completed.stdout.splitlines() if line]
    if len(lines) != 1:
        raise ValueError("exactly one GPU is required")
    fields = [field.strip() for field in lines[0].split(",")]
    observed = {
        "device": fields[0],
        "driver_version": fields[1],
        "memory_mib": int(fields[2]),
    }
    if observed != EXPECTED_HARDWARE:
        raise ValueError(f"hardware mismatch: {observed}")
    return observed


def selected_indices(value):
    if value is None:
        return list(range(TOTAL_ROWS))
    output = []
    for item in value.split(","):
        if "-" in item:
            lower, upper = map(int, item.split("-", 1))
            output.extend(range(lower, upper + 1))
        else:
            output.append(int(item))
    if (
        not output
        or output != sorted(set(output))
        or output[0] < 0
        or output[-1] >= TOTAL_ROWS
    ):
        raise ValueError("indices must be unique increasing rows in [0,3000)")
    return output


def load_checkpoint(paths, indices, plan, frozen_rows):
    checkpoint_dir = paths["checkpoints"]
    files = sorted(checkpoint_dir.glob("*.json"))
    expected_names = [f"{index:04d}.json" for index in indices[:len(files)]]
    if [path.name for path in files] != expected_names:
        raise ValueError("atomic checkpoints are not an exact plan prefix")
    payloads = [read_json(path) for path in files]
    binding = {
        "contract_sha256": sha256_file(paths["contract"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
    }
    if any(payload["binding"] != binding for payload in payloads):
        raise ValueError("checkpoint contract binding mismatch")
    rows = [payload["row"] for payload in payloads]
    timings = [payload["timing"] for payload in payloads]
    for offset, (row, timing) in enumerate(zip(rows, timings)):
        index = indices[offset]
        if (
            row.get("plan_index") != index
            or timing.get("plan_index") != index
            or row.get("raw_text_sha256") != plan[index]["raw_text_sha256"]
            or row.get("topology_sha256")
            != frozen_rows[index]["topology_sha256"]
            or row.get("stage") != STAGE
            or row.get("protocol_sha256") != sha256_file(paths["protocol"])
        ):
            raise ValueError(f"invalid checkpoint at offset {offset}")
    # The atomic combined checkpoint is authoritative if ledger appends
    # were interrupted between row and timing writes.
    atomic_write_bytes(paths["rows"], b"".join(map(canonical_bytes, rows)))
    atomic_write_bytes(paths["timings"], b"".join(map(canonical_bytes, timings)))
    return rows, timings


def aggregate(rows, timings, indices, profiles, wall_seconds, hardware):
    if len(rows) != len(indices) or len(timings) != len(indices):
        raise ValueError("aggregate requires the complete selected denominator")
    if [row["plan_index"] for row in rows] != indices:
        raise ValueError("profile rows are out of order")
    full = indices == list(range(TOTAL_ROWS))
    distinct_radius_rows = [
        radius
        for row in rows
        for requested, radius in row["radii"].items()
        if int(requested) == radius["effective_radius"]
    ]
    components = {
        key: sum(row["projection"][key] for row in rows)
        for key in (
            "feature_seconds",
            "union_sidecar_rate_seconds",
            "pair_rate_core_seconds",
            "measured_topology_seconds",
            "measured_union_sidecar_seconds",
            "measured_pruned_core_seconds",
        )
    }
    model_load_seconds = max(
        float(profile["model_load_seconds"]) for profile in profiles
    )
    measured_formula = (
        model_load_seconds
        + 2 * (
            components["feature_seconds"]
            + max(
                components["union_sidecar_rate_seconds"],
                components["measured_union_sidecar_seconds"],
            )
            + components["measured_pruned_core_seconds"]
            + components["measured_topology_seconds"]
        )
    )
    rate_formula = (
        model_load_seconds
        + 2 * (
            components["feature_seconds"]
            + components["union_sidecar_rate_seconds"]
            + components["pair_rate_core_seconds"]
            + components["measured_topology_seconds"]
        )
    )
    upper_bound = max(measured_formula, rate_formula)
    max_worker_rss = max(row["worker_peak_rss_bytes"] for row in timings)
    max_child_rss = max(row["child_peak_rss_bytes"] for row in timings)
    cohort_counts = dict(Counter(row["cohort"] for row in rows))
    gates = {
        "selected_rows_complete": len(rows) == len(indices),
        "original_requested_radii_exact": all(
            set(row["radii"]) == {str(b) for b in RADII}
            and all(
                radius["requested_radius"] == int(key)
                and radius["effective_radius"] == int(key)
                and radius["target_edge_counts"] == list(range(
                    max(1, row["original_edge_count"] - int(key)),
                    min(row["atom_count"] - 1,
                        row["original_edge_count"] + int(key)) + 1,
                ))
                for key, radius in row["radii"].items()
            )
            for row in rows
        ),
        "all_layer_pairs_le_200m": all(
            row["resource_counts"]["maximum_layer_pair_count"] <= PAIR_CAP
            for row in rows
        ),
        "all_radius_unions_le_2m": all(
            radius["all_allowed_union_span_count"] <= SPAN_CAP
            for radius in distinct_radius_rows
        ),
        "all_sidecars_le_50_gib": all(
            row["resource_counts"]["maximum_sidecar_bytes"]
            <= SIDECAR_CAP_BYTES
            for row in rows
        ),
        "worker_and_child_rss_le_16_gib": (
            max(max_worker_rss, max_child_rss) <= RSS_CAP_BYTES
        ),
        "feature_sha_collision_buckets_zero": all(
            row["resource_counts"]["sha256_collision_bucket_count"] == 0
            for row in rows
        ),
        "score_access_zero": all(
            row["score_access_count"] == 0 for row in rows
        ),
        "hardware_exact": hardware == EXPECTED_HARDWARE,
    }
    if full:
        gates.update({
            "all_3000_rows_retained": len(rows) == TOTAL_ROWS,
            "balanced_cohorts": cohort_counts == {
                cohort: 1000 for cohort in COHORTS
            },
            "all_42_refinement_rows_retained": sum(
                row["refinement_count"] > 0 for row in rows
            ) == 42,
            "all_59_zero_byte_gaps_retained": sum(
                row["zero_byte_gap_count"] for row in rows
            ) == 59,
            "two_replay_upper_bound_le_72_gpu_hours": (
                upper_bound / 3600 <= GPU_HOUR_CAP
            ),
        })
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": (
            "go" if full and all(gates.values())
            else "stop" if full
            else "pass" if all(gates.values())
            else "invalid"
        ),
        "terminal": (
            "V4_FULL_RADIUS_RESOURCE_GO"
            if full and all(gates.values())
            else "V4_FULL_RADIUS_RESOURCE_READINESS_STOP"
            if full
            else "V4_FULL_RADIUS_PROFILE_SMOKE_PASS"
            if all(gates.values())
            else "V4_FULL_RADIUS_PROFILE_SMOKE_INVALID"
        ),
        "selected_indices": indices,
        "full_denominator": full,
        "counts": {
            "rows": len(rows),
            "cohorts": cohort_counts,
            "refinement_rows": sum(
                row["refinement_count"] > 0 for row in rows
            ),
            "zero_byte_gaps": sum(
                row["zero_byte_gap_count"] for row in rows
            ),
            "distinct_target_layers": sum(
                len(radius["layers"]) for radius in distinct_radius_rows
            ),
            "deduplicated_legal_feature_texts": sum(
                row["resource_counts"][
                    "deduplicated_legal_feature_text_count"
                ]
                for row in rows
            ),
            "deduplicated_legal_input_tokens": sum(
                row["resource_counts"]["deduplicated_legal_input_tokens"]
                for row in rows
            ),
            "sum_distinct_radius_union_spans": sum(
                row["resource_counts"]["sum_distinct_radius_union_spans"]
                for row in rows
            ),
            "sum_distinct_layer_pruned_pairs": sum(
                row["resource_counts"]["sum_distinct_layer_pruned_pairs"]
                for row in rows
            ),
            "maximum_layer_pairs": max(
                row["resource_counts"]["maximum_layer_pair_count"]
                for row in rows
            ),
            "maximum_radius_union_spans": max(
                row["resource_counts"]["maximum_radius_union_span_count"]
                for row in rows
            ),
        },
        "runtime": {
            "profile_wall_seconds": wall_seconds,
            **components,
            "inherited_model_load_seconds": model_load_seconds,
            "two_replay_measured_core_formula_seconds": measured_formula,
            "two_replay_pair_rate_formula_seconds": rate_formula,
            "conservative_two_replay_upper_bound_seconds": upper_bound,
            "conservative_two_replay_upper_bound_gpu_hours": (
                upper_bound / 3600
            ),
        },
        "resources": {
            "maximum_worker_rss_bytes": max_worker_rss,
            "maximum_child_rss_bytes": max_child_rss,
            "observed_free_disk_bytes": shutil.disk_usage(
                Path.cwd()
            ).free,
        },
        "hardware": hardware,
        "gates": gates,
        "decision": {
            "new_wp4_authorized_to_freeze_and_execute": (
                False
            ),
            "wp5_certificate_authorized": False,
            "resource_ready": full and all(gates.values()),
            "successor_execution_authorized": False,
        },
        "execution_attestation": {
            "tokenizer_loads_upper_bound": min(len(indices), 6),
            "sbert_weight_reads": 0,
            "sbert_model_loads": 0,
            "cuda_initializations": 0,
            "region_computations": 0,
            "detector_score_reads": 0,
            "detector_score_computations": 0,
            "certificate_computations": 0,
            "generation_calls": 0,
            "network_provider_calls": 0,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    return result


def acquire_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)


def run(args):
    paths = paths_for(args.root, args.output_dir)
    readiness = read_json(paths["readiness"])
    if readiness.get("audit_status") != "pass" \
            or readiness.get("decision") != "go":
        raise RuntimeError("full-radius resource readiness is not Go")
    contract = validate_contract(paths, args)
    hardware = validate_hardware()
    profiles = load_profiles(paths["profile"])
    indices = selected_indices(args.indices)

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
    rows, timings = load_checkpoint(
        paths, indices, plan, frozen_rows
    )
    remaining = indices[len(rows):]
    if not remaining:
        if paths["result"].is_file():
            return read_json(paths["result"])
        result = aggregate(
            rows, timings, indices, profiles,
            sum(timing["elapsed_seconds"] for timing in timings), hardware,
        )
        result["runtime"]["profile_wall_seconds_recovered"] = True
        atomic_write_json(paths["result"], result)
        return result
    tasks = [
        (
            index,
            plan[index],
            frozen_rows[index],
            texts[index],
            contract["protocol_sha256"],
        )
        for index in remaining
    ]
    if paths["lock"].exists():
        raise RuntimeError("profile execution lock already exists")
    acquire_lock(paths["lock"])
    started = time.perf_counter()
    try:
        Path(args.scratch).mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=initialize_worker,
            initargs=(
                args.model_dir,
                args.nltk_data,
                args.core,
                args.scratch,
                profiles,
            ),
        ) as pool:
            for row, timing in pool.imap(
                profile_task,
                tasks,
                chunksize=1,
            ):
                atomic_write_json(
                    paths["checkpoints"] / f'{row["plan_index"]:04d}.json',
                    {
                        "binding": {
                            "contract_sha256": sha256_file(paths["contract"]),
                            "manifest_sha256": sha256_file(paths["manifest"]),
                        },
                        "row": row,
                        "timing": timing,
                    },
                )
                append_jsonl(paths["rows"], row)
                append_jsonl(paths["timings"], timing)
                rows.append(row)
                timings.append(timing)
                atomic_write_json(paths["heartbeat"], {
                    "schema_version": 1,
                    "stage": STAGE,
                    "status": "running",
                    "selected_rows": len(indices),
                    "completed_rows": len(rows),
                    "last_plan_index": row["plan_index"],
                    "updated_at": utc_now(),
                    "score_blind": True,
                })
        wall_seconds = time.perf_counter() - started
        result = aggregate(
            rows,
            timings,
            indices,
            profiles,
            wall_seconds,
            hardware,
        )
        atomic_write_json(paths["result"], result)
        paths["heartbeat"].unlink(missing_ok=True)
        return result
    finally:
        paths["lock"].unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--indices")
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be in [1,6]")
    return args


def main():
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        paths = paths_for(args.root, args.output_dir)
        if paths["output"].is_dir():
            atomic_write_json(paths["failure"], {
                "schema_version": 1,
                "stage": STAGE,
                "status": "invalid_or_incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
                "contains_raw_text": False,
            })
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
