"""Run one frozen detector-fit document through the WP2-v6 provider."""

import argparse
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from audit_certified_resegmentation_wp2_v2_design import (
    NATIVE_TOKEN_LIMIT,
    SYSTEMS,
    load_dataset,
    load_tokenizer,
    official_tokenizer_text,
    untruncated_token_count,
)
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
from certified_resegmentation_wp2_v6_features import (
    MAGIC,
    SEMSTAMP_MODE,
    SIMMARK_DIMENSION,
    SIMMARK_MODE,
    serialize_core_problem,
    write_sidecar,
)
from run_certified_resegmentation_wp2_detector_equivalence import (
    semstamp_core,
    simmark_functions,
    verify_inputs,
)


PROTOCOL_SHA256 = (
    "ddab1e0f8d3714dda22ae383395ea91e14ff10ac55d59e63b0708806d533a350"
)
SELECTION_SHA256 = (
    "413bdd37780b9c3e9e41807fcf80b2e8843951fb26af4570d81312544649e399"
)
PAIR_SAMPLE_PER_DOCUMENT = 512
PROBE_WALL_GATE_SECONDS = 600
RSS_GATE_BYTES = 16 * 1024 ** 3
CACHE_GATE_BYTES = 50 * 1024 ** 3


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_document(root, selection, system_id, slot):
    from nltk.tokenize import sent_tokenize

    frozen = next(
        (
            row
            for row in selection["systems"][system_id]["documents"]
            if row["selection_slot"] == slot
        ),
        None,
    )
    if frozen is None:
        raise ValueError(f"selection slot not found: {slot}")
    config = SYSTEMS[system_id]
    dataset = load_dataset(
        root / config["data_repo_dir"] / config["null_path"]
    )
    raw_value = dataset[frozen["row_index"]]["text"]
    if hash_raw_value(raw_value) != frozen["raw_text_sha256"]:
        raise ValueError("selected raw-value SHA256 mismatch")
    document = canonicalize_structured(
        raw_value,
        sentence_splitter=(
            sent_tokenize if isinstance(raw_value, str) else None
        ),
    )
    if (
        document.structural_tape_sha256
        != frozen["structural_tape_sha256"]
        or document.atom_count != frozen["atom_count"]
        or document.unit_count != frozen["unit_count"]
    ):
        raise ValueError("selected structural identity mismatch")
    return frozen, document


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


def dump_all_allowed_spans(core, document, mode, path):
    completed = subprocess.run(
        [
            str(core),
            "--dump-synthetic-sidecar",
            str(path),
            str(mode),
        ],
        input=baseline_payload(document),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "span-query dump failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    baseline = parse_core_output(completed.stdout, 1)[0]
    return baseline


def read_dumped_span_keys(path, expected_mode, expected_n):
    with path.open("rb") as handle:
        if handle.read(8) != MAGIC:
            raise ValueError("dumped sidecar magic mismatch")
        mode, token_count, count, dimension = struct.unpack(
            "<IIQI",
            handle.read(20),
        )
        if mode != expected_mode or token_count != expected_n:
            raise ValueError("dumped sidecar identity mismatch")
        expected_dimension = (
            1 if mode == SEMSTAMP_MODE else SIMMARK_DIMENSION
        )
        if dimension != expected_dimension:
            raise ValueError("dumped sidecar dimension mismatch")
        if mode == SEMSTAMP_MODE:
            handle.read(64)
        records = []
        for _ in range(count):
            start, end, allowed = struct.unpack("<IIB", handle.read(9))
            if allowed != 1:
                raise ValueError("all-allowed dump contained illegal span")
            records.append((start, end))
            feature_bytes = 1 if mode == SEMSTAMP_MODE else 4 * dimension
            if len(handle.read(feature_bytes)) != feature_bytes:
                raise ValueError("truncated dumped sidecar")
        if handle.read(1):
            raise ValueError("dumped sidecar has trailing bytes")
    return tuple(records)


def freeze_native_topology(document, system_id, tokenizer, all_spans):
    legality = {}
    token_counts = {}
    rendered_text = {}
    for start, end in all_spans:
        text = document.preprocess_unit(system_id, start, end)
        tokenizer_text = official_tokenizer_text(
            system_id,
            text,
            int(start > 0),
        )
        token_count = untruncated_token_count(tokenizer, tokenizer_text)
        key = (start, end)
        legality[key] = token_count <= NATIVE_TOKEN_LIMIT
        token_counts[key] = token_count
        rendered_text[key] = text

    if all(legality.values()):
        actual_spans = tuple(all_spans)
        pair_count = None
        topology_source = "all_allowed_query_set"
    else:
        queried = set()
        pair_count = 0

        def span_allowed(start, end):
            queried.add((start, end))
            try:
                return legality[(start, end)]
            except KeyError as error:
                raise AssertionError(
                    "actual query escaped all-allowed superset"
                ) from error

        def count_pair(*coordinates):
            nonlocal pair_count
            if len(coordinates) != 3:
                raise AssertionError("pair scorer arity mismatch")
            pair_count += 1
            return 0.0

        exact_grouped_sparse_pairwise_certificate(
            SparsePairwiseProblem(
                token_count=document.atom_count,
                original_boundaries=document.original_boundaries,
                max_boundary_moves=min(
                    3,
                    len(document.original_boundaries),
                ),
                span_allowed=span_allowed,
                pair_scorer=count_pair,
            )
        )
        actual_spans = tuple(sorted(queried))
        topology_source = "python_pruned_query_set"

    actual_legality = {
        key: legality[key] for key in actual_spans
    }
    actual_text = {
        key: rendered_text[key]
        for key in actual_spans
        if actual_legality[key]
    }
    return {
        "spans": actual_spans,
        "legality": actual_legality,
        "texts": actual_text,
        "token_counts": {
            key: token_counts[key] for key in actual_spans
        },
        "all_allowed": all(actual_legality.values()),
        "pair_count_if_pruned": pair_count,
        "topology_source": topology_source,
    }


def deduplicate_texts(text_by_span):
    texts = []
    span_to_text_index = {}
    identity_to_index = {}
    for span in sorted(text_by_span):
        text = text_by_span[span]
        identity = sha256_bytes(text.encode("utf-8"))
        index = identity_to_index.get(identity)
        if index is None:
            index = len(texts)
            identity_to_index[identity] = index
            texts.append(text)
        elif texts[index] != text:
            raise AssertionError("rendered-text SHA256 collision")
        span_to_text_index[span] = index
    return texts, span_to_text_index


def bounded_pair_sample_add(heap, system_id, raw_sha, triple):
    encoded = (
        f"{system_id}\0{raw_sha}\0"
        f"{triple[0]}\0{triple[1]}\0{triple[2]}"
    ).encode("ascii")
    rank = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    item = (-rank, triple)
    if len(heap) < PAIR_SAMPLE_PER_DOCUMENT:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def run_python_reference(
    document,
    system_id,
    raw_sha,
    topology,
    pair_value,
):
    sample_heap = []

    def span_allowed(start, end):
        try:
            return topology["legality"][(start, end)]
        except KeyError as error:
            raise AssertionError("reference queried unknown span") from error

    def scoring(left, middle, right):
        triple = (left, middle, right)
        bounded_pair_sample_add(
            sample_heap,
            system_id,
            raw_sha,
            triple,
        )
        return pair_value(left, middle, right)

    result = exact_grouped_sparse_pairwise_certificate(
        SparsePairwiseProblem(
            token_count=document.atom_count,
            original_boundaries=document.original_boundaries,
            max_boundary_moves=min(
                3,
                len(document.original_boundaries),
            ),
            span_allowed=span_allowed,
            pair_scorer=scoring,
            normalization=1.0,
            bias=0.0,
            threshold=0.0,
        )
    )
    endpoints = (
        0,
        *document.original_boundaries,
        document.atom_count,
    )
    p0_pairs = {
        (
            endpoints[index - 1],
            endpoints[index],
            endpoints[index + 1],
        )
        for index in range(1, len(endpoints) - 1)
    }
    sampled = {item[1] for item in sample_heap} | p0_pairs
    return result, tuple(sorted(sampled))


def run_core_replays(core, problem, mode, sidecar):
    payload = serialize_core_problem(problem, mode, sidecar)
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [str(core)],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "factorized core failed: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        outputs.append(completed.stdout)
    if outputs[0] != outputs[1]:
        raise AssertionError("factorized core replay changed bytes")
    return parse_core_output(outputs[0], 1)[0], sha256_bytes(outputs[0])


def compare_core(reference, observed):
    score_error = max(
        abs(reference.original_score - observed["original_score"]),
        abs(reference.certified_score - observed["certified_score"]),
    )
    fields = (
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
        "reachable_state_count",
        "transitions_evaluated",
        "peak_layer_states",
        "candidate_boundary_positions",
        "unique_span_checks",
        "unique_pair_scores",
    )
    mismatches = [
        field
        for field in fields
        if getattr(reference, field) != observed[field]
    ]
    return score_error, mismatches


def load_semstamp(root, model_dir, batch_size):
    model, get_mask, _official_detect = semstamp_core(
        root / "repos/SemStamp",
        model_dir,
    )
    model.batch_size = batch_size
    relation = bytes(
        int(current in set(
            int(value)
            for value in get_mask(
                3,
                0.25,
                previous,
            ).detach().cpu().tolist()
        ))
        for previous in range(8)
        for current in range(8)
    )
    return model, relation


def semstamp_features(model, texts):
    return np.asarray(model.get_hash(texts), dtype=np.uint8)


def load_simmark(model_dir):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_dir), device="cuda").eval()


def simmark_features(model, texts, batch_size):
    return np.asarray(model.encode(
        texts,
        prompt="Represent the sentence for cosine similarity:",
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ), dtype=np.float32)


def feature_sha256(features):
    array = np.ascontiguousarray(features)
    return sha256_bytes(array.tobytes(order="C"))


def main():
    import nltk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.environ["NLTK_DATA"] = str(args.nltk_data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    nltk.data.path.insert(0, str(args.nltk_data))
    if nltk.__version__ != SYSTEMS[args.system]["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    protocol_bytes = args.protocol.read_bytes()
    selection_bytes = args.selection.read_bytes()
    if sha256_bytes(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("WP2-v6 protocol SHA256 mismatch")
    if sha256_bytes(selection_bytes) != SELECTION_SHA256:
        raise ValueError("WP2-v5 selection SHA256 mismatch")
    selection = json.loads(selection_bytes)
    frozen, document = load_selected_document(
        args.root,
        selection,
        args.system,
        args.slot,
    )
    input_manifest = verify_inputs(
        args.root,
        args.model_dir,
        args.system,
    )
    tokenizer = load_tokenizer(args.model_dir, args.system)
    mode = (
        SEMSTAMP_MODE
        if args.system == "semstamp"
        else SIMMARK_MODE
    )

    model_load_started = time.perf_counter()
    if args.system == "semstamp":
        model, relation = load_semstamp(
            args.root,
            args.model_dir,
            args.batch_size,
        )
        simmark_official = None
    else:
        model = load_simmark(args.model_dir)
        relation = None
        simmark_official = simmark_functions(
            args.root / "repos/SimMark"
        )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    torch.cuda.reset_peak_memory_stats()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    dump_path = args.work_dir / (
        f"{args.system}_{args.slot}_all_allowed.bin"
    )
    sidecar_path = args.work_dir / (
        f"{args.system}_{args.slot}_features.bin"
    )
    for path in (dump_path, sidecar_path):
        if path.exists():
            path.unlink()

    probe_started = time.perf_counter()
    stage_seconds = {}
    started = time.perf_counter()
    baseline = dump_all_allowed_spans(
        args.core,
        document,
        mode,
        dump_path,
    )
    all_spans = read_dumped_span_keys(
        dump_path,
        mode,
        document.atom_count,
    )
    dump_path.unlink()
    stage_seconds["all_allowed_query_dump"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    topology = freeze_native_topology(
        document,
        args.system,
        tokenizer,
        all_spans,
    )
    stage_seconds["native_token_topology"] = (
        time.perf_counter() - started
    )
    texts, span_to_text_index = deduplicate_texts(topology["texts"])
    deduplicated_token_counts = {}
    for span, text_index in span_to_text_index.items():
        token_count = topology["token_counts"][span]
        previous = deduplicated_token_counts.setdefault(
            text_index,
            token_count,
        )
        if previous != token_count:
            raise AssertionError(
                "deduplicated text has inconsistent token counts"
            )

    started = time.perf_counter()
    with torch.inference_mode():
        if args.system == "semstamp":
            first_features = semstamp_features(model, texts)
            second_features = semstamp_features(model, texts)
        else:
            first_features = simmark_features(
                model,
                texts,
                args.batch_size,
            )
            second_features = simmark_features(
                model,
                texts,
                args.batch_size,
            )
    torch.cuda.synchronize()
    stage_seconds["two_batch_provider_replays"] = (
        time.perf_counter() - started
    )

    if first_features.shape != second_features.shape:
        raise AssertionError("provider replay changed feature shape")
    replay_max_abs_error = float(np.max(np.abs(
        first_features.astype(np.float64)
        - second_features.astype(np.float64)
    ))) if first_features.size else 0.0
    replay_gate = (
        replay_max_abs_error == 0.0
        if args.system == "semstamp"
        else replay_max_abs_error <= 1e-6
    )

    feature_by_span = {
        span: first_features[index]
        for span, index in span_to_text_index.items()
    }
    if args.system == "semstamp":
        def pair_value(left, middle, right):
            previous = int(feature_by_span[(left, middle)])
            current = int(feature_by_span[(middle, right)])
            return float(relation[previous * 8 + current])
    else:
        simmark_float64 = first_features.astype(
            np.float64,
            copy=True,
        )
        simmark_norms = np.linalg.norm(simmark_float64, axis=1)
        if (
            not np.all(np.isfinite(simmark_norms))
            or np.any(simmark_norms <= 0)
        ):
            raise ValueError("SimMark provider returned invalid norms")

        def pair_value(left, middle, right):
            left_index = span_to_text_index[(left, middle)]
            right_index = span_to_text_index[(middle, right)]
            cosine = float(
                np.dot(
                    simmark_float64[left_index],
                    simmark_float64[right_index],
                )
                / (
                    simmark_norms[left_index]
                    * simmark_norms[right_index]
                )
            )
            distance = (
                0.0
                if 0.68 <= cosine <= 0.76
                else min(
                    abs(cosine - 0.68),
                    abs(cosine - 0.76),
                )
            )
            return math.exp(-250.0 * distance)

    started = time.perf_counter()
    reference, sampled_pairs = run_python_reference(
        document,
        args.system,
        frozen["raw_text_sha256"],
        topology,
        pair_value,
    )
    stage_seconds["python_reference"] = time.perf_counter() - started

    sampled_spans = sorted({
        span
        for left, middle, right in sampled_pairs
        for span in ((left, middle), (middle, right))
    })
    started = time.perf_counter()
    scalar_features = {}
    with torch.inference_mode():
        for span in sampled_spans:
            text = topology["texts"][span]
            if args.system == "semstamp":
                scalar_features[span] = int(
                    semstamp_features(model, [text])[0]
                )
            else:
                scalar_features[span] = simmark_features(
                    model,
                    [text],
                    1,
                )[0]
    torch.cuda.synchronize()
    stage_seconds["official_scalar_sample"] = (
        time.perf_counter() - started
    )

    feature_error = 0.0
    cosine_error = 0.0
    pair_error = 0.0
    for span in sampled_spans:
        if args.system == "semstamp":
            feature_error = max(
                feature_error,
                float(
                    int(feature_by_span[span])
                    != int(scalar_features[span])
                ),
            )
        else:
            feature_error = max(
                feature_error,
                float(np.max(np.abs(
                    np.asarray(feature_by_span[span], dtype=np.float64)
                    - np.asarray(scalar_features[span], dtype=np.float64)
                ))),
            )
    for left, middle, right in sampled_pairs:
        if args.system == "semstamp":
            previous = scalar_features[(left, middle)]
            current = scalar_features[(middle, right)]
            scalar_value = float(relation[previous * 8 + current])
            pair_error = max(
                pair_error,
                abs(pair_value(left, middle, right) - scalar_value),
            )
        else:
            scalar_cosine = float(cosine_similarity(
                np.asarray(
                    scalar_features[(left, middle)]
                ).reshape(1, -1),
                np.asarray(
                    scalar_features[(middle, right)]
                ).reshape(1, -1),
            )[0, 0])
            batch_cosine = float(cosine_similarity(
                np.asarray(
                    feature_by_span[(left, middle)]
                ).reshape(1, -1),
                np.asarray(
                    feature_by_span[(middle, right)]
                ).reshape(1, -1),
            )[0, 0])
            cosine_error = max(
                cosine_error,
                abs(batch_cosine - scalar_cosine),
            )
            scalar_distance = simmark_official[
                "distance_to_intervals"
            ](
                [(0.68, 0.76)],
                scalar_cosine,
            )
            scalar_value = float(math.exp(-250 * scalar_distance))
            pair_error = max(
                pair_error,
                abs(pair_value(left, middle, right) - scalar_value),
            )

    started = time.perf_counter()
    records = []
    for span in topology["spans"]:
        allowed = topology["legality"][span]
        if not allowed:
            feature = None
        elif args.system == "semstamp":
            feature = int(feature_by_span[span])
        else:
            feature = feature_by_span[span]
        records.append((*span, allowed, feature))
    encoded_sidecar = write_sidecar(
        sidecar_path,
        mode,
        document.atom_count,
        records,
        relation=relation,
    )
    sidecar_sha = sha256_bytes(encoded_sidecar)
    del encoded_sidecar
    stage_seconds["sidecar_materialization"] = (
        time.perf_counter() - started
    )

    core_problem = SparsePairwiseProblem(
        token_count=document.atom_count,
        original_boundaries=document.original_boundaries,
        max_boundary_moves=min(
            3,
            len(document.original_boundaries),
        ),
        span_allowed=lambda start, end: topology["legality"][
            (start, end)
        ],
        pair_scorer=pair_value,
        normalization=1.0,
        bias=0.0,
        threshold=0.0,
    )
    started = time.perf_counter()
    observed, core_stdout_sha = run_core_replays(
        args.core,
        core_problem,
        mode,
        sidecar_path,
    )
    stage_seconds["two_core_replays"] = time.perf_counter() - started
    score_error, field_mismatches = compare_core(reference, observed)
    probe_elapsed = time.perf_counter() - probe_started

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    process_rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    tolerances = {
        "feature": 0.0 if args.system == "semstamp" else 1e-6,
        "cosine": 0.0 if args.system == "semstamp" else 1e-6,
        "pair": 0.0 if args.system == "semstamp" else 2.5e-4,
        "core_score": 1e-12 if args.system == "semstamp" else 1e-8,
    }
    gates = {
        "provider_replay": replay_gate,
        "scalar_feature_equivalence": (
            feature_error <= tolerances["feature"]
        ),
        "scalar_cosine_equivalence": (
            cosine_error <= tolerances["cosine"]
        ),
        "scalar_pair_equivalence": pair_error <= tolerances["pair"],
        "core_score_equivalence": (
            score_error <= tolerances["core_score"]
        ),
        "core_discrete_fields_match": not field_mismatches,
        "probe_wall_seconds_le_600": (
            args.slot != "nearest_300"
            or probe_elapsed <= PROBE_WALL_GATE_SECONDS
        ),
        "cuda_reserved_le_80_percent": (
            peak_reserved <= 0.8 * gpu_total
        ),
        "process_rss_le_16_gib": process_rss <= RSS_GATE_BYTES,
        "sidecar_le_50_gib": (
            sidecar_path.stat().st_size <= CACHE_GATE_BYTES
        ),
    }
    status = "pass" if all(gates.values()) else "stop"
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "wp2_v6_real_feature_provider",
        "system": args.system,
        "selection_slot": args.slot,
        "status": status,
        "identity": {
            "row_index": frozen["row_index"],
            "raw_text_sha256": frozen["raw_text_sha256"],
            "structural_tape_sha256": (
                frozen["structural_tape_sha256"]
            ),
            "atom_count": document.atom_count,
            "unit_count": document.unit_count,
            "contains_raw_text": False,
        },
        "inputs": {
            "protocol_sha256": sha256_bytes(protocol_bytes),
            "selection_sha256": sha256_bytes(selection_bytes),
            "core_sha256": sha256_file(args.core),
            "official": input_manifest,
        },
        "provider": {
            "batch_size": args.batch_size,
            "model_load_seconds": model_load_seconds,
            "unique_query_spans": len(topology["spans"]),
            "legal_query_spans": len(topology["texts"]),
            "all_query_spans_legal": topology["all_allowed"],
            "topology_source": topology["topology_source"],
            "deduplicated_feature_texts": len(texts),
            "deduplicated_input_tokens": sum(
                deduplicated_token_counts.values()
            ),
            "feature_shape": list(first_features.shape),
            "first_feature_sha256": feature_sha256(first_features),
            "second_feature_sha256": feature_sha256(second_features),
            "replay_max_abs_error": replay_max_abs_error,
        },
        "equivalence": {
            "sampled_pairs": len(sampled_pairs),
            "sampled_spans": len(sampled_spans),
            "feature_max_abs_error": feature_error,
            "cosine_max_abs_error": cosine_error,
            "pair_value_max_abs_error": pair_error,
            "core_score_max_abs_error": score_error,
            "core_field_mismatches": field_mismatches,
            "core_stdout_sha256": core_stdout_sha,
            "tolerances": tolerances,
        },
        "topology": {
            "baseline_unique_spans": baseline["unique_span_checks"],
            "baseline_unique_pairs": baseline["unique_pair_scores"],
            "actual_unique_spans": reference.unique_span_checks,
            "actual_unique_pairs": reference.unique_pair_scores,
            "threat_set_size": reference.threat_set_size,
        },
        "sidecar": {
            "path": str(sidecar_path),
            "bytes": sidecar_path.stat().st_size,
            "sha256": sidecar_sha,
        },
        "timing": {
            "probe_elapsed_seconds": probe_elapsed,
            "stages_seconds": stage_seconds,
        },
        "resources": {
            "process_peak_rss_bytes": process_rss,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "gpu_total_bytes": gpu_total,
        },
        "gates": gates,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "execution_attestation": {
            "model_loads": 1,
            "tokenizer_loads": 1,
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
        "status": status,
        "system": args.system,
        "slot": args.slot,
        "probe_elapsed_seconds": probe_elapsed,
        "unique_spans": reference.unique_span_checks,
        "unique_pairs": reference.unique_pair_scores,
        "feature_error": feature_error,
        "cosine_error": cosine_error,
        "pair_error": pair_error,
        "core_score_error": score_error,
        "peak_reserved_bytes": peak_reserved,
        "gates": gates,
    }, sort_keys=True))
    if status != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
