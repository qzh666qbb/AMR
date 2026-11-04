"""Run the frozen WP2-v11 full-release R1 equivalence audit."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from audit_certified_resegmentation_wp2_v2_design import frozen_identities
from certified_resegmentation_wp2_v3 import canonicalize_structured
from run_certified_resegmentation_wp2_detector_equivalence import (
    package_versions,
    semstamp_core,
    simmark_functions,
    verify_inputs,
)
from run_certified_resegmentation_wp2_v6_provider import (
    load_simmark,
    semstamp_features,
    simmark_features,
)


PROTOCOL_SHA256 = (
    "6cf3efedd69c877ca00af238e70e2ee9aa327b24fa0b2978b5865db2af9118f9"
)
R0_SHA256 = (
    "d3b87d868cc47a771e4bfae80c9cdfc704d9e5087b7c4fab13b132edcdd65713"
)
MANIFEST_SHA256 = {
    "semstamp": (
        "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218"
    ),
    "simmark": (
        "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
    ),
}
MANIFEST_PATH = {
    "semstamp": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/semstamp_v3_readiness_rows.jsonl"
    ),
    "simmark": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/simmark_v3_readiness_rows.jsonl"
    ),
}
V10_P0_PATH = (
    "experiments/certified_resegmentation_wp2_v10_semstamp1/p0"
)
EXPECTED_SEMSTAMP_EXCLUSION = 1432
SIMMARK_GAMMA = 0.5
TOLERANCE = 1e-6


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


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value))
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


def load_jsonl(path):
    with Path(path).open("r", encoding="ascii") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def float_sequence_sha256(values):
    payload = b"".join(struct.pack("<d", float(value)) for value in values)
    return sha256_bytes(payload)


def integer_sequence_sha256(values):
    payload = bytes(int(value) for value in values)
    return sha256_bytes(payload)


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


def materialize_and_validate(item, source, manifest_index):
    raw_value = item["raw_value"]
    document = canonicalize_structured(
        raw_value,
        sentence_splitter=sent_tokenize if isinstance(raw_value, str) else None,
    )
    observed = {
        "raw_text_sha256": document.raw_text_sha256,
        "structural_tape_sha256": document.structural_tape_sha256,
        "atom_count": document.atom_count,
        "unit_count": document.unit_count,
    }
    if observed != {key: source[key] for key in observed}:
        raise ValueError(
            f"structural identity mismatch at manifest index {manifest_index}"
        )
    return document


def validate_common_inputs(args):
    if sha256_file(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("WP2-v11 protocol SHA256 mismatch")
    if sha256_file(args.r0) != R0_SHA256:
        raise ValueError("WP2-v11 R0 SHA256 mismatch")
    if sha256_file(args.identity_manifest) != MANIFEST_SHA256[args.system]:
        raise ValueError("identity manifest SHA256 mismatch")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    nltk.data.path.insert(0, str(args.nltk_data))

    identity = verify_inputs(args.root, args.model_dir, args.system)
    items, _, _ = frozen_identities(args.root, args.system)
    sources = load_jsonl(args.identity_manifest)
    if len(items) != 2000 or len(sources) != 2000:
        raise ValueError("R1 requires exactly 2,000 frozen identities")
    for index, (item, source) in enumerate(zip(items, sources)):
        record = item["record"]
        observed = (
            source.get("cohort"),
            source.get("role"),
            source.get("row_index"),
            source.get("raw_text_sha256"),
        )
        expected = (
            item["cohort"],
            item["role"],
            record.row_index,
            record.raw_text_sha256,
        )
        if observed != expected:
            raise ValueError(f"identity mismatch at manifest index {index}")
    return identity, items, sources


class ReplayLSHModel:
    def __init__(self, regions):
        self._regions = iter(int(region) for region in regions)

    def get_hash(self, texts):
        if len(texts) != 1:
            raise ValueError("official detect_lsh requested a non-scalar batch")
        return [next(self._regions)]

    def assert_exhausted(self):
        try:
            next(self._regions)
        except StopIteration:
            return
        raise AssertionError("official detect_lsh did not consume all regions")


def semstamp_direct_score(regions, relation):
    if len(regions) < 2:
        raise ValueError("SemStamp requires at least two units")
    hits = [
        relation[int(previous) * 8 + int(current)]
        for previous, current in zip(regions[:-1], regions[1:])
    ]
    edge_count = len(hits)
    score = (
        sum(hits) - 0.25 * edge_count
    ) / math.sqrt(edge_count * 0.25 * 0.75)
    return hits, float(score)


def run_semstamp(args, items, sources):
    model, get_mask, official_detect = semstamp_core(
        args.root / "repos/SemStamp",
        args.model_dir,
    )
    model.batch_size = args.batch_size
    relation = bytes(
        int(current in {
            int(value)
            for value in get_mask(3, 0.25, previous).detach().cpu().tolist()
        })
        for previous in range(8)
        for current in range(8)
    )

    rows = []
    exclusions = []
    checkpoint_compared = 0
    checkpoint_region_mismatches = 0
    checkpoint_score_max_error = 0.0
    with torch.inference_mode():
        for index, (item, source) in enumerate(zip(items, sources)):
            try:
                document = materialize_and_validate(item, source, index)
            except ValueError as error:
                if index != EXPECTED_SEMSTAMP_EXCLUSION:
                    raise
                exclusions.append({
                    "manifest_index": index,
                    **identity_fields(source),
                    "reason_type": type(error).__name__,
                    "reason": str(error),
                    "detector_score_accessed": False,
                    "contains_raw_text": False,
                })
                continue

            official_units = tuple(document.original_units)
            direct_units = document.preprocess_partition("semstamp")
            if official_units != direct_units:
                raise AssertionError(
                    f"SemStamp P0 unit mismatch at manifest index {index}"
                )
            regions = np.asarray(
                semstamp_features(model, list(direct_units)),
                dtype=np.uint8,
            )
            if regions.shape != (len(direct_units),):
                raise ValueError(
                    f"invalid SemStamp regions at manifest index {index}"
                )
            replay_model = ReplayLSHModel(regions)
            official_score = float(official_detect(
                list(official_units),
                replay_model,
                lmbd=0.25,
                lsh_dim=3,
            ))
            replay_model.assert_exhausted()
            hits, direct_score = semstamp_direct_score(regions, relation)
            score_error = abs(official_score - direct_score)
            if score_error != 0.0:
                raise AssertionError(
                    f"SemStamp score mismatch at manifest index {index}"
                )

            checkpoint_match = None
            checkpoint_path = args.root / V10_P0_PATH / f"{index:04d}.json"
            if checkpoint_path.is_file():
                checkpoint = json.loads(
                    checkpoint_path.read_text(encoding="ascii")
                )
                checkpoint_compared += 1
                region_match = checkpoint.get("regions") == [
                    int(value) for value in regions
                ]
                checkpoint_region_mismatches += int(not region_match)
                checkpoint_score_error = abs(
                    float(checkpoint["z_score"]) - direct_score
                )
                checkpoint_score_max_error = max(
                    checkpoint_score_max_error,
                    checkpoint_score_error,
                )
                checkpoint_match = (
                    region_match and checkpoint_score_error == 0.0
                )
                if not checkpoint_match:
                    raise AssertionError(
                        f"v10 checkpoint mismatch at manifest index {index}"
                    )

            rows.append({
                "manifest_index": index,
                **identity_fields(source),
                "edge_count": len(hits),
                "regions_sha256": integer_sequence_sha256(regions),
                "hits_sha256": integer_sequence_sha256(hits),
                "hit_count": int(sum(hits)),
                "official_score": official_score,
                "direct_score": direct_score,
                "score_abs_error": score_error,
                "v10_checkpoint_exact": checkpoint_match,
                "contains_raw_text": False,
            })

    expected_exclusions = [EXPECTED_SEMSTAMP_EXCLUSION]
    if [item["manifest_index"] for item in exclusions] != expected_exclusions:
        raise AssertionError("SemStamp exclusion set changed")
    if checkpoint_compared != 1799:
        raise AssertionError("expected 1,799 SemStamp v10 checkpoints")
    return {
        "status": "pass",
        "scored_documents": len(rows),
        "pre_score_exclusions": exclusions,
        "checkpoint_comparison": {
            "compared": checkpoint_compared,
            "region_mismatches": checkpoint_region_mismatches,
            "score_max_abs_error": checkpoint_score_max_error,
        },
        "score_max_abs_error": max(
            row["score_abs_error"] for row in rows
        ),
        "rows": rows,
    }


def simmark_direct_trace(embeddings, gamma):
    distances = []
    soft_counts = []
    for previous, current in zip(embeddings[:-1], embeddings[1:]):
        distance = float(cosine_similarity(
            previous.reshape(1, -1),
            current.reshape(1, -1),
        )[0, 0])
        distances.append(distance)
        interval_distance = (
            0.0
            if 0.68 <= distance <= 0.76
            else min(abs(distance - 0.68), abs(distance - 0.76))
        )
        soft_counts.append(float(np.exp(-250 * interval_distance)))
    edge_count = len(distances)
    score = (
        sum(soft_counts) - gamma * edge_count
    ) / math.sqrt(edge_count * gamma * (1 - gamma) + 1e-12)
    return distances, soft_counts, float(score)


def max_abs_error(left, right):
    if len(left) != len(right):
        return math.inf
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )


def flatten_with_offsets(groups):
    flat = []
    offsets = [0]
    for group in groups:
        flat.extend(group)
        offsets.append(len(flat))
    return flat, offsets


def run_simmark(args, items, sources):
    official = simmark_functions(args.root / "repos/SimMark")
    official_model = load_simmark(args.model_dir)
    direct_model = load_simmark(args.model_dir)
    instruction = "Represent the sentence for cosine similarity:"
    pca_model = official["DummyPCA"]()

    documents = []
    official_units_by_document = []
    direct_units_by_document = []
    for index, (item, source) in enumerate(zip(items, sources)):
        document = materialize_and_validate(item, source, index)
        official_units = tuple(sent_tokenize(item["raw_value"]))
        if official_units != document.original_units:
            raise AssertionError(
                f"SimMark official units mismatch at manifest index {index}"
            )
        direct_units = document.preprocess_partition("simmark")
        expected_direct = tuple(
            unit if unit_index == 0 else " " + unit
            for unit_index, unit in enumerate(official_units)
        )
        if direct_units != expected_direct:
            raise AssertionError(
                f"SimMark preprocessing mismatch at manifest index {index}"
            )
        documents.append(document)
        official_units_by_document.append(official_units)
        direct_units_by_document.append(direct_units)

    raw_values = [item["raw_value"] for item in items]
    with torch.inference_mode():
        official_embeddings = official["calculate_embeddings"](
            raw_values,
            official_model,
            instruction=instruction,
        )
        flat_direct, offsets = flatten_with_offsets(direct_units_by_document)
        direct_flat_embeddings = simmark_features(
            direct_model,
            flat_direct,
            args.batch_size,
        )
    if len(official_embeddings) != 2000:
        raise ValueError("official SimMark embedding groups changed")

    rows = []
    global_embedding_error = 0.0
    global_distance_error = 0.0
    global_soft_error = 0.0
    global_score_error = 0.0
    for index, (
        item,
        source,
        document,
        official_units,
        official_group,
    ) in enumerate(zip(
        items,
        sources,
        documents,
        official_units_by_document,
        official_embeddings,
    )):
        direct_group = direct_flat_embeddings[offsets[index]:offsets[index + 1]]
        official_array = np.asarray(
            official_group.detach().cpu().numpy(),
            dtype=np.float32,
        )
        embedding_error = float(np.max(np.abs(
            official_array - direct_group
        )))
        official_distances = [
            float(value)
            for value in official["calc_dist"](
                list(official_units),
                "cosine",
                official_group,
                pca_model,
            )
        ]
        official_soft_counts = [
            float(np.exp(
                -250
                * official["distance_to_intervals"](
                    [(0.68, 0.76)], distance
                )
            ))
            for distance in official_distances
        ]
        official_score = float(official["detect"](
            list(official_units),
            "cosine",
            official_group,
            pca_model,
            gamma=SIMMARK_GAMMA,
            whole_interval=(0.68, 0.76),
            verbose=False,
            K=250,
        ))
        direct_distances, direct_soft_counts, direct_score = (
            simmark_direct_trace(direct_group, SIMMARK_GAMMA)
        )
        distance_error = max_abs_error(
            official_distances, direct_distances
        )
        soft_error = max_abs_error(
            official_soft_counts, direct_soft_counts
        )
        score_error = abs(official_score - direct_score)
        errors = (
            embedding_error,
            distance_error,
            soft_error,
            score_error,
        )
        if any(error > TOLERANCE for error in errors):
            raise AssertionError(
                f"SimMark equivalence mismatch at manifest index {index}: "
                f"{errors}"
            )
        global_embedding_error = max(global_embedding_error, embedding_error)
        global_distance_error = max(global_distance_error, distance_error)
        global_soft_error = max(global_soft_error, soft_error)
        global_score_error = max(global_score_error, score_error)
        rows.append({
            "manifest_index": index,
            **identity_fields(source),
            "edge_count": len(official_distances),
            "official_distances_sha256": float_sequence_sha256(
                official_distances
            ),
            "direct_distances_sha256": float_sequence_sha256(
                direct_distances
            ),
            "official_soft_counts_sha256": float_sequence_sha256(
                official_soft_counts
            ),
            "direct_soft_counts_sha256": float_sequence_sha256(
                direct_soft_counts
            ),
            "official_score": official_score,
            "direct_score": direct_score,
            "embedding_max_abs_error": embedding_error,
            "cosine_max_abs_error": distance_error,
            "soft_count_max_abs_error": soft_error,
            "score_abs_error": score_error,
            "contains_raw_text": False,
        })

    return {
        "status": "pass",
        "scored_documents": len(rows),
        "pre_score_exclusions": [],
        "equivalence_gamma": SIMMARK_GAMMA,
        "equivalence_gamma_is_performance_parameter": False,
        "maximum_errors": {
            "embedding": global_embedding_error,
            "cosine": global_distance_error,
            "soft_count": global_soft_error,
            "score": global_score_error,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=("semstamp", "simmark"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.nltk_data = args.nltk_data.resolve()
    args.protocol = args.protocol.resolve()
    args.r0 = args.r0.resolve()
    args.identity_manifest = args.identity_manifest.resolve()
    args.output = args.output.resolve()
    if args.batch_size != 64:
        raise ValueError("the frozen R1 batch size is 64")

    identity, items, sources = validate_common_inputs(args)
    torch.cuda.reset_peak_memory_stats()
    if args.system == "semstamp":
        execution = run_semstamp(args, items, sources)
    else:
        execution = run_simmark(args, items, sources)
    torch.cuda.synchronize()

    output = {
        "schema_version": 1,
        "created_on": "2026-09-25",
        "experiment": "certified_resegmentation_wp2_v11_detector_reproduction1",
        "stage": "R1_full_release_equivalence",
        "system": args.system,
        "status": execution["status"],
        "protocol_sha256": PROTOCOL_SHA256,
        "r0_sha256": R0_SHA256,
        "identity": identity,
        "environment": {
            **package_versions(),
            "platform": platform.platform(),
            "cuda_device": torch.cuda.get_device_name(0),
            "batch_size": args.batch_size,
        },
        "execution": execution,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "aggregate_rows_sha256": sha256_bytes(
            canonical_json_bytes(execution["rows"])
        ),
        "attestation": {
            "historically_observed_release_data": True,
            "exploratory_only": True,
            "performance_metrics_computed": False,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "certificate_calls": 0,
            "contains_raw_text": False,
            "network_access": False,
        },
    }
    atomic_write_json(args.output, output)
    print(json.dumps({
        "status": output["status"],
        "system": args.system,
        "scored_documents": execution["scored_documents"],
        "pre_score_exclusions": len(execution["pre_score_exclusions"]),
        "aggregate_rows_sha256": output["aggregate_rows_sha256"],
        "peak_cuda_bytes": output["peak_cuda_bytes"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
