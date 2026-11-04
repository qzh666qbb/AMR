"""Run the frozen WP2 official-core versus direct-list k=0 gate."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable, Iterator, List, Optional, Tuple

from datasets import Dataset, load_from_disk
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch

from certified_resegmentation_wp2_adapter import (
    canonicalize,
    cohort_sort_hash,
    make_source_record,
    split_null_cohort,
)


SYSTEMS = {
    "semstamp": {
        "code_revision": "97db73d11fd80f376a02b0a604d500627622f7e6",
        "data_revision": "bd6e948de2a6fc30f7670dfad347c2b79e252687",
        "repo_dir": "repos/SemStamp",
        "data_repo_dir": "repos/semstamp-data",
        "positive_path": (
            "c4-semstamp-pegasus-parrot/"
            "semstamp-pegasus-bigram=False"
        ),
        "null_path": "original-c4-texts",
        "model_revision": "a77dce8530e100ac1101981e4be390546c1cb282",
        "model_weights": {
            "pytorch_model.bin": (
                "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4"
                "a6a2e9dee"
            ),
        },
        "nltk_version": "3.8.1",
    },
    "simmark": {
        "code_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "data_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "repo_dir": "repos/SimMark",
        "data_repo_dir": "repos/SimMark",
        "positive_path": "watermarked/c4/c4-cosine",
        "null_path": "human/c4",
        "model_revision": "54e5ffb8d484de506e59443b07dc819fb15c7233",
        "model_weights": {
            "pytorch_model.bin": (
                "fb7f63752cf10103be938bc50dfad0b6fa1e63bc67b963471fc827838"
                "d9bbb41"
            ),
            "2_Dense/pytorch_model.bin": (
                "a6e82cba0876dacbadccdea565c9e19e29848d994d23968bd1343b8f"
                "0f762bdc"
            ),
        },
        "nltk_version": "3.9.1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_dataset(path: Path) -> Dataset:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{path} did not contain one Dataset")
    if "text" not in dataset.column_names:
        raise ValueError(f"{path} has no text field")
    return dataset


def extract_source_symbols(
    path: Path,
    names: set[str],
    namespace: dict,
) -> dict:
    """Execute only named function/class definitions from a pinned source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in names
    ]
    observed = {node.name for node in selected}
    missing = names - observed
    if missing:
        raise ValueError(f"missing source symbols in {path}: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def materialize_groups(root: Path, system_id: str, limit: int):
    config = SYSTEMS[system_id]
    data_root = root / config["data_repo_dir"]
    null_dataset = load_dataset(data_root / config["null_path"])
    positive_dataset = load_dataset(data_root / config["positive_path"])

    def records(dataset, role):
        relative_path = (
            f"{config[f'{role}_path']}/data-00000-of-00001.arrow"
        )
        return tuple(
            make_source_record(
                system_id=system_id,
                data_revision=config["data_revision"],
                row_index=index,
                raw_value=row["text"],
                relative_path=relative_path,
            )
            for index, row in enumerate(dataset)
        )

    null_records = records(null_dataset, "null")
    positive_records = records(positive_dataset, "positive")
    split = split_null_cohort(
        system_id=system_id,
        data_revision=config["data_revision"],
        null_records=null_records,
        positive_records=positive_records,
    )
    positive_sorted = tuple(sorted(
        split.positive_retained,
        key=lambda record: (cohort_sort_hash(record), record.row_index),
    ))
    selections = (
        ("detector_fit", "null", split.detector_fit[:limit]),
        (
            "certificate_calibration",
            "null",
            split.certificate_calibration[:limit],
        ),
        ("positive", "positive", positive_sorted[:limit]),
    )
    datasets = {"null": null_dataset, "positive": positive_dataset}
    output = []
    for group, role, selected in selections:
        for record in selected:
            output.append({
                "group": group,
                "role": role,
                "record": record,
                "raw_value": datasets[role][record.row_index]["text"],
            })
    return output


def canonical_document(item, system_id):
    value = item["raw_value"]
    document = canonicalize(
        value,
        sentence_splitter=sent_tokenize if isinstance(value, str) else None,
    )
    if document.unit_count < 2:
        raise ValueError("detector equivalence requires at least two units")
    official_units = (
        tuple(value) if isinstance(value, list) else tuple(sent_tokenize(value))
    )
    if official_units != document.original_units:
        raise AssertionError("canonical P0 differs from official P0")
    direct_units = document.preprocess_partition(system_id)
    return document, official_units, direct_units


def common_document_result(item, document, elapsed_seconds):
    record = item["record"]
    return {
        "group": item["group"],
        "role": item["role"],
        "row_index": record.row_index,
        "cohort_sort_sha256": cohort_sort_hash(record),
        "raw_text_sha256": record.raw_text_sha256,
        "canonical_tape_sha256": document.canonical_tape_sha256,
        "raw_field_type": document.raw_field_type,
        "unit_count": document.unit_count,
        "atom_count": document.atom_count,
        "edge_count": document.unit_count - 1,
        "elapsed_seconds": elapsed_seconds,
        "contains_raw_text": False,
    }


def pre_score_exclusion(item, error):
    record = item["record"]
    return {
        "group": item["group"],
        "role": item["role"],
        "row_index": record.row_index,
        "cohort_sort_sha256": cohort_sort_hash(record),
        "raw_text_sha256": record.raw_text_sha256,
        "reason_code": "invalid_p0_before_detector_score",
        "reason_type": type(error).__name__,
        "reason": str(error),
        "detector_score_accessed": False,
        "contains_raw_text": False,
    }


def semstamp_core(repo_dir: Path, model_dir: Path):
    from nearpy.hashes import RandomBinaryProjections
    from scipy.spatial.distance import cosine, hamming
    from sentence_transformers import SentenceTransformer

    model_symbols = extract_source_symbols(
        repo_dir / "sbert_lsh_model.py",
        {"batched", "LSHModel", "SBERTLSHModel"},
        {
            "np": np,
            "SentenceTransformer": SentenceTransformer,
            "RandomBinaryProjections": RandomBinaryProjections,
            "hamming": hamming,
            "cosine": cosine,
            "List": List,
            "Tuple": Tuple,
            "Callable": Callable,
            "Optional": Optional,
            "Iterator": Iterator,
        },
    )
    rng = torch.Generator("cuda")
    mask_symbols = extract_source_symbols(
        repo_dir / "sampling_lsh_utils.py",
        {"get_mask_from_seed"},
        {
            "torch": torch,
            "rng": rng,
            "hash_key": 15485863,
            "device": "cuda",
        },
    )
    detector_symbols = extract_source_symbols(
        repo_dir / "detection_utils.py",
        {"detect_lsh"},
        {
            "np": np,
            "get_mask_from_seed": mask_symbols["get_mask_from_seed"],
        },
    )
    model = model_symbols["SBERTLSHModel"](
        lsh_model_path=str(model_dir),
        device="cuda",
        batch_size=1,
        lsh_dim=3,
        sbert_type="base",
    )
    return (
        model,
        mask_symbols["get_mask_from_seed"],
        detector_symbols["detect_lsh"],
    )


def semstamp_trace(units, model, get_mask):
    regions = tuple(int(model.get_hash([unit])[0]) for unit in units)
    masks = []
    hits = []
    for previous, current in zip(regions[:-1], regions[1:]):
        mask_tensor = get_mask(3, 0.25, previous)
        mask = tuple(int(value) for value in mask_tensor.detach().cpu().tolist())
        masks.append(mask)
        hits.append(int(current in mask))
    edge_count = len(units) - 1
    hit_count = sum(hits)
    score = (
        hit_count - 0.25 * edge_count
    ) / np.sqrt(edge_count * 0.25 * 0.75)
    return {
        "regions": regions,
        "masks": tuple(masks),
        "hits": tuple(hits),
        "edge_scores": tuple(float(hit) for hit in hits),
        "document_score": float(score),
    }


def run_semstamp(root: Path, model_dir: Path, items):
    config = SYSTEMS["semstamp"]
    repo_dir = root / config["repo_dir"]
    model, get_mask, official_detect = semstamp_core(repo_dir, model_dir)
    results = []
    exclusions = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for item in items:
            started = time.perf_counter()
            try:
                document, official_units, direct_units = canonical_document(
                    item, "semstamp"
                )
            except ValueError as error:
                exclusions.append(pre_score_exclusion(item, error))
                continue
            official_score = float(
                official_detect(
                    list(official_units),
                    model,
                    lmbd=0.25,
                    lsh_dim=3,
                )
            )
            official_trace = semstamp_trace(
                official_units, model, get_mask
            )
            direct_trace = semstamp_trace(direct_units, model, get_mask)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            gates = {
                "units_exact": official_units == direct_units,
                "regions_exact": (
                    official_trace["regions"] == direct_trace["regions"]
                ),
                "masks_exact": (
                    official_trace["masks"] == direct_trace["masks"]
                ),
                "hits_exact": official_trace["hits"] == direct_trace["hits"],
                "official_trace_score_exact": (
                    official_score == official_trace["document_score"]
                ),
                "direct_score_exact": (
                    official_score == direct_trace["document_score"]
                ),
            }
            if not all(gates.values()):
                raise AssertionError(
                    f"SemStamp k=0 mismatch at row "
                    f"{item['record'].row_index}: {gates}"
                )
            result = common_document_result(item, document, elapsed)
            result.update({
                "official_score": official_score,
                "direct_score": direct_trace["document_score"],
                "regions_sha256": hashlib.sha256(
                    json.dumps(
                        direct_trace["regions"], separators=(",", ":")
                    ).encode("ascii")
                ).hexdigest(),
                "masks_sha256": hashlib.sha256(
                    json.dumps(
                        direct_trace["masks"], separators=(",", ":")
                    ).encode("ascii")
                ).hexdigest(),
                "hit_count": sum(direct_trace["hits"]),
                "gates": gates,
            })
            results.append(result)
    return {
        "documents": results,
        "pre_score_exclusions": exclusions,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def simmark_functions(repo_dir: Path):
    return extract_source_symbols(
        repo_dir / "detection.py",
        {
            "DummyPCA",
            "distance_to_intervals",
            "calculate_embeddings",
            "calc_dist",
            "detect",
        },
        {
            "np": np,
            "sent_tokenize": sent_tokenize,
            "cosine_similarity": cosine_similarity,
            "euclidean_distances": euclidean_distances,
        },
    )


def simmark_preprocessed_units(units):
    return tuple(
        unit if index == 0 else " " + unit
        for index, unit in enumerate(units)
    )


def simmark_direct_trace(
    preprocessed_units,
    embedder,
    instruction,
    gamma,
):
    embeddings = embedder.encode(
        list(preprocessed_units),
        prompt=instruction,
        batch_size=256,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    distances = []
    soft_counts = []
    for previous, current in zip(embeddings[:-1], embeddings[1:]):
        previous_array = previous.reshape(1, -1).cpu().detach().numpy()
        current_array = current.reshape(1, -1).cpu().detach().numpy()
        distance = float(cosine_similarity(
            previous_array, current_array
        )[0, 0])
        distances.append(distance)
        if 0.68 <= distance <= 0.76:
            interval_distance = 0.0
        else:
            interval_distance = min(
                abs(distance - 0.68), abs(distance - 0.76)
            )
        soft_counts.append(float(np.exp(-250 * interval_distance)))
    edge_count = len(preprocessed_units) - 1
    score = (
        sum(soft_counts) - gamma * edge_count
    ) / np.sqrt(edge_count * gamma * (1 - gamma) + 1e-12)
    return {
        "distances": tuple(distances),
        "soft_counts": tuple(soft_counts),
        "document_score": float(score),
    }


def max_abs_error(left, right):
    if len(left) != len(right):
        return math.inf
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )


def run_simmark(
    root: Path,
    model_dir: Path,
    items,
    equivalence_gamma=None,
):
    from sentence_transformers import SentenceTransformer

    config = SYSTEMS["simmark"]
    repo_dir = root / config["repo_dir"]
    official = simmark_functions(repo_dir)
    embedder = SentenceTransformer(str(model_dir), device="cuda").eval()
    instruction = "Represent the sentence for cosine similarity:"
    pca_model = official["DummyPCA"]()

    prepared = []
    exclusions = []
    fit_distances = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for item in items:
            try:
                document, official_units, direct_units = canonical_document(
                    item, "simmark"
                )
            except ValueError as error:
                exclusions.append(pre_score_exclusion(item, error))
                continue
            official_embeddings = official["calculate_embeddings"](
                [item["raw_value"]],
                embedder,
                instruction=instruction,
            )[0]
            official_distances = tuple(float(value) for value in official[
                "calc_dist"
            ](
                list(official_units),
                "cosine",
                official_embeddings,
                pca_model,
            ))
            if item["group"] == "detector_fit":
                fit_distances.extend(official_distances)
            prepared.append({
                "item": item,
                "document": document,
                "official_units": official_units,
                "direct_units": direct_units,
                "official_embeddings": official_embeddings,
                "official_distances": official_distances,
            })

        if equivalence_gamma is None:
            if not fit_distances:
                raise ValueError("SimMark preflight gamma has no fit distances")
            counts, bins = np.histogram(np.array(fit_distances), bins=1000)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mask = (bin_centers >= 0.68) & (bin_centers <= 0.76)
            gamma = float(np.sum(counts[mask]) / len(fit_distances))
            gamma_note = (
                "Equivalence-only gamma from the selected detector-fit "
                "subset; not the frozen full-cohort detector parameter."
            )
        else:
            gamma = float(equivalence_gamma)
            gamma_note = (
                "Explicit equivalence-only smoke gamma; not estimated and "
                "not the frozen full-cohort detector parameter."
            )
        if not 0 < gamma < 1:
            raise ValueError("SimMark preflight gamma is outside (0, 1)")

        results = []
        for prepared_item in prepared:
            started = time.perf_counter()
            item = prepared_item["item"]
            document = prepared_item["document"]
            official_units = prepared_item["official_units"]
            direct_units = prepared_item["direct_units"]
            official_distances = prepared_item["official_distances"]
            official_score = float(official["detect"](
                list(official_units),
                "cosine",
                prepared_item["official_embeddings"],
                pca_model,
                gamma=gamma,
                whole_interval=(0.68, 0.76),
                verbose=False,
                K=250,
            ))
            official_soft_counts = tuple(
                float(np.exp(
                    -250
                    * official["distance_to_intervals"](
                        [(0.68, 0.76)], distance
                    )
                ))
                for distance in official_distances
            )
            expected_preprocessed = simmark_preprocessed_units(
                official_units
            )
            direct_trace = simmark_direct_trace(
                direct_units,
                embedder,
                instruction,
                gamma,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            distance_error = max_abs_error(
                official_distances, direct_trace["distances"]
            )
            soft_count_error = max_abs_error(
                official_soft_counts, direct_trace["soft_counts"]
            )
            score_error = abs(
                official_score - direct_trace["document_score"]
            )
            gates = {
                "preprocessed_units_exact": (
                    expected_preprocessed == direct_units
                ),
                "cosine_max_abs_error_le_1e_6": distance_error <= 1e-6,
                "soft_count_max_abs_error_le_1e_6": (
                    soft_count_error <= 1e-6
                ),
                "score_abs_error_le_1e_6": score_error <= 1e-6,
            }
            if not all(gates.values()):
                raise AssertionError(
                    f"SimMark k=0 mismatch at row "
                    f"{item['record'].row_index}: {gates}"
                )
            result = common_document_result(item, document, elapsed)
            result.update({
                "official_score": official_score,
                "direct_score": direct_trace["document_score"],
                "cosine_max_abs_error": distance_error,
                "soft_count_max_abs_error": soft_count_error,
                "score_abs_error": score_error,
                "gates": gates,
            })
            results.append(result)
    return {
        "documents": results,
        "pre_score_exclusions": exclusions,
        "preflight_only_gamma": gamma,
        "preflight_gamma_fit_pair_count": len(fit_distances),
        "preflight_gamma_note": gamma_note,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def verify_inputs(root: Path, model_dir: Path, system_id: str):
    config = SYSTEMS[system_id]
    repo_dir = root / config["repo_dir"]
    if git_head(repo_dir) != config["code_revision"]:
        raise ValueError("official code revision mismatch")
    data_repo_dir = root / config["data_repo_dir"]
    if git_head(data_repo_dir) != config["data_revision"]:
        raise ValueError("data revision mismatch")
    if nltk.__version__ != config["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    observed_weights = {}
    for relative_path, expected in config["model_weights"].items():
        path = model_dir / relative_path
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"model weight SHA256 mismatch: {relative_path}")
        observed_weights[relative_path] = observed
    return {
        "code_revision": config["code_revision"],
        "data_revision": config["data_revision"],
        "model_revision": config["model_revision"],
        "model_weight_sha256": observed_weights,
        "official_source_sha256": {
            path.name: sha256(path)
            for path in (
                (
                    repo_dir / "sbert_lsh_model.py",
                    repo_dir / "sampling_lsh_utils.py",
                    repo_dir / "detection_utils.py",
                )
                if system_id == "semstamp"
                else (repo_dir / "detection.py",)
            )
        },
    }


def package_versions():
    import scipy
    import sentence_transformers
    import sklearn
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nltk": nltk.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=20)
    parser.add_argument("--equivalence-gamma", type=float)
    args = parser.parse_args()

    if args.limit_per_group < 1 or args.limit_per_group > 20:
        raise ValueError("limit-per-group must be in [1, 20]")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen detector contract")
    nltk.data.path.insert(0, str(args.nltk_data))

    identity = verify_inputs(args.root, args.model_dir, args.system)
    items = materialize_groups(
        args.root, args.system, args.limit_per_group
    )
    expected_count = 3 * args.limit_per_group
    if len(items) != expected_count:
        raise ValueError(
            f"expected {expected_count} frozen identities, got {len(items)}"
        )
    started = time.perf_counter()
    if args.system == "semstamp":
        if args.equivalence_gamma is not None:
            raise ValueError(
                "equivalence-gamma is only valid for SimMark smoke runs"
            )
        execution = run_semstamp(args.root, args.model_dir, items)
    else:
        execution = run_simmark(
            args.root,
            args.model_dir,
            items,
            equivalence_gamma=args.equivalence_gamma,
        )
    elapsed = time.perf_counter() - started

    scored_group_counts = {}
    for result in execution["documents"]:
        group = result["group"]
        scored_group_counts[group] = scored_group_counts.get(group, 0) + 1
    selected_group_counts = {}
    for item in items:
        group = item["group"]
        selected_group_counts[group] = selected_group_counts.get(group, 0) + 1
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "official_core_vs_direct_list_k0_equivalence",
        "system_id": args.system,
        "status": "pass",
        "identity": identity,
        "selection": {
            "ordering": "cohort_sort_sha256_then_row_index",
            "limit_per_group": args.limit_per_group,
            "selected_group_counts": selected_group_counts,
            "scored_group_counts": scored_group_counts,
            "selected_unique_document_count": len(items),
            "scored_unique_document_count": len(execution["documents"]),
            "pre_score_exclusion_count": len(
                execution["pre_score_exclusions"]
            ),
        },
        "environment": {
            **package_versions(),
            "platform": platform.platform(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "execution": execution,
        "elapsed_seconds": elapsed,
        "network_access_during_equivalence": False,
        "contains_raw_text": False,
        "execution_attestation": {
            "unique_documents_selected": len(items),
            "unique_documents_scored": len(execution["documents"]),
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "llm_judge_calls": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        output, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "system_id": args.system,
        "unique_document_count": len(execution["documents"]),
        "selected_group_counts": selected_group_counts,
        "scored_group_counts": scored_group_counts,
        "pre_score_exclusion_count": len(
            execution["pre_score_exclusions"]
        ),
        "peak_cuda_bytes": execution["peak_cuda_bytes"],
        "elapsed_seconds": elapsed,
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
