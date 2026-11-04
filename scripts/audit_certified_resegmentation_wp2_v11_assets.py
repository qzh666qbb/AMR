"""Run the score-blind WP2-v11 detector reproduction asset audit."""

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pyarrow.ipc as ipc

from certified_resegmentation_wp2_adapter import hash_raw_value


CREATED_ON = "2026-09-25"
TARGET = {
    "calibration_null": 1000,
    "independent_test_null": 1000,
    "clean_positive": 1000,
}

REPOSITORIES = {
    "semstamp_code": {
        "relative_path": "repos/SemStamp",
        "expected_revision": "97db73d11fd80f376a02b0a604d500627622f7e6",
        "expected_origin": "https://github.com/bohanhou14/SemStamp.git",
    },
    "semstamp_data": {
        "relative_path": "repos/semstamp-data",
        "expected_revision": "bd6e948de2a6fc30f7670dfad347c2b79e252687",
        "expected_origin": "https://github.com/abehou/semstamp-data.git",
    },
    "simmark_code_and_data": {
        "relative_path": "repos/SimMark",
        "expected_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "expected_origin": "https://github.com/DabiriAghdam/SimMark.git",
    },
}

SOURCE_FILES = {
    "semstamp_readme": (
        "semstamp_code",
        "README.md",
        "42892e7abf69a3d30d4d572672a1b787b76254a8af11a001c43f07fd4ed15c3b",
    ),
    "semstamp_detection": (
        "semstamp_code",
        "detection.py",
        "146d81abb5ede8b3e4d7822a1b720521aee217c75356f12bf102e301f0b8096b",
    ),
    "semstamp_lsh_model": (
        "semstamp_code",
        "sbert_lsh_model.py",
        "5e8735e0bd2b8530035b1a9b7951c5f1b002ab875bb706066be5b5925896963a",
    ),
    "semstamp_mask": (
        "semstamp_code",
        "sampling_lsh_utils.py",
        "5fe536616f8d4845cd7d19eddc622b055380dd9364499ce03d4e0b6a5dde2a65",
    ),
    "semstamp_statistic": (
        "semstamp_code",
        "detection_utils.py",
        "2ab984bc838cb36db5213398dcd162ddc6625500d9d52eb2cff813f74bb2d9cd",
    ),
    "simmark_readme": (
        "simmark_code_and_data",
        "README.md",
        "898ba58aec57766acae25050884f4a3dde33a9d4cf18b3c32a3a9d39ec49f3fd",
    ),
    "simmark_detection": (
        "simmark_code_and_data",
        "detection.py",
        "f54c907d9534c50b6e403cdadafcd88dd98eec6b5b2659a7a9d09029f726bd79",
    ),
    "simmark_sampling": (
        "simmark_code_and_data",
        "sampling.py",
        "b3529f0bad5776ec2077f3fe2cef96685909cc2adbdd9b1ed8b736ff4ce45261",
    ),
    "simmark_pca": (
        "simmark_code_and_data",
        "pca_model_16.pkl",
        "9fc91b35549cf4a67968100351a9f433075ac732197997c9960d6122a9e6214b",
    ),
}

PUBLISHED_REFERENCE_OUTPUTS = {
    "simmark_c4_cosine_results": {
        "repository": "simmark_code_and_data",
        "relative_path": "watermarked/c4/c4-cosine/results.txt",
    },
    "simmark_c4_cosine_distribution": {
        "repository": "simmark_code_and_data",
        "relative_path": (
            "watermarked/c4/c4-cosine/"
            "mode=last_token-cluster_size=50-distribution.txt"
        ),
    },
}

MODELS = {
    "semstamp_c4_sbert": {
        "relative_path": "models/semstamp-c4-sbert-a77dce8530e1",
        "repository": "AbeHou/SemStamp-c4-sbert",
        "revision": "a77dce8530e100ac1101981e4be390546c1cb282",
        "weights": {
            "pytorch_model.bin": (
                "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4"
                "a6a2e9dee"
            ),
        },
    },
    "semstamp_booksum_sbert": {
        "relative_path": None,
        "repository": "AbeHou/SemStamp-booksum-sbert",
        "revision": None,
        "weights": {},
    },
    "simmark_instructor_large": {
        "relative_path": "models/instructor-large-54e5ffb8d484",
        "repository": "hkunlp/instructor-large",
        "revision": "54e5ffb8d484de506e59443b07dc819fb15c7233",
        "weights": {
            "pytorch_model.bin": (
                "fb7f63752cf10103be938bc50dfad0b6fa1e63bc67b963471fc827838"
                "d9bbb41"
            ),
            "2_Dense/pytorch_model.bin": (
                "a6e82cba0876dacbadccdea565c9e19e29848d994d23968bd1343b8f"
                "0f762bdc"
            ),
        },
    },
}

DATASETS = {
    "semstamp_human_c4": {
        "repository": "semstamp_data",
        "path": "original-c4-texts/data-00000-of-00001.arrow",
        "role": "null",
    },
    "semstamp_positive_c4_pegasus": {
        "repository": "semstamp_data",
        "path": (
            "c4-semstamp-pegasus-parrot/"
            "semstamp-pegasus-bigram=False/data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "semstamp_positive_c4_gpt35": {
        "repository": "semstamp_data",
        "path": (
            "c4-semstamp-gpt3.5-bigram=False-cleaned/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "semstamp_human_booksum": {
        "repository": "semstamp_data",
        "path": "original-booksum-texts/data-00000-of-00001.arrow",
        "role": "null",
    },
    "semstamp_positive_booksum_pegasus": {
        "repository": "semstamp_data",
        "path": (
            "booksum-semstamp-pegasus-parrot/"
            "-pegasus-num_beams=25-bigram=False-cleaned/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "semstamp_positive_booksum_gpt35": {
        "repository": "semstamp_data",
        "path": (
            "booksum-semstamp-gpt3.5-bigram=False-cleaned/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "simmark_human_c4": {
        "repository": "simmark_code_and_data",
        "path": "human/c4/data-00000-of-00001.arrow",
        "role": "null",
    },
    "simmark_positive_c4_cosine": {
        "repository": "simmark_code_and_data",
        "path": "watermarked/c4/c4-cosine/data-00000-of-00001.arrow",
        "role": "positive",
    },
    "simmark_positive_c4_euclidean": {
        "repository": "simmark_code_and_data",
        "path": (
            "watermarked/c4/c4-euclidean-pca/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "simmark_human_booksum": {
        "repository": "simmark_code_and_data",
        "path": "human/booksum/data-00000-of-00001.arrow",
        "role": "null",
    },
    "simmark_positive_booksum_cosine": {
        "repository": "simmark_code_and_data",
        "path": (
            "watermarked/booksum/booksum-cosine/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "simmark_positive_booksum_euclidean": {
        "repository": "simmark_code_and_data",
        "path": (
            "watermarked/booksum/booksum-euclidean/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
    "simmark_human_tifu": {
        "repository": "simmark_code_and_data",
        "path": "human/tifu/data-00000-of-00001.arrow",
        "role": "null",
    },
    "simmark_positive_tifu_cosine": {
        "repository": "simmark_code_and_data",
        "path": "watermarked/tifu/tifu-cosine/data-00000-of-00001.arrow",
        "role": "positive",
    },
    "simmark_positive_tifu_euclidean": {
        "repository": "simmark_code_and_data",
        "path": (
            "watermarked/tifu/tifu-euclidean-pca/"
            "data-00000-of-00001.arrow"
        ),
        "role": "positive",
    },
}

CANDIDATES = (
    {
        "candidate_id": "semstamp_c4_lsh_pegasus",
        "system": "semstamp",
        "method": "lsh",
        "domain": "c4_realnewslike",
        "null_asset": "semstamp_human_c4",
        "positive_asset": "semstamp_positive_c4_pegasus",
        "model": "semstamp_c4_sbert",
        "provenance_rank": 1,
        "provenance_class": "exact_readme_command",
        "in_scope": True,
        "parameters": {"detection_mode": "lsh", "sp_dim": 3, "lmbd": 0.25},
        "evidence": ["SemStamp README.md:70-81", "detection.py:16-26"],
    },
    {
        "candidate_id": "semstamp_c4_lsh_gpt35",
        "system": "semstamp",
        "method": "lsh",
        "domain": "c4_realnewslike",
        "null_asset": "semstamp_human_c4",
        "positive_asset": "semstamp_positive_c4_gpt35",
        "model": "semstamp_c4_sbert",
        "provenance_rank": 2,
        "provenance_class": "tracked_release_asset_no_exact_readme_command",
        "in_scope": True,
        "parameters": {"detection_mode": "lsh", "sp_dim": 3, "lmbd": 0.25},
        "evidence": ["SemStamp README.md:81", "git tree at pinned data revision"],
    },
    {
        "candidate_id": "semstamp_booksum_lsh_pegasus",
        "system": "semstamp",
        "method": "lsh",
        "domain": "booksum",
        "null_asset": "semstamp_human_booksum",
        "positive_asset": "semstamp_positive_booksum_pegasus",
        "model": "semstamp_booksum_sbert",
        "provenance_rank": 3,
        "provenance_class": "readme_named_model_and_tracked_release_assets",
        "in_scope": True,
        "parameters": {"detection_mode": "lsh", "sp_dim": 3, "lmbd": 0.25},
        "evidence": ["SemStamp README.md:81,100", "git tree at pinned data revision"],
    },
    {
        "candidate_id": "simmark_c4_cosine",
        "system": "simmark",
        "method": "cosine",
        "domain": "c4_realnewslike",
        "null_asset": "simmark_human_c4",
        "positive_asset": "simmark_positive_c4_cosine",
        "model": "simmark_instructor_large",
        "provenance_rank": 1,
        "provenance_class": "exact_readme_command",
        "in_scope": True,
        "parameters": {"a": 0.68, "b": 0.76, "K": 250, "use_pca": False},
        "evidence": ["SimMark README.md:59-69,91-107", "detection.py:46-57"],
    },
    {
        "candidate_id": "simmark_booksum_cosine",
        "system": "simmark",
        "method": "cosine",
        "domain": "booksum",
        "null_asset": "simmark_human_booksum",
        "positive_asset": "simmark_positive_booksum_cosine",
        "model": "simmark_instructor_large",
        "provenance_rank": 2,
        "provenance_class": "readme_parameter_table_and_tracked_clean_asset",
        "in_scope": True,
        "parameters": {"a": 0.68, "b": 0.76, "K": 250, "use_pca": False},
        "evidence": ["SimMark README.md:81-107", "git tree at pinned revision"],
    },
    {
        "candidate_id": "simmark_tifu_cosine",
        "system": "simmark",
        "method": "cosine",
        "domain": "reddit_tifu",
        "null_asset": "simmark_human_tifu",
        "positive_asset": "simmark_positive_tifu_cosine",
        "model": "simmark_instructor_large",
        "provenance_rank": 3,
        "provenance_class": "readme_parameter_table_and_tracked_clean_asset",
        "in_scope": True,
        "parameters": {"a": 0.68, "b": 0.76, "K": 250, "use_pca": False},
        "evidence": ["SimMark README.md:81,91-107", "git tree at pinned revision"],
    },
    {
        "candidate_id": "simmark_c4_euclidean_pca",
        "system": "simmark",
        "method": "euclidean_pca",
        "domain": "c4_realnewslike",
        "null_asset": "simmark_human_c4",
        "positive_asset": "simmark_positive_c4_euclidean",
        "model": "simmark_instructor_large",
        "provenance_rank": 4,
        "provenance_class": "exact_readme_command_different_detector_variant",
        "in_scope": False,
        "parameters": {"a": 0.28, "b": 0.36, "K": 250, "use_pca": True},
        "evidence": ["SimMark README.md:59-69,91-107"],
    },
    {
        "candidate_id": "simmark_booksum_euclidean",
        "system": "simmark",
        "method": "euclidean",
        "domain": "booksum",
        "null_asset": "simmark_human_booksum",
        "positive_asset": "simmark_positive_booksum_euclidean",
        "model": "simmark_instructor_large",
        "provenance_rank": 5,
        "provenance_class": "readme_parameter_table_different_detector_variant",
        "in_scope": False,
        "parameters": {"a": 0.4, "b": 0.55, "K": 250, "use_pca": False},
        "evidence": ["SimMark README.md:81-107"],
    },
    {
        "candidate_id": "simmark_tifu_euclidean_pca",
        "system": "simmark",
        "method": "euclidean_pca",
        "domain": "reddit_tifu",
        "null_asset": "simmark_human_tifu",
        "positive_asset": "simmark_positive_tifu_euclidean",
        "model": "simmark_instructor_large",
        "provenance_rank": 6,
        "provenance_class": "readme_parameter_table_different_detector_variant",
        "in_scope": False,
        "parameters": {"a": 0.28, "b": 0.36, "K": 250, "use_pca": True},
        "evidence": ["SimMark README.md:81,91-107"],
    },
)

HISTORICAL_MANIFESTS = {
    "semstamp": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/semstamp_v3_readiness_rows.jsonl"
    ),
    "simmark": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/simmark_v3_readiness_rows.jsonl"
    ),
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(arguments, cwd=None):
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_blob(repo, relative_path):
    environment = dict(os.environ)
    environment["GIT_NO_LAZY_FETCH"] = "1"
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def repository_audit(root):
    results = {}
    for repository_id, config in REPOSITORIES.items():
        path = root / config["relative_path"]
        revision = run_text(["git", "rev-parse", "HEAD"], cwd=path)
        origin = run_text(["git", "remote", "get-url", "origin"], cwd=path)
        status = run_text(["git", "status", "--porcelain"], cwd=path)
        results[repository_id] = {
            "relative_path": config["relative_path"],
            "revision": revision,
            "expected_revision": config["expected_revision"],
            "revision_matches": revision == config["expected_revision"],
            "origin": origin,
            "expected_origin": config["expected_origin"],
            "origin_matches": origin == config["expected_origin"],
            "worktree_clean": status == "",
        }
    return results


def source_audit(root):
    results = {}
    for source_id, (repository_id, relative_path, expected) in SOURCE_FILES.items():
        repo = root / REPOSITORIES[repository_id]["relative_path"]
        actual = sha256_bytes(git_blob(repo, relative_path))
        results[source_id] = {
            "repository": repository_id,
            "relative_path": relative_path,
            "sha256": actual,
            "expected_sha256": expected,
            "sha256_matches": actual == expected,
        }
    return results


def reference_output_audit(root):
    results = {}
    for output_id, config in PUBLISHED_REFERENCE_OUTPUTS.items():
        repo = root / REPOSITORIES[config["repository"]]["relative_path"]
        blob = git_blob(repo, config["relative_path"])
        results[output_id] = {
            **config,
            "size_bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "contains_numerical_oracle": len(blob.strip()) > 0,
        }
    results["semstamp_c4_lsh_pegasus"] = {
        "repository": "semstamp_data",
        "relative_path": (
            "c4-semstamp-pegasus-parrot/semstamp-pegasus-bigram=False"
        ),
        "tracked_detector_output_paths": [],
        "contains_numerical_oracle": False,
    }
    return results


def model_audit(root):
    results = {}
    for model_id, config in MODELS.items():
        relative_path = config["relative_path"]
        present = relative_path is not None and (root / relative_path).is_dir()
        weights = {}
        for relative_weight, expected in config["weights"].items():
            path = root / relative_path / relative_weight
            actual = sha256_file(path) if path.is_file() else None
            weights[relative_weight] = {
                "present": path.is_file(),
                "sha256": actual,
                "expected_sha256": expected,
                "sha256_matches": actual == expected,
            }
        results[model_id] = {
            **config,
            "present": present,
            "weights": weights,
            "ready": (
                present
                and config["revision"] is not None
                and all(item["sha256_matches"] for item in weights.values())
            ),
        }
    return results


def dataset_audit(root):
    summaries = {}
    hash_sets = {}
    for dataset_id, config in DATASETS.items():
        repo = root / REPOSITORIES[config["repository"]]["relative_path"]
        blob = git_blob(repo, config["path"])
        table = ipc.open_stream(BytesIO(blob)).read_all()
        if "text" not in table.column_names:
            raise ValueError(f"{dataset_id} has no text field")
        raw_values = table["text"].to_pylist()
        hashes = [hash_raw_value(value) for value in raw_values]
        hash_set = frozenset(hashes)
        raw_types = Counter(
            "string" if isinstance(value, str) else "list"
            for value in raw_values
        )
        hash_sets[dataset_id] = hash_set
        summaries[dataset_id] = {
            **config,
            "arrow_sha256": sha256_bytes(blob),
            "rows": table.num_rows,
            "columns": table.column_names,
            "raw_field_type_counts": dict(sorted(raw_types.items())),
            "unique_raw_sha256": len(hash_set),
            "duplicate_rows": len(hashes) - len(hash_set),
            "contains_raw_text": False,
        }
    return summaries, hash_sets


def historical_audit(root):
    results = {}
    hash_sets = {}
    for system, relative_path in HISTORICAL_MANIFESTS.items():
        path = root / relative_path
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        hashes = frozenset(row["raw_text_sha256"] for row in rows)
        role_counts = Counter(row["role"] for row in rows)
        cohort_counts = Counter(row["cohort"] for row in rows)
        role_hashes = {
            role: frozenset(
                row["raw_text_sha256"]
                for row in rows
                if row["role"] == role
            )
            for role in role_counts
        }
        hash_sets[system] = {
            "all": hashes,
            **role_hashes,
        }
        results[system] = {
            "relative_path": relative_path,
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_raw_sha256": len(hashes),
            "role_counts": dict(sorted(role_counts.items())),
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "contains_raw_text": False,
            "confirmation_reuse_policy": (
                "all manifest identities count as observed, including "
                "pre-score exclusions"
            ),
        }
    return results, hash_sets


def overlap_record(left, right, hash_sets):
    return {
        "left": left,
        "right": right,
        "intersection": len(hash_sets[left] & hash_sets[right]),
    }


def candidate_audit(candidates, datasets, dataset_hashes, history_hashes, models):
    output = []
    for candidate in candidates:
        null_id = candidate["null_asset"]
        positive_id = candidate["positive_asset"]
        observed = history_hashes[candidate["system"]]["all"]
        null_hashes = dataset_hashes[null_id]
        positive_hashes = dataset_hashes[positive_id]
        fresh_null = null_hashes - observed
        fresh_positive = positive_hashes - observed
        fresh_cross_overlap = fresh_null & fresh_positive
        usable_fresh_null = len(fresh_null - fresh_cross_overlap)
        usable_fresh_positive = len(fresh_positive - fresh_cross_overlap)
        capacity = {
            "fresh_unique_null": len(fresh_null),
            "fresh_unique_positive": len(fresh_positive),
            "fresh_cross_role_overlap": len(fresh_cross_overlap),
            "usable_fresh_unique_null": usable_fresh_null,
            "usable_fresh_unique_positive": usable_fresh_positive,
            "required_unique_null": (
                TARGET["calibration_null"] + TARGET["independent_test_null"]
            ),
            "required_unique_positive": TARGET["clean_positive"],
        }
        capacity["target_capacity_pass"] = (
            usable_fresh_null >= capacity["required_unique_null"]
            and usable_fresh_positive >= capacity["required_unique_positive"]
        )
        diagnostics_ready = (
            candidate["in_scope"]
            and models[candidate["model"]]["ready"]
            and datasets[null_id]["unique_raw_sha256"] > 0
            and datasets[positive_id]["unique_raw_sha256"] > 0
        )
        output.append({
            **candidate,
            "published_asset_counts": {
                "null_rows": datasets[null_id]["rows"],
                "null_unique": datasets[null_id]["unique_raw_sha256"],
                "positive_rows": datasets[positive_id]["rows"],
                "positive_unique": datasets[positive_id]["unique_raw_sha256"],
                "cross_role_overlap": len(null_hashes & positive_hashes),
            },
            "historical_overlap": {
                "null_unique": len(null_hashes & observed),
                "positive_unique": len(positive_hashes & observed),
            },
            "fresh_capacity": capacity,
            "diagnostic_readiness": (
                "go" if diagnostics_ready else "stop"
            ),
            "fresh_confirmation_readiness": (
                "go" if (
                    candidate["in_scope"]
                    and diagnostics_ready
                    and capacity["target_capacity_pass"]
                ) else "stop"
            ),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    repositories = repository_audit(root)
    sources = source_audit(root)
    reference_outputs = reference_output_audit(root)
    models = model_audit(root)
    datasets, dataset_hashes = dataset_audit(root)
    history, history_hashes = historical_audit(root)
    candidates = candidate_audit(
        CANDIDATES,
        datasets,
        dataset_hashes,
        history_hashes,
        models,
    )

    selected_overlaps = [
        overlap_record(
            "simmark_human_c4", "semstamp_human_c4", dataset_hashes
        ),
        overlap_record(
            "simmark_human_booksum",
            "semstamp_human_booksum",
            dataset_hashes,
        ),
        overlap_record(
            "semstamp_positive_c4_pegasus",
            "semstamp_positive_c4_gpt35",
            dataset_hashes,
        ),
        overlap_record(
            "semstamp_positive_booksum_pegasus",
            "semstamp_positive_booksum_gpt35",
            dataset_hashes,
        ),
    ]

    c4_coverage = {}
    for system, null_id, positive_id in (
        (
            "semstamp",
            "semstamp_human_c4",
            "semstamp_positive_c4_pegasus",
        ),
        (
            "simmark",
            "simmark_human_c4",
            "simmark_positive_c4_cosine",
        ),
    ):
        observed = history_hashes[system]["all"]
        c4_coverage[system] = {
            "published_null_unique": len(dataset_hashes[null_id]),
            "observed_null_unique": len(dataset_hashes[null_id] & observed),
            "fresh_null_unique": len(dataset_hashes[null_id] - observed),
            "published_positive_unique": len(dataset_hashes[positive_id]),
            "observed_positive_unique": len(
                dataset_hashes[positive_id] & observed
            ),
            "fresh_positive_unique": len(
                dataset_hashes[positive_id] - observed
            ),
        }

    repository_gate = all(
        item["revision_matches"]
        and item["origin_matches"]
        and item["worktree_clean"]
        for item in repositories.values()
    )
    source_gate = all(item["sha256_matches"] for item in sources.values())
    diagnostic_go = {
        system: any(
            item["system"] == system
            and item["diagnostic_readiness"] == "go"
            for item in candidates
        )
        for system in ("semstamp", "simmark")
    }
    confirmation_go = {
        system: any(
            item["system"] == system
            and item["fresh_confirmation_readiness"] == "go"
            for item in candidates
        )
        for system in ("semstamp", "simmark")
    }

    output = {
        "schema_version": 1,
        "created_on": CREATED_ON,
        "audit_stage": "WP2-v11-R0",
        "audit_mode": "score_blind_provenance_and_capacity",
        "target_confirmation_capacity": TARGET,
        "repositories": repositories,
        "source_files": sources,
        "published_reference_outputs": reference_outputs,
        "models": models,
        "datasets": datasets,
        "historical_manifests": history,
        "selected_overlaps": selected_overlaps,
        "published_c4_historical_coverage": c4_coverage,
        "candidates": candidates,
        "gates": {
            "repository_identity": "pass" if repository_gate else "fail",
            "source_identity": "pass" if source_gate else "fail",
            "diagnostic_readiness": diagnostic_go,
            "fresh_confirmation_readiness": confirmation_go,
        },
        "decision": {
            "status": "partial_go_diagnostic_only",
            "r1_r2_existing_release_diagnostics": (
                "go" if all(diagnostic_go.values()) else "stop"
            ),
            "r4_fresh_confirmation": (
                "go" if all(confirmation_go.values()) else "stop"
            ),
            "umbrella_confirmatory_protocol_freeze": "stop",
            "reason_codes": [
                "published_c4_assets_fully_observed",
                "no_in_scope_official_pair_has_2000_fresh_unique_nulls",
                "semstamp_booksum_model_not_pinned_or_materialized",
            ],
            "required_next_action": (
                "separate an exploratory R1/R2 protocol from a new "
                "score-blind data acquisition protocol; do not reuse any "
                "historical identity for R4"
            ),
        },
        "score_blind_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "raw_text_emitted": False,
            "operations": [
                "git identity and source hashing",
                "Arrow schema and text identity hashing",
                "manifest identity set comparison",
                "duplicate and overlap counting",
            ],
        },
    }

    if not repository_gate or not source_gate:
        output["decision"]["status"] = "fail_closed_identity_mismatch"
        output["decision"]["r1_r2_existing_release_diagnostics"] = "stop"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": output["decision"]["status"],
        "r1_r2": output["decision"]["r1_r2_existing_release_diagnostics"],
        "r4": output["decision"]["r4_fresh_confirmation"],
        "candidate_count": len(candidates),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
