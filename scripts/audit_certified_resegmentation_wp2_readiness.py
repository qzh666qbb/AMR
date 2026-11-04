#!/usr/bin/env python3
"""Offline readiness audit for the certified-resegmentation WP2 protocol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "datasets",
    "nltk",
    "numpy",
    "scikit-learn",
    "sentence-transformers",
    "pyarrow",
)

OFFICIAL_PACKAGE_LOCKS = {
    "semstamp": {
        "datasets": "2.19.1",
        "nltk": "3.8.1",
        "numpy": "1.26.4",
        "pyarrow": "16.1.0",
        "scikit-learn": "1.5.0",
        "sentence-transformers": "3.0.1",
        "torch": "2.3.0",
        "transformers": "4.35.2",
    },
    "simmark": {
        "datasets": "3.1.0",
        "nltk": "3.9.1",
        "numpy": "1.24.4",
        "pyarrow": "17.0.0",
        "scikit-learn": "1.2.2",
        "sentence-transformers": "3.2.1",
        "torch": "2.4.1",
        "transformers": "4.46.3",
    },
}

CODE_REPOSITORIES = {
    "semstamp": {
        "repository": "bohanhou14/SemStamp",
        "revision": "97db73d11fd80f376a02b0a604d500627622f7e6",
        "license": None,
        "detection_py_git_blob_sha1": "bb811886d76084e562e2eb9acbbd1ec9bb7c04b9",
        "detection_utils_py_git_blob_sha1": (
            "ab6977fc558260c38a56f87bda3096835da00a42"
        ),
        "sampling_lsh_utils_py_git_blob_sha1": (
            "86e1eff66f46f33d6fd6da8dd34ac0dfca12aadf"
        ),
    },
    "simmark": {
        "repository": "DabiriAghdam/SimMark",
        "revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "license": "MIT",
        "detection_py_git_blob_sha1": "ae68070189a7184ca02e14ef8ef14fb58b233bf4",
    },
}

MODEL_SNAPSHOTS = {
    "semstamp": {
        "repository": "AbeHou/SemStamp-c4-sbert",
        "revision": "a77dce8530e100ac1101981e4be390546c1cb282",
        "snapshot_bytes": 438_964_159,
        "used_storage_bytes": 875_990_784,
        "files": 8,
        "license": None,
        "main_weight": {
            "path": "pytorch_model.bin",
            "bytes": 438_014_769,
            "lfs_sha256": (
                "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4a6a2e9dee"
            ),
        },
    },
    "simmark": {
        "repository": "hkunlp/instructor-large",
        "revision": "54e5ffb8d484de506e59443b07dc819fb15c7233",
        "snapshot_bytes": 1_346_259_440,
        "used_storage_bytes": 6_712_370_802,
        "files": 14,
        "license": "apache-2.0",
        "main_weight": {
            "path": "pytorch_model.bin",
            "bytes": 1_339_823_867,
            "lfs_sha256": (
                "fb7f63752cf10103be938bc50dfad0b6fa1e63bc67b963471fc827838d9bbb41"
            ),
        },
        "dense_weight": {
            "path": "2_Dense/pytorch_model.bin",
            "bytes": 3_146_603,
            "lfs_sha256": (
                "a6e82cba0876dacbadccdea565c9e19e29848d994d23968bd1343b8f0f762bdc"
            ),
        },
    },
}

DATA_RELEASES = {
    "semstamp": {
        "repository": "abehou/semstamp-data",
        "revision": "bd6e948de2a6fc30f7670dfad347c2b79e252687",
        "all_arrow_files": 30,
        "all_arrow_bytes": 59_889_432,
        "repository_license": None,
        "selected_load_from_disk_bytes": 4_759_718,
        "selected_assets": [
            {
                "role": "positive",
                "path": (
                    "c4-semstamp-pegasus-parrot/"
                    "semstamp-pegasus-bigram=False/data-00000-of-00001.arrow"
                ),
                "git_blob_sha1": "3fb779d1dd78ba284db119054e42e8258b49027a",
                "bytes": 2_226_368,
            },
            {
                "role": "null",
                "path": "original-c4-texts/data-00000-of-00001.arrow",
                "git_blob_sha1": "64afe98aa9694f54837badae0cad3fede5ef5510",
                "bytes": 2_532_336,
            },
        ],
    },
    "simmark": {
        "repository": "DabiriAghdam/SimMark",
        "revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "all_arrow_files": 45,
        "all_arrow_bytes": 87_365_416,
        "repository_license": "MIT",
        "dataset_metadata_license_field": "",
        "selected_load_from_disk_bytes": 3_659_152,
        "selected_assets": [
            {
                "role": "positive",
                "path": "watermarked/c4/c4-cosine/data-00000-of-00001.arrow",
                "git_blob_sha1": "bc8d8f2fa75773b46ca48380ef1ee8bf2077e518",
                "bytes": 1_051_856,
            },
            {
                "role": "null",
                "path": "human/c4/data-00000-of-00001.arrow",
                "git_blob_sha1": "263ef71411062841c43abfb7cf552599ef831d32",
                "bytes": 2_606_472,
            },
        ],
    },
}


def interpreter_probe(python_path: Path) -> dict[str, Any]:
    if not python_path.exists():
        return {"path": str(python_path), "exists": False}

    probe = (
        "import importlib.metadata as m,json,platform;"
        f"names={PACKAGE_NAMES!r};"
        "versions={};"
        "\nfor name in names:"
        "\n try: versions[name]=m.version(name)"
        "\n except m.PackageNotFoundError: versions[name]=None"
        "\ntry:"
        "\n import torch"
        "\n cuda=bool(torch.cuda.is_available())"
        "\n mps=bool(getattr(torch.backends,'mps',None) and "
        "torch.backends.mps.is_available())"
        "\nexcept Exception:"
        "\n cuda=False; mps=False"
        "\nprint(json.dumps({'python':platform.python_version(),"
        "'packages':versions,'cuda_available':cuda,'mps_available':mps},"
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python_path), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        return {
            "path": str(python_path),
            "exists": True,
            "probe_ok": False,
            "stderr": completed.stderr.strip(),
        }
    result = json.loads(completed.stdout)
    result.update(path=str(python_path), exists=True, probe_ok=True)
    return result


def exact_lock_match(
    environment: dict[str, Any], expected: dict[str, str]
) -> bool:
    if not environment.get("probe_ok"):
        return False
    observed = environment["packages"]
    return all(observed.get(name) == version for name, version in expected.items())


def cache_snapshot_path(repository: str, revision: str) -> Path:
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    owner, name = repository.split("/", 1)
    return hf_home / "hub" / f"models--{owner}--{name}" / "snapshots" / revision


def build_audit() -> dict[str, Any]:
    model_bytes = sum(item["snapshot_bytes"] for item in MODEL_SNAPSHOTS.values())
    all_arrow_bytes = sum(item["all_arrow_bytes"] for item in DATA_RELEASES.values())
    selected_data_bytes = sum(
        item["selected_load_from_disk_bytes"] for item in DATA_RELEASES.values()
    )
    minimum_payload_bytes = model_bytes + selected_data_bytes
    broad_payload_bytes = model_bytes + all_arrow_bytes

    environments = {
        "system_python": interpreter_probe(Path(sys.executable)),
        "venv": interpreter_probe(ROOT / ".venv" / "bin" / "python"),
        "venv_swan_positive": interpreter_probe(
            ROOT / ".venv-swan-positive" / "bin" / "python"
        ),
    }
    target_cache = {
        system: {
            "path": str(
                cache_snapshot_path(item["repository"], item["revision"])
            ),
            "exists": cache_snapshot_path(
                item["repository"], item["revision"]
            ).is_dir(),
        }
        for system, item in MODEL_SNAPSHOTS.items()
    }
    disk = shutil.disk_usage(ROOT)

    any_semstamp_lock = any(
        exact_lock_match(env, OFFICIAL_PACKAGE_LOCKS["semstamp"])
        for env in environments.values()
    )
    any_simmark_lock = any(
        exact_lock_match(env, OFFICIAL_PACKAGE_LOCKS["simmark"])
        for env in environments.values()
    )
    cuda_available = any(
        bool(env.get("cuda_available")) for env in environments.values()
    )
    models_cached = all(item["exists"] for item in target_cache.values())
    storage_reserve_bytes = broad_payload_bytes * 2 + 20 * 1024**3

    gates = {
        "protocol_frozen": True,
        "storage_reserve_available": disk.free >= storage_reserve_bytes,
        "target_model_snapshots_cached": models_cached,
        "semstamp_exact_dependency_lock_available": any_semstamp_lock,
        "simmark_exact_dependency_lock_available": any_simmark_lock,
        "cuda_available_for_semstamp_official_path": cuda_available,
        "semstamp_code_data_and_model_license_resolved": False,
        "simmark_dataset_use_basis_resolved": False,
        "payload_schema_row_and_overlap_audit_complete": False,
        "direct_unit_list_adapter_implemented_and_k0_verified": False,
        "certificate_calibration_and_test_cohorts_materialized": False,
        "explicit_payload_download_authorization_present": False,
        "explicit_detector_execution_authorization_present": False,
    }
    blockers = [name for name, passed in gates.items() if not passed]

    return {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "offline_local_readiness_only",
        "network_access_by_script": False,
        "scope": {
            "systems": [
                "SemStamp LSH on published C4 Pegasus clean positives",
                "SimMark cosine on published C4 clean positives",
            ],
            "generation": "forbidden",
            "model_or_dataset_payload_download": "not_authorized",
            "detector_execution": "not_authorized",
        },
        "pinned_metadata": {
            "code_repositories": CODE_REPOSITORIES,
            "model_snapshots": MODEL_SNAPSHOTS,
            "data_releases": DATA_RELEASES,
            "official_package_locks": OFFICIAL_PACKAGE_LOCKS,
            "metadata_source": (
                "Pinned GitHub recursive-tree and Hugging Face model API "
                "responses observed on 2026-09-22; no payload content fetched."
            ),
        },
        "resource_arithmetic": {
            "model_snapshot_bytes": model_bytes,
            "selected_dataset_load_from_disk_bytes": selected_data_bytes,
            "minimum_selected_payload_bytes": minimum_payload_bytes,
            "all_published_arrow_bytes": all_arrow_bytes,
            "broad_model_plus_all_arrow_bytes": broad_payload_bytes,
            "huggingface_used_storage_bytes_not_transfer_budget": sum(
                item["used_storage_bytes"] for item in MODEL_SNAPSHOTS.values()
            ),
            "storage_reserve_rule_bytes": storage_reserve_bytes,
            "note": (
                "Content-byte lower bounds exclude Git/HTTP overhead, package "
                "wheels, runtime caches, and derived detector caches."
            ),
        },
        "local_host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "interpreters": environments,
            "target_model_cache": target_cache,
        },
        "gates": gates,
        "blocking_gates": blockers,
        "system_decisions": {
            "semstamp_current_host": "no_go",
            "simmark_current_host": "no_go",
            "storage_capacity": (
                "pass" if gates["storage_reserve_available"] else "no_go"
            ),
            "wp2_current_host": "no_go",
            "wp2_authorized": False,
            "overall_certified_resegmentation_project": "conditional_go",
        },
        "execution_attestation": {
            "provider_api_calls": 0,
            "watermark_generation_calls": 0,
            "watermark_detector_calls": 0,
            "llm_judge_calls": 0,
            "amr_parser_calls": 0,
            "s2match_calls": 0,
            "model_payload_downloads": 0,
            "dataset_payload_downloads": 0,
            "external_gpu_jobs": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_audit()
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
