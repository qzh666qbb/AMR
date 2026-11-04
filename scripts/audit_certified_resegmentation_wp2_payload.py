"""Audit pinned WP2 Arrow payloads before any detector model is loaded."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from datasets import Dataset, load_from_disk

from certified_resegmentation_wp2_adapter import (
    make_source_record,
    split_null_cohort,
)


SYSTEMS = {
    "semstamp": {
        "data_repository": "abehou/semstamp-data",
        "data_revision": "bd6e948de2a6fc30f7670dfad347c2b79e252687",
        "repo_dir": "repos/semstamp-data",
        "positive": {
            "path": (
                "c4-semstamp-pegasus-parrot/"
                "semstamp-pegasus-bigram=False"
            ),
            "arrow": "data-00000-of-00001.arrow",
            "git_blob_sha1": "3fb779d1dd78ba284db119054e42e8258b49027a",
            "bytes": 2_226_368,
        },
        "null": {
            "path": "original-c4-texts",
            "arrow": "data-00000-of-00001.arrow",
            "git_blob_sha1": "64afe98aa9694f54837badae0cad3fede5ef5510",
            "bytes": 2_532_336,
        },
    },
    "simmark": {
        "data_repository": "DabiriAghdam/SimMark",
        "data_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "repo_dir": "repos/SimMark",
        "positive": {
            "path": "watermarked/c4/c4-cosine",
            "arrow": "data-00000-of-00001.arrow",
            "git_blob_sha1": "bc8d8f2fa75773b46ca48380ef1ee8bf2077e518",
            "bytes": 1_051_856,
        },
        "null": {
            "path": "human/c4",
            "arrow": "data-00000-of-00001.arrow",
            "git_blob_sha1": "263ef71411062841c43abfb7cf552599ef831d32",
            "bytes": 2_606_472,
        },
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path):
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def git_head(repo_dir):
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_dataset_directory(directory):
    loaded = load_from_disk(str(directory))
    if not isinstance(loaded, Dataset):
        raise TypeError(f"{directory} did not contain one Dataset")
    if "text" not in loaded.column_names:
        raise ValueError(f"{directory} has no text field")
    return loaded


def raw_field_type(value):
    if isinstance(value, str):
        return "string"
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        return "list"
    raise TypeError("text must be a string or list of strings")


def inspect_role(system_id, config, root, role):
    asset = config[role]
    dataset_dir = root / config["repo_dir"] / asset["path"]
    arrow_path = dataset_dir / asset["arrow"]
    if not arrow_path.is_file():
        raise FileNotFoundError(arrow_path)
    observed_size = arrow_path.stat().st_size
    observed_blob = git_blob_sha1(arrow_path)
    if observed_size != asset["bytes"]:
        raise ValueError(f"{role} Arrow byte size does not match")
    if observed_blob != asset["git_blob_sha1"]:
        raise ValueError(f"{role} Arrow Git blob SHA-1 does not match")

    dataset = load_dataset_directory(dataset_dir)
    rows = []
    type_counts = Counter()
    empty_raw_count = 0
    total_utf8_bytes = 0
    for row_index, row in enumerate(dataset):
        value = row["text"]
        field_type = raw_field_type(value)
        type_counts[field_type] += 1
        if field_type == "string":
            empty_raw_count += int(not value)
            total_utf8_bytes += len(value.encode("utf-8"))
        else:
            empty_raw_count += int(not value)
            total_utf8_bytes += sum(
                len(item.encode("utf-8")) for item in value
            )
        record = make_source_record(
            system_id=system_id,
            data_revision=config["data_revision"],
            row_index=row_index,
            raw_value=value,
            source_repository=config["data_repository"],
            relative_path=f"{asset['path']}/{asset['arrow']}",
        )
        rows.append({
            "record": record,
            "manifest": {
                "system_id": system_id,
                "data_revision": config["data_revision"],
                "source_repository": config["data_repository"],
                "relative_path": record.relative_path,
                "row_index": row_index,
                "role": role,
                "raw_field_type": field_type,
                "raw_text_sha256": record.raw_text_sha256,
            },
        })

    hash_counts = Counter(
        item["record"].raw_text_sha256 for item in rows
    )
    return {
        "dataset": dataset,
        "rows": rows,
        "summary": {
            "directory": str(dataset_dir),
            "arrow_path": str(arrow_path),
            "arrow_bytes": observed_size,
            "arrow_git_blob_sha1": observed_blob,
            "arrow_sha256": sha256(arrow_path),
            "row_count": len(dataset),
            "column_names": list(dataset.column_names),
            "features": dataset.features.to_dict(),
            "arrow_schema": str(dataset.data.schema),
            "raw_field_type_counts": dict(sorted(type_counts.items())),
            "empty_raw_count": empty_raw_count,
            "total_raw_utf8_bytes": total_utf8_bytes,
            "unique_raw_hash_count": len(hash_counts),
            "duplicate_row_count": sum(
                count - 1 for count in hash_counts.values()
            ),
            "duplicate_hash_count": sum(
                count > 1 for count in hash_counts.values()
            ),
        },
    }


def assign_statuses(split, positive_rows, null_rows):
    cohort_by_record = {}
    for cohort, records in (
        ("detector_fit", split.detector_fit),
        ("certificate_calibration", split.certificate_calibration),
        ("independent_test", split.independent_test),
    ):
        for record in records:
            cohort_by_record[record] = cohort
    positive_retained = frozenset(split.positive_retained)
    positive_overlap = frozenset(split.positive_overlap_excluded)

    output = []
    for item in null_rows:
        record = item["record"]
        manifest = dict(item["manifest"])
        if record in cohort_by_record:
            manifest["identity_status"] = "retained_unique_null"
            manifest["cohort"] = cohort_by_record[record]
        else:
            manifest["identity_status"] = "excluded_duplicate_null"
            manifest["cohort"] = None
        output.append(manifest)
    for item in positive_rows:
        record = item["record"]
        manifest = dict(item["manifest"])
        if record in positive_overlap:
            manifest["identity_status"] = "excluded_positive_null_overlap"
            manifest["cohort"] = None
        elif record in positive_retained:
            manifest["identity_status"] = "retained_positive"
            manifest["cohort"] = "positive"
        else:
            raise AssertionError("positive record has no identity status")
        output.append(manifest)
    return output


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(
                row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            handle.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
    return digest.hexdigest()


def audit_system(root, system_id, row_manifest):
    config = SYSTEMS[system_id]
    repo_dir = root / config["repo_dir"]
    observed_head = git_head(repo_dir)
    if observed_head != config["data_revision"]:
        raise ValueError("data repository revision does not match")

    positive = inspect_role(system_id, config, root, "positive")
    null = inspect_role(system_id, config, root, "null")
    positive_records = [item["record"] for item in positive["rows"]]
    null_records = [item["record"] for item in null["rows"]]
    split = split_null_cohort(
        system_id=system_id,
        data_revision=config["data_revision"],
        null_records=null_records,
        positive_records=positive_records,
    )
    rows = assign_statuses(split, positive["rows"], null["rows"])
    manifest_sha256 = write_jsonl(row_manifest, rows)

    null_hashes = {
        item["record"].raw_text_sha256 for item in null["rows"]
    }
    positive_hashes = {
        item["record"].raw_text_sha256 for item in positive["rows"]
    }
    overlap = null_hashes & positive_hashes
    return {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "pre_score_arrow_identity_audit",
        "system_id": system_id,
        "network_access_by_script": False,
        "source": {
            "repository": config["data_repository"],
            "expected_revision": config["data_revision"],
            "observed_revision": observed_head,
        },
        "positive": positive["summary"],
        "null": null["summary"],
        "cross_role_raw_hash_overlap_count": len(overlap),
        "cohort_split_before_eligibility": {
            "detector_fit": len(split.detector_fit),
            "certificate_calibration": len(
                split.certificate_calibration
            ),
            "independent_test": len(split.independent_test),
            "positive_retained": len(split.positive_retained),
            "positive_overlap_excluded": len(
                split.positive_overlap_excluded
            ),
            "null_duplicates_excluded": (
                split.attrition.null_duplicate_count
            ),
            "note": (
                "Identities are frozen before sentence/length eligibility; "
                "excluded rows will not be reassigned."
            ),
        },
        "row_identity_manifest": {
            "path": str(row_manifest),
            "rows": len(rows),
            "sha256": manifest_sha256,
            "contains_raw_text": False,
        },
        "gates": {
            "revision_match": True,
            "arrow_blob_identity_match": True,
            "datasets_load_from_disk": True,
            "text_field_present": True,
            "all_text_values_supported": True,
            "row_identity_frozen": True,
            "detector_score_accessed": False,
            "model_loaded": False,
            "eligibility_evaluated": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "datasets": __import__("datasets").__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "numpy": __import__("numpy").__version__,
            "nltk": __import__("nltk").__version__,
        },
        "execution_attestation": {
            "watermark_detector_calls": 0,
            "model_loads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_system(args.root, args.system, args.row_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        audit, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
