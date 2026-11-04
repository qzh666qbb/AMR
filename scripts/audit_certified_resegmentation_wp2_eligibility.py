"""Run the score-blind WP2 P0 and unit-length eligibility audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

from datasets import Dataset, load_from_disk
import nltk
from nltk.tokenize import sent_tokenize

from certified_resegmentation_wp2_adapter import (
    canonicalize,
    cohort_sort_hash,
    make_source_record,
    split_null_cohort,
)


SYSTEMS = {
    "semstamp": {
        "data_revision": "bd6e948de2a6fc30f7670dfad347c2b79e252687",
        "data_repo_dir": "repos/semstamp-data",
        "positive_path": (
            "c4-semstamp-pegasus-parrot/"
            "semstamp-pegasus-bigram=False"
        ),
        "null_path": "original-c4-texts",
        "nltk_version": "3.8.1",
    },
    "simmark": {
        "data_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "data_repo_dir": "repos/SimMark",
        "positive_path": "watermarked/c4/c4-cosine",
        "null_path": "human/c4",
        "nltk_version": "3.9.1",
    },
}

MIN_UNIT_ATOMS = 5
MAX_UNIT_ATOMS = 35
_ATOM_RE = re.compile(r"\S+")


def git_head(path):
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_dataset(path):
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{path} did not contain one Dataset")
    if "text" not in dataset.column_names:
        raise ValueError(f"{path} has no text field")
    return dataset


def records_for(dataset, system_id, config, role):
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


def frozen_identities(root, system_id):
    config = SYSTEMS[system_id]
    data_root = root / config["data_repo_dir"]
    observed_revision = git_head(data_root)
    if observed_revision != config["data_revision"]:
        raise ValueError("data revision mismatch")

    null_dataset = load_dataset(data_root / config["null_path"])
    positive_dataset = load_dataset(data_root / config["positive_path"])
    null_records = records_for(null_dataset, system_id, config, "null")
    positive_records = records_for(
        positive_dataset, system_id, config, "positive"
    )
    split = split_null_cohort(
        system_id=system_id,
        data_revision=config["data_revision"],
        null_records=null_records,
        positive_records=positive_records,
    )
    groups = (
        ("detector_fit", "null", split.detector_fit),
        (
            "certificate_calibration",
            "null",
            split.certificate_calibration,
        ),
        ("independent_test", "null", split.independent_test),
        ("positive", "positive", split.positive_retained),
    )
    datasets = {"null": null_dataset, "positive": positive_dataset}
    identities = []
    for cohort, role, records in groups:
        for record in records:
            identities.append({
                "cohort": cohort,
                "role": role,
                "record": record,
                "raw_value": datasets[role][record.row_index]["text"],
            })
    return identities, split, observed_revision


def audit_identity(item):
    record = item["record"]
    output = {
        "cohort": item["cohort"],
        "role": item["role"],
        "row_index": record.row_index,
        "cohort_sort_sha256": cohort_sort_hash(record),
        "raw_text_sha256": record.raw_text_sha256,
        "raw_field_type": record.raw_field_type,
        "contains_raw_text": False,
    }
    try:
        document = canonicalize(
            item["raw_value"],
            sentence_splitter=(
                sent_tokenize
                if isinstance(item["raw_value"], str)
                else None
            ),
        )
    except ValueError as error:
        output.update({
            "eligible": False,
            "exclusion_codes": ["invalid_p0"],
            "p0_error": str(error),
            "unit_count": None,
            "atom_count": None,
            "min_unit_atoms": None,
            "max_unit_atoms": None,
        })
        return output

    unit_atom_counts = tuple(
        len(_ATOM_RE.findall(unit)) for unit in document.original_units
    )
    exclusion_codes = []
    if document.unit_count < 2:
        exclusion_codes.append("unit_count_below_2")
    if min(unit_atom_counts) < MIN_UNIT_ATOMS:
        exclusion_codes.append("unit_below_5_atoms")
    if max(unit_atom_counts) > MAX_UNIT_ATOMS:
        exclusion_codes.append("unit_above_35_atoms")
    output.update({
        "eligible": not exclusion_codes,
        "exclusion_codes": exclusion_codes,
        "p0_error": None,
        "unit_count": document.unit_count,
        "atom_count": document.atom_count,
        "min_unit_atoms": min(unit_atom_counts),
        "max_unit_atoms": max(unit_atom_counts),
        "canonical_tape_sha256": document.canonical_tape_sha256,
    })
    return output


def summarize(rows):
    cohorts = {}
    for cohort in (
        "detector_fit",
        "certificate_calibration",
        "independent_test",
        "positive",
    ):
        selected = [row for row in rows if row["cohort"] == cohort]
        reason_counts = Counter(
            code
            for row in selected
            for code in row["exclusion_codes"]
        )
        cohorts[cohort] = {
            "selected": len(selected),
            "eligible": sum(row["eligible"] for row in selected),
            "excluded": sum(not row["eligible"] for row in selected),
            "exclusion_code_counts": dict(sorted(reason_counts.items())),
        }
    return cohorts


def write_jsonl(path, rows):
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(
                row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            handle.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    args = parser.parse_args()

    config = SYSTEMS[args.system]
    if nltk.__version__ != config["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(args.nltk_data))
    identities, split, revision = frozen_identities(args.root, args.system)
    rows = [audit_identity(item) for item in identities]
    cohorts = summarize(rows)

    gates = {
        "certificate_calibration_eligible_ge_399": (
            cohorts["certificate_calibration"]["eligible"] >= 399
        ),
        "independent_test_eligible_ge_400": (
            cohorts["independent_test"]["eligible"] >= 400
        ),
        "positive_eligible_ge_200": (
            cohorts["positive"]["eligible"] >= 200
        ),
    }
    if args.system == "simmark":
        gates["detector_fit_eligible_ge_200"] = (
            cohorts["detector_fit"]["eligible"] >= 200
        )

    args.row_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = write_jsonl(args.row_manifest, rows)
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "score_blind_full_p0_length_eligibility",
        "system_id": args.system,
        "status": "pass" if all(gates.values()) else "stop",
        "data_revision": revision,
        "parameters": {
            "atom_pattern": r"\S+",
            "min_unit_atoms": MIN_UNIT_ATOMS,
            "max_unit_atoms": MAX_UNIT_ATOMS,
            "min_unit_count": 2,
            "null_split": "20%/40%/remainder",
            "replacement_after_attrition": False,
        },
        "identity_attrition": {
            "null_input_count": split.attrition.null_input_count,
            "null_unique_count": split.attrition.null_unique_count,
            "null_duplicate_count": split.attrition.null_duplicate_count,
            "positive_input_count": split.attrition.positive_input_count,
            "positive_retained_count": (
                split.attrition.positive_retained_count
            ),
            "positive_overlap_count": split.attrition.positive_overlap_count,
        },
        "cohorts": cohorts,
        "gates": gates,
        "row_manifest": {
            "path": str(args.row_manifest),
            "rows": len(rows),
            "sha256": manifest_sha256,
            "contains_raw_text": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nltk": nltk.__version__,
        },
        "execution_attestation": {
            "model_loads": 0,
            "watermark_detector_calls": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
        },
        "contains_raw_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "system_id": args.system,
        "status": output["status"],
        "cohorts": cohorts,
        "gates": gates,
        "row_manifest_sha256": manifest_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
