"""Run the score-blind WP2-v2 lossless/native-context design audit."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess

from certified_resegmentation_wp2_adapter import (
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
        "tokenizer_files": (
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    },
    "simmark": {
        "data_revision": "8b91c97cc1d8cc4a219917cdff628ae750f8dd5a",
        "data_repo_dir": "repos/SimMark",
        "positive_path": "watermarked/c4/c4-cosine",
        "null_path": "human/c4",
        "nltk_version": "3.9.1",
        "tokenizer_files": (
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
}

CANDIDATES = (
    "V1_STRICT_5_35",
    "W_LOSSLESS_5_35",
    "L_STRICT_NATIVE_512",
    "V2_LOSSLESS_NATIVE_512",
)
PRIMARY_CANDIDATE = "V2_LOSSLESS_NATIVE_512"
INSTRUCTION = "Represent the sentence for cosine similarity:"
MIN_LEGACY_ATOMS = 5
MAX_LEGACY_ATOMS = 35
NATIVE_TOKEN_LIMIT = 512
EAGER_PAIR_LIMIT = 50_000_000
_ATOM_RE = re.compile(r"\S+")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head(path):
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_dataset(path):
    from datasets import Dataset, load_from_disk

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


def materialize_units(value, sentence_splitter):
    if isinstance(value, str):
        units = sentence_splitter(value)
        if not isinstance(units, list):
            raise TypeError("sentence splitter must return a list")
        field_type = "string"
    elif isinstance(value, list):
        units = value
        field_type = "list"
    else:
        raise TypeError("raw value must be a string or list")
    if not all(isinstance(unit, str) for unit in units):
        raise TypeError("every P0 unit must be a string")
    return field_type, tuple(units)


def official_sentence_splitter(value):
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(value)


def unit_layout(units):
    layouts = []
    for unit in units:
        matches = tuple(_ATOM_RE.finditer(unit))
        layouts.append({
            "atom_count": len(matches),
            "leading_whitespace": (
                len(unit[:matches[0].start()]) if matches else None
            ),
            "trailing_whitespace": (
                len(unit[matches[-1].end():]) if matches else None
            ),
        })
    return layouts


def p0_codes(field_type, units, layouts):
    base_codes = []
    strict_codes = []
    lossless_codes = []
    if not units:
        base_codes.append("empty_p0")
    if any(not unit or layout["atom_count"] == 0
           for unit, layout in zip(units, layouts)):
        base_codes.append("empty_or_atomless_unit")
    if any(unit != unit.strip() for unit in units):
        strict_codes.append("unit_edge_whitespace")

    missing_cross_unit_whitespace = 0
    if field_type == "list" and not base_codes:
        for left, right in zip(layouts, layouts[1:]):
            if (
                left["trailing_whitespace"] == 0
                and right["leading_whitespace"] == 0
            ):
                missing_cross_unit_whitespace += 1
    if missing_cross_unit_whitespace:
        lossless_codes.append("missing_cross_unit_whitespace")
    return (
        base_codes,
        strict_codes,
        lossless_codes,
        missing_cross_unit_whitespace,
    )


def official_tokenizer_text(system_id, unit, unit_index):
    if system_id == "semstamp":
        return unit.strip()
    preprocessed = unit if unit_index == 0 else " " + unit
    return (INSTRUCTION + preprocessed).strip()


def untruncated_token_count(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("tokenizer unexpectedly returned multiple rows")
        input_ids = input_ids[0]
    return len(input_ids)


def candidate_codes(
    base_codes,
    strict_codes,
    lossless_codes,
    unit_count,
    atom_counts,
    native_token_counts,
):
    structural = list(base_codes)
    if unit_count < 2:
        structural.append("unit_count_below_2")
    legacy = []
    native = []
    if not base_codes:
        if min(atom_counts) < MIN_LEGACY_ATOMS:
            legacy.append("unit_below_5_atoms")
        if max(atom_counts) > MAX_LEGACY_ATOMS:
            legacy.append("unit_above_35_atoms")
        if max(native_token_counts) > NATIVE_TOKEN_LIMIT:
            native.append("unit_above_512_native_tokens")

    return {
        "V1_STRICT_5_35": sorted(set(
            structural + strict_codes + legacy
        )),
        "W_LOSSLESS_5_35": sorted(set(
            structural + lossless_codes + legacy
        )),
        "L_STRICT_NATIVE_512": sorted(set(
            structural + strict_codes + native
        )),
        "V2_LOSSLESS_NATIVE_512": sorted(set(
            structural + lossless_codes + native
        )),
    }


def threat_size_upper(atom_count, unit_count, radius):
    original = unit_count - 1
    nonoriginal = atom_count - unit_count
    return sum(
        math.comb(original, moved) * math.comb(nonoriginal, moved)
        for moved in range(radius + 1)
        if moved <= original and moved <= nonoriginal
    )


def audit_identity(item, system_id, tokenizer):
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
        field_type, units = materialize_units(
            item["raw_value"], official_sentence_splitter
        )
    except (TypeError, ValueError) as error:
        codes = {
            candidate: ["invalid_p0"] for candidate in CANDIDATES
        }
        output.update({
            "p0_layout_sha256": None,
            "unit_count": None,
            "atom_count": None,
            "min_unit_atoms": None,
            "max_unit_atoms": None,
            "min_native_tokens": None,
            "max_native_tokens": None,
            "units_with_leading_whitespace": None,
            "units_with_trailing_whitespace": None,
            "missing_cross_unit_whitespace": None,
            "candidate_eligible": {
                candidate: False for candidate in CANDIDATES
            },
            "candidate_exclusion_codes": codes,
            "p0_error_type": type(error).__name__,
        })
        return output

    layouts = unit_layout(units)
    (
        base_codes,
        strict_codes,
        lossless_codes,
        missing_cross_unit_whitespace,
    ) = p0_codes(field_type, units, layouts)
    atom_counts = [layout["atom_count"] for layout in layouts]
    native_counts = []
    if not base_codes:
        native_counts = [
            untruncated_token_count(
                tokenizer,
                official_tokenizer_text(system_id, unit, index),
            )
            for index, unit in enumerate(units)
        ]
    codes = candidate_codes(
        base_codes=base_codes,
        strict_codes=strict_codes,
        lossless_codes=lossless_codes,
        unit_count=len(units),
        atom_counts=atom_counts,
        native_token_counts=native_counts,
    )
    atom_count = sum(atom_counts)
    eager_pair_upper = (
        math.comb(atom_count + 1, 3) if atom_count >= 2 else 0
    )
    output.update({
        "p0_layout_sha256": sha256_json({
            "raw_field_type": field_type,
            "units": units,
        }),
        "unit_count": len(units),
        "atom_count": atom_count,
        "min_unit_atoms": min(atom_counts) if atom_counts else None,
        "max_unit_atoms": max(atom_counts) if atom_counts else None,
        "min_native_tokens": min(native_counts) if native_counts else None,
        "max_native_tokens": max(native_counts) if native_counts else None,
        "units_with_leading_whitespace": sum(
            bool(layout["leading_whitespace"]) for layout in layouts
        ),
        "units_with_trailing_whitespace": sum(
            bool(layout["trailing_whitespace"]) for layout in layouts
        ),
        "missing_cross_unit_whitespace": missing_cross_unit_whitespace,
        "candidate_eligible": {
            candidate: not candidate_codes
            for candidate, candidate_codes in codes.items()
        },
        "candidate_exclusion_codes": codes,
        "eager_pair_key_upper_bound": eager_pair_upper,
        "threat_size_upper_bound": {
            str(radius): threat_size_upper(
                atom_count, len(units), radius
            )
            for radius in range(4)
        },
        "p0_error_type": None,
    })
    return output


def nearest_rank_quantiles(values):
    if not values:
        return None
    ordered = sorted(values)

    def at(probability):
        index = math.ceil(probability * len(ordered)) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def summarize(rows):
    cohorts = {}
    cohort_order = (
        "detector_fit",
        "certificate_calibration",
        "independent_test",
        "positive",
    )
    for cohort in cohort_order:
        selected = [row for row in rows if row["cohort"] == cohort]
        candidates = {}
        for candidate in CANDIDATES:
            reason_counts = Counter(
                code
                for row in selected
                for code in row["candidate_exclusion_codes"][candidate]
            )
            candidates[candidate] = {
                "eligible": sum(
                    row["candidate_eligible"][candidate] for row in selected
                ),
                "excluded": sum(
                    not row["candidate_eligible"][candidate]
                    for row in selected
                ),
                "exclusion_code_counts": dict(sorted(reason_counts.items())),
            }
        patterns = Counter(
            "|".join(
                candidate
                for candidate in CANDIDATES
                if row["candidate_eligible"][candidate]
            ) or "NONE"
            for row in selected
        )
        analyzable = [
            row for row in selected if row["atom_count"] is not None
        ]
        primary = [
            row for row in selected
            if row["candidate_eligible"][PRIMARY_CANDIDATE]
        ]
        cohorts[cohort] = {
            "selected": len(selected),
            "candidates": candidates,
            "eligibility_pattern_counts": dict(sorted(patterns.items())),
            "distribution": {
                key: nearest_rank_quantiles([
                    row[key] for row in analyzable if row[key] is not None
                ])
                for key in (
                    "atom_count",
                    "unit_count",
                    "min_unit_atoms",
                    "max_unit_atoms",
                    "max_native_tokens",
                )
            },
            "primary_complexity": {
                "eligible_rows": len(primary),
                "eager_pair_key_upper_bound": nearest_rank_quantiles([
                    row["eager_pair_key_upper_bound"] for row in primary
                ]),
                "threat_size_upper_bound_k3": nearest_rank_quantiles([
                    row["threat_size_upper_bound"]["3"] for row in primary
                ]),
                "rows_above_eager_pair_limit": sum(
                    row["eager_pair_key_upper_bound"] > EAGER_PAIR_LIMIT
                    for row in primary
                ),
            },
        }
    return cohorts


def primary_gates(system_id, cohorts):
    gates = {
        "certificate_calibration_eligible_ge_399": (
            cohorts["certificate_calibration"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 399
        ),
        "independent_test_eligible_ge_400": (
            cohorts["independent_test"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 400
        ),
        "positive_eligible_ge_200": (
            cohorts["positive"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 200
        ),
    }
    if system_id == "simmark":
        gates["detector_fit_eligible_ge_200"] = (
            cohorts["detector_fit"]["candidates"][
                PRIMARY_CANDIDATE
            ]["eligible"] >= 200
        )
    return gates


def write_jsonl(path, rows):
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_tokenizer(model_dir, system_id):
    from transformers import AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.model_max_length != NATIVE_TOKEN_LIMIT:
        raise ValueError(
            f"tokenizer max length is {tokenizer.model_max_length}, not 512"
        )
    expected_class = {
        "semstamp": "MPNetTokenizerFast",
        "simmark": "T5TokenizerFast",
    }[system_id]
    if type(tokenizer).__name__ != expected_class:
        raise ValueError(
            f"tokenizer class is {type(tokenizer).__name__}, "
            f"not {expected_class}"
        )
    return tokenizer


def tokenizer_manifest(model_dir, system_id):
    output = {}
    for name in SYSTEMS[system_id]["tokenizer_files"]:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = sha256_file(path)
    return output


def main():
    import nltk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    args = parser.parse_args()

    config = SYSTEMS[args.system]
    if nltk.__version__ != config["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(args.nltk_data))
    tokenizer_files = tokenizer_manifest(args.model_dir, args.system)
    tokenizer = load_tokenizer(args.model_dir, args.system)
    identities, split, revision = frozen_identities(args.root, args.system)
    rows = [
        audit_identity(item, args.system, tokenizer) for item in identities
    ]
    cohorts = summarize(rows)
    gates = primary_gates(args.system, cohorts)
    primary_rows = [
        row for row in rows
        if row["candidate_eligible"][PRIMARY_CANDIDATE]
    ]
    eager_limit_exceeded = any(
        row["eager_pair_key_upper_bound"] > EAGER_PAIR_LIMIT
        for row in primary_rows
    )

    args.row_manifest.parent.mkdir(parents=True, exist_ok=True)
    row_sha256 = write_jsonl(args.row_manifest, rows)
    output = {
        "schema_version": 1,
        "created_on": "2026-09-22",
        "audit_mode": "score_blind_wp2_v2_design_feasibility",
        "system_id": args.system,
        "status": "eligibility_go" if all(gates.values()) else "stop",
        "data_revision": revision,
        "primary_candidate": PRIMARY_CANDIDATE,
        "candidate_order": list(CANDIDATES),
        "parameters": {
            "atom_pattern": r"\S+",
            "legacy_min_unit_atoms": MIN_LEGACY_ATOMS,
            "legacy_max_unit_atoms": MAX_LEGACY_ATOMS,
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "min_unit_count": 2,
            "null_split": "20%/40%/remainder",
            "replacement_after_attrition": False,
            "simmark_instruction": (
                INSTRUCTION if args.system == "simmark" else None
            ),
            "eager_pair_key_limit": EAGER_PAIR_LIMIT,
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
        "primary_gates": gates,
        "implementation_gate": {
            "eager_pair_limit_exceeded": eager_limit_exceeded,
            "current_eager_build_pair_scores_status": (
                "stop_sparse_adapter_required"
                if eager_limit_exceeded
                else "pass"
            ),
        },
        "row_manifest": {
            "path": args.row_manifest.name,
            "rows": len(rows),
            "sha256": row_sha256,
            "contains_raw_text": False,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "model_max_length": tokenizer.model_max_length,
            "local_files_only": True,
            "file_sha256": tokenizer_files,
            "unit_calls": sum(
                row["unit_count"] or 0 for row in rows
                if row["min_native_tokens"] is not None
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nltk": nltk.__version__,
        },
        "execution_attestation": {
            "model_loads": 0,
            "watermark_detector_calls": 0,
            "detector_score_reads": 0,
            "watermark_generation_calls": 0,
            "provider_api_calls": 0,
            "tokenizer_loads": 1,
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
        "primary_gates": gates,
        "implementation_gate": output["implementation_gate"],
        "row_manifest_sha256": row_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
