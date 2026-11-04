"""Audit and allocate fresh WP2-v12 C4 assets without detector scores."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


CREATED_ON = "2026-09-25"
ORDER_SALT = "WP2-v12-c4-source-order-v1"
INSTRUCTION = "Represent the sentence for cosine similarity:"
NATIVE_TOKEN_LIMIT = 512

SOURCE = {
    "repository": "allenai/c4",
    "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "config": "realnewslike",
    "split": "validation",
    "relative_path": (
        "realnewslike/c4-validation.00000-of-00001.json.gz"
    ),
    "url": (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        "1588ec454efa1a09f29cd18ddd04fe05fc8653a2/"
        "realnewslike/c4-validation.00000-of-00001.json.gz"
    ),
    "size_bytes": 15_284_943,
    "sha256": "42ac859dc1c4d48d165ec602909403e2066ce1d4854149ed70b9ec9cc96dc65f",
}

COHORTS = (
    ("calibration_null", 1000),
    ("independent_test_null", 1000),
    ("generation_primary", 1000),
    ("generation_reserve", 500),
)

PARENT_FILES = {
    "protocol": {
        "relative_path": (
            "experiments/certified_resegmentation_wp2_v12_fresh_assets1/"
            "PROTOCOL.md"
        ),
        "sha256": (
            "0d8815e7bdf979d42118c56a8450ad1b11a5069c78e1ae9b5bffa233bcd3627e"
        ),
    },
    "v11_asset_audit": {
        "relative_path": (
            "experiments/"
            "certified_resegmentation_wp2_v11_detector_reproduction1/"
            "R0_ASSET_AUDIT.json"
        ),
        "sha256": (
            "d3b87d868cc47a771e4bfae80c9cdfc704d9e5087b7c4fab13b132edcdd65713"
        ),
    },
    "v11_conclusion": {
        "relative_path": (
            "experiments/"
            "certified_resegmentation_wp2_v11_detector_reproduction1/"
            "R1_R2_CONCLUSION.md"
        ),
        "sha256": (
            "7b630d7cf97c1f2827ddf49285c8af083d9dfc18adae3d9e623d08fd21087edf"
        ),
    },
}

HISTORICAL_MANIFESTS = {
    "semstamp": {
        "relative_path": (
            "experiments/certified_resegmentation_wp2_v3_readiness1/"
            "system/semstamp_v3_readiness_rows.jsonl"
        ),
        "sha256": (
            "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218"
        ),
    },
    "simmark": {
        "relative_path": (
            "experiments/certified_resegmentation_wp2_v3_readiness1/"
            "system/simmark_v3_readiness_rows.jsonl"
        ),
        "sha256": (
            "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668"
        ),
    },
}

TOKENIZER_MODELS = {
    "semstamp": {
        "relative_path": "models/semstamp-c4-sbert-a77dce8530e1",
        "revision": "a77dce8530e100ac1101981e4be390546c1cb282",
        "nltk_version": "3.8.1",
        "tokenizer_class": "MPNetTokenizerFast",
        "files": {
            "pytorch_model.bin": (
                "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4"
                "a6a2e9dee"
            ),
            "special_tokens_map.json": (
                "15a9b70f6a5defe4feac823a82e6412e5d5bfd8eab09d8dbcf5ac241f14a7756"
            ),
            "tokenizer.json": (
                "8c28ddc1c09f31165c1a52805d6c5318a40545b1b734a31143121c92cb339778"
            ),
            "tokenizer_config.json": (
                "3ea32220c2e21af4f0be95e2a43bdc8ac693f8c35c2255127ef680689b469461"
            ),
            "vocab.txt": (
                "dbd90cb94e2247bd4d4ccaecbf616d2290e66691d7d5e5bb81f063c2d0649ada"
            ),
        },
    },
    "simmark": {
        "relative_path": "models/instructor-large-54e5ffb8d484",
        "revision": "54e5ffb8d484de506e59443b07dc819fb15c7233",
        "nltk_version": "3.9.1",
        "tokenizer_class": "T5TokenizerFast",
        "files": {
            "pytorch_model.bin": (
                "fb7f63752cf10103be938bc50dfad0b6fa1e63bc67b963471fc827838"
                "d9bbb41"
            ),
            "2_Dense/pytorch_model.bin": (
                "a6e82cba0876dacbadccdea565c9e19e29848d994d23968bd1343b8f"
                "0f762bdc"
            ),
            "special_tokens_map.json": (
                "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"
            ),
            "spiece.model": (
                "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"
            ),
            "tokenizer.json": (
                "72fb264ef4324ec23bc4e8b94167134eaf031b2048fd51a2c2f5a329c7695847"
            ),
            "tokenizer_config.json": (
                "c31a902ec8f6335e196f500d15c9f974e5c69715b7d375642555c32113b3fdf3"
            ),
        },
    },
}

GENERATOR_MODEL = {
    "repository": "AbeHou/opt-1.3b-semstamp",
    "revision": "32662c77794b4f5cbe3d00bcf76fa8653b59bf73",
    "relative_path": "models/opt-1.3b-semstamp-32662c77794",
    "files": {
        "pytorch_model.bin": (
            "9f90897038cfabef26c1d797864e22d00720e3255666e8f9076c54fd43fecbbf"
        ),
    },
    "expected_weight_bytes": 5_263_157_438,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_text_sha256(text):
    if not isinstance(text, str):
        raise TypeError("C4 text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def order_key(identity):
    value = f"{ORDER_SALT}\0{identity}".encode("ascii")
    return hashlib.sha256(value).hexdigest()


def read_source_rows(path):
    rows = []
    invalid_rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                invalid_rows += 1
                continue
            rows.append({
                "source_row_index": row_index,
                "raw_text_sha256": raw_text_sha256(text),
                "text": text,
                "timestamp": value.get("timestamp"),
                "url": value.get("url"),
            })
    return rows, invalid_rows


def atomic_write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_jsonl(path, rows):
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ]
    value = "\n".join(lines) + "\n"
    atomic_write_text(path, value)
    return {
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "bytes": len(value.encode("utf-8")),
        "rows": len(rows),
    }


def relative_to_root(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def audit_expected_files(root, specifications):
    output = {}
    all_match = True
    for item_id, specification in specifications.items():
        path = root / specification["relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        matches = actual == specification["sha256"]
        all_match = all_match and matches
        output[item_id] = {
            **specification,
            "present": path.is_file(),
            "actual_sha256": actual,
            "sha256_matches": matches,
        }
    return output, all_match


def audit_model(root, specification):
    model_root = root / specification["relative_path"]
    files = {}
    for relative_path, expected_sha256 in specification["files"].items():
        path = model_root / relative_path
        actual_sha256 = sha256_file(path) if path.is_file() else None
        files[relative_path] = {
            "present": path.is_file(),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "sha256_matches": actual_sha256 == expected_sha256,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "relative_path": specification["relative_path"],
        "revision": specification["revision"],
        "present": model_root.is_dir(),
        "files": files,
        "ready": model_root.is_dir() and all(
            item["sha256_matches"] for item in files.values()
        ),
    }


def load_history(root):
    output = {}
    all_hashes = set()
    all_match = True
    for system, specification in HISTORICAL_MANIFESTS.items():
        path = root / specification["relative_path"]
        actual_sha256 = sha256_file(path) if path.is_file() else None
        matches = actual_sha256 == specification["sha256"]
        all_match = all_match and matches
        hashes = set()
        row_count = 0
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                hashes.add(row["raw_text_sha256"])
                row_count += 1
        all_hashes.update(hashes)
        output[system] = {
            **specification,
            "present": path.is_file(),
            "actual_sha256": actual_sha256,
            "sha256_matches": matches,
            "rows": row_count,
            "unique_raw_sha256": len(hashes),
        }
    return output, all_hashes, all_match


def tokenizer_text(system, unit, unit_index):
    if system == "semstamp":
        return unit.strip()
    prefixed = unit if unit_index == 0 else " " + unit
    return (INSTRUCTION + prefixed).strip()


def tokenizer_worker(args):
    import nltk
    from nltk.tokenize import sent_tokenize
    from transformers import AutoTokenizer

    system = args.worker_system
    specification = TOKENIZER_MODELS[system]
    if nltk.__version__ != specification["nltk_version"]:
        raise ValueError(
            f"{system} NLTK version {nltk.__version__} does not match "
            f"{specification['nltk_version']}"
        )
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if type(tokenizer).__name__ != specification["tokenizer_class"]:
        raise ValueError("tokenizer class mismatch")
    if tokenizer.model_max_length != NATIVE_TOKEN_LIMIT:
        raise ValueError("tokenizer native limit mismatch")

    source_rows, invalid_source_rows = read_source_rows(args.source_gzip)
    eligible = {}
    exclusions = Counter()
    for row in source_rows:
        text = row["text"]
        identity = row["raw_text_sha256"]
        try:
            units = sent_tokenize(text)
        except Exception:
            exclusions["sentence_split_error"] += 1
            continue
        if (
            len(units) < 2
            or any(
                not isinstance(unit, str)
                or not unit
                or unit != unit.strip()
                for unit in units
            )
        ):
            exclusions["invalid_or_short_p0"] += 1
            continue
        token_counts = [
            len(tokenizer(
                tokenizer_text(system, unit, unit_index),
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"])
            for unit_index, unit in enumerate(units)
        ]
        if max(token_counts) > NATIVE_TOKEN_LIMIT:
            exclusions["unit_above_native_limit"] += 1
            continue
        if identity in eligible:
            exclusions["duplicate_identity"] += 1
            continue
        eligible[identity] = {
            "source_row_index": row["source_row_index"],
            "unit_count": len(units),
            "max_native_tokens": max(token_counts),
        }

    print(json.dumps({
        "system": system,
        "nltk_version": nltk.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "source_sha256": sha256_file(args.source_gzip),
        "source_rows": len(source_rows),
        "invalid_source_rows": invalid_source_rows,
        "eligible": eligible,
        "exclusions": dict(sorted(exclusions.items())),
        "detector_model_loaded": False,
        "detector_scores_computed": 0,
    }, sort_keys=True, separators=(",", ":")))


def run_tokenizer_worker(
    script_path,
    python_path,
    source_path,
    nltk_data,
    model_dir,
    system,
):
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NLTK_DATA": str(nltk_data.resolve()),
    })
    completed = subprocess.run(
        [
            str(python_path),
            str(script_path),
            "--worker-system",
            system,
            "--source-gzip",
            str(source_path),
            "--nltk-data",
            str(nltk_data),
            "--model-dir",
            str(model_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not lines:
        raise ValueError(f"{system} tokenizer worker returned no JSON")
    return json.loads(lines[-1])


def power_analysis():
    from scipy.stats import beta, binom

    sample_size = 1000

    def cp_upper(successes):
        if successes == sample_size:
            return 1.0
        return float(beta.ppf(
            0.95,
            successes + 1,
            sample_size - successes,
        ))

    def cp_lower(successes):
        if successes == 0:
            return 0.0
        return float(beta.ppf(
            0.05,
            successes,
            sample_size - successes + 1,
        ))

    fpr_max_successes = max(
        successes
        for successes in range(sample_size + 1)
        if cp_upper(successes) <= 0.08
    )
    tpr_min_successes = max(
        800,
        next(
            successes
            for successes in range(sample_size + 1)
            if cp_lower(successes) > 0.75
        ),
    )
    certificate_min_successes = max(
        200,
        next(
            successes
            for successes in range(sample_size + 1)
            if cp_lower(successes) > 0.10
        ),
    )
    return {
        "sample_size": sample_size,
        "calibration_order_statistic": math.ceil(
            (sample_size + 1) * 0.95
        ),
        "strict_exceedance": True,
        "fpr_gate": {
            "cp_upper_limit": 0.08,
            "max_exceedances": fpr_max_successes,
            "cp_upper_at_boundary": cp_upper(fpr_max_successes),
            "power_at_true_rate_0_05": float(
                binom.cdf(fpr_max_successes, sample_size, 0.05)
            ),
        },
        "tpr_gate": {
            "raw_rate_minimum": 0.80,
            "cp_lower_minimum_exclusive": 0.75,
            "min_successes": tpr_min_successes,
            "cp_lower_at_boundary": cp_lower(tpr_min_successes),
            "power_at_true_rate_0_85": float(
                binom.sf(tpr_min_successes - 1, sample_size, 0.85)
            ),
        },
        "k1_certificate_gate": {
            "raw_rate_minimum": 0.20,
            "cp_lower_minimum_exclusive": 0.10,
            "min_successes": certificate_min_successes,
            "cp_lower_at_boundary": cp_lower(
                certificate_min_successes
            ),
            "power_at_true_rate_0_25": float(
                binom.sf(
                    certificate_min_successes - 1,
                    sample_size,
                    0.25,
                )
            ),
        },
    }


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "max": None}
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
    }


def require_main_arguments(parser, args):
    names = (
        "root",
        "output",
        "identity_output",
        "sealed_output",
        "semstamp_python",
        "simmark_python",
        "semstamp_nltk_data",
        "simmark_nltk_data",
        "semstamp_model_dir",
        "simmark_model_dir",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        parser.error(
            "main audit requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main_audit(args):
    root = args.root.resolve()
    source_path = args.source_gzip.resolve()
    source_actual_sha256 = (
        sha256_file(source_path) if source_path.is_file() else None
    )
    source_actual_bytes = (
        source_path.stat().st_size if source_path.is_file() else None
    )
    source_identity_pass = (
        source_actual_sha256 == SOURCE["sha256"]
        and source_actual_bytes == SOURCE["size_bytes"]
    )
    if not source_identity_pass:
        raise ValueError("source shard identity mismatch")

    parent_files, parent_files_pass = audit_expected_files(
        root, PARENT_FILES
    )
    historical, historical_hashes, historical_pass = load_history(root)
    tokenizer_models = {
        system: audit_model(root, specification)
        for system, specification in TOKENIZER_MODELS.items()
    }
    tokenizer_models_pass = all(
        item["ready"] for item in tokenizer_models.values()
    )
    if not tokenizer_models_pass:
        raise ValueError("tokenizer model identity mismatch")

    workers = {
        "semstamp": run_tokenizer_worker(
            Path(__file__).resolve(),
            args.semstamp_python,
            source_path,
            args.semstamp_nltk_data,
            args.semstamp_model_dir,
            "semstamp",
        ),
        "simmark": run_tokenizer_worker(
            Path(__file__).resolve(),
            args.simmark_python,
            source_path,
            args.simmark_nltk_data,
            args.simmark_model_dir,
            "simmark",
        ),
    }
    worker_identity_pass = all(
        worker["source_sha256"] == SOURCE["sha256"]
        and worker["nltk_version"]
        == TOKENIZER_MODELS[system]["nltk_version"]
        and worker["tokenizer_class"]
        == TOKENIZER_MODELS[system]["tokenizer_class"]
        and not worker["detector_model_loaded"]
        and worker["detector_scores_computed"] == 0
        for system, worker in workers.items()
    )
    if not worker_identity_pass:
        raise ValueError("tokenizer worker identity mismatch")

    source_rows, invalid_source_rows = read_source_rows(source_path)
    source_identity_counts = Counter(
        row["raw_text_sha256"] for row in source_rows
    )
    source_by_identity = {}
    for row in source_rows:
        source_by_identity.setdefault(row["raw_text_sha256"], row)

    eligible_by_system = {
        system: set(worker["eligible"])
        for system, worker in workers.items()
    }
    dual_eligible = set.intersection(*eligible_by_system.values())
    fresh_dual_eligible = dual_eligible - historical_hashes
    ordered_identities = sorted(
        fresh_dual_eligible,
        key=lambda identity: (order_key(identity), identity),
    )
    required_capacity = sum(count for _, count in COHORTS)
    capacity_pass = len(ordered_identities) >= required_capacity
    selected_identities = ordered_identities[:required_capacity]

    public_rows = []
    sealed_rows = []
    cursor = 0
    for cohort, count in COHORTS:
        for cohort_index, identity in enumerate(
            selected_identities[cursor:cursor + count]
        ):
            source_row = source_by_identity[identity]
            public_rows.append({
                "allocation_index": cursor + cohort_index,
                "cohort": cohort,
                "cohort_index": cohort_index,
                "source_row_index": source_row["source_row_index"],
                "raw_text_sha256": identity,
                "order_key_sha256": order_key(identity),
                "semstamp_unit_count": (
                    workers["semstamp"]["eligible"][identity]["unit_count"]
                ),
                "semstamp_max_native_tokens": (
                    workers["semstamp"]["eligible"][identity][
                        "max_native_tokens"
                    ]
                ),
                "simmark_unit_count": (
                    workers["simmark"]["eligible"][identity]["unit_count"]
                ),
                "simmark_max_native_tokens": (
                    workers["simmark"]["eligible"][identity][
                        "max_native_tokens"
                    ]
                ),
                "contains_raw_text": False,
            })
            sealed_rows.append({
                "allocation_index": cursor + cohort_index,
                "cohort": cohort,
                "cohort_index": cohort_index,
                "source_row_index": source_row["source_row_index"],
                "raw_text_sha256": identity,
                "text": source_row["text"],
                "timestamp": source_row["timestamp"],
                "url": source_row["url"],
            })
        cursor += count

    identity_manifest = write_jsonl(args.identity_output, public_rows)
    sealed_manifest = write_jsonl(args.sealed_output, sealed_rows)
    cohort_counts = Counter(row["cohort"] for row in public_rows)
    selected_set = {row["raw_text_sha256"] for row in public_rows}
    null_set = {
        row["raw_text_sha256"]
        for row in public_rows
        if row["cohort"].endswith("_null")
    }
    prompt_set = selected_set - null_set
    cohort_identity_sets = {
        cohort: {
            row["raw_text_sha256"]
            for row in public_rows
            if row["cohort"] == cohort
        }
        for cohort, _ in COHORTS
    }
    pairwise_cohort_overlaps = {}
    cohort_names = [cohort for cohort, _ in COHORTS]
    for left_index, left in enumerate(cohort_names):
        for right in cohort_names[left_index + 1:]:
            pairwise_cohort_overlaps[f"{left}__{right}"] = len(
                cohort_identity_sets[left] & cohort_identity_sets[right]
            )
    allocation_pass = (
        len(public_rows) == required_capacity
        and len(selected_set) == required_capacity
        and dict(cohort_counts) == dict(COHORTS)
        and not selected_set & historical_hashes
        and not null_set & prompt_set
        and all(value == 0 for value in pairwise_cohort_overlaps.values())
    )

    generator_model = audit_model(root, GENERATOR_MODEL)
    source_readiness = (
        source_identity_pass
        and parent_files_pass
        and historical_pass
        and tokenizer_models_pass
        and worker_identity_pass
        and capacity_pass
        and allocation_pass
    )
    semstamp_generation_readiness = (
        source_readiness and generator_model["ready"]
    )
    simmark_generation_readiness = (
        source_readiness
        and generator_model["ready"]
        and False
    )

    output = {
        "schema_version": 1,
        "created_on": CREATED_ON,
        "audit_stage": "WP2-v12-A0",
        "audit_mode": "score_blind_fresh_source_acquisition",
        "contains_raw_text": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "parent_files": parent_files,
        "source": {
            **SOURCE,
            "local_path": relative_to_root(source_path, root),
            "actual_size_bytes": source_actual_bytes,
            "actual_sha256": source_actual_sha256,
            "identity_pass": source_identity_pass,
            "parsed_rows": len(source_rows),
            "invalid_rows": invalid_source_rows,
            "unique_raw_sha256": len(source_identity_counts),
            "duplicate_rows": sum(
                count - 1
                for count in source_identity_counts.values()
                if count > 1
            ),
        },
        "historical_manifests": historical,
        "historical_union_unique_raw_sha256": len(historical_hashes),
        "tokenizer_models": tokenizer_models,
        "tokenizer_workers": {
            system: {
                key: value
                for key, value in worker.items()
                if key != "eligible"
            }
            for system, worker in workers.items()
        },
        "eligibility": {
            "semstamp_unique": len(eligible_by_system["semstamp"]),
            "simmark_unique": len(eligible_by_system["simmark"]),
            "dual_unique": len(dual_eligible),
            "dual_historical_overlap": len(
                dual_eligible & historical_hashes
            ),
            "fresh_dual_unique": len(fresh_dual_eligible),
            "required": required_capacity,
            "capacity_pass": capacity_pass,
        },
        "allocation": {
            "order_salt": ORDER_SALT,
            "cohort_targets": dict(COHORTS),
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "selected_unique_raw_sha256": len(selected_set),
            "selected_historical_overlap": len(
                selected_set & historical_hashes
            ),
            "null_prompt_overlap": len(null_set & prompt_set),
            "pairwise_cohort_overlaps": pairwise_cohort_overlaps,
            "semstamp_unit_count": distribution([
                row["semstamp_unit_count"] for row in public_rows
            ]),
            "simmark_unit_count": distribution([
                row["simmark_unit_count"] for row in public_rows
            ]),
            "semstamp_max_native_tokens": distribution([
                row["semstamp_max_native_tokens"] for row in public_rows
            ]),
            "simmark_max_native_tokens": distribution([
                row["simmark_max_native_tokens"] for row in public_rows
            ]),
            "allocation_pass": allocation_pass,
        },
        "identity_manifest": {
            **identity_manifest,
            "relative_path": relative_to_root(args.identity_output, root),
            "contains_raw_text": False,
        },
        "sealed_source_manifest": {
            **sealed_manifest,
            "relative_path": relative_to_root(args.sealed_output, root),
            "contains_raw_text": True,
            "public_copy_allowed": False,
        },
        "generator_model": {
            **GENERATOR_MODEL,
            **generator_model,
        },
        "simmark_contract": {
            "author_pooling_semantics_resolved": False,
            "permitted_pooling_setting": None,
            "reason": (
                "WP2-v11 S3 was an exploratory winning arm and cannot be "
                "promoted without independent upstream evidence"
            ),
        },
        "power_analysis": power_analysis(),
        "gates": {
            "source_artifact_identity": (
                "pass" if source_identity_pass else "fail"
            ),
            "parent_identity": "pass" if parent_files_pass else "fail",
            "historical_identity": "pass" if historical_pass else "fail",
            "dual_tokenizer_identity": (
                "pass" if tokenizer_models_pass else "fail"
            ),
            "dual_tokenizer_execution": (
                "pass" if worker_identity_pass else "fail"
            ),
            "fresh_source_capacity": "pass" if capacity_pass else "fail",
            "cohort_allocation": "pass" if allocation_pass else "fail",
            "generator_materialized": (
                "pass" if generator_model["ready"] else "fail"
            ),
            "simmark_author_pooling_semantics": "fail",
        },
        "decision": {
            "status": (
                "source_assets_ready_generation_blocked"
                if source_readiness
                else "stop_source_readiness_failed"
            ),
            "a0_source_readiness": (
                "go" if source_readiness else "stop"
            ),
            "a1_semstamp_generation": (
                "go" if semstamp_generation_readiness else "stop"
            ),
            "a1_simmark_generation": (
                "go" if simmark_generation_readiness else "stop"
            ),
            "b0_detector_health": "stop",
            "b1_exact_certificate": "stop",
            "reason_codes": [
                reason
                for condition, reason in (
                    (
                        not source_readiness,
                        "source_or_identity_gate_failed",
                    ),
                    (
                        not generator_model["ready"],
                        "generator_model_not_materialized",
                    ),
                    (
                        True,
                        "simmark_author_pooling_semantics_unresolved",
                    ),
                    (
                        True,
                        "generation_runner_and_smoke_not_yet_frozen",
                    ),
                )
                if condition
            ],
            "next_action": (
                "materialize and hash the pinned generator, implement the "
                "deterministic A1 runner, and obtain independent upstream "
                "evidence for SimMark pooling before any detector score"
            ),
        },
        "score_blind_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "tokenizer_loads": 2,
            "raw_text_emitted_to_public_artifacts": False,
            "sealed_raw_text_manifest_written": True,
            "operations": [
                "source shard hashing and JSON parsing",
                "historical raw identity set comparison",
                "NLTK sentence segmentation",
                "tokenizer-only native length validation",
                "salted deterministic cohort allocation",
                "binomial power calculation",
            ],
        },
    }
    atomic_write_text(
        args.output,
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
    )
    print(json.dumps({
        "status": output["decision"]["status"],
        "a0": output["decision"]["a0_source_readiness"],
        "fresh_dual_unique": len(fresh_dual_eligible),
        "selected": len(selected_set),
        "generator_ready": generator_model["ready"],
        "simmark_pooling_resolved": False,
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source-gzip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--sealed-output", type=Path)
    parser.add_argument("--semstamp-python", type=Path)
    parser.add_argument("--simmark-python", type=Path)
    parser.add_argument("--semstamp-nltk-data", type=Path)
    parser.add_argument("--simmark-nltk-data", type=Path)
    parser.add_argument("--semstamp-model-dir", type=Path)
    parser.add_argument("--simmark-model-dir", type=Path)
    parser.add_argument(
        "--worker-system",
        choices=sorted(TOKENIZER_MODELS),
    )
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.worker_system is not None:
        if args.nltk_data is None or args.model_dir is None:
            parser.error("worker mode requires --nltk-data and --model-dir")
        tokenizer_worker(args)
        return
    require_main_arguments(parser, args)
    main_audit(args)


if __name__ == "__main__":
    main()
