"""Audit and allocate the score-blind WP2-v13 SemStamp source queue."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


CREATED_ON = "2026-09-25"
PROTOCOL_SHA256 = (
    "e3fa05e532986077840651ae37a1ff1b9a43a9a52706d2337a2435420b274768"
)
SOURCE_SHA256 = (
    "42ac859dc1c4d48d165ec602909403e2066ce1d4854149ed70b9ec9cc96dc65f"
)
SOURCE_BYTES = 15_284_943
ORDER_SALT = "WP2-v13-semstamp-formal-source-order-v1"
ATTEMPT_COUNT = 1600
NATIVE_TOKEN_LIMIT = 512
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
GENERATOR_REVISION = "32662c77794b4f5cbe3d00bcf76fa8653b59bf73"
GENERATOR_MANIFEST_SHA256 = (
    "58e0b3b9442b8abe40cec3a4f4ab3a9a9086317c5ddb4e413cf04ebbb07470c6"
)
GENERATOR_WEIGHT_SHA256 = (
    "9f90897038cfabef26c1d797864e22d00720e3255666e8f9076c54fd43fecbbf"
)
GENERATOR_WEIGHT_BYTES = 5_263_157_438
EMBEDDER_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
EMBEDDER_FILES = {
    "pytorch_model.bin": (
        "f23f44b44fcad45f2f6178c3936ac7c7d8b16b09b3b70a0043b61c4a6a2e9dee"
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
}
OFFICIAL_FILES = {
    "sampling_utils.py": (
        "35164c4e99057617b93dcb98f4a507f34f8007b61536f7f7ab3333987f3e721e"
    ),
    "sampling_lsh_utils.py": (
        "5fe536616f8d4845cd7d19eddc622b055380dd9364499ce03d4e0b6a5dde2a65"
    ),
    "sbert_lsh_model.py": (
        "5e8735e0bd2b8530035b1a9b7951c5f1b002ab875bb706066be5b5925896963a"
    ),
}
PARENT_FILES = {
    "protocol": (
        "experiments/certified_resegmentation_wp2_v13_semstamp_generation1/"
        "PROTOCOL.md",
        PROTOCOL_SHA256,
    ),
    "v12_a0": (
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/"
        "A0_READINESS.json",
        "1496e754c912e105169bf6db63ac6c03062cbc081093887614740840073a3924",
    ),
    "v12_source_identities": (
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/"
        "SOURCE_IDENTITIES.jsonl",
        "ebc1b60759008bc97e0701eef7ba35baa5df35ba8fb1d7c4d8141fcf7ff9cceb",
    ),
    "v12_a1_contract": (
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/"
        "A1_SEMSTAMP_CONTRACT.md",
        "67e66412bd0a2930d0606719d4492a2b599162195080d1263e4063aa955e666d",
    ),
    "v12_a1_result": (
        "experiments/certified_resegmentation_wp2_v12_fresh_assets1/"
        "A1_SEMSTAMP_SMOKE.json",
        "86cae250106c7de5fc7d7f1025a696737e572d109c350f7e0dab18c270bf8faf",
    ),
}
HISTORICAL_MANIFESTS = {
    "semstamp": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/semstamp_v3_readiness_rows.jsonl",
        "fbda89650be020c396df2d4995b4c372069b612890086dc265d0edfe3e1f1218",
    ),
    "simmark": (
        "experiments/certified_resegmentation_wp2_v3_readiness1/"
        "system/simmark_v3_readiness_rows.jsonl",
        "cdc7140b774b5dc5f4e1c30c719f3cc65f36f6974190eaea20555f87f09aa668",
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def order_key(identity):
    payload = f"{ORDER_SALT}\0{identity}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def read_source(path):
    rows = []
    invalid = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for source_row_index, line in enumerate(handle):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                invalid += 1
                continue
            rows.append({
                "source_row_index": source_row_index,
                "raw_text_sha256": sha256_text(text),
                "text": text,
                "timestamp": value.get("timestamp"),
                "url": value.get("url"),
            })
    return rows, invalid


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
    value = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ) + "\n"
    atomic_write_text(path, value)
    return {
        "relative_path": None,
        "rows": len(rows),
        "bytes": len(value.encode("utf-8")),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0] if values else None,
        "median": statistics.median(values) if values else None,
        "max": values[-1] if values else None,
    }


def relative_to_root(path, root):
    return str(path.resolve().relative_to(root.resolve()))


def verify_file(path, expected):
    actual = sha256_file(path) if path.is_file() else None
    return {
        "relative_path": str(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "present": path.is_file(),
        "sha256_matches": actual == expected,
    }


def verify_parent_files(root):
    output = {}
    for name, (relative_path, expected) in PARENT_FILES.items():
        result = verify_file(root / relative_path, expected)
        result["relative_path"] = relative_path
        output[name] = result
    if not all(item["sha256_matches"] for item in output.values()):
        raise ValueError("parent file identity mismatch")
    return output


def load_historical_identities(root):
    output = {}
    identities = set()
    for name, (relative_path, expected) in HISTORICAL_MANIFESTS.items():
        path = root / relative_path
        result = verify_file(path, expected)
        if not result["sha256_matches"]:
            raise ValueError("historical manifest identity mismatch")
        rows = read_jsonl(path)
        row_identities = {row["raw_text_sha256"] for row in rows}
        identities.update(row_identities)
        result.update({
            "relative_path": relative_path,
            "rows": len(rows),
            "unique_raw_sha256": len(row_identities),
        })
        output[name] = result
    return output, identities


def verify_embedder(embedder_dir):
    files = {}
    for relative_path, expected in EMBEDDER_FILES.items():
        path = embedder_dir / relative_path
        result = verify_file(path, expected)
        result["relative_path"] = relative_path
        result["bytes"] = path.stat().st_size if path.is_file() else None
        files[relative_path] = result
    if not all(item["sha256_matches"] for item in files.values()):
        raise ValueError("embedder identity mismatch")
    return {
        "revision": EMBEDDER_REVISION,
        "files": files,
    }


def verify_generator(generator_dir):
    manifest = generator_dir / "MODEL_MANIFEST.sha256"
    if sha256_file(manifest) != GENERATOR_MANIFEST_SHA256:
        raise ValueError("generator manifest identity mismatch")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative_path = line.split(None, 1)
        relative_path = relative_path.strip()
        path = generator_dir / relative_path
        actual = sha256_file(path) if path.is_file() else None
        entries.append({
            "relative_path": relative_path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": actual == expected,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    by_path = {item["relative_path"]: item for item in entries}
    weight = by_path.get("pytorch_model.bin")
    if (
        not entries
        or not all(item["sha256_matches"] for item in entries)
        or weight is None
        or weight["actual_sha256"] != GENERATOR_WEIGHT_SHA256
        or weight["bytes"] != GENERATOR_WEIGHT_BYTES
    ):
        raise ValueError("generator identity mismatch")
    return {
        "revision": GENERATOR_REVISION,
        "manifest_sha256": GENERATOR_MANIFEST_SHA256,
        "files": entries,
    }


def verify_official_source(root):
    repository = root / "repos/SemStamp"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = {
        relative_path: sha256_file(repository / relative_path)
        for relative_path in OFFICIAL_FILES
    }
    if (
        head != SEMSTAMP_REVISION
        or status
        or files != OFFICIAL_FILES
    ):
        raise ValueError("official SemStamp source identity mismatch")
    return {
        "revision": head,
        "worktree_clean": True,
        "files": files,
    }


def verify_v12_pilot(root):
    result = json.loads(
        (root / PARENT_FILES["v12_a1_result"][0]).read_text(
            encoding="utf-8"
        )
    )
    if (
        result["status"] != "stop"
        or result["determinism"]["row_matches"] != 20
        or not result["determinism"]["all_rows_match"]
        or result["eligibility"]["eligible_rows"] != 36
    ):
        raise ValueError("v12 pilot summary mismatch")
    smoke_output_identities = set()
    failures = {}
    total_seconds = 0.0
    for replay_key in ("0", "1"):
        rows = result["replays"][replay_key]
        if len(rows) != 20:
            raise ValueError("v12 pilot row count mismatch")
        total_seconds += sum(row["elapsed_seconds"] for row in rows)
        replay_failures = []
        for row in rows:
            output_identity = row["output"]["raw_text_sha256"]
            smoke_output_identities.add(output_identity)
            if row["exclusion_codes"]:
                replay_failures.append(row["cohort_index"])
                if (
                    row["exclusion_codes"] != ["invalid_or_short_p0"]
                    or row["prompt_sha256"] != output_identity
                    or row["output"]["unit_count"] != 1
                ):
                    raise ValueError("v12 pilot failure mechanism mismatch")
        failures[replay_key] = replay_failures
    if failures != {"0": [2, 3], "1": [2, 3]}:
        raise ValueError("v12 pilot failure index mismatch")
    attestation = result["execution_attestation"]
    if any(attestation[key] != 0 for key in (
        "detector_imports",
        "detector_score_reads",
        "detector_scores_computed",
        "resegmentation_calls",
        "certificate_calls",
    )):
        raise ValueError("v12 pilot score-blind attestation mismatch")
    return {
        "result_sha256": PARENT_FILES["v12_a1_result"][1],
        "replay_output_matches": 20,
        "unique_outputs": len(smoke_output_identities),
        "unique_eligible_outputs": 18,
        "failed_cohort_indices": [2, 3],
        "failure_code": "invalid_or_short_p0",
        "failure_output_equals_prompt": True,
        "total_replay_seconds": total_seconds,
    }, smoke_output_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-gzip", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--embedder-dir", type=Path, required=True)
    parser.add_argument("--generator-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--sealed-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    source_actual_sha256 = sha256_file(args.source_gzip)
    source_actual_bytes = args.source_gzip.stat().st_size
    if (
        source_actual_sha256 != SOURCE_SHA256
        or source_actual_bytes != SOURCE_BYTES
    ):
        raise ValueError("source shard identity mismatch")

    parent_files = verify_parent_files(root)
    historical, historical_identities = load_historical_identities(root)
    v12_rows = read_jsonl(
        root / PARENT_FILES["v12_source_identities"][0]
    )
    v12_source_identities = {
        row["raw_text_sha256"] for row in v12_rows
    }
    if len(v12_rows) != 3500 or len(v12_source_identities) != 3500:
        raise ValueError("v12 source allocation mismatch")
    pilot, smoke_output_identities = verify_v12_pilot(root)
    embedder = verify_embedder(args.embedder_dir)
    generator = verify_generator(args.generator_dir)
    official_source = verify_official_source(root)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NLTK_DATA"] = str(args.nltk_data.resolve())
    import nltk
    from nltk.tokenize import sent_tokenize
    from transformers import AutoTokenizer

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.embedder_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if (
        type(tokenizer).__name__ != "MPNetTokenizerFast"
        or tokenizer.model_max_length != NATIVE_TOKEN_LIMIT
    ):
        raise ValueError("SemStamp tokenizer contract mismatch")

    source_rows, invalid_source_rows = read_source(args.source_gzip)
    source_counts = Counter(
        row["raw_text_sha256"] for row in source_rows
    )
    source_by_identity = {}
    eligible = {}
    exclusions = Counter()
    for row in source_rows:
        identity = row["raw_text_sha256"]
        source_by_identity.setdefault(identity, row)
        if source_counts[identity] != 1:
            exclusions["duplicate_source_identity"] += 1
            continue
        try:
            units = sent_tokenize(row["text"])
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
                unit.strip(),
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"])
            for unit in units
        ]
        if max(token_counts) > NATIVE_TOKEN_LIMIT:
            exclusions["unit_above_native_limit"] += 1
            continue
        eligible[identity] = {
            "unit_count": len(units),
            "max_native_tokens": max(token_counts),
        }

    forbidden = (
        historical_identities
        | v12_source_identities
        | smoke_output_identities
    )
    fresh_identities = set(eligible) - forbidden
    ordered = sorted(
        fresh_identities,
        key=lambda identity: (order_key(identity), identity),
    )
    selected = ordered[:ATTEMPT_COUNT]
    if len(selected) != ATTEMPT_COUNT:
        raise ValueError("insufficient fresh generation sources")

    public_rows = []
    sealed_rows = []
    for attempt_index, identity in enumerate(selected):
        source_row = source_by_identity[identity]
        public_rows.append({
            "attempt_index": attempt_index,
            "source_row_index": source_row["source_row_index"],
            "raw_text_sha256": identity,
            "order_key_sha256": order_key(identity),
            "unit_count": eligible[identity]["unit_count"],
            "max_native_tokens": eligible[identity][
                "max_native_tokens"
            ],
            "contains_raw_text": False,
        })
        sealed_rows.append({
            "attempt_index": attempt_index,
            "source_row_index": source_row["source_row_index"],
            "raw_text_sha256": identity,
            "text": source_row["text"],
            "timestamp": source_row["timestamp"],
            "url": source_row["url"],
        })

    identity_manifest = write_jsonl(args.identity_output, public_rows)
    identity_manifest["relative_path"] = relative_to_root(
        args.identity_output, root
    )
    identity_manifest["contains_raw_text"] = False
    sealed_manifest = write_jsonl(args.sealed_output, sealed_rows)
    sealed_manifest["relative_path"] = relative_to_root(
        args.sealed_output, root
    )
    sealed_manifest.update({
        "contains_raw_text": True,
        "public_copy_allowed": False,
    })
    selected_set = set(selected)
    allocation_pass = (
        len(selected_set) == ATTEMPT_COUNT
        and not selected_set & historical_identities
        and not selected_set & v12_source_identities
        and not selected_set & smoke_output_identities
    )
    if not allocation_pass:
        raise ValueError("generation source allocation failed")

    output = {
        "schema_version": 1,
        "created_on": CREATED_ON,
        "stage": "WP2-v13-G0",
        "mode": "score_blind_generation_source_allocation",
        "contains_raw_text": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "parent_files": parent_files,
        "source": {
            "repository": "allenai/c4",
            "revision": (
                "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            ),
            "config": "realnewslike",
            "split": "validation",
            "relative_path": (
                "realnewslike/c4-validation.00000-of-00001.json.gz"
            ),
            "actual_sha256": source_actual_sha256,
            "actual_bytes": source_actual_bytes,
            "parsed_rows": len(source_rows),
            "invalid_rows": invalid_source_rows,
            "unique_raw_sha256": len(source_counts),
        },
        "historical_manifests": historical,
        "v12_exclusions": {
            "source_identity_count": len(v12_source_identities),
            "smoke_output_identity_count": len(
                smoke_output_identities
            ),
            "pilot": pilot,
        },
        "model_contract": {
            "embedder": embedder,
            "generator": generator,
            "official_source": official_source,
        },
        "eligibility": {
            "tokenizer_class": type(tokenizer).__name__,
            "nltk_version": nltk.__version__,
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "eligible_before_identity_exclusions": len(eligible),
            "historical_overlap": len(
                set(eligible) & historical_identities
            ),
            "v12_source_overlap": len(
                set(eligible) & v12_source_identities
            ),
            "v12_smoke_output_overlap": len(
                set(eligible) & smoke_output_identities
            ),
            "fresh_after_all_exclusions": len(fresh_identities),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "allocation": {
            "order_salt": ORDER_SALT,
            "attempt_target": ATTEMPT_COUNT,
            "selected_unique": len(selected_set),
            "selected_historical_overlap": len(
                selected_set & historical_identities
            ),
            "selected_v12_source_overlap": len(
                selected_set & v12_source_identities
            ),
            "selected_v12_smoke_output_overlap": len(
                selected_set & smoke_output_identities
            ),
            "unit_count": distribution([
                row["unit_count"] for row in public_rows
            ]),
            "max_native_tokens": distribution([
                row["max_native_tokens"] for row in public_rows
            ]),
            "allocation_pass": allocation_pass,
        },
        "identity_manifest": identity_manifest,
        "sealed_source_manifest": sealed_manifest,
        "decision": {
            "status": "g0_go_g1_authorized",
            "g0_source_readiness": "go",
            "g1_formal_generation": "go",
            "h0_identity_freeze": "stop_until_g1_complete",
            "b0_detector_health": "stop",
            "b1_exact_certificate": "stop",
            "next_action": (
                "freeze and launch the G1 replacement-aware runner "
                "without detector calls"
            ),
        },
        "score_blind_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "generator_model_loads": 0,
            "tokenizer_loads": 1,
            "raw_text_emitted_to_public_artifacts": False,
            "sealed_raw_text_manifest_written": True,
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
        "fresh_after_all_exclusions": len(fresh_identities),
        "selected": len(selected_set),
        "identity_manifest_sha256": identity_manifest["sha256"],
        "sealed_manifest_sha256": sealed_manifest["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
