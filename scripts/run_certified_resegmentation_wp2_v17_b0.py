"""Run the frozen WP2-v17 SemStamp B0 detector-health experiment."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import time

import certified_resegmentation_wp2_v13_b0_support as support


STAGE = "WP2-v17-B0-SemStamp-detector-health"
PROTOCOL_SHA256 = (
    "435234b89955935272f7fc59c8d7e6d04de0c9ee579556f06151e4dd63458dd4"
)
CONTRACT_SHA256 = (
    "ff5256aa75675536ad3c8dc200341be0eff726ae73d06db7bb65c0a056240848"
)
H0_SHA256 = (
    "9fa13f542414a29734b367d0e73a04675ad212e8980aa7b1fafdc00468cdf5b4"
)
H0_BYTES = 7_004
H0_RUNNER_SHA256 = (
    "008b4a65fa73f706f0157f3973cb610bcd7e6bb422901b27769b8ccfb59d97f6"
)
V16_PROTOCOL_SHA256 = (
    "0449b363abff408007de8e6024c2d147b5aa586c00ad8b19a8b519e080c41cfe"
)
G1R_CONTRACT_SHA256 = (
    "c8344700cf6c420bfb4cbe0fe6fb4870febf70aa772a5c328d5ecfba5dfe3fe9"
)
G1R_RESULT_SHA256 = (
    "4ecf7f92aaf6bd554d6d3a50da41edb7ae60bb69ebbe69cc4886e5f69134d146"
)
COMBINED_SHA256 = (
    "cae59db1b40d8c9e14bb95f88999ef29639cde0a89f7baddfa41504b3e992598"
)
COMBINED_BYTES = 351_993
V12_IDENTITIES_SHA256 = (
    "ebc1b60759008bc97e0701eef7ba35baa5df35ba8fb1d7c4d8141fcf7ff9cceb"
)
V12_IDENTITIES_BYTES = 1_426_865
V12_SEALED_SHA256 = (
    "8a65987196c14fab5eb2d0bf5ff7ec23b4748607b32e076e8fbcc2595efa18b3"
)
V12_SEALED_BYTES = 10_537_171
V15_SEALED_SHA256 = (
    "b60bfdb8d84e358dca049f73a681609e23991c05e1f2327f1038c5d5e11d9404"
)
V15_SEALED_BYTES = 5_329_796
V16_SEALED_SHA256 = (
    "81bb32e33629b71dbfb4ac93a930c1eaeaa294387d938597106fa9aee0a693ba"
)
V16_SEALED_BYTES = 18_058
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
MODEL_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
SOURCE_SHA256 = {
    "sbert_lsh_model.py": (
        "5e8735e0bd2b8530035b1a9b7951c5f1b002ab875bb706066be5b5925896963a"
    ),
    "sampling_lsh_utils.py": (
        "5fe536616f8d4845cd7d19eddc622b055380dd9364499ce03d4e0b6a5dde2a65"
    ),
    "detection_utils.py": (
        "2ab984bc838cb36db5213398dcd162ddc6625500d9d52eb2cff813f74bb2d9cd"
    ),
}
MODEL_SHA256 = {
    "config.json": (
        "77d80f1542b994cb16f9b9dadd34e04e6ab7a0f79ca201ce309f3d7cf1c82b97"
    ),
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
EXPECTED_OVERLAPS = {
    "all_attempt_sources",
    "historical_all",
    "supplement_failed_outputs",
    "v12_all_sources",
    "v12_calibration_null",
    "v12_independent_test_null",
    "v12_smoke_outputs",
    "v15_failed_outputs",
    "v15_nonretained_outputs",
}
ZERO_CALL_FIELDS = {
    "certificate_calls",
    "detector_imports",
    "detector_model_loads",
    "detector_score_reads",
    "detector_scores_computed",
    "resegmentation_calls",
}
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
NATIVE_TOKEN_LIMIT = 512
REPLAY_COUNT = 2
BATCH_SIZE = 64
MIN_FREE_GIB = 5.0
MAX_VRAM_FRACTION = 0.80


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value):
    if not isinstance(value, str):
        raise TypeError("raw text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def jsonl_bytes(rows):
    return b"\n".join(canonical_bytes(row) for row in rows) + b"\n"


def atomic_write_bytes(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, mode)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value, mode=0o600):
    atomic_write_bytes(path, canonical_bytes(value) + b"\n", mode)


def relative_path(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def file_metadata(path, root=None):
    path = Path(path)
    result = {
        "sha256": support.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if root is not None:
        result["relative_path"] = relative_path(path, root)
    return result


def verify_file(path, expected_sha256, expected_bytes, label, mode=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    observed = file_metadata(path)
    if (
        observed["sha256"] != expected_sha256
        or observed["bytes"] != expected_bytes
    ):
        raise ValueError(f"{label} bytes/SHA mismatch")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise ValueError(f"{label} mode mismatch")
    return observed


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def package_versions():
    packages = (
        "nltk",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in packages},
    }


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside experiment root")
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "B0_CONTRACT.md",
        "h0": (
            root
            / "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1"
            / "H0_RESULT.json"
        ),
        "combined": (
            root
            / "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1"
            / "COMBINED_POSITIVE_IDENTITIES.jsonl"
        ),
        "v12_identities": (
            root
            / "experiments/certified_resegmentation_wp2_v12_fresh_assets1"
            / "SOURCE_IDENTITIES.jsonl"
        ),
        "v12_sealed": (
            root
            / "experiments/certified_resegmentation_wp2_v12_fresh_assets1"
            / "sealed/source_rows.jsonl"
        ),
        "v15_sealed": (
            root
            / "experiments/certified_resegmentation_wp2_v15_causal_prefix1"
            / "sealed/generation_checkpoint.json"
        ),
        "v16_sealed": (
            root
            / "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1"
            / "sealed/supplement_checkpoint.json"
        ),
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
        "support": Path(support.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "test": root / "scripts/test_run_certified_resegmentation_wp2_v17_b0.py",
        "plan": output_dir / "B0_PLAN.jsonl",
        "readiness": output_dir / "P0_READINESS.json",
        "checkpoint": output_dir / "sealed/b0_checkpoint.json",
        "heartbeat": output_dir / "B0_HEARTBEAT.json",
        "rows0": output_dir / "B0_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "B0_ROWS_REPLAY1.jsonl",
        "result": output_dir / "B0_RESULT.json",
    }


def validate_h0(h0):
    counts = h0.get("counts", {})
    decision = h0.get("decision", {})
    attestation = h0.get("execution_attestation", {})
    overlaps = h0.get("overlaps", {})
    if (
        h0.get("schema_version") != 1
        or h0.get("stage") != "WP2-v16-H0-rescue-identity-freeze"
        or h0.get("status") != "go"
        or h0.get("scope") != "identity_only_no_detector_scores"
        or h0.get("contains_raw_text") is not False
        or h0.get("runner_sha256") != H0_RUNNER_SHA256
        or decision.get("h0_identity_freeze") != "go"
        or decision.get("b0_detector_health")
        != "authorized_to_freeze_protocol"
        or decision.get("b1_exact_certificate") != "stop"
        or counts
        != {
            "combined_output_unique": 1000,
            "combined_positive": 1000,
            "combined_source_unique": 1000,
            "retained_parent": 995,
            "supplement_attempts": 6,
            "supplement_positive": 5,
        }
        or h0.get("resource_gate", {}).get("below_0_80") is not True
        or set(overlaps) != EXPECTED_OVERLAPS
        or any(value.get("count") != 0 for value in overlaps.values())
        or any(attestation.get(field) != 0 for field in ZERO_CALL_FIELDS)
        or attestation.get("raw_text_emitted_to_public_artifacts") is not False
        or attestation.get("v15_sealed_checkpoint_content_parsed") is not False
        or attestation.get("supplement_sealed_checkpoint_content_parsed") is not False
    ):
        raise ValueError("v16 H0 is not the frozen terminal identity Go")
    manifest = h0.get("combined_manifest", {})
    if (
        manifest.get("sha256") != COMBINED_SHA256
        or manifest.get("bytes") != COMBINED_BYTES
        or manifest.get("rows") != COHORT_SIZE
        or manifest.get("contains_raw_text") is not False
    ):
        raise ValueError("v16 combined manifest binding mismatch")
    expected_inputs = {
        "protocol": V16_PROTOCOL_SHA256,
        "g1r_contract": G1R_CONTRACT_SHA256,
        "g1r_result": G1R_RESULT_SHA256,
        "v12_source_identities": V12_IDENTITIES_SHA256,
        "v15_sealed_checkpoint": V15_SEALED_SHA256,
        "supplement_sealed_checkpoint": V16_SEALED_SHA256,
    }
    if any(
        h0.get("inputs", {}).get(label, {}).get("sha256") != expected
        for label, expected in expected_inputs.items()
    ):
        raise ValueError("v16 H0 parent binding mismatch")
    return h0


def validate_v12_public(rows):
    if len(rows) != 3500:
        raise ValueError("v12 public source count mismatch")
    expected_counts = {
        "calibration_null": 1000,
        "generation_primary": 1000,
        "generation_reserve": 500,
        "independent_test_null": 1000,
    }
    if Counter(row.get("cohort") for row in rows) != expected_counts:
        raise ValueError("v12 public cohort counts changed")
    seen_by_cohort = Counter()
    identities = set()
    for index, row in enumerate(rows):
        cohort = row.get("cohort")
        if (
            row.get("allocation_index") != index
            or row.get("cohort_index") != seen_by_cohort[cohort]
            or row.get("contains_raw_text") is not False
        ):
            raise ValueError("v12 public order/header mismatch")
        support.require_sha(row.get("raw_text_sha256"))
        support.require_integer(
            row.get("semstamp_unit_count"), 2, 2**31 - 1, "v12 unit count"
        )
        support.require_integer(
            row.get("semstamp_max_native_tokens"),
            1,
            NATIVE_TOKEN_LIMIT,
            "v12 native token count",
        )
        if row["raw_text_sha256"] in identities:
            raise ValueError("duplicate v12 public identity")
        identities.add(row["raw_text_sha256"])
        seen_by_cohort[cohort] += 1
    return rows


def validate_combined(rows):
    if len(rows) != COHORT_SIZE:
        raise ValueError("combined positive count mismatch")
    identities = set()
    sources = set()
    origins = Counter()
    for index, row in enumerate(rows):
        expected_origin = "v15_retained" if index < 995 else "v16_supplement"
        expected_origin_index = index if index < 995 else index - 995
        if (
            row.get("positive_index") != index
            or row.get("origin") != expected_origin
            or row.get("origin_index") != expected_origin_index
            or row.get("contains_raw_text") is not False
        ):
            raise ValueError("combined positive order/provenance mismatch")
        output_identity = support.require_sha(row.get("output_raw_text_sha256"))
        source_identity = support.require_sha(row.get("source_raw_text_sha256"))
        support.require_integer(row.get("attempt_index"), 0, 1599, "attempt index")
        support.require_integer(row.get("unit_count"), 2, 2**31 - 1, "unit count")
        support.require_integer(
            row.get("max_native_tokens"),
            1,
            NATIVE_TOKEN_LIMIT,
            "native token count",
        )
        if output_identity in identities or source_identity in sources:
            raise ValueError("combined positive identity duplication")
        identities.add(output_identity)
        sources.add(source_identity)
        origins[expected_origin] += 1
    if origins != {"v15_retained": 995, "v16_supplement": 5}:
        raise ValueError("combined positive origin counts changed")
    return rows


def map_null_texts(public_rows, sealed_rows):
    if len(sealed_rows) != len(public_rows):
        raise ValueError("v12 sealed/public row count mismatch")
    selected = {}
    for index, (public, sealed) in enumerate(zip(public_rows, sealed_rows)):
        if (
            sealed.get("allocation_index") != index
            or sealed.get("cohort") != public["cohort"]
            or sealed.get("cohort_index") != public["cohort_index"]
            or sealed.get("source_row_index") != public["source_row_index"]
            or sealed.get("raw_text_sha256") != public["raw_text_sha256"]
            or sha256_text(sealed.get("text")) != public["raw_text_sha256"]
        ):
            raise ValueError("v12 sealed/public mapping mismatch")
        if public["cohort"] in support.COHORTS[:2]:
            selected[(public["cohort"], public["cohort_index"])] = sealed["text"]
    if len(selected) != 2000:
        raise ValueError("null sealed mapping count mismatch")
    return selected


def validate_attempt_output(attempt, manifest):
    output = attempt.get("output", {})
    text = attempt.get("generated_text")
    if (
        attempt.get("attempt_index") != manifest["attempt_index"]
        or attempt.get("source_raw_text_sha256")
        != manifest["source_raw_text_sha256"]
        or attempt.get("seed") != manifest["seed"]
        or attempt.get("eligible") is not True
        or attempt.get("exclusion_codes") != []
        or output.get("raw_text_sha256")
        != manifest["output_raw_text_sha256"]
        or output.get("unit_count") != manifest["unit_count"]
        or output.get("max_native_tokens") != manifest["max_native_tokens"]
        or output.get("output_equals_prompt") is not False
        or text == attempt.get("prompt")
        or sha256_text(text) != manifest["output_raw_text_sha256"]
    ):
        raise ValueError("positive sealed/public mapping mismatch")
    return text


def map_positive_texts(combined, v15_checkpoint, v16_checkpoint):
    v15_attempts = v15_checkpoint.get("attempts")
    v16_attempts = v16_checkpoint.get("attempts")
    if (
        v15_checkpoint.get("stage")
        != "WP2-v15-G1-SemStamp-causal-prefix-generation"
        or v15_checkpoint.get("status") != "go"
        or v15_checkpoint.get("terminal_reason") != "target_eligible_reached"
        or not isinstance(v15_attempts, list)
        or len(v15_attempts) != 1111
        or [row.get("attempt_index") for row in v15_attempts] != list(range(1111))
        or v16_checkpoint.get("stage")
        != "WP2-v16-G1R-prompt-only-rescue"
        or v16_checkpoint.get("status") != "go"
        or v16_checkpoint.get("terminal_reason") != "target_supplement_reached"
        or not isinstance(v16_attempts, list)
        or len(v16_attempts) != 6
        or [row.get("attempt_index") for row in v16_attempts]
        != list(range(1111, 1117))
    ):
        raise ValueError("positive sealed checkpoint terminal mismatch")
    v15_by_attempt = {row["attempt_index"]: row for row in v15_attempts}
    supplement = [row for row in v16_attempts if row.get("eligible") is True]
    if len(supplement) != 5:
        raise ValueError("supplement eligible count mismatch")
    texts = []
    for manifest in combined:
        if manifest["origin"] == "v15_retained":
            attempt = v15_by_attempt.get(manifest["attempt_index"])
        else:
            attempt = supplement[manifest["origin_index"]]
        texts.append(validate_attempt_output(attempt, manifest))
    return texts


def materialize_inputs(paths):
    h0 = validate_h0(read_json(paths["h0"]))
    v12_public = validate_v12_public(read_jsonl(paths["v12_identities"]))
    combined = validate_combined(read_jsonl(paths["combined"]))
    plan = support.make_public_plan(v12_public, combined)
    nulls = map_null_texts(v12_public, read_jsonl(paths["v12_sealed"]))
    positives = map_positive_texts(
        combined,
        read_json(paths["v15_sealed"]),
        read_json(paths["v16_sealed"]),
    )
    texts = [
        nulls[(row["cohort"], row["cohort_index"])]
        if row["cohort"] != "positive"
        else positives[row["cohort_index"]]
        for row in plan
    ]
    if len(texts) != TOTAL_ROWS:
        raise ValueError("materialized text count mismatch")
    return h0, plan, texts


def verify_units(plan, texts, nltk_data, model_dir):
    import nltk
    from nltk.tokenize import sent_tokenize
    from transformers import AutoTokenizer

    if nltk.__version__ != "3.8.1":
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    nltk.data.find("tokenizers/punkt/english.pickle")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_dir).resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if type(tokenizer).__name__ != "MPNetTokenizerFast":
        raise ValueError("SemStamp tokenizer class mismatch")
    if tokenizer.model_max_length != NATIVE_TOKEN_LIMIT:
        raise ValueError("SemStamp tokenizer native limit mismatch")

    all_units = []
    for index, (planned, text) in enumerate(zip(plan, texts)):
        units = tuple(sent_tokenize(text))
        if (
            len(units) < 2
            or any(
                not isinstance(unit, str)
                or not unit
                or unit != unit.strip()
                for unit in units
            )
        ):
            raise ValueError(f"invalid sentence units at plan index {index}")
        token_counts = [
            len(
                tokenizer(
                    unit.strip(),
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
            )
            for unit in units
        ]
        if (
            len(units) != planned["unit_count"]
            or max(token_counts) != planned["max_native_tokens"]
            or max(token_counts) > NATIVE_TOKEN_LIMIT
        ):
            raise ValueError(f"unit/token structure drift at plan index {index}")
        all_units.append(units)
    return all_units


def verified_metadata(paths, model_dir):
    root = paths["root"]
    specifications = {
        "protocol": (paths["protocol"], PROTOCOL_SHA256, 7_702, None),
        "contract": (paths["contract"], CONTRACT_SHA256, 2_999, None),
        "v16_h0": (paths["h0"], H0_SHA256, H0_BYTES, None),
        "v16_combined": (
            paths["combined"],
            COMBINED_SHA256,
            COMBINED_BYTES,
            None,
        ),
        "v12_identities": (
            paths["v12_identities"],
            V12_IDENTITIES_SHA256,
            V12_IDENTITIES_BYTES,
            None,
        ),
        "v12_sealed": (
            paths["v12_sealed"],
            V12_SEALED_SHA256,
            V12_SEALED_BYTES,
            0o600,
        ),
        "v15_sealed": (
            paths["v15_sealed"],
            V15_SEALED_SHA256,
            V15_SEALED_BYTES,
            0o600,
        ),
        "v16_sealed": (
            paths["v16_sealed"],
            V16_SEALED_SHA256,
            V16_SEALED_BYTES,
            0o600,
        ),
    }
    observed = {}
    for label, (path, sha256, size, mode) in specifications.items():
        verify_file(path, sha256, size, label, mode)
        observed[label] = file_metadata(path, root)
        if mode is not None:
            observed[label]["mode"] = "0600"

    if paths["support"] != Path(support.__file__).resolve():
        raise ValueError("support module path mismatch")
    for label in ("runner", "support", "test"):
        path = paths[label]
        if not path.is_file():
            raise ValueError(f"{label} file missing")
        observed[label] = file_metadata(path, root)

    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    observed["official_source"] = {}
    for name, expected in SOURCE_SHA256.items():
        path = paths["repo"] / name
        if support.file_sha256(path) != expected:
            raise ValueError(f"official source mismatch: {name}")
        observed["official_source"][name] = file_metadata(path, root)
    observed["model"] = {"revision": MODEL_REVISION, "files": {}}
    for name, expected in MODEL_SHA256.items():
        path = Path(model_dir) / name
        if support.file_sha256(path) != expected:
            raise ValueError(f"model file mismatch: {name}")
        observed["model"]["files"][name] = file_metadata(path, root)
    return observed


def build_preflight(args):
    if args.batch_size != BATCH_SIZE:
        raise ValueError("frozen embedding batch size is 64")
    if args.min_free_gib != MIN_FREE_GIB:
        raise ValueError("frozen minimum free-space gate is 5 GiB")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("TRANSFORMERS_OFFLINE=1 is required")
    paths = paths_for(args.root, args.output_dir)
    metadata_values = verified_metadata(paths, args.embedder_dir)
    required_free = int(args.min_free_gib * 1024**3)
    if shutil.disk_usage(paths["root"]).free < required_free:
        raise OSError("free disk is below the frozen 5 GiB gate")
    _, plan, texts = materialize_inputs(paths)
    units = verify_units(plan, texts, args.nltk_data, args.embedder_dir)
    plan_encoded = jsonl_bytes(plan)
    readiness = {
        "schema_version": 1,
        "stage": f"{STAGE}-P0",
        "status": "pass",
        "scope": "identity_text_structure_only_no_detector",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "artifacts": metadata_values,
        "plan": {
            "rows": len(plan),
            "cohort_counts": dict(Counter(row["cohort"] for row in plan)),
            "canonical_sha256": support.digest(plan),
            "jsonl_sha256": hashlib.sha256(plan_encoded).hexdigest(),
            "jsonl_bytes": len(plan_encoded),
            "contains_raw_text": False,
        },
        "structure": {
            "unit_count_sha256": support.digest(
                [row["unit_count"] for row in plan]
            ),
            "max_native_tokens_sha256": support.digest(
                [row["max_native_tokens"] for row in plan]
            ),
            "native_token_limit": NATIVE_TOKEN_LIMIT,
            "all_rows_exact": True,
        },
        "environment": package_versions(),
        "resource_gate": {
            "minimum_free_gib": MIN_FREE_GIB,
            "free_disk_pass": True,
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "sealed_inputs_parsed": True,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "contains_raw_text": False,
    }
    return paths, plan, units, plan_encoded, readiness


class ReplayLSHModel:
    def __init__(self, regions):
        self._regions = tuple(int(region) for region in regions)
        self._index = 0

    def get_hash(self, texts):
        if len(texts) != 1:
            raise ValueError("official detect_lsh requested a non-scalar batch")
        if self._index >= len(self._regions):
            raise AssertionError("official detect_lsh over-consumed regions")
        value = self._regions[self._index]
        self._index += 1
        return [value]

    def assert_exhausted(self):
        if self._index != len(self._regions):
            raise AssertionError("official detect_lsh did not consume all regions")


def validate_checkpoint(checkpoint, plan, relation):
    return support.validate_checkpoint(
        checkpoint, plan, relation, CONTRACT_SHA256
    )


def write_heartbeat(path, replay, completed, started, stage, **extra):
    value = {
        "schema_version": 1,
        "stage": stage,
        "status": "running" if stage != "terminal" else extra.pop("status"),
        "replay": replay,
        "completed_in_replay": completed,
        "completed_total": replay * TOTAL_ROWS + completed,
        "target_total": REPLAY_COUNT * TOTAL_ROWS,
        "started_at": started,
        "updated_at": utc_now(),
        "elapsed_seconds": time.monotonic() - extra.pop("monotonic_started"),
        "contains_raw_text": False,
        **extra,
    }
    atomic_write_json(path, value)


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": ordered[len(ordered) // 4],
        "median": ordered[len(ordered) // 2],
        "q75": ordered[(3 * len(ordered)) // 4],
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def run_p1(args):
    paths, plan, units, plan_encoded, readiness = build_preflight(args)
    if not paths["readiness"].is_file() or not paths["plan"].is_file():
        raise ValueError("P0 readiness and plan must exist before P1")
    if canonical_bytes(read_json(paths["readiness"])) != canonical_bytes(readiness):
        raise ValueError("current inputs/code differ from frozen P0 readiness")
    if paths["plan"].read_bytes() != plan_encoded:
        raise ValueError("frozen P0 plan bytes changed")
    readiness_sha256 = support.file_sha256(paths["readiness"])
    plan_file_sha256 = support.file_sha256(paths["plan"])

    import numpy as np
    import torch
    from run_certified_resegmentation_wp2_detector_equivalence import semstamp_core

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started_at = utc_now()
    monotonic_started = time.monotonic()
    write_heartbeat(
        paths["heartbeat"],
        0,
        0,
        started_at,
        "loading_detector",
        monotonic_started=monotonic_started,
    )
    load_started = time.monotonic()
    model, get_mask, official_detect = semstamp_core(
        paths["repo"], Path(args.embedder_dir).resolve()
    )
    model.batch_size = BATCH_SIZE
    relation = bytes(
        int(
            current
            in {
                int(value)
                for value in get_mask(3, 0.25, previous)
                .detach()
                .cpu()
                .tolist()
            }
        )
        for previous in range(8)
        for current in range(8)
    )
    relation = support.validate_relation(relation)
    torch.cuda.synchronize()
    model_load_seconds = time.monotonic() - load_started

    checkpoint_path = paths["checkpoint"]
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        validate_checkpoint(checkpoint, plan, relation)
    else:
        checkpoint = {
            "schema_version": 1,
            "contract_sha256": CONTRACT_SHA256,
            "plan_sha256": support.digest(plan),
            "replays": {"0": [], "1": []},
        }
        validate_checkpoint(checkpoint, plan, relation)
        support.atomic_write_checkpoint(checkpoint_path, checkpoint)

    with torch.inference_mode():
        for replay in range(REPLAY_COUNT):
            key = str(replay)
            entries = checkpoint["replays"][key]
            write_heartbeat(
                paths["heartbeat"],
                replay,
                len(entries),
                started_at,
                "detector_health",
                monotonic_started=monotonic_started,
                model_load_seconds=model_load_seconds,
            )
            for index in range(len(entries), TOTAL_ROWS):
                observed = np.asarray(
                    model.get_hash(list(units[index])),
                    dtype=np.uint8,
                )
                if (
                    observed.shape != (plan[index]["unit_count"],)
                    or np.any(observed > 7)
                ):
                    raise ValueError(f"invalid region output at plan index {index}")
                regions = [int(value) for value in observed]
                replay_model = ReplayLSHModel(regions)
                official_score = float(
                    official_detect(
                        list(units[index]),
                        replay_model,
                        lmbd=0.25,
                        lsh_dim=3,
                    )
                )
                replay_model.assert_exhausted()
                public = support.make_public_score(
                    plan[index], regions, relation, official_score
                )
                entry = {"public": public, "regions": regions}
                if replay and canonical_bytes(entry) != canonical_bytes(
                    checkpoint["replays"]["0"][index]
                ):
                    raise ValueError(f"replay mismatch at plan index {index}")
                checkpoint["replays"][key].append(entry)
                support.atomic_write_checkpoint(checkpoint_path, checkpoint)
                write_heartbeat(
                    paths["heartbeat"],
                    replay,
                    index + 1,
                    started_at,
                    "detector_health",
                    monotonic_started=monotonic_started,
                    model_load_seconds=model_load_seconds,
                )

    state = validate_checkpoint(checkpoint, plan, relation)
    if not state["complete"]:
        raise AssertionError("both B0 replays did not complete")
    summary = support.summarize_checkpoint(
        checkpoint, plan, relation, CONTRACT_SHA256
    )
    replay_rows = [
        [entry["public"] for entry in checkpoint["replays"][str(replay)]]
        for replay in range(REPLAY_COUNT)
    ]
    rows_encoded = [jsonl_bytes(rows) for rows in replay_rows]
    atomic_write_bytes(paths["rows0"], rows_encoded[0])
    atomic_write_bytes(paths["rows1"], rows_encoded[1])
    if rows_encoded[0] != rows_encoded[1]:
        raise AssertionError("public replay files differ")

    torch.cuda.synchronize()
    peak_cuda_bytes = torch.cuda.max_memory_allocated()
    total_cuda_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_fraction = peak_cuda_bytes / total_cuda_bytes
    stats = summary["statistics"]
    calibration_scores = [
        row["direct_score"]
        for row in replay_rows[0]
        if row["cohort"] == "calibration_null"
    ]
    score_groups = {
        cohort: [
            row["direct_score"]
            for row in replay_rows[0]
            if row["cohort"] == cohort
        ]
        for cohort in support.COHORTS
    }
    gates = {
        **summary["gates"],
        "no_sample_exclusion_or_supplement": len(replay_rows[0]) == TOTAL_ROWS,
        "no_nonfinite_scores": all(
            math.isfinite(value)
            for values in score_groups.values()
            for value in values
        ),
        "identity_and_structure_exact": True,
        "peak_vram_below_0_80": vram_fraction < MAX_VRAM_FRACTION,
        "p0_readiness_exact": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "go" if passed else "stop",
        "terminal_reason": (
            "b0_detector_health_go"
            if passed
            else "b0_stop_without_certificate"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": support.file_sha256(paths["runner"]),
        "support_sha256": support.file_sha256(paths["support"]),
        "test_sha256": support.file_sha256(paths["test"]),
        "p0_readiness_sha256": readiness_sha256,
        "plan": {
            "rows": TOTAL_ROWS,
            "canonical_sha256": support.digest(plan),
            "file_sha256": plan_file_sha256,
        },
        "detector": {
            "semstamp_revision": SEMSTAMP_REVISION,
            "data_revision": SEMSTAMP_DATA_REVISION,
            "model_revision": MODEL_REVISION,
            "relation_sha256": support.RELATION_SHA256,
            "batch_size": BATCH_SIZE,
            "device": "cuda:0",
            "arithmetic": "FP32",
            "model_load_seconds": model_load_seconds,
        },
        "statistics": stats,
        "diagnostics": {
            "threshold_ties": sum(
                score == stats["calibration"]["threshold"]
                for score in calibration_scores
            ),
            "score_sha256": {
                cohort: support.digest(values)
                for cohort, values in score_groups.items()
            },
            "score_distribution": {
                cohort: distribution(values)
                for cohort, values in score_groups.items()
            },
        },
        "gates": gates,
        "counts": {
            "cohort_size": COHORT_SIZE,
            "rows_per_replay": TOTAL_ROWS,
            "replays": REPLAY_COUNT,
            "official_detect_lsh_calls": REPLAY_COUNT * TOTAL_ROWS,
            "direct_score_computations": REPLAY_COUNT * TOTAL_ROWS,
            "sample_exclusions": 0,
            "sample_supplements": 0,
            "nonfinite_scores": 0,
            "replay_mismatches": 0,
            "official_direct_mismatches": 0,
        },
        "outputs": {
            "replay0": {
                **file_metadata(paths["rows0"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "replay1": {
                **file_metadata(paths["rows1"], paths["root"]),
                "rows": TOTAL_ROWS,
                "contains_raw_text": False,
            },
            "sealed_checkpoint": {
                **file_metadata(checkpoint_path, paths["root"]),
                "mode": "0600",
                "contains_raw_text": False,
            },
        },
        "runtime": {
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - monotonic_started,
            "device_name": torch.cuda.get_device_name(0),
            "peak_cuda_bytes": peak_cuda_bytes,
            "total_cuda_bytes": total_cuda_bytes,
            "peak_vram_fraction": vram_fraction,
            "environment": package_versions(),
        },
        "execution_attestation": {
            "detector_imports": 1,
            "detector_model_loads": 1,
            "detector_scores_computed": REPLAY_COUNT * TOTAL_ROWS,
            "generation_calls": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "decision": {
            "b0_detector_health": "go" if passed else "stop",
            "b1_exact_certificate": (
                "authorized_to_freeze_separate_protocol"
                if passed
                else "stop"
            ),
            "b1_started": False,
        },
        "contains_raw_text": False,
    }
    atomic_write_json(paths["result"], result)
    write_heartbeat(
        paths["heartbeat"],
        1,
        TOTAL_ROWS,
        started_at,
        "terminal",
        monotonic_started=monotonic_started,
        status=result["status"],
        terminal_reason=result["terminal_reason"],
        result_sha256=support.file_sha256(paths["result"]),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--embedder-dir", type=Path, required=True)
    value.add_argument("--nltk-data", type=Path, required=True)
    value.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    value.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    value.add_argument("--preflight-only", action="store_true")
    return value


def main():
    args = parser().parse_args()
    if args.preflight_only:
        paths, _, _, plan_encoded, readiness = build_preflight(args)
        atomic_write_bytes(paths["plan"], plan_encoded)
        atomic_write_json(paths["readiness"], readiness)
        print(json.dumps(readiness, sort_keys=True), flush=True)
        return
    run_p1(args)


if __name__ == "__main__":
    main()
