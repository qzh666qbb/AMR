"""Independently audit the terminal WP2-v17 B0 public evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

import certified_resegmentation_wp2_v13_b0_support as support


STAGE = "WP2-v17-B0-SemStamp-detector-health"
PROTOCOL_SHA256 = (
    "435234b89955935272f7fc59c8d7e6d04de0c9ee579556f06151e4dd63458dd4"
)
CONTRACT_SHA256 = (
    "ff5256aa75675536ad3c8dc200341be0eff726ae73d06db7bb65c0a056240848"
)
RUNNER_SHA256 = (
    "a15c7bfc65b6b7bea62bbeb012484532e389972e7d6d5bc45833e2a98d0f5c72"
)
SUPPORT_SHA256 = (
    "692f26230b7f335f715b23a05d2984d617d064e89db0d7ad0fa23fc4c2025166"
)
TEST_SHA256 = (
    "d7b466b606c285da455b3911d98e6ea59fcd8762356578797108ddf452188d4b"
)
P0_SHA256 = (
    "e1f8990e2612018eadccba2020346710a638efc1e73180389b4d8a54206e2725"
)
PLAN_SHA256 = (
    "a1427a9978751c756ad5072aeb45bb6cb393713d8339d62ff6ef4397591fc4e2"
)
RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
REPLAY_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
CHECKPOINT_SHA256 = (
    "b4fbd2d9d610fe170f1a4700ac35c2b1c405b1a145e8bc1d32cd7d7ec0f8d909"
)
RELATION_HEX = (
    "0000010000010000000001000000010000000100000100000100000000000100"
    "0000010000000001000100000100000001000001000000000000000100000100"
)
RELATION = bytes.fromhex(RELATION_HEX)
FORBIDDEN_PUBLIC_KEYS = {
    "source_text",
    "prompt",
    "generated_text",
    "raw_text",
    "text",
}


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


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(
                f"raw public key at {location}: {sorted(overlap)!r}"
            )
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def verify_sha(path, expected, label):
    if not Path(path).is_file() or support.file_sha256(path) != expected:
        raise ValueError(f"{label} SHA mismatch")


def atomic_write_json(path, value):
    path = Path(path)
    encoded = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def close_float(left, right):
    return abs(float(left) - float(right)) <= 1e-12


def validate_statistics(observed, expected):
    if set(observed) != set(expected):
        raise ValueError("statistics groups changed")
    for group in observed:
        if set(observed[group]) != set(expected[group]):
            raise ValueError("statistics fields changed")
        for field, value in observed[group].items():
            reference = expected[group][field]
            if isinstance(value, float):
                if not close_float(value, reference):
                    raise ValueError(f"statistics drift: {group}.{field}")
            elif value != reference:
                raise ValueError(f"statistics drift: {group}.{field}")


def run(root, output_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = {
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "B0_CONTRACT.md",
        "runner": root / "scripts/run_certified_resegmentation_wp2_v17_b0.py",
        "support": root / "scripts/certified_resegmentation_wp2_v13_b0_support.py",
        "test": root / "scripts/test_run_certified_resegmentation_wp2_v17_b0.py",
        "p0": output_dir / "P0_READINESS.json",
        "plan": output_dir / "B0_PLAN.jsonl",
        "result": output_dir / "B0_RESULT.json",
        "rows0": output_dir / "B0_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "B0_ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/b0_checkpoint.json",
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "contract": CONTRACT_SHA256,
        "runner": RUNNER_SHA256,
        "support": SUPPORT_SHA256,
        "test": TEST_SHA256,
        "p0": P0_SHA256,
        "plan": PLAN_SHA256,
        "result": RESULT_SHA256,
        "rows0": REPLAY_SHA256,
        "rows1": REPLAY_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
    }
    for label, digest in expected.items():
        verify_sha(paths[label], digest, label)
    if stat.S_IMODE(paths["checkpoint"].stat().st_mode) != 0o600:
        raise ValueError("terminal checkpoint is not mode 0600")
    if hashlib.sha256(RELATION).hexdigest() != support.RELATION_SHA256:
        raise ValueError("auditor relation bytes mismatch")
    relation = support.validate_relation(RELATION)

    p0 = read_json(paths["p0"])
    if (
        p0.get("status") != "pass"
        or p0.get("scope") != "identity_text_structure_only_no_detector"
        or p0.get("contains_raw_text") is not False
        or p0.get("plan", {}).get("rows") != 3000
        or p0.get("plan", {}).get("jsonl_sha256") != PLAN_SHA256
        or any(
            p0.get("execution_attestation", {}).get(field) != 0
            for field in (
                "detector_imports",
                "detector_model_loads",
                "detector_scores_computed",
                "generation_calls",
                "resegmentation_calls",
                "certificate_calls",
            )
        )
    ):
        raise ValueError("P0 readiness is not the frozen model-free Pass")

    plan = support.validate_plan(read_jsonl(paths["plan"]))
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    checkpoint = read_json(paths["checkpoint"])
    reject_raw_public_keys([p0, plan, rows0, rows1, checkpoint])
    if len(rows0) != 3000 or canonical_bytes(rows0) != canonical_bytes(rows1):
        raise ValueError("public replay rows are incomplete or different")
    for index, row in enumerate(rows0):
        support.validate_public_score(row, plan[index])
    state = support.validate_checkpoint(
        checkpoint, plan, relation, CONTRACT_SHA256
    )
    if not state["complete"]:
        raise ValueError("terminal checkpoint is incomplete")
    checkpoint_rows = [
        entry["public"] for entry in checkpoint["replays"]["0"]
    ]
    if canonical_bytes(checkpoint_rows) != canonical_bytes(rows0):
        raise ValueError("checkpoint/public rows mismatch")

    summary = support.summarize_checkpoint(
        checkpoint, plan, relation, CONTRACT_SHA256
    )
    result = read_json(paths["result"])
    reject_raw_public_keys(result)
    if (
        result.get("schema_version") != 1
        or result.get("stage") != STAGE
        or result.get("status") != "go"
        or result.get("terminal_reason") != "b0_detector_health_go"
        or result.get("protocol_sha256") != PROTOCOL_SHA256
        or result.get("contract_sha256") != CONTRACT_SHA256
        or result.get("runner_sha256") != RUNNER_SHA256
        or result.get("support_sha256") != SUPPORT_SHA256
        or result.get("test_sha256") != TEST_SHA256
        or result.get("p0_readiness_sha256") != P0_SHA256
        or result.get("contains_raw_text") is not False
        or result.get("plan", {}).get("file_sha256") != PLAN_SHA256
        or result.get("decision", {}).get("b0_detector_health") != "go"
        or result.get("decision", {}).get("b1_exact_certificate")
        != "authorized_to_freeze_separate_protocol"
        or result.get("decision", {}).get("b1_started") is not False
        or not all(result.get("gates", {}).values())
    ):
        raise ValueError("B0 result terminal Go contract mismatch")
    validate_statistics(result["statistics"], summary["statistics"])
    if result["outputs"]["replay0"]["sha256"] != REPLAY_SHA256:
        raise ValueError("replay 0 output binding mismatch")
    if result["outputs"]["replay1"]["sha256"] != REPLAY_SHA256:
        raise ValueError("replay 1 output binding mismatch")
    if result["outputs"]["sealed_checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("checkpoint output binding mismatch")
    counts = result.get("counts", {})
    if (
        counts.get("rows_per_replay") != 3000
        or counts.get("replays") != 2
        or counts.get("official_detect_lsh_calls") != 6000
        or counts.get("direct_score_computations") != 6000
        or any(
            counts.get(field) != 0
            for field in (
                "sample_exclusions",
                "sample_supplements",
                "nonfinite_scores",
                "replay_mismatches",
                "official_direct_mismatches",
            )
        )
    ):
        raise ValueError("B0 result count mismatch")

    audit = {
        "schema_version": 1,
        "stage": f"{STAGE}-terminal-audit",
        "status": "go",
        "scope": "public_and_sealed_structural_recalculation_no_model_load",
        "bindings": expected,
        "checks": {
            "p0_model_free_pass": True,
            "plan_3000_exact": True,
            "two_replays_byte_exact": True,
            "checkpoint_complete_and_private": True,
            "all_public_rows_reconstructed": True,
            "statistics_recomputed": True,
            "all_b0_gates_pass": True,
            "raw_text_absent_from_public_artifacts": True,
        },
        "statistics": summary["statistics"],
        "decision": {
            "b0_detector_health": "confirmed_go",
            "b1_exact_certificate": "not_started",
            "next_action": "freeze a separate B1 protocol before any certificate run",
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
            "certificate_calls": 0,
        },
        "contains_raw_text": False,
    }
    atomic_write_json(output, audit)
    print(json.dumps(audit, sort_keys=True))
    return audit


def main():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    args = value.parse_args()
    run(args.root, args.output_dir, args.output)


if __name__ == "__main__":
    main()
