"""Run the SemStamp certified-defense v2 WP1 refined-atom audit."""

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import tempfile

import run_certified_resegmentation_wp2_v17_b0 as v17
from semstamp_refined_atom_renderer import (
    RefinedAtomRendererError,
    canonicalize_refined_bytes,
)


STAGE = "semstamp_certified_defense_v2_wp1_threat_model"
PROTOCOL_SHA256 = (
    "3a77a6c1d9f8334e49f1def49c2be106ccbac07a0ea9ae2c110b94e8baccd206"
)
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
REPLAY_COUNT = 2
LEGACY_FAILURE_COUNT = 42
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_digest(value):
    return sha256_bytes(canonical_bytes(value))


def jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) for row in rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def atomic_write_bytes(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
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
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value, mode=0o600):
    atomic_write_bytes(path, canonical_bytes(value), mode)


def file_metadata(path, root=None):
    path = Path(path)
    result = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if root is not None:
        result["relative_path"] = str(
            path.resolve().relative_to(Path(root).resolve())
        )
    return result


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(
                f"raw text key in public artifact at {location}: {overlap}"
            )
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("output directory must be inside root")
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "threat_model": output_dir / "THREAT_MODEL.md",
        "admissible": output_dir / "ADMISSIBLE_RESEGMENTATION.md",
        "threat_json": output_dir / "THREAT_MODEL.json",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "rows0": output_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "P0_RENDERER_ROWS_REPLAY1.jsonl",
        "replay0": output_dir / "REPLAY0_RESULT.json",
        "replay1": output_dir / "REPLAY1_RESULT.json",
        "result": output_dir / "RESULT.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "runner": Path(__file__).resolve(),
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "renderer_test": (
            root / "scripts/test_semstamp_refined_atom_renderer.py"
        ),
        "runner_test": (
            root / "scripts/test_semstamp_certified_defense_v2_wp1.py"
        ),
        "auditor": (
            root / "scripts/audit_semstamp_certified_defense_v2_wp1.py"
        ),
        "v17_dir": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
        ),
        "v1_dir": (
            root
            / "experiments/semstamp_certified_defense_v1_wp3_detector_health1"
        ),
    }


def parse_manifest(path):
    entries = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in entries
        ):
            raise ValueError(f"invalid input manifest line {line_number}")
        entries[relative] = digest
    if not entries:
        raise ValueError("input manifest is empty")
    return entries


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen",
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "cohorts": list(COHORTS),
        "cohort_size": COHORT_SIZE,
        "total_rows": TOTAL_ROWS,
        "replay_count": REPLAY_COUNT,
        "legacy_failure_count": LEGACY_FAILURE_COUNT,
        "nltk_version": "3.8.1",
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    artifacts = {
        "protocol_sha256": paths["protocol"],
        "threat_model_sha256": paths["threat_model"],
        "admissible_contract_sha256": paths["admissible"],
        "threat_model_json_sha256": paths["threat_json"],
        "runner_sha256": paths["runner"],
        "renderer_sha256": paths["renderer"],
        "renderer_test_sha256": paths["renderer_test"],
        "runner_test_sha256": paths["runner_test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["manifest"],
    }
    for field, path in artifacts.items():
        if contract.get(field) != sha256_file(path):
            raise ValueError(f"contract artifact mismatch: {field}")
    manifest = parse_manifest(paths["manifest"])
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(manifest):
        raise ValueError("contract/input manifest path set mismatch")
    for relative, digest in manifest.items():
        meta = inputs[relative]
        if (
            not isinstance(meta, dict)
            or meta.get("sha256") != digest
            or type(meta.get("bytes")) is not int
            or meta["bytes"] < 1
        ):
            raise ValueError(f"contract input metadata mismatch: {relative}")
    return contract, manifest


def verify_inputs(paths, contract, manifest):
    observed = {}
    for relative, expected_sha in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
        ):
            raise ValueError(f"input is missing or escapes root: {relative}")
        meta = contract["inputs"][relative]
        if (
            sha256_file(path) != expected_sha
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"input bytes/SHA mismatch: {relative}")
        expected_mode = meta.get("mode")
        if (
            expected_mode is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != expected_mode
        ):
            raise ValueError(f"input mode mismatch: {relative}")
        observed[relative] = {
            "bytes": meta["bytes"],
            "sha256": expected_sha,
            **({"mode": expected_mode} if expected_mode else {}),
        }
    return observed


def validate_parent_evidence(paths):
    v17_result = read_json(paths["v17_dir"] / "B0_RESULT.json")
    v17_audit = read_json(paths["v17_dir"] / "B0_AUDIT.json")
    v1_readiness = read_json(paths["v1_dir"] / "P0_READINESS.json")
    v1_audit = read_json(paths["v1_dir"] / "AUDIT.json")
    legacy_indices = v1_readiness["renderer"]["failed_plan_indices"]
    if (
        v17_result.get("status") != "go"
        or not all(v17_result.get("gates", {}).values())
        or v17_audit.get("status") != "go"
        or not all(v17_audit.get("checks", {}).values())
        or v1_readiness.get("status") != "stop"
        or v1_readiness.get("terminal_reason") != "renderer_contract_stop"
        or v1_readiness["renderer"].get("invalid_rows")
        != LEGACY_FAILURE_COUNT
        or len(legacy_indices) != LEGACY_FAILURE_COUNT
        or len(set(legacy_indices)) != LEGACY_FAILURE_COUNT
        or v1_audit.get("status") != "stop"
        or v1_audit.get("counts", {}).get("invalid_rows")
        != LEGACY_FAILURE_COUNT
    ):
        raise ValueError("parent v17/v1 evidence is not the frozen terminal")
    return tuple(legacy_indices)


def materialize_inputs(paths):
    b0_paths = v17.paths_for(paths["root"], paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    if (
        v17.jsonl_bytes(plan)
        != (paths["v17_dir"] / "B0_PLAN.jsonl").read_bytes()
        or len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
    ):
        raise ValueError("materialized inputs differ from frozen v17 plan")
    return plan, texts


def load_nltk(nltk_data, contract):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != contract["nltk_version"]:
        raise ValueError("NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    resource = nltk.data.find("tokenizers/punkt/english.pickle")
    with resource.open() as handle:
        payload_sha256 = sha256_bytes(handle.read())
    if payload_sha256 != contract["punkt_payload_sha256"]:
        raise ValueError("Punkt payload SHA mismatch")
    return sent_tokenize


def renderer_row(index, planned, document, legacy_failure_indices):
    evidence = document.public_evidence()
    if (
        document.raw_utf8_sha256 != planned["raw_text_sha256"]
        or document.unit_count != planned["unit_count"]
        or evidence["byte_reconstruction_sha256"]
        != document.raw_utf8_sha256
        or not set(document.zero_byte_gap_boundaries).issubset(
            document.original_boundaries
        )
    ):
        raise ValueError(f"renderer identity mismatch at row {index}")
    legality_sha256 = canonical_digest({
        "scope": "b0_only_inherited_v17_native_legality",
        "native_token_limit": 512,
        "original_unit_atom_spans": [
            list(span) for span in document.original_unit_atom_spans
        ],
        "v17_max_native_tokens": planned["max_native_tokens"],
    })
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "status": "pass",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "max_native_tokens": planned["max_native_tokens"],
        "legacy_v1_alignment_failure": index in legacy_failure_indices,
        **evidence,
        "legality_sha256": legality_sha256,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row


def failure_row(index, planned, raw, units, error, legacy_failure_indices):
    row = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "status": "invalid_or_incomplete",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": sha256_bytes(raw),
        "raw_utf8_bytes": len(raw),
        "unit_count": planned["unit_count"],
        "observed_nltk_unit_count": len(units),
        "legacy_v1_alignment_failure": index in legacy_failure_indices,
        "failure_code": error.code,
        "failure_detail": error.detail,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row


def build_rows(paths, contract, legacy_failure_indices, nltk_data):
    plan, texts = materialize_inputs(paths)
    sent_tokenize = load_nltk(nltk_data, contract)
    rows = []
    for index, (planned, text) in enumerate(zip(plan, texts)):
        raw = text.encode("utf-8")
        units = sent_tokenize(text)
        try:
            document = canonicalize_refined_bytes(
                raw,
                lambda _value, frozen=units: frozen,
            )
        except RefinedAtomRendererError as error:
            rows.append(
                failure_row(
                    index,
                    planned,
                    raw,
                    units,
                    error,
                    legacy_failure_indices,
                )
            )
        else:
            rows.append(
                renderer_row(
                    index,
                    planned,
                    document,
                    legacy_failure_indices,
                )
            )
    return rows


def summarize_rows(rows):
    valid = [row for row in rows if row["status"] == "pass"]
    failures = [row for row in rows if row["status"] != "pass"]
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "invalid_rows": len(failures),
        "cohort_counts": dict(Counter(row["cohort"] for row in rows)),
        "raw_identity_unique": (
            len({row["raw_text_sha256"] for row in rows}) == len(rows)
        ),
        "failure_code_counts": dict(
            Counter(row["failure_code"] for row in failures)
        ),
        "lexical_span_total": sum(
            row["lexical_span_count"] for row in valid
        ),
        "refined_atom_total": sum(row["atom_count"] for row in valid),
        "refinement_total": sum(row["refinement_count"] for row in valid),
        "rows_with_refinement": sum(
            row["refinement_count"] > 0 for row in valid
        ),
        "zero_byte_gap_total": sum(
            row["zero_byte_gap_count"] for row in valid
        ),
        "rows_with_zero_byte_gap": sum(
            row["zero_byte_gap_count"] > 0 for row in valid
        ),
        "legacy_v1_failures_present": sum(
            row["legacy_v1_alignment_failure"] for row in rows
        ),
        "legacy_v1_failures_repaired": sum(
            row["legacy_v1_alignment_failure"] and row["status"] == "pass"
            for row in rows
        ),
        "topology_sha256": canonical_digest(
            [row["topology_sha256"] for row in valid]
        ),
        "legality_sha256": canonical_digest(
            [row["legality_sha256"] for row in valid]
        ),
        "reconstruction_sha256": canonical_digest(
            [row["byte_reconstruction_sha256"] for row in valid]
        ),
    }


def prepare(args):
    paths = paths_for(args.root, args.output_dir)
    contract, manifest = validate_contract(paths)
    observed_inputs = verify_inputs(paths, contract, manifest)
    legacy_failure_indices = validate_parent_evidence(paths)
    return paths, contract, observed_inputs, legacy_failure_indices


def run_replay(args, replay_index):
    paths, contract, observed_inputs, legacy_indices = prepare(args)
    rows = build_rows(paths, contract, legacy_indices, args.nltk_data)
    encoded = jsonl_bytes(rows)
    output = paths[f"rows{replay_index}"]
    atomic_write_bytes(output, encoded)
    summary = summarize_rows(rows)
    replay_result = {
        "schema_version": 1,
        "stage": STAGE,
        "phase": f"replay{replay_index}",
        "status": "pass" if summary["invalid_rows"] == 0 else "stop",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "renderer_rows": {
            **file_metadata(output, paths["root"]),
            "rows": TOTAL_ROWS,
            "contains_raw_text": False,
        },
        "summary": summary,
        "artifacts": {
            "runner": file_metadata(paths["runner"], paths["root"]),
            "renderer": file_metadata(paths["renderer"], paths["root"]),
            "inputs": observed_inputs,
        },
        "environment": {
            "python": platform.python_version(),
            "nltk": metadata.version("nltk"),
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
            "score_access_count_during_construction": 0,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(replay_result)
    atomic_write_json(paths[f"replay{replay_index}"], replay_result)
    if replay_index == 1 and paths["rows0"].is_file():
        if paths["rows0"].read_bytes() != encoded:
            raise ValueError("TOPOLOGY_NONDETERMINISTIC")
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(replay_result, sort_keys=True))
    return replay_result


def finalize(args):
    paths, contract, _observed_inputs, legacy_indices = prepare(args)
    if not all(
        paths[name].is_file()
        for name in ("rows0", "rows1", "replay0", "replay1")
    ):
        raise ValueError("both replay artifacts are required")
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    replay0 = read_json(paths["replay0"])
    replay1 = read_json(paths["replay1"])
    reject_raw_public_keys([rows0, rows1, replay0, replay1])
    summary = summarize_rows(rows0)
    exact_replay = paths["rows0"].read_bytes() == paths["rows1"].read_bytes()
    gates = {
        "fixed_3000_rows": (
            len(rows0) == TOTAL_ROWS
            and len(rows1) == TOTAL_ROWS
            and summary["cohort_counts"]
            == {cohort: COHORT_SIZE for cohort in COHORTS}
        ),
        "all_raw_identities_unique": summary["raw_identity_unique"],
        "all_original_partitions_exact": summary["invalid_rows"] == 0,
        "all_raw_bytes_reconstructed": all(
            row.get("byte_reconstruction_sha256")
            == row.get("raw_text_sha256")
            for row in rows0
        ),
        "all_zero_byte_gaps_authorized": all(
            row.get("zero_byte_gaps_all_original") is True
            for row in rows0
        ),
        "legacy_42_retained": (
            summary["legacy_v1_failures_present"] == LEGACY_FAILURE_COUNT
            and len(legacy_indices) == LEGACY_FAILURE_COUNT
        ),
        "legacy_42_repaired": (
            summary["legacy_v1_failures_repaired"] == LEGACY_FAILURE_COUNT
        ),
        "two_replays_byte_exact": exact_replay,
        "replay_summaries_exact": (
            replay0.get("status") == "pass"
            and replay1.get("status") == "pass"
            and replay0.get("summary") == replay1.get("summary")
        ),
        "score_access_zero": all(
            row.get("score_access_count_during_construction") == 0
            for row in rows0
        ),
        "model_gpu_generation_calls_zero": all(
            result["execution_attestation"][field] == 0
            for result in (replay0, replay1)
            for field in (
                "certificate_calls",
                "detector_imports",
                "detector_model_loads",
                "detector_score_reads",
                "detector_scores_computed",
                "generation_calls",
                "gpu_calls",
                "score_access_count_during_construction",
            )
        ),
        "public_artifacts_contain_no_raw_text": True,
    }
    status = "pass" if all(gates.values()) else "stop"
    result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "terminal_reason": (
            "wp1_execution_pass_pending_independent_audit"
            if status == "pass"
            else "wp1_definition_stop"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "renderer_contract": contract["renderer_contract"],
        "gates": gates,
        "counts": summary,
        "outputs": {
            "replay0": file_metadata(paths["rows0"], paths["root"]),
            "replay1": file_metadata(paths["rows1"], paths["root"]),
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
            "score_access_count_during_construction": 0,
            "raw_text_emitted_to_public_artifacts": False,
            "wp2_started": False,
            "wp3_started": False,
        },
        "decision": {
            "wp1": (
                "pending_independent_audit"
                if status == "pass"
                else "stop"
            ),
            "wp2": "not_authorized",
            "wp2_started": False,
            "wp3_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(result)
    atomic_write_json(paths["result"], result)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))
    if status != "pass":
        raise RuntimeError("WP1 definition audit failed")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("replay0", "replay1", "finalize"),
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    try:
        if args.phase == "replay0":
            run_replay(args, 0)
        elif args.phase == "replay1":
            run_replay(args, 1)
        else:
            finalize(args)
    except Exception as error:
        atomic_write_json(paths["last_failure"], {
            "schema_version": 1,
            "stage": STAGE,
            "status": "invalid_or_incomplete",
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "contains_raw_text": False,
        })
        raise


if __name__ == "__main__":
    main()
