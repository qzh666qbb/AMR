"""Run the model-free v2 WP3 P0 refined-renderer readiness gate."""

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile

import run_certified_resegmentation_wp2_v17_b0 as v17
from semstamp_refined_atom_renderer import (
    RefinedAtomRendererError,
    canonicalize_refined_bytes,
)


STAGE = "semstamp_certified_defense_v2_wp3_detector_health_p0"
PROTOCOL_SHA256 = (
    "3d6d071e64a9bba12d5cf97d78b1f8ca8ce73c8318def7cbc2fef6bc0299acb3"
)
WP1_PROTOCOL_SHA256 = (
    "3a77a6c1d9f8334e49f1def49c2be106ccbac07a0ea9ae2c110b94e8baccd206"
)
SEMSTAMP_REVISION = "97db73d11fd80f376a02b0a604d500627622f7e6"
SEMSTAMP_DATA_REVISION = "bd6e948de2a6fc30f7670dfad347c2b79e252687"
MODEL_REVISION = "a77dce8530e100ac1101981e4be390546c1cb282"
RELATION_SHA256 = (
    "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
)
PUNKT_SHA256 = (
    "5cad3758596392364e3be9803dbd7ebeda384b68937b488a01365f5551bb942c"
)
V17_B0_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
V17_B0_AUDIT_SHA256 = (
    "bb152276a01c1b7a78671061a30ac6219a3905fcafebb454ba2a270782a8f342"
)
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
REPLAY_COUNT = 2
LEGACY_FAILURE_COUNT = 42
EXPECTED_ZERO_GAP_COUNT = 59
SMOKE_PER_COHORT = 32
SMOKE_ROWS = 96
NATIVE_TOKEN_LIMIT = 512
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


def renderer_digest(value):
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )


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
    result = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
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
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "rows0": output_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "P0_RENDERER_ROWS_REPLAY1.jsonl",
        "smoke0": output_dir / "P0_SMOKE_PLAN_REPLAY0.jsonl",
        "smoke1": output_dir / "P0_SMOKE_PLAN_REPLAY1.jsonl",
        "smoke": output_dir / "SMOKE_PLAN.jsonl",
        "replay0": output_dir / "REPLAY0_RESULT.json",
        "replay1": output_dir / "REPLAY1_RESULT.json",
        "readiness": output_dir / "P0_READINESS.json",
        "last_failure": output_dir / "LAST_FAILURE.json",
        "runner": Path(__file__).resolve(),
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "test": root / "scripts/test_semstamp_certified_defense_v2_wp3_p0.py",
        "auditor": root / "scripts/audit_semstamp_certified_defense_v2_wp3_p0.py",
        "v17_dir": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
        ),
        "v1_wp3_dir": (
            root
            / "experiments/semstamp_certified_defense_v1_wp3_detector_health1"
        ),
        "v2_wp1_dir": (
            root
            / "experiments/semstamp_certified_defense_v2_wp1_threat_model1"
        ),
        "v2_wp2_dir": (
            root
            / "experiments/semstamp_certified_defense_v2_wp2_algorithm1"
        ),
        "repo": root / "repos/SemStamp",
        "data_repo": root / "repos/semstamp-data",
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
        "expected_zero_gap_count": EXPECTED_ZERO_GAP_COUNT,
        "smoke_per_cohort": SMOKE_PER_COHORT,
        "smoke_rows": SMOKE_ROWS,
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "nltk_version": "3.8.1",
        "punkt_payload_sha256": PUNKT_SHA256,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
        "smoke_execution_authorized": False,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    artifacts = {
        "runner_sha256": paths["runner"],
        "renderer_sha256": paths["renderer"],
        "test_sha256": paths["test"],
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


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision mismatch")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision mismatch")
    return observed


def validate_parent_evidence(paths):
    v17_readiness = read_json(paths["v17_dir"] / "P0_READINESS.json")
    v1_readiness = read_json(paths["v1_wp3_dir"] / "P0_READINESS.json")
    v1_audit = read_json(paths["v1_wp3_dir"] / "AUDIT.json")
    wp1_result = read_json(paths["v2_wp1_dir"] / "RESULT.json")
    wp1_audit = read_json(paths["v2_wp1_dir"] / "AUDIT.json")
    wp2_result = read_json(paths["v2_wp2_dir"] / "RESULT.json")
    wp2_audit = read_json(paths["v2_wp2_dir"] / "AUDIT.json")
    legacy_indices = tuple(
        v1_readiness["renderer"]["failed_plan_indices"]
    )
    if (
        sha256_file(paths["v17_dir"] / "B0_RESULT.json")
        != V17_B0_RESULT_SHA256
        or sha256_file(paths["v17_dir"] / "B0_AUDIT.json")
        != V17_B0_AUDIT_SHA256
        or v17_readiness.get("status") != "pass"
        or not all(v17_readiness.get("gates", {}).values())
        or v1_readiness.get("status") != "stop"
        or v1_readiness.get("terminal_reason") != "renderer_contract_stop"
        or v1_audit.get("status") != "stop"
        or len(legacy_indices) != LEGACY_FAILURE_COUNT
        or wp1_result.get("status") != "pass"
        or not all(wp1_result.get("gates", {}).values())
        or wp1_audit.get("status") != "go"
        or wp2_result.get("terminal")
        != "WP2_ALGORITHM_GO_FOR_V2_WP3_DESIGN"
        or wp2_result.get("wp3_started") is not False
        or wp2_audit.get("status") != "go"
    ):
        raise ValueError("parent evidence is not at the frozen terminal")
    return legacy_indices


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


def structural_row(index, planned, document, legacy_indices):
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
        "native_token_limit": NATIVE_TOKEN_LIMIT,
        "original_unit_atom_spans": [
            list(span) for span in document.original_unit_atom_spans
        ],
        "v17_max_native_tokens": planned["max_native_tokens"],
    })
    return {
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "status": "pass",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "max_native_tokens": planned["max_native_tokens"],
        "legacy_v1_alignment_failure": index in legacy_indices,
        **evidence,
        "legality_sha256": legality_sha256,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }


def p0_row(index, planned, document, legacy_indices, parent_row):
    structural = structural_row(
        index, planned, document, legacy_indices
    )
    expected_parent = {
        "schema_version": 1,
        "protocol_sha256": WP1_PROTOCOL_SHA256,
        **structural,
    }
    if canonical_bytes(expected_parent) != canonical_bytes(parent_row):
        raise ValueError(f"v2 WP1 structural drift at row {index}")
    row = {
        "schema_version": 1,
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA256,
        **structural,
        "wp1_reference_exact": True,
        "wp1_reference_row_sha256": sha256_bytes(
            canonical_bytes(parent_row)
        ),
    }
    reject_raw_public_keys(row)
    return row


def failure_row(index, planned, raw, units, error):
    row = {
        "schema_version": 1,
        "stage": STAGE,
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
        "failure_code": error.code,
        "failure_detail": error.detail,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
    }
    reject_raw_public_keys(row)
    return row


def select_smoke_rows(rows):
    selected = []
    for cohort in COHORTS:
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        if len(cohort_rows) != COHORT_SIZE:
            raise ValueError(f"renderer cohort count mismatch: {cohort}")
        mandatory = sorted(
            (
                row for row in cohort_rows
                if (
                    row["refinement_count"] > 0
                    or row["zero_byte_gap_count"] > 0
                )
            ),
            key=lambda row: row["plan_index"],
        )
        if len(mandatory) > SMOKE_PER_COHORT:
            raise ValueError(
                f"mandatory smoke rows exceed budget for {cohort}"
            )
        mandatory_indices = {
            row["plan_index"] for row in mandatory
        }
        remaining = sorted(
            (
                row for row in cohort_rows
                if row["plan_index"] not in mandatory_indices
            ),
            key=lambda row: (
                row["atom_count"],
                row["unit_count"],
                row["raw_text_sha256"],
                row["plan_index"],
            ),
        )
        fill_count = SMOKE_PER_COHORT - len(mandatory)
        ranks = [
            ((2 * index + 1) * len(remaining)) // (2 * fill_count)
            for index in range(fill_count)
        ]
        if len(set(ranks)) != fill_count:
            raise AssertionError("smoke quantile ranks are not unique")
        choices = [
            ("mandatory_refinement", None, row)
            for row in mandatory
        ] + [
            ("structural_quantile", rank, remaining[rank])
            for rank in ranks
        ]
        for cohort_selection_index, (
            selection_class,
            structural_rank,
            row,
        ) in enumerate(choices):
            selected.append({
                "schema_version": 1,
                "stage": STAGE,
                "protocol_sha256": PROTOCOL_SHA256,
                "smoke_index": len(selected),
                "cohort": cohort,
                "cohort_selection_index": cohort_selection_index,
                "selection_class": selection_class,
                "structural_rank": structural_rank,
                "plan_index": row["plan_index"],
                "cohort_index": row["cohort_index"],
                "raw_text_sha256": row["raw_text_sha256"],
                "atom_count": row["atom_count"],
                "unit_count": row["unit_count"],
                "refinement_count": row["refinement_count"],
                "zero_byte_gap_count": row["zero_byte_gap_count"],
                "topology_sha256": row["topology_sha256"],
                "legality_sha256": row["legality_sha256"],
                "wp1_reference_row_sha256": (
                    row["wp1_reference_row_sha256"]
                ),
                "score_access_count_during_construction": 0,
                "contains_raw_text": False,
            })
    risk_indices = {
        row["plan_index"]
        for row in rows
        if (
            row["refinement_count"] > 0
            or row["zero_byte_gap_count"] > 0
        )
    }
    selected_indices = {row["plan_index"] for row in selected}
    if (
        len(selected) != SMOKE_ROWS
        or len(selected_indices) != SMOKE_ROWS
        or not risk_indices.issubset(selected_indices)
        or Counter(row["cohort"] for row in selected)
        != {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
    ):
        raise AssertionError("invalid risk-complete smoke selection")
    return selected


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
        "refined_atom_total": sum(
            row["atom_count"] for row in valid
        ),
        "refinement_total": sum(
            row["refinement_count"] for row in valid
        ),
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
        "wp1_reference_exact": all(
            row["wp1_reference_exact"] is True for row in valid
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
        "wp1_reference_rows_sha256": canonical_digest(
            [row["wp1_reference_row_sha256"] for row in valid]
        ),
    }


def summarize_smoke(plan):
    return {
        "rows": len(plan),
        "cohort_counts": dict(Counter(
            row["cohort"] for row in plan
        )),
        "mandatory_rows": sum(
            row["selection_class"] == "mandatory_refinement"
            for row in plan
        ),
        "quantile_rows": sum(
            row["selection_class"] == "structural_quantile"
            for row in plan
        ),
        "refinement_rows": sum(
            row["refinement_count"] > 0 for row in plan
        ),
        "zero_gap_rows": sum(
            row["zero_byte_gap_count"] > 0 for row in plan
        ),
        "zero_gap_total": sum(
            row["zero_byte_gap_count"] for row in plan
        ),
        "score_access_zero": all(
            row["score_access_count_during_construction"] == 0
            for row in plan
        ),
    }


def prepare(args):
    paths = paths_for(args.root, args.output_dir)
    contract, manifest = validate_contract(paths)
    observed_inputs = verify_inputs(paths, contract, manifest)
    legacy_indices = validate_parent_evidence(paths)
    parent_rows = read_jsonl(
        paths["v2_wp1_dir"] / "P0_RENDERER_ROWS_REPLAY0.jsonl"
    )
    if (
        len(parent_rows) != TOTAL_ROWS
        or sha256_file(
            paths["v2_wp1_dir"] / "P0_RENDERER_ROWS_REPLAY0.jsonl"
        )
        != contract["wp1_reference_rows_sha256"]
    ):
        raise ValueError("v2 WP1 reference rows mismatch")
    return (
        paths,
        contract,
        observed_inputs,
        legacy_indices,
        parent_rows,
    )


def build_replay(args):
    (
        paths,
        contract,
        observed_inputs,
        legacy_indices,
        parent_rows,
    ) = prepare(args)
    plan, texts = materialize_inputs(paths)
    sent_tokenize = load_nltk(args.nltk_data, contract)
    rows = []
    for index, (planned, text, parent_row) in enumerate(
        zip(plan, texts, parent_rows)
    ):
        raw = text.encode("utf-8")
        units = sent_tokenize(text)
        try:
            document = canonicalize_refined_bytes(
                raw,
                lambda _value, frozen=units: frozen,
            )
        except RefinedAtomRendererError as error:
            rows.append(failure_row(
                index, planned, raw, units, error
            ))
        else:
            rows.append(p0_row(
                index,
                planned,
                document,
                legacy_indices,
                parent_row,
            ))
    smoke_plan = (
        select_smoke_rows(rows)
        if all(row["status"] == "pass" for row in rows)
        else []
    )
    return (
        paths,
        contract,
        observed_inputs,
        rows,
        smoke_plan,
    )


def run_replay(args, replay_index):
    paths, contract, observed_inputs, rows, smoke_plan = build_replay(args)
    rows_encoded = jsonl_bytes(rows)
    smoke_encoded = jsonl_bytes(smoke_plan)
    atomic_write_bytes(paths[f"rows{replay_index}"], rows_encoded)
    atomic_write_bytes(paths[f"smoke{replay_index}"], smoke_encoded)
    summary = summarize_rows(rows)
    smoke_summary = summarize_smoke(smoke_plan)
    status = (
        "pass"
        if (
            summary["invalid_rows"] == 0
            and smoke_summary["rows"] == SMOKE_ROWS
        )
        else "stop"
    )
    replay = {
        "schema_version": 1,
        "stage": STAGE,
        "phase": f"replay{replay_index}",
        "status": status,
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "renderer_contract": contract["renderer_contract"],
        "renderer_rows": {
            **file_metadata(
                paths[f"rows{replay_index}"], paths["root"]
            ),
            "rows": TOTAL_ROWS,
            "contains_raw_text": False,
        },
        "smoke_plan": {
            **file_metadata(
                paths[f"smoke{replay_index}"], paths["root"]
            ),
            "rows": len(smoke_plan),
            "contains_raw_text": False,
        },
        "summary": summary,
        "smoke_summary": smoke_summary,
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
            "raw_text_emitted_to_public_artifacts": False,
            "sealed_inputs_parsed": True,
            "smoke_executed": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(replay)
    atomic_write_json(paths[f"replay{replay_index}"], replay)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(replay, sort_keys=True))
    if status != "pass":
        raise RuntimeError("WP3 P0 replay stopped")
    return replay


def finalize(args):
    paths, contract, _inputs, _legacy, _parent = prepare(args)
    required = (
        "rows0", "rows1", "smoke0", "smoke1", "replay0", "replay1"
    )
    if not all(paths[name].is_file() for name in required):
        raise ValueError("both complete P0 replays are required")
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    smoke0 = read_jsonl(paths["smoke0"])
    smoke1 = read_jsonl(paths["smoke1"])
    replay0 = read_json(paths["replay0"])
    replay1 = read_json(paths["replay1"])
    reject_raw_public_keys([
        rows0, rows1, smoke0, smoke1, replay0, replay1
    ])
    summary = summarize_rows(rows0)
    smoke_summary = summarize_smoke(smoke0)
    rows_exact = paths["rows0"].read_bytes() == paths["rows1"].read_bytes()
    smoke_exact = (
        paths["smoke0"].read_bytes() == paths["smoke1"].read_bytes()
    )
    risk_indices = {
        row["plan_index"]
        for row in rows0
        if (
            row["refinement_count"] > 0
            or row["zero_byte_gap_count"] > 0
        )
    }
    smoke_indices = {row["plan_index"] for row in smoke0}
    gates = {
        "contract_and_inputs_exact": True,
        "parent_terminals_exact": True,
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
        "legacy_42_retained_and_repaired": (
            summary["legacy_v1_failures_present"]
            == LEGACY_FAILURE_COUNT
            and summary["legacy_v1_failures_repaired"]
            == LEGACY_FAILURE_COUNT
        ),
        "expected_59_zero_gaps": (
            summary["zero_byte_gap_total"] == EXPECTED_ZERO_GAP_COUNT
        ),
        "wp1_structural_reference_exact": (
            summary["wp1_reference_exact"] is True
        ),
        "renderer_replays_byte_exact": rows_exact,
        "smoke_plans_byte_exact": smoke_exact,
        "replay_summaries_exact": (
            replay0.get("status") == "pass"
            and replay1.get("status") == "pass"
            and replay0.get("summary") == replay1.get("summary")
            and replay0.get("smoke_summary")
            == replay1.get("smoke_summary")
        ),
        "smoke_96_and_balanced": (
            len(smoke0) == SMOKE_ROWS
            and smoke_summary["cohort_counts"]
            == {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        ),
        "smoke_covers_all_refinement_risk_rows": (
            len(risk_indices) == LEGACY_FAILURE_COUNT
            and risk_indices.issubset(smoke_indices)
            and smoke_summary["mandatory_rows"] == LEGACY_FAILURE_COUNT
            and smoke_summary["zero_gap_total"]
            == EXPECTED_ZERO_GAP_COUNT
        ),
        "score_access_zero": all(
            row.get("score_access_count_during_construction") == 0
            for row in (*rows0, *smoke0)
        ),
        "model_gpu_score_certificate_calls_zero": all(
            replay["execution_attestation"][field] == 0
            for replay in (replay0, replay1)
            for field in (
                "certificate_calls",
                "detector_imports",
                "detector_model_loads",
                "detector_score_reads",
                "detector_scores_computed",
                "generation_calls",
                "gpu_calls",
            )
        ),
        "smoke_not_executed": all(
            replay["execution_attestation"]["smoke_executed"] is False
            for replay in (replay0, replay1)
        ),
        "public_artifacts_contain_no_raw_text": True,
    }
    status = "pass" if all(gates.values()) else "stop"
    if status == "pass":
        atomic_write_bytes(paths["smoke"], paths["smoke0"].read_bytes())
    else:
        paths["smoke"].unlink(missing_ok=True)
    readiness = {
        "schema_version": 1,
        "stage": STAGE,
        "status": status,
        "terminal_reason": (
            "p0_pass_pending_independent_audit"
            if status == "pass"
            else "WP3_P0_STOP"
        ),
        "scope": "refined_renderer_all_3000_model_free_double_replay",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "renderer_contract": contract["renderer_contract"],
        "gates": gates,
        "counts": summary,
        "smoke_selection": {
            **smoke_summary,
            "algorithm": (
                "mandatory_refinement_then_structural_midpoint_quantiles_v2"
            ),
            "status": "frozen_not_executed" if status == "pass" else "stop",
            "smoke_plan_sha256": (
                sha256_file(paths["smoke"])
                if status == "pass" else None
            ),
            "score_blind": True,
        },
        "outputs": {
            name: file_metadata(paths[name], paths["root"])
            for name in ("rows0", "rows1", "smoke0", "smoke1")
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
            "raw_text_emitted_to_public_artifacts": False,
            "sealed_inputs_parsed": True,
            "smoke_executed": False,
        },
        "decision": {
            "p0": (
                "pending_independent_audit"
                if status == "pass" else "stop"
            ),
            "smoke": "not_authorized",
            "smoke_started": False,
            "full_detector_replay_started": False,
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(readiness)
    atomic_write_json(paths["readiness"], readiness)
    paths["last_failure"].unlink(missing_ok=True)
    print(json.dumps(readiness, sort_keys=True))
    if status != "pass":
        raise RuntimeError("WP3 P0 finalize stopped")
    return readiness


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
