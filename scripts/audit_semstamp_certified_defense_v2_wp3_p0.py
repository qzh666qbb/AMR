"""Independently audit the model-free v2 WP3 P0 evidence."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import audit_semstamp_certified_defense_v2_wp1 as wp1_audit
import run_certified_resegmentation_wp2_v17_b0 as v17


STAGE = "semstamp_certified_defense_v2_wp3_detector_health_p0"
AUDIT_STAGE = f"{STAGE}_independent_audit"
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
LEGACY_FAILURE_COUNT = 42
EXPECTED_ZERO_GAP_COUNT = 59
SMOKE_PER_COHORT = 32
SMOKE_ROWS = 96
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}
FORBIDDEN_MODULE_ROOTS = {
    "sentence_transformers",
    "torch",
    "transformers",
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


def jsonl_bytes(rows):
    return b"".join(canonical_bytes(row) for row in rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


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


def atomic_write_json(path, value):
    atomic_write_bytes(path, canonical_bytes(value))


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw text key at {location}: {overlap}")
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def assert_model_stack_absent():
    loaded = set(sys.modules)
    overlap = {
        module
        for module in loaded
        if module.split(".", 1)[0] in FORBIDDEN_MODULE_ROOTS
    }
    if overlap:
        raise ValueError(f"forbidden model modules imported: {sorted(overlap)}")
    if (
        "run_certified_resegmentation_wp2_detector_equivalence"
        in loaded
    ):
        raise ValueError("forbidden detector module imported")


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


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def paths_for(root, output_dir, output):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output = Path(output).resolve()
    if (
        not output_dir.is_relative_to(root)
        or not output.is_relative_to(output_dir)
    ):
        raise ValueError("output paths must remain inside the experiment")
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
        "audit": output,
        "decision": output_dir / "P0_GO_NO_GO.md",
        "final": output_dir / "FINAL_RESULT.json",
        "runner": root / "scripts/run_semstamp_certified_defense_v2_wp3_p0.py",
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "test": root / "scripts/test_semstamp_certified_defense_v2_wp3_p0.py",
        "auditor": Path(__file__).resolve(),
        "independent_basis": (
            root / "scripts/audit_semstamp_certified_defense_v2_wp1.py"
        ),
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


def validate_contract_and_inputs(paths):
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
        "replay_count": 2,
        "legacy_failure_count": LEGACY_FAILURE_COUNT,
        "expected_zero_gap_count": EXPECTED_ZERO_GAP_COUNT,
        "smoke_per_cohort": SMOKE_PER_COHORT,
        "smoke_rows": SMOKE_ROWS,
        "native_token_limit": 512,
        "nltk_version": "3.8.1",
        "punkt_payload_sha256": PUNKT_SHA256,
        "semstamp_revision": SEMSTAMP_REVISION,
        "semstamp_data_revision": SEMSTAMP_DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "relation_sha256": RELATION_SHA256,
        "smoke_execution_authorized": False,
    }
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("contract constants mismatch")
    artifact_fields = {
        "protocol_sha256": paths["protocol"],
        "runner_sha256": paths["runner"],
        "renderer_sha256": paths["renderer"],
        "test_sha256": paths["test"],
        "auditor_sha256": paths["auditor"],
        "independent_basis_sha256": paths["independent_basis"],
        "input_manifest_sha256": paths["manifest"],
    }
    if any(
        contract.get(field) != sha256_file(path)
        for field, path in artifact_fields.items()
    ):
        raise ValueError("contract artifact binding mismatch")

    manifest = parse_manifest(paths["manifest"])
    if set(manifest) != set(contract.get("inputs", {})):
        raise ValueError("manifest/contract input set mismatch")
    for relative, expected_sha in manifest.items():
        relative_path = Path(relative)
        path = (paths["root"] / relative_path).resolve()
        metadata = contract["inputs"][relative]
        if (
            relative_path.is_absolute()
            or not path.is_relative_to(paths["root"])
            or not path.is_file()
            or sha256_file(path) != expected_sha
            or path.stat().st_size != metadata.get("bytes")
            or metadata.get("sha256") != expected_sha
        ):
            raise ValueError(f"input drift: {relative}")
        if (
            metadata.get("mode") is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}"
            != metadata["mode"]
        ):
            raise ValueError(f"input mode drift: {relative}")
    if git_head(paths["repo"]) != SEMSTAMP_REVISION:
        raise ValueError("SemStamp revision drift")
    if git_head(paths["data_repo"]) != SEMSTAMP_DATA_REVISION:
        raise ValueError("semstamp-data revision drift")
    return contract, manifest


def validate_parent_terminals(paths):
    v17_readiness = read_json(paths["v17_dir"] / "P0_READINESS.json")
    v1_readiness = read_json(paths["v1_wp3_dir"] / "P0_READINESS.json")
    v1_audit = read_json(paths["v1_wp3_dir"] / "AUDIT.json")
    wp1_result = read_json(paths["v2_wp1_dir"] / "RESULT.json")
    wp1_result_audit = read_json(paths["v2_wp1_dir"] / "AUDIT.json")
    wp2_result = read_json(paths["v2_wp2_dir"] / "RESULT.json")
    wp2_result_audit = read_json(paths["v2_wp2_dir"] / "AUDIT.json")
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
        or wp1_result_audit.get("status") != "go"
        or wp2_result.get("terminal")
        != "WP2_ALGORITHM_GO_FOR_V2_WP3_DESIGN"
        or wp2_result.get("wp3_started") is not False
        or wp2_result_audit.get("status") != "go"
    ):
        raise ValueError("parent evidence is not at the frozen terminal")
    return legacy_indices


def independently_select_smoke(rows):
    selected = []
    for cohort in COHORTS:
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        if len(cohort_rows) != COHORT_SIZE:
            raise ValueError(f"independent cohort mismatch: {cohort}")
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
            raise ValueError(f"independent smoke budget exceeded: {cohort}")
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
    if (
        len(selected) != SMOKE_ROWS
        or len({row["plan_index"] for row in selected}) != SMOKE_ROWS
    ):
        raise ValueError("independent smoke selection is invalid")
    return selected


def load_nltk(nltk_data, contract):
    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != contract["nltk_version"]:
        raise ValueError("auditor NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    resource = nltk.data.find("tokenizers/punkt/english.pickle")
    with resource.open() as handle:
        if sha256_bytes(handle.read()) != contract["punkt_payload_sha256"]:
            raise ValueError("auditor Punkt payload mismatch")
    return sent_tokenize


def execution_attestation():
    return {
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
    }


def decision_markdown(audit_sha256, readiness_sha256, counts):
    return (
        "# SemStamp Certified Defense v2 - WP3 P0 Go/No-Go\n\n"
        "Date: 2026-10-08  \n"
        "Decision: **WP3_P0_GO_FOR_SMOKE_DESIGN**\n\n"
        "## Evidence\n\n"
        f"- Independent rows recomputed: `{counts['rows']}/3000`\n"
        f"- Legacy renderer failures repaired: "
        f"`{counts['legacy_v1_failures_repaired']}/42`\n"
        f"- Authorized zero-byte gaps: "
        f"`{counts['zero_byte_gap_total']}/59`\n"
        f"- Frozen smoke rows: `{counts['smoke_rows']}/96`\n"
        f"- Independent audit SHA256: `{audit_sha256}`\n"
        f"- Final readiness SHA256: `{readiness_sha256}`\n\n"
        "## Scope\n\n"
        "This decision authorizes only a separate 96-row detector-smoke "
        "protocol design. It does not authorize model loading, GPU use, "
        "smoke execution, a 3,000-row detector replay, WP4, or certificate "
        "execution.\n"
    ).encode("ascii")


def audit(root, output_dir, output, nltk_data):
    assert_model_stack_absent()
    paths = paths_for(root, output_dir, output)
    contract, _manifest = validate_contract_and_inputs(paths)
    legacy_indices = validate_parent_terminals(paths)

    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    smoke0 = read_jsonl(paths["smoke0"])
    smoke1 = read_jsonl(paths["smoke1"])
    smoke = read_jsonl(paths["smoke"])
    replay0 = read_json(paths["replay0"])
    replay1 = read_json(paths["replay1"])
    pending_readiness = read_json(paths["readiness"])
    parent_rows = read_jsonl(
        paths["v2_wp1_dir"] / "P0_RENDERER_ROWS_REPLAY0.jsonl"
    )
    reject_raw_public_keys([
        rows0,
        rows1,
        smoke0,
        smoke1,
        smoke,
        replay0,
        replay1,
        pending_readiness,
        parent_rows,
    ])
    if (
        len(rows0) != TOTAL_ROWS
        or len(parent_rows) != TOTAL_ROWS
        or paths["rows0"].read_bytes() != paths["rows1"].read_bytes()
        or paths["smoke0"].read_bytes() != paths["smoke1"].read_bytes()
        or paths["smoke0"].read_bytes() != paths["smoke"].read_bytes()
        or any(replay.get("status") != "pass" for replay in (replay0, replay1))
        or pending_readiness.get("status") != "pass"
        or pending_readiness.get("terminal_reason")
        != "p0_pass_pending_independent_audit"
        or not all(pending_readiness.get("gates", {}).values())
    ):
        raise ValueError("runner evidence is not a complete pending pass")
    for artifact in (
        paths["rows0"],
        paths["rows1"],
        paths["smoke0"],
        paths["smoke1"],
        paths["smoke"],
        paths["replay0"],
        paths["replay1"],
        paths["readiness"],
    ):
        if stat.S_IMODE(artifact.stat().st_mode) != 0o600:
            raise ValueError(f"public artifact mode drift: {artifact.name}")

    b0_paths = v17.paths_for(paths["root"], paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    if (
        len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
        or v17.jsonl_bytes(plan)
        != (paths["v17_dir"] / "B0_PLAN.jsonl").read_bytes()
    ):
        raise ValueError("independent input materialization mismatch")
    sent_tokenize = load_nltk(nltk_data, contract)

    expected_rows = []
    observed_legacy = []
    for index, (planned_value, text, observed, parent) in enumerate(
        zip(plan, texts, rows0, parent_rows)
    ):
        planned = {
            **planned_value,
            "_protocol_sha256": WP1_PROTOCOL_SHA256,
        }
        independent_parent = wp1_audit.independently_build_row(
            index,
            planned,
            text,
            sent_tokenize(text),
            legacy_indices,
        )
        legacy_observed = independent_parent.pop(
            "_legacy_failure_independently_observed"
        )
        if legacy_observed:
            observed_legacy.append(index)
        if canonical_bytes(independent_parent) != canonical_bytes(parent):
            raise ValueError(f"independent WP1 parent mismatch at {index}")
        structural = {
            key: value
            for key, value in independent_parent.items()
            if key not in {"schema_version", "protocol_sha256"}
        }
        expected = {
            "schema_version": 1,
            "stage": STAGE,
            "protocol_sha256": PROTOCOL_SHA256,
            **structural,
            "wp1_reference_exact": True,
            "wp1_reference_row_sha256": sha256_bytes(
                canonical_bytes(independent_parent)
            ),
        }
        if canonical_bytes(expected) != canonical_bytes(observed):
            raise ValueError(f"independent P0 row mismatch at {index}")
        expected_rows.append(expected)
    if tuple(observed_legacy) != legacy_indices:
        raise ValueError("legacy failures were not independently reproduced")
    if jsonl_bytes(expected_rows) != paths["rows0"].read_bytes():
        raise ValueError("independent renderer row bytes mismatch")

    expected_smoke = independently_select_smoke(expected_rows)
    if jsonl_bytes(expected_smoke) != paths["smoke"].read_bytes():
        raise ValueError("independent smoke plan bytes mismatch")

    risk_indices = {
        row["plan_index"]
        for row in expected_rows
        if row["refinement_count"] or row["zero_byte_gap_count"]
    }
    smoke_indices = {row["plan_index"] for row in expected_smoke}
    zero_gap_total = sum(
        row["zero_byte_gap_count"] for row in expected_rows
    )
    counts = {
        "rows": len(expected_rows),
        "cohorts": dict(Counter(
            row["cohort"] for row in expected_rows
        )),
        "raw_identities": len({
            row["raw_text_sha256"] for row in expected_rows
        }),
        "legacy_v1_failures": len(legacy_indices),
        "legacy_v1_failures_repaired": sum(
            row["legacy_v1_alignment_failure"]
            for row in expected_rows
        ),
        "rows_with_refinement": sum(
            row["refinement_count"] > 0 for row in expected_rows
        ),
        "refinement_total": sum(
            row["refinement_count"] for row in expected_rows
        ),
        "zero_byte_gap_total": zero_gap_total,
        "smoke_rows": len(expected_smoke),
        "smoke_cohorts": dict(Counter(
            row["cohort"] for row in expected_smoke
        )),
        "smoke_mandatory_rows": sum(
            row["selection_class"] == "mandatory_refinement"
            for row in expected_smoke
        ),
    }
    all_attestations = (
        replay0["execution_attestation"],
        replay1["execution_attestation"],
        pending_readiness["execution_attestation"],
    )
    zero_fields = (
        "certificate_calls",
        "detector_imports",
        "detector_model_loads",
        "detector_score_reads",
        "detector_scores_computed",
        "generation_calls",
        "gpu_calls",
    )
    checks = {
        "contract_and_inputs_exact": True,
        "parent_terminals_exact": True,
        "fixed_3000_rows_accounted": len(expected_rows) == TOTAL_ROWS,
        "all_raw_identities_unique": counts["raw_identities"] == TOTAL_ROWS,
        "two_renderer_replays_byte_exact": (
            paths["rows0"].read_bytes() == paths["rows1"].read_bytes()
        ),
        "all_rows_independently_recomputed": True,
        "all_wp1_parent_rows_independently_recomputed": True,
        "all_original_units_exact": True,
        "all_raw_bytes_reconstructed": all(
            row["byte_reconstruction_sha256"]
            == row["raw_text_sha256"]
            for row in expected_rows
        ),
        "all_zero_byte_gaps_are_b0": all(
            row["zero_byte_gaps_all_original"] is True
            for row in expected_rows
        ),
        "legacy_42_failures_reproduced": (
            len(observed_legacy) == LEGACY_FAILURE_COUNT
        ),
        "legacy_42_rows_repaired": (
            counts["legacy_v1_failures_repaired"]
            == LEGACY_FAILURE_COUNT
        ),
        "expected_59_zero_gaps": (
            zero_gap_total == EXPECTED_ZERO_GAP_COUNT
        ),
        "two_smoke_plans_byte_exact": (
            paths["smoke0"].read_bytes() == paths["smoke1"].read_bytes()
        ),
        "smoke_plan_independently_recomputed": True,
        "smoke_96_and_balanced": (
            len(expected_smoke) == SMOKE_ROWS
            and counts["smoke_cohorts"]
            == {cohort: SMOKE_PER_COHORT for cohort in COHORTS}
        ),
        "smoke_covers_all_risk_rows": (
            len(risk_indices) == LEGACY_FAILURE_COUNT
            and risk_indices.issubset(smoke_indices)
            and counts["smoke_mandatory_rows"] == LEGACY_FAILURE_COUNT
        ),
        "model_gpu_score_certificate_calls_zero": all(
            attestation[field] == 0
            for attestation in all_attestations
            for field in zero_fields
        ),
        "smoke_not_executed": all(
            attestation["smoke_executed"] is False
            for attestation in all_attestations
        ),
        "raw_text_absent_from_public_artifacts": True,
        "canonical_private_outputs": True,
        "forbidden_model_imports_absent": True,
    }
    if not all(checks.values()):
        raise ValueError("independent P0 checks did not all pass")
    assert_model_stack_absent()

    pending_readiness_sha256 = sha256_file(paths["readiness"])
    audit_result = {
        "schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "go",
        "terminal_reason": "WP3_P0_GO_FOR_SMOKE_DESIGN",
        "scope": "independent_model_free_raw_to_row_and_smoke_recalculation",
        "bindings": {
            "protocol": sha256_file(paths["protocol"]),
            "contract": sha256_file(paths["contract"]),
            "manifest": sha256_file(paths["manifest"]),
            "runner": sha256_file(paths["runner"]),
            "renderer": sha256_file(paths["renderer"]),
            "test": sha256_file(paths["test"]),
            "auditor": sha256_file(paths["auditor"]),
            "independent_basis": sha256_file(paths["independent_basis"]),
            "rows0": sha256_file(paths["rows0"]),
            "rows1": sha256_file(paths["rows1"]),
            "smoke0": sha256_file(paths["smoke0"]),
            "smoke1": sha256_file(paths["smoke1"]),
            "smoke": sha256_file(paths["smoke"]),
            "runner_readiness": pending_readiness_sha256,
        },
        "checks": checks,
        "counts": counts,
        "execution_attestation": execution_attestation(),
        "decision": {
            "p0": "go",
            "smoke": "authorized_to_freeze_design_only",
            "smoke_execution_authorized": False,
            "smoke_started": False,
            "full_detector_replay_started": False,
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(audit_result)
    atomic_write_json(paths["audit"], audit_result)
    audit_sha256 = sha256_file(paths["audit"])

    final_readiness = {
        **pending_readiness,
        "status": "go",
        "terminal_reason": "WP3_P0_GO_FOR_SMOKE_DESIGN",
        "independent_audit": {
            "status": "go",
            "sha256": audit_sha256,
            "runner_readiness_sha256": pending_readiness_sha256,
            "all_checks_pass": True,
        },
        "decision": {
            "p0": "go",
            "smoke": "authorized_to_freeze_design_only",
            "smoke_execution_authorized": False,
            "smoke_started": False,
            "full_detector_replay_started": False,
            "wp4_started": False,
        },
    }
    reject_raw_public_keys(final_readiness)
    atomic_write_json(paths["readiness"], final_readiness)
    readiness_sha256 = sha256_file(paths["readiness"])

    atomic_write_bytes(
        paths["decision"],
        decision_markdown(audit_sha256, readiness_sha256, counts),
    )
    final_result = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "go",
        "terminal": "WP3_P0_GO_FOR_SMOKE_DESIGN",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "input_manifest_sha256": sha256_file(paths["manifest"]),
        "audit_sha256": audit_sha256,
        "readiness_sha256": readiness_sha256,
        "decision_sha256": sha256_file(paths["decision"]),
        "counts": counts,
        "checks": checks,
        "execution_attestation": execution_attestation(),
        "decision": {
            "p0": "go",
            "next_action": "freeze_separate_96_row_detector_smoke_protocol",
            "smoke_execution_authorized": False,
            "smoke_started": False,
            "full_detector_replay_started": False,
            "wp4_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(final_result)
    atomic_write_json(paths["final"], final_result)
    print(json.dumps(final_result, sort_keys=True))
    return final_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.output, args.nltk_data)


if __name__ == "__main__":
    main()
