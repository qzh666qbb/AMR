"""Independently audit SemStamp certified-defense WP3 terminal evidence."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

import certified_resegmentation_wp2_v13_b0_support as support
import run_certified_resegmentation_wp2_v17_b0 as v17


STAGE = "semstamp_certified_defense_v1_wp3_detector_health"
COHORTS = ("calibration_null", "independent_test_null", "positive")
TOTAL_ROWS = 3000
COHORT_SIZE = 1000
SMOKE_ROWS = 96
SMOKE_PER_COHORT = 32
V17_THRESHOLD = 1.6666666666666667
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def parse_manifest(path):
    entries = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or relative in entries:
            raise ValueError("invalid or duplicate manifest entry")
        entries[relative] = digest
    return entries


def select_smoke(renderer_rows, protocol_sha256):
    selected = []
    for cohort in COHORTS:
        ordered = sorted(
            (row for row in renderer_rows if row["cohort"] == cohort),
            key=lambda row: (
                row["atom_count"],
                row["unit_count"],
                row["raw_text_sha256"],
                row["plan_index"],
            ),
        )
        if len(ordered) != COHORT_SIZE:
            raise ValueError("renderer cohort count mismatch")
        for selection_index in range(SMOKE_PER_COHORT):
            rank = (
                (2 * selection_index + 1) * COHORT_SIZE
            ) // (2 * SMOKE_PER_COHORT)
            row = ordered[rank]
            selected.append({
                "schema_version": 1,
                "protocol_sha256": protocol_sha256,
                "smoke_index": len(selected),
                "cohort": cohort,
                "cohort_selection_index": selection_index,
                "structural_rank": rank,
                "plan_index": row["plan_index"],
                "cohort_index": row["cohort_index"],
                "raw_text_sha256": row["raw_text_sha256"],
                "atom_count": row["atom_count"],
                "unit_count": row["unit_count"],
                "topology_sha256": row["topology_sha256"],
                "legality_sha256": row["legality_sha256"],
                "score_access_count_during_construction": 0,
                "contains_raw_text": False,
            })
    return selected


def base_score_row(row):
    return {
        key: row[key]
        for key in support.PLAN_FIELDS | support.SCORE_FIELDS
    }


def alignment_failure(text, units):
    atoms = tuple(re.finditer(r"\S+", text))
    if len(atoms) < 2:
        return "INSUFFICIENT_ATOMS", "atom_count_lt_2"
    starts = {match.start() for match in atoms}
    ends = {match.end() for match in atoms}
    cursor = 0
    for unit_index, unit in enumerate(units):
        if not unit or unit != unit.strip() or re.search(r"\S+", unit) is None:
            return "P0_ALIGNMENT_FAILED", f"invalid_unit_{unit_index}"
        match_start = text.find(unit, cursor)
        if match_start < 0:
            return "P0_ALIGNMENT_FAILED", f"unit_not_found_{unit_index}"
        skipped = text[cursor:match_start]
        if skipped and not skipped.isspace():
            return (
                "P0_ALIGNMENT_FAILED",
                f"non_whitespace_before_unit_{unit_index}",
            )
        match_end = match_start + len(unit)
        start_aligned = match_start in starts
        end_aligned = match_end in ends
        if not start_aligned and not end_aligned:
            return (
                "P0_ALIGNMENT_FAILED",
                f"both_endpoints_inside_atoms_at_unit_{unit_index}",
            )
        if not start_aligned:
            return (
                "P0_ALIGNMENT_FAILED",
                f"start_inside_atom_at_unit_{unit_index}",
            )
        if not end_aligned:
            return (
                "P0_ALIGNMENT_FAILED",
                f"end_inside_atom_at_unit_{unit_index}",
            )
        cursor = match_end
    if text[cursor:] and not text[cursor:].isspace():
        return "P0_ALIGNMENT_FAILED", "non_whitespace_trailing_text"
    return None


def validate_scored_rows(rows, expected_indices, v17_rows, renderer_rows):
    if (
        len(rows) != len(expected_indices)
        or [row.get("plan_index") for row in rows] != expected_indices
    ):
        raise ValueError("scored row order/count mismatch")
    for offset, row in enumerate(rows):
        plan_index = expected_indices[offset]
        reference = v17_rows[plan_index]
        renderer = renderer_rows[plan_index]
        base = base_score_row(row)
        support.validate_public_score(
            base,
            {key: reference[key] for key in support.PLAN_FIELDS},
        )
        if canonical_bytes(base) != canonical_bytes(reference):
            raise ValueError(f"v17 score mismatch at row {plan_index}")
        expected_renderer = {
            "renderer_contract": "fixed_raw_utf8_bytes_v1",
            "atom_count": renderer["atom_count"],
            "original_boundaries": renderer["original_boundaries"],
            "original_units_sha256": renderer["original_units_sha256"],
            "topology_sha256": renderer["topology_sha256"],
            "legality_sha256": renderer["legality_sha256"],
            "byte_reconstruction_sha256": (
                renderer["byte_reconstruction_sha256"]
            ),
            "v17_reference_exact": True,
        }
        if any(row.get(key) != value for key, value in expected_renderer.items()):
            raise ValueError(f"renderer score binding mismatch at {plan_index}")


def audit(root, output_dir, output, nltk_data):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = {
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "readiness": output_dir / "P0_READINESS.json",
        "renderer_rows": output_dir / "P0_RENDERER_ROWS.jsonl",
        "smoke_plan": output_dir / "SMOKE_PLAN.jsonl",
        "smoke_result": output_dir / "SMOKE_RESULT.json",
        "smoke_rows0": output_dir / "SMOKE_ROWS_REPLAY0.jsonl",
        "smoke_rows1": output_dir / "SMOKE_ROWS_REPLAY1.jsonl",
        "smoke_checkpoint": output_dir / "sealed/smoke_checkpoint.json",
        "result": output_dir / "RESULT.json",
        "rows0": output_dir / "ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "ROWS_REPLAY1.jsonl",
        "checkpoint": output_dir / "sealed/full_checkpoint.json",
        "runner": root / "scripts/run_semstamp_certified_defense_v1_wp3.py",
        "renderer": root / "scripts/semstamp_fixed_byte_renderer.py",
        "test": root / "scripts/test_semstamp_certified_defense_v1_wp3.py",
        "auditor": Path(__file__).resolve(),
        "v17_dir": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
        ),
        "v17_rows": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
            / "B0_ROWS_REPLAY0.jsonl"
        ),
        "v17_result": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
            / "B0_RESULT.json"
        ),
    }
    contract = read_json(paths["contract"])
    protocol_sha256 = contract["protocol_sha256"]
    bindings = {
        label: sha256_file(path)
        for label, path in paths.items()
        if (
            label not in {"v17_dir", "v17_rows", "v17_result"}
            and path.is_file()
        )
    }
    expected_artifacts = {
        "protocol": protocol_sha256,
        "manifest": contract["input_manifest_sha256"],
        "runner": contract["runner_sha256"],
        "renderer": contract["renderer_sha256"],
        "test": contract["test_sha256"],
        "auditor": contract["auditor_sha256"],
    }
    if any(bindings[label] != digest for label, digest in expected_artifacts.items()):
        raise ValueError("contract artifact binding mismatch")
    manifest = parse_manifest(paths["manifest"])
    if set(manifest) != set(contract["inputs"]):
        raise ValueError("manifest/contract input set mismatch")
    for relative, digest in manifest.items():
        path = root / relative
        meta = contract["inputs"][relative]
        if (
            sha256_file(path) != digest
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"input drift: {relative}")

    readiness = read_json(paths["readiness"])
    renderer_rows = read_jsonl(paths["renderer_rows"])
    smoke_plan = (
        read_jsonl(paths["smoke_plan"])
        if paths["smoke_plan"].is_file()
        else []
    )
    reject_raw_public_keys([readiness, renderer_rows, smoke_plan])
    if (
        readiness.get("status") not in {"pass", "stop"}
        or readiness.get("scope")
        != "fixed_byte_renderer_all_3000_model_free"
        or len(renderer_rows) != TOTAL_ROWS
        or [row.get("plan_index") for row in renderer_rows]
        != list(range(TOTAL_ROWS))
        or Counter(row.get("cohort") for row in renderer_rows)
        != {cohort: COHORT_SIZE for cohort in COHORTS}
        or any(
            row.get("score_access_count_during_construction") != 0
            for row in renderer_rows
        )
        or any(
            readiness["execution_attestation"].get(field) != 0
            for field in (
                "certificate_calls",
                "detector_imports",
                "detector_model_loads",
                "detector_score_reads",
                "detector_scores_computed",
                "generation_calls",
                "score_access_count_during_construction",
            )
        )
    ):
        raise ValueError("P0 readiness contract failed")
    valid_rows = [row for row in renderer_rows if row.get("status") == "pass"]
    failed_rows = [
        row for row in renderer_rows
        if row.get("status") == "invalid_or_incomplete"
    ]
    if readiness["renderer"]["renderer_rows_sha256"] != sha256_file(
        paths["renderer_rows"]
    ):
        raise ValueError("P0 aggregate digest mismatch")

    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != "3.8.1":
        raise ValueError("auditor NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    nltk.data.find("tokenizers/punkt/english.pickle")
    b0_paths = v17.paths_for(root, paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    independent_failures = []
    for index, (planned, text, row) in enumerate(
        zip(plan, texts, renderer_rows)
    ):
        units = sent_tokenize(text)
        failure = alignment_failure(text, units)
        if (
            row["plan_index"] != index
            or row["raw_text_sha256"]
            != hashlib.sha256(text.encode("utf-8")).hexdigest()
            or row["raw_text_sha256"] != planned["raw_text_sha256"]
            or row["unit_count"] != planned["unit_count"]
        ):
            raise ValueError(f"P0 identity drift at row {index}")
        if failure is None:
            if (
                row.get("status") != "pass"
                or row.get("byte_reconstruction_sha256")
                != row.get("raw_text_sha256")
            ):
                raise ValueError(f"unexpected renderer failure at row {index}")
        else:
            code, detail = failure
            independent_failures.append((index, code, detail))
            if (
                row.get("status") != "invalid_or_incomplete"
                or row.get("failure_code") != code
                or row.get("failure_detail") != detail
                or row.get("observed_nltk_unit_count") != len(units)
            ):
                raise ValueError(f"failure evidence drift at row {index}")
    if (
        len(valid_rows) + len(failed_rows) != TOTAL_ROWS
        or len(independent_failures) != len(failed_rows)
        or readiness["renderer"]["valid_rows"] != len(valid_rows)
        or readiness["renderer"]["invalid_rows"] != len(failed_rows)
        or readiness["renderer"]["failed_plan_indices"]
        != [index for index, _code, _detail in independent_failures]
    ):
        raise ValueError("P0 failure counts are not reproducible")

    if readiness["status"] == "stop":
        forbidden_downstream = (
            "smoke_plan",
            "smoke_result",
            "smoke_rows0",
            "smoke_rows1",
            "smoke_checkpoint",
            "result",
            "rows0",
            "rows1",
            "checkpoint",
        )
        if (
            not failed_rows
            or readiness.get("terminal_reason") != "renderer_contract_stop"
            or readiness["smoke_selection"].get("status") != "not_started"
            or any(paths[label].exists() for label in forbidden_downstream)
        ):
            raise ValueError("P0 Stop did not prevent downstream execution")
        audit_result = {
            "schema_version": 1,
            "stage": f"{STAGE}_p0_stop_audit",
            "status": "stop",
            "terminal_reason": "wp3_renderer_contract_stop",
            "scope": "model_free_independent_p0_recalculation",
            "bindings": bindings,
            "checks": {
                "contract_and_inputs_exact": True,
                "all_3000_rows_accounted": True,
                "alignment_failures_independently_reproduced": True,
                "p0_score_access_zero": True,
                "downstream_model_execution_absent": True,
                "raw_text_absent_from_public_artifacts": True,
            },
            "counts": {
                "rows": TOTAL_ROWS,
                "valid_rows": len(valid_rows),
                "invalid_rows": len(failed_rows),
                "failure_codes": dict(Counter(
                    row["failure_code"] for row in failed_rows
                )),
                "failure_cohorts": dict(Counter(
                    row["cohort"] for row in failed_rows
                )),
            },
            "decision": {
                "wp3_detector_health": "stop_before_model_load",
                "wp4": "not_authorized",
                "wp4_started": False,
            },
            "execution_attestation": {
                "certificate_calls": 0,
                "detector_imports": 0,
                "detector_model_loads": 0,
                "detector_scores_computed": 0,
            },
            "contains_raw_text": False,
        }
        atomic_write_json(output, audit_result)
        print(json.dumps(audit_result, sort_keys=True))
        return audit_result

    if (
        readiness["renderer"]["topology_sha256"]
        != canonical_digest([row["topology_sha256"] for row in valid_rows])
        or readiness["renderer"]["legality_sha256"]
        != canonical_digest([row["legality_sha256"] for row in valid_rows])
    ):
        raise ValueError("P0 pass aggregate digest mismatch")
    rebuilt_smoke = select_smoke(renderer_rows, protocol_sha256)
    if canonical_bytes(rebuilt_smoke) != canonical_bytes(smoke_plan):
        raise ValueError("smoke selection is not independently reproducible")

    v17_rows = read_jsonl(paths["v17_rows"])
    v17_result = read_json(paths["v17_result"])
    smoke_result = read_json(paths["smoke_result"])
    smoke_rows0 = read_jsonl(paths["smoke_rows0"])
    smoke_rows1 = read_jsonl(paths["smoke_rows1"])
    reject_raw_public_keys([smoke_result, smoke_rows0, smoke_rows1])
    smoke_indices = [row["plan_index"] for row in smoke_plan]
    if (
        smoke_result.get("status") != "pass"
        or not all(smoke_result.get("gates", {}).values())
        or len(smoke_rows0) != SMOKE_ROWS
        or paths["smoke_rows0"].read_bytes()
        != paths["smoke_rows1"].read_bytes()
    ):
        raise ValueError("smoke terminal evidence failed")
    validate_scored_rows(
        smoke_rows0,
        smoke_indices,
        v17_rows,
        renderer_rows,
    )

    result = read_json(paths["result"])
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    reject_raw_public_keys([result, rows0, rows1])
    if (
        result.get("stage") != STAGE
        or result.get("status") != "go"
        or result.get("terminal_reason")
        != "wp3_detector_health_go_for_wp4_design"
        or not all(result.get("gates", {}).values())
        or len(rows0) != TOTAL_ROWS
        or paths["rows0"].read_bytes() != paths["rows1"].read_bytes()
        or result.get("decision", {}).get("wp4")
        != "authorized_to_freeze_design"
        or result.get("decision", {}).get("wp4_started") is not False
    ):
        raise ValueError("full terminal evidence failed")
    validate_scored_rows(
        rows0,
        list(range(TOTAL_ROWS)),
        v17_rows,
        renderer_rows,
    )
    score_groups = {
        cohort: [
            row["direct_score"] for row in rows0 if row["cohort"] == cohort
        ]
        for cohort in COHORTS
    }
    statistics = support.summarize_scores(
        score_groups["calibration_null"],
        score_groups["independent_test_null"],
        score_groups["positive"],
    )
    if (
        canonical_bytes(statistics) != canonical_bytes(result["statistics"])
        or canonical_bytes(statistics)
        != canonical_bytes(v17_result["statistics"])
        or statistics["calibration"]["threshold"] != V17_THRESHOLD
    ):
        raise ValueError("statistics are not independently reproducible")
    for checkpoint in (paths["smoke_checkpoint"], paths["checkpoint"]):
        if stat.S_IMODE(checkpoint.stat().st_mode) != 0o600:
            raise ValueError("checkpoint mode is not 0600")

    audit_result = {
        "schema_version": 1,
        "stage": f"{STAGE}_terminal_audit",
        "status": "go",
        "scope": "model_free_independent_terminal_recalculation",
        "bindings": bindings,
        "checks": {
            "contract_and_inputs_exact": True,
            "p0_all_3000_fixed_byte_rows_exact": True,
            "p0_score_access_zero": True,
            "smoke_selection_independently_rebuilt": True,
            "smoke_96_two_replays_exact": True,
            "full_3000_two_replays_exact": True,
            "official_direct_rows_valid": True,
            "v17_reference_exact": True,
            "statistics_recomputed": True,
            "all_health_and_resource_gates_pass": True,
            "sealed_checkpoints_private": True,
            "raw_text_absent_from_public_artifacts": True,
        },
        "statistics": statistics,
        "decision": {
            "wp3_detector_health": "confirmed_go",
            "wp4": "not_started",
            "next_action": "freeze a separate WP4 protocol before execution",
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
        },
        "contains_raw_text": False,
    }
    atomic_write_json(output, audit_result)
    print(json.dumps(audit_result, sort_keys=True))
    return audit_result


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
