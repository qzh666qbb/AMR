"""Independently audit WP2-v18 B1 readiness or terminal artifacts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
)


STAGE = "WP2-v18-B1-SemStamp-exact-certificate"
PROTOCOL_SHA256 = (
    "0da27bc53e06e76ba5f1d25aefe3bf787893c0fa4421113dddb1a995892ab82d"
)
B0_RESULT_SHA256 = (
    "c17a382d6946fe93e09df34f420b2ff07a9ccc9daad184b55c71d23118cf9e9e"
)
B0_AUDIT_SHA256 = (
    "bb152276a01c1b7a78671061a30ac6219a3905fcafebb454ba2a270782a8f342"
)
B0_PLAN_SHA256 = (
    "a1427a9978751c756ad5072aeb45bb6cb393713d8339d62ff6ef4397591fc4e2"
)
B0_ROWS_SHA256 = (
    "75d0fe012215958727da715a0e916240837dd2ef0daac6b02c78c6500410824f"
)
B0_CHECKPOINT_SHA256 = (
    "b4fbd2d9d610fe170f1a4700ac35c2b1c405b1a145e8bc1d32cd7d7ec0f8d909"
)
PROFILE_SHA256 = (
    "4701a4b6d77f2d0c20e189853ec896d70ac205df05659ee38e576b5010119f6d"
)
CORE_SOURCE_SHA256 = (
    "d77d23c0e44a343c154e93d9897ee3e40c1a05194d9200108ff7e619f52e1360"
)
B0_THRESHOLD = 1.6666666666666667
DEPENDENCY_SHA256 = {
    "scripts/audit_certified_resegmentation_wp2_v4_correctness.py": (
        "e6e3cc41c4c10844f16932652223674f3999739356e37c8c0511d2891eb703d2"
    ),
    "scripts/certified_resegmentation_wp2_v3.py": (
        "31e89a3be64780ae3d5537fc93ac4715b2ed3a88427fac20b0fe2007831a39d3"
    ),
    "scripts/certified_resegmentation_wp2_v4.py": (
        "fb6ee4aa1ce944ec9c6a6a96ca103bcdf68ad2d8f4e67444c5337786c83853ec"
    ),
    "scripts/certified_resegmentation_wp2_v6_features.py": (
        "16ba74b614780e7d64f0bc51d14b0fd554b4ed5cc51d4a3af4f54f2cf2484ae3"
    ),
    "scripts/certified_resegmentation_wp2_v13_b0_support.py": (
        "692f26230b7f335f715b23a05d2984d617d064e89db0d7ad0fa23fc4c2025166"
    ),
    "scripts/run_certified_resegmentation_wp2_v10_semstamp.py": (
        "f003aff6aa8245a7fdb619c7a5a7af5544fb1fb51de2df0de2ab6774b6d86583"
    ),
    "scripts/run_certified_resegmentation_wp2_v17_b0.py": (
        "a15c7bfc65b6b7bea62bbeb012484532e389972e7d6d5bc45833e2a98d0f5c72"
    ),
}
TOTAL_ROWS = 3000
COHORT_SIZE = 1000
RADII = tuple(range(4))
SPAN_GATE = 2_000_000
PAIR_GATE = 50_000_000
LEGACY_PAIR_DIAGNOSTIC = 5_000_000
CACHE_GATE_BYTES = 50 * 1024**3
GPU_HOUR_GATE = 72.0
MIN_FREE_GIB = 5.0
REPLAY_COUNT = 2
BATCH_SIZE = 64
TOKENIZER_CHUNK_SIZE = 512
TEXT_CHUNK_SIZE = 4096
FORBIDDEN_PUBLIC_KEYS = {
    "text",
    "raw_text",
    "raw_value",
    "prompt",
    "generated_text",
    "original_units",
    "structural_tape",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
    payload = canonical_bytes(value)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def no_forbidden_keys(value):
    if isinstance(value, dict):
        if FORBIDDEN_PUBLIC_KEYS.intersection(value):
            return False
        return all(no_forbidden_keys(item) for item in value.values())
    if isinstance(value, list):
        return all(no_forbidden_keys(item) for item in value)
    return True


def score_from_hits(hits, edge_count):
    return float(
        (float(hits) - 0.25 * edge_count)
        / math.sqrt(edge_count * 0.25 * 0.75)
    )


def clopper_pearson_lower(successes, trials, confidence=0.95):
    if successes == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(
        trials - successes, trials, confidence
    )


def certificate_digest(value):
    selected = {
        key: value[key]
        for key in (
            "requested_radius",
            "effective_radius",
            "exact_min_hits",
            "exact_score",
            "cheap_min_hits",
            "cheap_score",
            "threat_set_size",
            "reachable_state_count",
            "unique_span_count",
            "unique_pair_count",
            "witness_boundaries",
            "witness_score",
            "feature_cache_sha256",
        )
    }
    return canonical_digest(selected)


def paths_for(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    v17 = (
        root
        / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "protocol": output_dir / "PROTOCOL.md",
        "contract": output_dir / "B1_CONTRACT.json",
        "workload": output_dir / "P0_WORKLOAD_ROWS.jsonl",
        "readiness": output_dir / "P0_READINESS.json",
        "p0_audit": output_dir / "P0_AUDIT.json",
        "rows0": output_dir / "B1_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "B1_ROWS_REPLAY1.jsonl",
        "runtime": output_dir / "B1_RUNTIME.json",
        "result": output_dir / "B1_RESULT.json",
        "audit": output_dir / "B1_AUDIT.json",
        "b0_result": v17 / "B0_RESULT.json",
        "b0_audit": v17 / "B0_AUDIT.json",
        "b0_plan": v17 / "B0_PLAN.jsonl",
        "b0_rows": v17 / "B0_ROWS_REPLAY0.jsonl",
        "b0_checkpoint": v17 / "sealed/b0_checkpoint.json",
        "profile": (
            root
            / "experiments/certified_resegmentation_wp2_v6_feature1"
            / "semstamp_workload.json"
        ),
        "core_source": (
            root / "scripts/certified_resegmentation_wp2_v6_core.cpp"
        ),
        "runner": (
            root / "scripts/run_certified_resegmentation_wp2_v18_b1.py"
        ),
        "auditor": Path(__file__).resolve(),
        "test": (
            root / "scripts/test_certified_resegmentation_wp2_v18_b1.py"
        ),
    }


def validate_contract(paths):
    contract = read_json(paths["contract"])
    required = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "frozen",
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(paths["runner"]),
        "auditor_sha256": sha256_file(paths["auditor"]),
        "test_sha256": sha256_file(paths["test"]),
        "core_source_sha256": CORE_SOURCE_SHA256,
        "b0_result_sha256": B0_RESULT_SHA256,
        "b0_audit_sha256": B0_AUDIT_SHA256,
        "b0_plan_sha256": B0_PLAN_SHA256,
        "b0_rows_sha256": B0_ROWS_SHA256,
        "b0_checkpoint_sha256": B0_CHECKPOINT_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "radii": list(RADII),
        "replay_count": REPLAY_COUNT,
        "span_gate": SPAN_GATE,
        "pair_gate": PAIR_GATE,
        "legacy_pair_diagnostic": LEGACY_PAIR_DIAGNOSTIC,
        "gpu_hour_gate": GPU_HOUR_GATE,
        "minimum_free_gib": MIN_FREE_GIB,
        "embedding_batch_size": BATCH_SIZE,
        "tokenizer_chunk_size": TOKENIZER_CHUNK_SIZE,
        "text_chunk_size": TEXT_CHUNK_SIZE,
        "dependency_sha256": DEPENDENCY_SHA256,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"contract mismatch: {mismatches}")
    for relative_path, expected in DEPENDENCY_SHA256.items():
        if sha256_file(paths["root"] / relative_path) != expected:
            raise ValueError(f"dependency changed: {relative_path}")
    return contract


def validate_parent_files(paths):
    expected = {
        "protocol": PROTOCOL_SHA256,
        "b0_result": B0_RESULT_SHA256,
        "b0_audit": B0_AUDIT_SHA256,
        "b0_plan": B0_PLAN_SHA256,
        "b0_rows": B0_ROWS_SHA256,
        "b0_checkpoint": B0_CHECKPOINT_SHA256,
        "profile": PROFILE_SHA256,
        "core_source": CORE_SOURCE_SHA256,
    }
    for key, digest in expected.items():
        if sha256_file(paths[key]) != digest:
            raise ValueError(f"{key} SHA256 mismatch")
    result = read_json(paths["b0_result"])
    audit = read_json(paths["b0_audit"])
    if (
        result.get("status") != "go"
        or result.get("statistics", {}).get("calibration", {}).get("threshold")
        != B0_THRESHOLD
        or not all(result.get("gates", {}).values())
        or audit.get("status") != "go"
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("parent B0 is not an audited Go")


def validate_workload_row(row, index, plan):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("plan_index") != index
        or row.get("cohort") != plan["cohort"]
        or row.get("cohort_index") != plan["cohort_index"]
        or row.get("raw_text_sha256") != plan["raw_text_sha256"]
        or row.get("unit_count") != plan["unit_count"]
        or row.get("edge_count") != plan["unit_count"] - 1
        or len(row.get("original_boundaries", [])) != row.get("edge_count")
        or row.get("original_boundaries")
        != sorted(set(row.get("original_boundaries", [])))
        or set(row.get("radii", {})) != {"0", "1", "2", "3"}
        or row.get("contains_raw_text") is not False
        or not no_forbidden_keys(row)
    ):
        raise ValueError(f"workload row mismatch at {index}")
    for radius in RADII:
        value = row["radii"][str(radius)]
        if (
            value["requested_radius"] != radius
            or value["effective_radius"]
            != min(radius, row["edge_count"])
            or not 1 <= value["legal_span_count"]
            <= value["unique_span_count"]
            <= value["all_allowed_unique_spans"]
            or not 0 <= value["unique_pair_count"]
            <= value["all_allowed_unique_pairs"]
            or value["threat_set_size"] < 1
        ):
            raise ValueError(
                f"workload radius mismatch at {index}, k={radius}"
            )


def audit_readiness(paths, contract):
    validate_parent_files(paths)
    plan = read_jsonl(paths["b0_plan"])
    rows = read_jsonl(paths["workload"])
    readiness = read_json(paths["readiness"])
    if len(plan) != TOTAL_ROWS or len(rows) != TOTAL_ROWS:
        raise ValueError("readiness row count mismatch")
    for index, row in enumerate(rows):
        validate_workload_row(row, index, plan[index])
    max_spans = max(
        radius["unique_span_count"]
        for row in rows for radius in row["radii"].values()
    )
    max_pairs = max(
        radius["unique_pair_count"]
        for row in rows for radius in row["radii"].values()
    )
    max_all_spans = max(
        radius["all_allowed_unique_spans"]
        for row in rows for radius in row["radii"].values()
    )
    max_all_pairs = max(
        radius["all_allowed_unique_pairs"]
        for row in rows for radius in row["radii"].values()
    )
    max_sidecar = max(
        row["resource_counts"]["maximum_sidecar_bytes"] for row in rows
    )
    expected_counts = {
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "total_maximum_legal_input_tokens": sum(
            row["resource_counts"]["maximum_legal_input_tokens"]
            for row in rows
        ),
        "total_radius_unique_spans": sum(
            value["unique_span_count"]
            for row in rows for value in row["radii"].values()
        ),
        "total_radius_unique_pairs": sum(
            value["unique_pair_count"]
            for row in rows for value in row["radii"].values()
        ),
        "max_document_pruned_spans": max_spans,
        "max_document_pruned_pairs": max_pairs,
        "max_document_all_allowed_spans": max_all_spans,
        "max_document_all_allowed_pairs": max_all_pairs,
        "max_document_sidecar_bytes": max_sidecar,
        "documents_over_legacy_5m_pair_diagnostic": sum(
            max(value["unique_pair_count"] for value in row["radii"].values())
            > 5_000_000
            for row in rows
        ),
    }
    gates = {
        "all_3000_rows_completed": len(rows) == TOTAL_ROWS,
        "all_identity_and_structure_exact": True,
        "all_renderers_restore_original_units": True,
        "max_pruned_spans_le_2000000": max_spans <= SPAN_GATE,
        "max_pruned_pairs_le_50000000": max_pairs <= PAIR_GATE,
        "max_all_allowed_spans_le_2000000": max_all_spans <= SPAN_GATE,
        "max_all_allowed_pairs_le_50000000": max_all_pairs <= PAIR_GATE,
        "max_sidecar_le_50_gib": max_sidecar <= CACHE_GATE_BYTES,
        "free_disk_ge_5_gib": (
            readiness.get("resource_gate", {}).get("observed_free_bytes", 0)
            >= 5 * 1024**3
        ),
        "two_replay_estimate_le_72_gpu_hours": (
            readiness.get("runtime_upper_bound", {}).get(
                "conservative_two_replay_gpu_hours", math.inf
            )
            <= GPU_HOUR_GATE
        ),
    }
    checks = {
        "contract_valid": True,
        "parent_b0_valid": True,
        "workload_rows_sha_exact": (
            readiness.get("workload_rows", {}).get("sha256")
            == sha256_file(paths["workload"])
        ),
        "workload_counts_exact": readiness.get("counts") == expected_counts,
        "gates_recomputed_exact": readiness.get("gates") == gates,
        "readiness_go": (
            readiness.get("status") == "go"
            and readiness.get("terminal_reason") == "b1_readiness_go"
            and all(gates.values())
        ),
        "protocol_contract_runner_bound": (
            readiness.get("protocol_sha256") == PROTOCOL_SHA256
            and readiness.get("contract_sha256")
            == sha256_file(paths["contract"])
            and readiness.get("runner_sha256")
            == contract["runner_sha256"]
        ),
        "public_artifacts_contain_no_raw_text": (
            no_forbidden_keys(readiness) and all(
                no_forbidden_keys(row) for row in rows
            )
        ),
        "score_blind_attestation_zero": all(
            readiness.get("execution_attestation", {}).get(key) == 0
            for key in (
                "detector_source_imports",
                "detector_model_loads",
                "region_computations",
                "certificate_score_computations",
                "threshold_computations",
                "generation_calls",
            )
        ),
    }
    return {
        "schema_version": 1,
        "stage": f"{STAGE}-P0-audit",
        "status": "pass" if all(checks.values()) else "fail",
        "protocol_sha256": PROTOCOL_SHA256,
        "contract_sha256": sha256_file(paths["contract"]),
        "readiness_sha256": sha256_file(paths["readiness"]),
        "workload_rows_sha256": sha256_file(paths["workload"]),
        "counts": expected_counts,
        "gates": gates,
        "checks": checks,
        "audited_at": utc_now(),
        "contains_raw_text": False,
    }


def validate_certificate_row(row, index, plan, b0_row, workload):
    if (
        row.get("schema_version") != 1
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        or row.get("plan_index") != index
        or row.get("cohort") != plan["cohort"]
        or row.get("cohort_index") != plan["cohort_index"]
        or row.get("raw_text_sha256") != plan["raw_text_sha256"]
        or row.get("unit_count") != plan["unit_count"]
        or row.get("edge_count") != plan["unit_count"] - 1
        or row.get("original_hits") != b0_row["hit_count"]
        or row.get("original_score") != b0_row["direct_score"]
        or row.get("original_regions_sha256")
        != b0_row["regions_sha256"]
        or set(row.get("certificates", {})) != {"0", "1", "2", "3"}
        or not no_forbidden_keys(row)
    ):
        raise ValueError(f"certificate row mismatch at {index}")
    previous = math.inf
    for radius in RADII:
        value = row["certificates"][str(radius)]
        exact_hits = value["exact_min_hits"]
        witness = value.get("witness_boundaries")
        if (
            not isinstance(witness, list)
            or len(witness) != row["edge_count"]
            or any(
                isinstance(boundary, bool)
                or not isinstance(boundary, int)
                for boundary in witness
            )
            or witness != sorted(set(witness))
            or any(
                boundary <= 0 or boundary >= row["atom_count"]
                for boundary in witness
            )
            or len(
                set(workload["original_boundaries"]) - set(witness)
            ) > value["effective_radius"]
        ):
            raise ValueError(
                f"certificate witness mismatch at {index}, k={radius}"
            )
        if (
            value["requested_radius"] != radius
            or value["effective_radius"]
            != min(radius, row["edge_count"])
            or exact_hits > previous
            or value["cheap_min_hits"] > exact_hits
            or value["cheap_min_hits"]
            != max(
                0,
                row["original_hits"] - 5 * value["effective_radius"],
            )
            or exact_hits > row["original_hits"]
            or value["exact_score"]
            != score_from_hits(exact_hits, row["edge_count"])
            or value["cheap_score"]
            != score_from_hits(value["cheap_min_hits"], row["edge_count"])
            or value["witness_score"] != value["exact_score"]
            or len(value["witness_boundaries"]) != row["edge_count"]
            or value["unique_span_count"]
            != workload["radii"][str(radius)]["unique_span_count"]
            or value["unique_pair_count"]
            != workload["radii"][str(radius)]["unique_pair_count"]
            or value["certificate_sha256"] != certificate_digest(value)
        ):
            raise ValueError(
                f"certificate invariant mismatch at {index}, k={radius}"
            )
        previous = exact_hits
    if (
        row["certificates"]["0"]["exact_min_hits"] != row["original_hits"]
        or row["certificates"]["0"]["exact_score"] != row["original_score"]
        or row["certificates"]["0"]["witness_boundaries"]
        != workload["original_boundaries"]
    ):
        raise ValueError(f"k=0/B0 mismatch at {index}")


def classify(scores, threshold, confidence, lower):
    successes = sum(score > threshold for score in scores)
    result = {
        "threshold": threshold,
        "strict_exceedances": successes,
        "trials": len(scores),
        "rate": successes / len(scores),
        "one_sided_cp_confidence": confidence,
    }
    if lower:
        result["one_sided_cp_lower"] = clopper_pearson_lower(
            successes, len(scores), confidence
        )
    else:
        result["one_sided_cp_upper"] = clopper_pearson_upper(
            successes, len(scores), confidence
        )
    return result


def pair(exact_scores, cheap_scores, threshold):
    counts = Counter()
    for exact, cheap in zip(exact_scores, cheap_scores):
        exact_pass = exact > threshold
        cheap_pass = cheap > threshold
        key = (
            "both_certify" if exact_pass and cheap_pass
            else "exact_only" if exact_pass
            else "cheap_only" if cheap_pass
            else "neither"
        )
        counts[key] += 1
    return {
        key: counts[key]
        for key in (
            "both_certify",
            "exact_only",
            "cheap_only",
            "neither",
        )
    }


def reconstruct_statistics(rows, runtime):
    groups = {
        cohort: [row for row in rows if row["cohort"] == cohort]
        for cohort in (
            "calibration_null",
            "independent_test_null",
            "positive",
        )
    }
    if any(len(value) != COHORT_SIZE for value in groups.values()):
        raise ValueError("terminal cohort count mismatch")
    calibrations = {}
    thresholds = {}
    analyses = {}
    paired = {}
    for radius in RADII:
        key = str(radius)
        scores = [
            row["certificates"][key]["exact_score"]
            for row in groups["calibration_null"]
        ]
        calibration = split_conformal_calibrate(scores, 0.05)
        calibrations[key] = calibration.to_dict()
        thresholds[radius] = calibration.threshold
    for radius in RADII:
        key = str(radius)
        exact_test = [
            row["certificates"][key]["exact_score"]
            for row in groups["independent_test_null"]
        ]
        exact_positive = [
            row["certificates"][key]["exact_score"]
            for row in groups["positive"]
        ]
        cheap_test = [
            row["certificates"][key]["cheap_score"]
            for row in groups["independent_test_null"]
        ]
        cheap_positive = [
            row["certificates"][key]["cheap_score"]
            for row in groups["positive"]
        ]
        analyses[key] = {
            "exact_recalibrated": {
                "test": classify(
                    exact_test,
                    thresholds[radius],
                    0.975 if radius == 1 else 0.95,
                    False,
                ),
                "positive": classify(
                    exact_positive, thresholds[radius], 0.95, True
                ),
            },
            "exact_fixed_b0": {
                "test": classify(exact_test, B0_THRESHOLD, 0.95, False),
                "positive": classify(
                    exact_positive, B0_THRESHOLD, 0.95, True
                ),
            },
            "cheap_fixed_b0": {
                "test": classify(cheap_test, B0_THRESHOLD, 0.95, False),
                "positive": classify(
                    cheap_positive, B0_THRESHOLD, 0.95, True
                ),
            },
            "cheap_at_exact_threshold": {
                "test": classify(
                    cheap_test, thresholds[radius], 0.95, False
                ),
                "positive": classify(
                    cheap_positive, thresholds[radius], 0.95, True
                ),
            },
        }
        paired[key] = {
            "fixed_b0_test": pair(
                exact_test, cheap_test, B0_THRESHOLD
            ),
            "fixed_b0_positive": pair(
                exact_positive, cheap_positive, B0_THRESHOLD
            ),
            "exact_threshold_test": pair(
                exact_test, cheap_test, thresholds[radius]
            ),
            "exact_threshold_positive": pair(
                exact_positive, cheap_positive, thresholds[radius]
            ),
        }
    main = analyses["1"]["exact_recalibrated"]
    gates = {
        "k1_test_cp975_upper_le_0_08": (
            main["test"]["one_sided_cp_upper"] <= 0.08
        ),
        "k1_positive_rate_ge_0_20": main["positive"]["rate"] >= 0.20,
        "k1_positive_cp95_lower_gt_0_10": (
            main["positive"]["one_sided_cp_lower"] > 0.10
        ),
        "clean_k0_positive_rate_ge_0_80": (
            analyses["0"]["exact_recalibrated"]["positive"]["rate"]
            >= 0.80
        ),
        "k0_threshold_equals_b0": thresholds[0] == B0_THRESHOLD,
        "two_full_replays_exact": (
            runtime["replay_rows_sha256"]["0"]
            == runtime["replay_rows_sha256"]["1"]
        ),
        "all_cheap_only_counts_zero": all(
            table["cheap_only"] == 0
            for radius in paired.values()
            for table in radius.values()
        ),
        "peak_vram_below_0_80": runtime["peak_vram_fraction"] < 0.80,
        "all_3000_rows_completed": len(rows) == TOTAL_ROWS,
    }
    return calibrations, analyses, paired, gates


def audit_terminal(paths, contract, p0_audit):
    plan = read_jsonl(paths["b0_plan"])
    b0_rows = read_jsonl(paths["b0_rows"])
    workload = read_jsonl(paths["workload"])
    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    runtime = read_json(paths["runtime"])
    result = read_json(paths["result"])
    if not (
        len(plan) == len(b0_rows) == len(workload)
        == len(rows0) == len(rows1) == TOTAL_ROWS
    ):
        raise ValueError("terminal row count mismatch")
    for index, row in enumerate(rows0):
        validate_certificate_row(
            row, index, plan[index], b0_rows[index], workload[index]
        )
    if rows0 != rows1:
        raise ValueError("certificate replay row mismatch")
    calibrations, analyses, paired, gates = reconstruct_statistics(
        rows0, runtime
    )
    checks = {
        "p0_audit_pass": p0_audit["status"] == "pass",
        "contract_valid": True,
        "two_replays_byte_exact": (
            sha256_file(paths["rows0"]) == sha256_file(paths["rows1"])
            == runtime["replay_rows_sha256"]["0"]
            == runtime["replay_rows_sha256"]["1"]
        ),
        "all_rows_reconstructed": True,
        "calibration_exact": result.get("calibration") == calibrations,
        "analyses_exact": result.get("analyses") == analyses,
        "paired_counts_exact": result.get("paired") == paired,
        "gates_exact": result.get("gates") == gates,
        "status_exact": result.get("status")
        == ("go" if all(gates.values()) else "stop"),
        "protocol_and_code_bound": (
            result.get("protocol_sha256") == PROTOCOL_SHA256
            and result.get("contract_sha256")
            == sha256_file(paths["contract"])
            and result.get("runner_sha256") == contract["runner_sha256"]
        ),
        "runtime_metric_count_exact": (
            runtime.get("runtime_metric_rows") == REPLAY_COUNT * TOTAL_ROWS
        ),
        "public_artifacts_contain_no_raw_text": all(
            no_forbidden_keys(value)
            for value in (runtime, result, rows0, rows1)
        ),
    }
    return {
        "schema_version": 1,
        "stage": f"{STAGE}-terminal-audit",
        "status": "pass" if all(checks.values()) else "fail",
        "decision": result.get("status"),
        "checks": checks,
        "gates": gates,
        "artifacts": {
            key: {
                "sha256": sha256_file(paths[key]),
                "bytes": paths[key].stat().st_size,
            }
            for key in (
                "protocol",
                "contract",
                "workload",
                "readiness",
                "p0_audit",
                "rows0",
                "rows1",
                "runtime",
                "result",
            )
        },
        "audited_at": utc_now(),
        "contains_raw_text": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("readiness", "terminal"), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = paths_for(args.root, args.output_dir)
    contract = validate_contract(paths)
    p0_audit = audit_readiness(paths, contract)
    atomic_write_json(paths["p0_audit"], p0_audit)
    if p0_audit["status"] != "pass":
        raise ValueError("independent B1 readiness audit failed")
    if args.phase == "readiness":
        print(json.dumps(p0_audit, sort_keys=True), flush=True)
        return
    terminal = audit_terminal(paths, contract, p0_audit)
    atomic_write_json(paths["audit"], terminal)
    if terminal["status"] != "pass":
        raise ValueError("independent B1 terminal audit failed")
    print(json.dumps(terminal, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
