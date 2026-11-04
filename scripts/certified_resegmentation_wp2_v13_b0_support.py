"""Model-free B0 support. Importing this module never loads a detector.

This is preparation code, not a frozen B0 runner or a readiness verdict.
Raw-text/tokenizer mapping and runtime/model attestation remain separate gates.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from certified_resegmentation_calibration import (
    clopper_pearson_upper,
    split_conformal_calibrate,
    strict_exceedance_count,
)


COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
RELATION_SHA256 = "4a8c517498d800a398414485a7e218e871a24d79a72238920b0c3f4ca29c53f1"
H0_RUNNER_SHA256 = "ec5a54b2d51a3f79d8808fb6b0e92c99fbffd18a3e4fc3d7fa1ee72947db7396"
OVERLAP_LABELS = {
    "v13_attempt_sources", "v12_all_sources", "v12_calibration_null",
    "v12_independent_test_null", "historical_all", "historical_positive",
    "v12_smoke_outputs", "g1_failed_outputs",
}
PLAN_FIELDS = {
    "plan_index", "cohort", "cohort_index", "raw_text_sha256",
    "unit_count", "max_native_tokens",
}
SCORE_FIELDS = {
    "schema_version", "edge_count", "hit_count", "regions_sha256", "hits_sha256",
    "relation_sha256", "direct_score", "official_score", "score_abs_error",
    "contains_raw_text",
}


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha(value):
    if (
        not isinstance(value, str) or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid SHA256")
    return value


def require_integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {label}")
    return value


def finite_score(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("score must be finite numeric, not bool")
    return float(value)


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_h0(h0):
    if (
        h0.get("schema_version") != 1
        or h0.get("stage") != "WP2-v13-H0-identity-freeze"
        or h0.get("status") != "go"
        or h0.get("scope") != "identity_only_no_detector_scores"
        or h0.get("contains_raw_text") is not False
        or h0.get("runner_sha256") != H0_RUNNER_SHA256
        or h0.get("decision", {}).get("h0_identity_freeze") != "go"
        or h0.get("decision", {}).get("b0_detector_health")
        != "authorized_to_freeze_protocol"
        or h0.get("decision", {}).get("b1_exact_certificate") != "stop"
        or h0.get("resource_gate", {}).get("below_0_80") is not True
    ):
        raise ValueError("H0 is not a terminal authorized identity Go")
    counts = h0.get("counts", {})
    for field in ("positive", "positive_unique", "v12_calibration_null",
                  "v12_independent_test_null"):
        require_integer(counts.get(field), COHORT_SIZE, COHORT_SIZE, field)
    attempts = require_integer(counts.get("g1_attempts"), 1000, 1600, "attempts")
    require_integer(counts.get("g1_failures"), attempts - 1000, attempts - 1000,
                    "failures")
    overlaps = h0.get("overlaps", {})
    if set(overlaps) != OVERLAP_LABELS:
        raise ValueError("H0 overlap evidence missing")
    for evidence in overlaps.values():
        require_integer(evidence.get("count"), 0, 0, "overlap count")
    attestation = h0.get("execution_attestation", {})
    for field in ("detector_imports", "detector_model_loads", "detector_score_reads",
                  "detector_scores_computed", "resegmentation_calls", "certificate_calls"):
        require_integer(attestation.get(field), 0, 0, field)
    for field in ("sealed_checkpoint_content_parsed", "raw_text_emitted_to_public_artifacts"):
        if attestation.get(field) is not False:
            raise ValueError("H0 score-blind attestation failed")


def validate_plan(plan):
    if len(plan) != 3 * COHORT_SIZE:
        raise ValueError("plan must contain exactly 3000 identities")
    identities = []
    for index, row in enumerate(plan):
        if set(row) != PLAN_FIELDS:
            raise ValueError("unexpected plan fields")
        require_integer(row["plan_index"], index, index, "plan index")
        require_integer(row["cohort_index"], index % 1000, index % 1000, "cohort index")
        if row["cohort"] != COHORTS[index // 1000]:
            raise ValueError("cohort order drift")
        identities.append(require_sha(row["raw_text_sha256"]))
        require_integer(row["unit_count"], 2, 2**31 - 1, "unit count")
        require_integer(row["max_native_tokens"], 1, 512, "native token count")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate/overlapping cohort identities")
    return plan


def make_public_plan(v12_rows, positives):
    """Keep manifest order; this pure helper alone does NOT authorize B0."""
    if len(v12_rows) != 3500 or len(positives) != 1000:
        raise ValueError("source manifest counts changed")
    plan = []
    for cohort in COHORTS:
        rows = positives if cohort == "positive" else [
            row for row in v12_rows if row.get("cohort") == cohort
        ]
        if len(rows) != COHORT_SIZE:
            raise ValueError("cohort count changed")
        for index, row in enumerate(rows):
            positive = cohort == "positive"
            source_index = row.get("positive_index" if positive else "cohort_index")
            require_integer(source_index, index, index, "manifest order")
            if row.get("contains_raw_text") is not False:
                raise ValueError("manifest is not explicitly public")
            plan.append({
                "plan_index": len(plan), "cohort": cohort, "cohort_index": index,
                "raw_text_sha256": row.get(
                    "output_raw_text_sha256" if positive else "raw_text_sha256"),
                "unit_count": row.get("unit_count" if positive else "semstamp_unit_count"),
                "max_native_tokens": row.get(
                    "max_native_tokens" if positive else "semstamp_max_native_tokens"),
            })
    return validate_plan(plan)


def load_public_plan(root, h0_path, expected_h0_sha256):
    """Read terminal public manifests bound to a future frozen H0 SHA.

    Does not open the sealed checkpoint; its runtime validation is still required.
    """
    if file_sha256(h0_path) != require_sha(expected_h0_sha256):
        raise ValueError("H0 file SHA mismatch")
    h0 = json.loads(Path(h0_path).read_text(encoding="utf-8"))
    validate_h0(h0)
    root = Path(root).resolve()

    def bound_public_file(label):
        meta = h0["inputs"][label]
        relative = Path(meta["relative_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("input path escapes experiment root")
        if (
            file_sha256(path) != require_sha(meta["sha256"])
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"{label} bytes/SHA mismatch")
        return path

    g1 = json.loads(bound_public_file("g1_result").read_text(encoding="utf-8"))
    if (
        g1.get("stage") != "WP2-v13-G1-SemStamp-generation"
        or g1.get("status") != "go"
        or g1.get("terminal_reason") != "target_eligible_reached"
        or g1.get("counts", {}).get("eligible_count") != 1000
        or g1.get("counts", {}).get("attempt_count") != h0["counts"]["g1_attempts"]
    ):
        raise ValueError("G1 is not the expected Go terminal")
    public_rows = {}
    for label in ("v12_source_identities", "positive_identities"):
        public_rows[label] = [
            json.loads(line)
            for line in bound_public_file(label).read_text(encoding="utf-8").splitlines()
            if line
        ]
    return make_public_plan(
        public_rows["v12_source_identities"], public_rows["positive_identities"])


def validate_relation(relation):
    values = tuple(relation)
    if (
        len(values) != 64
        or any(type(v) is not int or v not in (0, 1) for v in values)
    ):
        raise ValueError("relation must contain 64 binary bytes")
    encoded = bytes(values)
    if (
        hashlib.sha256(encoded).hexdigest() != RELATION_SHA256
        or any(sum(values[i:i + 8]) != 2 for i in range(0, 64, 8))
    ):
        raise ValueError("frozen directed relation mismatch")
    return encoded


def score_from_counts(hits, edges):
    require_integer(edges, 1, 2**31 - 1, "edge count")
    require_integer(hits, 0, edges, "hit count")
    return (hits - 0.25 * edges) / math.sqrt(edges * 0.25 * 0.75)


def make_public_score(planned, regions, relation, official_score):
    relation = validate_relation(relation)
    regions = list(regions)
    if len(regions) != planned["unit_count"]:
        raise ValueError("region/unit count mismatch")
    for region in regions:
        require_integer(region, 0, 7, "region")
    hits = [relation[a * 8 + b] for a, b in zip(regions, regions[1:])]
    direct = score_from_counts(sum(hits), len(hits))
    official = finite_score(official_score)
    if direct != official:
        raise ValueError("official/direct score error must be exactly zero")
    row = {
        **planned, "schema_version": 1,
        "edge_count": len(hits), "hit_count": sum(hits),
        "regions_sha256": hashlib.sha256(bytes(regions)).hexdigest(),
        "hits_sha256": hashlib.sha256(bytes(hits)).hexdigest(),
        "relation_sha256": RELATION_SHA256,
        "direct_score": direct, "official_score": official, "score_abs_error": 0.0,
        "contains_raw_text": False,
    }
    validate_public_score(row, planned)
    return row


def validate_public_score(row, planned):
    if set(row) != PLAN_FIELDS | SCORE_FIELDS:
        raise ValueError("unexpected score row fields")
    if canonical_bytes({key: row[key] for key in PLAN_FIELDS}) != canonical_bytes(planned):
        raise ValueError("score identity or structure drift")
    if (
        type(row["schema_version"]) is not int or row["schema_version"] != 1
        or row["contains_raw_text"] is not False
        or row["relation_sha256"] != RELATION_SHA256
    ):
        raise ValueError("score row header mismatch")
    require_integer(row["edge_count"], planned["unit_count"] - 1,
                    planned["unit_count"] - 1, "edge count")
    expected = score_from_counts(row["hit_count"], row["edge_count"])
    if (
        finite_score(row["direct_score"]) != expected
        or finite_score(row["official_score"]) != expected
        or finite_score(row["score_abs_error"]) != 0
    ):
        raise ValueError("score/count/official mismatch")
    require_sha(row["regions_sha256"])
    require_sha(row["hits_sha256"])


def summarize_scores(calibration, test, positive):
    """Generic statistics for historical regression too; not a B0 Go verdict."""
    groups = [tuple(finite_score(v) for v in scores)
              for scores in (calibration, test, positive)]
    if any(not group for group in groups):
        raise ValueError("empty cohort")
    calibrated = split_conformal_calibrate(groups[0], 0.05)
    result = {"calibration": calibrated.to_dict()}
    for name, group in zip(("test", "positive"), groups[1:]):
        hits = strict_exceedance_count(group, calibrated.threshold)
        result[name] = {
            "trials": len(group), "strict_exceedances": hits, "rate": hits / len(group),
            "one_sided_cp_confidence": 0.95,
            "one_sided_cp_upper": clopper_pearson_upper(hits, len(group), 0.95),
            "one_sided_cp_lower": 1 - clopper_pearson_upper(len(group) - hits, len(group), 0.95),
        }
    return result


def validate_checkpoint(checkpoint, plan, relation, contract_sha256):
    validate_plan(plan)
    require_sha(contract_sha256)
    if (
        set(checkpoint) != {"schema_version", "contract_sha256", "plan_sha256", "replays"}
        or checkpoint["schema_version"] != 1
        or checkpoint["contract_sha256"] != contract_sha256
        or checkpoint["plan_sha256"] != digest(plan)
        or set(checkpoint["replays"]) != {"0", "1"}
    ):
        raise ValueError("checkpoint contract/plan/header drift")
    zero, one = (checkpoint["replays"][key] for key in ("0", "1"))
    if (
        not isinstance(zero, list) or not isinstance(one, list)
        or len(zero) > len(plan) or len(one) > len(plan)
        or (one and len(zero) != len(plan))
    ):
        raise ValueError("invalid replay prefix")
    for replay in (zero, one):
        for index, entry in enumerate(replay):
            if set(entry) != {"public", "regions"}:
                raise ValueError("unexpected checkpoint entry")
            rebuilt = make_public_score(
                plan[index], entry["regions"], relation, entry["public"]["official_score"])
            if canonical_bytes(rebuilt) != canonical_bytes(entry["public"]):
                raise ValueError("checkpoint region/hit/row evidence mismatch")
    for index, entry in enumerate(one):
        if canonical_bytes(entry) != canonical_bytes(zero[index]):
            raise ValueError("replay mismatch")
    return {"completed": [len(zero), len(one)], "complete": len(one) == len(plan)}


def summarize_checkpoint(checkpoint, plan, relation, contract_sha256):
    state = validate_checkpoint(checkpoint, plan, relation, contract_sha256)
    if not state["complete"]:
        raise ValueError("summary forbidden before both full replays finish")
    rows = [entry["public"] for entry in checkpoint["replays"]["0"]]
    stats = summarize_scores(*[
        [row["direct_score"] for row in rows if row["cohort"] == cohort]
        for cohort in COHORTS
    ])
    gates = {
        "fpr_cp95_upper_le_0_08": stats["test"]["one_sided_cp_upper"] <= 0.08,
        "tpr_ge_0_80": stats["positive"]["rate"] >= 0.80,
        "tpr_cp95_lower_gt_0_75": stats["positive"]["one_sided_cp_lower"] > 0.75,
        "two_full_replays_exact": True,
        "official_direct_error_zero": True,
    }
    return {
        "scope": "model_free_validation_only_requires_separate_runtime_attestation",
        "statistics": stats, "gates": gates,
        "statistics_pass": all(gates.values()),
        "public_rows_sha256": digest(rows), "contract_sha256": contract_sha256,
        "b1_authorized": False,
    }


def atomic_write_checkpoint(path, value):
    """Atomic 0600 JSON write; caller must validate the checkpoint first."""
    encoded = canonical_bytes(value) + b"\n"
    path = Path(path)
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
