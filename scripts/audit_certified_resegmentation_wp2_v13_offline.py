"""Replay B0 preparation on synthetic data and public historical scores only."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import certified_resegmentation_wp2_v13_b0_support as b0


HISTORICAL_R1_SHA256 = "4517c64cda80a034048ad7b315bb4657a9a3e3988937c5068fda17641c911023"
HISTORICAL_R2_SHA256 = "398e88d37d49bb8b1bb1d88a7602bde8d34618880ac33843d9b0c6408ff9f6bf"


def run_audit(root):
    base = root / "experiments/certified_resegmentation_wp2_v11_detector_reproduction1"
    inputs = {}
    for name, expected in (("r1_semstamp.json", HISTORICAL_R1_SHA256),
                           ("r2_semstamp.json", HISTORICAL_R2_SHA256)):
        path = base / name
        if b0.file_sha256(path) != expected:
            raise ValueError(f"historical SHA mismatch: {name}")
        inputs[name] = {"sha256": expected, "bytes": path.stat().st_size}
    rows = json.loads((base / "r1_semstamp.json").read_text())["execution"]["rows"]
    for row in rows:
        if not (
            b0.score_from_counts(row["hit_count"], row["edge_count"])
            == row["direct_score"] == row["official_score"]
        ):
            raise AssertionError("historical score/count/official mismatch")
    history = b0.summarize_scores(*[
        [row["direct_score"] for row in rows if row["cohort"] == cohort]
        for cohort in ("certificate_calibration", "independent_test", "positive")
    ])
    previous = json.loads((base / "r2_semstamp.json").read_text())["M1_V10_SPLIT"]
    if history["calibration"] != previous["calibration"]:
        raise AssertionError("historical calibration mismatch")
    cp_errors = []
    for cohort in ("test", "positive"):
        for key, value in history[cohort].items():
            if key in ("one_sided_cp_lower", "one_sided_cp_upper"):
                cp_errors.append(abs(value - previous[cohort][key]))
            elif value != previous[cohort][key]:
                raise AssertionError("historical count/rate mismatch")
    if max(cp_errors) > 1e-12:
        raise AssertionError("historical CP numerical mismatch")

    plan = [{
        "plan_index": index,
        "cohort": b0.COHORTS[index // 1000],
        "cohort_index": index % 1000,
        "raw_text_sha256": hashlib.sha256(f"offline-fixture-{index}".encode()).hexdigest(),
        "unit_count": 4, "max_native_tokens": 12,
    } for index in range(3000)]
    relation = bytes(int(b in (a, (a + 3) % 8)) for a in range(8) for b in range(8))
    contract = b0.digest({"scope": "synthetic_fixture_only_not_a_B0_protocol"})
    checkpoint = {
        "schema_version": 1, "contract_sha256": contract,
        "plan_sha256": b0.digest(plan), "replays": {},
    }
    # No runtime override is offered by b0_support; this mock is fixture-local.
    with patch.object(b0, "RELATION_SHA256", hashlib.sha256(relation).hexdigest()):
        for replay in ("0", "1"):
            entries = []
            for row in plan:
                positive = row["cohort"] == "positive"
                regions = [0, 0, 0, 0] if positive else [0, 1, 2, 3]
                entries.append({
                    "public": b0.make_public_score(row, regions, relation, 3.0 if positive else -1.0),
                    "regions": regions,
                })
            checkpoint["replays"][replay] = entries
        synthetic = b0.summarize_checkpoint(checkpoint, plan, relation, contract)
    if any(name in sys.modules for name in (
        "torch", "transformers", "sentence_transformers", "nltk")):
        raise AssertionError("offline audit imported a model/tokenizer dependency")
    return {
        "schema_version": 1, "status": "pass",
        "scope": "offline_preparation_not_B0_readiness_or_detector_health",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "historical_inputs": inputs,
        "historical_regression": {
            "rows": len(rows), "scores_exact": True, "counts_and_threshold_exact": True,
            "cp_max_abs_error_vs_frozen_linux": max(cp_errors),
            "cp_cross_platform_tolerance": 1e-12,
            "statistics": history,
        },
        "synthetic_3000_by_2_replay": synthetic,
        "attestation": {
            "detector_calls": 0, "model_imports": 0, "tokenizer_imports": 0,
            "sealed_inputs_read": 0, "real_certificate_calls": 0,
            "new_real_scores_computed": 0, "historical_score_recomputation_only": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b0.canonical_bytes(result) + b"\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
