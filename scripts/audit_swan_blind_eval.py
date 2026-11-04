"""Audit exported whole-document results and write an aggregate SHA manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.swan_blind_protocol import digest
from scripts.analyze_swan_blind_eval import ATTACKS, save_json


def audit(folder):
    freeze = json.loads((folder / "FREEZE.json").read_text())
    readiness = json.loads((folder / "READINESS.json").read_text())
    fingerprint = digest(freeze)
    if readiness["freeze_sha256"] != fingerprint or not readiness["ready"]:
        raise ValueError("Readiness/freeze binding failed")
    heartbeat_path = folder / "heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}
    if heartbeat and heartbeat["freeze_sha256"] != fingerprint:
        raise ValueError("Heartbeat/freeze binding failed")
    checks = {
        "readiness_binds_freeze": True, "readiness_passed": True,
        "historical_scores_unchanged": hashlib.sha256((ROOT /
            "experiments/swan_multiview_development1/dataset5_eval/PROTOCOL_RESULTS.json"
            ).read_bytes()).hexdigest() == freeze["historical_scores_sha256"],
        "independent_human_test_available": False,
        "full_cohort_complete": False, "permanent_halt_present": (folder / "HALT.json").exists(),
        "evaluation_methods_completed": [],
    }
    if (folder / "RESULTS.json").exists():
        results = json.loads((folder / "RESULTS.json").read_text())
        calibration = json.loads((folder / "CALIBRATION.json").read_text())
        documents = json.loads((folder / "DOCUMENT_SCORES.json").read_text())
        if results["freeze_sha256"] != fingerprint \
                or documents["freeze_sha256"] != fingerprint:
            raise ValueError("Result/freeze binding failed")
        methods = results["complete_methods"]
        checks["evaluation_methods_completed"] = methods
        checks["full_cohort_complete"] = methods == freeze["evaluation_methods"]
        records = documents["documents"]
        if set(methods) != {r["method"] for r in records}:
            raise ValueError("Method result and document snapshots differ")
        if len({r["record_id"] for r in records}) != len(records):
            raise ValueError("Duplicate document records")
        expected_ids = freeze["source_ids"]["positive"]
        for method in methods:
            data = [r for r in records if r["method"] == method]
            expected = {(arm, doc, attack) for arm in ("positive", "paired_negative")
                        for doc in expected_ids for attack in ATTACKS}
            if {(r["arm"], r["document_id"], r["attack"]) for r in data} != expected \
                    or len(data) != 420:
                raise ValueError("Completed method has incomplete document denominator")
            if any(r["complete"] and len(r["unit_bank_max"]) != r["N_expected"] for r in data):
                raise ValueError("Complete document lacks all unit scores")
            config_method = "P0" if method.startswith("P0") else method
            if any(r["configuration_sha256"] !=
                   freeze["configuration_sha256"][config_method] for r in data):
                raise ValueError("Document configuration differs from freeze")
            if any(r["split"] != ("calibration" if r["document_id"] in
                   freeze["calibration_ids"] else "test") for r in data):
                raise ValueError("Document split differs from freeze")
            cal_rows = [r for r in data if r["arm"] == "paired_negative"
                        and r["split"] == "calibration"]
            cal_manifest = digest([(r["record_id"], r["text_sha256"]) for r in cal_rows])
            if calibration[method]["negative_manifest_sha256"] != cal_manifest:
                raise ValueError("Calibration negative manifest mismatch")
            if {r["document_id"] for r in cal_rows} != set(freeze["calibration_ids"]):
                raise ValueError("Calibration IDs differ from freeze")
            for op in results["methods"][method]["operating_points"].values():
                for attack in ATTACKS:
                    measured = op["conditions"][attack]
                    if any(measured[key]["denominator"] != 15 for key in ("TPR", "FPR")):
                        raise ValueError("Primary test denominator changed")
                    for key in ("TPR", "FPR"):
                        m = measured[key]
                        if m["rate"] is not None and m["rate"] != \
                                m["decisions_positive"] / m["denominator"]:
                            raise ValueError("Rate does not retain whole denominator")
        checks["complete_method_denominators_verified"] = True
        checks["calibration_manifest_verified"] = True
        checks["document_status_counts"] = dict(Counter(
            r["method"] + (":complete" if r["complete"] else ":unresolved")
            for r in records))
        checks["full_cohort_complete"] = checks["full_cohort_complete"] and \
            heartbeat.get("state") == "complete" and not checks["permanent_halt_present"]
    save_json(folder / "AUDIT.json", {"freeze_sha256": fingerprint, **checks})
    extensions = {".json", ".csv", ".pdf", ".svg", ".png", ".md"}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.iterdir()) if p.is_file()
              and p.suffix in extensions and p.name != "SHA256SUMS.json"}
    save_json(folder / "SHA256SUMS.json", hashes)
    return checks


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    print(json.dumps(audit(cli.parse_args().output), indent=2))
