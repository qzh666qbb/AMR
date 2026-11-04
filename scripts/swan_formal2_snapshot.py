"""Read-only export of ongoing experiments and W05 formal continuation."""
import argparse
import base64
import json
from pathlib import Path

import swan_successors2_snapshot as base


def snapshot(export=False):
    base.JOBS.update({
        "formal_null": ("normal_formal2_runtime", "normal_formal2_queue", "null"),
        "formal_alignment": ("normal_formal2_runtime/alignment", "normal_formal2_scoring_queue", "alignment"),
        "confirmation_generation": (
            "v4_confirmation2_runtime/experiments/swan_v4_independent_confirmation2/generation",
            "v4_confirmation2_queue", "generate"),
        "confirmation": (
            "v4_confirmation2_runtime/experiments/swan_v4_independent_confirmation2",
            "v4_confirmation2_queue", "score"),
    })
    payload = base.snapshot(export)
    null = base.ROOT / "normal_formal2_runtime"
    for label, directory in (("formal_null", null), ("formal_alignment", null / "alignment")):
        job = payload["jobs"][label]
        if label == "formal_null":
            job["null_document_records"] = len(list((directory / "private/null_documents").glob("*.json")))
        names = (("SOURCE_REFERENCE.json", "MISSING_RESPONSE_POLICY.json", "LLM_NULL_BUDGET.json",
                  "MISSING_RESPONSE_DISPOSITION.json", "LLM_NULL_GENERATION_SEAL.json")
                 if label == "formal_null" else
                 ("MACHINE_CALIBRATION.json", "BASELINE_SELECTION.json", "COVERAGE_AND_INFERENCE.json"))
        for name in names:
            path = directory / name
            if path.is_file():
                job[name] = {"sha256": base.sha(path)}
                if export:
                    payload["exports"][label + "/" + name] = {
                        "sha256": base.sha(path),
                        "base64": base64.b64encode(path.read_bytes()).decode()}
        if job["machine_closure"]:
            if label == "formal_null":
                audit = base.read(directory / "AUDIT.json")
                for name, expected in audit["files"].items():
                    assert base.sha(directory / name) == expected
                job["final_sealed_files_verified"] = len(audit["files"])
            else:
                audit = base.read(directory / "AUDIT.json")
                for field, filename in (
                    ("prediction_seal_sha256", "PREDICTION_SEAL.json"),
                    ("raw_graph_seal_sha256", "RAW_GRAPH_SEAL.json"),
                    ("calibration_sha256", "MACHINE_CALIBRATION.json"),
                    ("single_view_pretest_seal_sha256", "BASELINE_SELECTION.json")):
                    assert audit[field] == base.sha(directory / filename)
                seal = base.read(directory / "RAW_GRAPH_SEAL.json")
                for name, expected in seal["files"].items():
                    assert base.sha(directory / name) == expected
                predictions = base.read(directory / "PREDICTION_SEAL.json")
                assert predictions["predictions_sha256"] == base.sha(directory / "private/PREDICTIONS.json")
                job["final_graph_files_verified"] = len(seal["files"])
                job["complete_prediction_rows"] = len(base.read(directory / "private/PREDICTIONS.json"))
                job["aggregate_metrics"] = base.read(directory / "RESULT.json")
    folder = base.ROOT / "normal_formal2_scoring_queue"
    for name in ("COORDINATOR.json", "COORDINATOR_STATUS.json"):
        if (folder / name).exists():
            payload["jobs"]["formal_alignment"][name] = base.read(folder / name)
    confirmation = base.ROOT / "v4_confirmation2_runtime/experiments/swan_v4_independent_confirmation2"
    if confirmation.exists():
        job = payload["jobs"]["confirmation"]
        job["scored_rows"] = len(list((confirmation / "results/rows").glob("*.json")))
        job["inherited_document_records"] = 30
        for name in ("GENERATION_COMPLETE.json", "GENERATION_READINESS.json",
                     "SOURCE_REFERENCE.json", "MISSING_RESPONSE_POLICY.json",
                     "MISSING_RESPONSE_DISPOSITION.json", "FINAL_VERIFICATION.json"):
            path = confirmation / name
            if path.is_file():
                job[name] = {"sha256": base.sha(path)}
                if export:
                    payload["exports"]["confirmation/" + name] = {
                        "sha256": base.sha(path), "base64": base64.b64encode(path.read_bytes()).decode()}
        if (confirmation / "HEARTBEAT.json").exists():
            beat = base.read(confirmation / "HEARTBEAT.json")
            job["HEARTBEAT.json"].update({k: beat[k] for k in
                ("stage", "completed_documents", "budget") if k in beat})
        if job["machine_closure"]:
            audit = base.read(confirmation / "AUDIT.json")
            assert audit["row_count"] == len(audit["rows"]) == 8500
            for name, expected in audit["rows"].items():
                path = confirmation / name
                assert base.sha(path) == expected
                row = base.read(path)
                if row.get("evidence_path"):
                    assert base.sha(confirmation / row["evidence_path"]) == row["evidence_sha256"]
            verified = base.read(confirmation / "FINAL_VERIFICATION.json")
            assert verified["status"] == "PASS"
            assert verified["result_sha256"] == base.sha(confirmation / "RESULT.json")
            assert verified["audit_sha256"] == base.sha(confirmation / "AUDIT.json")
            job["final_score_rows_verified"] = 8500
            job["aggregate_metrics"] = base.read(confirmation / "RESULT.json")
        if payload["jobs"]["confirmation_generation"]["machine_closure"]:
            seal = base.read(confirmation / "generation/AUDIT.json")
            for name, expected in seal["files"].items():
                assert base.sha(confirmation / name) == expected
            payload["jobs"]["confirmation_generation"]["sealed_files_verified"] = len(seal["files"])
    for label, directory in (("formal_null", null), ("calibration", base.ROOT / "independent_calibration2_runtime")):
        path = directory / "HEARTBEAT.json"
        if path.exists():
            beat = base.read(path)
            payload["jobs"][label]["HEARTBEAT.json"].update(
                {k: beat[k] for k in ("eligible", "budget", "at") if k in beat})
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    print(json.dumps(snapshot(parser.parse_args().export), sort_keys=True))
