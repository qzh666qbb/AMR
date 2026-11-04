"""Derive official unit counts and statistic capacity from completed scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import ATTACKS, save_json


def diagnose(folder):
    result = json.loads((folder / "RESULTS.json").read_text())
    rows = json.loads((folder / "DOCUMENT_SCORES.json").read_text())["documents"]
    out = {"freeze_sha256": result["freeze_sha256"],
           "derived_from_completed_document_scores": True,
           "no_new_detector_calls": True,
           "bound": "G <= N implies z <= sqrt(3N) for lambda=0.25",
           "methods": {}}
    for method in ("P0_raw", "P0_strict"):
        if method not in result["methods"]:
            continue
        measured = result["methods"][method]
        data = [row for row in rows if row["method"] == method]
        conditions = {}
        for attack in ATTACKS:
            groups = {}
            for arm, split in (("paired_negative", "calibration"),
                               ("paired_negative", "test"), ("positive", "test")):
                cohort = [row for row in data if row["attack"] == attack
                          and row["arm"] == arm and row["split"] == split]
                complete = [row for row in cohort if row["complete"]]
                values = [v for row in complete for v in row["unit_bank_max"]]
                greens = sum(v is not None and v >= .65 for v in values)
                bounds = {}
                for name, op in measured["operating_points"].items():
                    threshold = op["document_threshold"]
                    bounds[name] = {
                        "document_threshold": threshold,
                        "complete_documents_unable_to_exceed_threshold_even_if_all_units_green":
                            sum(math.sqrt(3 * row["N_expected"]) <= threshold
                                for row in complete) if threshold is not None else None,
                        "unresolved_documents": len(cohort) - len(complete),
                        "full_denominator": len(cohort)}
                groups[arm + ":" + split] = {
                    "complete_documents": len(complete), "denominator": len(cohort),
                    "actual_N_histogram": dict(Counter(row["N_expected"] for row in cohort)),
                    "complete_scored_units": len(values), "complete_green_units": greens,
                    "green_rate_in_complete_documents": greens / len(values) if values else None,
                    "statistic_capacity": bounds}
            conditions[attack] = groups
        out["methods"][method] = conditions
    save_json(folder / "OFFICIAL_DIAGNOSTICS.json", out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    out = diagnose(parser.parse_args().output)
    print(json.dumps({"derived_methods": list(out["methods"])}, indent=2))
