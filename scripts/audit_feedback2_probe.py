"""Audit frozen probe records and cache accounting without API or model calls."""

import argparse
import json
import math
import os
from pathlib import Path

from run_confirmation_experiment import digest
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_length_probe import OUTPUT, assess, check


def audit(folder):
    import nltk
    manifest = check(folder)
    runtime = json.loads((folder / "runtime.json").read_text())
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Runtime manifest mismatch")
    original = ROOT / "experiments/confirmation_three_arm30/preparation.json"
    prior_ids = {r["prompt_id"] for r in json.loads(original.read_text())["prompts"]}
    if prior_ids & {r["prompt_id"] for r in manifest["prompts"]}:
        raise ValueError("Historical prompt overlap")
    summary = json.loads((folder / "summary.json").read_text())
    evidence = {str(p.relative_to(folder)): sha(p) for p in
                (folder / "manifest.json", folder / "runtime.json", folder / "summary.json")}
    successes, checked = {}, 0
    for job in manifest["jobs"]:
        path = folder / "evaluated" / f"{job['id']}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        call_path = folder / "calls" / f"{job['id']}-000.json"
        call = json.loads(call_path.read_text())
        if (call["status"] != "complete" or call["request"] != job["request"]
                or call["request_sha256"] != digest(job["request"])
                or call["response"]["model"] != runtime["model_returned"]):
            raise ValueError("API request or model identity mismatch")
        if (row["job_sha256"] != digest(job) or row["runtime_sha256"] != digest(runtime)
                or row["text_sha256"] != text_sha(row["text"])
                or row["text"] != call["response"]["content"].strip()
                or row["finish_reason"] != call["response"]["finish_reason"]):
            raise ValueError("Generated text checkpoint mismatch")
        detection = row["detection"]
        details = detection["details"]
        if nltk.sent_tokenize(row["text"]) != [d["text"] for d in details]:
            raise ValueError("Segmentation mismatch")
        for detail in details:
            scores = detail["bank_scores"]
            if len(scores) != 50 or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
                raise ValueError("Invalid bank scores")
            if max(scores) != detail["score"]:
                raise ValueError("Maximum bank score mismatch")
            cache_root = folder / "sentence_cache" / digest(runtime)
            score_path = cache_root / "score" / f'{digest({"amr": detail["amr"]})}.json'
            parse_path = cache_root / "parse" / f'{digest({"text": detail["text"]})}.json'
            score_cache = json.loads(score_path.read_text())
            parse_cache = json.loads(parse_path.read_text())
            if score_cache["value"] != scores or score_cache["value_sha256"] != digest(scores):
                raise ValueError("Score cache mismatch")
            if parse_cache["value_sha256"] != digest(parse_cache["value"]):
                raise ValueError("Parse cache checksum mismatch")
            if parse_cache["value"] != [{k: detail[k] for k in ("text", "raw_amr", "amr")}]:
                raise ValueError("Parse cache mismatch")
            for item in (score_path, parse_path):
                evidence[str(item.relative_to(folder))] = sha(item)
        n = len(details)
        green = sum(d["score"] >= .65 for d in details)
        z = (green - n * .25) / math.sqrt(n * .25 * .75)
        if n != detection["n"] or green != detection["green"] or not math.isclose(z, detection["z"], abs_tol=1e-12):
            raise ValueError("Detector aggregation mismatch")
        target = details[0]["bank_scores"][job["template_index"]] if n == 1 else None
        if target != row["target_score"]:
            raise ValueError("Target score mismatch")
        expected = assess(row["text"], row["finish_reason"], details, target, job["word_range"])
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("Feasibility label mismatch")
        successes[row["stratum"]] = successes.get(row["stratum"], 0) + row["joint_success"]
        for item in (path, call_path):
            evidence[str(item.relative_to(folder))] = sha(item)
        checked += 1
    for stratum, values in summary["strata"].items():
        if successes.get(stratum, 0) != values["joint_success"]:
            raise ValueError("Summary count mismatch")
    complete = checked == len(manifest["jobs"])
    if summary["paragraph_pilot_gate_passed"] != (
        all(v["joint_success"] >= 4 for v in summary["strata"].values()) if complete else None
    ):
        raise ValueError("Expansion gate mismatch")
    payload = {
        "status": "passed" if complete else "passed_completed_records_only",
        "checked_jobs": checked, "planned_jobs": len(manifest["jobs"]),
        "new_api_calls": 0, "new_parser_calls": 0, "new_s2match_calls": 0,
        "mismatches": 0, "input_sha256": evidence,
        "audit_script_sha256": sha(Path(__file__)),
        "limitation": "Recomputed identities, labels and aggregation from saved model outputs; not an independent parser or stochastic S2Match rerun",
    }
    write_json(folder / "audit.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "input_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    audit(args.output)


if __name__ == "__main__":
    main()
