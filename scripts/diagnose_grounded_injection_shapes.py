"""Exploratory triple-count upper bounds from already-cached injection graphs."""

import contextlib
import io
import json
from pathlib import Path
import statistics
import sys

from run_grounded_injection_pilot import OUTPUT
from run_segmentation_intervention import ROOT, sha, text_sha, write_json

MATCHER_PATH = ROOT / "third_party/SWAN/amr-metric-suite/py3-Smatch-and-S2match/smatch"


def triple_bound(left, right):
    total = sum(left) + sum(right)
    return 2 * sum(min(a, b) for a, b in zip(left, right)) / total if total else 0.


def diagnose(folder):
    sys.path.insert(0, str(MATCHER_PATH))
    from amr_py3 import AMR

    inputs, cache = {}, {}
    runtime = json.loads((folder / "runtime.json").read_text())
    from run_confirmation_experiment import digest
    for path in (folder / "sentence_cache" / digest(runtime) / "score").glob("*.json"):
        saved = json.loads(path.read_text())
        key = saved["key"]
        if "injection_amr" not in key:
            continue
        if saved["value_sha256"] != digest(saved["value"]):
            raise ValueError("Cached match checksum mismatch")
        inputs[str(path.relative_to(folder))] = sha(path)
        cache[text_sha(key["injection_amr"]), text_sha(key["target"])] = saved
    rows = []
    parsed = {}

    def counts(text):
        if text not in parsed:
            messages = io.StringIO()
            try:
                with contextlib.redirect_stderr(messages):
                    graph = AMR.parse_AMR_line(text, False)
                if graph is None:
                    raise ValueError("Matcher graph parser returned None")
                parsed[text] = {"counts": [len(v) for v in graph.get_triples()],
                                "parser_messages": messages.getvalue(), "error": None}
            except Exception as exc:
                parsed[text] = {"counts": None, "parser_messages": messages.getvalue(), "error": type(exc).__name__}
        return parsed[text]

    bank_path = ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"
    bank_counts = [counts(r["normalized_template_amr"])["counts"] for r in json.loads(bank_path.read_text())]
    if len(bank_counts) != 50 or any(c is None for c in bank_counts):
        raise ValueError("Expected 50 countable bank templates")
    for path in sorted((folder / "sentences").glob("*-swan_grounded-*.json")):
        result = json.loads(path.read_text())
        inputs[str(path.relative_to(folder))] = sha(path)
        for attempt, event in enumerate(result["match_events"]):
            saved = cache[event["amr_sha256"], event["target_sha256"]]
            if saved["value"] != [event["matched"], event["score"]]:
                raise ValueError("Match event not equal to cached observation")
            left, right = counts(saved["key"]["injection_amr"]), counts(saved["key"]["target"])
            bound = triple_bound(left["counts"], right["counts"]) if left["counts"] and right["counts"] else None
            bank_bound = max(triple_bound(left["counts"], c) for c in bank_counts) if left["counts"] else None
            rows.append({
                "prompt_id": result["prompt_id"], "slot": result["slot"], "attempt": attempt,
                "candidate_triples": left["counts"], "target_triples": right["counts"],
                "count_upper_bound": bound, "stored_score": event["score"],
                "maximum_bank_count_upper_bound": bank_bound,
                "bound_below_threshold": bound < .75 if bound is not None else None,
                "score_exceeds_bound": event["score"] > bound + 1e-6 if bound is not None else None,
                "parse_errors": [v["error"] for v in (left, right) if v["error"]],
                "parser_messages": [v["parser_messages"] for v in (left, right) if v["parser_messages"]],
            })
    scored = [r for r in rows if r["count_upper_bound"] is not None]
    summary = {
        "attempts": len(rows), "countable_attempts": len(scored),
        "upper_bound_below_acceptance": sum(r["bound_below_threshold"] for r in scored),
        "even_best_bank_bound_below_acceptance": sum(r["maximum_bank_count_upper_bound"] < .75 for r in scored),
        "observed_bound_violations": sum(r["score_exceeds_bound"] for r in scored),
        "parser_error_attempts": sum(bool(r["parse_errors"]) for r in rows),
        "parser_message_attempts": sum(bool(r["parser_messages"]) for r in rows),
        "candidate_triple_mean": statistics.mean(sum(r["candidate_triples"]) for r in scored) if scored else None,
        "target_triple_mean": statistics.mean(sum(r["target_triples"]) for r in scored) if scored else None,
    }
    payload = {
        "status": "completed_exploratory_shape_diagnostic", "summary": summary, "rows": rows,
        "definition": "2*sum(min(candidate_count,type_template_count) per instance/attribute/relation type)/(candidate_total+template_total)",
        "assumption": "One-to-one node mapping with each compatible triple contributing at most one; bound is necessary, not sufficient",
        "checks": "Stored scores compared to the bound; violations would invalidate using it as an implementation bound",
        "new_api_calls": 0, "new_amr_model_calls": 0, "new_s2match_optimization_calls": 0,
        "input_sha256": inputs, "script_sha256": sha(Path(__file__)), "matcher_parser_sha256": sha(MATCHER_PATH / "amr_py3.py"),
        "bank_sha256": sha(bank_path),
        "limits": "Exploratory within the inspected pilot; repeated attempts/cached graphs not independent; no claim that larger templates solve quality or watermark robustness",
    }
    write_json(folder / "shape_diagnostic.json", payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    diagnose(OUTPUT)
