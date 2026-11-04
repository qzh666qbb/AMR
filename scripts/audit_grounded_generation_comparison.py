"""Offline replay of grounded generation requests, reviews, metrics and advancement."""

import argparse
import json
import os
from pathlib import Path

from run_confirmation_experiment import CallJournal, digest, request_body
from run_grounded_generation_comparison import (
    ARMS, BASELINE_REVIEW, CAP, OUTPUT, check, decode_baseline_review,
    decode_fact_review, evaluate, fact_body, generation_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


def audit(folder):
    import nltk
    manifest = check(folder)
    evidence = {}
    def load(path):
        evidence[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())
    load(folder / "manifest.json")
    runtime = load(folder / "runtime.json")
    summary = load(folder / "summary.json")
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Runtime identity mismatch")
    sources = {r["prompt_id"]: r for r in manifest["prompts"]}
    if len(sources) != 3 or any(r["shard_row"] <= 284 for r in sources.values()):
        raise ValueError("Unexpected cohort or known development overlap")
    calls = [load(p) for p in sorted((folder / "calls").glob("*.json"))]
    if len(calls) != CAP or any(c["status"] != "complete" or digest(c["request"]) != c["request_sha256"]
                                or c["response"]["model"] != runtime["model_returned"] for c in calls):
        raise ValueError("Call inventory or identity mismatch")

    class Replay(CallJournal):
        def request(self, scope, body):
            path = folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
            if not path.exists():
                raise ValueError("Replay would require new API call")
            return super().request(scope, body)

    api = Replay(folder, "", 0)
    api.model_returned = runtime["model_returned"]
    api.request("probe", request_body("Reply exactly OK.", 0, 16))
    responses = {(pid, arm): [] for pid in sources for arm in ARMS}
    for job in manifest["schedule"]:
        pid, arm, slot = job["prompt_id"], job["arm"], job["slot"]
        previous = responses[pid, arm]
        if slot != len(previous):
            raise ValueError("Unexpected sequential order")
        body = generation_body(sources[pid]["source"], arm, slot, [r["content"].strip() for r in previous])
        response = api.request(f"generate-{pid}-{arm}-{slot}", body)
        saved = load(folder / "generated" / f"{pid}-{arm}-{slot}.json")
        expected = {**job, "runtime_sha256": digest(runtime), "request_sha256": digest(body), "response": response}
        if saved != expected:
            raise ValueError("Generated artifact mismatch")
        previous.append(response)
    expected_records = []
    for pid, arm in responses:
        group = responses[pid, arm]
        text = " ".join(r["content"].strip() for r in group).strip()
        sr = api.request(f"style-{pid}-{arm}", request_body(BASELINE_REVIEW + text, 0, 700))
        fr = api.request(f"facts-{pid}-{arm}", fact_body(sources[pid]["source"], text))
        if sr["finish_reason"] != "stop" or fr["finish_reason"] != "stop":
            raise ValueError("Truncated judgment")
        style, facts = decode_baseline_review(sr["content"]), decode_fact_review(fr["content"])
        row = {
            "prompt_id": pid, "arm": arm, "text": text, "text_sha256": text_sha(text),
            "source_sha256": sources[pid]["source_sha256"], "runtime_sha256": digest(runtime),
            "response_ids": [r["id"] for r in group], "style_review": style, "fact_review": facts,
            **evaluate(text, group, arm, style, facts, nltk.sent_tokenize),
        }
        if load(folder / "evaluated" / f"{pid}-{arm}.json") != row:
            raise ValueError("Evaluation replay differs")
        expected_records.append({k: v for k, v in row.items() if k not in ("text", "runtime_sha256", "response_ids")})
    if (summary["records"] != expected_records or len(list((folder / "generated").glob("*.json"))) != 18
            or len(list((folder / "evaluated").glob("*.json"))) != 6):
        raise ValueError("Summary or allocated cell count mismatch")
    for arm in ARMS:
        selected = [r for r in expected_records if r["arm"] == arm]
        expected = {
            "allocated": 3, "complete": 3,
            "joint_pass": sum(r["source_feasibility_pass"] for r in selected),
            "gate_pass_counts": {key: sum(r["gates"][key] for r in selected) for key in (
                "style_pass", "factual_pass", "structure_pass", "total_length_pass", "sentence_length_pass")},
        }
        if summary["groups"][arm] != expected:
            raise ValueError("Group denominators or counts mismatch")
    ready = [arm for arm in ARMS if summary["groups"][arm]["joint_pass"] >= 2]
    if (summary["ready_for_injection_pilot_preparation"] != ready
            or summary["preferred_workflow_if_ready"] != (ready[0] if ready else None)):
        raise ValueError("Frozen advancement rule mismatch")
    if (summary["status"] != "complete_grounded_comparison" or summary["api_attempts"] != CAP
            or summary["api_complete"] != CAP or sum(api.counters.values()) != CAP):
        raise ValueError("Completion or cost mismatch")
    for key in ("prompt_tokens", "completion_tokens"):
        if summary["api_usage"][key] != sum(c["response"].get("usage", {}).get(key, 0) for c in calls):
            raise ValueError("Token accounting mismatch")
    result = {
        "status": "passed", "generation_calls_replayed": 18, "review_calls_replayed": 12,
        "source_documents": 3, "outputs_checked": 6, "requests_replayed": sum(api.counters.values()),
        "mismatches": 0, "new_api_calls": api.sent, "new_amr_or_detector_calls": 0,
        "input_sha256": evidence, "audit_script_sha256": sha(Path(__file__)),
        "limitation": "Recomputes record consistency and labels, not independent factual or stylistic review.",
    }
    write_json(folder / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    audit(args.output)


if __name__ == "__main__":
    main()
