"""Two-call developmental sanity check, separate from the frozen generator comparison."""

import argparse
import json
import os
from pathlib import Path
import time

from feedback2_runtime import Heartbeat, claim
from run_confirmation_experiment import CallJournal, digest, freeze, request_body
from run_segmentation_intervention import ROOT, sha, text_sha
from run_swan_sentence_growth import BASELINE_REVIEW, decode_baseline_review
from supervise_confirmation_experiment import read_credential

OUTPUT = ROOT / "experiments/swan_onepass_control1"
PARENT = ROOT / "experiments/swan_generator_comparison1"
INSTRUCTION = (
    "Write one coherent paragraph of 100 to 160 words continuing the opening below. "
    "This is a fictional news-style writing exercise, so you may invent plausible details. "
    "Each sentence must add distinct information; do not restate the opening or previous sentences. "
    "Do not include a title, list, commentary, or the opening itself. Output only the paragraph.\n\nOPENING:\n"
)


def gates(text, review, finish_reason):
    words = len(text.split())
    quality = all(review[k] >= 4 for k in ("fluency", "coherence", "nonredundancy")) and not review["severe_problem"]
    return {"words": words, "quality_pass": quality, "length_pass": 100 <= words <= 160,
            "format_pass": len([p for p in text.split("\n") if p.strip()]) == 1,
            "finish_pass": finish_reason == "stop"}


def run(folder, key):
    parent = json.loads((PARENT / "manifest.json").read_text())
    runtime = json.loads((PARENT / "runtime.json").read_text())
    prompt = parent["prompts"][0]
    paths = [Path(__file__), PARENT / "manifest.json", PARENT / "runtime.json",
             ROOT / "scripts/run_confirmation_experiment.py", ROOT / "scripts/run_swan_sentence_growth.py"]
    manifest = {
        "kind": "two_call_unwatermarked_sanity_probe", "prompt_id": prompt["prompt_id"],
        "reused_development_prompt": True, "api_cap": 2,
        "request": request_body(INSTRUCTION + prompt["generation_prompt"], .7, 320, top_p=.9),
        "review_prompt": BASELINE_REVIEW, "model_returned": runtime["model_returned"],
        "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "limits": [
            "No watermark generation, detection, ASR or FPR measurement",
            "Not an independent test: same already-inspected developmental opening",
            "One-shot output budget and explicit fictional/nonrepetition instruction differ from the sequential comparison",
            "May invent details; output is fictional, not a factual summary of the source article",
            "Cannot isolate generation granularity as a cause of any quality difference",
        ],
    }
    freeze(folder / "manifest.json", manifest)
    with Heartbeat(folder) as heartbeat:
        api = CallJournal(folder, key, 2)
        api.model_returned = manifest["model_returned"]
        started = time.monotonic()
        heartbeat.update(stage="generation")
        response = api.request("generation", manifest["request"])
        text = response["content"].strip()
        heartbeat.update(stage="quality")
        review_body = request_body(BASELINE_REVIEW + text, 0, 700)
        review_response = api.request("review", review_body)
        if review_response["finish_reason"] != "stop":
            raise ValueError("Truncated one-pass quality review")
        review = decode_baseline_review(review_response["content"])
        result = {
            "status": "complete_development_probe", "manifest_sha256": sha(folder / "manifest.json"),
            "text": text, "text_sha256": text_sha(text), "finish_reason": response["finish_reason"],
            "review": review, "gates": gates(text, review, response["finish_reason"]),
        }
        result["all_gates_pass"] = all(v for k, v in result["gates"].items() if k != "words")
        freeze(folder / "result.json", result)
        calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
        if len(calls) != 2 or any(c["status"] != "complete" for c in calls):
            raise ValueError("Unexpected probe call inventory")
        expected = {"generation": manifest["request"], "review": review_body}
        for call in calls:
            if call["request"] != expected[call["scope"]] or call["request_sha256"] != digest(call["request"]):
                raise ValueError("Probe request audit failed")
        audit = {
            "status": "passed", "calls_checked": 2, "mismatches": 0,
            "result_sha256": sha(folder / "result.json"), "manifest_sha256": sha(folder / "manifest.json"),
            "call_sha256": {p.name: sha(p) for p in (folder / "calls").glob("*.json")},
            "api_usage": {k: sum(c["response"].get("usage", {}).get(k, 0) for c in calls)
                          for k in ("prompt_tokens", "completion_tokens")},
            "note": "Journal and label consistency check, not independent quality review",
        }
        freeze(folder / "audit.json", audit)
        heartbeat.update(stage="complete", new_api_calls=api.sent, elapsed_work_seconds=time.monotonic() - started)
        print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    key = read_credential(args.key_file) if args.key_file else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("Expected single nonempty API credential")
    with claim(args.output / ".runner.lock"):
        run(args.output, key)


if __name__ == "__main__":
    main()
