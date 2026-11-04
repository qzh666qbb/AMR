"""Offline source-screening, generation-trace, quality and detection replay."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from audit_confirmation_experiment import verify_score
from run_confirmation_experiment import CallJournal, digest, request_body
from run_grounded_injection_pilot import (
    ARMS, BASELINE_REVIEW, CAP, DOCUMENTS, OUTPUT, SLOTS, check,
    decode_baseline_review, decode_fact_review, execute_slot, fact_body, load_upstream, screen_source, source_metrics,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


def audit(folder):
    import nltk
    from utils.amr_utils import load_amr_bank
    manifest = check(folder)
    inputs = {}
    def load(path):
        inputs[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())
    load(folder / "manifest.json")
    runtime = load(folder / "runtime.json")
    summary = load(folder / "summary.json")
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Runtime/manifest mismatch")
    sources = {r["prompt_id"]: r for r in manifest["prompts"]}
    if len(sources) != DOCUMENTS or any(r["shard_row"] < 288 for r in sources.values()):
        raise ValueError("Cohort size or historical overlap")
    considered = list(sources.values()) + manifest["screened_rejections"]
    if sorted(r["shard_row"] for r in considered) != list(range(288, manifest["scanned_rows"])):
        raise ValueError("Source-screening inventory has gaps or duplicates")
    for record in considered:
        if (text_sha(record["source"]) != record["source_sha256"]
                or screen_source(record["source"], nltk.sent_tokenize) != record["screening"]):
            raise ValueError("Source-screening evidence changed")
    calls = [load(p) for p in sorted((folder / "calls").glob("*.json"))]
    if len(calls) > CAP or any(c["status"] != "complete" or c["request_sha256"] != digest(c["request"])
                              or c["response"]["model"] != runtime["model_returned"] for c in calls):
        raise ValueError("Request status, hash, model or budget mismatch")

    class Replay(CallJournal):
        def request(self, scope, body):
            path = folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
            if not path.exists():
                raise ValueError("Replay would require new API call")
            return super().request(scope, body)

    class ReplayDetector:
        bank = load_amr_bank(str(ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"))
        parser = SimpleNamespace(tokenizer=lambda text: {"input_ids": [0]}, max_sent_len=1024)

        def memo(self, kind, key, compute):
            path = folder / "sentence_cache" / digest(runtime) / kind / f"{digest(key)}.json"
            if not path.exists():
                raise ValueError("Replay would require new model inference")
            row = load(path)
            if row["key"] != key or digest(row["value"]) != row["value_sha256"]:
                raise ValueError("Cache identity mismatch")
            return row["value"]

    api = Replay(folder, "", 0)
    api.model_returned = runtime["model_returned"]
    api.request("probe", request_body("Reply exactly OK.", 0, 16))
    detector, module = ReplayDetector(), load_upstream()
    previous = {(pid, arm): [] for pid in sources for arm in ARMS}
    for job in manifest["schedule"]:
        pid, arm, slot = job["prompt_id"], job["arm"], job["slot"]
        group = previous[pid, arm]
        if slot != len(group):
            raise ValueError("Unexpected sequential generation order")
        saved = load(folder / "sentences" / f"{pid}-{arm}-{slot}.json")
        actual = execute_slot(job, sources[pid]["source"], group, module, detector, api)
        if (actual != {k: v for k, v in saved.items() if k not in ("runtime_sha256", "elapsed_seconds")}
                or saved["runtime_sha256"] != digest(runtime)):
            raise ValueError("Full generation trace differs")
        group.append(saved)
    expected_rows = []
    for (pid, arm), group in previous.items():
        if len(group) != SLOTS:
            raise ValueError("Incomplete arm")
        text = " ".join(s["text"] for s in group).strip()
        sr = api.request(f"style-{pid}-{arm}", request_body(BASELINE_REVIEW + text, 0, 700))
        fr = api.request(f"facts-{pid}-{arm}", fact_body(sources[pid]["source"], text))
        if sr["finish_reason"] != "stop" or fr["finish_reason"] != "stop":
            raise ValueError("Truncated review")
        style, facts = decode_baseline_review(sr["content"]), decode_fact_review(fr["content"])
        metrics = source_metrics(group, style, facts, nltk.sent_tokenize)
        quality = load(folder / "quality" / f"{pid}-{arm}.json")
        if quality != {"text_sha256": text_sha(text), "runtime_sha256": digest(runtime),
                       "style": style, "facts": facts, "metrics": metrics}:
            raise ValueError("Quality gate reconstruction mismatch")
        row = load(folder / "evaluated" / f"{pid}-{arm}.json")
        scored = row["detection"]
        if {k: v for k, v in row.items() if k != "detection"} != {
            "prompt_id": pid, "arm": arm, "runtime_sha256": digest(runtime),
            "source_sha256": sources[pid]["source_sha256"], "text": text, "text_sha256": text_sha(text),
            "sentences": group, "quality_sha256": digest(quality), "style_review": style,
            "fact_review": facts, "metrics": metrics,
        }:
            raise ValueError("Evaluated artifact mismatch")
        verify_score(scored)
        details = scored["details"] if scored else []
        if nltk.sent_tokenize(text) != [d["text"] for d in details]:
            raise ValueError("Detector segmentation mismatch")
        for detail in details:
            parsed = detector.memo("parse", {"text": detail["text"]}, None)
            scores = detector.memo("score", {"amr": detail["amr"]}, None)
            if parsed != [{k: detail[k] for k in ("text", "raw_amr", "amr")}] or scores != detail["bank_scores"]:
                raise ValueError("Detector cache differs")
        wm = arm == "swan_grounded"
        full = all(s["accepted"] is True for s in group) if wm else None
        detected = bool(scored and scored["z"] > 2.33)
        expected_rows.append({
            "prompt_id": pid, "arm": arm, **metrics, "style_review": style, "fact_review": facts,
            "accepted_slots": sum(s["accepted"] is True for s in group) if wm else None,
            "generation_calls": sum(s["attempts"] for s in group), "full_injection": full, "detected": detected,
            "joint_usable": bool(metrics["source_pass"] and full and detected) if wm else None,
            "detection": {k: scored[k] for k in ("n", "green", "z")} if scored else None,
        })
    if summary["records"] != expected_rows:
        raise ValueError("Feasibility table mismatch")
    joint = sum(r["joint_usable"] for r in expected_rows if r["arm"] == "swan_grounded")
    if (summary["joint_usable_watermarked"] != joint
            or summary["decision"] != ("prepare_fresh_validation" if joint >= 3 else "stop_this_configuration")
            or summary["status"] != "complete_grounded_injection_pilot"
            or summary["allocated_documents"] != DOCUMENTS):
        raise ValueError("Frozen decision or allocation mismatch")
    if any(len(list((folder / name).glob("*.json"))) != expected for name, expected in (
        ("sentences", 50), ("quality", 10), ("evaluated", 10))):
        raise ValueError("Artifact inventory mismatch")
    if (sum(api.counters.values()) != len(calls) or summary["api_attempts"] != len(calls)
            or summary["api_complete"] != len(calls)):
        raise ValueError("Unaccounted API calls")
    for key in ("prompt_tokens", "completion_tokens"):
        if summary["api_usage"][key] != sum(c["response"].get("usage", {}).get(key, 0) for c in calls):
            raise ValueError("Token usage mismatch")
    result = {
        "status": "passed", "source_documents": DOCUMENTS, "sentence_slots_replayed": 50,
        "outputs_checked": 10, "requests_replayed": sum(api.counters.values()), "mismatches": 0,
        "new_api_calls": api.sent, "new_parser_calls": 0, "new_s2match_calls": 0,
        "input_sha256": inputs, "audit_script_sha256": sha(Path(__file__)),
        "limitation": "Trace and gate consistency using saved stochastic scores; not independent model or human review",
    }
    write_json(folder / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for k, v in {"NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
                 "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt"}.items():
        os.environ.setdefault(k, str(ROOT / v))
    audit(args.output)


if __name__ == "__main__":
    main()
