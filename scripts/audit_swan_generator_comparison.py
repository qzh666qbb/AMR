"""Replay the complete generator comparison using only saved requests and caches."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from audit_confirmation_experiment import verify_score
from prepare_confirmation_experiment import normalized_hash
from run_confirmation_experiment import CallJournal, digest, request_body
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_generator_comparison import (
    ARMS, BASELINE_REVIEW, CAP, OUTPUT, SLOTS, check, context_for,
    decode_baseline_review, execute_slot, load_upstream, source_gates,
)


def audit(folder):
    import nltk
    from injection import injection_amr_utils as local
    from utils.amr_utils import load_amr_bank

    manifest = check(folder)
    evidence = {}
    def load(path):
        evidence[str(path.relative_to(folder))] = sha(path)
        return json.loads(path.read_text())
    load(folder / "manifest.json")
    runtime = load(folder / "runtime.json")
    summary = load(folder / "summary.json")
    if runtime["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Runtime manifest mismatch")
    calls = [load(p) for p in sorted((folder / "calls").glob("*.json"))]
    if len(calls) > CAP:
        raise ValueError("API budget exceeded")
    for c in calls:
        if (c["status"] != "complete" or c["request_sha256"] != digest(c["request"])
                or c["response"]["model"] != runtime["model_returned"]):
            raise ValueError("Incomplete call or API identity mismatch")

    class ReplayJournal(CallJournal):
        def request(self, scope, body):
            path = folder / "calls" / f"{scope}-{self.counters.get(scope, 0):03d}.json"
            if not path.exists():
                raise ValueError("Replay would issue a new request")
            return super().request(scope, body)

    class ReplayDetector:
        bank = load_amr_bank(str(ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"))
        parser = SimpleNamespace(tokenizer=lambda text: {"input_ids": [0]}, max_sent_len=1024)

        def memo(self, kind, key, compute):
            path = folder / "sentence_cache" / digest(runtime) / kind / f"{digest(key)}.json"
            if not path.exists():
                raise ValueError("Replay would require new parsing or matching")
            row = load(path)
            if row["key"] != key or row["value_sha256"] != digest(row["value"]):
                raise ValueError("Cache checksum mismatch")
            return row["value"]

    api = ReplayJournal(folder, "", 0)
    api.model_returned = runtime["model_returned"]
    api.request("probe", request_body("Reply exactly OK.", 0, 16))
    detector = ReplayDetector()
    upstream = load_upstream()
    modules = {"upstream": upstream, "local_adapted": local, "unwatermarked": upstream}
    previous = {arm: [] for arm in ARMS}
    prompt = manifest["prompts"][0]
    for job in manifest["schedule"]:
        arm, slot = job["arm"], job["slot"]
        saved = load(folder / "sentences" / arm / f"{slot:03d}.json")
        if slot != len(previous[arm]) or saved["runtime_sha256"] != digest(runtime):
            raise ValueError("Slot order or runtime mismatch")
        actual = execute_slot(arm, slot, context_for(prompt, previous[arm]), prompt["prompt_id"], modules[arm], detector, api)
        expected = {k: v for k, v in saved.items() if k not in ("runtime_sha256", "elapsed_seconds")}
        if actual != expected:
            raise ValueError("Full injection trace replay diverged")
        previous[arm].append(saved)
    if len(list((folder / "sentences").rglob("*.json"))) != len(manifest["schedule"]):
        raise ValueError("Unexpected sentence checkpoint")
    for arm in ARMS:
        sentences = previous[arm]
        if len(sentences) != SLOTS:
            raise ValueError("Incomplete arm")
        source = " ".join(s["text"] for s in sentences).strip()
        response = api.request(f"review-{arm}", request_body(BASELINE_REVIEW + source, 0, 700))
        if response["finish_reason"] != "stop":
            raise ValueError("Truncated quality review")
        review = decode_baseline_review(response["content"])
        gates = source_gates(review, sentences)
        quality = load(folder / "quality" / f"{arm}.json")
        if quality != {"text_sha256": text_sha(source), "runtime_sha256": digest(runtime), "review": review, "gates": gates}:
            raise ValueError("Frozen source judgment or gate mismatch")
        row = load(folder / "evaluated" / f"{arm}.json")
        if (row["sentences"] != sentences or row["text"] != source or row["text_sha256"] != text_sha(source)
                or row["quality_sha256"] != digest(quality) or row["quality_review"] != review
                or row["gates"] != gates or row["words"] != len(source.split())
                or row["runtime_sha256"] != digest(runtime)):
            raise ValueError("Evaluated source identity mismatch")
        scored = row["detection"]
        verify_score(scored)
        details = scored["details"] if scored else []
        if nltk.sent_tokenize(source) != [d["text"] for d in details]:
            raise ValueError("Source segmentation mismatch")
        for detail in details:
            parsed = detector.memo("parse", {"text": detail["text"]}, None)
            scores = detector.memo("score", {"amr": detail["amr"]}, None)
            if parsed != [{k: detail[k] for k in ("text", "raw_amr", "amr")}] or scores != detail["bank_scores"]:
                raise ValueError("Detector differs from frozen cache")
        unique = len({normalized_hash(d["text"]) for d in details}) / max(1, len(details))
        if unique != row["unique_sentence_fraction"]:
            raise ValueError("Uniqueness mismatch")
        expected = {
            "status": "complete", "completed_slots": SLOTS, "words": len(source.split()),
            "quality_review": review, "gates": gates, "unique_sentence_fraction": unique,
            "accepted_slots": sum(s["accepted"] is True for s in sentences) if arm != "unwatermarked" else None,
            "generation_calls": sum(s["attempts"] for s in sentences),
            "finish_reasons": [s["finish_reason"] for s in sentences],
            "detection": {k: scored[k] for k in ("n", "green", "z")} if scored else None,
        }
        if summary["arms"][arm] != expected:
            raise ValueError("Summary arm mismatch")
    if sum(api.counters.values()) != len(calls) or summary["api_attempts"] != len(calls):
        raise ValueError("Unaccounted request")
    for key in ("prompt_tokens", "completion_tokens"):
        if sum(c["response"].get("usage", {}).get(key, 0) for c in calls) != summary["api_usage"][key]:
            raise ValueError("Token accounting mismatch")
    if summary["status"] != "complete_development_comparison":
        raise ValueError("Completion state mismatch")
    result = {
        "status": "passed", "sentence_slots_replayed": len(manifest["schedule"]),
        "requests_replayed": sum(api.counters.values()), "source_judgments_checked": 3,
        "detection_results_checked": 3, "mismatches": 0, "new_api_calls": api.sent,
        "new_parser_calls": 0, "new_s2match_calls": 0,
        "input_sha256": evidence, "audit_script_sha256": sha(Path(__file__)),
        "limitation": "Stored-observation trace replay, not independent stochastic repeatability.",
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
