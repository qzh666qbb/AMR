"""Checkpointed SWAN injection and fixed-budget three-arm confirmation runner.

Credentials are read only from the environment or non-echoing piped stdin.
Each paid call is journaled before transmission; uncertain calls are never
silently retried. A partial batch is an integration pilot, not a final result.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

from prepare_confirmation_experiment import ARMS, check, normalized_hash
from run_segmentation_intervention import ROOT, amr_body, sha, text_sha, write_json

MODEL = "deepseek-v4-flash"
PRE_REVIEW_LABEL_FIX = "d26e8243049eb9d9f3165b15a62f4897ec332ab9f0d05bc995e98421b8f09c99"
URL = "https://api.deepseek.com/chat/completions"
REVIEW = (
    "Evaluate four rewrites against SOURCE. For each candidate score semantic preservation 1-5 and mark "
    "major_error true only for changed entities, numbers, polarity, modality, attribution, events, or relations. "
    "Return JSON only as an array of four objects with keys candidate, score, major_error, reason.\n\n"
)


def digest(value):
    return text_sha(json.dumps(value, sort_keys=True))


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Frozen artifact differs: {path.name}")
    else:
        write_json(path, payload)
    return payload


class CallJournal:
    def __init__(self, folder, key, max_calls):
        self.folder, self.key, self.max_calls = folder, key, max_calls
        self.sent, self.counters, self.model_returned = 0, {}, None

    def request(self, scope, body):
        import requests

        ordinal = self.counters.get(scope, 0)
        self.counters[scope] = ordinal + 1
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        request_hash = digest(body)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_sha256"] != request_hash:
                raise ValueError("Replay request changed; inspect incomplete scope before resuming")
            if saved["status"] != "complete":
                raise RuntimeError("Uncertain/failed API call journal exists; refusing automatic retry")
            response = saved["response"]
        else:
            if self.sent >= self.max_calls:
                raise RuntimeError("Run API call cap reached; completed checkpoints preserved")
            saved = {"scope": scope, "ordinal": ordinal, "request_sha256": request_hash,
                     "request": body, "status": "in_flight", "transport_attempts": 1,
                     "started_at": datetime.now(timezone.utc).isoformat()}
            write_json(path, saved)
            self.sent += 1
            start = time.monotonic()
            try:
                result = requests.post(URL, headers={"Authorization": "Bearer " + self.key},
                                       json=body, timeout=(20, 240))
                saved["http_status"] = result.status_code
                if result.status_code != 200:
                    raise RuntimeError(f"HTTP {result.status_code}")
                payload = result.json()
                choice = payload["choices"][0]
                response = {"id": payload["id"], "model": payload["model"],
                            "usage": payload.get("usage", {}), "content": choice["message"]["content"],
                            "finish_reason": choice["finish_reason"]}
                if not isinstance(response["content"], str) or not response["content"].strip():
                    raise ValueError("Empty API content")
                saved.update(status="complete", response=response)
            except Exception as exc:
                saved.update(status="failed_or_uncertain", error_type=type(exc).__name__)
                raise RuntimeError(f"API call stopped ({type(exc).__name__}); inspect journal {path.name}") from None
            finally:
                saved["elapsed_seconds"] = time.monotonic() - start
                write_json(path, saved)
        if self.model_returned is not None and response["model"] != self.model_returned:
            raise ValueError("Returned model changed within frozen experiment")
        return response


def request_body(prompt, temperature, max_tokens, system=None, top_p=None):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    if top_p is not None:
        body["top_p"] = top_p
    return body


def decode_reviews(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    values = json.loads(text)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("Expected four semantic reviews")
    result = {}
    for row in values:
        k = row["candidate"]
        if isinstance(k, str) and re.fullmatch(r"(?:CANDIDATE )?[0-3]", k):
            k = int(k[-1])
        if type(k) is not int or k not in range(4) or k in result:
            raise ValueError("Invalid or duplicate review identity")
        if type(row["score"]) is not int or row["score"] not in range(1, 6) or type(row["major_error"]) is not bool:
            raise ValueError("Invalid semantic review fields")
        result[k] = {**row, "candidate": k}
    return result


def accept_review_label_fix(folder, config):
    """Only admit the audited label parser change with unchanged old judgments."""
    current = folder / "runtime.json"
    if not current.exists():
        return config
    old = json.loads(current.read_text())
    key = str(Path(__file__).relative_to(ROOT))
    if old["dependencies"].get(key) != PRE_REVIEW_LABEL_FIX:
        return config
    comparable = {**config, "dependencies": {**config["dependencies"], key: PRE_REVIEW_LABEL_FIX}}
    if comparable != old:
        return config
    checked = []
    for path in sorted((folder / "quality").glob("*.json")):
        quality = json.loads(path.read_text())
        row = quality["rows"][0]
        response_path = folder / "calls" / f'review-{row["prompt_id"]}-{row["arm"]}-000.json'
        response = json.loads(response_path.read_text())
        reviews = decode_reviews(response["response"]["content"])
        if any(reviews[r["candidate"]] != r["review"] for r in quality["rows"]):
            raise ValueError("Review label normalization changes existing judgment")
        checked.append(str(path.relative_to(folder)))
    amendment = {
        "original_runtime_sha256": digest(old), "original_runner_sha256": PRE_REVIEW_LABEL_FIX,
        "runner_sha256": config["dependencies"][key],
        "change": "Accept only CANDIDATE 0..3 or digit-string identities; keep strict scores, booleans and uniqueness",
        "existing_quality_checkpoints_verified": checked,
    }
    destination = folder / "implementation_amendments" / "review_label_parser.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if any(saved[k] != amendment[k] for k in amendment if k != "existing_quality_checkpoints_verified"):
            raise ValueError("Implementation amendment mismatch")
    else:
        freeze(destination, amendment)
    return old


class Detector:
    def __init__(self):
        import amrlib
        import torch
        sys.path.insert(0, str(ROOT / "third_party/SWAN"))
        from utils import amr_utils
        torch.set_num_threads(4)
        self.utils = amr_utils
        self.parser = amrlib.load_stog_model(
            model_dir=str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
            tok_name_or_path=str(ROOT / "models/bart-large-tokenizer"), device="cpu", batch_size=1)
        amr_utils.stog = self.parser
        self.bank = amr_utils.load_amr_bank(str(ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json"))
        if len(self.bank) != 50:
            raise ValueError("Expected 50 templates")

    def parse(self, text):
        import nltk
        import penman
        details = []
        for sentence in nltk.sent_tokenize(text):
            if len(self.parser.tokenizer(sentence)["input_ids"]) > self.parser.max_sent_len:
                raise ValueError("Refusing truncated AMR input")
            raw = self.parser.parse_sents([sentence])[0]
            if not raw:
                raise ValueError("AMR parser returned empty graph")
            graph = amr_body(raw)
            penman.decode(graph)
            graph = self.utils.normalize_amr_variables(self.utils.generate_template_amr(
                self.utils.normalize_amr_variables(graph)))
            penman.decode(graph)
            details.append({"text": sentence, "raw_amr": raw, "amr": graph})
        if not details:
            raise ValueError("Empty paragraph")
        return details

    def score(self, details):
        matched = []
        for item in details:
            scores = [float(self.utils.compute_s2match_score(item["amr"], t)) for t in self.bank]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
                raise ValueError("Invalid S2Match output")
            matched.append({**item, "bank_scores": scores, "score": max(scores)})
        n = len(matched)
        g = sum(item["score"] >= .65 for item in matched)
        return {"n": n, "green": g, "z": (g - .25 * n) / math.sqrt(n * .25 * .75), "details": matched}


def runtime(folder, api, amend_unstarted=False):
    check(folder)
    probe = api.request("probe", request_body("Reply exactly OK.", 0, 16))
    if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
        raise ValueError("API smoke response unexpected")
    api.model_returned = probe["model"]
    paths = [
        Path(__file__), folder / "preparation.json",
        ROOT / "scripts/evaluate_adaptive_retry.py",
        ROOT / "third_party/SWAN/injection/injection_amr_utils.py",
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json",
        ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json",
        ROOT / "vectors/glove.6B.100d.txt",
    ]
    paths += [p for d in ("model_parse_xfm_bart_large-v0_1_0", "bart-large-tokenizer")
              for p in (ROOT / "models" / d).glob("*") if p.is_file()]
    config = {
        "model_requested": MODEL, "model_returned": probe["model"],
        "endpoint": URL, "probe_id": probe["id"], "prompt_count": 30,
        "injection": {"sentences": 5, "max_tokens": 50, "temperature": .7, "top_p": .9,
                      "max_trials": 50, "max_templates": 10, "retries_per_template": 5,
                      "acceptance_s2match": .75, "seed": 20260912,
                      "logic": "Existing SWAN amr_instruction_completion with logged API adapter",
                      "rng": "Per-sentence seed; S2Match internal random restarts are nondeterministic"},
        "attack": {"k": 4, "temperature": .7, "max_tokens": 1400, "adaptive_rescue": False},
        "review": {"template": REVIEW, "temperature": 0, "max_tokens": 700, "expose_arm_or_z": False},
        "detector": {"device": "cpu", "beams": 4, "batch": 1, "lambda": .25,
                     "match_threshold": .65, "escape_threshold": 2.33},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "amrlib", "penman", "spacy", "nltk", "numpy", "requests")},
        "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "failure_policy": "Transport or parse errors stop execution; completed slots preserved, no silent replacement",
        "incomplete_injection_recovery": "Explicit fresh scope, counting all prior transmitted requests against 50-call sentence cap",
    }
    current = folder / "runtime.json"
    config = accept_review_label_fix(folder, config)
    if current.exists() and json.loads(current.read_text()) != config and amend_unstarted:
        if any(list((folder / sub).glob("*.json")) for sub in ("injection", "baselines", "attacks", "quality", "evaluated")):
            raise ValueError("Cannot amend runtime after sentence or candidate completion")
        old = json.loads(current.read_text())
        freeze(folder / "superseded_runtime" / f"{digest(old)}.json", old)
        write_json(current, config)
    freeze(folder / "runtime.json", config)
    print(json.dumps({"stage": "runtime_verified", "model_returned": probe["model"]}), flush=True)
    return config


def recovery_scope(folder, prefix, allow):
    prior = [json.loads(p.read_text()) for p in (folder / "calls").glob(f"{prefix}-*.json")]
    if not prior:
        return prefix, 0
    if not allow:
        raise RuntimeError("Incomplete injection scope exists; explicit recovery required")
    if len(prior) >= 50:
        raise RuntimeError("Incomplete sentence consumed all 50 calls; do not extend its budget")
    return f"{prefix}-recovery{len(prior)}", len(prior)


def generate_baseline(folder, prompt, detector, api, config, recover=False):
    from injection import injection_amr_utils as injection

    pid = prompt["prompt_id"]
    output = folder / "baselines" / f"{pid}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["runtime_sha256"] != digest(config) or saved["prompt_sha256"] != prompt["generation_prompt_sha256"]:
            raise ValueError("Baseline checkpoint mismatch")
        return saved
    context = prompt["generation_prompt"]
    sentences = []
    for slot in range(5):
        path = folder / "injection" / f"{pid}-{slot}.json"
        context_hash = text_sha(context)
        if path.exists():
            row = json.loads(path.read_text())
            if row["context_sha256"] != context_hash or row["runtime_sha256"] != digest(config):
                raise ValueError("Sentence injection checkpoint mismatch")
        else:
            scope, spent = recovery_scope(folder, f"inject-{pid}-{slot}", recover)
            if spent:
                freeze(folder / "recovery" / f"{scope}.json", {
                    "reason": "Incomplete sentence recovery after interrupted transport",
                    "prior_transmitted_calls": spent, "remaining_calls": 50 - spent,
                    "policy": "Retain all previous requests; restart same sentence with remaining budget only",
                })
            class LoggedManager:
                def __init__(self, **kwargs):
                    pass

                def generate(self, user_text, system_text, max_tokens, temperature, top_p):
                    return api.request(scope, request_body(user_text, temperature, max_tokens,
                                                           system_text, top_p))["content"].strip()

            original = injection.BedrockManager
            injection.BedrockManager = LoggedManager
            random.seed(int(text_sha(f"20260912:{pid}:{slot}")[:16], 16))
            started = time.monotonic()
            try:
                text, attempts, template, score, accepted = injection.amr_instruction_completion(
                    context=context, amr_bank=detector.bank, model_id=MODEL, region="us-east-1",
                    max_new_tokens=50, temperature=.7, top_p=.9, max_trials=50 - spent,
                    amr_examples_path=str(ROOT / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json"),
                )
            finally:
                injection.BedrockManager = original
            row = {"text": text, "attempts": attempts + spent, "recovery_spent_calls": spent, "chosen_template": template,
                   "s2match": score, "accepted": accepted, "context_sha256": context_hash,
                   "runtime_sha256": digest(config), "elapsed_seconds": time.monotonic() - started}
            freeze(path, row)
            print(json.dumps({"stage": "injection", "id": pid, "slot": slot,
                              "attempts": attempts, "accepted": accepted, "s2match": score}), flush=True)
        sentences.append(row)
        context = prompt["generation_prompt"] + " " + " ".join(s["text"] for s in sentences)
    text = " ".join(s["text"] for s in sentences).strip()
    old = json.loads((ROOT / "experiments/planE_e2_joint250/candidates.json").read_text())["rows"]
    if normalized_hash(text) in {normalized_hash(r["source"]) for r in old}:
        raise ValueError("New baseline overlaps historical source")
    detection = detector.score(detector.parse(text)) if text else None
    baseline = {"prompt_id": pid, "prompt_sha256": prompt["generation_prompt_sha256"],
                "runtime_sha256": digest(config), "text": text, "text_sha256": text_sha(text),
                "sentences": sentences, "all_injection_accepted": all(s["accepted"] for s in sentences),
                "detection": detection, "complete": True}
    freeze(output, baseline)
    return baseline


def run_attacks(folder, baseline, templates, api):
    pid, source = baseline["prompt_id"], baseline["text"]
    rows = []
    for k in range(4):
        for arm in ARMS:
            path = folder / "attacks" / f"{pid}-{arm}-{k}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["source_sha256"] != text_sha(source):
                    raise ValueError("Attack baseline mismatch")
            else:
                response = api.request(f"attack-{pid}-{arm}-{k}",
                                       request_body(templates[arm].format(source, candidate=k), .7, 1400)) if source else None
                row = {"prompt_id": pid, "arm": arm, "candidate": k, "source": source,
                       "source_sha256": text_sha(source),
                       "attack": response["content"].strip() if response else "",
                       "finish_reason": response["finish_reason"] if response else "empty_baseline"}
                freeze(path, row)
            rows.append(row)
    return rows


def review_and_detect(folder, baseline, rows, detector, api):
    from evaluate_adaptive_retry import gates

    pid = baseline["prompt_id"]
    evaluated = []
    for arm in ARMS:
        group = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["candidate"])
        path = folder / "quality" / f"{pid}-{arm}.json"
        group_hash = digest(group)
        if path.exists():
            quality = json.loads(path.read_text())
            if quality["group_sha256"] != group_hash:
                raise ValueError("Quality checkpoint mismatch")
        else:
            prompt = REVIEW + "SOURCE:\n" + baseline["text"] + "\n\n" + "\n\n".join(
                f'CANDIDATE {r["candidate"]}:\n{r["attack"]}' for r in group)
            response = api.request(f"review-{pid}-{arm}", request_body(prompt, 0, 700))
            if response["finish_reason"] != "stop":
                raise ValueError("Truncated semantic review; do not silently relabel")
            judgments = decode_reviews(response["content"])
            labeled = []
            for row in group:
                details = detector.parse(row["attack"]) if row["attack"] else []
                surface = gates(row, [d["amr"] for d in details])
                judgment = judgments[row["candidate"]]
                valid = (all(surface.values()) and judgment["score"] >= 4 and not judgment["major_error"]
                         and row["finish_reason"] == "stop")
                labeled.append({**row, "details": details, "surface": surface, "review": judgment,
                                "quality_valid": valid})
            quality = {"group_sha256": group_hash, "rows": labeled,
                       "no_feedback_candidate": next((r["candidate"] for r in labeled if r["quality_valid"]), None)}
            freeze(path, quality)
        evaluated.append((arm, quality))
    # All quality-only selections are persisted before reading any attack z.
    for arm, quality in evaluated:
        path = folder / "evaluated" / f"{pid}-{arm}.json"
        if path.exists():
            if json.loads(path.read_text())["quality_sha256"] != digest(quality):
                raise ValueError("Detection checkpoint mismatch")
            continue
        scored = [{**row, "detection": detector.score(row["details"]) if row["details"] else None}
                  for row in quality["rows"]]
        eligible = [r for r in scored if r["quality_valid"] and r["detection"]]
        best = min(eligible, key=lambda r: (r["detection"]["z"], r["candidate"])) if eligible else None
        selected = next((r for r in eligible if r["candidate"] == quality["no_feedback_candidate"]), None)
        freeze(path, {"quality_sha256": digest(quality), "prompt_id": pid, "arm": arm, "rows": scored,
                      "feedback_candidate": best["candidate"] if best else None,
                      "feedback_success": bool(best and best["detection"]["z"] <= 2.33),
                      "no_feedback_candidate": quality["no_feedback_candidate"],
                      "no_feedback_success": bool(selected and selected["detection"]["z"] <= 2.33)})
        print(json.dumps({"stage": "evaluated", "id": pid, "arm": arm, "quality_candidates": len(eligible),
                          "best_z": best["detection"]["z"] if best else None}), flush=True)


def summarize(folder):
    preparation = json.loads((folder / "preparation.json").read_text())
    planned = len(preparation["prompts"])
    baselines = [json.loads(p.read_text()) for p in (folder / "baselines").glob("*.json")]
    arms = {}
    for arm in ARMS:
        rows = [json.loads(p.read_text()) for p in (folder / "evaluated").glob(f"*-{arm}.json")]
        arms[arm] = {"completed_paragraphs": len(rows),
                     "feedback_successes": sum(r["feedback_success"] for r in rows),
                     "no_feedback_successes": sum(r["no_feedback_success"] for r in rows),
                     "valid_asr": sum(r["feedback_success"] for r in rows) / planned if len(rows) == planned else None}
    calls = [json.loads(p.read_text()) for p in (folder / "calls").glob("*.json")]
    costs = {}
    for row in calls:
        stage = row["scope"].split("-")[0]
        cost = costs.setdefault(stage, {"attempted_calls": 0, "complete_calls": 0,
                                       "prompt_tokens": 0, "completion_tokens": 0, "elapsed_seconds": 0.})
        cost["attempted_calls"] += 1
        cost["elapsed_seconds"] += row.get("elapsed_seconds", 0)
        if row["status"] == "complete":
            cost["complete_calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                cost[k] += row["response"]["usage"].get(k, 0)
    result = {"status": "complete" if all(r["completed_paragraphs"] == planned for r in arms.values())
              else "integration_pilot_incomplete", "planned_paragraphs": planned,
              "completed_baselines": len(baselines),
              "fully_accepted_injection_paragraphs": sum(r["all_injection_accepted"] for r in baselines),
              "initially_detected_baselines": sum(bool(r["detection"] and r["detection"]["z"] > 2.33) for r in baselines),
              "arms": arms, "api_costs": costs,
              "note": "No full-cohort ASR until all planned slots complete; no reselection of prompts."}
    write_json(folder / "execution_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarize"))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/confirmation_three_arm30")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--recover-incomplete-injection", action="store_true")
    parser.add_argument("--amend-unstarted-runtime", action="store_true")
    args = parser.parse_args()
    for k, v in {"NLTK_DATA": ".nltk_data", "HF_HOME": ".hf-cache",
                 "GLOVE_VECTORS_PATH": "vectors/glove.6B.100d.txt"}.items():
        os.environ.setdefault(k, str(ROOT / v))
    if args.stage == "summarize":
        print(json.dumps(summarize(args.output), indent=2))
        return
    if not 1 <= args.limit <= 30 or args.max_calls < 1:
        parser.error("limit must be 1..30 and max-calls positive")
    if args.key_stdin:
        if sys.stdin.isatty():
            raise RuntimeError("Use piped stdin, not an echoing terminal")
        print("Waiting for credential on stdin", flush=True)
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("API credential missing")
    api = CallJournal(args.output, key, args.max_calls)
    try:
        config = runtime(args.output, api, args.amend_unstarted_runtime)
        detector = Detector()
        preparation = json.loads((args.output / "preparation.json").read_text())
        for prompt in preparation["prompts"][:args.limit]:
            baseline = generate_baseline(args.output, prompt, detector, api, config, args.recover_incomplete_injection)
            rows = run_attacks(args.output, baseline, preparation["attack"]["templates"], api)
            review_and_detect(args.output, baseline, rows, detector, api)
    finally:
        print(json.dumps(summarize(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
