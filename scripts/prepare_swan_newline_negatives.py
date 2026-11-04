"""Freeze and generate additional unwatermarked S6 newline documents."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from src.core.swan_blind_protocol import digest

OUT = ROOT / "experiments/swan_multiview_development1/newline_negative_expansion1"
SOURCE = ROOT / "experiments/swan_multiview_development1/threshold5_reassessment1/P0_S6"
SEED = 2026102306
SYSTEM = "Write natural English prose. Return only the requested JSON object."
SETTINGS = {
    "model": "deepseek-flash", "temperature": .9, "max_tokens": 1400,
    "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
}
DOMAINS = (
    "a neighborhood library", "a school science club", "a community garden",
    "a local history museum", "a small bakery", "a bicycle repair shop",
    "a coastal research station", "a neighborhood animal shelter",
    "a youth sports club", "a farmers market", "a regional railway station",
    "a volunteer fire brigade", "a pottery workshop", "a village music school",
    "a small software studio", "a mountain visitor center",
    "a community theater", "a university archives office",
    "a riverside conservation group", "a family furniture workshop",
)
ACTIVITIES = (
    "preparing a public demonstration", "organizing supplies for a new project",
    "repairing equipment after a routine inspection", "welcoming new volunteers",
    "adjusting plans after unexpected rain", "coordinating a weekend event",
    "reviewing records from the previous month", "teaching newcomers a practical skill",
    "moving materials to a different building", "responding to a delivery delay",
)
PROMPT = """Write an ordinary fictional account about {domain} {activity}.
The place is called {place}. Use {person_a} and {person_b} as participants where natural.
Return exactly {sentences} short, specific declarative English sentences as JSON:
{{"sentences":["First sentence.", "Next sentence."]}}
Aim for about {words} words per sentence, with 5 to 12 words in each.
Keep one consistent topic and describe several different concrete events.
Do not repeat claims or use quotations, headings, numbering, or abbreviated names.
Each sentence should end in one period. Return only this JSON object."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(text):
    return hashlib.sha256(" ".join(
        unicodedata.normalize("NFKC", text).casefold().split()).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if (OUT / "GENERATION_FREEZE.json").exists():
        raise RuntimeError("Existing generation freeze cannot be replaced")
    private = OUT / "private"
    private.mkdir(parents=True, mode=0o700)
    old = json.loads((SOURCE / "RESULTS.json").read_text())["methods"]["S6"]
    seed_docs = json.loads((ROOT /
        "experiments/swan_multiview_development1/dataset5_paired_neg/documents.json"
    ).read_text())["documents"]
    # Only numerical style metadata is used. No previous source, text or graph
    # enters generation. Assignments are fixed before any completion or score.
    profiles = [(len(d["slots"]), round(d["word_count"] / len(d["slots"])))
                for d in seed_docs]
    rng = random.Random(SEED)
    indices = list(range(200))
    rng.shuffle(indices)
    calibration = set(indices[:100])
    prompts = []
    for i in range(200):
        count, length = profiles[i % len(profiles)]
        values = {
            "domain": DOMAINS[i % 20], "activity": ACTIVITIES[i // 20],
            "place": ("Meadowhaven", "Riverton", "Ashford", "Brookfield", "Hillcrest",
                      "Elmwood", "Lakewood", "Fairview", "Oakridge", "Westhaven")[i // 20],
            "person_a": ("Alex", "Morgan", "Sam", "Taylor", "Jordan", "Casey", "Riley",
                         "Drew", "Cameron", "Robin", "Avery", "Quinn", "Blake", "Hayden",
                         "Reese", "Emery", "Skyler", "Dakota", "Parker", "Logan")[i % 20],
            "person_b": ("Jamie", "Chris", "Lee", "Pat", "Ash", "Devon", "Shawn",
                         "Kelly", "Jesse", "Terry")[i // 20],
            "sentences": count, "words": length,
        }
        prompts.append({
            "document_id": f"expanded-negative-{i:03d}",
            "split": "calibration" if i in calibration else "test",
            "domain_index": i % 20, "activity_index": i // 20,
            "requested_sentences": count, "requested_mean_words": length,
            "prompt": PROMPT.format(**values),
        })
    save_json(private / "PROMPTS.json", {"documents": prompts})
    save_json(OUT / "SOURCE_FREEZE.json", json.loads((SOURCE / "SOURCE_FREEZE.json").read_text()))
    source_rows = json.loads((SOURCE / "SOURCE_DOCUMENT_SCORES.json").read_text())
    source_rows["documents"] = [
        r for r in source_rows["documents"]
        if r["method"] == "S6" and r["split"] == "test"
    ]
    save_json(OUT / "HISTORICAL_CONTROL_SCORES.json", source_rows)
    generation = {
        "schema": "swan-newline-negative-generation/v1", "created_at": now(),
        "seed": SEED, "calibration_count": 100, "test_count": 100,
        "total_documents": 200, "dispatch_cap": 200, "workers": 4,
        "automatic_retries": 0, "resampling_or_score_screening": False,
        "api_url": "https://api.deepseek.com/chat/completions",
        "generation_settings": SETTINGS, "system_prompt": SYSTEM,
        "prompt_pattern": PROMPT, "domains": DOMAINS, "activities": ACTIVITIES,
        "prompts_sha256": sha(private / "PROMPTS.json"),
        "source_freeze_sha256": sha(OUT / "SOURCE_FREEZE.json"),
        "source_control_sha256": sha(OUT / "HISTORICAL_CONTROL_SCORES.json"),
        "implementation_sha256": sha(Path(__file__)),
        "fixed_unit_threshold": old["unit_threshold"],
        "fixed_document_threshold": old["condition_guard_5pct"]["document_threshold"],
        "recalibration_rule": "complete-calibration-negative-order-ceil((n+1)*0.95)",
        "calibration_fallback": "threshold_1_if_fewer_than_19_complete",
        "document_statistic": "fraction_of_all_blind_S6_windows_with_bankmax_ge_unit",
        "decision_rule": "statistic > document_threshold",
        "failures": "document_abstain_retained_in_frozen_denominator",
        "sampling_scope": "new-generated-unwatermarked-scenario-cohort",
        "independent_human_test": False,
        "generation_change": "same-provider-model-temperature; document-JSON prompt replaces slotwise continuation",
        "positive_tpr_scope": "reused_historical_heldout15_development_control",
        "threshold_uses_test": False,
    }
    save_json(OUT / "GENERATION_FREEZE.json", generation)
    print(json.dumps({"frozen": True, "calibration": 100, "test": 100,
                      "generation_sha256": sha(OUT / "GENERATION_FREEZE.json")}))


def load_key():
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    for path in (ROOT / ".secrets/legacy_api.txt", ROOT / "api.txt"):
        if path.exists():
            keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]+", path.read_text()))
            if len(keys) == 1:
                return keys.pop()
    raise RuntimeError("Configured generation credential unavailable")


def audit():
    generation = json.loads((OUT / "GENERATION_FREEZE.json").read_text())
    prompts = json.loads((OUT / "private/PROMPTS.json").read_text())["documents"]
    prior_docs, prior_sentences = set(), set()
    inventory = []
    # Inspect all existing document cohorts plus semantic-sequence originals.
    # Responses and credentials are never included in this audit inventory.
    paths = list((ROOT / "experiments").rglob("documents.json"))
    paths += list((ROOT / "experiments").glob("semantic_sequence*/private/*DOCUMENT*.json*"))
    for path in sorted(set(paths)):
        if OUT in path.parents:
            continue
        inventory.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        try:
            values = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                      if path.suffix == ".jsonl" else [json.loads(path.read_text())])
        except (ValueError, UnicodeError):
            raise RuntimeError("Unreadable prior document cohort: " + path.name)
        def visit(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("text", "source_text", "original_text") and isinstance(v, str):
                        prior_docs.add(identity(v))
                    if k in ("slots", "sentences", "carriers") and isinstance(v, list):
                        for s in v:
                            t = s.get("text") if isinstance(s, dict) else s
                            if isinstance(t, str):
                                prior_sentences.add(identity(t))
                    visit(v)
            elif isinstance(obj, list):
                for v in obj:
                    visit(v)
        for value in values:
            visit(value)
    seen, all_sentences, documents = {}, set(), []
    counts = Counter()
    for p in prompts:
        receipt = json.loads((OUT / "private/calls" / (p["document_id"] + ".response.json")).read_text())
        sentences, reasons = [], []
        content = receipt.get("content", "")
        try:
            decoded = json.loads(content)
            if set(decoded) != {"sentences"} or not isinstance(decoded["sentences"], list) \
                    or not all(isinstance(s, str) for s in decoded["sentences"]):
                raise ValueError("Invalid schema")
            sentences = decoded["sentences"]
        except (ValueError, TypeError):
            reasons.append("generation_schema_failure")
        text = "\n".join(sentences)
        if receipt.get("finish_reason") != "stop":
            reasons.append("generation_finish_reason")
        if len(sentences) != p["requested_sentences"]:
            reasons.append("generation_sentence_count")
        if any(not re.search(r"\.\s*$", s) or len(re.findall(r"[.!?]", s)) != 1
               or "\n" in s or not 5 <= len(s.split()) <= 12 for s in sentences):
            reasons.append("generation_surface_failure")
        normalized = identity(text)
        sentence_ids = [identity(s) for s in sentences]
        if normalized in seen or normalized in prior_docs:
            reasons.append("duplicate_or_prior_document")
        if len(set(sentence_ids)) != len(sentence_ids) \
                or any(s in all_sentences or s in prior_sentences for s in sentence_ids):
            reasons.append("duplicate_or_prior_sentence")
        seen[normalized] = p["document_id"]
        all_sentences.update(sentence_ids)
        counts.update(reasons)
        documents.append({
            **{k: v for k, v in p.items() if k != "prompt"},
            "text": text, "sentences": sentences, "word_count": len(text.split()),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "normalized_text_sha256": normalized, "eligible": not reasons,
            "failure_reasons": reasons, "label": "generated_unwatermarked",
        })
    save_json(OUT / "private/DOCUMENTS.json", {"documents": documents})
    test = {d["normalized_text_sha256"] for d in documents if d["split"] == "test"}
    cal = {d["normalized_text_sha256"] for d in documents if d["split"] == "calibration"}
    result = {
        "schema": "swan-newline-negative-input-audit/v1",
        "generation_freeze_sha256": sha(OUT / "GENERATION_FREEZE.json"),
        "input_manifest_sha256": sha(OUT / "private/DOCUMENTS.json"),
        "total_documents": len(documents), "calibration": 100, "test": 100,
        "eligible": sum(d["eligible"] for d in documents),
        "failures": dict(counts), "calibration_test_overlap": len(cal & test),
        "prior_inventory_sha256": digest(inventory), "prior_document_files": len(inventory),
        "independent_human_test": False, "semantic_quality": "not_independently_human_reviewed",
        "detector_calls_before_freeze": 0, "silent_resampling": False,
        "ready": len(documents) == generation["total_documents"] and not cal & test
                 and not counts["duplicate_or_prior_document"],
    }
    save_json(OUT / "PRIOR_DOCUMENT_INVENTORY.json", {"files": inventory})
    save_json(OUT / "INPUT_AUDIT.json", result)
    print(json.dumps(result))


def generate():
    private = OUT / "private"
    calls = private / "calls"
    calls.mkdir(mode=0o700, exist_ok=True)
    frozen = json.loads((OUT / "GENERATION_FREEZE.json").read_text())
    if frozen["implementation_sha256"] != sha(Path(__file__)) \
            or frozen["prompts_sha256"] != sha(private / "PROMPTS.json"):
        raise RuntimeError("Generation inputs changed")
    if (OUT / "GENERATION_HALT.json").exists():
        raise RuntimeError("Permanent generation halt")
    for intent in calls.glob("*.intent.json"):
        if not intent.with_name(intent.name.replace(".intent.", ".response.")).exists():
            save_json(OUT / "GENERATION_HALT.json",
                      {"reason": "unresolved_dispatch", "permanent": True, "at": now()})
            raise RuntimeError("Unresolved dispatch cannot be retried")
    key, lock, stopped = load_key(), threading.RLock(), threading.Event()
    prompts = json.loads((private / "PROMPTS.json").read_text())["documents"]
    def heartbeat(state):
        with lock:
            responses = list(calls.glob("*.response.json"))
            save_json(OUT / "generation_heartbeat.json", {
                "at": now(), "state": state, "completed": len(responses),
                "expected": 200, "freeze_sha256": sha(OUT / "GENERATION_FREEZE.json"),
            })
    def request(p):
        response_path = calls / (p["document_id"] + ".response.json")
        if response_path.exists():
            return
        with lock:
            if stopped.is_set():
                return
            body = {**SETTINGS, "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": p["prompt"]}]}
            save_json(calls / (p["document_id"] + ".intent.json"),
                      {"at": now(), "request_sha256": digest(body), "document_id": p["document_id"]})
        try:
            req = urllib.request.Request(
                frozen["api_url"], data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=60) as stream:
                payload = json.load(stream)
            choice = payload["choices"][0]
            receipt = {"at": now(), "content": choice["message"]["content"],
                       "finish_reason": choice["finish_reason"], "usage": payload.get("usage", {}),
                       "response_model": payload.get("model"), "response_id": payload.get("id")}
            with lock:
                save_json(response_path, receipt)
                heartbeat("running")
        except Exception as exc:
            with lock:
                stopped.set()
                save_json(OUT / "GENERATION_HALT.json", {
                    "reason": "dispatch_failure", "exception_type": type(exc).__name__,
                    "document_id": p["document_id"], "at": now(), "permanent": True})
                heartbeat("halted")
            raise RuntimeError("Generation dispatch failed; exception type " + type(exc).__name__) from None
    heartbeat("running")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request, p) for p in prompts]
        for future in as_completed(futures):
            future.result()
    heartbeat("complete")
    receipts = [json.loads(p.read_text()) for p in calls.glob("*.response.json")]
    save_json(OUT / "GENERATION_SUMMARY.json", {
        "complete": len(receipts) == 200, "calls": len(receipts), "at": now(),
        "response_models": sorted({str(r.get("response_model")) for r in receipts}),
        "usage": dict(Counter({k: sum(r.get("usage", {}).get(k, 0) for r in receipts)
                              for k in ("prompt_tokens", "completion_tokens", "total_tokens")})),
        "retries": 0, "resampling": 0,
    })
    audit()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("prepare", "generate", "audit"))
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "prepare":
        prepare()
    elif args.phase == "audit":
        audit()
    else:
        with (OUT / "private/generation.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generate()
