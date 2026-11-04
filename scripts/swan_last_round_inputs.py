"""One frozen historical source and one non-resampling generation batch."""
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
from scripts.prepare_swan_newline_negatives import load_key
from src.core.swan_blind_protocol import digest, _punkt

OUT = ROOT / "experiments/swan_multiview_development1/final_completion1"
PRIVATE = OUT / "private"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
SOURCE_SIZE = 156987808
URL = "https://api.deepseek.com/chat/completions"
SETTINGS = {"model": "deepseek-flash", "temperature": .9, "max_tokens": 1200,
            "thinking": {"type": "disabled"}}
SYSTEM = "Write natural English prose. Return only the one requested sentence."
FRAMEWORK = """Write the next sentence of a short, coherent English account.
Background article: {title}
Background paragraph: {background}
Previous sentences: {context}
Write one complete sentence with 5 to 12 whitespace-separated words.
Use specific concrete events and avoid repeating an earlier claim.
Do not use headings, quotations, numbering, abbreviated names, or literal AMR placeholders.
End with exactly one period and output only the sentence.
AMR is a graph representation of meaning. NE means a named entity, N a noun,
and X an unspecified concept. Substitute real words for these placeholders.
Examples of AMR structures and possible realizations:
{examples}
Additional semantic constraint:
{constraint}"""
WORD = re.compile(r"\S+")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(text):
    return hashlib.sha256(" ".join(unicodedata.normalize(
        "NFKC", text).casefold().split()).encode()).hexdigest()


def within_budget():
    c = json.loads((OUT / "CONTROL.json").read_text())
    return datetime.now(timezone.utc) < datetime.fromisoformat(c["deadline_at"]) \
        and not any((OUT / n).exists() for n in ("HALT.json", "DEADLINE_STOP.json"))


def prior_inventory():
    texts, articles, inventory = set(), set(), []
    names = ("documents", "source_manifest", "sealed_sources", "source_assign",
             "source_records", "source_cohort")
    paths = sorted(p for p in (ROOT / "experiments").rglob("*")
                   if p.suffix in (".json", ".jsonl")
                   and any(n in p.name.casefold() for n in names)
                   and OUT not in p.parents)
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    if key in ("text", "source_text", "original_text",
                               "reference_text", "generation_prompt"):
                        texts.add(identity(value))
                    elif key in ("article_title", "source_title", "article_id"):
                        articles.add(identity(value))
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    for p in paths:
        if p.stat().st_size > 200_000_000:
            raise RuntimeError("Prior source inventory too large")
        inventory.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p)})
        values = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] \
            if p.suffix == ".jsonl" else [json.loads(p.read_text())]
        for value in values:
            visit(value)
    return texts, articles, inventory


def reconstruct(text):
    # WikiText raw still contains documented token placeholders and spacing.
    text = text.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+([\])])", r"\1", text)
    return " ".join(text.split())


def source():
    if (OUT / "SOURCE_FREEZE.json").exists() or (OUT / "SOURCE_STOP.json").exists():
        raise RuntimeError("Source acquisition already decided")
    if not within_budget():
        raise RuntimeError("Deadline reached")
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    p = PRIVATE / "human_source.parquet"
    marker = PRIVATE / "SOURCE_DISPATCH.json"
    if marker.exists():
        raise RuntimeError("Uncertain source download cannot be repeated")
    save_json(marker, {"revision": REVISION, "file": SOURCE_FILE, "at": now()})
    url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/{SOURCE_FILE}"
    try:
        import requests
        import pyarrow.parquet as pq
        size = 0
        with requests.get(url, stream=True, timeout=(15, 45)) as response:
            response.raise_for_status()
            with p.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not within_budget():
                        raise RuntimeError("Source download deadline")
                    size += len(chunk)
                    if size > 250_000_000:
                        raise RuntimeError("Source download cap")
                    stream.write(chunk)
        if size != SOURCE_SIZE:
            raise RuntimeError("Source length differs from pinned metadata")
        prior_text, prior_articles, inventory = prior_inventory()
        rejected, candidates, seen = Counter(), [], set()
        article = None
        article_seen = set()
        scanned = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1024, columns=["text"]):
            for raw in batch.column(0).to_pylist():
                scanned += 1
                if scanned > 100_000:
                    break
                heading = re.fullmatch(r"\s*=\s+([^=]+?)\s+=\s*", raw or "")
                if heading:
                    article = reconstruct(heading.group(1))
                    continue
                if not article or identity(article) in article_seen:
                    continue
                text = reconstruct(raw or "")
                sentences = _punkt().tokenize(text)
                if not 96 <= len(WORD.findall(text)) <= 180 or not 3 <= len(sentences) <= 20 \
                        or any(t in text for t in ("<unk>", " = ", "@")):
                    rejected["length_or_markup"] += 1
                    continue
                aid, tid = identity(article), identity(text)
                if aid in prior_articles or tid in prior_text:
                    rejected["prior_article_or_text_overlap"] += 1
                    continue
                if tid in seen:
                    rejected["duplicate_paragraph"] += 1
                    continue
                article_seen.add(aid)
                seen.add(tid)
                candidates.append({"source_article_sha256": aid, "article_title": article,
                                   "text": text, "sentences": sentences,
                                   "normalized_text_sha256": tid, "row": scanned - 1})
            if scanned >= 100_000:
                break
        # Source-only deterministic selection; no detector output is consulted.
        candidates.sort(key=lambda d: digest({"seed": 2026102307,
                                             "article": d["source_article_sha256"]}))
        if len(candidates) < 300:
            raise RuntimeError("Fewer than 300 qualifying independent articles")
        selected = candidates[:300]
        human = []
        for i, d in enumerate(selected):
            human.append({**d, "document_id": f"human-{i:03d}",
                          "source": "human", "arm": "negative", "eligible": True,
                          "failure_reasons": [],
                          "split": "calibration" if i < 100 else "test"})
        save_json(PRIVATE / "HUMAN_DOCUMENTS.json", {"documents": human})
        save_json(OUT / "SOURCE_FREEZE.json", {
            "schema": "swan-last-round-source/v1", "created_at": now(),
            "repository": "Salesforce/wikitext", "revision": REVISION,
            "source_file": SOURCE_FILE, "source_sha256": sha(p),
            "download_bytes": size, "scanned_rows": scanned,
            "qualified_articles": len(candidates), "selected_articles": 300,
            "calibration_articles": 100, "test_articles": 200,
            "historical_origin": "WikiText Wikipedia extraction released in 2016",
            "license": ["cc-by-sa-3.0", "gfdl"],
            "source_url": url,
            "reconstruction": "replace @-@/@,@/@.@; punctuation/bracket spacing; whitespace collapse",
            "selection": "first qualifying paragraph per article then seeded-SHA sort",
            "eligible_words": [96, 180], "eligible_sentence_count": [3, 20],
            "human_manifest_sha256": sha(PRIVATE / "HUMAN_DOCUMENTS.json"),
            "identity_manifest": [{k: d[k] for k in (
                "document_id", "source_article_sha256", "normalized_text_sha256",
                "split", "row")} for d in human],
            "prior_inventory": inventory, "prior_article_overlap": 0,
            "prior_normalized_text_overlap": 0, "rejections": dict(rejected),
            "source_title_inventory_limitation": "only titles explicitly retained by prior manifests",
            "implementation_sha256": sha(__file__),
        })
        print(json.dumps({"source_frozen": True, "human_documents": len(human),
                          "qualified_articles": len(candidates)}), flush=True)
    except Exception as exc:
        save_json(OUT / "SOURCE_STOP.json", {
            "at": now(), "reason": type(exc).__name__, "permanent": True,
            "download_bytes": p.stat().st_size if p.exists() else 0,
            "no_alternative_source": True,
        })
        raise


def freeze_generation():
    if (OUT / "GENERATION_FREEZE.json").exists():
        raise RuntimeError("Cannot replace generation freeze")
    load_key()  # Verify configured access without exposing it.
    source_freeze = json.loads((OUT / "SOURCE_FREEZE.json").read_text())
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    if sha(PRIVATE / "HUMAN_DOCUMENTS.json") != source_freeze["human_manifest_sha256"]:
        raise RuntimeError("Human source changed")
    bank_path = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
    example_path = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/artifacts/normalized_amr_examples.json"
    bank = json.loads(bank_path.read_text())
    examples = json.loads(example_path.read_text())
    example_text = "\n".join(
        f"AMR: {e['normalized_template_amr']}\nSentence: {e['sentence']}"
        for e in examples)
    rng = random.Random(2026102308)
    template_ids = list(range(100)) * 8
    rng.shuffle(template_ids)
    docs = []
    for i in range(100):
        docs.append({"document_id": f"generated-cal-{i:03d}", "split": "calibration",
                     "arm": "negative", "source_index": i, "pair_id": None})
    for i in range(100):
        docs.append({"document_id": f"generated-neg-{i:03d}", "split": "test",
                     "arm": "negative", "source_index": 100 + i,
                     "pair_id": f"pair-{i:03d}" if i < 50 else None})
    for i in range(50):
        docs.append({"document_id": f"generated-pos-{i:03d}", "split": "test",
                     "arm": "positive", "source_index": 100 + i,
                     "pair_id": f"pair-{i:03d}",
                     "template_ids": template_ids[16*i:16*(i+1)]})
    for d in docs:
        h = human[d["source_index"]]
        d.update(source="generated", source_article_sha256=h["source_article_sha256"],
                 requested_sentences=16)
    save_json(PRIVATE / "GENERATION_ASSIGNMENTS.json", {"documents": docs})
    save_json(PRIVATE / "PROMPT_RESOURCES.json", {"framework": FRAMEWORK,
              "examples": example_text, "templates": [
                  b["normalized_template_amr"] for b in bank]})
    save_json(OUT / "GENERATION_FREEZE.json", {
        "schema": "swan-last-round-generation/v1", "created_at": now(),
        "settings": SETTINGS, "system_prompt": SYSTEM, "prompt_framework": FRAMEWORK,
        "source_freeze_sha256": sha(OUT / "SOURCE_FREEZE.json"),
        "assignments_sha256": sha(PRIVATE / "GENERATION_ASSIGNMENTS.json"),
        "prompt_resources_sha256": sha(PRIVATE / "PROMPT_RESOURCES.json"),
        "bank_sha256": sha(bank_path), "example_sha256": sha(example_path),
        "implementation_sha256": sha(__file__), "dispatch_cap": 4000,
        "documents": 250, "sentences_per_document": 16, "workers": 4,
        "same_framework_and_parameters": True, "shared_topic_pairs": 50,
        "label_policy": "generation_intent_including_injection_failures",
        "context_policy": "all previously returned sentences in this document",
        "negative_constraint": "No additional graph constraint; continue the account naturally.",
        "positive_constraint": "Realize the following AMR structure, using actual words:\n{template}",
        "automatic_retries": 0, "resampling": False, "score_screening": False,
        "dispatch_journal": "atomic-before-network; uncertain attempts never redispatched",
    })
    print(json.dumps({"generation_frozen": True, "documents": 250, "cap": 4000}))


def generate():
    if not (OUT / "CANDIDATE_SELECTION.json").exists():
        raise RuntimeError("Window candidate must be frozen before generation")
    freeze = json.loads((OUT / "GENERATION_FREEZE.json").read_text())
    if sha(__file__) != freeze["implementation_sha256"]:
        raise RuntimeError("Frozen generation implementation changed")
    if sha(PRIVATE / "GENERATION_ASSIGNMENTS.json") != freeze["assignments_sha256"] \
            or sha(PRIVATE / "PROMPT_RESOURCES.json") != freeze["prompt_resources_sha256"]:
        raise RuntimeError("Frozen prompts changed")
    calls = PRIVATE / "calls"
    calls.mkdir(mode=0o700, exist_ok=True)
    lock = (PRIVATE / "generation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resources = json.loads((PRIVATE / "PROMPT_RESOURCES.json").read_text())
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    docs = json.loads((PRIVATE / "GENERATION_ASSIGNMENTS.json").read_text())["documents"]
    key = load_key()
    mutex = threading.Lock()
    total_dispatches = len(list(calls.glob("*.dispatch.json")))
    completed_docs = 0

    def one_doc(d):
        nonlocal total_dispatches, completed_docs
        context, slots = [], []
        h = human[d["source_index"]]
        for index in range(16):
            callid = f"{d['document_id']}-{index:02d}"
            dispatch, receipt = calls / (callid + ".dispatch.json"), calls / (callid + ".receipt.json")
            template = d.get("template_ids", [None] * 16)[index]
            constraint = freeze["negative_constraint"] if template is None \
                else freeze["positive_constraint"].format(template=resources["templates"][template])
            prompt = resources["framework"].format(
                title=h["article_title"], background=h["text"], context=" ".join(context),
                examples=resources["examples"], constraint=constraint)
            payload = {**SETTINGS, "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            if receipt.exists():
                response = json.loads(receipt.read_text())
            elif dispatch.exists():
                response = {"text": "", "error": "uncertain_prior_dispatch"}
                save_json(receipt, response)
            elif not within_budget():
                response = {"text": "", "error": "deadline_not_dispatched"}
            else:
                with mutex:
                    if total_dispatches >= 4000:
                        raise RuntimeError("Dispatch cap reached")
                    fd = os.open(dispatch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w") as stream:
                        json.dump({"call_id": callid, "at": now(),
                                   "request_sha256": digest(payload)}, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    total_dispatches += 1
                request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer " + key})
                try:
                    with urllib.request.urlopen(request, timeout=55) as reply:
                        data = json.load(reply)
                    choice = data["choices"][0]
                    response = {"text": choice["message"]["content"].strip(),
                                "finish_reason": choice["finish_reason"],
                                "response_model": data.get("model"),
                                "usage": data.get("usage", {}), "at": now()}
                except Exception as exc:
                    response = {"text": "", "error": type(exc).__name__, "at": now()}
                save_json(receipt, response)
            text = response.get("text", "")
            slots.append({"slot_id": index, "text": text, "target_template_id": template,
                          "error": response.get("error"),
                          "finish_reason": response.get("finish_reason")})
            context.append(text)
        save_json(PRIVATE / "generated" / (d["document_id"] + ".json"),
                  {**d, "slots": slots})
        with mutex:
            completed_docs += 1
            save_json(OUT / "GENERATION_HEARTBEAT.json", {
                "at": now(), "documents_committed": completed_docs,
                "generation_dispatches": total_dispatches, "cap": 4000})

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one_doc, d) for d in docs]
        for future in as_completed(futures):
            future.result()
    audit_generation()


def audit_generation():
    assignments = json.loads((PRIVATE / "GENERATION_ASSIGNMENTS.json").read_text())["documents"]
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    prior_text, _, inventory = prior_inventory()
    seen, docs, counts = set(), [], Counter()
    for d in assignments:
        p = PRIVATE / "generated" / (d["document_id"] + ".json")
        data = json.loads(p.read_text()) if p.exists() else {
            **d, "slots": [{"text": "", "error": "document_not_completed"}] * 16}
        sentences = [s["text"] for s in data["slots"]]
        text = " ".join(sentences)
        reasons = []
        if any(s.get("error") for s in data["slots"]):
            reasons.append("generation_call_failure")
        if any(s.get("finish_reason") != "stop" for s in data["slots"]):
            reasons.append("generation_finish_reason")
        if any(not 5 <= len(WORD.findall(t)) <= 12 or len(re.findall(r"[.!?]", t)) != 1
               or not t.endswith(".") or "\n" in t for t in sentences):
            reasons.append("generation_surface_failure")
        tid = identity(text)
        if tid in seen or tid in prior_text:
            reasons.append("duplicate_or_prior_document")
        if len({identity(s) for s in sentences}) != 16:
            reasons.append("within_document_duplicate_sentence")
        seen.add(tid)
        counts.update(reasons)
        docs.append({**d, "text": text, "sentences": sentences, "eligible": not reasons,
                     "failure_reasons": reasons, "normalized_text_sha256": tid,
                     "target_template_ids": d.get("template_ids")})
    cal = {d["source_article_sha256"] for d in docs + human if d["split"] == "calibration"}
    test = {d["source_article_sha256"] for d in docs + human if d["split"] == "test"}
    if cal & test:
        raise RuntimeError("Source-family calibration/test overlap")
    save_json(PRIVATE / "DOCUMENTS.json", {"documents": human + docs})
    dispatches = len(list((PRIVATE / "calls").glob("*.dispatch.json")))
    save_json(OUT / "INPUT_AUDIT.json", {
        "created_at": now(), "source_family_overlap": len(cal & test),
        "generation_dispatches": dispatches, "dispatch_cap_pass": dispatches <= 4000,
        "all_frozen_documents_retained": len(docs) == 250 and len(human) == 300,
        "generated_eligible": sum(d["eligible"] for d in docs),
        "human_documents": len(human), "input_failure_counts": dict(counts),
        "source_family_scope": "human and generated sources shared within split, never across splits",
        "prior_inventory": inventory,
        "eligible_by_arm_split": dict(Counter(
            f"{d['arm']}:{d['split']}" for d in docs if d["eligible"])),
        "input_manifest_sha256": sha(PRIVATE / "DOCUMENTS.json"),
        "injection_failure_exclusion": False,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--phase", choices=("source", "freeze", "generate", "audit"), required=True)
    args = cli.parse_args()
    {"source": source, "freeze": freeze_generation, "generate": generate,
     "audit": audit_generation}[args.phase]()
