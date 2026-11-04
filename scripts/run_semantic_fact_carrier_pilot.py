"""Frozen development pilot. At-most-once API calls; no AMR or blind decoder."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/core"))
from semantic_fact_carrier import (ALPHABET_VERSION, SYMBOLS, classify_facts,
                                   generation_messages, paraphrase_messages,
                                   plan, validate_evidence)

OUT = ROOT / "experiments/semantic_fact_carrier_pilot1"
PRIVATE = Path.home() / ".cache/amr/semantic_fact_carrier_pilot1"
API_URL = "https://api.deepseek.com/chat/completions"
PARAMETERS = {"model": "deepseek-flash", "temperature": 0.7, "top_p": 1,
              "max_tokens": 320, "n": 1, "stream": False,
              "presence_penalty": 0, "frequency_penalty": 0,
              "thinking": {"type": "disabled"}}
BUDGET = {"max_calls": 192, "concurrency": 3, "max_wall_seconds": 7200,
          "max_input_tokens_bound": 4096, "input_peak_usd_per_million": 0.30,
          "output_peak_usd_per_million": 1.20, "max_reserved_usd": 0.35}
GATES = {"writing_min": 0.90, "paraphrase_preserved_min": 0.90,
         "cluster_bootstrap_lower_min": 0.80, "bootstrap_replicates": 10000,
         "bootstrap_seed": 20261026, "unit": "24 paired scenes"}
FIELDS = ("sender", "recipient", "object", "transfer_status")
FACT_LABELS = ["readability", *FIELDS, "explicit_status", "evidence_json",
               "quality", "notes"]
PAIR_LABELS = ["facts_preserved", "explicit_status_preserved",
               "event_order_preserved", "evidence_json", "notes"]


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load(path):
    return json.loads(Path(path).read_text())


class TerminalHalt(RuntimeError):
    pass


class Exclusive:
    def __init__(self, private):
        self.path = private / "runner.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stream = self.path.open("a")
        fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *args):
        self.stream.close()


def read_key(access):
    config = load(access)
    keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]+",
                          Path(config["credential_file"]).read_text()))
    if len(keys) != 1:
        raise RuntimeError("Expected exactly one configured provider key")
    return keys.pop()


def prepare(out, private):
    if (out / "FREEZE.json").exists() or (private / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze is immutable; create a new version")
    rows = plan()
    assert len(rows) == 96 and len({r["item_id"] for r in rows}) == 96
    assert Counter(r["intended_symbol"] for r in rows) == dict.fromkeys("ABCD", 24)
    required = ["PROTOCOL.md", "PRICE_SOURCE.json", "TEST_RESULT.json"]
    if any(not (out / p).exists() for p in required):
        raise RuntimeError("Protocol, primary price source and test evidence required")
    tests = load(out / "TEST_RESULT.json")
    if tests.get("status") != "PASS" or tests["tests_run"] < 8:
        raise RuntimeError("Meaningful tests have not passed")
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    frozen_plan = [{"row": r, "original_messages": generation_messages(r)} for r in rows]
    atomic(private / "PLAN.json", frozen_plan)
    atomic(out / "ALPHABET.json", {"version": ALPHABET_VERSION, "symbols": SYMBOLS,
                                 "fields": FIELDS, "erasure_is_failure": True,
                                 "historical_template_labels_reused": False})
    atomic(out / "PLAN_METADATA.json", {
        "split": "development_only", "originals": 96, "paraphrases": 96,
        "scenes": 24, "symbols_per_scene": list(SYMBOLS),
        "counts": dict(Counter(r["intended_symbol"] for r in rows)),
        "planned_ids": [r["item_id"] for r in rows],
        "one_sample_per_request_no_resampling": True,
        "paraphrase_source": "literal nonempty original receipt with finish_reason=stop",
    })
    sources = ["scripts/run_semantic_fact_carrier_pilot.py",
               "src/core/semantic_fact_carrier.py",
               "scripts/test_semantic_fact_carrier_pilot.py"]
    for source in sources:
        if tests["code_sha256"].get(source) != sha(ROOT / source):
            raise RuntimeError("Source changed after tests")
    public = required + ["ALPHABET.json", "PLAN_METADATA.json"]
    freeze = {"created_at": now(), "version": out.name, "alphabet": ALPHABET_VERSION,
              "api_url": API_URL, "parameters": PARAMETERS, "budget": BUDGET,
              "gates": GATES, "private_plan_sha256": sha(private / "PLAN.json"),
              "code_sha256": {p: sha(ROOT / p) for p in sources},
              "public_sha256": {p: sha(out / p) for p in public},
              "authorization": "user 2026-10-26: 继续吧; new development carrier pilot",
              "forbidden_pre_human_gate": ["AMR", "embedding", "training", "blind_decode"],
              "historical_runs_resumed": False}
    atomic(private / "FREEZE.json", freeze)
    atomic(out / "FREEZE.json", freeze)
    return freeze


def verify(out, private):
    freeze = load(out / "FREEZE.json")
    if freeze != load(private / "FREEZE.json"):
        raise TerminalHalt("Public/private freeze mismatch")
    checks = {p: sha(ROOT / p) == s for p, s in freeze["code_sha256"].items()}
    checks.update({p: sha(out / p) == s for p, s in freeze["public_sha256"].items()})
    checks["private_plan"] = sha(private / "PLAN.json") == freeze["private_plan_sha256"]
    checks["runtime_contract"] = (
        freeze["parameters"] == PARAMETERS and freeze["budget"] == BUDGET
        and freeze["gates"] == GATES and freeze["api_url"] == API_URL)
    if not all(checks.values()):
        raise TerminalHalt("Frozen source or input drift")
    return checks


def precheck(out, private, access):
    verify(out, private)
    config = load(access)
    key = read_key(access)
    del key
    result = {"at": now(), "status": "GO_GENERATION_ONLY",
              "credential_configured": True, "provider_ping_calls": 0,
              "packages": {p: md.version(p) for p in ("requests", "supervisor")},
              "configured_python_exists": Path(config["python"]).is_file(),
              "cuda_required_for_this_stage": False, "cuda_verified": False,
              "input_bound": "UTF-8 JSON message bytes + 512 framing tokens",
              "max_192_call_peak_usd_bound": 192 * (4096 * .30 + 320 * 1.20) / 1e6,
              "human_gate": "NOT_EVALUATED"}
    if not result["configured_python_exists"]:
        raise RuntimeError("Configured Python missing")
    atomic(out / "READINESS.json", result)
    return result


class Provider:
    def __init__(self, out, private, key, transport=requests.post):
        self.out, self.private, self.key = out, private, key
        self.transport = transport
        self.lock = threading.Lock()
        self.stop = threading.Event()
        start = private / "START.json"
        if not start.exists():
            atomic(start, {"started_at": now(), "epoch": time.time()})
        self.start = load(start)
        self.ledger = {}
        self.replays = 0
        if (private / "HALT.json").exists():
            raise TerminalHalt("Permanent halt exists; never resume this manifest")
        for path in (private / "requests").glob("*/INTENT.json"):
            identity = path.parent.name
            intent = load(path)
            receipt_path = path.parent / "RECEIPT.json"
            if not receipt_path.exists():
                self.halt("uncertain_intent_without_receipt")
            receipt = load(receipt_path)
            if receipt["body_sha256"] != intent["body_sha256"]:
                self.halt("receipt_intent_mismatch")
            if receipt["kind"] == "transport_uncertain":
                self.halt("uncertain_transport_receipt")
            self.ledger[identity] = intent

    def halt(self, reason):
        self.stop.set()
        atomic(self.private / "HALT.json", {"at": now(), "reason": reason,
                                           "terminal": True, "retry_allowed": False})
        raise TerminalHalt(reason)

    def request(self, identity, messages):
        body = {**PARAMETERS, "messages": messages}
        body_hash = digest(body)
        folder = self.private / "requests" / identity
        with self.lock:
            if self.stop.is_set():
                raise TerminalHalt("Another worker halted")
            if identity in self.ledger:
                intent = self.ledger[identity]
                receipt = load(folder / "RECEIPT.json")
                if intent["body_sha256"] != body_hash or receipt["body_sha256"] != body_hash:
                    self.halt("replay_prompt_drift")
                self.replays += 1
                return receipt
            if time.time() - self.start["epoch"] >= BUDGET["max_wall_seconds"]:
                self.halt("wall_budget")
            input_bound = len(encode(messages)) + 512
            reservation = (input_bound * .30 + PARAMETERS["max_tokens"] * 1.20) / 1e6
            if input_bound > BUDGET["max_input_tokens_bound"]:
                self.halt("input_capacity")
            if (len(self.ledger) >= BUDGET["max_calls"] or
                    sum(x["reserved_usd"] for x in self.ledger.values()) + reservation
                    > BUDGET["max_reserved_usd"]):
                self.halt("call_or_cost_budget")
            intent = {"at": now(), "body_sha256": body_hash,
                      "input_token_bound": input_bound, "reserved_usd": reservation}
            # Literal body first; intent durably persisted immediately before dispatch.
            atomic(folder / "BODY.json", body)
            atomic(folder / "INTENT.json", intent)
            self.ledger[identity] = intent
        try:
            response = self.transport(API_URL, json=body, allow_redirects=False,
                                      timeout=(15, 90),
                                      headers={"Authorization": "Bearer " + self.key})
            # Preserve the raw receipt outside workspace; never print provider payload.
            raw = response.text
            try:
                payload = json.loads(raw)
            except (ValueError, TypeError):
                payload = None
            receipt = {"at": now(), "kind": "http_receipt",
                       "http_status": response.status_code, "body_sha256": body_hash,
                       "raw": raw, "payload": payload}
            atomic(folder / "RECEIPT.json", receipt)
            usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
            if (usage.get("prompt_tokens", 0) > intent["input_token_bound"]
                    or usage.get("completion_tokens", 0) > PARAMETERS["max_tokens"]):
                with self.lock:
                    self.halt("provider_usage_exceeded_reserved_bound")
            return receipt
        except TerminalHalt:
            raise
        except Exception as error:
            atomic(folder / "RECEIPT.json", {
                "at": now(), "kind": "transport_uncertain", "body_sha256": body_hash,
                "exception_type": type(error).__name__})
            with self.lock:
                self.halt("transport_outcome_uncertain")


def extract(receipt):
    if receipt.get("kind") != "http_receipt" or receipt.get("http_status") != 200:
        return {"status": "provider_failure", "text": "", "model": None, "usage": {}}
    payload = receipt.get("payload")
    try:
        choice = payload["choices"][0]
        text = choice["message"]["content"]
        finish = choice["finish_reason"]
        if not isinstance(text, str):
            raise ValueError
        return {"status": ("complete" if text.strip() and finish == "stop"
                           else "truncated" if finish == "length" else "empty_or_nonstop"),
                "text": text, "finish_reason": finish, "model": payload.get("model"),
                "usage": payload.get("usage", {})}
    except (KeyError, IndexError, TypeError, ValueError):
        return {"status": "malformed_receipt", "text": "", "model": None, "usage": {}}


def summarize(out, private, state):
    rows = load(private / "PLAN.json")
    counts, models, words, usages = Counter(), Counter(), {}, Counter()
    results = []
    text_hashes = {c: Counter() for c in ("original", "paraphrase")}
    unchanged = 0
    missing = {"status": "not_generated", "text": "", "model": None, "usage": {}}
    for entry in rows:
        identity = entry["row"]["item_id"]
        doc_path = private / "documents" / (identity + ".json")
        doc = load(doc_path) if doc_path.exists() else {}
        public = {"item_id": identity, "intended_symbol": entry["row"]["intended_symbol"]}
        for condition in ("original", "paraphrase"):
            data = doc.get(condition, missing)
            counts[f"{condition}:{data['status']}"] += 1
            public[condition] = data["status"]
            words.setdefault(condition, []).append(len(data["text"].split()))
            if data["text"]:
                text_hashes[condition][digest(data["text"])] += 1
        if (doc.get("original", missing)["text"]
                and doc.get("original", missing)["text"] == doc.get("paraphrase", missing)["text"]):
            unchanged += 1
        results.append(public)
    intents = sorted((private / "requests").glob("*/INTENT.json"))
    uncertain = 0
    for intent_path in intents:
        rp = intent_path.parent / "RECEIPT.json"
        if not rp.exists():
            uncertain += 1
            continue
        receipt = load(rp)
        if receipt["kind"] == "transport_uncertain":
            uncertain += 1
        data = extract(receipt)
        if data.get("model"):
            models[data["model"]] += 1
        usage = data.get("usage", {})
        for field in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
                      "prompt_cache_miss_tokens"):
            usages[field] += usage.get(field, 0)
    cost_bound = (usages["prompt_tokens"] * .30 + usages["completion_tokens"] * 1.20) / 1e6
    result = {"at": now(), "state": state, "planned_originals": 96, "planned_paraphrases": 96,
              "counts": dict(counts), "calls_dispatched": len(intents),
              "uncertain_dispatches": uncertain, "provider_models_reported": dict(models),
              "usage": dict(usages), "observed_tokens_peak_cost_upper_usd": cost_bound,
              "reserved_usd": sum(load(p)["reserved_usd"] for p in intents),
              "actual_bill_usd": None, "semantic_writing_success": None,
              "human_completed": 0, "human_gate": "NOT_EVALUATED",
              "parser_calls": 0, "embedding_calls": 0, "blind_decode_calls": 0,
              "unchanged_paraphrase_pairs": unchanged,
              "duplicate_nonempty_texts": {c: sum(n - 1 for n in v.values())
                                            for c, v in text_hashes.items()},
              "word_counts": {k: {"min": min(v), "max": max(v),
                                   "outside_soft_35_70": sum(not 35 <= n <= 70 for n in v)}
                              for k, v in words.items()},
              "results": results}
    atomic(out / "GENERATION_SUMMARY.json", result)
    atomic(out / "heartbeat.json", {k: v for k, v in result.items()
                                    if k not in ("results", "word_counts")})
    return result


def run(out, private, access):
    with Exclusive(private):
        verify(out, private)
        provider = Provider(out, private, read_key(access))
        entries = load(private / "PLAN.json")

        def item(entry):
            row = entry["row"]
            identity = row["item_id"]
            original = extract(provider.request(identity + "_original", entry["original_messages"]))
            if original["status"] == "complete":
                paraphrase = extract(provider.request(
                    identity + "_paraphrase", paraphrase_messages(original["text"])))
            else:
                paraphrase = {"status": "source_failure_not_called", "text": "",
                              "model": None, "usage": {}}
            atomic(private / "documents" / (identity + ".json"),
                   {"item_id": identity, "original": original, "paraphrase": paraphrase})
            return identity

        failure = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=BUDGET["concurrency"]) as pool:
            iterator = iter(entries)
            pending = {pool.submit(item, next(iterator)) for _ in range(BUDGET["concurrency"])}
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except TerminalHalt as error:
                        failure = type(error).__name__
                        provider.stop.set()
                    except Exception as error:
                        failure = type(error).__name__
                        try:
                            provider.halt("worker_exception")
                        except TerminalHalt:
                            pass
                    if not provider.stop.is_set():
                        entry = next(iterator, None)
                        if entry is not None:
                            pending.add(pool.submit(item, entry))
                summarize(out, private, "HALTED" if provider.stop.is_set() else "RUNNING")
        result = summarize(out, private, "HALTED" if failure else "GENERATION_FINISHED")
        if failure:
            raise TerminalHalt("Pilot halted; no retry or historical manifest rescue")
        return result


REVIEW_INSTRUCTIONS = """# 真人事实核对（开发小试）

本目录为一位评审者的独立材料；标签全空白。不用LLM代填，不互看答案。
先完成 facts.csv，再打开 pairs.csv。两个任务ID独立打乱，没有目标符号或计划答案。
公共角色、物品只用于指明待核对事实；不要依据它们补出文本未表达的内容。

facts.csv 共192行：直接读文本，填实际 sender、recipient、object、transfer_status。
readability=clear/unclear/contradictory/unavailable；不清楚的事实填unknown。
状态只能 completed/planned_not_occurred/other/unknown：
completed 要有明确已完成、收到物品的事实；
planned_not_occurred 要同时有明确计划和该转移尚未发生；
只凭过去时、将来时、愿望、可能、引语、承诺，不把状态补为确定事实。
explicit_status=yes/no/unknown；quality=1/2/3/4/5/unknown（1很差，5自然连贯）。
evidence_json 为四字段对象，每字段是字符证据数组：
{"sender":[{"start":0,"end":4,"quote":"Mina"}],"recipient":[...],"object":[...],"transfer_status":[...]}
位置从0开始，左闭右开；quote 必须逐字等于文本对应区间。状态证据可以有多个片段。
明确事实要有证据；未知字段填unknown、证据用[]。不要改文本和公共字段。
text_words.txt 提供索引。notes 为解释，不能留空；无文本保留该行并填unavailable。

pairs.csv 共96行：对照原句和复述，填 facts_preserved、
explicit_status_preserved、event_order_preserved=yes/no/unknown。
判断源文实际表达的事实是否保留，不判断是否符合计划；错误原句被忠实复述仍可yes。
evidence_json={"source":[证据片段],"paraphrase":[证据片段]}；每个yes都需两侧精确证据。
缺失文本不能填yes；notes 必填。背景事件与转移的先后也要核对。

最后填写reviewer.json：真实评审者代号、UTC完成时间、
human_attestation=true、independent_attestation=true。无需实名。
两位评审有分歧时由第三位真人裁决；unknown不自动转yes。
交回完整目录，原ID/文本/字段不可改。admin目录仅由实验管理员保管。
本材料含封存生成文本，仅用于本研究，不放入公开复现包。
"""


def write_csv(path, rows, labels):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + labels)
        writer.writeheader()
        writer.writerows({**r, **{k: "" for k in labels}} for r in rows)
    os.chmod(path, 0o600)


def review(out, private):
    verify(out, private)
    folder = out / "private_review"
    manifest_path = folder / "admin/MANIFEST.json"
    if manifest_path.exists():
        raise RuntimeError("Review packet already sealed")
    if load(out / "GENERATION_SUMMARY.json")["state"] != "GENERATION_FINISHED":
        raise RuntimeError("Finish or archive generation before review")
    rows = load(private / "PLAN.json")
    sources = {"facts": [], "pairs": []}
    document_hashes = {}
    for entry in rows:
        row = entry["row"]
        identity = row["item_id"]
        path = private / "documents" / (identity + ".json")
        document = load(path)
        document_hashes[identity] = sha(path)
        contract = row["public_contract"]
        for condition in ("original", "paraphrase"):
            sources["facts"].append((
                {"role_one": contract["role_one"], "role_two": contract["role_two"],
                 "public_object": contract["object"], "text": document[condition]["text"]},
                {"item": identity, "condition": condition}))
        sources["pairs"].append((
            {"source": document["original"]["text"], "paraphrase": document["paraphrase"]["text"]},
            {"item": identity}))
    keys, files = {}, {}
    for reviewer in (1, 2):
        target = folder / f"reviewer{reviewer}"
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        (target / "INSTRUCTIONS.md").write_text(REVIEW_INSTRUCTIONS)
        atomic(target / "reviewer.json", {"reviewer_code": "", "completed_at": "",
                                        "human_attestation": False,
                                        "independent_attestation": False})
        keys[str(reviewer)] = {}
        index = []
        for task, entries in sources.items():
            order = list(range(len(entries)))
            random.Random(20261026 + reviewer * 100 + len(task)).shuffle(order)
            packet, mapping = [], {}
            for j, i in enumerate(order):
                payload, identity = entries[i]
                alias = hashlib.sha256(f"fact1:{reviewer}:{task}:{j}".encode()).hexdigest()[:16]
                packet.append({"item_id": alias, **payload})
                mapping[alias] = {"identity": identity, "payload": payload}
                if task == "facts":
                    index.append(alias + "\n" + "\n".join(
                        f"[{m.start()},{m.end()}) {m.group()}"
                        for m in re.finditer(r"\S+", payload["text"])))
            keys[str(reviewer)][task] = mapping
            write_csv(target / (task + ".csv"), packet,
                      FACT_LABELS if task == "facts" else PAIR_LABELS)
        (target / "text_words.txt").write_text("\n\n".join(index) + "\n")
        for p in target.iterdir():
            os.chmod(p, 0o600)
            files[str(p.relative_to(folder))] = sha(p)
    atomic(folder / "admin/KEYS.json", keys)
    manifest = {"at": now(), "counts_per_reviewer": {"facts": 192, "pairs": 96},
                "files": files, "source_sha256": document_hashes,
                "keys_sha256": sha(folder / "admin/KEYS.json")}
    atomic(manifest_path, manifest)
    atomic(out / "HUMAN_REVIEW_STATUS.json", {
        "at": now(), "reviewers": 2, "counts_per_reviewer": manifest["counts_per_reviewer"],
        "blank_labels": True, "human_completed": 0, "gate": "NOT_EVALUATED",
        "private_packet_manifest_sha256": sha(manifest_path)})
    return {"reviewers": 2, "facts_per_reviewer": 192, "pairs_per_reviewer": 96,
            "completed": 0}


def validate_fact(row):
    if row["readability"] not in ("clear", "unclear", "contradictory", "unavailable"):
        raise ValueError("readability missing")
    if any(not row[k].strip() for k in FIELDS) or not row["notes"].strip():
        raise ValueError("facts or notes missing")
    if row["explicit_status"] not in ("yes", "no", "unknown"):
        raise ValueError("explicit_status missing")
    if row["quality"] not in ("1", "2", "3", "4", "5", "unknown"):
        raise ValueError("quality missing")
    if row["transfer_status"] not in ("completed", "planned_not_occurred", "other", "unknown"):
        raise ValueError("invalid state")
    evidence = json.loads(row["evidence_json"])
    if set(evidence) != set(FIELDS):
        raise ValueError("four evidence fields required")
    for field in FIELDS:
        spans = evidence[field]
        if not isinstance(spans, list):
            raise ValueError("evidence must be an array")
        if spans and not validate_evidence(row["text"], spans):
            raise ValueError("evidence not literal")
        if row[field] != "unknown" and not validate_evidence(row["text"], spans):
            raise ValueError("known fact needs evidence")
    if row["readability"] == "clear" and any(row[k] == "unknown" for k in FIELDS):
        raise ValueError("clear row has unknown fact")
    if row["readability"] == "unavailable" and row["text"]:
        raise ValueError("unavailable row has source text")
    return {k: row[k] for k in FACT_LABELS}


def validate_pair(row):
    if any(row[k] not in ("yes", "no", "unknown") for k in PAIR_LABELS[:3]):
        raise ValueError("pair labels missing")
    if not row["notes"].strip():
        raise ValueError("pair notes missing")
    evidence = json.loads(row["evidence_json"])
    if set(evidence) != {"source", "paraphrase"}:
        raise ValueError("pair evidence keys missing")
    for side in evidence:
        if not isinstance(evidence[side], list):
            raise ValueError("pair evidence must be an array")
        if evidence[side] and not validate_evidence(row[side], evidence[side]):
            raise ValueError("pair evidence not literal")
    if any(row[k] == "yes" for k in PAIR_LABELS[:3]) and any(
            not validate_evidence(row[k], evidence[k]) for k in evidence):
        raise ValueError("yes pair needs both source evidences")
    return {k: row[k] for k in PAIR_LABELS}


def semantic_signature(task, labels):
    fields = (["readability", *FIELDS, "explicit_status"] if task == "facts"
              else PAIR_LABELS[:3])
    return tuple(labels[k].strip().casefold() for k in fields)


def fact_symbol(label, contract):
    if label["readability"] != "clear" or label["explicit_status"] != "yes":
        return None
    return classify_facts(label, contract).symbol


def bootstrap(values):
    rng = random.Random(GATES["bootstrap_seed"])
    draws = sorted(sum(values[rng.randrange(24)] for _ in range(24)) / 24
                   for _ in range(GATES["bootstrap_replicates"]))
    return [draws[249], draws[9749]]


def valid_completed_at(metadata):
    try:
        completed_at = datetime.fromisoformat(metadata["completed_at"].replace("Z", "+00:00"))
        return completed_at.tzinfo is not None and completed_at <= datetime.now(timezone.utc)
    except (ValueError, KeyError, TypeError):
        return False


def collect(out, private):
    verify(out, private)
    folder = out / "private_review"
    manifest = load(folder / "admin/MANIFEST.json")
    if sha(folder / "admin/KEYS.json") != manifest["keys_sha256"]:
        raise RuntimeError("Private review key changed")
    for identity, signature in manifest["source_sha256"].items():
        if sha(private / "documents" / (identity + ".json")) != signature:
            raise RuntimeError("Source document changed")
    previous = out / "HUMAN_COLLECTION.json"
    if previous.exists() and load(previous).get("status") == "COMPLETE":
        result = load(previous)
        if any(sha(folder / p) != s for p, s in result["human_input_sha256"].items()):
            raise RuntimeError("Completed human labels are sealed; create a new audit version")
        return result
    keys = load(folder / "admin/KEYS.json")
    errors, reviewers, codes = [], {}, []
    for reviewer in (1, 2):
        code = str(reviewer)
        directory = folder / f"reviewer{reviewer}"
        metadata = load(directory / "reviewer.json")
        if not (metadata.get("reviewer_code", "").strip()
                and metadata.get("human_attestation") is True
                and metadata.get("independent_attestation") is True):
            errors.append({"reviewer": reviewer, "reason": "missing_human_attestation"})
        if not valid_completed_at(metadata):
            errors.append({"reviewer": reviewer, "reason": "invalid_completion_time"})
        codes.append(metadata.get("reviewer_code", "").strip())
        reviewers[code] = {}
        for task in ("facts", "pairs"):
            with (directory / (task + ".csv")).open(newline="") as stream:
                packet = list(csv.DictReader(stream))
            expected = keys[code][task]
            if (len(packet) != len(expected)
                    or {r["item_id"] for r in packet} != set(expected)):
                errors.append({"reviewer": reviewer, "task": task, "reason": "wrong_full_denominator"})
                continue
            for row in packet:
                ref = expected[row["item_id"]]
                try:
                    if any(row[k] != v for k, v in ref["payload"].items()):
                        raise ValueError("public payload edited")
                    labels = validate_fact(row) if task == "facts" else validate_pair(row)
                    reviewers[code][(task, digest(ref["identity"]))] = {
                        "identity": ref["identity"], "labels": labels, "payload": ref["payload"]}
                except (ValueError, KeyError, TypeError) as error:
                    errors.append({"reviewer": reviewer, "task": task,
                                   "item_id": row["item_id"], "reason": str(error)})
    if codes[0] and codes[0] == codes[1]:
        errors.append({"reason": "reviewers_must_be_distinct"})
    if errors:
        atomic(out / "HUMAN_COLLECTION.json", {
            "at": now(), "status": "INCOMPLETE_OR_INVALID", "error_count": len(errors),
            "human_gate": "NOT_EVALUATED", "human_labels_claimed": False})
        atomic(folder / "admin/COLLECTION_ERRORS.json", errors)
        return {"status": "INCOMPLETE_OR_INVALID", "error_count": len(errors)}
    # Agreement is by actual facts, never target symbols. Quality differences stay separate.
    resolved, disputes = {}, []
    for identity in sorted(reviewers["1"]):
        one = reviewers["1"][identity]
        two = reviewers["2"][identity]
        task = identity[0]
        if semantic_signature(task, one["labels"]) == semantic_signature(task, two["labels"]):
            resolved[identity] = one
        else:
            disputes.append({"task": task, "identity": one["identity"],
                             "payload": one["payload"], "reviewer1": one["labels"],
                             "reviewer2": two["labels"]})
    atomic(folder / "admin/DISPUTES.json", disputes)
    adjudication = folder / "adjudicator"
    if disputes:
        if not (adjudication / "KEYS.json").exists():
            adjudication.mkdir(parents=True, exist_ok=True, mode=0o700)
            mapping = {}
            for task in ("facts", "pairs"):
                packet = []
                for i, dispute in enumerate(disputes):
                    if dispute["task"] != task:
                        continue
                    alias = digest({"adjudication": i, "task": task})[:16]
                    packet.append({"item_id": alias, **dispute["payload"]})
                    mapping[alias] = {"task": task, "identity": dispute["identity"],
                                      "payload": dispute["payload"]}
                if packet:
                    write_csv(adjudication / (task + ".csv"), packet,
                              FACT_LABELS if task == "facts" else PAIR_LABELS)
            atomic(adjudication / "KEYS.json", mapping)
            atomic(adjudication / "reviewer.json", {"reviewer_code": "", "completed_at": "",
                                                  "human_attestation": False,
                                                  "independent_attestation": False})
            (adjudication / "INSTRUCTIONS.md").write_text(REVIEW_INSTRUCTIONS)
        meta = load(adjudication / "reviewer.json")
        if not (meta.get("reviewer_code", "").strip() not in ("", *codes)
                and meta.get("human_attestation") is True
                and meta.get("independent_attestation") is True
                and valid_completed_at(meta)):
            atomic(out / "HUMAN_COLLECTION.json", {
                "at": now(), "status": "AWAITING_HUMAN_ADJUDICATION",
                "disputes": len(disputes), "human_gate": "NOT_EVALUATED"})
            return {"status": "AWAITING_HUMAN_ADJUDICATION", "disputes": len(disputes)}
        mapping = load(adjudication / "KEYS.json")
        expected_mapping = {}
        for i, dispute in enumerate(disputes):
            alias = digest({"adjudication": i, "task": dispute["task"]})[:16]
            expected_mapping[alias] = {k: dispute[k] for k in ("task", "identity", "payload")}
        if mapping != expected_mapping:
            raise RuntimeError("Adjudication source or dispute set changed")
        filled = {}
        for task in ("facts", "pairs"):
            path = adjudication / (task + ".csv")
            if not path.exists():
                continue
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    ref = mapping[row["item_id"]]
                    if row["item_id"] in filled or ref["task"] != task:
                        raise RuntimeError("Duplicate or misplaced adjudication")
                    if any(row[k] != v for k, v in ref["payload"].items()):
                        raise RuntimeError("Adjudication payload edited")
                    label = validate_fact(row) if task == "facts" else validate_pair(row)
                    filled[row["item_id"]] = {**ref, "labels": label}
        if set(filled) != set(mapping):
            raise RuntimeError("Incomplete adjudication")
        for data in filled.values():
            resolved[(data["task"], digest(data["identity"]))] = data
    rows = load(private / "PLAN.json")
    write, preserve, para_write = [], [], []
    by_symbol = {s: {"denominator": 24, "writing_correct": 0,
                     "paraphrase_target_correct": 0, "paraphrase_preserved": 0} for s in SYMBOLS}
    confusion = {condition: {s: dict.fromkeys([*SYMBOLS, "erasure"], 0) for s in SYMBOLS}
                 for condition in ("original", "paraphrase")}
    readability = {condition: Counter() for condition in ("original", "paraphrase")}
    for entry in rows:
        row = entry["row"]
        identity = row["item_id"]
        labels = {condition: resolved[("facts", digest(
            {"item": identity, "condition": condition}))]["labels"]
                  for condition in ("original", "paraphrase")}
        pair = resolved[("pairs", digest({"item": identity}))]["labels"]
        write.append(fact_symbol(labels["original"], row["public_contract"]) == row["intended_symbol"])
        para_write.append(fact_symbol(labels["paraphrase"], row["public_contract"]) == row["intended_symbol"])
        preserve.append(all(pair[k] == "yes" for k in PAIR_LABELS[:3]))
        symbol = row["intended_symbol"]
        by_symbol[symbol]["writing_correct"] += write[-1]
        by_symbol[symbol]["paraphrase_target_correct"] += para_write[-1]
        by_symbol[symbol]["paraphrase_preserved"] += preserve[-1]
        for condition in labels:
            actual = fact_symbol(labels[condition], row["public_contract"])
            confusion[condition][symbol][actual or "erasure"] += 1
            readability[condition][labels[condition]["readability"]] += 1
    cluster = lambda flags: [sum(flags[i:i + 4]) / 4 for i in range(0, 96, 4)]
    writing_ci, preserve_ci = bootstrap(cluster(write)), bootstrap(cluster(preserve))
    passed = (sum(write) / 96 >= GATES["writing_min"]
              and sum(preserve) / 96 >= GATES["paraphrase_preserved_min"]
              and writing_ci[0] >= GATES["cluster_bootstrap_lower_min"]
              and preserve_ci[0] >= GATES["cluster_bootstrap_lower_min"])
    atomic(folder / "admin/RESOLVED.json", list(resolved.values()))
    quality = {r: dict(Counter(data["labels"]["quality"] for identity, data in values.items()
                               if identity[0] == "facts")) for r, values in reviewers.items()}
    human_files = [p for r in (1, 2) for p in
                   (f"reviewer{r}/facts.csv", f"reviewer{r}/pairs.csv", f"reviewer{r}/reviewer.json")]
    if disputes:
        human_files += [str(p.relative_to(folder)) for p in adjudication.iterdir()
                        if p.suffix == ".csv" or p.name in ("reviewer.json", "KEYS.json")]
    outcome = {"at": now(), "status": "COMPLETE", "denominator": 96,
               "writing_correct": sum(write), "paraphrase_preserved": sum(preserve),
               "paraphrase_target_correct_diagnostic": sum(para_write),
               "writing_cluster_95ci": writing_ci, "preserved_cluster_95ci": preserve_ci,
               "human_gate": "GO_FACT_LAYER_PREPARATION" if passed else "NO_GO",
               "parser_run_authorized_by_this_result": False,
               "disputes_adjudicated": len(disputes), "unknown_counts_as_failure": True,
               "by_symbol": by_symbol, "target_vs_actual_facts": confusion,
               "readability": {k: dict(v) for k, v in readability.items()},
               "quality_histogram_per_reviewer_full_192_denominator": quality,
               "human_input_sha256": {p: sha(folder / p) for p in human_files}}
    atomic(out / "HUMAN_COLLECTION.json", outcome)
    return outcome


def start(out, private, access):
    precheck(out, private, access)
    config = load(access)
    runtime = private / "supervisor"
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    daemon_config = runtime / "supervisord.conf"
    if (runtime / "supervisord.pid").exists():
        raise RuntimeError("Supervisor PID exists; inspect status before starting")
    command = [config["python"], str(Path(__file__).resolve()), "run",
               "--out", str(out), "--private", str(private), "--access-config", str(access)]
    quoted = " ".join('"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"' for s in command)
    daemon_config.write_text(
        f"[unix_http_server]\nfile={runtime}/supervisor.sock\nchmod=0600\n"
        f"[supervisord]\nlogfile={runtime}/daemon.log\npidfile={runtime}/supervisord.pid\n"
        f"childlogdir={runtime}\n"
        "[rpcinterface:supervisor]\nsupervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface\n"
        f"[supervisorctl]\nserverurl=unix://{runtime}/supervisor.sock\n"
        f"[program:fact_carrier]\ncommand={quoted}\ndirectory={ROOT}\n"
        "autostart=true\nautorestart=unexpected\nexitcodes=0,2\nstartsecs=0\nstartretries=0\n"
        f"stdout_logfile={runtime}/runner.log\nstderr_logfile={runtime}/error.log\n"
        "stopasgroup=true\nkillasgroup=true\n")
    os.chmod(daemon_config, 0o600)
    subprocess.run([config["python"], "-m", "supervisor.supervisord",
                    "-c", str(daemon_config)], check=True, capture_output=True)
    return {"status": "SUPERVISOR_STARTED", "program": "fact_carrier"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "precheck", "start",
                                           "run", "status", "review", "collect"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--access-config", type=Path, default=Path("/tmp/amr_fact_carrier_access.json"))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.out, args.private)
        elif args.command == "verify":
            result = {"status": "PASS", "checks": verify(args.out, args.private)}
        elif args.command in ("precheck", "start", "run"):
            result = globals()[args.command](args.out, args.private, args.access_config)
        elif args.command == "status":
            result = load(args.out / "heartbeat.json")
        else:
            result = globals()[args.command](args.out, args.private)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except TerminalHalt as error:
        print(json.dumps({"status": "TERMINAL_HALT", "reason": str(error)}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
