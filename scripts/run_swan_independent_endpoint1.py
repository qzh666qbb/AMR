"""Independent-bank writer, shared 16-candidate attacks, and joint-writer comparison."""
import argparse
import csv
import fcntl
import gc
import json
from pathlib import Path
import re
import shutil
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
import run_swan_independent_banks1 as construction
import run_swan_independent_calibration1 as calibration
import swan_independent_endpoint1_budget as budget
from swan_final1_common import freeze
from swan_independent_endpoint1_core import (
    ARMS, SLOTS, ATTEMPTS, BUDGETS, order, template_schedule, attack_messages,
    accepted, client_score, client_threshold, rate)
from swan_independent_calibration1_core import (
    FLOWS, BANK_KEYS, CONDITIONS, variants, document_statistics, detected)
from semantic_fact_raw_amr import audit_raw, replay_signature
from swan_restricted_feedback1_server import session

OUT = ROOT / "experiments/swan_independent_endpoint1"
BANK = ROOT / "experiments/swan_independent_banks1"
CAL = ROOT / "experiments/swan_independent_calibration1"
OFFICIAL = "third_party/SWAN_official_2ca0c207_clean/injection/injection_amr_utils.py"
EXAMPLES = "third_party/SWAN_official_2ca0c207_clean/amr_bank/artifacts/normalized_amr_examples.json"
CODE = (
    "scripts/run_swan_independent_endpoint1.py", "scripts/swan_independent_endpoint1_core.py",
    "scripts/swan_independent_endpoint1_budget.py", "scripts/test_swan_independent_endpoint1.py",
    "scripts/symbol_prompt_completion1_budget.py", "scripts/swan_final1_common.py",
    "scripts/swan_restricted_feedback1_server.py", "scripts/swan_restricted_feedback1_client.py",
    "scripts/collect_swan_independent_endpoint1_quality.py",
)


def tokenizer(bank):
    import nltk
    nltk.data.path.insert(0, str(bank / "private/code/.nltk_data"))
    if nltk.__version__ != rt.read(bank / "FREEZE.json")["normalizer"]["nltk"]:
        raise RuntimeError("Sentence tokenizer changed")
    return nltk.tokenize.PunktTokenizer("english")


def prepare(out, bank, cal):
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Choose an unfrozen successor")
    parent = calibration.verify(cal, bank)
    tok = tokenizer(bank)
    sources = []
    for role in ("pilot", "main"):
        for row in rt.read(bank / "private/cohorts" / (role + ".json")):
            if rt.digest(row["text"]) != row["window_sha256"]:
                raise RuntimeError("Source text changed")
            sources.append({"source_id": row["allocation_id"], "role": role,
                            "source_text_sha256": row["window_sha256"],
                            "prompt": tok.tokenize(row["text"])[0]})
    if len(sources) != 72 or len({r["source_id"] for r in sources}) != 72:
        raise RuntimeError("Expected all 12 pilot plus 60 main sources")
    out.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(mode=0o700, exist_ok=True)
    (out / ".gitignore").write_text("private/\n*.log\n")
    freeze(out / "private/SOURCES.json", sources)
    freeze(out / "private/ORDER.json", order(sources))
    # All imports, including src.core's indirect dependencies, are isolated.
    shutil.copytree(bank / "private/code", out / "private/code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(cal / "private/code", out / "private/code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (*CODE, OFFICIAL, EXAMPLES):
        target = out / "private/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    freeze(out / "FREEZE.json", {
        "version": "independent-bank-endpoint/v1",
        "bank_freeze_sha256": rt.sha(bank / "FREEZE.json"),
        "calibration_freeze_sha256": rt.sha(cal / "FREEZE.json"),
        "parser": parent["parser"], "glove_sha256": parent["glove_sha256"],
        "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": .8},
        "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
        "sources": 72, "pilot": 12, "main": 60, "arms": list(ARMS),
        "slots_per_arm": SLOTS, "attempts_per_slot": ATTEMPTS,
        "writer_max_calls": 230400, "writer_cost_guard_usd": 40,
        "attack_candidates": 16, "attack_max_calls": 4608, "attack_cost_guard_usd": 10,
        "query_budgets": list(BUDGETS),
        "no_pilot_result_selection": True,
        "first_sentence_context_not_in_detector_text": True,
        "joint_writer": "A on even slots, B on odd slots, 8 each",
        "pooled_writer": "sample uniformly from 100 capacity slots, overlaps retained",
        "generation_acceptance": "official prompts, >=.75, complete single sentence; no fallback filtering",
        "quality_after_selection": True, "two_independent_reviewers_required": True,
    })
    return verify(out, bank, cal)


def verify(out, bank, cal):
    f = rt.read(out / "FREEZE.json")
    calibration.verify(cal, bank)
    if (rt.sha(bank / "FREEZE.json") != f["bank_freeze_sha256"]
            or rt.sha(cal / "FREEZE.json") != f["calibration_freeze_sha256"]):
        raise RuntimeError("Parent contract drift")
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Endpoint frozen file drift: " + name)
    return f


def check_seal(out, name):
    data = rt.read(out / name)
    if data["freeze_sha256"] != rt.sha(out / "FREEZE.json"):
        raise RuntimeError("Output seal identity changed")
    for relative, expected in data["files"].items():
        if rt.sha(out / relative) != expected:
            raise RuntimeError("Output artifact drift: " + relative)
    return data


def seal(out, name, files, **extra):
    return freeze(out / name, {"freeze_sha256": rt.sha(out / "FREEZE.json"),
                              "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(files)},
                              **extra})


def bind(out, bank, cal):
    for parent, status in ((bank, "INDEPENDENT_BANKS_CONSTRUCTED"),
                           (cal, "CALIBRATION_AND_INDEPENDENT_NULL_COMPLETE")):
        t = rt.read(parent / "TERMINAL.json")
        if t["status"] != status or any(t[k] != rt.sha(parent / p) for k, p in (
                ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
            raise RuntimeError("Parent terminal not complete or not authentic")
    audit = rt.read(cal / "AUDIT.json")
    if audit["thresholds_sha256"] != rt.sha(cal / "THRESHOLDS.json"):
        raise RuntimeError("Calibration threshold drift")
    banks = calibration.bind_banks(out, bank)
    cal_binding = rt.read(cal / "private/BANK_BINDING.json")
    if banks != cal_binding["banks"]:
        raise RuntimeError("Positive and negative banks differ")
    freeze(out / "private/DETECTOR_BINDING.json", {
        "calibration_terminal_sha256": rt.sha(cal / "TERMINAL.json"),
        "calibration_audit_sha256": rt.sha(cal / "AUDIT.json"),
        "thresholds_sha256": rt.sha(cal / "THRESHOLDS.json"),
        "thresholds": rt.read(cal / "THRESHOLDS.json")})
    return banks


def sender(key_file):
    if stat.S_IMODE(key_file.stat().st_mode) & 0o077:
        raise PermissionError("Credential file permissions")
    keys = [s.strip() for s in key_file.read_text().splitlines()
            if re.fullmatch(r"sk-[A-Za-z0-9_-]+", s.strip())]
    if len(keys) != 1:
        raise RuntimeError("Expected one configured credential")
    import requests
    def send(body):
        r = requests.post("https://api.deepseek.com/chat/completions",
                          data=json.dumps(body, ensure_ascii=True).encode(),
                          headers={"Authorization": "Bearer " + keys[0], "Content-Type": "application/json"},
                          timeout=(20, 240), allow_redirects=False)
        if r.status_code != 200:
            raise RuntimeError("Provider HTTP failure; unresolved intent must be audited")
        return r.json()
    return send


def save_row(path, row):
    return freeze(path, {"row": row, "sha256": rt.digest(json.dumps(row, sort_keys=True))})["row"]


def read_row(path, binding):
    envelope = rt.read(path)
    row = envelope["row"]
    if (envelope["sha256"] != rt.digest(json.dumps(row, sort_keys=True))
            or any(row[k] != v for k, v in binding.items())):
        raise RuntimeError("Row checksum or binding changed")
    return row


def disk(out):
    if shutil.disk_usage(out).free < 512 * 1024**2:
        raise rt.ResourcePause("Insufficient disk headroom")


def raw_template(out, text, parser, normalizer, folder):
    identity = rt.digest(text)
    path = out / "private" / folder / (identity + ".json")
    if path.exists():
        raw = rt.read(path)
    else:
        n = len(parser.model.tokenizer(text, truncation=False)["input_ids"])
        raw = (construction.excluded(text, parser.freeze_sha, n) if n >= 1024 else
               {**parser.fresh(text), "case_id": identity})
        freeze(path, raw)
    check = audit_raw(raw, {"case_id": identity, "input_text_sha256": identity}, parser.freeze_sha)
    if not check["structurally_eligible"]:
        return None
    value = normalizer(raw["amr"])
    return value["template"] if value["eligible"] else None


def write_documents(out, bank, assets, banks, key_file, f):
    contract = rt.sha(out / "FREEZE.json")
    journal = budget.journal(out, contract, "writer")
    if (out / "WRITING_COMPLETE.json").exists():
        check_seal(out, "WRITING_COMPLETE.json")
        return
    journal.send = sender(key_file)
    normalizer, tok = construction.Normalizer(bank), tokenizer(bank)
    matcher, parser = calibration.Matcher(out, assets), rt.Parser(assets, out, f)
    probe = "The researcher described the result."
    pair = [{**parser.fresh(probe), "case_id": rt.digest(probe)} for _ in range(2)]
    if replay_signature(pair[0]) != replay_signature(pair[1]):
        raise RuntimeError("Writer parser replay differs")
    freeze(out / "WRITER_PRECHECK.json", {
        "parser_two_replays_equal": True, "probe_sha256": rt.digest(probe),
        "S2Match_self_replays": [matcher.fresh(banks["a"][0], banks["a"][0]) for _ in range(2)]})
    if rt.read(out / "WRITER_PRECHECK.json")["S2Match_self_replays"] != [1., 1.]:
        raise RuntimeError("Writer S2Match self match differs")
    ns = {"json": json}
    rt.isolate(out / "private/code" / OFFICIAL,
               ["build_system_prompt", "build_user_prompt", "load_amr_examples"], ns)
    examples = ns["load_amr_examples"](str(out / "private/code" / EXAMPLES), 3)
    sources = {r["source_id"]: r for r in rt.read(out / "private/SOURCES.json")}
    texts = {(key, arm): [] for key in sources for arm in ARMS}
    for index, assignment in enumerate(rt.read(out / "private/ORDER.json")):
        key, arm, slot = (assignment[k] for k in ("source_id", "arm", "slot"))
        slot_id = rt.digest(key + "/" + arm + "/" + str(slot))
        path = out / "private/slots" / (slot_id + ".json")
        context = sources[key]["prompt"] + " " + " ".join(texts[key, arm])
        binding = {**assignment, "context_sha256": rt.digest(context), "freeze_sha256": contract}
        if path.exists():
            record = read_row(path, binding)
            for call_id, expected in record["attempt_files"].items():
                if rt.sha(out / "private/attempts" / (call_id + ".json")) != expected:
                    raise RuntimeError("Writer attempt changed")
        else:
            hashes = {}
            for draw, (bank_key, target) in enumerate(template_schedule(banks, key, arm, slot)):
                prompt = ns["build_user_prompt"](target, examples, context)
                messages = [{"role": "system", "content": ns["build_system_prompt"]()},
                            {"role": "user", "content": prompt}]
                for within in range(5):
                    disk(out)
                    call_id = slot_id + "-" + str(draw * 5 + within)
                    response = journal.request(call_id, messages)
                    event_path = out / "private/attempts" / (call_id + ".json")
                    event_binding = {"call_id": call_id, "target": target, "bank": bank_key,
                                     "text": response["text"], "finish_reason": response["finish_reason"],
                                     "freeze_sha256": contract}
                    if event_path.exists():
                        event = read_row(event_path, event_binding)
                    else:
                        text = response["text"]
                        template = (raw_template(out, text, parser, normalizer, "writer_graphs")
                                    if text.strip() else None)
                        score = matcher.pair(template, target) if template is not None else None
                        event = save_row(event_path, {**event_binding, "score": score,
                            "template_comparisons": int(template is not None),
                            "accepted": accepted(text, response["finish_reason"], score, tok)})
                    hashes[call_id] = rt.sha(event_path)
                    if event["accepted"]:
                        break
                if event["accepted"]:
                    break
            record = save_row(path, {**binding, "text": event["text"], "accepted": event["accepted"],
                                    "fallback": not event["accepted"], "attempt_files": hashes,
                                    "attempts": len(hashes)})
        texts[key, arm].append(record["text"])
        rt.atomic(out / "HEARTBEAT.json", {"phase": "writing", "slots": index + 1,
                                          "total_slots": 72 * 4 * SLOTS,
                                          "budget": journal.totals(), "at": rt.now()})
    for (key, arm), values in texts.items():
        freeze(out / "private/documents" / (key + "-" + arm + ".json"), {
            "source_id": key, "role": sources[key]["role"], "arm": arm,
            "text": " ".join(values), "slots": SLOTS, "text_sha256": rt.digest(" ".join(values))})
    files = [p for folder in ("documents", "slots", "attempts", "writer_graphs", "provider/writer")
             for p in (out / "private" / folder).rglob("*") if p.is_file()]
    seal(out, "WRITING_COMPLETE.json", files, documents=len(texts), provider=journal.totals())
    matcher.db.close()
    del parser, normalizer, matcher
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def generate_attacks(out, key_file):
    check_seal(out, "WRITING_COMPLETE.json")
    journal = budget.journal(out, rt.sha(out / "FREEZE.json"), "attack")
    if (out / "ATTACK_GENERATION_COMPLETE.json").exists():
        check_seal(out, "ATTACK_GENERATION_COMPLETE.json")
        return
    journal.send = sender(key_file)
    for source in rt.read(out / "private/SOURCES.json"):
        for arm in ARMS:
            doc_id = source["source_id"] + "-" + arm
            doc = rt.read(out / "private/documents" / (doc_id + ".json"))
            for index in range(16):
                disk(out)
                call_id = rt.digest(doc_id) + "-" + str(index)
                result = journal.request(call_id, attack_messages(doc["text"], index))
                freeze(out / "private/candidates" / (call_id + ".json"), {
                    "doc_id": doc_id, "source_id": source["source_id"], "arm": arm,
                    "role": source["role"], "candidate_id": str(index), "text": result["text"],
                    "text_sha256": rt.digest(result["text"]), "finish_reason": result["finish_reason"],
                    "source_clean_sha256": doc["text_sha256"]})
                rt.atomic(out / "HEARTBEAT.json", {"phase": "attack_generation",
                    "budget": journal.totals(), "at": rt.now()})
    files = [p for folder in ("candidates", "provider/attack")
             for p in (out / "private" / folder).rglob("*") if p.is_file()]
    count = len(rt.read(out / "private/SOURCES.json")) * len(ARMS) * 16
    if len(list((out / "private/candidates").glob("*.json"))) != count:
        raise RuntimeError("Generated candidate denominator differs")
    seal(out, "ATTACK_GENERATION_COMPLETE.json", files, candidates=count, provider=journal.totals())


def evaluation_inputs(out, bank):
    tok, docs = tokenizer(bank), []
    for source in rt.read(out / "private/SOURCES.json"):
        for arm in ARMS:
            doc_id = source["source_id"] + "-" + arm
            base = rt.read(out / "private/documents" / (doc_id + ".json"))
            common = {k: base[k] for k in ("source_id", "role", "arm")}
            # Empty output remains in the denominator, with no parse input.
            transform_error = None
            try:
                transforms = variants(base["text"], tok) if base["text"].strip() else dict.fromkeys(CONDITIONS, "")
            except ValueError as exc:
                transform_error = str(exc)
                transforms = dict.fromkeys(CONDITIONS, "")
                transforms["clean"] = base["text"]
            for condition, text in transforms.items():
                docs.append({**common, "doc_id": doc_id, "condition": condition,
                             "candidate_id": None, "text": text, "text_sha256": rt.digest(text),
                             "sentences": tok.tokenize(text),
                             "transform_error": transform_error if condition != "clean" else None})
            for index in range(16):
                candidate = rt.read(out / "private/candidates" / (rt.digest(doc_id) + "-" + str(index) + ".json"))
                docs.append({**common, **candidate, "condition": "paraphrase",
                             "sentences": tok.tokenize(candidate["text"])})
    freeze(out / "private/EVALUATION_INPUTS.json", docs)
    return docs


def score_documents(out, bank, assets, banks, f):
    if (out / "SCORING_COMPLETE.json").exists():
        check_seal(out, "SCORING_COMPLETE.json")
        return
    docs = evaluation_inputs(out, bank)
    # Fresh evaluation graphs, all sealed before any document score/selection.
    construction.parse(out, assets, f, docs)
    normalizer, matcher = construction.Normalizer(bank), calibration.Matcher(out, assets)
    cache = {}
    for index, doc in enumerate(docs):
        ident = rt.digest(doc["doc_id"] + "/" + doc["condition"] + "/" + str(doc["candidate_id"]))
        binding = {k: doc[k] for k in ("source_id", "role", "arm", "doc_id", "condition",
                                     "candidate_id", "text_sha256")}
        binding["freeze_sha256"] = rt.sha(out / "FREEZE.json")
        binding["transform_error"] = doc.get("transform_error")
        path = out / "private/rows" / (ident + ".json")
        if path.exists():
            read_row(path, binding)
            continue
        sentence_scores = []
        for text in doc["sentences"]:
            identity = rt.digest(text)
            if identity not in cache:
                raw = rt.read(out / "private/raw_graphs" / (identity + ".json"))
                check = audit_raw(raw, {"case_id": identity, "input_text_sha256": identity},
                                  binding["freeze_sha256"])
                normalized = normalizer(raw["amr"]) if check["structurally_eligible"] else None
                cache[identity] = (normalized["template"] if normalized and normalized["eligible"] else None)
            template = cache[identity]
            sentence_scores.append({
                key: max(matcher.pair(template, target) for target in banks[key])
                if template is not None else None for key in BANK_KEYS})
        save_row(path, {**binding, **document_statistics(sentence_scores),
                       "sentence_scores": sentence_scores,
                       "logical_comparisons": {
                           "public": 50 * sum(r["public"] is not None for r in sentence_scores),
                           "a": 50 * sum(r["a"] is not None for r in sentence_scores),
                           "b": 50 * sum(r["b"] is not None for r in sentence_scores),
                           "joint": 100 * sum(r["a"] is not None for r in sentence_scores),
                           "pooled": 100 * sum(r["a"] is not None for r in sentence_scores)}})
        rt.atomic(out / "HEARTBEAT.json", {"phase": "endpoint_scoring", "rows": index + 1,
                                          "total": len(docs), "at": rt.now()})
        disk(out)
    matcher.db.close()
    expected = {rt.digest(d["doc_id"] + "/" + d["condition"] + "/" + str(d["candidate_id"]))
                for d in docs}
    if {p.stem for p in (out / "private/rows").glob("*.json")} != expected:
        raise RuntimeError("Scored row denominator differs")
    seal(out, "SCORING_COMPLETE.json", list((out / "private/rows").glob("*.json")),
         rows=len(docs), evaluation_inputs_sha256=rt.sha(out / "private/EVALUATION_INPUTS.json"),
         raw_graph_seal_sha256=rt.sha(out / "RAW_GRAPH_SEAL.json"))


def select_attacks(out):
    check_seal(out, "SCORING_COMPLETE.json")
    thresholds = rt.read(out / "private/DETECTOR_BINDING.json")["thresholds"]
    rows = [read_row(p, {}) for p in sorted((out / "private/rows").glob("*.json"))]
    lookup = {(r["doc_id"], r["condition"], r["candidate_id"]): r for r in rows}
    if len(lookup) != len(rows):
        raise RuntimeError("Duplicate evaluation row")
    for source in rt.read(out / "private/SOURCES.json"):
        for arm in ARMS:
            doc_id = source["source_id"] + "-" + arm
            candidates = [rt.read(out / "private/candidates" / (rt.digest(doc_id) + "-" + str(i) + ".json"))
                          for i in range(16)]
            clean = lookup[doc_id, "clean", None]
            for oracle in ("public", arm):
                scores = {str(i): client_score(lookup[doc_id, "paraphrase", str(i)]["scores"][oracle])
                          for i in range(16)}
                threshold = client_threshold(thresholds["clean"][oracle])
                for queries in BUDGETS:
                    for mode in ("none", "label", "score"):
                        result = session(candidates, scores, threshold, mode, queries)
                        selected = result["selection"]["selected"]
                        selected_row = lookup[doc_id, "paraphrase", selected]
                        clean_hits = {flow: detected(clean["scores"][flow], thresholds["clean"][flow])
                                      for flow in FLOWS}
                        attack_hits = {flow: detected(selected_row["scores"][flow], thresholds["clean"][flow])
                                       for flow in FLOWS}
                        freeze(out / "private/selections" / (
                            doc_id + "-" + oracle + "-" + mode + "-" + str(queries) + ".json"), {
                            "source_id": source["source_id"], "role": source["role"], "arm": arm,
                            "doc_id": doc_id, "oracle": oracle, "mode": mode, "budget": queries,
                            **result, "clean_detected": clean_hits, "attack_detected": attack_hits,
                            "selected_abstained": selected_row["abstain"],
                            "selected_raw_score": selected_row["scores"][oracle],
                            "selection_quality_labels_seen": 0})
    files = list((out / "private/selections").glob("*.json"))
    count = len(rt.read(out / "private/SOURCES.json")) * len(ARMS) * 2 * 3 * 3
    if len(files) != count:
        raise RuntimeError("Attack session denominator differs")
    seal(out, "SELECTIONS_COMPLETE.json", files, sessions=count, attacker_target_templates_seen=0)
    return rows


def summarize(out, rows):
    thresholds = rt.read(out / "private/DETECTOR_BINDING.json")["thresholds"]
    selections = [rt.read(p) for p in sorted((out / "private/selections").glob("*.json"))]
    groups, comparisons = [], []
    for role in ("pilot", "main"):
        for arm in ARMS:
            baseline = [r for r in rows if r["role"] == role and r["arm"] == arm and r["condition"] == "clean"]
            groups.append({"role": role, "arm": arm, "sources": len(baseline),
                           "unfiltered_clean_detection": rate(sum(detected(
                               r["scores"][arm], thresholds["clean"][arm]) for r in baseline), len(baseline)),
                           "abstentions": sum(r["abstain"] for r in baseline),
                           "sentence_counts": [r["sentences"] for r in baseline],
                           "word_counts": [len(re.findall(r"\w+", rt.read(
                               out / "private/documents" / (r["doc_id"] + ".json"))["text"]))
                               for r in baseline]})
            for oracle in ("public", arm):
                for mode in ("none", "label", "score"):
                    for q in BUDGETS:
                        chosen = [r for r in selections if (r["role"], r["arm"], r["oracle"], r["mode"], r["budget"])
                                  == (role, arm, oracle, mode, q)]
                        eligible = [r for r in chosen if r["clean_detected"][arm]]
                        proxy_success = [r for r in eligible
                                         if r["clean_detected"]["public"] and not r["attack_detected"]["public"]]
                        comparisons.append({"role": role, "arm": arm, "oracle": oracle, "mode": mode,
                            "budget": q, "sources": len(chosen), "queries": sum(r["selection"]["queries"] for r in chosen),
                            "unfiltered_escape_given_target_clean": rate(
                                sum(not r["attack_detected"][arm] for r in eligible), len(eligible)),
                            "unfiltered_transfer_given_proxy_escape": rate(
                                sum(not r["attack_detected"][arm] for r in proxy_success), len(proxy_success))
                                if oracle == "public" else None,
                            "ASR": None, "ATR": None, "human_quality_complete": False})
    slots = [read_row(p, {}) for p in (out / "private/slots").glob("*.json")]
    costs = {arm: {"slots": sum(r["arm"] == arm for r in slots),
                  "attempts": sum(r["attempts"] for r in slots if r["arm"] == arm),
                  "accepted_slots": sum(r["accepted"] for r in slots if r["arm"] == arm),
                  "fallback_slots": sum(r["fallback"] for r in slots if r["arm"] == arm)}
             for arm in ARMS}
    attempts = [read_row(p, {}) for p in (out / "private/attempts").glob("*.json")]
    for arm in ARMS:
        calls = {key for r in slots if r["arm"] == arm for key in r["attempt_files"]}
        costs[arm]["template_comparisons"] = sum(r["template_comparisons"] for r in attempts
                                                if r["call_id"] in calls)
    boundary = []
    for role in ("pilot", "main"):
        for arm in ARMS:
            for condition in CONDITIONS:
                subset = [r for r in rows if (r["role"], r["arm"], r["condition"]) == (role, arm, condition)]
                for reference in ("clean", condition) if condition != "clean" else ("clean",):
                    boundary.append({"role": role, "arm": arm, "condition": condition,
                        "threshold_condition": reference,
                        "unfiltered_detection": rate(sum(detected(r["scores"][arm],
                            thresholds[reference][arm]) for r in subset), len(subset)),
                        "abstentions": sum(r["abstain"] for r in subset),
                        "transform_failures": sum(r.get("transform_error") is not None for r in subset)})
    return {"status": "MACHINE_ENDPOINT_COMPLETE_HUMAN_QUALITY_PENDING",
            "writer_groups": groups, "attacks": comparisons, "writer_costs": costs,
            "boundary_conditions": boundary,
            "detector_logical_comparisons": {flow: sum(r["logical_comparisons"][flow] for r in rows)
                                              for flow in FLOWS},
            "fixed_pool_nonadaptive": True,
            "feedback_escape_equality_is_structural_when_any_candidate_escapes": True,
            "TPR": None, "ASR": None, "ATR": None, "independent_human_reviews": 0}


def review_packet(out):
    # Review every clean text and the UNION of sealed selections; no resampling.
    selections = [rt.read(p) for p in (out / "private/selections").glob("*.json")]
    chosen = {(r["doc_id"], r["selection"]["selected"]) for r in selections}
    items, mapping = [], {}
    for p in sorted((out / "private/documents").glob("*.json")):
        doc = rt.read(p)
        doc_id = doc["source_id"] + "-" + doc["arm"]
        identity = rt.digest("review/clean/" + doc_id)
        mapping[identity] = {"doc_id": doc_id, "candidate_id": None}
        items.append({"item_id": identity, "reference": "", "text": doc["text"], "kind": "clean"})
    for doc_id, candidate_id in sorted(chosen):
        clean = rt.read(out / "private/documents" / (doc_id + ".json"))
        candidate = rt.read(out / "private/candidates" / (rt.digest(doc_id) + "-" + candidate_id + ".json"))
        identity = rt.digest("review/attack/" + doc_id + "/" + candidate_id)
        mapping[identity] = {"doc_id": doc_id, "candidate_id": candidate_id}
        items.append({"item_id": identity, "reference": clean["text"], "text": candidate["text"], "kind": "rewrite"})
    items.sort(key=lambda r: r["item_id"])
    freeze(out / "private/quality/BLIND_ITEMS.json", items)
    freeze(out / "private/quality/EVALUATOR_MAPPING.json", mapping)
    freeze(out / "private/quality/CONTRACT.json", {
        "items_sha256": rt.sha(out / "private/quality/BLIND_ITEMS.json"),
        "mapping_sha256": rt.sha(out / "private/quality/EVALUATOR_MAPPING.json"),
        "selection_seal_sha256": rt.sha(out / "SELECTIONS_COMPLETE.json"),
        "required_reviewers": 2, "items": len(items),
        "labels": ["natural", "coherent", "preserves_meaning"],
        "disagreement": "third human adjudication required; no majority or automatic relabel",
        "quality_may_not_change_selection": True})
    for reviewer in (1, 2):
        path = out / "private/quality" / ("reviewer_" + str(reviewer) + ".csv")
        if not path.exists():
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "item_id", "packet_sha256", "reviewer_id", "natural", "coherent", "preserves_meaning"])
                writer.writeheader()
                writer.writerows({"item_id": r["item_id"],
                                  "packet_sha256": rt.sha(out / "private/quality/BLIND_ITEMS.json")} for r in items)


def run(out, bank, cal, assets, key_file):
    f = verify(out, bank, cal)
    if (out / "TERMINAL.json").exists():
        t = rt.read(out / "TERMINAL.json")
        if any(t[k] != rt.sha(out / p) for k, p in (
                ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
            raise RuntimeError("Final artifacts changed")
        return
    if (out / "HALT.json").exists():
        raise RuntimeError("Integrity halt requires successor")
    banks = bind(out, bank, cal)
    write_documents(out, bank, assets, banks, key_file, f)
    generate_attacks(out, key_file)
    score_documents(out, bank, assets, banks, f)
    rows = select_attacks(out)
    review_packet(out)
    freeze(out / "RESULT.json", summarize(out, rows))
    freeze(out / "AUDIT.json", {
        "freeze_sha256": rt.sha(out / "FREEZE.json"),
        "seals": {name: rt.sha(out / name) for name in (
            "WRITING_COMPLETE.json", "ATTACK_GENERATION_COMPLETE.json", "RAW_GRAPH_SEAL.json",
            "SCORING_COMPLETE.json", "SELECTIONS_COMPLETE.json")},
        "same_positive_negative_parser_normalizer_matcher_statistics": True,
        "provider": {phase: budget.journal(out, rt.sha(out / "FREEZE.json"), phase).totals()
                     for phase in ("writer", "attack")},
        "missing_human_quality": True, "scientific_closure": False})
    freeze(out / "TERMINAL.json", {"status": "MACHINE_ENDPOINT_COMPLETE_HUMAN_QUALITY_PENDING",
        "result_sha256": rt.sha(out / "RESULT.json"), "audit_sha256": rt.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--calibration", type=Path, default=CAL)
    ap.add_argument("--assets", type=Path, default=ROOT)
    ap.add_argument("--key-file", type=Path)
    args = ap.parse_args()
    if args.action in ("prepare", "verify"):
        f = (prepare if args.action == "prepare" else verify)(args.out, args.bank, args.calibration)
        print(json.dumps({k: f[k] for k in ("sources", "arms", "writer_max_calls", "attack_max_calls")}))
        return
    with (args.out / "private/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        try:
            run(args.out, args.bank, args.calibration, args.assets, args.key_file)
        except Exception as exc:
            resource = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            money = type(exc).__name__ == "BudgetPause"
            rt.atomic(args.out / ("PAUSED.json" if resource or money else "HALT.json"), {
                "at": rt.now(), "reason_type": type(exc).__name__,
                "resource_resume_allowed": resource, "scientific_result": None,
                "budget_increase_not_automatic": money})
            if resource or money:
                sys.exit(75 if resource else 76)
            raise
        finally:
            rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
