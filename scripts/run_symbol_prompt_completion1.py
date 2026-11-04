"""Paired old/revised prompt recipes on all original 24 scenes and templates."""
import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import re
import shutil
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import run_semantic_sequence_stage1b as original
import run_multiscale_transfer as runtime
import run_symbol_paraphrase_completion1 as inference
import run_symbol_sequence_diagnostic as diagnostic
import symbol_prompt_completion1_budget as budget
from src.core.semantic_sequence import equal_composition_sequences, exact_costs, rank_truth, unordered_costs
from swan_final1_common import freeze, load, sha256

P2 = ROOT / "第二篇_语义消息写入与读回"
BASE = P2 / "02_实验/semantic_sequence_stage1_run2"
DEFAULT = P2 / "02_实验/semantic_sequence_prompt_completion1"
ORIGINAL_SHA = "b3d017549843dba801b597648c34674b588aa6488128ce9c9bc130f6f0a3721a"
ARMS = ("old", "revised")
REPAIR = (
    "Realize the semantic relations in the supplied abstract AMR graph in natural English. "
    "Treat repeated AMR variables as references to the same entity or event. "
    "Preserve who acts, who is affected, and the recipient or destination. "
    "Preserve nested events and their attachment. V is a predicate placeholder. "
    "N is a common concept placeholder. NE represents the named entity structure "
    "specified by the graph. A name with :op1 and :op2 requires two parts of the "
    "same entity's name, not two different named people; three operands require "
    "three name parts. X denotes an unknown semantic value corresponding to "
    "amr-unknown under the official normalizer. Do not replace it with a concrete "
    "answer or an arbitrary ordinary noun. Express the unknown value in the role "
    "specified by the graph. Do not assume every X can be expressed by negation. "
    "Respect :mode imperative, :polite, :polarity, and requests versus assertions. "
    "Do not turn a command into a past-tense report. Variable names are not words "
    "to emit. Output English text only, no AMR."
)


def revised_scenario(text):
    before = ("Use those names when named entities are needed; do not introduce "
              "other proper names. Use clear, matter-of-fact prose in the past tense, ")
    after = ("Use those participants when named entities are needed. If the graph "
             "requires a multipart name, extend a participant's name consistently "
             "with invented name parts. Use questions, commands, requests, or "
             "statements as required by the target meaning. Use clear prose, ")
    if text.count(before) != 1:
        raise RuntimeError("Original scenario contract differs")
    return text.replace(before, after)


def messages(prompt, prior, slot, template, arm):
    if arm == "old":
        return original.carrier_messages(prompt, prior, slot, template)
    if arm != "revised":
        raise ValueError("Unknown arm")
    result = original.carrier_messages(revised_scenario(prompt), prior, slot, None)
    result[-1]["content"] += "\n\n" + REPAIR + "\nTARGET AMR\n" + template
    return result


def template_contracts(bank, pool):
    import penman
    result = []
    for index in pool:
        text = bank[index]["normalized_template_amr"]
        graph = penman.decode(text)
        concepts = {t.source: t.target for t in graph.instances()}
        incoming = Counter(t.target for t in graph.edges())
        unknown_roles = sorted(role for _, role, target in graph.triples
                               if role != ":instance" and (concepts.get(target) == "X" or target == "X"))
        name_nodes = {target for _, role, target in graph.triples if role == ":name"}
        names = [sum(source == variable and role.startswith(":op") for source, role, _ in graph.triples)
                 for variable in sorted(name_nodes)]
        result.append({
            "template": index, "graph_sha256": runtime.digest(text),
            "unknown_roles": unknown_roles, "name_operand_counts": names,
            "reentrant_nodes": sum(n > 1 for n in incoming.values()),
            "imperative": any(role == ":mode" and target == "imperative" for _, role, target in graph.triples),
            "polarity_X": ":polarity" in unknown_roles,
            "semantic_contract": "PENDING_HUMAN_SOURCE_GRAPH_INTERPRETATION",
            "mechanical_description_only": True,
        })
    return result


def prepare(out):
    if out.exists():
        raise RuntimeError("Choose a new successor directory")
    if sha256(Path(original.__file__)) != ORIGINAL_SHA:
        raise RuntimeError("Original generator changed")
    source = BASE / "symbol_paraphrase_completion1"
    parent = inference.verify(source)
    symbol_map = load(BASE / "SYMBOL_MAP.json")
    prompts, assignments = original.build_manifests(symbol_map)
    if (prompts != diagnostic.jsonl(BASE / "private/PROMPTS.jsonl")
            or assignments != diagnostic.jsonl(BASE / "private/ASSIGNMENTS.jsonl")):
        raise RuntimeError("Original scene or assignment replay differs")
    out.mkdir(parents=True)
    (out / "private").mkdir(mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    for folder in ("code", "spacy_model"):
        shutil.copytree(source / "private" / folder, out / "private" / folder)
    for name in ("BANK.json", "propbank_verbs.json"):
        shutil.copyfile(source / "private" / name, out / "private" / name)
    sources = ["scripts/run_symbol_prompt_completion1.py", "scripts/symbol_prompt_completion1_budget.py",
               "scripts/test_symbol_prompt_completion1.py", "scripts/run_semantic_sequence_stage1b.py",
               "scripts/swan_final1_common.py", "scripts/run_research_completion_queue.py"]
    for relative in sources:
        dst = out / "private/code" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dst)
    original.jsonl(out / "private/PROMPTS.jsonl", prompts)
    original.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl", assignments)
    contracts = template_contracts(load(out / "private/BANK.json"), parent["pool"])
    freeze(out / "TEMPLATE_CONTRACTS.json", contracts)
    order = []
    for doc in range(24):
        for slot in list(range(8)) + ["paraphrase"]:
            arms = list(ARMS)
            random.Random(f"prompt-completion1/{doc}/{slot}").shuffle(arms)
            order.extend({"doc_id": doc, "slot": slot, "arm": arm} for arm in arms)
    freeze(out / "private/DISPATCH_ORDER.json", order)
    protocol = """# 原24模板同来源提示配对补实验

2026-11-02。24原场景、原12/12分组、原192计划位置逐项重放一致。
两臂均写入同一目标：旧提示逐字调用旧生成函数；修订臂同时修正X、姓名、
语气和角色解释。估计整体提示配方及其上下文累积效应，不拆称单个提示组件因果效应。
每文逐槽交错两臂，槽内顺序预冻结；各臂只读自己的历史文本。
384载体+48整段复述，最多432请求，模型参数保持原1600token/.9/top_p1。
每位置第一次响应保留，空文本、截断、不合长度均保留，不看图或读出分数补样。
零HTTP重试；未确定消费不重复调用。费用保护10美元，历史峰值费率
输入0.30/输出1.20美元每百万token作保守核算，不宣称实时账单价格。
请求上限32768字节加1024token预留；每次先持久预留，再发送。
预算或请求长度上限暂停，不改输入；显存/磁盘压力保存断点恢复。

24模板均有机械结构合同，全部保留，包括polarity X歧义。没有逐模板真人源图
解释时仅运行探索性提示/解析/计划身份诊断，不称生成语义实现率或修复成功。
先封存全部生成，再制作不带目标的原句槽流及spaCy句界复述流。
冻结FP32 BART/官方归一化/S2Match，4束/1024原生边界，逐句失败算擦除。
原句身份采用已知生成槽，只是诊断；复述无人工映射时不报符号身份准确率。
盲分句原文和复述均在相同2520等频候选上计算精确DP hard/soft/无序成本。
全部预测和成本先封存，后读计划符号。顺序排名不是完整消息恢复成功。
报告12/12已暴露分组和24来源配对差，按来源bootstrap 10000次、seed20261102；
没有显著性筛选、不挑子集、不调阈值，不报告正式FPR/TPR。
此对照尚未加入新语料embedding读出；不得据此主张AMR优于其他表示。
真人两人独立核验及分歧裁决另行回收；自动结果与人评完成状态分开。
"""
    (out / "PROTOCOL.md").write_text(protocol)
    files = {str(p.relative_to(out)): sha256(p) for p in sorted((out / "private").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    files["TEMPLATE_CONTRACTS.json"] = sha256(out / "TEMPLATE_CONTRACTS.json")
    plan = {k: parent[k] for k in ("parser", "scorer", "pool", "groups", "template_to_symbol", "split")}
    plan.update({
        "version": "paired-original-dictionary-prompt/v1", "files": files,
        "protocol_sha256": sha256(out / "PROTOCOL.md"),
        "parent_freeze_sha256": sha256(source / "FREEZE.json"),
        "original_generator_sha256": ORIGINAL_SHA, "parameters": budget.PARAMETERS,
        "resource_gate": {"vram_fraction_max": .8},
        "execution": {"resumable_gpu_pressure": True, "single_parser": True},
        "calls": 432, "max_nanodollars": budget.MONEY_CAP, "temperature": .1,
        "human_truth_available": False,
    })
    freeze(out / "FREEZE.json", plan)
    freeze(out / "VALIDATION.json", {
        "scenes_and_slots_identical_to_original": True, "templates": len(contracts),
        "planned_positions_per_arm": 192, "requests": len(order),
        "DP_oracle": diagnostic.oracle_validation(equal_composition_sequences()),
        "CUDA_or_API_calls": 0,
    })


def verify(out):
    plan = inference.verify(out)
    for relative, expected in plan["files"].items():
        if relative.startswith("private/code/"):
            if sha256(ROOT / relative[len("private/code/"):]) != expected:
                raise RuntimeError("Executing dependency differs: " + relative)
    return plan


def call_id(item):
    return f"d{item['doc_id']:02d}-{item['arm']}-{item['slot']}"


def read_credential(path):
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("Credential must not be group/world readable")
    keys = [s.strip() for s in path.read_text().splitlines()
            if re.fullmatch(r"sk-[A-Za-z0-9_-]+", s.strip())]
    if len(keys) != 1:
        raise ValueError("Expected one configured credential")
    return keys[0]


def generate(out, key_file):
    verify(out)
    contract = sha256(out / "FREEZE.json")
    journal = budget.Journal(out, contract)
    if (out / "GENERATION_COMPLETE.json").exists():
        verify_generation(out)
        return
    import requests
    key = read_credential(key_file)

    def send(body):
        r = requests.post(original.API_URL, data=json.dumps(body, ensure_ascii=True).encode(),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                          timeout=(15, 180), allow_redirects=False)
        if r.status_code != 200:
            raise budget.IntegrityError("provider_HTTP_failure")
        return r.json()

    journal.send = send
    prompts = diagnostic.jsonl(out / "private/PROMPTS.jsonl")
    assignments = diagnostic.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl")
    bank = load(out / "private/BANK.json")
    retained = {(doc, arm): [] for doc in range(24) for arm in ARMS}
    paths = {}
    for i, item in enumerate(load(out / "private/DISPATCH_ORDER.json")):
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise runtime.ResourcePause("disk_headroom_below_512MiB")
        doc, arm, slot = item["doc_id"], item["arm"], item["slot"]
        prior = " ".join(retained[doc, arm])
        request = original.paraphrase_messages(prior) if slot == "paraphrase" else messages(
            prompts[doc]["prompt"], prior, slot,
            bank[assignments[doc]["slots"][slot]["intended_template"]]["normalized_template_amr"], arm)
        record = journal.request(call_id(item), request)
        relative = "private/completions/" + call_id(item) + ".json"
        freeze(out / relative, {**item, **record, "contract_sha256": contract,
                               "surface": original.surface_audit(record["text"])})
        paths[relative] = sha256(out / relative)
        if slot != "paraphrase":
            retained[doc, arm].append(record["text"])
        runtime.atomic(out / "HEARTBEAT.json", {
            "stage": "generating", "completed": i + 1, "total": 432,
            "budget": journal.totals(), "at": runtime.now()})
    freeze(out / "GENERATION_COMPLETE.json", {
        "contract_sha256": contract, "files": paths, "budget": journal.totals(),
        "all_generation_before_detector": True, "independent_human_rows": 0,
    })
    verify_generation(out)


def verify_generation(out):
    seal = load(out / "GENERATION_COMPLETE.json")
    contract = sha256(out / "FREEZE.json")
    expected = {"private/completions/" + call_id(x) + ".json"
                for x in load(out / "private/DISPATCH_ORDER.json")}
    if seal["contract_sha256"] != contract or set(seal["files"]) != expected or len(expected) != 432:
        raise RuntimeError("Generation identity or denominator changed")
    journal = budget.Journal(out, contract)
    if journal.totals() != seal["budget"] or journal.totals()["calls"] != 432:
        raise RuntimeError("Generation accounting changed")
    for relative, expected_sha in seal["files"].items():
        if sha256(out / relative) != expected_sha:
            raise RuntimeError("Generation output changed")
        r = load(out / relative)
        if r["contract_sha256"] != contract:
            raise RuntimeError("Completion contract changed")
        cid = call_id(r)
        payload = load(out / "private/calls" / (cid + ".json"))
        parsed = budget.parse(payload, journal.calls[cid]["intent"]["allowance"])
        if any(r[k] != v for k, v in parsed.items()):
            raise RuntimeError("Completion differs from provider ledger")
    return seal


def materialize(out):
    """Only text, stream labels and document IDs go to the inference input."""
    import spacy
    verify_generation(out)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    streams = []
    for doc in range(24):
        for arm in ARMS:
            slots = [load(out / f"private/completions/d{doc:02d}-{arm}-{slot}.json")["text"]
                     for slot in range(8)]
            para = load(out / f"private/completions/d{doc:02d}-{arm}-paraphrase.json")["text"]
            for condition, units in (
                    ("slots", [{"text": text} for text in slots]),
                    ("clean", diagnostic.segment(" ".join(slots), nlp)),
                    ("paraphrase", diagnostic.segment(para, nlp))):
                for unit in units:
                    unit["text"] = diagnostic.normalize(unit["text"])
                    unit["text_sha256"] = runtime.digest(unit["text"])
                streams.append({"doc_id": doc, "arm": arm + "/" + condition, "units": units})
    freeze(out / "private/INPUTS.json", streams)
    freeze(out / "INPUT_SEAL.json", {
        "generation_sha256": sha256(out / "GENERATION_COMPLETE.json"),
        "inputs_sha256": sha256(out / "private/INPUTS.json"), "spacy": spacy.__version__,
        "streams": len(streams), "slot_occurrences": 384,
        "sentence_occurrences": sum(len(s["units"]) for s in streams),
    })


def review_packet(out):
    verify_generation(out)
    assignments = diagnostic.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl")
    bank = load(out / "private/BANK.json")
    records, linkage = [], {}
    for doc in range(24):
        for arm in ARMS:
            para = load(out / f"private/completions/d{doc:02d}-{arm}-paraphrase.json")["text"]
            for slot in range(8):
                item_id = runtime.digest(f"prompt-human/{doc}/{arm}/{slot}")[:20]
                template = assignments[doc]["slots"][slot]["intended_template"]
                records.append({
                    "item_id": item_id, "template_id": template,
                    "target_graph": bank[template]["normalized_template_amr"],
                    "original_text": load(out / f"private/completions/d{doc:02d}-{arm}-{slot}.json")["text"],
                    "paraphrase_paragraph": para,
                })
                linkage[item_id] = {"doc_id": doc, "arm": arm, "slot": slot}
    records.sort(key=lambda r: r["item_id"])
    freeze(out / "private/HUMAN_LINKAGE.json", linkage)
    folder = out / "private/human_review"
    freeze(folder / "PACKET.json", {
        "items": records, "arm_and_scores_hidden": True, "source_graph_truth_available": False,
        "instructions": "Interpret template semantics independently first; unresolved X roles must be uncertain. "
                        "Judge text realization only when target meaning is defined. "
                        "For paraphrases locate the same event; allow split/merge/reorder or absent events. "
                        "Character spans use Python start-inclusive/end-exclusive offsets.",
        "template_contracts": load(out / "TEMPLATE_CONTRACTS.json"),
    })
    for reviewer in ("reviewer1", "reviewer2"):
        path = folder / (reviewer + ".csv")
        if not path.exists():
            with path.open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "item_id", "reviewer_id", "is_human", "target_meaning_defined",
                    "semantic_realization", "grammar_pass", "paraphrase_event_preserved",
                    "paraphrase_spans_json", "notes"])
                writer.writeheader()
                writer.writerows({"item_id": r["item_id"]} for r in records)


def collect(out):
    """No supplied answer is inferred from model scores or planned identities."""
    review_packet(out)
    folder = out / "private/human_review"
    items = {r["item_id"]: r for r in load(folder / "PACKET.json")["items"]}
    fields = ("target_meaning_defined", "semantic_realization", "grammar_pass", "paraphrase_event_preserved")
    received, people = [], []
    for name in ("reviewer1", "reviewer2"):
        accepted, ids, seen = {}, set(), set()
        with (folder / (name + ".csv")).open(newline="") as stream:
            for r in csv.DictReader(stream):
                key = r["item_id"]
                if key not in items or key in seen:
                    raise ValueError("Review item duplicated or outside frozen packet")
                seen.add(key)
                if not any(r[k].strip() for k in fields + ("reviewer_id", "is_human")):
                    continue
                if r["is_human"].lower() != "true" or not r["reviewer_id"].strip():
                    raise ValueError("Actual identified human required")
                if any(r[k] not in ("yes", "no", "uncertain") for k in fields):
                    raise ValueError("Incomplete human fields")
                if r["target_meaning_defined"] != "yes" and r["semantic_realization"] != "uncertain":
                    raise ValueError("Undefined target cannot receive semantic truth")
                spans = json.loads(r["paraphrase_spans_json"] or "[]")
                if not isinstance(spans, list) or any(
                    not isinstance(s, list) or len(s) != 2 or any(type(i) is not int for i in s)
                    or not 0 <= s[0] < s[1] <= len(items[key]["paraphrase_paragraph"]) for s in spans):
                    raise ValueError("Invalid event evidence span")
                if r["paraphrase_event_preserved"] == "yes" and not spans:
                    raise ValueError("Preserved event needs evidence")
                ids.add(r["reviewer_id"].strip())
                accepted[key] = {**{k: r[k] for k in fields}, "spans": spans}
            if len(ids) > 1:
                raise ValueError("One reviewer per independent file")
        received.append(accepted)
        people.append(ids)
    if people[0] & people[1]:
        raise ValueError("Two distinct reviewers required")
    common = set(received[0]) & set(received[1])
    agreed = {k for k in common if received[0][k] == received[1][k]}
    complete = len(agreed) == 384 and all(received[0][k]["target_meaning_defined"] == "yes" for k in agreed)
    runtime.atomic(out / "HUMAN_COLLECTION_AUDIT.json", {
        "valid_submissions": [len(x) for x in received], "agreed": len(agreed),
        "requires_adjudication": len(common - agreed), "planned": 384,
        "human_semantic_collection_complete": complete,
        "full_research_closure": False,
        "answer_sha256": {name: sha256(folder / (name + ".csv")) for name in ("reviewer1", "reviewer2")},
    })


def paired_summary(pairs):
    delta = np.asarray([p["revised"] - p["old"] for p in pairs])
    rng = np.random.default_rng(20261102)
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"pairs": len(pairs), "old_mean": float(np.mean([p["old"] for p in pairs])),
            "revised_mean": float(np.mean([p["revised"] for p in pairs])),
            "mean_difference": float(delta.mean()),
            "bootstrap95_by_source": np.quantile(samples, [.025, .975]).tolist()}


class Parser(runtime.Parser):
    def fresh(self, text):
        length = len(self.model.tokenizer(text, truncation=False)["input_ids"])
        if not text.strip() or length >= self.model.max_sent_len:
            return {"amr": None, "amr_sha256": None, "valid": False,
                    "text_sha256": runtime.digest(text), "token_length": length,
                    "events": [], "attempts": [], "decode_events": [],
                    "beam_lengths": [], "clipped": length >= self.model.max_sent_len,
                    "seconds": 0, "freeze_sha256": self.freeze_sha,
                    "erasure_reason": "empty_text" if not text.strip() else "native_input_limit"}
        return super().fresh(text)


def score(out, assets):
    f = verify(out)
    input_seal = load(out / "INPUT_SEAL.json")
    if input_seal["inputs_sha256"] != sha256(out / "private/INPUTS.json"):
        raise RuntimeError("Scoring input changed")
    if (out / "PREDICTION_SEAL.json").exists():
        inference.verify_outputs(out, load(out / "PREDICTION_SEAL.json"))
        return
    import torch
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    if not torch.cuda.is_available():
        raise runtime.ResourcePause("CUDA_unavailable")
    inputs = load(out / "private/INPUTS.json")
    texts = list(dict.fromkeys(u["text"] for s in inputs for u in s["units"]))
    parser, scorer = Parser(assets, out, f), runtime.Scorer(assets, out)
    if sha256(scorer.glove) != f["scorer"]["glove_sha256"]:
        raise RuntimeError("Scorer vector identity changed")
    replays = []
    for text in texts[:2]:
        a, b = parser.fresh(text), parser.fresh(text)
        key = lambda r: (r["amr_sha256"], diagnostic.eligible_parse(r, parser.freeze_sha))
        if key(a) != key(b):
            raise RuntimeError("Parser double replay differs")
        path = out / "private/parses" / (runtime.digest(text) + ".json")
        if path.exists():
            if key(load(path)) != key(a):
                raise RuntimeError("Resumed parser differs")
        else:
            runtime.atomic(path, a)
        first = inference.graph_scores(parser, scorer, text, f)
        if first != inference.graph_scores(parser, scorer, text, f):
            raise RuntimeError("Dictionary double replay differs")
        replays.append({"text_sha256": runtime.digest(text), "exact_replay": True})
    runtime.atomic(out / "GPU_PRECHECK.json", {
        "passed": True, "replays": replays, "dtype": "float32",
        "gpu": torch.cuda.get_device_name(0), "freeze_sha256": sha256(out / "FREEZE.json")})
    for i, text in enumerate(texts):
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise runtime.ResourcePause("disk_headroom_below_512MiB")
        parse_path = "private/parses/" + runtime.digest(text) + ".json"
        path = out / "private/predictions" / (runtime.digest(text) + ".json")
        if path.exists():
            r = load(path)
            if (r["freeze_sha256"] != parser.freeze_sha or r["text_sha256"] != runtime.digest(text)
                    or r["parse_path"] != parse_path or r["parse_sha256"] != sha256(out / parse_path)):
                raise RuntimeError("Prediction resume identity changed")
            continue
        q, reason = inference.graph_scores(parser, scorer, text, f)
        freeze(path, {"text_sha256": runtime.digest(text), "q": q, "reason": reason,
                      "freeze_sha256": parser.freeze_sha, "parse_path": parse_path,
                      "parse_sha256": sha256(out / parse_path)})
        runtime.atomic(out / "HEARTBEAT.json", {
            "stage": "scoring", "completed": i + 1, "total": len(texts), "at": runtime.now()})
        runtime.atomic(out / "private/SCORING_DIAGNOSTICS.json", scorer.events)
    candidates, costs, keys = equal_composition_sequences(), [], []
    for stream in inputs:
        qs = [load(out / "private/predictions" / (u["text_sha256"] + ".json"))["q"]
              for u in stream["units"]]
        for mode in ("hard", "soft", "unordered"):
            values = unordered_costs(qs, candidates) if mode == "unordered" else exact_costs(qs, candidates, mode)
            if mode == "unordered" and np.ptp(values) != 0:
                raise RuntimeError("Equal-composition unordered costs vary")
            costs.append(values)
            keys.append([stream["doc_id"], stream["arm"], mode])
    np.savez_compressed(out / "private/ORDER_COSTS.npz", costs=np.stack(costs), candidates=candidates)
    freeze(out / "private/ORDER_KEYS.json", keys)
    paths = [out / "private/ORDER_KEYS.json", out / "private/ORDER_COSTS.npz",
             out / "INPUT_SEAL.json", out / "private/INPUTS.json"]
    paths += list((out / "private/parses").glob("*.json"))
    paths += list((out / "private/predictions").glob("*.json"))
    freeze(out / "PREDICTION_SEAL.json", {
        "freeze_sha256": sha256(out / "FREEZE.json"),
        "files": {str(p.relative_to(out)): sha256(p) for p in sorted(paths)},
        "reference_access_before_seal": False})


def evaluate(out):
    f = verify(out)
    verify_generation(out)
    inference.verify_outputs(out, load(out / "PREDICTION_SEAL.json"))
    assignments = {x["doc_id"]: x for x in diagnostic.jsonl(out / "private/EVALUATOR_ASSIGNMENTS.jsonl")}
    streams = load(out / "private/INPUTS.json")
    identities, paired = {}, {}
    for arm in ARMS:
        rows = []
        for s in streams:
            if s["arm"] != arm + "/slots":
                continue
            if len(s["units"]) != 8:
                raise RuntimeError("Slot denominator differs")
            for slot, unit in enumerate(s["units"]):
                pred = load(out / "private/predictions" / (unit["text_sha256"] + ".json"))
                rows.append({"doc_id": s["doc_id"], "slot": slot, "q": pred["q"], "reason": pred["reason"],
                             "target": f["template_to_symbol"][str(assignments[s["doc_id"]]["slots"][slot]["intended_template"])]})
        if len(rows) != 192:
            raise RuntimeError("All 192 positions required per arm")
        identities[arm] = {split: diagnostic.identity([r for r in rows if r["doc_id"] in f["split"][split]])
                           for split in ("development", "readout")}
        for doc in range(24):
            paired.setdefault(doc, {})[arm] = diagnostic.identity([r for r in rows if r["doc_id"] == doc])["balanced_accuracy"]
    arrays = np.load(out / "private/ORDER_COSTS.npz", allow_pickle=False)
    candidates = equal_composition_sequences()
    if not np.array_equal(arrays["candidates"], candidates):
        raise RuntimeError("Order candidate set differs")
    keys = load(out / "private/ORDER_KEYS.json")
    expected_keys = {(doc, arm + "/" + condition, mode) for doc in range(24) for arm in ARMS
                     for condition in ("slots", "clean", "paraphrase") for mode in ("hard", "soft", "unordered")}
    if set(map(tuple, keys)) != expected_keys or arrays["costs"].shape != (432, 2520):
        raise RuntimeError("Order condition denominator differs")
    orders = {}
    for (doc, label, mode), costs in zip(keys, arrays["costs"]):
        arm, condition = label.split("/")
        truth = [s["intended_state"] for s in assignments[doc]["slots"]]
        value = rank_truth(costs, candidates, truth)["pairwise_win_rate"]
        orders.setdefault(condition + "/" + mode, {}).setdefault(doc, {})[arm] = value
    surface = {arm: Counter() for arm in ARMS}
    for item in load(out / "private/DISPATCH_ORDER.json"):
        if item["slot"] == "paraphrase":
            continue
        r = load(out / "private/completions" / (call_id(item) + ".json"))
        surface[item["arm"]].update({k: int(r["surface"][k]) for k in (
            "nonempty", "target_length_met", "single_sentence_heuristic", "placeholder_literal")})
    result = {
        "status": "COMPUTATION_COMPLETE_HUMAN_SEMANTICS_PENDING",
        "identity_known_slot_planned_target": identities,
        "paired_identity_difference": paired_summary(list(paired.values())),
        "paired_equal_frequency_order": {k: paired_summary(list(v.values())) for k, v in orders.items()},
        "surface_diagnostics": {k: dict(v) for k, v in surface.items()},
        "source_pairs": 24, "positions_per_arm": 192,
        "paraphrase_symbol_identity": None, "independent_human_rows": 0,
        "semantic_realization_rate": None, "formal_FPR": None, "formal_TPR": None,
        "budget": load(out / "GENERATION_COMPLETE.json")["budget"],
        "provider_metadata": dict(Counter(
            str(load(out / "private/completions" / (call_id(x) + ".json"))["fingerprint"])
            for x in load(out / "private/DISPATCH_ORDER.json"))),
    }
    freeze(out / "RESULT.json", result)
    freeze(out / "AUDIT.json", {
        "all_24_templates_retained": True, "same_original_scenes_and_slots": True,
        "all_432_first_responses_retained": True, "prediction_seal_before_reference": True,
        "order_candidates_equal_frequency_2520": True,
        "no_human_semantic_claim": True, "full_research_closure": False,
        "result_sha256": sha256(out / "RESULT.json"),
    })
    freeze(out / "TERMINAL.json", {
        "status": "COMPUTATION_COMPLETE", "result_sha256": sha256(out / "RESULT.json"),
        "audit_sha256": sha256(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "generate", "score", "evaluate", "collect"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--assets", type=Path, default=ROOT)
    ap.add_argument("--key-file", type=Path)
    args = ap.parse_args()
    os.umask(0o077)
    if args.action == "prepare":
        prepare(args.out)
        print(json.dumps(load(args.out / "VALIDATION.json")))
        return
    verify(args.out)
    if args.action == "verify":
        print(json.dumps({"frozen_dependencies_verified": True}))
        return
    with (args.out / "private/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.out / "HALT.json").exists():
            raise RuntimeError("Integrity halt remains immutable")
        started = time.monotonic()
        try:
            if args.action == "generate":
                generate(args.out, args.key_file)
                collect(args.out)
            elif args.action == "collect":
                collect(args.out)
            else:
                if not (args.out / "GENERATION_COMPLETE.json").exists():
                    raise budget.BudgetPause("generation_incomplete")
                materialize(args.out)
                collect(args.out)
                if args.action == "score":
                    score(args.out, args.assets)
                evaluate(args.out)
        except Exception as exc:
            resource = isinstance(exc, runtime.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
            paused = resource or isinstance(exc, budget.BudgetPause)
            runtime.atomic(args.out / ("PAUSED.json" if paused else "HALT.json"), {
                "type": type(exc).__name__, "at": runtime.now(), "scientific_result": None,
                "resume_allowed": paused})
            if paused:
                sys.exit(75 if resource else 76)
            raise
        finally:
            runtime.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                           {"elapsed_seconds": time.monotonic() - started, "at": runtime.now()})


if __name__ == "__main__":
    main()
