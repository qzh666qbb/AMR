"""Freeze and run new-text SWAN diagnostics and conditional blind transfer.

No provider calls, fitting, injected-template lookup, or legacy feature-cache reads.
Artifacts containing text/graphs/checkpoints remain in the experiment's private/.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/semantic_sequence_stage1_run2"
VERSION = "new-text-multiscale-transfer/v3-exact-workload"
PARSER_OUTPUT_VERSION = "accepted-graph/v2-exact-metadata"


class ResourcePause(RuntimeError):
    """A resumable resource shortage, never an experimental outcome."""


def gpu_capacity(out, freeze, phase, additional_mib):
    if not freeze.get("execution", {}).get("resumable_gpu_pressure"):
        return
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    used, total = map(float, result.stdout.strip().split(","))
    limit = freeze["resource_gate"]["vram_fraction_max"] * total
    snapshot = {"at": now(), "phase": phase, "used_mib": used, "total_mib": total,
                "additional_reservation_mib": additional_mib, "limit_mib": limit,
                "pid": os.getpid(), "admitted": used + additional_mib < limit}
    atomic(out / "GPU_RESOURCE_STATE.json", snapshot)
    if not snapshot["admitted"]:
        raise ResourcePause("GPU capacity unavailable; checkpoint and release this worker")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def isolate(path, names, namespace):
    nodes = [node for node in ast.parse(Path(path).read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Source isolation failed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def edited_text(arm, condition):
    if condition in ("clean", "paraphrase"):
        return arm["text" if condition == "clean" else "paraphrase"]
    pieces = [row["text"] for row in arm["slots"]]
    for i, text in enumerate(pieces):
        if condition == "split":
            tokens = list(re.finditer(r"\S+", text))
            if len(tokens) < 2:
                raise ValueError("Frozen split operation has fewer than two words")
            at = tokens[len(tokens) // 2 - 1].end()
            if text[:at].endswith((".", "!", "?")):
                raise ValueError("Frozen split operation hits an existing boundary")
            pieces[i] = text[:at] + "." + text[at:]
        elif condition == "merge" and i % 2 == 0 or condition in ("comma", "space"):
            replacement = "," if condition in ("merge", "comma") else ""
            pieces[i], count = re.subn(r"[.!?]+(?=[\"')\]]*\s*$)", replacement, text.strip())
            if count != 1:
                raise ValueError("Frozen boundary operation lacks terminal punctuation")
        elif condition != "merge":
            raise ValueError("Unknown condition")
    return " ".join(pieces)


def unique_parse_inputs(documents, conditions):
    """Count actual text-key cache work, without parser calls or score access."""
    from src.core.multiscale_blind import blind_geometry
    return {w["text"] for doc in documents for arm in doc["arms"].values()
            for condition in conditions for rows in blind_geometry(arm["texts"][condition]).values()
            for w in rows}


def prepare(root, out, prior_gpu_seconds=0.0):
    import spacy
    sys.path.insert(0, str(root))
    from src.core.multiscale_blind import blind_geometry

    if out.exists():
        raise ValueError("Use a new output directory; freezes cannot be overwritten")
    out.mkdir(parents=True)
    (out / ".gitignore").write_text("private/\n*.log\n")
    private = out / "private"
    code = private / "code"
    run = root / BASE
    assignments = [json.loads(s) for s in
                   (run / "private/ASSIGNMENTS.jsonl").read_text().splitlines()]
    dev = [a["doc_id"] for a in sorted(assignments, key=lambda a: a["half_index"])
           if a["split"] == "development"]
    test = [a["doc_id"] for a in sorted(assignments, key=lambda a: a["half_index"])
            if a["split"] == "readout"]
    assert len(dev) == len(test) == 12 and not set(dev) & set(test)
    assert len(assignments) == 24
    sources = [
        "scripts/run_multiscale_transfer.py", "src/core/multiscale_blind.py",
        "src/core/multiscale_fusion.py", "src/core/parser_recovery_audit.py",
        "src/core/bank_scorer.py",
        "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py",
        "third_party/swan_s2match_runtime_711a231d/s2match.py",
        "third_party/swan_s2match_runtime_711a231d/amr_py3.py",
    ]
    for path in sources:
        destination = code / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / path, destination)
    for directory in (code / "src", code / "src/core"):
        (directory / "__init__.py").write_text("")
    runtime = root / "third_party/swan_s2match_runtime_711a231d"
    for path in runtime.glob("*.py"):
        shutil.copy2(path, code / "third_party/swan_s2match_runtime_711a231d" / path.name)
    checkpoints = []
    models = root / "experiments/swan_multiscale_attention_development1/validation1/runs"
    names = [f"{m}_20261023" for m in ("raw_s6", "s6", "mean")]
    names += [f"{m}_{s}" for m in ("gate", "attention") for s in range(20261023, 20261028)]
    for name in names:
        destination = private / "models" / (name + ".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(models / name / "MODEL.pt", destination)
        checkpoints.append({"name": name, "file": str(destination.relative_to(out)),
                            "sha256": sha(destination)})
    bank_source = root / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
    shutil.copy2(bank_source, private / "BANK.json")
    bank = read(private / "BANK.json")
    assert len(bank) == 100
    shutil.copy2(run / "alignment_review1/private/propbank_verbs.json",
                 private / "propbank_verbs.json")
    normalizer = spacy.load("en_core_web_sm")
    shutil.copytree(normalizer.path, private / "spacy_model")
    documents, provenance, counts = [], [], {}
    for docid in dev + test:
        path = run / f"private/documents/d{docid:02d}.json"
        d = read(path)
        assert d["doc_id"] == docid
        arms = {}
        for armname in ("positive", "unwatermarked"):
            arm = d["arms"][armname]
            assert len(arm["slots"]) == 8
            assert [s["slot"] for s in arm["slots"]] == list(range(8))
            assert digest(arm["text"]) == arm["text_sha256"]
            assert digest(arm["paraphrase"]) == arm["paraphrase_sha256"]
            carriers, cursor = [], 0
            for slot in arm["slots"]:
                start = arm["text"].find(slot["text"], cursor)
                assert start >= cursor
                end = start + len(slot["text"])
                carriers.append({"start": start, "end": end, "text": slot["text"],
                                 "slot": slot["slot"]})
                cursor = end
            texts = {c: edited_text(arm, c) for c in
                     ("clean", "paraphrase", "split", "merge", "comma", "space")}
            arms[armname] = {"texts": texts, "carriers": carriers,
                             "text_sha256": {c: digest(t) for c, t in texts.items()}}
            for condition, text in texts.items():
                geometry = blind_geometry(text)
                unique = {w["text"] for scale in geometry.values() for w in scale}
                if condition == "clean":
                    unique.update(c["text"] for c in carriers)
                counts[f"d{docid:02d}/{armname}/{condition}"] = {
                    "S6": len(geometry["S6"]), "S12": len(geometry["S12"]),
                    "unique_parse_inputs": len(unique), "words": len(text.split())}
        documents.append({"doc_id": docid, "split": d["split"], "arms": arms})
        provenance.append({"file": str(path.relative_to(root)), "sha256": sha(path)})
    atomic(private / "DOCUMENTS.json", documents)
    atomic(out / "INPUT_INVENTORY.json", counts)
    sentences = {c["text"] for d in documents for a in d["arms"].values() for c in a["carriers"]}
    pre_docs = [d for d in documents if d["doc_id"] in dev[:2]]
    all_conditions = ["clean", "paraphrase", "split", "merge", "comma", "space"]
    full_inputs = unique_parse_inputs(documents, all_conditions)
    clean_inputs = unique_parse_inputs(documents, ["clean"])
    pre_inputs = unique_parse_inputs(pre_docs, ["clean"]) - sentences
    workload = {
        "basis": "Exact unique text keys; already completed original-sentence keys excluded from feature work",
        "original_sentence_unique_inputs": len(sentences),
        "unique_all_parse_inputs": len(full_inputs | sentences),
        "unique_clean_parse_inputs": len(clean_inputs | sentences),
        "feature_inputs_after_signal": {
            "all_conditions": len(full_inputs - sentences), "clean_only": len(clean_inputs - sentences)},
        "precheck_feature_inputs_after_signal": len(pre_inputs),
        "all_parse_key_manifest_sha256": digest("\n".join(sorted(digest(t) for t in full_inputs | sentences))),
        "precheck_parse_key_manifest_sha256": digest("\n".join(sorted(digest(t) for t in pre_inputs))),
        "no_score_access": True}
    atomic(out / "WORKLOAD.json", workload)
    parser_assets = read(run / "alignment_review1/FREEZE.json")["parser"]["inference_assets"]
    frozen = {
        "version": VERSION, "frozen_at": now(),
        "scope": "24-pair exploratory diagnostics; no original Stage1B gate rewrite",
        "split": {"calibration": dev[:6], "signal_decision": dev[6:],
                  "development": dev, "readout": test, "precheck": dev[:2]},
        "input_provenance": provenance,
        "documents_sha256": sha(private / "DOCUMENTS.json"),
        "original_generation_manifest_sha256": sha(run / "MANIFEST_GENERATION.sha256"),
        "checkpoints": checkpoints,
        "bank_sha256": sha(private / "BANK.json"),
        "bank_order": "Original JSON row order, zero-based 0..99; all columns retained",
        "sources": {str(p.relative_to(out)): sha(p) for p in sorted(code.rglob("*")) if p.is_file()},
        "normalizer_assets": {str(p.relative_to(out)): sha(p)
                              for p in sorted((private / "spacy_model").rglob("*")) if p.is_file()},
        "propbank_verbs_sha256": sha(private / "propbank_verbs.json"),
        "parser": {"assets": parser_assets, "dtype": "float32", "tf32": False,
                   "batch_size": 1, "num_beams": 4, "input_limit": 1024, "output_limit": 1024,
                   "invalid_or_bound_error": "Exclude span scores, retain missing features and denominators",
                   "unbound_warning": "Permanent halt; same manifest cannot resume"},
        "scorer": {"baseline_normalization": "Official AST-isolated classifier/template functions",
                   "fusion_normalization": "Frozen project concept classes used by old models",
                   "cosine": True, "cutoff": 0.5, "diffsense": 0.5, "mwp": "split",
                   "do_not_mark_quotes": False, "hill_climb_iterations": 5,
                   "rng": "Per normalized graph/template SHA seed; no-argument reseeding is a no-op",
                   "event_extractor": "Deterministic outgoing BFS depth3/node12; no dangling edges",
                   "old_cache_reuse": False, "glove_sha256": sha(root / "vectors/glove.6B.100d.txt")},
        "features": {"S6": "6 whitespace words, stride1, complete tail",
                     "S12": "12 whitespace words, stride6, complete tail",
                     "Ssub": "Max project-normalized event scores from each S12 candidate",
                     "oracle": "Per carrier/column max over overlapping candidates; diagnostic only",
                     "blind": "S12 anchors; S6 max over geometric overlaps; no slot or target input",
                     "document_score": "Mean anchor statistics; any all-missing anchor makes document abstain"},
        "calibration": {"data": "Only first6 development negative docs, clean condition",
                        "carrier_alpha": [0.01, 0.05], "document_alpha": [0.01, 0.05],
                        "test_tuning": False, "refit_or_refit_normalizer": False,
                        "independent_human_test_fpr": None},
        "clean_gate": {"population": "Last6 development pairs only",
                       "method": "Official-normalized full clean sentence bank-max",
                       "carrier_auroc_min": 0.7, "document_paired_win_min": 0.7,
                       "document_aggregation": "Mean of all8 sentence bank-max values; missing=-inf",
                       "ties": "half win", "readout_ignored": True,
                       "interpretation": "Exploratory conditional decision, not confirmatory significance"},
        "resource_gate": {"precheck_docs": dev[:2], "vram_fraction_max": 0.8,
                          "projected_gpu_hours_max": 12,
                          "wall_hours_max": 24, "prior_gpu_seconds": prior_gpu_seconds,
                          "projection": "Signal-only pilot then exact unique feature-text workload, factor1.5; include prior attempts",
                          "workload_sha256": sha(out / "WORKLOAD.json")},
        "stage_order": ["signal_all_clean_sentences", "features_all_clean",
                        "frozen_model_clean_evaluation", "conditional_blind_attacks"],
        "cache_scope": "Fresh directory only; within-successor parse/score deduplication allowed",
        "conditions": ["clean", "paraphrase", "split", "merge", "comma", "space"],
        "no_generation": True, "no_score_filtering": True,
    }
    atomic(out / "FREEZE.json", frozen)
    (out / "PROTOCOL.md").write_text(
        "# 新语料多尺度迁移诊断：冻结协议\n\n"
        "本轮只使用现有 24 对文本。先检查全部配对 clean 原句的 SWAN bank-max 信号，"
        "再重算 S6、S12 和事件子图对全部 100 模板的分数。所有正例均保留；不按分数筛选。\n\n"
        "旧 raw S6、归一化 S6、等权模型及门控/注意力各 5 个种子的参数、归一化器和银行列顺序"
        "在新评分前冻结。不会训练、选择最好种子或根据读出结果调阈值。\n\n"
        "原开发集 12 对按 half_index 顺序分为前 6 对负例校准和后 6 对信号决策；"
        "原读出集 12 对单独报告。载体校准有 48 个相关负例，仅提供探索性 5% 工作点；"
        "1% 校准不足。文档校准只有 6 篇，1%/5% 工作点均不足，报告 AUROC 和配对胜率。\n\n"
        "clean 信号门只看后 6 个开发文档：官方规范化的原句 bank-max 载体 AUROC ≥ 0.7，"
        "且各 8 原句均分的文档配对胜率 ≥ 0.7。通过后自动扩展复述、split、merge、comma、space；"
        "否则结束攻击阶段并建议修生成通道。这是小样本开发诊断。\n\n"
        "盲评分只接收文本，按空白词枚举完整 S6(6/1) 与 S12(12/6) 窗口并覆盖尾部。"
        "以 S12 窗口为锚，将重叠 S6 最大分、S12 分、其事件子图最大分组成 [N,3,100]。"
        "逐锚应用冻结模型，文档分数取均值。任一锚完全缺失则文档 abstain，并保留在分母。\n\n"
        "已知原句位置仅用于单独的载体诊断，不能替代盲文档指标。负例接受相同确定性句界编辑；"
        "每种条件单独报告假阳性观察。无独立人类测试集，不主张 1% 文档误报率。\n\n"
        "AMR 使用 CUDA FP32、batch1、4 beams；不截断。恢复与 roundtrip 错误绑定到输入后排除其分数。"
        "无法绑定的 warning 永久停止。S2Match 保留 5 次 hill-climb，以图对 SHA 确定随机流；"
        "其无参数 seed() 不再重置为系统时间。官方与项目规范化口径分开记录。"
        "事件子图改为确定顺序、无悬空边的 BFS，旧模型迁移须考虑该输入变化。\n\n"
        "先仅运行前两对开发文档的原句，检查两次 parser/score 重放与 <80% 显存。"
        "全部 384 个原句匹配与独立信号门写入后，才开始三尺度特征。"
        "特征预检只使用前两对开发文档，投影按解析缓存实际使用的唯一文本键计算，"
        "扣除已经完成的原句解析，并保留全部可能条件、1.5 安全系数及旧尝试消耗，"
        "上限 12 GPU 小时和 24 墙钟小时。按文档原子存档、断点恢复并由 Supervisor 托管。\n\n"
        "所有参数、输入、代码、银行、模型与资产 SHA 见 FREEZE.json；文本、图、模型与缓存仅存 private/。\n")
    print(json.dumps({"prepared": True, "pairs": len(documents),
                      "calibration": dev[:6], "signal": dev[6:], "readout": test,
                      "checkpoints": len(checkpoints), "freeze_sha256": sha(out / "FREEZE.json")}))


class PairRandom(random.Random):
    def seed(self, a=None, version=2):
        if a is not None:
            super().seed(a, version)


class BoundLogger:
    def __init__(self):
        self.events = []

    def debug(self, *args):
        pass

    def error(self, *args):
        self.events.append({"level": "ERROR", "message": " ".join(map(str, args))})


class Scorer:
    def __init__(self, root, out):
        import penman
        import spacy
        code = out / "private/code"
        sys.path.insert(0, str(code / "third_party/swan_s2match_runtime_711a231d"))
        import amr_py3
        import s2match
        self.mod, self.penman = s2match, penman
        self.logger = BoundLogger()
        amr_py3.log_helper = self.logger
        s2match.log_helper = self.logger
        self.glove = root / "vectors/glove.6B.100d.txt"
        vectors = s2match.load_vecs(str(self.glove))
        s2match.load_vecs = lambda _: vectors
        namespace = dict(penman=penman, re=re, logger=self.logger,
                         defaultdict=defaultdict, deque=deque,
                         verb_pattern=re.compile(r"(.*)-\d+$"),
                         verb_lemmas=set(read(out / "private/propbank_verbs.json")),
                         nlp=spacy.load(out / "private/spacy_model"))
        isolate(code / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py",
                ["classify_concept", "generate_template_amr", "normalize_amr_variables"], namespace)
        # BoundLogger also captures official normalizer warnings as invalid scores.
        self.logger.warning = self.logger.error
        self.official = namespace["generate_template_amr"]
        self.canonical = namespace["normalize_amr_variables"]
        project = dict(re=re, _TEMPLATE_CONCEPTS={"ADJ", "LOC", "N", "NE", "NUM", "V", "X"},
                       _NUMERIC_CONCEPTS={"date-entity", "monetary-quantity", "ordinal-entity",
                                          "percentage-entity", "temporal-quantity"},
                       _LOCATION_CONCEPTS={"city", "continent", "country", "province", "state"},
                       _SENSE_RE=re.compile(r".+-\d+$"))
        isolate(code / "src/core/bank_scorer.py", ["_concept_class"], project)
        self.classify = project["_concept_class"]
        self.bank = [b["normalized_template_amr"] for b in read(out / "private/BANK.json")]
        self.db = sqlite3.connect(out / "private/scores.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT)")
        self.calls, self.cache_hits = 0, 0
        diagnostics = out / "private/SCORING_DIAGNOSTICS.json"
        self.events = read(diagnostics) if diagnostics.exists() else []
        self.norms = {}

    def normalize(self, amr, kind):
        key = (amr, kind)
        if key not in self.norms:
            clean = "\n".join(s for s in amr.splitlines() if not s.startswith("#"))
            self.logger.events.clear()
            if kind == "official":
                normalized = self.official(clean)
            else:
                graph = self.penman.decode(clean)
                named = {s for s, r, t in graph.triples if r == ":name"}
                triples = [(s, r, self.classify(t, s in named) if r == ":instance"
                            else '"NE"' if t.startswith('"') and t.endswith('"') else t)
                           for s, r, t in graph.triples]
                normalized = self.penman.encode(self.penman.Graph(triples, top=graph.top))
            normalized = self.canonical(normalized)
            if self.logger.events:
                raise ValueError("Bound normalizer diagnostic")
            self.norms[key] = normalized
        return self.norms[key]

    def pair(self, normalized, template):
        self.logger.events.clear()
        self.mod.random = PairRandom(int(digest(normalized + "\0" + template), 16))
        try:
            _, _, value = self.mod.compute_s2match_from_strings(
                normalized, template, vectors_file=str(self.glove),
                similarity_function="cosine", cutoff=0.5, diffsense=0.5,
                mwp="split", do_not_mark_quotes=False)
            if self.logger.events or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("S2Match invalid score or bound error")
            return float(value)
        except Exception as exc:
            self.events.append({"graph_sha256": digest(normalized),
                                "template_sha256": digest(template),
                                "reason": type(exc).__name__,
                                "bound_errors": list(self.logger.events)})
            return None
        finally:
            self.calls += 1

    def scores(self, amr, kind):
        if amr is None:
            return [None] * 100
        try:
            normalized = self.normalize(amr, kind)
        except Exception as exc:
            self.events.append({"graph_sha256": digest(amr), "normalizer": kind,
                                "reason": type(exc).__name__})
            return [None] * 100
        key = digest(normalized)
        prior = self.db.execute("SELECT value FROM scores WHERE key=?", (key,)).fetchone()
        if prior:
            self.cache_hits += 1
            return json.loads(prior[0])
        values = [self.pair(normalized, t) for t in self.bank]
        self.db.execute("INSERT INTO scores VALUES (?,?)", (key, json.dumps(values)))
        self.db.commit()
        return values


class Parser:
    def __init__(self, root, out, freeze):
        import torch
        import amrlib
        import amrlib.models.parse_xfm.inference as inference
        import penman
        from penman.models.amr import model as amr_model
        self.torch, self.inference, self.penman, self.amr_model = torch, inference, penman, amr_model
        self.out, self.freeze_sha = out, sha(out / "FREEZE.json")
        self.capacity_freeze = freeze
        gpu_capacity(out, freeze, "before_model_load", 3072)
        source = out / "private/code/src/core/parser_recovery_audit.py"
        ns = dict(globals(), _LOCK=threading.RLock(), AUDIT_VERSION="amrlib-recovery-audit:v1")
        isolate(source, ["RecoveryHandler", "capture_recovery_logs", "extract_amr_graph"], ns)
        self.capture = ns["capture_recovery_logs"]
        self.extract_graph = ns["extract_amr_graph"]
        assert torch.cuda.is_available()
        torch.set_num_threads(2)
        torch.manual_seed(20261023)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for asset in freeze["parser"]["assets"]:
            assert sha(root / asset["path"]) == asset["sha256"]
        events = []
        with self.capture({"phase": "load", "span_index": None, "beam_index": None}, events):
            self.model = amrlib.load_stog_model(
                model_dir=str(root / "models/model_parse_xfm_bart_large-v0_1_0"),
                tok_name_or_path=str(root / "models/bart-large-tokenizer"),
                device="cuda:0", batch_size=1, num_beams=4)
        if events:
            atomic(out / "private/LOAD_EVENTS.json", events)
            raise RuntimeError("Unbound parser model-load diagnostic")
        assert all(p.dtype == torch.float32 for p in self.model.model.parameters())
        assert self.model.max_sent_len == self.model.max_graph_len == 1024
        self.beam_lengths = []
        original_generate = self.model.model.generate
        def generate(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            for ids in outputs:
                values = ids.tolist()
                while values and values[-1] == self.model.tokenizer.pad_token_id:
                    values.pop()
                self.beam_lengths.append({"length": len(values),
                                         "eos": bool(values) and values[-1] == self.model.tokenizer.eos_token_id})
            return outputs
        self.model.model.generate = generate
        self.calls, self.cache_hits = 0, 0

    def fresh(self, text):
        gpu_capacity(self.out, self.capacity_freeze, "before_parse", 512)
        n_tokens = len(self.model.tokenizer(text, truncation=False)["input_ids"])
        if n_tokens >= self.model.max_sent_len:
            raise RuntimeError("Native parser input limit reached; no truncation")
        self.beam_lengths.clear()
        events, attempts = [], []
        state = {"phase": "generation", "span_index": None, "beam_index": None}
        original = self.inference.PenmanDeSerializer
        beam = 0
        def deserializer(*args, **kwargs):
            nonlocal beam
            before = len(events)
            state.update(phase="deserialize", span_index=0, beam_index=beam)
            try:
                result = original(*args, **kwargs)
                graph = result.get_graph_string()
            finally:
                state.update(phase="generation", span_index=None, beam_index=None)
            attempts.append({"beam_index": beam, "accepted": graph is not None,
                             "raw_sha256": digest(args[0]), "graph_sha256": digest(graph) if graph else None,
                             "events": events[before:]})
            beam += 1
            return result
        start = time.monotonic()
        self.inference.PenmanDeSerializer = deserializer
        try:
            with self.capture(state, events):
                outputs = self.model.parse_sents([text])
        finally:
            self.inference.PenmanDeSerializer = original
        if any(e["span_index"] is None for e in events):
            atomic(self.out / "private/UNBOUND_EVENTS.json", events)
            raise RuntimeError("Unbound parser diagnostic; permanent halt")
        if len(outputs) != 1:
            raise RuntimeError("Parser output count differs from input")
        output = outputs[0]
        amr = self.extract_graph(output, text) if output else output
        accepted = [a for a in attempts if a["accepted"]]
        if amr and len(accepted) != 1:
            raise RuntimeError("Parser output lacks unique accepted beam")
        if amr:
            if digest(amr) != accepted[0]["graph_sha256"]:
                raise RuntimeError("Parser output differs from accepted beam")
        decode_events = []
        valid = bool(amr)
        if amr:
            try:
                with self.capture({"phase": "roundtrip", "span_index": 0, "beam_index": None}, decode_events):
                    graph = self.penman.decode(amr, model=self.amr_model)
                    roundtrip = self.penman.decode(self.penman.encode(graph, model=self.amr_model),
                                                  model=self.amr_model)
                valid = bool(graph.instances()) and graph.top in graph.variables()
                valid &= graph.top == roundtrip.top and sorted(graph.triples) == sorted(roundtrip.triples)
            except Exception as exc:
                decode_events.append({"level": "ERROR", "reason": type(exc).__name__})
                valid = False
        clipped = any(r["length"] >= 1024 and not r["eos"] for r in self.beam_lengths)
        error = any(e["level"] in ("ERROR", "CRITICAL") for e in events + decode_events)
        self.calls += 1
        return {"amr": amr, "amr_sha256": digest(amr) if amr else None,
                "parser_output_version": PARSER_OUTPUT_VERSION,
                "upstream_output_sha256": digest(output) if output else None,
                "valid": bool(valid and not error and not clipped),
                "text_sha256": digest(text), "token_length": n_tokens,
                "events": events, "attempts": attempts, "decode_events": decode_events,
                "beam_lengths": list(self.beam_lengths), "clipped": clipped,
                "seconds": time.monotonic() - start, "freeze_sha256": self.freeze_sha}

    def parse(self, text):
        path = self.out / "private/parses" / (digest(text) + ".json")
        if path.exists():
            result = read(path)
            if result.get("parser_output_version") != PARSER_OUTPUT_VERSION:
                raise RuntimeError("Parser cache format changed; use a new frozen runtime/cache")
            assert result["freeze_sha256"] == self.freeze_sha
            assert result["text_sha256"] == digest(text)
            assert result["amr_sha256"] == (digest(result["amr"]) if result["amr"] else None)
            self.cache_hits += 1
        else:
            result = self.fresh(text)
            atomic(path, result)
        return result["amr"] if result["valid"] else None


def verify(out, freeze):
    assert sha(out / "private/DOCUMENTS.json") == freeze["documents_sha256"]
    assert sha(out / "private/BANK.json") == freeze["bank_sha256"]
    if "workload_sha256" in freeze["resource_gate"]:
        assert sha(out / "WORKLOAD.json") == freeze["resource_gate"]["workload_sha256"]
    for path, value in {**freeze["sources"], **freeze["normalizer_assets"]}.items():
        assert sha(out / path) == value
    for model in freeze["checkpoints"]:
        assert sha(out / model["file"]) == model["sha256"]
    assert sha(out / "private/propbank_verbs.json") == freeze["propbank_verbs_sha256"]


def process_signal(doc, parser, scorer, out):
    """Both arms, all eight original sentences, same official bank-max pipeline."""
    destination = out / "private/signal" / f"d{doc['doc_id']:02d}.json"
    if destination.exists():
        prior = read(destination)
        assert prior["freeze_sha256"] == sha(out / "FREEZE.json")
        return prior
    started = time.monotonic()
    record = {"doc_id": doc["doc_id"], "arms": {},
              "freeze_sha256": sha(out / "FREEZE.json")}
    for name in ("positive", "unwatermarked"):
        carriers = doc["arms"][name]["carriers"]
        assert len(carriers) == 8
        scores = [scorer.scores(parser.parse(c["text"]), "official") for c in carriers]
        record["arms"][name] = {
            "sentence_scores": scores,
            "sentence_sha256": [digest(c["text"]) for c in carriers]}
    record["seconds"] = time.monotonic() - started
    atomic(destination, record)
    atomic(out / "private/SCORING_DIAGNOSTICS.json", scorer.events)
    return record


def signal_analysis(out):
    freeze = read(out / "FREEZE.json")
    records = {r["doc_id"]: r for r in
               (read(p) for p in sorted((out / "private/signal").glob("*.json")))}
    assert set(records) == set(freeze["split"]["development"] + freeze["split"]["readout"])
    result = {"at": now(), "freeze_sha256": sha(out / "FREEZE.json"), "pairs": len(records),
              "sentences_per_arm": 192, "score_shape": [384, 100], "populations": {}}
    for population in ("calibration", "signal_decision", "readout"):
        ids = freeze["split"][population]
        p, n, pd, nd = [], [], [], []
        for docid in ids:
            positive = bank_max(records[docid]["arms"]["positive"]["sentence_scores"])
            negative = bank_max(records[docid]["arms"]["unwatermarked"]["sentence_scores"])
            p += positive
            n += negative
            pd.append(mean_complete(positive))
            nd.append(mean_complete(negative))
        result["populations"][population] = {
            "pairs": len(ids), "carrier_auroc": auc(p, n), "carrier_paired_win": win(p, n),
            "document_auroc": auc(pd, nd), "document_paired_win": win(pd, nd),
            "positive_abstain": sum(not math.isfinite(v) for v in p),
            "negative_abstain": sum(not math.isfinite(v) for v in n),
            "positive_mean": safe_value(mean_complete(p)),
            "negative_mean": safe_value(mean_complete(n))}
    decision = result["populations"]["signal_decision"]
    gate = {"passed": decision["carrier_auroc"] >= freeze["clean_gate"]["carrier_auroc_min"]
            and decision["document_paired_win"] >= freeze["clean_gate"]["document_paired_win_min"],
            "carrier_auroc": decision["carrier_auroc"],
            "document_paired_win": decision["document_paired_win"],
            "decision_pairs": 6, "readout_used": False, "freeze_sha256": sha(out / "FREEZE.json")}
    atomic(out / "SIGNAL_RESULTS.json", result)
    atomic(out / "CLEAN_GATE.json", gate)
    atomic(out / "SIGNAL_COMPLETE.json", {
        "at": now(), "pairs": len(records), "sentences": 384,
        "signal_results_sha256": sha(out / "SIGNAL_RESULTS.json"),
        "freeze_sha256": sha(out / "FREEZE.json")})
    return gate


def score_blind_text(text, parser, scorer):
    """Candidate extraction and full-bank scoring take text only, no oracle labels."""
    from src.core.multiscale_blind import blind_geometry, event_subgraphs, maximum, assemble_features
    geometry = blind_geometry(text)
    amrs = {w["text"]: parser.parse(w["text"])
            for scale in geometry.values() for w in scale}
    short = [scorer.scores(amrs[w["text"]], "project") for w in geometry["S6"]]
    long = [scorer.scores(amrs[w["text"]], "project") for w in geometry["S12"]]
    event, extraction_errors = [], []
    for w in geometry["S12"]:
        amr = amrs[w["text"]]
        try:
            graphs = event_subgraphs(amr) if amr else []
        except Exception as exc:
            graphs = []
            extraction_errors.append({"text_sha256": digest(w["text"]), "reason": type(exc).__name__})
        event.append(maximum([scorer.scores(g, "project") for g in graphs]))
    return {"text_sha256": digest(text), "words": len(text.split()), "geometry": geometry,
            "candidate_scores": {"S6": short, "S12": long, "Ssub": event},
            "blind_features": assemble_features(geometry, short, long, event),
            "extractor_errors": extraction_errors}


def process_document(doc, condition, parser, scorer, out):
    sys.path.insert(0, str(out / "private/code"))
    from src.core.multiscale_blind import assemble_features
    assert read(out / "SIGNAL_COMPLETE.json")["sentences"] == 384
    assert read(out / "SIGNAL_COMPLETE.json")["freeze_sha256"] == sha(out / "FREEZE.json")
    if condition != "clean":
        assert read(out / "CLEAN_GATE.json")["passed"] and (out / "CLEAN_COMPLETE.json").exists()
    destination = out / "private/records" / f"d{doc['doc_id']:02d}_{condition}.json"
    if destination.exists():
        prior = read(destination)
        assert prior["freeze_sha256"] == sha(out / "FREEZE.json")
        return prior
    started = time.monotonic()
    record = {"doc_id": doc["doc_id"], "condition": condition, "arms": {},
              "freeze_sha256": sha(out / "FREEZE.json")}
    for name, arm in doc["arms"].items():
        text = arm["texts"][condition]
        assert digest(text) == arm["text_sha256"][condition]
        a = score_blind_text(text, parser, scorer)
        if condition == "clean":
            signal = read(out / "private/signal" / f"d{doc['doc_id']:02d}.json")
            a["official_sentence_scores"] = signal["arms"][name]["sentence_scores"]
            candidates = a["candidate_scores"]
            a["carrier_features"] = assemble_features(
                a["geometry"], candidates["S6"], candidates["S12"], candidates["Ssub"], arm["carriers"])
        record["arms"][name] = a
    record["seconds"] = time.monotonic() - started
    atomic(destination, record)
    atomic(out / "private/SCORING_DIAGNOSTICS.json", scorer.events)
    return record


def work(root, out, precheck_only=False):
    import fcntl
    import importlib.metadata as md
    import torch
    lock = (out / "private/work.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "HALT.json").exists():
        raise RuntimeError("Permanent halt; same manifest cannot resume")
    freeze = read(out / "FREEZE.json")
    verify(out, freeze)
    if (out / "COMPLETE.json").exists():
        return
    if (out / "PAUSED.json").exists():
        (out / "PAUSED.json").unlink()
    started = time.monotonic()
    ledger_path = out / "private/RESOURCE_LEDGER.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"sessions": []}
    prior_seconds = freeze["resource_gate"]["prior_gpu_seconds"] + sum(
        row["seconds"] for row in ledger["sessions"])
    def progress(phase, **details):
        elapsed = time.monotonic() - started
        atomic(out / "heartbeat.json", {
            "phase": phase, "at": now(), "session_seconds": elapsed,
            "conservative_total_gpu_hours": (prior_seconds + elapsed) / 3600,
            "parser_calls": parser.calls, "s2match_calls": scorer.calls, **details})
        if (prior_seconds + elapsed) / 3600 >= freeze["resource_gate"]["projected_gpu_hours_max"]:
            raise RuntimeError("Conservative total GPU budget reached")
        if (prior_seconds + elapsed) / 3600 >= freeze["resource_gate"]["wall_hours_max"]:
            raise RuntimeError("Wall-time budget reached")
    try:
        parser = Parser(root, out, freeze)
        scorer = Scorer(root, out)
        assert sha(scorer.glove) == freeze["scorer"]["glove_sha256"]
        if not (out / "RUNTIME.json").exists():
            atomic(out / "RUNTIME.json", {
                "freeze_sha256": sha(out / "FREEZE.json"),
                "versions": {p: md.version(p) for p in ("torch", "amrlib", "transformers", "spacy", "penman")},
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "dtype": "float32", "tf32": False, "glove_sha256": sha(scorer.glove)})
        documents = read(out / "private/DOCUMENTS.json")
        lookup = {d["doc_id"]: d for d in documents}
        if ledger["sessions"] and freeze.get("execution", {}).get("resumable_gpu_pressure"):
            doc = lookup[freeze["split"]["precheck"][0]]
            from src.core.multiscale_blind import blind_geometry
            geometry = blind_geometry(doc["arms"]["positive"]["texts"]["clean"])
            samples = [doc["arms"]["unwatermarked"]["carriers"][0]["text"],
                       geometry["S6"][0]["text"], geometry["S12"][0]["text"]]
            replay = []
            for text in samples:
                cached = out / "private/parses" / (digest(text) + ".json")
                if cached.exists():
                    saved, fresh = read(cached), parser.fresh(text)
                    assert saved["amr_sha256"] == fresh["amr_sha256"]
                    assert saved["valid"] == fresh["valid"]
                    replay.append({"text_sha256": digest(text), "exact_parse_sha": True})
            atomic(out / "private/resume_replays" / f"{time.time_ns()}.json", {
                "at": now(), "freeze_sha256": sha(out / "FREEZE.json"), "samples": replay})
        workload = read(out / "WORKLOAD.json")
        if not (out / "SIGNAL_PRECHECK.json").exists():
            t0 = time.monotonic()
            for docid in freeze["split"]["precheck"]:
                process_signal(lookup[docid], parser, scorer, out)
                progress("signal_precheck", last_doc_id=docid)
            elapsed = time.monotonic() - t0
            sample = lookup[freeze["split"]["precheck"][0]]["arms"]["positive"]["carriers"][0]["text"]
            first, second = parser.fresh(sample), parser.fresh(sample)
            replay = first["amr_sha256"] == second["amr_sha256"] and first["valid"] == second["valid"]
            amr = first["amr"] if first["valid"] else None
            score_replay = {}
            if amr:
                for kind in ("official", "project"):
                    normalized = scorer.normalize(amr, kind)
                    one = [scorer.pair(normalized, t) for t in scorer.bank]
                    two = [scorer.pair(normalized, t) for t in scorer.bank]
                    score_replay[kind] = one == two and all(x is not None for x in one)
            projection = (prior_seconds + elapsed / 32 * 384 * 1.5) / 3600
            vram = torch.cuda.max_memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory
            passed = (replay and len(score_replay) == 2 and all(score_replay.values())
                      and vram < 0.8 and projection < 12)
            atomic(out / "SIGNAL_PRECHECK.json", {
                "passed": passed, "parser_replay": replay, "score_replay": score_replay,
                "seconds": elapsed, "original_sentences": 32,
                "projected_signal_gpu_hours_with_safety_factor_and_prior": projection,
                "peak_reserved_fraction": vram, "freeze_sha256": sha(out / "FREEZE.json")})
        if not read(out / "SIGNAL_PRECHECK.json")["passed"]:
            atomic(out / "STOP.json", {"reason": "Signal-only precheck failed", "at": now()})
            return
        for index, doc in enumerate(documents):
            process_signal(doc, parser, scorer, out)
            progress("signal", completed_pairs=index + 1, total_pairs=24)
        gate = signal_analysis(out)
        progress("signal_complete", completed_pairs=24, clean_gate=gate["passed"])
        if not (out / "PRECHECK.json").exists():
            t0 = time.monotonic()
            for docid in freeze["split"]["precheck"]:
                process_document(lookup[docid], "clean", parser, scorer, out)
                progress("feature_precheck", last_doc_id=docid)
            elapsed = time.monotonic() - t0
            conditions = freeze["conditions"] if gate["passed"] else ["clean"]
            scope = "all_conditions" if gate["passed"] else "clean_only"
            planned = workload["feature_inputs_after_signal"][scope]
            pre_n = workload["precheck_feature_inputs_after_signal"]
            projection = (prior_seconds + time.monotonic() - started
                          + elapsed / max(1, pre_n) * planned * 1.5) / 3600
            vram = torch.cuda.max_memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory
            atomic(out / "PRECHECK.json", {
                "passed": vram < 0.8 and projection < 12,
                "seconds": elapsed, "planned_inputs_upper_bound": planned,
                "precheck_inputs": pre_n, "conditions_planned": conditions,
                "workload_basis": workload["basis"],
                "workload_sha256": sha(out / "WORKLOAD.json"),
                "projected_gpu_hours_with_safety_factor_and_prior": projection,
                "peak_reserved_fraction": vram, "freeze_sha256": sha(out / "FREEZE.json")})
        if not read(out / "PRECHECK.json")["passed"]:
            atomic(out / "STOP.json", {"reason": "Feature readiness precheck failed", "at": now()})
            return
        if precheck_only:
            return
        for index, doc in enumerate(documents):
            process_document(doc, "clean", parser, scorer, out)
            progress("features_clean", completed_pairs=index + 1, total_pairs=24)
        analysis = analyze(out, partial=False)
        atomic(out / "CLEAN_COMPLETE.json", {"at": now(), "pairs": 24,
                                            "freeze_sha256": sha(out / "FREEZE.json")})
        if analysis["clean_gate"]["passed"]:
            for condition in freeze["conditions"][1:]:
                for index, doc in enumerate(documents):
                    process_document(doc, condition, parser, scorer, out)
                    progress(condition, completed_pairs=index + 1, total_pairs=24)
                analyze(out, partial=False)
        atomic(out / "COMPLETE.json", {"at": now(), "attack_stage_executed": analysis["clean_gate"]["passed"],
                                       "seconds": time.monotonic() - started,
                                       "freeze_sha256": sha(out / "FREEZE.json")})
        analyze(out, partial=False)
        atomic(out / "heartbeat.json", {"phase": "complete", "at": now(),
                                       "attack_stage_executed": analysis["clean_gate"]["passed"]})
    except (ResourcePause, torch.cuda.OutOfMemoryError) as exc:
        pause = {"at": now(), "reason": type(exc).__name__ + ": " + str(exc),
                 "freeze_sha256": sha(out / "FREEZE.json"),
                 "scientific_outcome": None,
                 "resume_scope": "Same successor only; verified completed cache records"}
        if (out / "GPU_RESOURCE_STATE.json").exists():
            pause["last_gpu_snapshot"] = read(out / "GPU_RESOURCE_STATE.json")
        atomic(out / "private/resource_pauses" / f"{time.time_ns()}.json", pause)
        atomic(out / "PAUSED.json", pause)
        raise ResourcePause(pause["reason"]) from exc
    except BaseException as exc:
        atomic(out / "HALT.json", {"at": now(), "reason": type(exc).__name__ + ": " + str(exc),
                                  "freeze_sha256": sha(out / "FREEZE.json")})
        raise
    finally:
        ledger["sessions"].append({"ended_at": now(), "seconds": time.monotonic() - started})
        ledger["prior_attempts_seconds"] = freeze["resource_gate"]["prior_gpu_seconds"]
        ledger["conservative_gpu_hours"] = (
            ledger["prior_attempts_seconds"] + sum(row["seconds"] for row in ledger["sessions"])) / 3600
        atomic(ledger_path, ledger)


def safe_value(value):
    return float(value) if math.isfinite(float(value)) else None


def auc(positive, negative):
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def win(positive, negative):
    if not positive or len(positive) != len(negative):
        return None
    return sum((p > n) + 0.5 * (p == n) for p, n in zip(positive, negative)) / len(positive)


def bank_max(rows):
    return [max([v for v in r if v is not None], default=-math.inf) for r in rows]


def mean_complete(values):
    return sum(values) / len(values) if values and all(math.isfinite(v) for v in values) else -math.inf


def analyze(out, partial=False):
    import numpy as np
    import torch
    sys.path.insert(0, str(out / "private/code"))
    from src.core.multiscale_fusion import MultiScaleAttentionFusion, EmpiricalThreshold
    freeze = read(out / "FREEZE.json")
    verify(out, freeze)
    torch.set_num_threads(2)
    records = [read(p) for p in sorted((out / "private/records").glob("*.json"))]
    models = {}
    for m in freeze["checkpoints"]:
        checkpoint = torch.load(out / m["file"], map_location="cpu", weights_only=False)
        model = MultiScaleAttentionFusion(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[m["name"]] = model
    result = {"version": VERSION, "at": now(), "freeze_sha256": sha(out / "FREEZE.json"),
              "completed_records": len(records), "independent_human_test_fpr": None,
              "no_refit_no_test_tuning": True, "conditions": {}, "feature_shapes": {}}
    clean = {r["doc_id"]: r for r in records if r["condition"] == "clean"}
    if (out / "CLEAN_GATE.json").exists():
        result["clean_gate"] = read(out / "CLEAN_GATE.json")
        result["official_signal"] = read(out / "SIGNAL_RESULTS.json")
    else:
        result["clean_gate"] = {"passed": None, "status": "pending_complete_development_decision"}
    # All model inference is fixed; calibration uses only the frozen clean negatives.
    score_rows = {}
    feature_rows, feature_metadata, blind_rows, blind_metadata = [], [], [], []
    for r in records:
        condition, docid = r["condition"], r["doc_id"]
        for arm, a in r["arms"].items():
            for view in ("blind", "carrier"):
                key = view + "_features"
                if key not in a:
                    continue
                features = torch.tensor([[[np.nan if v is None else v for v in row] for row in item]
                                         for item in a[key]], dtype=torch.float32)
                if view == "carrier":
                    feature_rows.extend(a[key])
                    feature_metadata.extend([{"doc_id": docid, "arm": arm, "slot": i}
                                             for i in range(len(a[key]))])
                else:
                    blind_rows.extend(a[key])
                    blind_metadata.extend([{"doc_id": docid, "arm": arm, "condition": condition,
                                            "anchor": i} for i in range(len(a[key]))])
                with torch.no_grad():
                    for name, model in models.items():
                        values = model(features)["statistic"].tolist()
                        score_rows[(docid, condition, arm, view, name)] = (
                            [mean_complete(values)] if view == "blind" else values)
            if condition == "clean":
                values = bank_max(a["official_sentence_scores"])
                score_rows[(docid, condition, arm, "carrier", "official_sentence")] = values
                score_rows[(docid, condition, arm, "sentence_doc", "official_sentence")] = [mean_complete(values)]
    if feature_rows:
        tensor = torch.tensor([[[np.nan if v is None else v for v in row] for row in item]
                               for item in feature_rows], dtype=torch.float32)
        torch.save({"scores": tensor, "metadata": feature_metadata,
                    "freeze_sha256": sha(out / "FREEZE.json"),
                    "bank_sha256": freeze["bank_sha256"]}, out / "private/CLEAN_FEATURES.pt")
        result["feature_shapes"]["clean_carrier"] = list(tensor.shape)
    if blind_rows:
        tensor = torch.tensor([[[np.nan if v is None else v for v in row] for row in item]
                               for item in blind_rows], dtype=torch.float32)
        torch.save({"scores": tensor, "metadata": blind_metadata,
                    "freeze_sha256": sha(out / "FREEZE.json"),
                    "bank_sha256": freeze["bank_sha256"]}, out / "private/BLIND_FEATURES.pt")
        result["feature_shapes"]["blind_all_conditions"] = list(tensor.shape)
    def get(ids, condition, arm, view, name):
        return [value for d in ids for value in score_rows.get((d, condition, arm, view, name), [])]
    calibration = freeze["split"]["calibration"]
    for condition in sorted({r["condition"] for r in records}):
        result["conditions"][condition] = {}
        available = {r["doc_id"] for r in records if r["condition"] == condition}
        for population in ("signal_decision", "readout"):
            ids = [d for d in freeze["split"][population] if d in available]
            group = {"pairs": len(ids), "expected_pairs": len(freeze["split"][population]), "methods": {}}
            for name in ["official_sentence"] + list(models):
                views = ("carrier", "sentence_doc") if name == "official_sentence" else ("carrier", "blind")
                for view in views:
                    p, n = [get(ids, condition, arm, view, name) for arm in ("positive", "unwatermarked")]
                    if not p and not n:
                        continue
                    metric = {"positive_n": len(p), "negative_n": len(n), "auroc": auc(p, n),
                              "paired_win": win(p, n),
                              "positive_abstain": sum(not math.isfinite(v) for v in p),
                              "negative_abstain": sum(not math.isfinite(v) for v in n),
                              "working_points": {}}
                    cal = get(calibration, "clean", "unwatermarked", view, name)
                    for alpha in (0.01, 0.05):
                        if len([d for d in calibration if d in clean]) != 6 or not cal:
                            row = {"status": "pending_complete_calibration"}
                        else:
                            threshold = EmpiricalThreshold.fit(cal, alpha)
                            row = threshold.to_dict()
                            if threshold.threshold is not None:
                                pp, nn = threshold.predict(p), threshold.predict(n)
                                row.update(tpr=sum(pp) / len(pp), fpr=sum(nn) / len(nn),
                                           positive_exceedances=sum(pp), negative_exceedances=sum(nn))
                        metric["working_points"][str(alpha)] = row
                    group["methods"][f"{name}/{view}"] = metric
            result["conditions"][condition][population] = group
    atomic(out / "RESULTS.json", result)
    prediction_rows = [{"doc_id": d, "condition": c, "arm": a, "view": v, "method": m,
                        "statistics": [safe_value(x) for x in xs]}
                       for (d, c, a, v, m), xs in score_rows.items()]
    atomic(out / "private/PREDICTIONS.json", prediction_rows)
    report(out, result)
    return result


def report(out, result):
    gate = result["clean_gate"]
    complete = (out / "COMPLETE.json").exists()
    status = "已完成" if complete else "运行中"
    lines = ["# 新语料 SWAN 与冻结多尺度模型迁移诊断", "",
             f"状态：{status}；已完成 {result['completed_records']} 个配对文档条件。", "",
             "## 原句检测信号", ""]
    if gate["passed"] is None:
        lines += ["信号决策仍待后 6 个开发文档完成，当前没有 Go/No-Go 结论。", ""]
    else:
        lines += [f"开发信号门：{'Go' if gate['passed'] else 'No-Go'}。官方原句 bank-max 的载体 "
                  f"AUROC={gate['carrier_auroc']:.3f}，文档配对胜率={gate['document_paired_win']:.3f}。"
                  "该决策只使用冻结的 6 对开发文档，未使用读出集。", ""]
        if not gate["passed"]:
            lines += ["原句尚未通过预设稳定信号门；攻击阶段按协议不执行，应先修生成通道。"
                      "这项结论来自正负例实际检测分数，与此前 0/24 严格完整图覆盖检查不同。", ""]
    lines += ["## 冻结模型比较", "",
              "各门控/注意力种子单独报告，未按新语料结果挑选或组合。"
              "表中的盲文档分数只依赖文本窗口；载体分数使用已知原句位置，仅作为诊断。", ""]
    for condition, populations in result["conditions"].items():
        for population, group in populations.items():
            if not group["methods"]:
                continue
            lines += [f"### {condition} · {population}（{group['pairs']}/{group['expected_pairs']} 对）", "",
                      "| 模型/口径 | AUROC | 配对胜率 | TPR@探索性5% | 观察FPR |",
                      "|---|---:|---:|---:|---:|"]
            for name, metric in group["methods"].items():
                wp = metric["working_points"]["0.05"]
                def fmt(v):
                    return "不足/缺失" if v is None else f"{v:.3f}"
                lines.append(f"| {name} | {fmt(metric['auroc'])} | {fmt(metric['paired_win'])} | "
                             f"{fmt(wp.get('tpr'))} | {fmt(wp.get('fpr'))} |")
            lines.append("")
    lines += ["## 解释边界与复现", "",
              "校准与测试文档完全分离，旧参数及归一化器保持冻结。新三尺度输入来自本轮重新解析和匹配，"
              f"当前 clean 载体张量形状为 {result['feature_shapes'].get('clean_carrier', '尚未完成')}。", "",
              "48 个校准载体来自 6 篇相关文档，5% 工作点仅供探索；1% 校准不足。"
              "文档校准只有 6 篇，1%/5% 文档误报率都不能估计到目标工作点。"
              "条件对应的负例接受相同编辑，报告各条件观察值。没有独立人类测试 FPR。", "",
              "正例与负例的长度不完全匹配；匹配信号可能包含长度或生成风格偏差，"
              "不能直接作为语义结构注入成功或投稿级鲁棒性证据。"
              "事件子图提取器修复了顺序不定和悬空边；S2Match 改为图对 SHA 随机流以支持重放，保留 5 次 hill-climb。"
              "这些是旧流程迁移时必须考虑的差异。", "",
              "证据：本目录 FREEZE.json、PRECHECK.json、RUNTIME.json、RESULTS.json 与 SHA256SUMS.json。"
              "完整文本、图、模型、预测与 CLEAN_FEATURES.pt 留在 private/，不纳入公开包。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    atomic(out / "SHA256SUMS.json", {
        str(p.relative_to(out)): sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name not in ("SHA256SUMS.json", "heartbeat.json") and not p.name.endswith(".log")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "work", "precheck", "analyze"))
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior-gpu-seconds", type=float, default=0.0)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.out.resolve(), prior_gpu_seconds=args.prior_gpu_seconds)
    elif args.mode == "analyze":
        analyze(args.out.resolve(), partial=True)
    else:
        try:
            work(args.root.resolve(), args.out.resolve(), precheck_only=args.mode == "precheck")
        except ResourcePause:
            raise SystemExit(75)


if __name__ == "__main__":
    main()
