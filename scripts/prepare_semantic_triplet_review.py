"""Real text/original-parse triplets with blank, gated human annotations.

No text generation, inference, inferred human labels, or automatic gold graphs.
Raw materials are confined to the ignored private_review/ directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import warnings
import zipfile

import penman
from penman.exceptions import DecodeError, LayoutError
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semantic_triplet_review1"
PRIVATE = Path.home() / ".cache/amr/semantic_triplet_review1"
PRIOR = Path.home() / ".cache/amr/semantic_sequence_readout1"
OLD = ROOT / "experiments/semantic_sequence_readout1"
PACK = OUT / "private_review"
SEED = 20261025
sys.path.insert(0, str(ROOT / "scripts"))
from run_semantic_channel_audit import atomic, sha


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def bytes_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def identity_blank():
    return {"reviewer_code": "", "completed_at": "",
            "human_attestation": False, "independent_attestation": False}


def text_blank(sample):
    return {
        "sample_id": sample, "status": "", "realized_meaning": "",
        "facts": [],
        "fact_schema_help": {
            "predicate": "实际发生/询问/要求的谓词，不能从目标图猜",
            "participants": [{"role": "agent/patient/recipient/.../unknown", "entity": "文本中的参与者"}],
            "polarity": "positive/negative/unknown",
            "modality": "asserted/possible/obligatory/intended/unknown",
            "evidence": [{"source": "carrier/context", "start": 0, "end": 0, "quote": ""}],
        },
        "context_required": None, "speech_act": "", "notes": "",
    }


def graph_blank(sample):
    return {
        "sample_id": sample, "status": "",
        "original_graph_fidelity": "", "fact_alignment": [],
        "alignment_schema_help": {
            "fact_index": 0, "fidelity": "preserved/missing/changed/unknown",
            "original_graph_evidence": "支持判断的原图三元组；缺失时说明查找依据",
            "reason": "",
        },
        "human_revision_attestation": False, "revision_reason": "", "notes": "",
    }


def target_blank(sample):
    return {"sample_id": sample, "target_realized": "", "evidence": [], "notes": ""}


def markdown_text_case(case):
    text, context = case["carrier"], case["context"]
    return (
        f"# 文本事实样本 {case['sample_id']}\n\n"
        "先独立记录实际意义。此阶段没有目标、解析图或模型结果。\n\n"
        f"## 解析器实际收到的原句\n\n{text}\n\n"
        f"## 原段落上下文\n\n{context}\n\n"
        "若必须依靠上下文才能判断，请标记 context_required=true。原解析器仅收到上面的原句。\n\n"
        "## 字符索引\n\n字符位置从 0 开始，左闭右开；JSON evidence.quote 必须与原文完全相同。\n\n"
        "```text\n" + "\n".join(f"[{i},{i+1}) {repr(c)}" for i, c in enumerate(text)) + "\n```\n"
    )


def markdown_graph_case(case):
    record = case["original_parse"]
    triples = []
    try:
        graph = penman.decode(record["amr"], model=AMR_MODEL)
        triples = [f"| `{s}` | `{r}` | `{t}` |" for s, r, t in graph.triples]
    except Exception:
        triples = ["| — | — | 原图无法严格解码；保留原字节供人工审查 |"]
    return (
        f"# 原图核验与人工修订 {case['sample_id']}\n\n"
        f"## 原句\n\n{case['carrier']}\n\n"
        f"## 原上下文\n\n{case['context']}\n\n"
        "## 原解析图\n\n```penman\n" + (record.get("amr") or "") + "\n```\n\n"
        "## AMR 模型规范方向后的三元组\n\n"
        "| 源 | 关系 | 目标 |\n|---|---|---|\n" + "\n".join(triples) + "\n\n"
        "依据先前已封存的文本事实逐项检查。human_revised.penman 初始为空；"
        "即使原图正确，也要人工明确确认并提交图。不得用目标图替代。\n"
    )


def archive(folder, filename):
    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                output.write(path, arcname=str(path.relative_to(folder)))


def prepare():
    if (PACK / "admin/MANIFEST.json").exists():
        raise RuntimeError("Existing review packets are immutable; use a new version")
    selection = read(PRIVATE / "SELECTION.json")
    if sha(OUT / "PROTOCOL.md") != selection["protocol_sha256"]:
        raise RuntimeError("Selection protocol changed")
    if sha(OLD / "RESULT.json") != selection["original_result_sha256"]:
        raise RuntimeError("Source result changed")
    if sha(PRIOR / "ASSIGNMENTS.json") != selection["assignments_sha256"]:
        raise RuntimeError("Source assignments changed")
    assignments = read(PRIOR / "ASSIGNMENTS.json")
    cases, manifest_cases = {}, {}
    for item in selection["selected"]:
        sid = item["sample_id"]
        document = PRIOR / "documents" / f"d{item['doc_id']:02d}.json"
        if sha(document) != item["document_sha256"]:
            raise RuntimeError("Source document changed")
        arm = read(document)["arms"]["positive"]
        carrier = arm["slots"][item["slot"]]
        if bytes_sha(carrier) != item["text_sha256"]:
            raise RuntimeError("Carrier SHA changed")
        parse_path = PRIVATE / "parses" / (item["text_sha256"] + ".json")
        record = read(parse_path)
        if record["freeze_sha256"] != selection["original_freeze_sha256"]:
            raise RuntimeError("Original parse identity changed")
        if record["text_sha256"] != bytes_sha(carrier):
            raise RuntimeError("Parser saw a different input")
        planned = assignments[item["doc_id"]]["slots"][item["slot"]]
        case = {**item, "carrier": carrier, "context": arm["text"],
                "target_amr": planned["target"], "target_markers": planned["markers"],
                "target_entities": planned.get("entities"), "original_parse": record}
        folder = PACK / "admin/cases" / sid
        folder.mkdir(parents=True, exist_ok=True)
        atomic(folder / "case.json", case)
        save_text(folder / "original_parse.penman", record.get("amr") or "")
        atomic(folder / "original_parse_record.json", record)
        atomic(folder / "human_confirmed_text.json", {
            "sample_id": sid, "status": "PENDING_HUMAN_REVIEW", "realized_meaning": None, "facts": None})
        save_text(folder / "human_revised.penman", "")
        atomic(folder / "comparison.json", {
            "sample_id": sid, "text_status": "PENDING", "original_parse_available": True,
            "original_strict_valid": record["valid"], "human_revision_status": "PENDING",
            "intended_symbol": item["planned_symbol"],
            "original_whole_sentence_W_prediction": item["original_prediction"],
            "writing_failure": "NOT_EVALUATED", "parsing_failure": "NOT_EVALUATED",
            "classification_failure": "NOT_EVALUATED", "corrected_graph_scored": False})
        save_text(folder / "TRIPLET.md",
                  f"# 三方对照 {sid}\n\n本页为管理端，包含目标与模型输出，不用于文本盲评。\n\n"
                  f"## 真实原句\n\n{carrier}\n\n"
                  "## 真人确认文本\n\n待两位真人独立确认与第三人裁决；见 human_confirmed_text.json。\n\n"
                  f"## 原解析图\n\n```penman\n{record.get('amr') or ''}\n```\n\n"
                  "## 人工修订图\n\nhuman_revised.penman 当前为空，尚未人工修订。\n\n"
                  f"## 写入目标（管理端）\n\n```penman\n{planned['target']}\n```\n")
        cases[sid] = case
        manifest_cases[sid] = {"case_sha256": sha(folder / "case.json"),
                              "carrier_sha256": item["text_sha256"],
                              "original_parse_record_sha256": sha(folder / "original_parse_record.json")}
    inputs = {}
    for reviewer in (1, 2):
        folder = PACK / f"reviewer{reviewer}"
        ids = list(cases)
        random.Random(SEED + reviewer).shuffle(ids)
        atomic(folder / "reviewer.json", identity_blank())
        atomic(folder / "ORDER.json", ids)
        save_text(folder / "INSTRUCTIONS.md",
                  "# 第一阶段：只读文本\n\n"
                  "每条 text.md 包含原句和上下文；编辑 text_response.json。\n"
                  "status 填 confirmed 或 unknown；confirmed 表示你独立确认实际意义，不表示目标已经写入。\n"
                  "confirmed 必须填写 realized_meaning、至少一条 facts 和原文 evidence 范围；"
                  "unknown 保留行并解释。\n"
                  "context_required 填 true/false；speech_act 填 statement/question/request/command/unknown。\n"
                  "不看其他评审者、目标图、原解析、模型结果，不用 LLM 代填。\n"
                  "完成全部 12 条后填写 reviewer.json 中代号、完成时间及两项真人/独立声明。\n"
                  "请不要修改 text_input.json、ORDER.json 或说明文件。\n\n"
                  f"在仓库根目录运行：\n\n```bash\n"
                  f".venv-swan-positive/bin/python scripts/prepare_semantic_triplet_review.py collect-text --reviewer {reviewer}\n"
                  "```\n\n校验通过后解锁图阶段，文本答案封存，不再改写。\n")
        inputs[f"reviewer{reviewer}/ORDER.json"] = sha(folder / "ORDER.json")
        inputs[f"reviewer{reviewer}/INSTRUCTIONS.md"] = sha(folder / "INSTRUCTIONS.md")
        for sid in ids:
            case = cases[sid]
            base = folder / "01_text" / sid
            atomic(base / "text_input.json", {"sample_id": sid, "carrier": case["carrier"], "context": case["context"],
                                           "parser_input_scope": "carrier_only"})
            save_text(base / "text.md", markdown_text_case(case))
            atomic(base / "text_response.json", text_blank(sid))
            for name in ("text_input.json", "text.md"):
                inputs[str((base / name).relative_to(PACK))] = sha(base / name)
        archive(folder, PACK / f"reviewer{reviewer}_text.zip")
    # Adjudicator is only opened after both reviewers complete all stages.
    save_text(PACK / "admin/INSTRUCTIONS.md",
              "# 管理端\n\ncases/ 是真实文本—原解析—空白人工图的三方对照。\n"
              "含目标、抽样类别和预测，不交给文本盲评者。\n"
              "两人独立完成文本、图修订与目标核验后，运行 prepare-adjudication 生成第三人裁决包。\n"
              "未完成前不得将 pending 改成已确认，也不得把原图复制成已修订图。\n")
    frozen = {"at": now(), "schema": "semantic-triplet-review/v1",
              "selection_sha256": sha(PRIVATE / "SELECTION.json"),
              "protocol_sha256": sha(OUT / "PROTOCOL.md"),
              "source_sha256": sha(Path(__file__)),
              "cases": manifest_cases, "reviewer_inputs_sha256": inputs, "planned": 12,
              "reviewers": 2, "blank_human_fields": True}
    atomic(PACK / "admin/MANIFEST.json", frozen)
    atomic(OUT / "PREPARATION_STATUS.json", {
        "at": now(), "prepared_samples": 12, "original_graphs": 12,
        "source_documents": len({c["doc_id"] for c in cases.values()}),
        "distinct_templates": len({c["template_id"] for c in cases.values()}),
        "original_strict_invalid_count": sum(not c["original_parse"]["valid"] for c in cases.values()),
        "human_confirmed_texts": 0, "human_revised_graphs": 0, "adjudicated_gold_triplets": 0,
        "human_fields_blank": True, "new_model_calls": 0,
        "manifest_sha256": sha(PACK / "admin/MANIFEST.json"),
        "reviewer_text_zip_sha256": {str(r): sha(PACK / f"reviewer{r}_text.zip") for r in (1, 2)},
        "raw_packet_gitignored": True})
    return {"prepared": 12, "human_confirmed": 0, "human_revised": 0}


def checked_cases():
    manifest = read(PACK / "admin/MANIFEST.json")
    if sha(Path(__file__)) != manifest["source_sha256"] or sha(OUT / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise RuntimeError("Frozen review source/protocol changed")
    cases = {}
    for sid, hashes in manifest["cases"].items():
        folder = PACK / "admin/cases" / sid
        if sha(folder / "case.json") != hashes["case_sha256"]:
            raise RuntimeError("Case changed")
        if sha(folder / "original_parse_record.json") != hashes["original_parse_record_sha256"]:
            raise RuntimeError("Original parse record changed")
        case = read(folder / "case.json")
        if bytes_sha(case["carrier"]) != hashes["carrier_sha256"]:
            raise RuntimeError("Carrier changed")
        if (folder / "original_parse.penman").read_text() != (case["original_parse"].get("amr") or ""):
            raise RuntimeError("Original PENMAN changed")
        cases[sid] = case
    for rel, expected in manifest["reviewer_inputs_sha256"].items():
        if sha(PACK / rel) != expected:
            raise RuntimeError("Immutable reviewer input changed")
    return cases


def check_identity(path):
    person = read(path)
    if not person.get("reviewer_code") or not person.get("completed_at"):
        raise ValueError("Missing reviewer identity/time")
    datetime.fromisoformat(person["completed_at"].replace("Z", "+00:00"))
    if person.get("human_attestation") is not True or person.get("independent_attestation") is not True:
        raise ValueError("Missing actual human/independent declarations")
    return person


def validate_evidence(evidence, case):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Evidence required")
    for item in evidence:
        source = item.get("source")
        if source not in ("carrier", "context"):
            raise ValueError("Evidence source invalid")
        text = case[source]
        a, b = item.get("start"), item.get("end")
        if type(a) is not int or type(b) is not int or not (0 <= a < b <= len(text)):
            raise ValueError("Evidence coordinates invalid")
        if text[a:b] != item.get("quote"):
            raise ValueError("Evidence quote differs from original")


def validate_text(value, case):
    if value.get("sample_id") != case["sample_id"]:
        raise ValueError("Sample identity changed")
    if value.get("status") not in ("confirmed", "unknown"):
        raise ValueError("Text annotation pending/invalid")
    if type(value.get("context_required")) is not bool:
        raise ValueError("Context dependency required")
    if value.get("speech_act") not in ("statement", "question", "request", "command", "unknown"):
        raise ValueError("Speech act invalid")
    if value["status"] == "unknown":
        if not value.get("notes"):
            raise ValueError("Uncertainty explanation required")
        return
    if not value.get("realized_meaning") or not value.get("facts") or not isinstance(value["facts"], list):
        raise ValueError("Realized meaning/facts required")
    for fact in value["facts"]:
        if not isinstance(fact, dict) or not fact.get("predicate"):
            raise ValueError("Actual predicate required")
        if not isinstance(fact.get("participants"), list):
            raise ValueError("Participant list required")
        for participant in fact["participants"]:
            if not participant.get("role") or not participant.get("entity"):
                raise ValueError("Participant role/entity required")
        if fact.get("polarity") not in ("positive", "negative", "unknown"):
            raise ValueError("Polarity invalid")
        if fact.get("modality") not in ("asserted", "possible", "obligatory", "intended", "unknown"):
            raise ValueError("Modality invalid")
        validate_evidence(fact.get("evidence"), case)


def snapshot_input_paths(folder, exclude_suffix="response.json"):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and not p.name.endswith(exclude_suffix) and p.name != "human_revised.penman"}


def collect_text(reviewer):
    cases = checked_cases()
    folder = PACK / f"reviewer{reviewer}"
    errors, labels = [], {}
    try:
        person = check_identity(folder / "reviewer.json")
    except (ValueError, TypeError, KeyError):
        person = {}
        errors.append("missing_human_identity_or_attestation")
    for sid, case in cases.items():
        try:
            value = read(folder / "01_text" / sid / "text_response.json")
            validate_text(value, case)
            labels[sid] = value
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            errors.append(sid + ":text_incomplete_or_invalid")
    status = {"at": now(), "reviewer": reviewer, "planned": 12, "valid_submitted_rows": len(labels),
              "complete": not errors, "errors": errors, "meaning_correctness": "HUMAN_JUDGMENT_NOT_AUTO_CHECKED"}
    atomic(OUT / f"REVIEWER{reviewer}_TEXT_CHECK.json", status)
    if errors:
        return status
    seal_path = PACK / "admin" / f"reviewer{reviewer}_TEXT_SEAL.json"
    if seal_path.exists():
        verify_text_seal(reviewer, cases)
        return status
    labels_path = PACK / "admin" / f"reviewer{reviewer}_TEXT_LABELS.json"
    atomic(labels_path, {"identity": person, "labels": labels})
    seal = {"at": now(), "identity_sha256": sha(folder / "reviewer.json"),
            "responses_sha256": {sid: sha(folder / "01_text" / sid / "text_response.json") for sid in cases},
            "labels_sha256": sha(labels_path)}
    atomic(seal_path, seal)
    stage = folder / "02_graph"
    for sid, case in cases.items():
        base = stage / sid
        atomic(base / "submitted_text.json", labels[sid])
        save_text(base / "original_parse.penman", case["original_parse"].get("amr") or "")
        save_text(base / "graph.md", markdown_graph_case(case))
        save_text(base / "human_revised.penman", "")
        atomic(base / "graph_response.json", graph_blank(sid))
    atomic(stage / "graph_reviewer.json", identity_blank())
    save_text(stage / "INSTRUCTIONS.md",
              "# 第二阶段：图核验与真人修订\n\n文本答案已封存，不改写第一阶段文件。\n"
              "逐事实填写 fact_alignment；original_graph_fidelity 填 faithful/missing/changed/unknown。\n"
              "status 填 corrected/unchanged/unrecoverable/unknown；corrected/unchanged 必须由真人提交 human_revised.penman。\n"
              "human_revision_attestation=true 表示真人依据实际文本完成图核验，不是助手或模型代填。\n"
              "每项事实给原图 evidence 与理由。上下文事实不要强迫图猜出未输入的信息。\n"
              "完成全部样本后填写 graph_reviewer.json（代号须与文本阶段相同）。\n\n"
              f"```bash\n.venv-swan-positive/bin/python scripts/prepare_semantic_triplet_review.py collect-graph --reviewer {reviewer}\n```\n")
    atomic(PACK / "admin" / f"reviewer{reviewer}_GRAPH_INPUT_SEAL.json", snapshot_input_paths(stage))
    archive(stage, PACK / f"reviewer{reviewer}_graph.zip")
    return status


def verify_text_seal(reviewer, cases):
    folder = PACK / f"reviewer{reviewer}"
    seal = read(PACK / "admin" / f"reviewer{reviewer}_TEXT_SEAL.json")
    if sha(folder / "reviewer.json") != seal["identity_sha256"]:
        raise RuntimeError("Text reviewer identity changed after seal")
    for sid in cases:
        if sha(folder / "01_text" / sid / "text_response.json") != seal["responses_sha256"][sid]:
            raise RuntimeError("Text answer changed after seal")
    labels = PACK / "admin" / f"reviewer{reviewer}_TEXT_LABELS.json"
    if sha(labels) != seal["labels_sha256"]:
        raise RuntimeError("Sealed text labels changed")
    return read(labels)


def check_stage_inputs(reviewer, stage):
    inputs = read(PACK / "admin" / f"reviewer{reviewer}_{stage.upper()}_INPUT_SEAL.json")
    folder = PACK / f"reviewer{reviewer}" / {"graph": "02_graph", "target": "03_target"}[stage]
    for rel, expected in inputs.items():
        if rel.endswith("_reviewer.json"):
            continue
        if sha(folder / rel) != expected:
            raise RuntimeError("Released stage input changed")


def normalized_graph(text):
    if not text.strip():
        raise ValueError("Human revised graph is empty")
    messages = []
    class Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())
    capture = Capture()
    logger = logging.getLogger("penman")
    logger.addHandler(capture)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = penman.decode(text, model=AMR_MODEL)
            roundtrip = penman.decode(penman.encode(graph, model=AMR_MODEL), model=AMR_MODEL)
        instances = list(graph.instances())
        variables = {t.source for t in instances}
        if messages or caught or not instances or graph.top not in variables:
            raise ValueError("Graph warning/root/instances invalid")
        if len(variables) != len(instances):
            raise ValueError("Repeated variable declarations")
        if sorted(graph.triples) != sorted(roundtrip.triples) or graph.top != roundtrip.top:
            raise ValueError("Graph roundtrip mismatch")
        adjacency = {v: set() for v in variables}
        for s, r, t in graph.triples:
            if s not in variables:
                raise ValueError("Undeclared source variable")
            if t in variables and r != ":instance":
                adjacency[s].add(t)
                adjacency[t].add(s)
        visited, todo = set(), [graph.top]
        while todo:
            v = todo.pop()
            if v in visited:
                continue
            visited.add(v)
            todo.extend(adjacency[v]-visited)
        if visited != variables:
            raise ValueError("Disconnected graph")
        return graph
    except (DecodeError, LayoutError) as exc:
        raise ValueError("Invalid PENMAN") from exc
    finally:
        logger.removeHandler(capture)


def validate_graph(value, graph_text, case, text_label):
    if value.get("sample_id") != case["sample_id"]:
        raise ValueError("Sample identity mismatch")
    if value.get("status") not in ("corrected", "unchanged", "unrecoverable", "unknown"):
        raise ValueError("Graph review pending/invalid")
    if value.get("original_graph_fidelity") not in ("faithful", "missing", "changed", "unknown"):
        raise ValueError("Original graph fidelity missing")
    if value.get("human_revision_attestation") is not True or not value.get("revision_reason"):
        raise ValueError("Actual human revision declaration/reason required")
    facts = text_label.get("facts") or []
    alignment = value.get("fact_alignment")
    if not isinstance(alignment, list) or {a.get("fact_index") for a in alignment} != set(range(len(facts))):
        raise ValueError("All confirmed text facts need alignment")
    if len(alignment) != len(facts):
        raise ValueError("Duplicated fact alignment")
    for item in alignment:
        if item.get("fidelity") not in ("preserved", "missing", "changed", "unknown") or not item.get("reason"):
            raise ValueError("Fact fidelity/reason invalid")
        if not item.get("original_graph_evidence"):
            raise ValueError("Graph evidence/search basis required")
    if value["status"] in ("corrected", "unchanged"):
        normalized_graph(graph_text)
        if text_label["status"] != "confirmed":
            raise ValueError("Cannot establish revised gold from unknown text")
        if value["status"] == "unchanged":
            # Preserve content honestly: identical original bytes required for unchanged.
            if graph_text.strip() != (case["original_parse"].get("amr") or "").strip():
                raise ValueError("Changed graph marked unchanged")
            if not case["original_parse"]["valid"]:
                raise ValueError("Invalid original graph cannot be marked unchanged")
    elif graph_text.strip():
        raise ValueError("Unknown/unrecoverable must not assert a corrected graph")


def collect_graph(reviewer):
    cases = checked_cases()
    if not (PACK / "admin" / f"reviewer{reviewer}_TEXT_SEAL.json").exists():
        result = {"complete": False, "reviewer": reviewer, "valid_submitted_rows": 0, "planned": 12,
                  "errors": ["text_stage_not_sealed_graph_stage_locked"]}
        atomic(OUT / f"REVIEWER{reviewer}_GRAPH_CHECK.json", result)
        return result
    texts = verify_text_seal(reviewer, cases)
    check_stage_inputs(reviewer, "graph")
    stage = PACK / f"reviewer{reviewer}/02_graph"
    labels, errors = {}, []
    try:
        person = check_identity(stage / "graph_reviewer.json")
        if person["reviewer_code"] != texts["identity"]["reviewer_code"]:
            raise ValueError("Graph reviewer differs from text reviewer")
    except (ValueError, TypeError, KeyError):
        person = {}
        errors.append("missing_or_different_human_identity")
    for sid, case in cases.items():
        try:
            value = read(stage / sid / "graph_response.json")
            graph = (stage / sid / "human_revised.penman").read_text()
            validate_graph(value, graph, case, texts["labels"][sid])
            labels[sid] = {"response": value, "human_revised_graph": graph}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            errors.append(sid + ":graph_incomplete_or_invalid")
    status = {"at": now(), "reviewer": reviewer, "complete": not errors, "planned": 12,
              "valid_submitted_rows": len(labels), "errors": errors,
              "semantic_correctness": "HUMAN_JUDGMENT_NOT_AUTO_CHECKED"}
    atomic(OUT / f"REVIEWER{reviewer}_GRAPH_CHECK.json", status)
    if errors:
        return status
    labels_path = PACK / "admin" / f"reviewer{reviewer}_GRAPH_LABELS.json"
    if labels_path.exists():
        verify_graph_seal(reviewer, cases)
        return status
    atomic(labels_path, {"identity": person, "labels": labels})
    seal = {"identity_sha256": sha(stage / "graph_reviewer.json"),
            "responses_sha256": {sid: sha(stage / sid / "graph_response.json") for sid in cases},
            "graphs_sha256": {sid: sha(stage / sid / "human_revised.penman") for sid in cases},
            "labels_sha256": sha(labels_path)}
    atomic(PACK / "admin" / f"reviewer{reviewer}_GRAPH_SEAL.json", seal)
    target_stage = PACK / f"reviewer{reviewer}/03_target"
    atomic(target_stage / "target_reviewer.json", identity_blank())
    for sid, case in cases.items():
        atomic(target_stage / sid / "target_input.json", {
            "sample_id": sid, "carrier": case["carrier"], "context": case["context"],
            "target_meaning_graph": case["target_amr"], "target_markers": case["target_markers"],
            "entity_contract": case["target_entities"],
            "sealed_text_response": texts["labels"][sid]})
        atomic(target_stage / sid / "target_response.json", target_blank(sid))
        save_text(target_stage / sid / "target.md",
                  f"# 写入核验 {sid}\n\n目标只在实际文本事实已封存后展示。\n\n"
                  f"## 原句\n\n{case['carrier']}\n\n"
                  f"## 目标意义图\n\n```penman\n{case['target_amr']}\n```\n\n"
                  "target_realized 填 yes/no/unknown，给原文证据和理由。"
                  "不得根据目标反改已封存的文本事实或修订图。\n")
    save_text(target_stage / "INSTRUCTIONS.md",
              "# 第三阶段：目标写入核验\n\ntarget_realized 填 yes/no/unknown，"
              "evidence 采用文本阶段相同范围格式。失败/不确定保留，不修改前阶段。\n"
              "完成后填写 target_reviewer.json，身份与前两阶段一致。\n\n"
              f"```bash\n.venv-swan-positive/bin/python scripts/prepare_semantic_triplet_review.py collect-target --reviewer {reviewer}\n```\n")
    atomic(PACK / "admin" / f"reviewer{reviewer}_TARGET_INPUT_SEAL.json", snapshot_input_paths(target_stage))
    archive(target_stage, PACK / f"reviewer{reviewer}_target.zip")
    return status


def verify_graph_seal(reviewer, cases):
    verify_text_seal(reviewer, cases)
    stage = PACK / f"reviewer{reviewer}/02_graph"
    seal = read(PACK / "admin" / f"reviewer{reviewer}_GRAPH_SEAL.json")
    if sha(stage / "graph_reviewer.json") != seal["identity_sha256"]:
        raise RuntimeError("Graph reviewer changed after seal")
    for sid in cases:
        if (sha(stage / sid / "graph_response.json") != seal["responses_sha256"][sid] or
                sha(stage / sid / "human_revised.penman") != seal["graphs_sha256"][sid]):
            raise RuntimeError("Human graph response changed after seal")
    path = PACK / "admin" / f"reviewer{reviewer}_GRAPH_LABELS.json"
    if sha(path) != seal["labels_sha256"]:
        raise RuntimeError("Sealed human graph labels changed")
    return read(path)


def validate_target(value, case):
    if value.get("sample_id") != case["sample_id"] or value.get("target_realized") not in ("yes", "no", "unknown"):
        raise ValueError("Target judgment pending/invalid")
    if not value.get("notes"):
        raise ValueError("Target judgment reason required")
    if value["target_realized"] != "unknown":
        validate_evidence(value.get("evidence"), case)


def collect_target(reviewer):
    cases = checked_cases()
    if not (PACK / "admin" / f"reviewer{reviewer}_GRAPH_SEAL.json").exists():
        result = {"complete": False, "reviewer": reviewer, "valid_submitted_rows": 0, "planned": 12,
                  "errors": ["graph_stage_not_sealed_target_stage_locked"]}
        atomic(OUT / f"REVIEWER{reviewer}_TARGET_CHECK.json", result)
        return result
    graphs = verify_graph_seal(reviewer, cases)
    check_stage_inputs(reviewer, "target")
    stage = PACK / f"reviewer{reviewer}/03_target"
    labels, errors = {}, []
    try:
        person = check_identity(stage / "target_reviewer.json")
        if person["reviewer_code"] != graphs["identity"]["reviewer_code"]:
            raise ValueError("Target reviewer differs")
    except (ValueError, TypeError, KeyError):
        person = {}
        errors.append("missing_or_different_human_identity")
    for sid, case in cases.items():
        try:
            value = read(stage / sid / "target_response.json")
            validate_target(value, case)
            labels[sid] = value
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            errors.append(sid + ":target_incomplete_or_invalid")
    result = {"at": now(), "reviewer": reviewer, "planned": 12, "valid_submitted_rows": len(labels),
              "complete": not errors, "errors": errors, "formal_failure_attribution": "NOT_ADJUDICATED"}
    atomic(OUT / f"REVIEWER{reviewer}_TARGET_CHECK.json", result)
    if not errors:
        destination = PACK / "admin" / f"reviewer{reviewer}_TARGET_LABELS.json"
        if destination.exists():
            seal = read(PACK / "admin" / f"reviewer{reviewer}_TARGET_SEAL.json")
            if sha(destination) != seal["labels_sha256"]:
                raise RuntimeError("Sealed target labels changed")
            for sid in cases:
                if sha(stage / sid / "target_response.json") != seal["responses_sha256"][sid]:
                    raise RuntimeError("Target response changed after seal")
            if sha(stage / "target_reviewer.json") != seal["identity_sha256"]:
                raise RuntimeError("Target reviewer changed after seal")
            return result
        atomic(destination, {"identity": person, "labels": labels})
        atomic(PACK / "admin" / f"reviewer{reviewer}_TARGET_SEAL.json", {
            "identity_sha256": sha(stage / "target_reviewer.json"),
            "responses_sha256": {sid: sha(stage / sid / "target_response.json") for sid in cases},
            "labels_sha256": sha(destination)})
    return result


def prepare_adjudication():
    cases = checked_cases()
    for reviewer in (1, 2):
        if not collect_target(reviewer)["complete"]:
            return {"complete": False, "errors": ["two_reviewers_all_stages_required"], "gold_triplets": 0}
    labels = {
        r: {stage: read(PACK / "admin" / f"reviewer{r}_{stage}_LABELS.json")
            for stage in ("TEXT", "GRAPH", "TARGET")} for r in (1, 2)}
    identities = {labels[r]["TEXT"]["identity"]["reviewer_code"] for r in (1, 2)}
    if len(identities) != 2:
        return {"complete": False, "errors": ["two_distinct_humans_required"], "gold_triplets": 0}
    stage = PACK / "adjudicator"
    if stage.exists():
        raise RuntimeError("Adjudication packet already prepared")
    atomic(stage / "reviewer.json", identity_blank())
    for sid, case in cases.items():
        atomic(stage / sid / "evidence.json", {
            "sample_id": sid, "carrier": case["carrier"], "context": case["context"],
            "original_amr": case["original_parse"].get("amr"),
            "target_amr": case["target_amr"],
            "reviewer1": {s: labels[1][s]["labels"][sid] for s in ("TEXT", "GRAPH", "TARGET")},
            "reviewer2": {s: labels[2][s]["labels"][sid] for s in ("TEXT", "GRAPH", "TARGET")}})
        atomic(stage / sid / "final_text_response.json", text_blank(sid))
        atomic(stage / sid / "final_graph_response.json", graph_blank(sid))
        atomic(stage / sid / "final_target_response.json", target_blank(sid))
        save_text(stage / sid / "human_revised.penman", "")
    save_text(stage / "INSTRUCTIONS.md",
              "# 第三人裁决\n\n独立裁决实际文本事实、原图保留/损失、最终修订图与目标写入。\n"
              "不要依据模型分类结果修订。允许 unknown；金标准不确定项保留。\n"
              "格式与前阶段一致。reviewer.json 必须是不同于前两人的真人代号。\n\n"
              "```bash\n.venv-swan-positive/bin/python scripts/prepare_semantic_triplet_review.py collect-adjudication\n```\n")
    atomic(PACK / "admin/ADJUDICATION_INPUT_SEAL.json", {
        str((stage / sid / "evidence.json").relative_to(stage)): sha(stage / sid / "evidence.json") for sid in cases})
    archive(stage, PACK / "adjudicator.zip")
    return {"prepared_adjudication_cases": 12, "gold_triplets": 0}


def collect_adjudication():
    cases = checked_cases()
    stage = PACK / "adjudicator"
    if not stage.exists():
        return {"complete": False, "gold_triplets": 0, "errors": ["adjudication_not_opened"]}
    for rel, expected in read(PACK / "admin/ADJUDICATION_INPUT_SEAL.json").items():
        if sha(stage / rel) != expected:
            raise RuntimeError("Adjudication evidence changed")
    errors, final = [], {}
    try:
        person = check_identity(stage / "reviewer.json")
        codes = {verify_text_seal(r, cases)["identity"]["reviewer_code"] for r in (1, 2)}
        if person["reviewer_code"] in codes:
            raise ValueError("Third distinct human required")
    except (ValueError, TypeError, KeyError):
        errors.append("missing_distinct_human_adjudicator")
        person = {}
    for sid, case in cases.items():
        try:
            text = read(stage / sid / "final_text_response.json")
            graph = read(stage / sid / "final_graph_response.json")
            target = read(stage / sid / "final_target_response.json")
            graph_text = (stage / sid / "human_revised.penman").read_text()
            validate_text(text, case)
            validate_graph(graph, graph_text, case, text)
            validate_target(target, case)
            final[sid] = {"text": text, "graph": graph, "human_revised_graph": graph_text, "target": target}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            errors.append(sid + ":adjudication_pending_or_invalid")
    ready = not errors
    usable = sum(v["text"]["status"] == "confirmed" and
                 v["graph"]["status"] in ("corrected", "unchanged") for v in final.values())
    status = {"at": now(), "complete": ready, "submitted_adjudications": len(final), "planned": 12,
              "gold_triplets": usable if ready else 0, "errors": errors,
              "classification_rescoring": "NOT_RUN"}
    atomic(OUT / "ADJUDICATION_CHECK.json", status)
    if ready:
        atomic(PACK / "admin/ADJUDICATED_LABELS.json", {"identity": person, "labels": final})
        for sid, value in final.items():
            base = PACK / "admin/cases" / sid
            atomic(base / "human_confirmed_text.json", value["text"])
            (base / "human_revised.penman").write_text(value["human_revised_graph"])
            atomic(base / "adjudicated_graph_response.json", value["graph"])
            atomic(base / "adjudicated_target_response.json", value["target"])
            summary = read(base / "comparison.json")
            summary.update(text_status=value["text"]["status"],
                           human_revision_status=value["graph"]["status"],
                           target_realized=value["target"]["target_realized"],
                           original_graph_fidelity=value["graph"]["original_graph_fidelity"],
                           formal_attribution="REQUIRES_CONTEXT_CHECK_AND_FROZEN_CLASSIFIER_RESCORING")
            atomic(base / "comparison.json", summary)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect-text", "collect-graph", "collect-target",
                                          "prepare-adjudication", "collect-adjudication"))
    parser.add_argument("--reviewer", type=int, choices=(1, 2))
    args = parser.parse_args()
    os.umask(0o077)
    if args.action.startswith("collect-") and args.action != "collect-adjudication" and args.reviewer is None:
        parser.error("--reviewer is required for reviewer stages")
    handlers = {
        "prepare": prepare, "collect-text": lambda: collect_text(args.reviewer),
        "collect-graph": lambda: collect_graph(args.reviewer),
        "collect-target": lambda: collect_target(args.reviewer),
        "prepare-adjudication": prepare_adjudication, "collect-adjudication": collect_adjudication,
    }
    print(json.dumps(handlers[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
