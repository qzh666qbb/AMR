"""Package a single assistant's annotations of frozen semantic triplets.

This entry point only reads prior materials and authored local annotations.
It performs no inference, classifier scoring, human collection, or networking.
Raw texts, candidate graphs, and row-level annotations stay gitignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import warnings

import penman
from penman.models.amr import model as AMR_MODEL

from prepare_semantic_triplet_review import checked_cases, normalized_graph

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/semantic_triplet_review1/private_review"
OUT = ROOT / "experiments/semantic_triplet_model_review1"
PACK = OUT / "private_review"
LABELS = {
    "EXPLICIT_CONTRACT_CONFLICT": "明确合同不符",
    "UNCERTAIN": "待定",
    "SUPPORTED_ON_CHECKED_REQUIREMENTS": "检查范围内支持",
    "CORE_ACCEPTABLE": "核心意义可接受",
    "EXPLICIT_SEMANTIC_ERROR": "明确语义错误",
    "ONTOLOGY_UNCERTAIN": "实体类型待定",
    "retained_original": "保留原图",
    "minimal_semantic_revision": "修订明确错误",
    "conservative_type_generalization": "保守泛化类型",
    "semantic_and_structure_revision": "修订词义与结构",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def evidence(case, quote, source="carrier"):
    text = case[source]
    start = text.find(quote)
    require(quote and start >= 0, "Evidence absent from frozen input")
    return {"source": source, "start": start, "end": start + len(quote), "quote": quote}


def expand_evidence(value, case):
    if isinstance(value, list):
        return [expand_evidence(x, case) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: expand_evidence(v, case) for k, v in value.items() if k != "quotes"}
    if "quotes" in value:
        result["evidence"] = [evidence(case, q) for q in value["quotes"]]
    return result


def verify_evidence(value, case):
    if isinstance(value, list):
        return sum(verify_evidence(x, case) for x in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    if {"quote", "start", "end", "source"} <= set(value):
        require(case[value["source"]][value["start"]:value["end"]] == value["quote"],
                "Evidence span mismatch")
        count += 1
    for item in value.values():
        count += verify_evidence(item, case)
    return count


def original_graph(text):
    # A permissive read is solely for inspecting original triples. It does not
    # override the frozen parser's stricter validity/warning result.
    logger = logging.getLogger("penman")
    previous = logger.disabled
    logger.disabled = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return penman.decode(text, model=AMR_MODEL)
    finally:
        logger.disabled = previous


def scope_checks(graph, case):
    concepts = {t.source: t.target for t in graph.instances()}
    names = []
    degree_frames = 0
    for node, concept in concepts.items():
        if concept == "name":
            parts = sorted(((int(r[3:]), json.loads(t)) for s, r, t in graph.triples
                            if s == node and re.fullmatch(r":op\d+", r)))
            name = " ".join(word for _, word in parts)
            require(name and name in case["carrier"], "Candidate invents a name")
            names.append(name)
        if concept == "have-degree-91":
            for role in (":ARG1", ":ARG2", ":ARG3"):
                require(sum(s == node and r == role for s, r, t in graph.triples) == 1,
                        "Candidate degree frame has missing/repeated domain/attribute/degree")
            degree_frames += 1
    require(not AMR_MODEL.errors(graph), "AMR role/model errors")
    return {"names_in_parser_input": names, "degree_frames_checked": degree_frames,
            "scope_check_limit": "names and authored edits; not full semantic certification"}


def human_pending(cases):
    for sid in cases:
        folder = SOURCE / "admin/cases" / sid
        require(not (folder / "human_revised.penman").read_text().strip(),
                "Human revision no longer blank")
        text = read(folder / "human_confirmed_text.json")
        require(text["status"] == "PENDING_HUMAN_REVIEW" and text["facts"] is None,
                "Human text attestation changed")
        for reviewer in (1, 2):
            folder_r = SOURCE / f"reviewer{reviewer}"
            identity = read(folder_r / "reviewer.json")
            require(not identity["human_attestation"] and not identity["independent_attestation"],
                    "Human identity unexpectedly attested")
            response = read(folder_r / "01_text" / sid / "text_response.json")
            require(response["status"] == "" and response["facts"] == [],
                    "Human text response is not blank")
    return {"confirmed_texts": 0, "revised_graphs": 0, "attested_reviewers": 0,
            "reviewer_text_responses_blank": 2 * len(cases)}


def sample_markdown(case, annotation, graph_text, delta, checks):
    def quotes(items):
        return "；".join(f'[{e["start"]},{e["end"]}) `{e["quote"]}`'
                        for e in items) or "见原记录或目标合同"
    facts = "\n".join(f'- {f["meaning"]} 证据：{quotes(f["evidence"])}'
                      for f in annotation["facts"])
    issues = "\n".join(
        f'- **{i["code"]}**：{i["reason"]} 证据：{quotes(i["evidence"])}。'
        f' 原图规范方向三元组：`{json.dumps(i["graph_triples"], ensure_ascii=False)}`'
        for i in annotation["parsing"]["issues"]) or "未发现本轮可高把握定位的明确错误。"
    pending = "\n".join(f'- {x}' for x in annotation["parsing"]["open_questions"])
    return (
        f'# 模型辅助对照样本 {case["sample_id"]}\n\n'
        '单助手事后审查；已见目标和原预测。真人确认与修订仍为 0，候选不是语义金标准。\n\n'
        f'抽样层：计划 {case["planned_symbol"]} / 原 W 输出 '
        f'{case["original_prediction"]}（0=A，1=B，2=C，3=D，null=擦除）。\n\n'
        f'## 原解析器完整输入\n\n{case["carrier"]}\n\n'
        f'## 原段落（仅用于读者理解）\n\n{case["context"]}\n\n'
        f'## 实际意义与证据\n\n{annotation["realized_meaning"]}\n\n'
        f'言语行为：{annotation["speech_act"]}；判定需要上下文：'
        f'{annotation["context_required"]}。问题/请求不解释为已发生的断言。\n\n{facts}\n\n'
        f'## 写入合同\n\n```penman\n{case["target_amr"]}\n```\n\n'
        f'{case["target_entities"]}\n\n'
        f'判定：**{annotation["writing"]["status"]}**；把握：'
        f'{annotation["writing"]["confidence"]}。\n\n{annotation["writing"]["reason"]}\n\n'
        f'## 原解析图\n\n```penman\n{case["original_parse"]["amr"]}\n```\n\n'
        f'原冻结严格有效性：{case["original_parse"]["valid"]}。'
        '完整 warning 保留于 original_parse_record.json。\n\n'
        f'## 原图语义核验\n\n判定：**{annotation["parsing"]["status"]}**；'
        f'把握：{annotation["parsing"]["confidence"]}。\n\n'
        f'{annotation["parsing"]["reason"]}\n\n{issues}\n\n'
        f'## 模型修订候选\n\n处理：{annotation["revision"]["action"]}。\n\n'
        f'{annotation["revision"]["reason"]}\n\n```penman\n{graph_text}\n```\n\n'
        f'规范方向三元组变更：删除 {len(delta["removed"])}，新增 {len(delta["added"])}；'
        '完整差异见 graph_delta.json。变量名变更不等同于独立语义修改。\n\n'
        '候选通过格式、模型角色、root、变量、连通性、roundtrip、'
        '姓名范围及相关 degree 检查；格式有效不证明语义正确。\n\n'
        f'## 尚未裁定\n\n{pending}\n\n'
        '分类归因：**UNRESOLVED**。修订图未在原冻结神经分类器下重新评分，'
        '因此没有 corrected prediction、分类恢复率或“只由分类导致”的判定。\n'
    )


def build():
    cases = checked_cases()
    raw = read(PACK / "ANNOTATIONS.json")
    revisions = read(PACK / "REVISED_GRAPHS.json")
    require(len(cases) == 12 and set(raw["cases"]) == set(cases), "Sample coverage changed")
    require(raw["reviewer_type"] == "single_assistant_model_review"
            and not any(raw[k] for k in ("human_attestation", "blind_review", "independent_gold")),
            "Reviewer provenance invalid")
    before = inventory(SOURCE)
    snapshot = PACK / "SOURCE_SNAPSHOT.json"
    if snapshot.exists():
        require(read(snapshot) == before, "Prior review packet changed since first build")
    else:
        put(snapshot, before)
    pending = human_pending(cases)
    summaries, audits = [], []
    for sid, case in sorted(cases.items()):
        annotation = expand_evidence(raw["cases"][sid], case)
        require(annotation["context_required"] is False, "Unexpected context-only assertion")
        for absence in annotation["writing"].get("absence_checks", []):
            require(absence["absent"] not in case[absence["source"]], "Absence check failed")
            require(absence["absent"] in case[absence["required_by"]], "Missing requirement source")
        target = normalized_graph(case["target_amr"])
        for triple in annotation["writing"]["target_triples"]:
            require(tuple(triple) in target.triples, "Cited target triple absent")
        original = original_graph(case["original_parse"]["amr"])
        for issue in annotation["parsing"]["issues"]:
            for triple in issue["graph_triples"]:
                require(tuple(triple) in original.triples, "Cited original triple absent")
        action = annotation["revision"]["action"]
        require((sid in revisions) == (action != "retained_original"), "Revision coverage mismatch")
        graph_text = revisions.get(sid, case["original_parse"]["amr"])
        graph = normalized_graph(graph_text)
        checks = scope_checks(graph, case)
        concepts = {t.target for t in graph.instances()}
        if "bearing" in case["carrier"]:
            require("bearing" in concepts and "bear-01" not in concepts
                    and '"Susan"' not in graph_text, "Bearing revision ineffective")
        if sid in ("821b0c2efbfb", "cecda439c5fb"):
            require("felt" in concepts and "feel-05" not in concepts, "Material revision ineffective")
        delta = {"removed": sorted(set(original.triples) - set(graph.triples)),
                 "added": sorted(set(graph.triples) - set(original.triples))}
        annotation.update({
            "sample_id": sid, "reviewer_type": raw["reviewer_type"],
            "human_attestation": False, "independent_gold": False, "blind_review": False,
            "classification": {"status": "UNRESOLVED", "corrected_graph_scored": False,
                               "corrected_prediction": None},
            "original_strict_valid": case["original_parse"]["valid"],
            "source_provenance": {"carrier_sha256": case["text_sha256"],
                                  "case_sha256": sha(SOURCE / "admin/cases" / sid / "case.json"),
                                  "original_freeze_sha256": case["original_parse"]["freeze_sha256"]},
        })
        evidence_count = verify_evidence(annotation, case)
        folder = PACK / "cases" / sid
        put(folder / "source_case.json", case)
        put(folder / "original_parse.penman", case["original_parse"]["amr"])
        put(folder / "original_parse_record.json", case["original_parse"])
        put(folder / "model_revised.penman", graph_text)
        put(folder / "model_annotation.json", annotation)
        put(folder / "graph_delta.json", delta)
        put(folder / "MODEL_COMPARISON.md", sample_markdown(
            case, annotation, graph_text, delta, checks))
        audits.append({"sample_id": sid, "candidate_format_pass": True,
                       "evidence_spans_checked": evidence_count, **checks})
        summaries.append({"sample_id": sid, "planned_symbol": case["planned_symbol"],
                          "original_W_prediction": case["original_prediction"],
                          "writing": annotation["writing"]["status"],
                          "parsing": annotation["parsing"]["status"],
                          "revision": action, "original_strict_valid": case["original_parse"]["valid"],
                          "classification": "UNRESOLVED"})
    require(before == inventory(SOURCE), "Original review packet mutated")
    aggregate = {
        "sample_count": len(cases), "reviewer_type": raw["reviewer_type"],
        "selection": "purposive: four symbols by correct/wrong/erased; not population prevalence",
        "writing": dict(Counter(s["writing"] for s in summaries)),
        "parsing": dict(Counter(s["parsing"] for s in summaries)),
        "revision": dict(Counter(s["revision"] for s in summaries)),
        "writing_and_explicit_parsing_conflict": sum(
            s["writing"] == "EXPLICIT_CONTRACT_CONFLICT"
            and s["parsing"] == "EXPLICIT_SEMANTIC_ERROR" for s in summaries),
        "original_strict_invalid": sum(not s["original_strict_valid"] for s in summaries),
        "classification_unresolved": len(cases), "corrected_graph_scored": 0,
        "candidate_format_valid": len(cases), "human_review": pending,
        "no_new_parser_provider_requests": True, "no_classifier_scoring": True,
        "original_readout_NoGo_unchanged": True,
    }
    put(OUT / "RESULT.json", aggregate)
    put(PACK / "CASE_INDEX.json", summaries)
    table = ("| 样本 | 写入 | 原图语义 | 图处理 |\n|---|---|---|---|\n" + "\n".join(
        f'| [{s["sample_id"]}](cases/{s["sample_id"]}/MODEL_COMPARISON.md) | '
        f'{LABELS[s["writing"]]} | {LABELS[s["parsing"]]} | '
        f'{LABELS[s["revision"]]} |' for s in summaries))
    put(PACK / "MODEL_COMPARISON_SAMPLES.md",
        "# 12 条模型辅助三方对照样本册\n\n"
        "真实原文、原解析、模型意义解释与修订候选均已逐条整理。"
        "单助手已见目标和原输出；不是盲评或真人金标准。"
        "原真人表单全部保持空白。\n\n" + table + "\n\n" +
        "\n\n---\n\n".join((PACK / "cases" / s["sample_id"] / "MODEL_COMPARISON.md").read_text()
                          for s in summaries))
    report_table = table.replace("(cases/", "(private_review/cases/")
    put(OUT / "REPORT.md",
        "# 语义写入—解析—分类的模型辅助分层审查\n\n"
        "2026-10-25。已审查原封存的 12 条样本；本轮发现 7 条明确写入合同不符、"
        "4 条原图存在明确语义错误，分类阶段的独立责任仍未裁定。"
        "两类问题重叠 3 条，不能把计数相加。"
        "这些是单助手的事后解释，不能替代真人确认，也不能估计总体错误率。\n\n"
        "## 输入与判定依据\n\n"
        "样本固定为四个符号各一条原模型正确、错误、擦除，共 12 条，覆盖 9 篇和 12 个模板。"
        "载体整句、原段落、目标合同与原 parser 记录来自既有三方包；"
        "抽样中的“模型正确”只表示预测命中计划标签，不表示语义写入成功。"
        "原句级 parser 仅收到载体，未收到整段上下文。\n\n"
        "判定分别使用原文字符证据、目标角色/姓名合同、原图规范方向三元组。"
        "逐条证据与不确定性见[完整样本册](private_review/MODEL_COMPARISON_SAMPLES.md)。"
        "输入身份由既有 SHA manifest 校验，完整源包在本轮前后逐文件哈希一致。"
        "本轮聚合可由 [RESULT.json](RESULT.json)、私有 CASE_INDEX 与 audit 复核。\n\n"
        "## 分层结果\n\n"
        "| 维度 | 本组结果 | 解释 |\n|---|---|---|\n"
        "| 写入 | 7 条明确合同不符 | 疑问类别/作用域、topic 与执行者混淆，或必需姓名缺失 |\n"
        "| 写入 | 1 条在检查要求内支持 | 请求、礼貌及内容 domain 相容；不证明全模板或解码 |\n"
        "| 写入 | 4 条不确定 | 抽象 V 缺少 sense，或内嵌 wh 与码本直接未知节点不易等同 |\n"
        "| 原图语义 | 4 条明确错误 | 非人部件误成人/事件、材料误为感觉、性质或 degree 错接 |\n"
        "| 原图语义 | 7 条核心可接受 | 未发现高把握错误；不代表全图真人确证 |\n"
        "| 原图语义 | 1 条本体待定 | 撤去无充分依据的 city 类型，保留实体类型不确定性 |\n"
        "| 原冻结通道 | 1 条严格无效 | 原 deserialize warning 保留；不因新图有效解除 |\n"
        "| 修订候选 | 4 条明确错误修订、1 条保守泛化、7 条保留原图 | 12/12 格式及范围检查通过 |\n"
        "| 分类 | 12 条 UNRESOLVED | 修订图没有调用原冻结神经分类器重算 |\n"
        "| 真人 | 0 条确认、0 条修订 | 原两位评审者表单及声明均保持空白 |\n\n"
        "## 可定位的故障与解释边界\n\n"
        "写入方面，参数疑问与是否疑问不同；在请求中担任执行者也不等于请求内容的 topic。"
        "必需姓名的字符级缺失可直接由文本与合同核验。"
        "这些问题发生在文本/合同接口，修正图中词义无法补回未写入的目标关系。"
        "解析方面，家具转盘部件被误拆为人名与事件，材料被误解为感觉谓词，"
        "以及性质和比较 domain 的错接，均有具体词组与原图三元组证据。"
        "修订只针对这些证据，未用目标图强行补上角色或姓名。\n\n"
        "官方 [AMR Questions](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#questions)"
        "规定嵌入 whether 可用 truth-value 与 polarity-of；某些嵌入 wh 可使用名词/反向角色。"
        "因此不能把“内层没有 amr-unknown”单独认定为解析故障。"
        "[Core roles](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#core-roles)"
        "说明 ARG 编号依赖具体 predicate sense；抽象 V 未固定 sense 的差异要保留不确定性。"
        "[Degree](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#degree)"
        "用于核对比较的 domain、attribute 和 degree；"
        "本轮引用规范的检索字节 SHA 与摘录存于私有 AMR_SPEC_REFERENCE.json。"
        "这些规范判断依据一份官方来源，未取得第三位标注者的独立语义验证。\n\n"
        "## 逐条索引\n\n" + report_table + "\n\n"
        "## 分类归因仍缺少的对照\n\n"
        "同一冻结分类器对原图与修订候选图的成对评分，才能测试明确图修订能否改变读出。"
        "即使修订后标签改变，也不能据此自动确认文本已经满足写入合同，"
        "或证明修订图是独立语义金标准。"
        "本轮只完成已授权的本地意义审查与候选图整理，没有运行该对照，"
        "没有新 parser/provider 调用或后台实验；原 readout1 No-Go 不变。\n\n"
        "## 校验与复核\n\n"
        "候选通过 PENMAN/AMR 模型角色、root、变量唯一性、连通性与 roundtrip；"
        "证据字符范围、姓名在输入内、相关 degree domain 唯一性及原图/目标三元组引用均已检查。"
        "这只确认文件可处理、证据引用准确，不能代替语义裁决。"
        "公开摘要与校验入口不含封存全文；"
        "private_review 含原始文本、原图与逐条候选，已被 gitignore 排除。\n\n"
        "```bash\n.venv-swan-positive/bin/python scripts/build_semantic_triplet_model_review.py --verify\n```\n")
    put(OUT / "README.md",
        "# 模型辅助语义三方对照\n\n"
        "阅读 [REPORT.md](REPORT.md) 查看分层发现；"
        "[完整样本册](private_review/MODEL_COMPARISON_SAMPLES.md) 含 12 条原文、"
        "原图、解释、修订候选及字符证据。\n\n"
        "不是真人/独立/盲评金标准。分类未重算，原实验 No-Go 不变。"
        "原 semantic_triplet_review1 的真人待办全部保持空白。\n")
    put(PACK / "INPUT_PROVENANCE.json", {
        "source_tree_sha256": before,
        "protocol_sha256": sha(OUT / "PROTOCOL.md"),
        "annotations_sha256": sha(PACK / "ANNOTATIONS.json"),
        "revision_input_sha256": sha(PACK / "REVISED_GRAPHS.json"),
        "spec_reference_sha256": sha(PACK / "AMR_SPEC_REFERENCE.json"),
        "build_source_sha256": sha(Path(__file__)),
        "case_original_freeze_sha256": sorted(
            {c["original_parse"]["freeze_sha256"] for c in cases.values()}),
    })
    put(PACK / "AUDIT.json", {"source_tree_unchanged": True, "cases": audits,
                             "human_review": pending})
    put(OUT / "AUDIT.json", {"source_cases_sha_pass": 12, "source_tree_unchanged": True,
                            "candidate_checks_pass": 12,
                            "evidence_spans_checked": sum(x["evidence_spans_checked"] for x in audits),
                            "human_inputs_unchanged": True, "human_review": pending,
                            "classification_scored": 0})
    put(OUT / "CLOSEOUT.json", {
        "status": "MODEL_REVIEW_COMPLETE_CLASSIFICATION_UNRESOLVED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pending_requests": 0, "background_tasks_started": 0,
        "human_review_completed": 0, "corrected_graph_scored": 0,
        "original_experiment_status": "NO_GO_UNCHANGED"})
    manifest = inventory(OUT)
    manifest.pop("DELIVERY_SHA256.json", None)
    manifest["../../scripts/" + Path(__file__).name] = sha(Path(__file__))
    put(OUT / "DELIVERY_SHA256.json", manifest)
    return aggregate


def verify():
    checked_cases()
    manifest = read(OUT / "DELIVERY_SHA256.json")
    for rel, expected in manifest.items():
        require(sha(OUT / rel) == expected, "Delivery SHA mismatch: " + rel)
    require(inventory(SOURCE) == read(PACK / "SOURCE_SNAPSHOT.json"),
            "Source packet no longer matches original snapshot")
    cases = checked_cases()
    human_pending(cases)
    for sid, case in cases.items():
        folder = PACK / "cases" / sid
        graph = normalized_graph((folder / "model_revised.penman").read_text())
        scope_checks(graph, case)
        annotation = read(folder / "model_annotation.json")
        verify_evidence(annotation, case)
        require(not annotation["human_attestation"]
                and not annotation["classification"]["corrected_graph_scored"],
                "Review provenance changed")
    ignored = subprocess.run(
        ["git", "check-ignore", str(PACK / "MODEL_COMPARISON_SAMPLES.md")],
        cwd=ROOT, capture_output=True, text=True)
    require(ignored.returncode == 0, "Raw review package is not gitignored")
    return {"delivery_sha_verified": len(manifest), "candidate_graphs_verified": len(cases),
            "source_tree_unchanged": True, "human_completed": 0,
            "classifier_calls": 0, "raw_review_gitignored": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else build(), ensure_ascii=False, indent=2))
