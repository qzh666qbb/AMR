"""Prepare a separately sealed v2 blind-review entry and replay the saved 171."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_semantic_fact_graph_readout_v2 import (
    V4, ASSISTANT, DEFAULT_OUT as V2, source_audit, sha, dump, load, verify, evidence_audit,
)
from scripts.collect_semantic_graph_fact_reviews_v2 import collect
from src.core.semantic_fact_layers import actual_signature
from src.core.semantic_fact_carrier import classify_facts
from src.core.semantic_graph_fact_reader_v2 import VERSION, read_graph_facts

DEFAULT_OUT = ROOT / "report_assets/semantic_fact_formal_review_v2_171_20261026"
OLD_UI = ROOT / "report_assets/semantic_fact_formal_review_start_20261026"
CODE = [
    "scripts/prepare_semantic_fact_formal_review_v2.py",
    "scripts/collect_semantic_graph_fact_reviews_v2.py",
    "scripts/collect_semantic_graph_fact_reviews.py",
    "scripts/run_semantic_fact_graph_readout_v2.py",
    "src/core/semantic_graph_fact_reader_v2.py",
    "src/core/semantic_graph_fact_reader.py",
    "src/core/semantic_fact_carrier.py",
    "src/core/semantic_fact_layers.py",
]


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seal(out):
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256.json"}
    dump(out / "SHA256.json", {"algorithm": "SHA256", "self_hash_excluded": True, "files": files})
    verify(out, files)


def prepare(out, port):
    if out.exists():
        raise RuntimeError("Use a new v2 review directory")
    source = source_audit()
    verify(V2, load(V2 / "SHA256.json")["files"])
    out.mkdir(parents=True)
    private = out / "private_review"
    private.mkdir(mode=0o700)
    supporting = out / "supporting_code"
    supporting.mkdir()
    (out / ".gitignore").write_text("private_review/\nindex.html\n__pycache__/\n", encoding="utf-8")
    canonical = load(V4 / "private_review/FORMAL_CANONICAL_INPUTS.json")
    original_readouts = load(V2 / "private_development/GRAPH_ONLY_READOUTS.json")
    subset_path = ASSISTANT / "private_assistant_attribution/NEW_171_CASES.json"
    subset = load(subset_path)
    focus = [{"canonical_index": r["canonical_index"], "case_id": r["case_id"]}
             for r in subset]
    if len(focus) != 171 or len({r["case_id"] for r in focus}) != 171:
        raise RuntimeError("171-row identity mismatch")
    if len(canonical) != 192 or len({r["case_id"] for r in canonical}) != 192:
        raise RuntimeError("192-row canonical identity mismatch")
    dump(private / "FOCUS171_NO_ANSWERS.json", focus)
    shutil.copy2(V4 / "private_review/FORMAL_CANONICAL_INPUTS.json",
                 private / "FORMAL_CANONICAL_INPUTS.json")
    dump(private / "GRAPH_ONLY_READOUTS_V2.json", [
        {"case_id": r["case_id"], "graph_readback": r["v2"]} for r in original_readouts])
    for name in CODE:
        dest = supporting / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    binding = value_digest({
        "reader_version": VERSION, "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "canonical_sha256": sha(private / "FORMAL_CANONICAL_INPUTS.json"),
        "graph_readouts_sha256": sha(private / "GRAPH_ONLY_READOUTS_V2.json"),
        "focus171_sha256": sha(private / "FOCUS171_NO_ANSWERS.json"),
    })
    for number in (1, 2):
        source_packet = V4 / f"private_review/reviewer{number}"
        dest = private / f"reviewer{number}"
        shutil.copytree(source_packet, dest, ignore=shutil.ignore_patterns("__pycache__"))
        dest.chmod(0o700)
        meta = load(dest / "reviewer.json")
        if any(meta[k] is not False for k in (
                "human_attestation", "independent_attestation", "no_model_or_target_answers_used",
                "raw_graph_read_without_text_answers")):
            raise RuntimeError("Source packet is not a blank template")
        meta.update(review_binding_sha256=binding, review_protocol="semantic-formal-review:v2")
        dump(dest / "reviewer.json", meta)
        instructions = (dest / "INSTRUCTIONS.md").read_text(encoding="utf-8")
        instructions += (
            "\n本包绑定独立的 v2 回收入口；不要混用 v1 草稿或回收命令。"
            "\n原图阶段不提供读图器输出或规则提示，文字阶段仍必须在原图锁定后开始。"
            "\nreviewer.json 的 review_binding_sha256 为版本绑定信息，请原样保留。"
            "\n此前已接触助手归因或模型答案的人，不能填写本包的独立盲评声明。\n"
        )
        (dest / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
        for phase in ("raw_graph", "text", "revision"):
            answers = load(dest / (phase + "_answers.json"))
            if len(answers) != 192 or any(r.get("reviewed") is not False for r in answers):
                raise RuntimeError("Source answer packet is not blank")
        zip_path = private / f"reviewer{number}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dest.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)
        zip_path.chmod(0o600)
    html = (OLD_UI / "index.html").read_text(encoding="utf-8")
    if html.count("function fresh(code)") != 1 or html.count("function key(code)") != 1:
        raise RuntimeError("Review UI template drift")
    html = html.replace("图事实 · 独立评审工作台", "图事实 · v2 独立评审工作台")
    html = html.replace("const PHASES=", "const REVIEW_BINDING='" + binding + "';\nconst PHASES=", 1)
    html = html.replace("semantic-formal-review-ui:v1", "semantic-formal-review-ui:v2")
    html = html.replace("reviewer:{reviewer_code:code,", (
        "reviewer:{reviewer_code:code,review_binding_sha256:REVIEW_BINDING,"
        "review_protocol:'semantic-formal-review:v2',"))
    html = html.replace(
        "return'semantic-formal-review:v1:'+SOURCE_SHA+':'+code",
        "return'semantic-formal-review:v2:'+REVIEW_BINDING+':'+code")
    html = html.replace(
        "st.source_sha256!==SOURCE_SHA",
        "st.source_sha256!==SOURCE_SHA||st.reviewer?.review_binding_sha256!==REVIEW_BINDING")
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(OLD_UI / "serve_review.py", out / "serve_review.py")
    freeze = {
        "version": "semantic-fact-formal-review:v2-171-focus",
        "reader_version": VERSION, "rows": 192, "focus_rows": 171,
        "canonical_sha256": sha(private / "FORMAL_CANONICAL_INPUTS.json"),
        "graph_readouts_sha256": sha(private / "GRAPH_ONLY_READOUTS_V2.json"),
        "focus171_sha256": sha(private / "FOCUS171_NO_ANSWERS.json"),
        "review_binding_sha256": binding, "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "entry_sha256": sha(out / "index.html"), "server_sha256": sha(out / "serve_review.py"),
        "structural_exclusions": 20, "focus171_structural_exclusions": 19,
        "independent_human_reviewers_required": 2,
        "source_v4_freeze_sha256": sha(V4 / "FREEZE.json"),
        "source_v2_result_sha256": sha(V2 / "RESULT.json"),
        "assistant_subset_sha256": sha(subset_path),
        "assistant_revisions_not_in_independent_review_packets": True,
        "model_or_target_answers_not_in_independent_review_packets": True,
    }
    dump(out / "FREEZE.json", freeze)
    blank_audit = collect(out, private / "reviewer1", private / "reviewer2",
                          private / "initial_collection")
    if blank_audit["all192"]["formal_complete"] or blank_audit["focus171"]["formal_complete"]:
        raise RuntimeError("Blank packets incorrectly counted as human evidence")
    dump(out / "FORMAL_ENTRY_AUDIT.json", blank_audit)
    # The extraction source is an exposed assistant container. Only its graph
    # and public contract fields enter the reader; this is never a blind review.
    replay_inputs = [
        {"case_id": r["case_id"], "contract": r["public_contract"],
         "revised_penman": r["revised_penman"]} for r in subset]
    readbacks = []
    for row in replay_inputs:
        result = read_graph_facts(row["revised_penman"], row["contract"])
        evidence_audit(result)
        readbacks.append({"case_id": row["case_id"], "graph_readback": result})
    dump(private / "ASSISTANT_REVISED_GRAPH_READOUTS_171.json", readbacks)
    replay_hash = sha(private / "ASSISTANT_REVISED_GRAPH_READOUTS_171.json")
    # Save before the fact comparison phase; reader inputs have no references.
    dump(out / "REPLAY_PHASE_BOUNDARY.json", {
        "rows": 171, "graph_readouts_sha256": replay_hash,
        "reader_inputs": ["saved_revised_penman", "public_roles_object_contract"],
        "assistant_container_exposure": True, "independent_review": False,
        "fact_comparison_performed_before_this_save": False,
    })
    subset_ref = {r["case_id"]: r for r in load(subset_path)}
    original_comparisons = {
        r["case_id"]: r for r in load(V2 / "private_development/COMPARISONS.json")}
    results = []
    for row in readbacks:
        reference = subset_ref[row["case_id"]]
        original = original_comparisons[row["case_id"]]
        readback = row["graph_readback"]
        if readback["original_penman_sha256"] != reference["revised_penman_sha256"]:
            raise RuntimeError("Saved assistant revision SHA differs")
        text_facts = reference["text_fact_candidates"]
        text_sig = actual_signature(text_facts)
        revised_match = text_sig is not None and actual_signature(readback["facts"]) == text_sig
        eligible = reference["structurally_eligible"]
        text_symbol = classify_facts(text_facts, reference["public_contract"]).symbol if text_sig else None
        results.append({
            "canonical_index": reference["canonical_index"], "case_id": reference["case_id"],
            "condition": reference["condition"], "structurally_eligible": eligible,
            "v1_original_match": original["v1_four_fact_match"],
            "v2_original_match": original["v2_four_fact_match"],
            "v2_original_primary_prediction": original["v2_primary_prediction"],
            "saved_assistant_revision_match_v1": reference["candidate_four_fact_match"],
            "v2_saved_revision_match": revised_match,
            "v2_saved_revision_primary_prediction": readback["prediction"] if eligible else None,
            "v2_saved_revision_raw_prediction": readback["prediction"],
            "assistant_text_symbol": text_symbol,
            "category": reference["category"], "v2_saved_revision_facts": readback["facts"],
            "formal_confirmed": False, "independent_gold": False,
        })
    dump(private / "DEVELOPMENT_COMPARISONS_171.json", results)
    columns = [k for k in results[0] if k != "v2_saved_revision_facts"]
    with (private / "CASES171.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in results)
    restored_original = [r["canonical_index"] for r in results
                         if r["structurally_eligible"] and not r["v1_original_match"] and r["v2_original_match"]]
    summary = {
        "status": "V2_FORMAL_ENTRY_PREPARED_INDEPENDENT_EVIDENCE_PENDING",
        "reader_version": VERSION, "focus_denominator": 171, "full_denominator": 192,
        "focus_structurally_eligible": 152, "focus_structural_exclusions": 19,
        "v1_original_primary_four_fact_matches": sum(
            r["structurally_eligible"] and r["v1_original_match"] for r in results),
        "v2_original_primary_four_fact_matches": sum(
            r["structurally_eligible"] and r["v2_original_match"] for r in results),
        "newly_restored_original_indexes": restored_original,
        "v2_saved_revision_raw_four_fact_matches": sum(r["v2_saved_revision_match"] for r in results),
        "v2_saved_revision_primary_four_fact_matches": sum(
            r["structurally_eligible"] and r["v2_saved_revision_match"] for r in results),
        "saved_revision_remaining_eligible_erasures": [
            r["canonical_index"] for r in results
            if r["structurally_eligible"] and r["v2_saved_revision_primary_prediction"] is None],
        "saved_revision_eligible_complete_disagreements": [
            r["canonical_index"] for r in results if r["structurally_eligible"]
            and r["v2_saved_revision_primary_prediction"] is not None and not r["v2_saved_revision_match"]],
        "original_eligible_success_regressions": [
            r["canonical_index"] for r in results
            if r["structurally_eligible"] and r["v1_original_match"] and not r["v2_original_match"]],
        "full_formal_confirmed": 0, "focus_formal_confirmed": 0,
        "formal_entry_collection_audit": blank_audit,
        "new_parser_provider_detector_embedding_calls": 0,
        "saved_assistant_revision_reader_calls": 171,
        "development_reference": "NONINDEPENDENT_ASSISTANT_TEXT_FACT_CANDIDATES",
        "cannot_claim_all171_recovered": True,
    }
    dump(out / "RESULT.json", summary)
    original = summary["v2_original_primary_four_fact_matches"]
    repaired = summary["v2_saved_revision_primary_four_fact_matches"]
    remaining = summary["saved_revision_remaining_eligible_erasures"]
    relative = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    report = f"""# v2 正式独立归因入口与 171 条恢复核验

本包绑定 semantic-graph-fact-reader:v2-development。独立评审入口已准备；
现有两个正式答案包均为空，正式确认 0/171，完整队列仍为 0/192。
“171 条”精确绑定此前 NEW_171_CASES.json，排除首批 21 条，不是待定 23 条或全部 192 条。

## 171 条开发核验

| 全量分母 171，保留结构排除 | 完整四事实与助手一致 |
|---|---:|
| v1 原图 | {summary['v1_original_primary_four_fact_matches']} |
| v2 原图 | {original} |
| v2 已保存的助手修订图 | {repaired} |

171 条中有 152 条结构合格、19 条冻结排除；全部继续保留在分母。
原图仅因 v2 规则新增恢复 {len(restored_original)} 条，索引为 {restored_original}。
首批 21 条中的索引 26 是另外一条作用域恢复，因此全 192 条新增 17 与本批新增 16 不矛盾。
原有合格成功案例退化为 {len(summary['original_eligible_success_regressions'])}。

助手修订图仍有合格擦除 {len(remaining)} 条：{remaining}。
另有 {len(summary['saved_revision_eligible_complete_disagreements'])} 条合格非擦除四事实不一致。
原图恢复和助手修订后恢复是两个不同结果；助手修订不证明原图已恢复，更不能证明正式责任层。
不能宣称 171 条全部恢复。

每个新读图调用仅接收图和公共角色／物品，先保存 171 条修订图读出，再计算助手事实对照；
抽取容器包含先前助手材料，已明确记录非独立暴露。没有把助手材料导入真人答案。
171 条新的证据三元组均核对为各自保存图中的字面三元组，未调用 parser、provider、detector 或 GPU。

## 正式评审与回收

两名尚未接触助手／模型答案的真实评审，分别完成全 192 条原图、文字、独立修订图三阶段。
每人 576 条答案；先锁定原图，再加载文字，最后独立修订；分歧由第三人裁决。
正式报告单独汇总本批 171，完整 192 分母和 20 条冻结排除仍保留。

原图页面只包含图和公共合同，没有助手修订、文字事实、v2 预测、规则提示或计划符号。
页面草稿与 reviewer.json 绑定 v2 版本，不能混入 v1 草稿。
回收器复用冻结的真人声明、时间顺序、SHA 和字面证据校验；新增 v2 绑定与两套修订图共同核验。
未填写的真实声明保持 false。初始回收核验合法独立证据为 [0, 0]；正式确认 0。

页面：<http://127.0.0.1:{port}/>。页面启动状态以 START.md 和 STATUS.json 的实测结果为准。
原图离线入口：[index.html](index.html)。
[评审包 1](private_review/reviewer1.zip) · [评审包 2](private_review/reviewer2.zip)。

```bash
PYTHONPATH=/tmp/amr_fact_readout_deps python3 scripts/collect_semantic_graph_fact_reviews_v2.py \\
  --experiment {relative} \\
  --reviewer1 /path/to/completed/reviewer1 \\
  --reviewer2 /path/to/completed/reviewer2 \\
  --out /tmp/semantic_fact_v2_human_collection
```

评审收集器只使用独立提交事实、原始规范输入和封印 v2 原图读出，
不读取助手事实或行政计划符号。只有真实独立材料通过后，original_restoration_verified /
revised_restoration_verified 才能计数为正式恢复。机器不能证明真人身份或自动证明语义蕴含。

## 证据文件

- [开发核验汇总](RESULT.json)
- [171 条逐条开发 CSV](private_review/CASES171.csv)
- [171 条逐条开发 JSON](private_review/DEVELOPMENT_COMPARISONS_171.json)
- [初始正式回收核验](FORMAL_ENTRY_AUDIT.json)
- [版本与输入冻结](FREEZE.json)
- [源 SHA 核验](SOURCE_SHA256.json)
- [交付 SHA](SHA256.json)

私有文本、图和答案位于 private_review，不属于公开材料。原 v1、v4、
首次正式入口、全 192 助手包和 v2 开发包均未修改。
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "START.md").write_text(
        f"# v2 正式独立评审入口\n\n启动服务后访问 <http://127.0.0.1:{port}/>。\n"
        "两位独立真人的空白包和回收说明见 [报告](REPORT.md)。现有正式确认 0/171。\n\n"
        f"```bash\nPYTHONPATH=/tmp/amr_fact_readout_deps python3 {relative}/serve_review.py --port {port}\n```\n",
        encoding="utf-8")
    after = source_audit()
    verify(V2, load(V2 / "SHA256.json")["files"])
    if after["v2_code_sha256"] != source["v2_code_sha256"]:
        raise RuntimeError("v2 source code changed during preparation")
    dump(out / "SOURCE_SHA256.json", {
        "previous_sources": source, "verified_after": after["verified_at"],
        "new_code_sha256": freeze["code_sha256"],
        "source_old_ui_sha256": sha(OLD_UI / "index.html"),
        "source_old_server_sha256": sha(OLD_UI / "serve_review.py"),
        "source_assistant_subset_sha256": sha(subset_path),
    })
    seal(out)
    print(json.dumps({k: summary[k] for k in (
        "focus_denominator", "v1_original_primary_four_fact_matches", "v2_original_primary_four_fact_matches",
        "v2_saved_revision_primary_four_fact_matches", "saved_revision_remaining_eligible_erasures",
        "full_formal_confirmed", "focus_formal_confirmed")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    prepare(args.out.resolve(), args.port)
