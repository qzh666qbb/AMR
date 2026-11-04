"""Separate v2 entrypoint: graph-only reading or sealed development replay.

Replay persists all original-graph outputs before decoding assistant references.
It never imports revised graphs into the reader or rewrites the v1 experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.semantic_fact_carrier import classify_facts
from src.core.semantic_fact_layers import actual_signature
from src.core.semantic_graph_fact_reader import read_graph_facts as read_v1
from src.core.semantic_graph_fact_reader_v2 import VERSION, RULES, read_graph_facts

V4 = ROOT / "experiments/semantic_fact_layers_v4_readout1"
ASSISTANT = ROOT / "report_assets/semantic_fact_assistant_attribution_all192_20261026"
ANALYSIS = ROOT / "report_assets/semantic_fact_unresolved23_analysis_20261026"
DEFAULT_OUT = ROOT / "report_assets/semantic_graph_fact_reader_v2_20261026"
CODE = [
    "src/core/semantic_graph_fact_reader_v2.py",
    "scripts/run_semantic_fact_graph_readout_v2.py",
    "src/core/semantic_graph_fact_reader.py",
    "src/core/semantic_fact_carrier.py",
    "src/core/semantic_fact_layers.py",
]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_equal(left, right):
    # In-memory literal triples are tuples; persisted JSON triples are lists.
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def verify(folder, mapping):
    for name, expected in mapping.items():
        if sha(folder / name) != expected:
            raise RuntimeError("Sealed file changed: " + str(folder / name))
    return len(mapping)


def source_audit():
    freeze = load(V4 / "FREEZE.json")
    def manifest(folder):
        value = load(folder / "SHA256.json")
        return value["files"] if "files" in value else value

    maps = {
        str(V4.relative_to(ROOT)): manifest(V4),
        str(ASSISTANT.relative_to(ROOT)): manifest(ASSISTANT),
        str(ANALYSIS.relative_to(ROOT)): manifest(ANALYSIS),
        **freeze["source_maps"],
    }
    counts = {name: verify(ROOT / name, mapping) for name, mapping in maps.items()}
    verify(ROOT, freeze["code_sha256"])
    return {
        "verified_at": utc(), "sealed_file_counts": counts, "maps": maps,
        "frozen_v1_code_sha256": freeze["code_sha256"],
        "v2_code_sha256": {name: sha(ROOT / name) for name in CODE},
        "v4_freeze_sha256": sha(V4 / "FREEZE.json"),
        "graph_inputs_sha256": sha(V4 / "private_review/GRAPH_INPUTS_NO_TARGET.json"),
        "reference_payloads_decoded_in_source_audit": False,
    }


def evidence_audit(result):
    original = {tuple(t) for t in result["normalized_triples"]}
    proofs = list(result["evidence_triples"].values())
    for event in result["events"]:
        proofs += list(event["evidence"].values())
        proofs += [hint["evidence_triples"] for hint in event.get("recipient_hints", [])]
    proofs += [trace["evidence_triples"] for trace in result.get("rule_trace", [])]
    proofs += [event["evidence_triples"] for event in result.get("unsupported_events", [])]
    if not all(tuple(triple) in original for proof in proofs for triple in proof):
        raise RuntimeError("Evidence contains a derived/nonliteral triple")
    if result["prediction"] is not None and any(not result["evidence_triples"][k] for k in result["evidence_triples"]):
        raise RuntimeError("A complete prediction has an empty field proof")


def run_tests(path, out):
    spec = importlib.util.spec_from_file_location("reader_v2_counterexamples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    names = []

    def enumerate_tests(value):
        if isinstance(value, unittest.TestSuite):
            for child in value:
                enumerate_tests(child)
        else:
            names.append(value.id())

    enumerate_tests(suite)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / "TEST_LOG.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "successful": result.wasSuccessful(), "test_ids": names,
        "source_sha256": sha(path),
        "synthetic_rule_counterexamples_are_not_evaluation_data": True,
    }
    dump(out / "TEST_RESULT.json", summary)
    if not result.wasSuccessful():
        raise RuntimeError("Reader v2 tests failed; see TEST_LOG.txt")
    return summary


def input_rows(path):
    value = load(path)
    rows = value if isinstance(value, list) else [value]
    for row in rows:
        if set(row) - {"case_id", "original_penman", "contract"}:
            raise ValueError("Graph input accepts only case_id, original_penman, contract")
        if not isinstance(row.get("original_penman"), str) or not isinstance(row.get("contract"), dict):
            raise ValueError("Each row needs original_penman and public contract")
    return rows


def write_csv(path, rows):
    columns = ["canonical_index", "case_id", "condition", "structurally_eligible",
               "v1_primary_prediction", "v2_primary_prediction", "prediction_changed",
               "v1_four_fact_match", "v2_four_fact_match", "rules_applied",
               "v2_sender", "v2_recipient", "v2_object", "v2_transfer_status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{k: row[k] for k in columns if k in row},
                "rules_applied": "|".join(row["rules_applied"]),
                **{"v2_" + k: row["v2_facts"][k]
                   for k in ("sender", "recipient", "object", "transfer_status")},
            })


def metrics(rows, version):
    prediction = version + "_primary_prediction"
    match = version + "_four_fact_match"
    return {
        "denominator": len(rows),
        "complete_four_fact_matches": sum(r["structurally_eligible"] and r[match] for r in rows),
        "symbol_matches": sum(r[prediction] is not None and r[prediction] == r["assistant_text_symbol"] for r in rows),
        "erasures": sum(r[prediction] is None for r in rows),
        "non_erased_symbol_disagreements": sum(
            r[prediction] is not None and r[prediction] != r["assistant_text_symbol"] for r in rows),
        "reference": "NONINDEPENDENT_ASSISTANT_TEXT_FACT_CANDIDATES",
    }


def report(out, result, focus):
    primary = result["all192"]
    display_out = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    lines = [
        "# 读图器 v2：让步作用域与移动／接收规则",
        "",
        f"原始 23 条疑问中，{result['focus23']['new_complete_four_fact_matches']} 条按 v2 开发规则恢复完整读出："
        "11 条让步作用域、6 条移动／接收组合；6 条继续擦除。"
        "这里的恢复指同一批原图与非独立助手文字事实的一致性，不是正式语义正确率或责任层结论。",
        "",
        "v1 模块、v4 冻结实验、原图、助手归因包和真人评审材料均保持原 SHA。"
        "v2 使用独立模块与入口；192 条原图的两个版本读出先保存，再加载助手事实对照。",
        "",
        "## 规则边界",
        "",
        "让步规则仅接受直接连接的计划交付和同物品、同方向、完整绑定的 negative＋yet 交付，"
        "要求唯一局部 decide-01／plan-01 及同一具名计划者。"
        "内部作用域遍历将现实否定解释为局部计划的并列断言；原 concession 的方向和三元组不改写。"
        "外层 say／want／plan／or、condition、purpose、future、even-if 仍阻断读出。"
        "这种局部作用域解释是待独立判定的开发假设，并非 AMR 规范保证。"
        "have-concession-91 的重述形式本轮不新增作用域豁免，不能把任意让步边视为可自由翻转。[cite:3]",
        "",
        "移动规则支持 slide-01 与 pass-05，并要求明确、非否定、非计划／条件／未来的 receive-01。"
        "发送者来自移动 ARG0，接收者来自 receive ARG0；遗漏来源可以联合绑定，明确未绑定或冲突来源不能当作遗漏。"
        "slide 的 beneficiary／ARG4／direction 仅作一致性约束，空间 ARG2 不作为人名。"
        "共享物品节点优先；不同节点必须完整描述一致且双方唯一配对，描述中未被读出的限定关系也不能被忽略。"
        "重复候选、冲突目标、未绑定方向代词或未绑定竞争移动均擦除。"
        "物品描述相同并不独立证明两事件同指，跨事件连接仍是显式开发假设。",
        "",
        "PropBank pass.05 的 ARG0／ARG1／ARG2 分别是给予者／易手物／接收者；"
        "pass.04 是允许经过，不能扩入 give 白名单。[cite:1] "
        "slide.01 的标准核心角色只有 causer 和 moving thing，ARG4 不能当作规范接收者。[cite:2]",
        "",
        "失败时，顶层汇总保留 receive 的具名接收者与完整物品。"
        "三项角色／物品部分事实保留不构成完整转移状态，ABCD 继续擦除。"
        "局部解码 warning 也会在 v2 单独调用时阻断预测，冻结的 20 条结构排除继续留在全量分母中。",
        "",
        "## 23 条原图结果",
        "",
        "| 规范索引 | 原因族 | v1 | v2 | 与助手四事实一致 |",
        "|---|---|---|---|---|",
    ]
    for row in focus:
        lines.append(f"| {row['canonical_index']} | {row['family']} | "
                     f"{row['v1_primary_prediction'] or '擦除'} | "
                     f"{row['v2_primary_prediction'] or '擦除'} | "
                     f"{'是' if row['v2_four_fact_match'] else '否'} |")
    lines += [
        "",
        "继续擦除：54／55 的 ink:instrument 限定、77 的 sample:ARG0 限定本轮不扩白名单；"
        "58 的 i 仍未绑定为 Ivo，不能靠计划／让步规则猜姓名；151 的 pass-04 不支持交付；"
        "183 的 direction→she 未绑定，不能用配对接收者回填该代词。"
        "151 仍保留已知接收者和物品，183 还保留已知发送者，状态继续 unknown。",
        "",
        "## 192 条回放与退化核对",
        "",
        "| 冻结结构资格后的全量分母 | v1 | v2 |",
        "|---|---:|---:|",
        f"| 分母 | 192 | 192 |",
        f"| 完整四事实与助手一致 | {primary['v1']['complete_four_fact_matches']} | {primary['v2']['complete_four_fact_matches']} |",
        f"| 符号与助手一致 | {primary['v1']['symbol_matches']} | {primary['v2']['symbol_matches']} |",
        f"| 擦除 | {primary['v1']['erasures']} | {primary['v2']['erasures']} |",
        f"| 未擦除符号不一致 | {primary['v1']['non_erased_symbol_disagreements']} | {primary['v2']['non_erased_symbol_disagreements']} |",
        "",
        f"结构资格仍为 172 条合格、20 条排除。原有合格成功案例退化数为 "
        f"{len(result['eligible_success_regressions'])}；原有合格非擦除读出的预测或四事实变化数为 "
        f"{len(result['eligible_non_erased_changes'])}。"
        "17 条新增完整读出全部在这 23 条疑问内。"
        "68／69 的局部 warning 图从 v1 原始 A 改成 v2 原始擦除；两行一直属于冻结排除，主口径均为擦除。",
        "",
        "四事实对照使用助手核对的实际文字候选，不使用计划符号；因此数字可能与基于此前模型文字事实的 v4 表格不同。"
        "这组数字反映开发对照，不能作为 independent FPR／TPR、泛化准确率或正式归因。",
        "",
        "## 验证与复现",
        "",
        f"{result['tests']['tests_run']} 项测试通过，包括原 v1 的 21 项兼容测试及新增规则反例。"
        "测试图是明确标注的合成单元用例，不计入 192 条观察或任何效果统计。"
        "192 条原始归一化三元组在 v1／v2 间完全一致；顶层、事件、提示和规则轨迹的证据均逐条核对为原图三元组。"
        "两次 v2 原图回放结果一致；v1 重放与冻结输出逐项一致。全部源封印在前后核验通过。",
        "",
        "运行环境为 Python 与 penman 1.3.1，不需要 GPU、parser、provider 或 detector。",
        "",
        "```bash",
        "PYTHONPATH=/tmp/amr_fact_readout_deps python3 scripts/run_semantic_fact_graph_readout_v2.py read --input graph_contract.json --out /tmp/readout_v2.json",
        f"PYTHONPATH=/tmp/amr_fact_readout_deps python3 {display_out}/supporting_code/test_reader_v2.py",
        f"PYTHONPATH=/tmp/amr_fact_readout_deps python3 scripts/run_semantic_fact_graph_readout_v2.py replay --tests-source {display_out}/supporting_code/test_reader_v2.py --out /tmp/semantic_reader_v2_replay",
        "```",
        "",
        "read 输入是仅含 original_penman、contract 的 JSON 对象或对象列表；case_id 可选。"
        "contract 严格限定 role_one、role_two、object。额外目标／答案字段会被拒绝。"
        "新 replay 目录必须不存在，避免覆写已保存结果。跨目录运行可设 AMR_READER_ROOT 为仓库根目录。",
        "",
        "开发输出与全部三元组位于 private_development/；代码快照和测试位于 supporting_code/。"
        "SHA256.json 覆盖交付文件且排除自身。公开共享时只取规则、汇总、脚本和 SHA，"
        "private_development 含私有原图派生材料，不作为公开包。",
        "",
        "正式独立真人确认仍为 0/192。v2 不更新原归因层标签，不把开发恢复追溯写成 v1 成功。",
        "",
        "## 证据与规则来源",
        "",
        "- [逐条对照与规则轨迹](private_development/COMPARISONS.json)",
        "- [23 条逐行表](private_development/CASES23.csv)",
        "- [保存后才加载参考的阶段记录](PRE_REFERENCE_SAVE.json)",
        "- [汇总与验证](RESULT.json)",
        "- [封印核验](SOURCE_SHA256.json)",
        "- [先前 23 条分析](../semantic_fact_unresolved23_analysis_20261026/REPORT.md)",
        '- <a id="cite-1"></a>[cite:1] PropBank pass frames（单一一手规范源）：https://raw.githubusercontent.com/propbank/propbank-frames/master/frames/pass.xml',
        '- <a id="cite-2"></a>[cite:2] PropBank slide frames（单一一手规范源）：https://raw.githubusercontent.com/propbank/propbank-frames/master/frames/slide.xml',
        '- <a id="cite-3"></a>[cite:3] AMR concession annotation guidance（单一一手规范源）：https://uhermjakob.github.io/amr/lib/popup/concession.html',
        "",
        "frame 与让步关系的上述规范约束来自各自一手规范。"
        "局部 scope 修复和跨事件补全的有效性来自本批开发回放与规则反例；没有独立评审证明它们普遍正确。",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay(out, tests_source):
    import penman
    if penman.__version__ != "1.3.1":
        raise RuntimeError("Replay requires frozen decoder penman==1.3.1")
    if out.exists():
        raise RuntimeError("Use a new output directory; existing artifacts are not overwritten")
    source = source_audit()
    out.mkdir(parents=True)
    private = out / "private_development"
    private.mkdir(mode=0o700)
    supporting = out / "supporting_code"
    supporting.mkdir()
    try:
        for name in CODE:
            target = supporting / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        shutil.copy2(tests_source, supporting / "test_reader_v2.py")
        shutil.copy2(V4 / "private_review/frozen_code/test_semantic_graph_fact_reader.py",
                     supporting / "test_reader_v1_frozen.py")
        test_result = run_tests(tests_source, out)
        inputs = input_rows(V4 / "private_review/GRAPH_INPUTS_NO_TARGET.json")
        if len(inputs) != 192 or len({r["case_id"] for r in inputs}) != 192:
            raise RuntimeError("192-row input identity drift")
        readbacks = []
        for index, row in enumerate(inputs):
            old = read_v1(row["original_penman"], row["contract"])
            new = read_graph_facts(row["original_penman"], row["contract"])
            if new["normalized_triples"] != old["normalized_triples"]:
                raise RuntimeError("Original graph normalization changed")
            evidence_audit(new)
            readbacks.append({"canonical_index": index, "case_id": row["case_id"],
                              "v1": old, "v2": new})
        dump(private / "GRAPH_ONLY_READOUTS.json", readbacks)
        persisted_hash = sha(private / "GRAPH_ONLY_READOUTS.json")
        # Crucially, no assistant reference is decoded above this boundary.
        dump(out / "PRE_REFERENCE_SAVE.json", {
            "at": utc(), "reader_version": VERSION, "rows": len(readbacks),
            "graph_inputs_sha256": source["graph_inputs_sha256"],
            "graph_only_outputs_sha256": persisted_hash,
            "assistant_reference_loaded": False,
            "revised_graphs_read_by_reader": False,
            "code_sha256": source["v2_code_sha256"],
        })
        print("192 graph-only readbacks persisted; loading assistant comparison now", flush=True)
        assistant = {r["case_id"]: r for r in load(ASSISTANT / "private_assistant_attribution/CASES.json")}
        frozen_rows = {r["case_id"]: r for r in load(V4 / "private_review/ROWS.json")}
        frozen_outputs = {r["case_id"]: r["graph_readback"]
                          for r in load(V4 / "private_review/GRAPH_ONLY_READOUTS.json")}
        focused = {r["case_id"]: r for r in load(ANALYSIS / "private_evidence/CASES23.json")}
        ids = {r["case_id"] for r in inputs}
        if set(assistant) != ids or set(frozen_rows) != ids or set(frozen_outputs) != ids or len(focused) != 23:
            raise RuntimeError("Reference/source row identity mismatch")
        comparisons, focus, raw_changes, eligible_changes, regressions = [], [], [], [], []
        for input_row, row in zip(inputs, readbacks):
            case_id, index = row["case_id"], row["canonical_index"]
            reference, frozen = assistant[case_id], frozen_rows[case_id]
            old, new = row["v1"], row["v2"]
            eligible = frozen["structurally_eligible"]
            if reference["canonical_index"] != index or reference["public_contract"] != input_row["contract"]:
                raise RuntimeError("Reference index or public contract drift")
            if reference["original_penman_sha256"] != new["original_penman_sha256"]:
                raise RuntimeError("Original literal identity drift")
            if eligible != reference["structurally_eligible"] or reference["formal_attribution_confirmed"]:
                raise RuntimeError("Structural qualification or evidence-status drift")
            if not json_equal(frozen_outputs[case_id], old):
                raise RuntimeError("v1 replay differs from frozen graph-only output")
            if not json_equal(read_graph_facts(input_row["original_penman"], input_row["contract"]), new):
                raise RuntimeError("v2 deterministic replay differs")
            facts = reference["text_fact_candidates"]
            symbol = classify_facts(facts, input_row["contract"]).symbol
            old_primary = old["prediction"] if eligible else None
            new_primary = new["prediction"] if eligible else None
            result = {
                "canonical_index": index, "case_id": case_id, "condition": frozen["condition"],
                "structurally_eligible": eligible, "assistant_text_symbol": symbol,
                "v1_primary_prediction": old_primary, "v2_primary_prediction": new_primary,
                "v1_raw_prediction": old["prediction"], "v2_raw_prediction": new["prediction"],
                "prediction_changed": old_primary != new_primary,
                "v1_four_fact_match": actual_signature(old["facts"]) == actual_signature(facts)
                    if actual_signature(facts) is not None else False,
                "v2_four_fact_match": actual_signature(new["facts"]) == actual_signature(facts)
                    if actual_signature(facts) is not None else False,
                "v1_facts": old["facts"], "v2_facts": new["facts"],
                "rules_applied": sorted({t["rule"] for t in new["rule_trace"]}),
                "formal_attribution_confirmed": False, "independent_gold": False,
            }
            comparisons.append(result)
            if old["prediction"] != new["prediction"]:
                raw_changes.append(index)
            if eligible and old_primary is not None and (
                    old_primary != new_primary or actual_signature(old["facts"]) != actual_signature(new["facts"])):
                eligible_changes.append(index)
            if eligible and result["v1_four_fact_match"] and not result["v2_four_fact_match"]:
                regressions.append(index)
            if case_id in focused:
                focus.append({**result, "family": focused[case_id]["family"],
                              "pattern": focused[case_id]["pattern"]})
        if sum(not r["structurally_eligible"] for r in comparisons) != 20:
            raise RuntimeError("Frozen 20 structural exclusions changed")
        dump(private / "COMPARISONS.json", comparisons)
        dump(private / "CASES23.json", focus)
        write_csv(private / "CASES192.csv", comparisons)
        write_csv(private / "CASES23.csv", focus)
        after = source_audit()
        if after["v2_code_sha256"] != source["v2_code_sha256"] or after["maps"] != source["maps"]:
            raise RuntimeError("Code/source drift during replay")
        dump(out / "SOURCE_SHA256.json", {**source, "post_replay_verified_at": after["verified_at"]})
        summary = {
            "status": "DEVELOPMENT_REPLAY_COMPLETE_NO_INDEPENDENT_GOLD", "at": utc(),
            "reader_version": VERSION, "rules": RULES, "penman_version": penman.__version__,
            "all192": {"v1": metrics(comparisons, "v1"), "v2": metrics(comparisons, "v2")},
            "by_condition": {condition: {
                version: metrics([r for r in comparisons if r["condition"] == condition], version)
                for version in ("v1", "v2")} for condition in ("original", "paraphrase")},
            "focus23": {
                "rows": 23,
                "new_complete_four_fact_matches": sum(not r["v1_four_fact_match"] and r["v2_four_fact_match"] for r in focus),
                "remaining_erasures": [r["canonical_index"] for r in focus if r["v2_primary_prediction"] is None],
                "per_family": {family: {
                    "rows": sum(r["family"] == family for r in focus),
                    "v2_complete_four_fact_matches": sum(
                        r["family"] == family and r["v2_four_fact_match"] for r in focus)}
                    for family in sorted({r["family"] for r in focus})},
            },
            "tests": test_result, "structurally_eligible": 172, "structural_exclusions": 20,
            "eligible_success_regressions": regressions,
            "eligible_non_erased_changes": eligible_changes,
            "raw_prediction_changes": raw_changes,
            "primary_prediction_changes": [r["canonical_index"] for r in comparisons if r["prediction_changed"]],
            "primary_improvements_outside_focus23": [
                r["canonical_index"] for r in comparisons if r["prediction_changed"] and r["case_id"] not in focused],
            "unchanged_normalized_original_graphs": 192, "literal_evidence_audit_passed_rows": 192,
            "v1_exact_frozen_replay_rows": 192, "v2_deterministic_replay_rows": 192,
            "graph_only_outputs_saved_before_reference_join_sha256": persisted_hash,
            "graph_only_outputs_unchanged_after_join": sha(private / "GRAPH_ONLY_READOUTS.json") == persisted_hash,
            "v1_readout_calls": 192, "v2_readout_calls": 384,
            "new_parser_provider_detector_embedding_calls": 0,
            "revised_graph_reader_calls": 0, "formal_confirmed": 0,
            "assistant_reference_is_nonindependent": True, "administrative_plan_payloads_decoded": False,
            "reader_inputs_contain_only_original_penman_and_public_contract": True,
            "sealed_source_counts": source["sealed_file_counts"],
        }
        dump(out / "RESULT.json", summary)
        report(out, summary, focus)
        files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256.json"}
        dump(out / "SHA256.json", {"algorithm": "SHA256", "self_hash_excluded": True, "files": files})
        print(json.dumps({"output": str(out), "focus23": summary["focus23"],
                          "all192": summary["all192"], "tests": test_result["tests_run"]}, ensure_ascii=False))
    except Exception as exc:
        dump(out / "HALT.json", {"at": utc(), "status": "HALTED", "error": str(exc),
                                "formal_attribution_confirmed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reading = sub.add_parser("read", help="Read a graph/public-contract JSON; no references")
    reading.add_argument("--input", type=Path, required=True)
    reading.add_argument("--out", type=Path, required=True)
    running = sub.add_parser("replay", help="Replay original 192 graphs, then compare assistant candidates")
    running.add_argument("--out", type=Path, default=DEFAULT_OUT)
    running.add_argument("--tests-source", type=Path,
                         default=DEFAULT_OUT / "supporting_code/test_reader_v2.py")
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.out.resolve(), args.tests_source.resolve())
    else:
        if args.out.exists():
            raise RuntimeError("Existing readout will not be overwritten")
        rows = input_rows(args.input)
        results = []
        for row in rows:
            result = read_graph_facts(row["original_penman"], row["contract"])
            evidence_audit(result)
            results.append({"case_id": row.get("case_id"), "graph_readback": result})
        dump(args.out, results)
        print(json.dumps({"rows": len(results), "output": str(args.out), "reader_version": VERSION}))


if __name__ == "__main__":
    main()
