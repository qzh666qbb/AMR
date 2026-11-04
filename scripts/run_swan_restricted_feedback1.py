"""Actual restricted-process replay on the complete historical E2 candidate pool."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
from swan_restricted_feedback1_server import session

OUT = ROOT / "experiments/swan_restricted_feedback1"
ORIGINAL = ROOT / "experiments/planE_e2_joint30"
CODE = ("scripts/run_swan_restricted_feedback1.py",
        "scripts/swan_restricted_feedback1_client.py", "scripts/swan_restricted_feedback1_server.py",
        "scripts/test_swan_restricted_feedback1.py", "scripts/run_multiscale_transfer.py")


def immutable(path, payload):
    if path.exists():
        if rt.read(path) != payload:
            raise RuntimeError("Immutable replay artifact changed")
    else:
        rt.atomic(path, payload)


def prepare():
    inputs = rt.read(ORIGINAL / "candidates.json")
    scored = rt.read(ORIGINAL / "scored_candidates.json")
    if inputs["n"] != 30 or inputs["k"] != 4 or len(scored) != 120:
        raise RuntimeError("Complete original 30 by 4 pool required")
    source = {(r["index"], r["candidate"]): r for r in inputs["rows"]}
    seen = set()
    for row in scored:
        key = row["index"], row["candidate"]
        if key in seen or {k: row[k] for k in ("index", "candidate", "source", "attack")} != source[key]:
            raise RuntimeError("Historical text identity or unique candidate mismatch")
        seen.add(key)
    if seen != {(i, k) for i in range(30) for k in range(4)}:
        raise RuntimeError("Incomplete historical source-candidate product")
    private = OUT / "private"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("candidates.json", "scored_candidates.json"):
        target = private / name
        if not target.exists():
            shutil.copyfile(ORIGINAL / name, target)
        if rt.sha(target) != rt.sha(ORIGINAL / name):
            raise RuntimeError("Historical source SHA differs")
    immutable(OUT / "FREEZE.json", {
        "version": "fixed-pool-restricted-feedback-historical-replay/v1", "at": rt.now(),
        "input_sha256": {name: rt.sha(private / name)
                         for name in ("candidates.json", "scored_candidates.json")},
        "code_sha256": {name: rt.sha(ROOT / name) for name in CODE},
        "protocol_sha256": rt.sha(OUT / "PROTOCOL.md"),
        "sources": 30, "candidates_per_source": 4, "budgets": [1, 4],
        "modes": ["none", "label", "score"], "threshold": 2.33,
        "strict_greater_than": True, "independent_calibration": False,
        "legacy_detector_cache": True, "human_quality": None,
    })


def run():
    f = rt.read(OUT / "FREEZE.json")
    for name, expected in f["code_sha256"].items():
        if rt.sha(ROOT / name) != expected:
            raise RuntimeError("Frozen code differs")
    for name, expected in f["input_sha256"].items():
        if rt.sha(OUT / "private" / name) != expected:
            raise RuntimeError("Frozen input differs")
    if rt.sha(OUT / "PROTOCOL.md") != f["protocol_sha256"]:
        raise RuntimeError("Frozen protocol differs")
    contract = rt.sha(OUT / "FREEZE.json")
    rows = rt.read(OUT / "private/scored_candidates.json")
    predictions = []
    for i in range(30):
        pool = sorted([r for r in rows if r["index"] == i], key=lambda r: r["candidate"])
        candidates = [{"candidate_id": str(r["candidate"]), "text": r["attack"]} for r in pool]
        scores = {str(r["candidate"]): r["z_score"] for r in pool}
        for budget in f["budgets"]:
            for mode in f["modes"]:
                path = OUT / "private/transcripts" / f"{i:02d}-{mode}-{budget}.json"
                if path.exists():
                    saved = rt.read(path)
                    receipt = path.with_suffix(".sha.json")
                    if not receipt.exists() or rt.read(receipt)["sha256"] != rt.sha(path):
                        raise RuntimeError("Transcript incomplete or altered; no silent reuse")
                    if saved["freeze_sha256"] != contract:
                        raise RuntimeError("Session freeze changed")
                else:
                    result = session(candidates, scores, f["threshold"], mode, budget)
                    saved = {"source": i, "mode": mode, "budget": budget,
                             "freeze_sha256": contract, **result}
                    immutable(path, saved)
                    immutable(path.with_suffix(".sha.json"), {"sha256": rt.sha(path)})
                predictions.append(saved)
    # Finish and seal every selection before assembling any quality packet.
    immutable(OUT / "private/PREDICTIONS.json", predictions)
    immutable(OUT / "PREDICTION_SEAL.json", {
        "freeze_sha256": contract, "predictions_sha256": rt.sha(OUT / "private/PREDICTIONS.json"),
        "quality_answers_read": False, "target_Bank_read_by_attacker": False})
    packet = [{"source_id": str(i), "source": next(r["source"] for r in rows if r["index"] == i),
               "candidates": [{"candidate_id": str(r["candidate"]), "text": r["attack"],
                               "meaning_preserved": None, "fluent": None}
                              for r in rows if r["index"] == i]} for i in range(30)]
    immutable(OUT / "private/QUALITY_PACKET_UNFILLED.json", {
        "human_attestation": False, "reviewer_code": "", "prediction_seal_sha256":
        rt.sha(OUT / "PREDICTION_SEAL.json"), "sources": packet})
    summary = []
    for budget in f["budgets"]:
        for mode in f["modes"]:
            selected = [r for r in predictions if r["budget"] == budget and r["mode"] == mode]
            summary.append({"budget": budget, "mode": mode, "sources": len(selected),
                            "not_detected_nominal": sum(not r["evaluation"]["detected"] for r in selected),
                            "actual_queries": sum(r["selection"]["queries"] for r in selected)})
    differences = {str(budget): sum(
        next(r["selection"]["selected"] for r in predictions
             if r["source"] == i and r["budget"] == budget and r["mode"] == "label") !=
        next(r["selection"]["selected"] for r in predictions
             if r["source"] == i and r["budget"] == budget and r["mode"] == "score")
        for i in range(30)) for budget in f["budgets"]}
    result = {"status": "HISTORICAL_RESTRICTED_PROCESS_REPLAY_COMPLETE",
              "independent_sources": 30, "candidate_texts": 120, "process_sessions": len(predictions),
              "nominal_replay_summary": summary, "label_vs_score_selection_differences": differences,
              "total_queries": sum(r["selection"]["queries"] for r in predictions),
              "new_neural_detector_calls": 0, "new_provider_calls": 0,
              "live_blackbox_attack": False, "independent_target_Bank": False,
              "new_calibration": False, "quality_validated_ASR": None, "ATR": None,
              "source_clean_TPR": None, "human_reviews": 0, "budget16_executed": False}
    immutable(OUT / "RESULT.json", result)
    immutable(OUT / "AUDIT.json", {
        "complete_30x2x3_sessions": len(predictions) == 180,
        "label_responses_boolean_only": all(
            set(t["response"]) == {"label"} and type(t["response"]["label"]) is bool
            for r in predictions if r["mode"] == "label" for t in r["transcript"]),
        "feedback_query_budget_equal": all(r["selection"]["queries"] == r["budget"]
                                          for r in predictions if r["mode"] != "none"),
        "no_feedback_zero_queries": all(r["selection"]["queries"] == 0
                                       for r in predictions if r["mode"] == "none"),
        "attacker_invocation_has_no_evaluator_paths": True,
        "attacker_environment_has_no_secret_variables": True,
        "quality_labels_used_for_selection": False,
        "fresh_scientific_blackbox_endpoint_complete": False,
        "baseline_clean_detection_not_reconstructed": True,
    })
    immutable(OUT / "TERMINAL.json", {"status": result["status"],
                                     "result_sha256": rt.sha(OUT / "RESULT.json"),
                                     "audit_sha256": rt.sha(OUT / "AUDIT.json")})
    report = """# 二元反馈受限进程回放 v1

已实际执行30来源、120个历史候选、180次独立攻击进程会话。
两种反馈在预算1/4下分别消耗相同查询，无反馈实际0查询。
攻击进程仅通过stdin获取bool或单一连续分数；目标分数表不传给二元进程。
选择全部封SHA后才建立质量回收包，旧仅最佳候选的30条模型审查未用于重选。

这是旧E2固定候选池及旧检测分数的离线回放，使用名义阈值2.33。
没有新独立校准、源clean命中核验、完整质量真值或新目标库；
下列未命中数不是质量有效ASR、TPR或秘密Bank ATR。
旧候选只有4个，未用复制候选伪造16查询实验。

| 最大预算 | 反馈 | 名义未命中/30 | 实际查询总数 |
|---|---|---:|---:|
"""
    for row in summary:
        report += f"| {row['budget']} | {row['mode']} | {row['not_detected_nominal']}/30 | {row['actual_queries']} |\n"
    report += """
固定池中，只要出现一个未命中候选，二元first-escape和连续分数最小值都会
选到未命中。因此两者未命中计数相同属于策略结构，不能推出黑盒能力等价；
具体文本选择与语义质量可能不同，仍需独立质量评审及后继自适应比较。
X01的新独立Bank与正常长度1172来源另已冻结，尚无新库攻击结果。
"""
    path = OUT / "REPORT.md"
    if path.exists() and path.read_text() != report:
        raise RuntimeError("Replay report differs")
    path.write_text(report)
    print(json.dumps(result))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "run"))
    args = ap.parse_args()
    globals()[args.action]()
