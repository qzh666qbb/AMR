"""Known-position four-symbol diagnostics from NEW frozen transfer scores.

No inference, fitting, test tuning, or rewriting of original Stage1B decisions.
The existing design-v1 T=.10 and 24-template symbol map remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    out, run = args.out, args.run
    frozen = json.loads((out / "FREEZE.json").read_text())
    symbol_map = json.loads((run / "SYMBOL_MAP.json").read_text())
    assert not symbol_map["cross_symbol_collisions"]
    assignments = {r["doc_id"]: r for r in
                   map(json.loads, (run / "private/ASSIGNMENTS.jsonl").read_text().splitlines())}
    mapping = {int(k): "ABCD".index(v) for k, v in symbol_map["template_to_symbol"].items()}
    pool = sorted(mapping)
    groups = [[i for i, t in enumerate(pool) if mapping[t] == c] for c in range(4)]
    assert len(pool) == 24 and all(len(g) == 6 for g in groups)
    records = {r["doc_id"]: r for r in
               (json.loads(p.read_text()) for p in sorted((out / "private/records").glob("*_clean.json")))}
    signal_ids = []
    for path in sorted((out / "private/signal").glob("*.json")):
        signal = json.loads(path.read_text())
        assert signal["freeze_sha256"] == sha(out / "FREEZE.json")
        signal_ids.append(signal["doc_id"])
        if signal["doc_id"] not in records:
            records[signal["doc_id"]] = {
                "doc_id": signal["doc_id"],
                "arms": {arm: {"official_sentence_scores": a["sentence_scores"]}
                         for arm, a in signal["arms"].items()}}
    records = [records[d] for d in sorted(records)]
    assert all(r.get("freeze_sha256", sha(out / "FREEZE.json")) == sha(out / "FREEZE.json")
               for r in records)
    result = {
        "schema": "new-text-known-position-symbols/v1",
        "analysis_source_sha256": sha(__file__),
        "scope": "Fresh-score exploratory clean carrier diagnostic; intended labels, not realized semantic truth",
        "freeze_sha256": sha(out / "FREEZE.json"),
        "symbol_map_sha256": sha(run / "SYMBOL_MAP.json"),
        "assignments_sha256": sha(run / "private/ASSIGNMENTS.jsonl"),
        "signal_complete_sha256": sha(out / "SIGNAL_COMPLETE.json") if (out / "SIGNAL_COMPLETE.json").exists() else None,
        "temperature": 0.10, "temperature_source": "Existing design-v1 fixed main temperature, no tuning",
        "completed_pairs": len(records), "no_old_score_cache": True,
        "official_signal_pairs": len(signal_ids),
        "no_class_filtering_no_refit": True,
        "blind_identity_f1": None, "paraphrase_known_position_metrics": None,
        "sequence_win_2520": None, "write_parse_classifier_causal_attribution": None,
        "negative_symbol_labels_are_not_paired_targets": True,
        "populations": {},
    }
    for population in ("development", "readout"):
        ids = frozen["split"][population]
        selected = [r for r in records if r["doc_id"] in ids]
        methods = {k: {"support": np.zeros(4), "confusion": np.zeros((4, 5)), "nll": [],
                       "brier": [], "high_confident": 0, "high_error": 0.0}
                   for k in ("official_sentence", "S6", "S12", "Ssub")}
        for r in selected:
            a = r["arms"]["positive"]
            slots = assignments[r["doc_id"]]["slots"]
            assert len(slots) == len(a["official_sentence_scores"]) == 8
            if "carrier_features" in a:
                assert len(a["carrier_features"]) == 8
            for i, slot in enumerate(slots):
                target = mapping[slot["intended_template"]]
                assert "ABCD"[target] == slot["intended_state"]
                vectors = {"official_sentence": a["official_sentence_scores"][i]}
                if "carrier_features" in a:
                    vectors.update({k: a["carrier_features"][i][j]
                                    for j, k in enumerate(("S6", "S12", "Ssub"))})
                for name, vector in vectors.items():
                    m = methods[name]
                    m["support"][target] += 1
                    complete = all(vector[t] is not None and math.isfinite(vector[t]) for t in pool)
                    if not complete:
                        q = np.ones(4) / 4
                        m["confusion"][target, 4] += 1
                    else:
                        values = np.array([vector[t] for t in pool])
                        mass = np.exp((values - values.max()) / 0.10)
                        q = np.array([mass[g].sum() for g in groups]) / mass.sum()
                        best = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
                        m["confusion"][target, best] += 1 / len(best)
                        if q.max() >= 0.8:
                            m["high_confident"] += 1
                            m["high_error"] += 1 - (1 / len(best) if target in best else 0)
                    reference = np.zeros(4)
                    reference[target] = 1
                    m["nll"].append(-math.log(max(float(q[target]), 1e-300)))
                    m["brier"].append(float(np.square(q - reference).sum()))
        rows = {}
        for name, m in methods.items():
            n = int(m["support"].sum())
            if not n:
                continue
            recalls = np.divide(np.diag(m["confusion"][:, :4]), m["support"],
                                out=np.zeros(4), where=m["support"] > 0)
            rows[name] = {
                "planned_carriers": n, "class_support": m["support"].tolist(),
                "balanced_accuracy": float(recalls.mean()), "class_recalls": recalls.tolist(),
                "confusion_true_ABCD_pred_ABCD_erasure": m["confusion"].tolist(),
                "erased_carriers": int(m["confusion"][:, 4].sum()),
                "nll": float(np.mean(m["nll"])), "brier": float(np.mean(m["brier"])),
                "maximum_recall_gap": float(recalls.max() - recalls.min()),
                "high_confidence_coverage": m["high_confident"] / n,
                "high_confidence_error": m["high_error"] / m["high_confident"] if m["high_confident"] else None,
            }
        result["populations"][population] = {"completed_pairs": len(selected), "expected_pairs": len(ids),
                                             "methods": rows}
        background_counts = np.zeros(5)
        background_weights = []
        for r in selected:
            for vector in r["arms"]["unwatermarked"]["official_sentence_scores"]:
                if any(vector[t] is None or not math.isfinite(vector[t]) for t in pool):
                    background_counts[4] += 1
                    continue
                values = np.asarray([vector[t] for t in pool])
                mass = np.exp((values - values.max()) / 0.10)
                q = np.asarray([mass[g].sum() for g in groups]) / mass.sum()
                best = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
                background_counts[best] += 1 / len(best)
                background_weights.append(q)
        result["populations"][population]["negative_official_symbol_distribution"] = {
            "ABCD_erasure_counts": background_counts.tolist(),
            "mean_weights": np.mean(background_weights, axis=0).tolist() if background_weights else None,
            "classification_accuracy": None,
        }
    (out / "SYMBOL_DIAGNOSTIC.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    text = [
        "# 新文本四类符号读出诊断", "",
        "只使用本轮重新计算的 clean 载体分数；已知原句位置，主标签是生成前的计划符号。"
        "它衡量计划与读出的一致性，不是实际语义实现率、盲定位 F1 或正式序列水印检出率。", "",
        "沿用封存的 24 模板四类字典及既有主温度 T=0.10。不训练分类器，不根据新结果改温度、"
        "字典或筛选正例。任何必需模板分数缺失均擦除，保留计划分母。", "",
    ]
    for population, p in result["populations"].items():
        text += [f"## {population}：{p['completed_pairs']}/{p['expected_pairs']} 对", "",
                 "| 表示 | 计划载体 | 四类 BA | 擦除 | NLL |",
                 "|---|---:|---:|---:|---:|"]
        for name, m in p["methods"].items():
            text.append(f"| {name} | {m['planned_carriers']} | {m['balanced_accuracy']:.3f} | "
                        f"{m['erased_carriers']} | {m['nll']:.3f} |")
        text.append("")
        for name, m in p["methods"].items():
            text += [f"### {name} 各类召回与混淆", "",
                     "| 真值 | 召回 | 读成A | 读成B | 读成C | 读成D | 擦除 |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
            for i, label in enumerate("ABCD"):
                row = m["confusion_true_ABCD_pred_ABCD_erasure"][i]
                text.append(f"| {label} | {m['class_recalls'][i]:.3f} | "
                            + " | ".join(f"{x:g}" for x in row) + " |")
            text.append("")
        background = p["negative_official_symbol_distribution"]
        text += ["配对无水印原句的 A/B/C/D/擦除预测数量为 "
                 f"{background['ABCD_erasure_counts']}。负例不使用配对计划符号作为真值，"
                 "该分布仅用于检查自然文本的类别偏向。", ""]
    text += [
        "## 解释边界", "",
        "S6/S12/Ssub 使用与已知原句重叠的候选最大分，不能称为盲读出。Ssub 是多子图取最大，"
        "不是序列协议中每个锚点选一个子图的 E1，因此不能把更高的比较次数算作 E1 收益。", "",
        "当前没有独立的实际结构标签或 parser 金标准，不能从混淆矩阵直接分解"
        "“生成没写入／解析丢失／分类器错误”。复述位置仍需独立映射。"
        "Embedding、事件锚点盲提取和 2,520 个等频排列的顺序检验未由本脚本完成。", "",
        "全部混淆矩阵、各类召回和置信错误率见 SYMBOL_DIAGNOSTIC.json。"
        "数据 SHA 和模型/解析器/评分参数见 FREEZE.json。", "",
    ]
    (out / "SYMBOL_DIAGNOSTIC.md").write_text("\n".join(text))
    print(json.dumps({"pairs": len(records),
                      "populations": {k: {m: v["balanced_accuracy"] for m, v in p["methods"].items()}
                                      for k, p in result["populations"].items()}}))


if __name__ == "__main__":
    main()
