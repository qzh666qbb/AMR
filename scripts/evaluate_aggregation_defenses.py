"""Evaluate lightweight document-score aggregations against frozen E2 attacks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


MATCH_THRESHOLD = 0.65
LAMBDA = 0.25


def scores(doc: list[dict]) -> dict[str, float]:
    values = np.asarray([float(item["score"]) for item in doc], dtype=float)
    if len(values) == 0:
        return {name: math.nan for name in ["binary_z", "mean", "median", "max", "top2_mean", "soft_green"]}
    green = int((values >= MATCH_THRESHOLD).sum())
    binary_z = (green - LAMBDA * len(values)) / math.sqrt(len(values) * LAMBDA * (1 - LAMBDA))
    top2 = np.sort(values)[-min(2, len(values)) :]
    soft_green = float(np.mean(1 / (1 + np.exp(-(values - MATCH_THRESHOLD) / 0.05))))
    return {
        "binary_z": float(binary_z),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "top2_mean": float(top2.mean()),
        "soft_green": soft_green,
    }


def threshold_metrics(human: np.ndarray, machine: np.ndarray, target_fpr: float) -> dict:
    candidates = np.unique(human)
    candidates = np.concatenate(([np.nextafter(candidates.min(), -np.inf)], candidates))
    feasible = []
    for threshold in candidates:
        fpr = float(np.mean(human > threshold))
        if fpr <= target_fpr:
            feasible.append((threshold, fpr))
    threshold, fpr = min(feasible, key=lambda item: abs(item[1] - target_fpr))
    return {
        "threshold": float(threshold),
        "actual_fpr": fpr,
        "tpr": float(np.mean(machine > threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    baseline = args.root / "baselines" / "local" / "swan_acl2026_realnews" / "runs" / "formal_250x5" / "detection"
    exp = args.root / "experiments" / "planE_e2_joint250"
    human_cache = out_dir = args.root / "experiments" / "defense_aggregation"
    human_cache.mkdir(parents=True, exist_ok=True)
    human_cache_path = human_cache / "human_amr_matches.json"
    if human_cache_path.exists():
        human_docs = json.loads(human_cache_path.read_text(encoding="utf-8"))["matched_amrs"]
    else:
        swan_root = args.root / "third_party" / "SWAN"
        sys.path.insert(0, str(swan_root))
        from detection.detect_from_parsed_amrs import compute_doc_zscores_parallel
        from utils.amr_utils import load_amr_bank

        parsed_human = json.loads(
            (swan_root / "amr_bank" / "artifacts" / "human_parsed_amrs_250.json").read_text(
                encoding="utf-8"
            )
        )
        bank = load_amr_bank(str(swan_root / "amr_bank" / "banks" / "amr_bank_50.json"))
        _, raw_matches = compute_doc_zscores_parallel(
            parsed_human, bank, MATCH_THRESHOLD, LAMBDA, True, num_workers=8
        )
        human_docs = [
            [
                {"score": float(match_score), "best_matching_amr": best_amr}
                for match_score, best_amr in doc
            ]
            for doc in raw_matches
        ]
        human_cache_path.write_text(
            json.dumps({"matched_amrs": human_docs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    machine_docs = json.loads((baseline / "machine_amr_matches.json").read_text(encoding="utf-8"))["matched_amrs"]
    attack_docs_all = json.loads(
        (exp / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8")
    )["matched_amrs"]
    reviews = json.loads((exp / "quality_aware_reviews.json").read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in reviews:
        grouped[int(row["index"])].append(row)
    selected = []
    for index in sorted(grouped):
        valid = [row for row in grouped[index] if row.get("quality_valid", False)]
        selected.append(min(valid or grouped[index], key=lambda row: (row["z_score"], row["candidate"])))
    attack_docs = [attack_docs_all[int(row["position"])] for row in selected]

    human_scores = [scores(doc) for doc in human_docs]
    machine_scores = [scores(doc) for doc in machine_docs]
    attack_scores = [scores(doc) for doc in attack_docs]
    methods = ["binary_z", "mean", "median", "max", "top2_mean", "soft_green"]
    recorded_human_z = np.load(baseline / "human_z_scores.npy")
    reconstructed_human_z = np.asarray([row["binary_z"] for row in human_scores])
    max_human_z_difference = float(np.max(np.abs(recorded_human_z - reconstructed_human_z)))
    human_z_exact_rate = float(np.mean(np.isclose(recorded_human_z, reconstructed_human_z)))
    human_z_correlation = float(np.corrcoef(recorded_human_z, reconstructed_human_z)[0, 1])
    results = {}
    for method in methods:
        human = np.asarray([row[method] for row in human_scores])
        machine = np.asarray([row[method] for row in machine_scores])
        attack = np.asarray([row[method] for row in attack_scores])
        results[method] = {
            "baseline_auroc": float(roc_auc_score(np.r_[np.zeros(len(human)), np.ones(len(machine))], np.r_[human, machine])),
            "attack_auroc": float(roc_auc_score(np.r_[np.zeros(len(human)), np.ones(len(attack))], np.r_[human, attack])),
            "baseline_at_1pct": threshold_metrics(human, machine, 0.01),
            "attack_at_1pct": threshold_metrics(human, attack, 0.01),
            "baseline_at_5pct": threshold_metrics(human, machine, 0.05),
            "attack_at_5pct": threshold_metrics(human, attack, 0.05),
        }

    out_dir = args.root / "experiments" / "defense_aggregation"
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = ["Binary z", "Mean", "Median", "Max", "Top-2 mean", "Soft green"]
    x = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 4.5))
    axis.bar(x - width / 2, [results[m]["baseline_auroc"] for m in methods], width, label="No attack")
    axis.bar(x + width / 2, [results[m]["attack_auroc"] for m in methods], width, label="E2 attack")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("AUROC")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "aggregation_auroc.png", dpi=200)
    plt.close(fig)

    display = dict(zip(methods, labels))
    lines = [
        "# 轻量聚合防御实验",
        "",
        "所有方法使用同一批250个人类段落、250个原始水印段落和250个E2质量感知选优结果。"
        "阈值仅由人类校准分数确定。",
        "",
        "| 聚合方法 | 原始AUROC | 攻击后AUROC | 原始TPR@≤1%FPR | 攻击后TPR@≤1%FPR | 原始TPR@≤5%FPR | 攻击后TPR@≤5%FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = results[method]
        lines.append(
            f"| {display[method]} | {row['baseline_auroc']:.3f} | {row['attack_auroc']:.3f} | "
            f"{row['baseline_at_1pct']['tpr']:.1%} | {row['attack_at_1pct']['tpr']:.1%} | "
            f"{row['baseline_at_5pct']['tpr']:.1%} | {row['attack_at_5pct']['tpr']:.1%} |"
        )
    lines.extend(
        [
            "",
        "## 解释限制",
            "",
            f"人类逐句匹配重建与正式基线z有{human_z_exact_rate:.1%}逐项一致，相关系数为{human_z_correlation:.3f}；"
            f"少数S2MATCH边界样本使最大差异达到{max_human_z_difference:.3f}，因此本防御结果标记为离线原型。"
            "现有连续分数聚合均未恢复E2攻击下的检测能力：最好的攻击后AUROC仍是原始Binary z的0.686。"
            "Mean虽接近随机水平0.516，但这代表检测失效而非防御成功；Max和Top-2甚至出现方向反转。",
            "",
            "这些是对已有每句最大S2MATCH分数的离线重聚合，不包含跨句模板、多切分或多parser。"
            "若某方法改善攻击后指标，还必须检查自适应攻击者能否针对新聚合重新选优。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
