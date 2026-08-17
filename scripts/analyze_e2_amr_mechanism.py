"""Compare baseline and quality-aware E2 selections at the AMR graph level."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import penman


MATCH_THRESHOLD = 0.65
ESCAPE_THRESHOLD = 2.33


def graph_features(amr: str) -> dict[str, int]:
    try:
        graph = penman.decode(amr)
    except Exception:
        return {"parse_ok": 0, "instances": 0, "relations": 0, "attributes": 0, "reentrancies": 0}
    instances = graph.instances()
    relations = graph.edges()
    attributes = graph.attributes()
    incoming = Counter(edge.target for edge in relations)
    reentrancies = sum(max(0, count - 1) for count in incoming.values())
    return {
        "parse_ok": 1,
        "instances": len(instances),
        "relations": len(relations),
        "attributes": len(attributes),
        "reentrancies": reentrancies,
    }


def doc_features(amrs: list[str], matches: list[dict]) -> dict[str, float]:
    graphs = [graph_features(amr) for amr in amrs]
    scores = np.asarray([float(item["score"]) for item in matches], dtype=float)
    return {
        "sentences": len(amrs),
        "amr_parse_rate": float(np.mean([item["parse_ok"] for item in graphs])) if graphs else 0.0,
        "instances": sum(item["instances"] for item in graphs),
        "relations": sum(item["relations"] for item in graphs),
        "attributes": sum(item["attributes"] for item in graphs),
        "reentrancies": sum(item["reentrancies"] for item in graphs),
        "mean_best_match": float(scores.mean()) if len(scores) else math.nan,
        "green_count": int((scores >= MATCH_THRESHOLD).sum()),
        "green_rate": float((scores >= MATCH_THRESHOLD).mean()) if len(scores) else math.nan,
    }


def corr(rows: list[dict], x_name: str, y_name: str) -> float:
    x = np.asarray([row[x_name] for row in rows], dtype=float)
    y = np.asarray([row[y_name] for row in rows], dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def mean(rows: list[dict], name: str) -> float:
    return float(np.mean([row[name] for row in rows])) if rows else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    exp = args.root / "experiments" / "planE_e2_joint250"
    baseline = args.root / "baselines" / "local" / "swan_acl2026_realnews" / "runs" / "formal_250x5"
    reviews = json.loads((exp / "quality_aware_reviews.json").read_text(encoding="utf-8"))
    attack_amrs = json.loads((exp / "parsed.json").read_text(encoding="utf-8"))
    attack_matches = json.loads(
        (exp / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8")
    )["matched_amrs"]
    baseline_amrs = json.loads((baseline / "parsed_amrs.json").read_text(encoding="utf-8"))
    baseline_matches = json.loads(
        (baseline / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8")
    )["matched_amrs"]
    baseline_z = np.load(baseline / "detection" / "machine_z_scores.npy")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in reviews:
        grouped[int(row["index"])].append(row)
    selected = []
    for index in sorted(grouped):
        valid = [row for row in grouped[index] if row.get("quality_valid", False)]
        selected.append(min(valid or grouped[index], key=lambda row: (row["z_score"], row["candidate"])))

    rows = []
    for index, selected_row in enumerate(selected):
        position = int(selected_row["position"])
        before = doc_features(baseline_amrs[index], baseline_matches[index])
        after = doc_features(attack_amrs[position], attack_matches[position])
        row = {
            "index": index,
            "candidate": selected_row["candidate"],
            "quality_valid": bool(selected_row.get("quality_valid", False)),
            "escaped": float(selected_row["z_score"]) <= ESCAPE_THRESHOLD,
            "valid_asr": bool(selected_row.get("quality_valid", False))
            and float(selected_row["z_score"]) <= ESCAPE_THRESHOLD,
            "baseline_z": float(baseline_z[index]),
            "attack_z": float(selected_row["z_score"]),
            "z_drop": float(baseline_z[index]) - float(selected_row["z_score"]),
        }
        for name, value in before.items():
            row[f"before_{name}"] = value
        for name, value in after.items():
            row[f"after_{name}"] = value
            if name not in {"amr_parse_rate"}:
                row[f"delta_{name}"] = value - before[name]
        rows.append(row)

    summary = {
        "n": len(rows),
        "quality_valid": int(sum(row["quality_valid"] for row in rows)),
        "escaped": int(sum(row["escaped"] for row in rows)),
        "valid_asr": int(sum(row["valid_asr"] for row in rows)),
        "means": {},
        "correlations_with_z_drop": {},
    }
    measures = ["sentences", "instances", "relations", "attributes", "reentrancies", "mean_best_match", "green_rate"]
    for measure in measures:
        summary["means"][measure] = {
            "before": mean(rows, f"before_{measure}"),
            "after": mean(rows, f"after_{measure}"),
            "delta": mean(rows, f"delta_{measure}"),
        }
        summary["correlations_with_z_drop"][f"delta_{measure}"] = corr(rows, f"delta_{measure}", "z_drop")
    summary["selected_amr_parse_rate"] = mean(rows, "after_amr_parse_rate")

    groups = {
        "valid_asr": [row for row in rows if row["valid_asr"]],
        "quality_failure": [row for row in rows if not row["quality_valid"]],
        "quality_valid_not_escaped": [
            row for row in rows if row["quality_valid"] and not row["escaped"]
        ],
    }
    group_summary = {
        key: {
            "n": len(value),
            "mean_z_drop": mean(value, "z_drop"),
            "mean_sentence_delta": mean(value, "delta_sentences"),
            "mean_green_rate_delta": mean(value, "delta_green_rate"),
            "mean_instance_delta": mean(value, "delta_instances"),
            "mean_relation_delta": mean(value, "delta_relations"),
        }
        for key, value in groups.items()
    }

    out_dir = args.root / "experiments" / "mechanism_e2_amr"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "groups": group_summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(
        [row["delta_sentences"] for row in rows],
        [row["z_drop"] for row in rows],
        alpha=0.5,
        s=18,
    )
    axes[0].set(xlabel="Attack - baseline sentence count", ylabel="z-score drop")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(
        [row["delta_green_rate"] for row in rows],
        [row["z_drop"] for row in rows],
        alpha=0.5,
        s=18,
        color="#7c3aed",
    )
    axes[1].set(xlabel="Attack - baseline green-sentence rate", ylabel="z-score drop")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "amr_change_vs_z_drop.png", dpi=200)
    plt.close(fig)

    lines = [
        "# E2 选优候选的AMR机制分析",
        "",
        "## 总体结果",
        "",
        f"分析250个质量感知选优结果：{summary['quality_valid']}个质量合格，"
        f"{summary['escaped']}个低于阈值，{summary['valid_asr']}个同时满足两项。",
        f"选中候选AMR图可解码率为{summary['selected_amr_parse_rate']:.1%}。",
        "",
        "| 图/检测特征 | 原始均值 | 攻击后均值 | 平均变化 | 与z降幅相关系数 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "sentences": "句子/AMR数",
        "instances": "实例节点数",
        "relations": "关系边数",
        "attributes": "属性数",
        "reentrancies": "重入数",
        "mean_best_match": "平均最大模板匹配",
        "green_rate": "绿色句比例",
    }
    for measure in measures:
        values = summary["means"][measure]
        correlation = summary["correlations_with_z_drop"][f"delta_{measure}"]
        lines.append(
            f"| {labels[measure]} | {values['before']:.3f} | {values['after']:.3f} | "
            f"{values['delta']:.3f} | {correlation:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结果分组",
            "",
            "| 分组 | n | 平均z降幅 | 平均句数变化 | 绿色句率变化 | 节点变化 | 关系边变化 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    group_labels = {
        "valid_asr": "质量合格且逃逸",
        "quality_failure": "质量失败",
        "quality_valid_not_escaped": "质量合格但未逃逸",
    }
    for key, values in group_summary.items():
        lines.append(
            f"| {group_labels[key]} | {values['n']} | {values['mean_z_drop']:.3f} | "
            f"{values['mean_sentence_delta']:.3f} | {values['mean_green_rate_delta']:.3f} | "
            f"{values['mean_instance_delta']:.3f} | {values['mean_relation_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "z-score由超过0.65阈值的绿色句数量和总句数共同决定，因此绿色句率变化最接近检测器的直接机制。"
            "节点、关系和重入变化属于相关证据，不能单独证明语义结构变化是因果来源。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
