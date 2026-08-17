"""Milestone A ablations using the frozen E2-250 four-candidate pool.

This script does not generate new text. It repeatedly subsamples the existing
four candidates per paragraph so candidate-budget and selection-policy effects
can be estimated without API-call randomness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


THRESHOLD = 2.33


def auc_score(human: np.ndarray, machine: np.ndarray) -> float:
    """AUC as P(machine > human), with half credit for ties."""
    values = np.concatenate([human, machine])
    labels = np.concatenate(
        [np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - len(machine) * (len(machine) + 1) / 2) / (len(machine) * len(human)))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+(?:\s+|$)", text.strip())))


def token_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)*\b", text))


def entity_proxy_count(text: str) -> int:
    spans = re.findall(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*\b", text)
    return len(spans)


def number_count(text: str) -> int:
    return len(re.findall(r"\b\d+(?:[.,]\d+)*\b", text))


def summarize(selected: list[dict], human_z: np.ndarray) -> dict:
    z_scores = np.asarray([row["z_score"] for row in selected], dtype=float)
    escaped = z_scores <= THRESHOLD
    quality = np.asarray([bool(row.get("quality_valid", False)) for row in selected])
    valid = escaped & quality
    low, high = wilson(int(valid.sum()), len(selected))
    return {
        "n": len(selected),
        "mean_z": float(z_scores.mean()),
        "median_z": float(np.median(z_scores)),
        "escape_rate": float(escaped.mean()),
        "quality_rate": float(quality.mean()),
        "valid_asr": float(valid.mean()),
        "valid_asr_ci_low": low,
        "valid_asr_ci_high": high,
        "auroc": auc_score(human_z, z_scores),
    }


def choose(rows: list[dict], policy: str, rng: np.random.Generator) -> dict:
    if policy == "random":
        return rows[int(rng.integers(0, len(rows)))]
    if policy == "min_z":
        return min(rows, key=lambda row: (row["z_score"], row["candidate"]))
    if policy == "quality_only":
        valid = [row for row in rows if row.get("quality_valid", False)]
        return (valid or rows)[int(rng.integers(0, len(valid or rows)))]
    if policy == "quality_then_min_z":
        valid = [row for row in rows if row.get("quality_valid", False)]
        return min(valid or rows, key=lambda row: (row["z_score"], row["candidate"]))
    raise ValueError(policy)


def aggregate_repeats(repeats: list[dict]) -> dict:
    keys = [
        "mean_z",
        "median_z",
        "escape_rate",
        "quality_rate",
        "valid_asr",
        "auroc",
    ]
    output = {"repeats": len(repeats)}
    for key in keys:
        values = np.asarray([row[key] for row in repeats], dtype=float)
        output[key] = float(values.mean())
        output[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
    return output


def length_bin(n_sentences: int) -> str:
    if n_sentences <= 3:
        return "2-3"
    if n_sentences <= 6:
        return "4-6"
    if n_sentences <= 10:
        return "7-10"
    return "11+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    exp = args.root / "experiments" / "planE_e2_joint250"
    rows = json.loads((exp / "quality_aware_reviews.json").read_text(encoding="utf-8"))
    human_z = np.load(exp / "detection" / "human_z_scores.npy")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["index"])].append(row)
    if len(grouped) != 250 or any(len(items) != 4 for items in grouped.values()):
        raise RuntimeError("Expected exactly 250 paragraphs with four reviewed candidates each")
    for items in grouped.values():
        items.sort(key=lambda row: int(row["candidate"]))

    rng = np.random.default_rng(args.seed)
    policies = ["random", "min_z", "quality_only", "quality_then_min_z"]
    budget_results: dict[str, dict] = {}
    for k in range(1, 5):
        budget_results[str(k)] = {}
        for policy in policies:
            repeats = []
            for _ in range(args.repeats):
                selected = []
                for index in sorted(grouped):
                    pool = grouped[index]
                    positions = rng.choice(len(pool), size=k, replace=False)
                    subset = [pool[int(position)] for position in positions]
                    selected.append(choose(subset, policy, rng))
                repeats.append(summarize(selected, human_z))
            budget_results[str(k)][policy] = aggregate_repeats(repeats)

    full_selected = [choose(grouped[index], "quality_then_min_z", rng) for index in sorted(grouped)]
    baseline_z_path = (
        args.root
        / "baselines"
        / "local"
        / "swan_acl2026_realnews"
        / "runs"
        / "formal_250x5"
        / "detection"
        / "machine_z_scores.npy"
    )
    baseline_z = np.load(baseline_z_path)
    if len(baseline_z) != len(full_selected):
        raise RuntimeError("Formal baseline z-score count does not match E2 paragraphs")
    strata: dict[str, list[dict]] = defaultdict(list)
    paragraph_metadata = []
    for position, row in enumerate(full_selected):
        sentences = sentence_count(row["source"])
        tokens = token_count(row["source"])
        entities = entity_proxy_count(row["source"])
        numbers = number_count(row["source"])
        bin_name = length_bin(sentences)
        strata[bin_name].append(row)
        paragraph_metadata.append(
            {
                "index": row["index"],
                "sentences": sentences,
                "tokens": tokens,
                "entity_proxy_count": entities,
                "number_count": numbers,
                "baseline_z": float(baseline_z[position]),
                "length_bin": bin_name,
                "z_score": row["z_score"],
                "quality_valid": row.get("quality_valid", False),
                "escaped": row["z_score"] <= THRESHOLD,
            }
        )
    stratum_results = {key: summarize(value, human_z) for key, value in sorted(strata.items())}

    token_values = np.asarray([row["tokens"] for row in paragraph_metadata])
    quartile_edges = np.quantile(token_values, [0.25, 0.5, 0.75])
    token_strata: dict[str, list[dict]] = defaultdict(list)
    for meta, selected in zip(paragraph_metadata, full_selected):
        quartile = int(np.searchsorted(quartile_edges, meta["tokens"], side="right")) + 1
        meta["token_quartile"] = f"Q{quartile}"
        token_strata[f"Q{quartile}"].append(selected)
    token_stratum_results = {
        key: summarize(value, human_z) for key, value in sorted(token_strata.items())
    }

    predictor_names = [
        "sentences",
        "tokens",
        "entity_proxy_count",
        "number_count",
        "baseline_z",
    ]
    predictors = np.asarray(
        [[row[name] for name in predictor_names] for row in paragraph_metadata], dtype=float
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(predictors)
    regression = {}
    targets = {
        "escape": np.asarray([row["escaped"] for row in paragraph_metadata], dtype=int),
        "quality_valid": np.asarray(
            [row["quality_valid"] for row in paragraph_metadata], dtype=int
        ),
    }
    targets["valid_asr"] = targets["escape"] & targets["quality_valid"]
    for target_name, target in targets.items():
        model = LogisticRegression(max_iter=5000, random_state=args.seed)
        model.fit(scaled, target)
        regression[target_name] = {
            "intercept": float(model.intercept_[0]),
            "accuracy": float(model.score(scaled, target)),
            "coefficients_per_sd": {
                name: float(value) for name, value in zip(predictor_names, model.coef_[0])
            },
            "odds_ratios_per_sd": {
                name: float(math.exp(value))
                for name, value in zip(predictor_names, model.coef_[0])
            },
        }

    output = {
        "protocol": {
            "source": "experiments/planE_e2_joint250/quality_aware_reviews.json",
            "paragraphs": len(grouped),
            "available_candidates_per_paragraph": 4,
            "threshold": THRESHOLD,
            "repeats": args.repeats,
            "seed": args.seed,
            "note": "K=8/16 require new API generation and are not estimated here.",
        },
        "candidate_budget": budget_results,
        "length_strata": stratum_results,
        "token_strata": token_stratum_results,
        "logistic_regression": regression,
        "paragraph_metadata": paragraph_metadata,
    }
    out_dir = args.root / "experiments" / "milestone_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    ks = np.arange(1, 5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for policy, label in [
        ("random", "Random"),
        ("min_z", "Minimum z"),
        ("quality_only", "Quality only"),
        ("quality_then_min_z", "Quality then minimum z"),
    ]:
        axes[0].plot(
            ks,
            [budget_results[str(k)][policy]["valid_asr"] for k in ks],
            marker="o",
            label=label,
        )
        axes[1].plot(
            ks,
            [budget_results[str(k)][policy]["mean_z"] for k in ks],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Candidates per paragraph (K)", ylabel="Valid ASR", xticks=ks)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[1].set(xlabel="Candidates per paragraph (K)", ylabel="Mean selected z-score", xticks=ks)
    axes[1].axhline(THRESHOLD, color="black", linestyle="--", linewidth=1, label="z=2.33")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_dir / "candidate_budget_ablation.png", dpi=200)
    plt.close(fig)

    lines = [
        "# Milestone A：候选预算与选择策略消融",
        "",
        "## 协议",
        "",
        f"复用 E2-250 的冻结候选池（250 段×4候选），每个 K/策略进行 {args.repeats} 次共享候选池子采样。",
        "K=8/16 需要新增 API 候选，本报告不做外推。",
        "",
        "## 候选预算结果",
        "",
        "| K | 策略 | 平均z | 逃逸率 | 质量通过率 | Valid ASR | AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "random": "随机选择",
        "min_z": "最低z",
        "quality_only": "仅质量随机",
        "quality_then_min_z": "质量后最低z",
    }
    for k in range(1, 5):
        for policy in policies:
            result = budget_results[str(k)][policy]
            lines.append(
                f"| {k} | {labels[policy]} | {result['mean_z']:.3f} | "
                f"{result['escape_rate']:.1%} | {result['quality_rate']:.1%} | "
                f"{result['valid_asr']:.1%} | {result['auroc']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 段落长度分层（K=4，质量后最低z）",
            "",
            "| 句数层 | n | 平均z | 逃逸率 | 质量通过率 | Valid ASR | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, result in stratum_results.items():
        lines.append(
            f"| {key} | {result['n']} | {result['mean_z']:.3f} | {result['escape_rate']:.1%} | "
            f"{result['quality_rate']:.1%} | {result['valid_asr']:.1%} | {result['auroc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Token长度四分位（K=4，质量后最低z）",
            "",
            "| 长度层 | n | 平均z | 逃逸率 | 质量通过率 | Valid ASR | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, result in token_stratum_results.items():
        lines.append(
            f"| {key} | {result['n']} | {result['mean_z']:.3f} | {result['escape_rate']:.1%} | "
            f"{result['quality_rate']:.1%} | {result['valid_asr']:.1%} | {result['auroc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 多变量逻辑回归（标准化系数对应的优势比）",
            "",
            "| 目标 | 句数 | token数 | 实体代理数 | 数字数 | 原始z |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target_name in ["escape", "quality_valid", "valid_asr"]:
        odds = regression[target_name]["odds_ratios_per_sd"]
        lines.append(
            f"| {target_name} | {odds['sentences']:.3f} | {odds['tokens']:.3f} | "
            f"{odds['entity_proxy_count']:.3f} | {odds['number_count']:.3f} | "
            f"{odds['baseline_z']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- K=1–4 是现有候选池内的重复子采样结果，不包含新增生成成本。",
            "- 长度分层样本高度不均衡；回归结果是探索性相关分析，不应解释为因果效应。",
            "- `quality_valid` 沿用冻结评审记录，不在本脚本中重新评价语义。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
