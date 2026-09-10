"""Analyze frozen Plan E strict candidates under alternate SWAN bank subsets."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/candidate_budget16_pilot30/feedback_driven_strict30"
TARGET = ROOT / "experiments/planE_transfer_strict30"
THRESHOLD = 2.33


def exact_ci(successes: int, total: int) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return [lower, upper]


def main() -> None:
    audit = json.loads((SOURCE / "assistant_audit.json").read_text(encoding="utf-8"))["rows"]
    source_scores = np.load(SOURCE / "detection/machine_z_scores.npy")
    if len(audit) != len(source_scores):
        raise ValueError("Audit rows and source scores are not aligned")

    valid = np.array([bool(row["assistant_quality_valid"]) for row in audit])
    paragraph = np.array([int(row["index"]) for row in audit])
    source_escape = source_scores <= THRESHOLD
    source_successes = sorted(set(paragraph[valid & source_escape].tolist()))

    results = {}
    for name in ("first25", "random25"):
        payload = json.loads((TARGET / f"{name}.json").read_text(encoding="utf-8"))
        target_scores = np.array([row["z_score"] for row in payload["documents"]], dtype=float)
        if len(target_scores) != len(audit):
            raise ValueError(f"{name}: target scores and audit rows are not aligned")

        target_escape = target_scores <= THRESHOLD
        retained = [
            index
            for index in source_successes
            if bool(np.any(valid & (paragraph == index) & target_escape))
        ]
        target_known_valid = sorted(set(paragraph[valid & target_escape].tolist()))
        results[name] = {
            "bank_size": payload["contract"]["bank_size"],
            "all_candidate_escape": int(np.sum(target_escape)),
            "all_candidates": len(target_scores),
            "known_strict_valid_candidate_escape": int(np.sum(valid & target_escape)),
            "known_strict_valid_candidates": int(np.sum(valid)),
            "target_known_strict_success_paragraphs": target_known_valid,
            "target_known_strict_valid_asr_lower_bound": len(target_known_valid) / 30,
            "source_strict_success_paragraphs": source_successes,
            "retained_source_success_paragraphs": retained,
            "atr": len(retained) / len(source_successes),
            "atr_exact_ci95": exact_ci(len(retained), len(source_successes)),
        }

    output = {
        "scope": "Frozen Plan E strict candidates rescored with two 25-template subsets of the original public 50-template SWAN bank",
        "threshold": THRESHOLD,
        "source_bank_size": 50,
        "source_candidate_escape": int(np.sum(source_escape)),
        "source_candidates": len(source_scores),
        "source_strict_success_paragraphs": source_successes,
        "quality_label": "Existing AI-assistant strict audit; not human-subject annotation",
        "results": results,
        "limitations": [
            "The target banks are subsets of the source bank, not independently constructed banks.",
            "The attack texts were frozen, but the underlying watermarked texts were not generated with the target subsets.",
            "ATR has denominator 9, so its exact confidence interval is wide.",
            "Target-bank clean detection power and target-specific human-score calibration were not re-established in this test.",
        ],
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Plan E 冻结候选的 Bank 子集迁移测试",
        "",
        "本测试冻结 Plan E 严格实验的 120 个候选及既有质量标签，仅更换检测 Bank 后重新评分。",
        "",
        "| 目标 Bank | 全候选逃逸 | 已审计严格候选逃逸 | 源成功段落保留 | ATR | 95% 精确区间 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        lo, hi = row["atr_exact_ci95"]
        lines.append(
            f"| {name} | {row['all_candidate_escape']}/{row['all_candidates']} | "
            f"{row['known_strict_valid_candidate_escape']}/{row['known_strict_valid_candidates']} | "
            f"{len(row['retained_source_success_paragraphs'])}/{len(row['source_strict_success_paragraphs'])} | "
            f"{row['atr']:.1%} | [{lo:.1%}, {hi:.1%}] |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "两个子 Bank 上观测 ATR 均为 100%，且源 Bank 下的 16 个严格合格逃逸候选全部维持逃逸。",
        "但目标 Bank 只是原 50 模板 Bank 的子集，不能代表独立秘密 Bank；子集检测下全候选逃逸数还从源 Bank 的 26/120 上升到 34/120 和 30/120，表明目标检测器整体更弱。",
        "因此该结果只能作为 Bank 删减扰动下的初步迁移证据，不能作为跨独立 Bank 的正式 ATR 结论。",
        "",
        "质量标签来自既有 AI 助手严格审计，不应表述为真人评审。",
    ]
    (TARGET / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: row["atr"] for name, row in results.items()}))


if __name__ == "__main__":
    main()
