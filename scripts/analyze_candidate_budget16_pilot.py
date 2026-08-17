"""Merge detection chunks and analyze the K=1/2/4/8/16 candidate budget pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


THRESHOLD = 2.33
K_VALUES = (1, 2, 4, 8, 16)
REPEATS = 1000
SEED = 20260818


def metric(
    selected: np.ndarray, human: np.ndarray, selected_quality: np.ndarray | None = None
) -> dict[str, float]:
    result = {
        "mean_z": float(selected.mean()),
        "escape_rate": float(np.mean(selected <= THRESHOLD)),
        "auroc": float(
            roc_auc_score(
                np.r_[np.zeros(len(human)), np.ones(len(selected))],
                np.r_[human, selected],
            )
        ),
    }
    if selected_quality is not None:
        result["selected_quality_rate"] = float(np.mean(selected_quality))
        result["automatic_valid_asr"] = float(np.mean((selected <= THRESHOLD) & selected_quality))
    return result


root = Path(__file__).resolve().parents[1]
exp = root / "experiments" / "candidate_budget16_pilot30"
rows = json.loads((exp / "rows.json").read_text(encoding="utf-8"))
validation = json.loads((exp / "validation.json").read_text(encoding="utf-8"))["rows"]

z_chunks = []
human_chunks = []
parsed = []
matches = []
for chunk_index in range(4):
    chunk = exp / "chunks" / f"chunk{chunk_index}"
    z_chunks.append(np.load(chunk / "detection" / "machine_z_scores.npy"))
    human_chunks.append(np.load(chunk / "detection" / "human_z_scores.npy"))
    parsed.extend(json.loads((chunk / "parsed_amrs.json").read_text(encoding="utf-8")))
    matches.extend(
        json.loads((chunk / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8"))[
            "matched_amrs"
        ]
    )
human = np.load(
    root
    / "baselines"
    / "local"
    / "swan_acl2026_realnews"
    / "runs"
    / "formal_250x5"
    / "detection"
    / "human_z_scores.npy"
)
human_reference_disagreements = [int(np.sum(candidate != human)) for candidate in human_chunks]
z = np.concatenate(z_chunks)
if not (len(rows) == len(validation) == len(parsed) == len(matches) == len(z) == 480):
    raise RuntimeError("Merged artifact lengths do not equal 480")

z_matrix = z.reshape(30, 16)
quality = np.asarray(
    [row["numbers_preserved"] and row["negation_preserved"] for row in validation], dtype=bool
).reshape(30, 16)
nonempty = np.asarray(
    [np.mean([bool(amr.strip()) for amr in doc]) if doc else 0.0 for doc in parsed]
).reshape(30, 16)

rng = np.random.default_rng(SEED)
summary = {}
for k in K_VALUES:
    prefix = z_matrix[:, :k]
    prefix_quality = quality[:, :k]
    min_selected = prefix.min(axis=1)
    min_indices = prefix.argmin(axis=1)
    min_selected_quality = prefix_quality[np.arange(30), min_indices]
    quality_selected = np.empty(30)
    quality_selected_pass = np.empty(30, dtype=bool)
    quality_available = np.any(prefix_quality, axis=1)
    for paragraph in range(30):
        allowed = np.flatnonzero(prefix_quality[paragraph])
        pool = allowed if len(allowed) else np.arange(k)
        selected_index = pool[np.argmin(prefix[paragraph, pool])]
        quality_selected[paragraph] = prefix[paragraph, selected_index]
        quality_selected_pass[paragraph] = prefix_quality[paragraph, selected_index]

    random_runs = []
    subset_min_runs = []
    subset_quality_runs = []
    for _ in range(REPEATS):
        random_pick = rng.integers(0, 16, size=30)
        random_runs.append(
            metric(
                z_matrix[np.arange(30), random_pick],
                human,
                quality[np.arange(30), random_pick],
            )
        )
        chosen = np.stack([rng.choice(16, size=k, replace=False) for _ in range(30)])
        subset = np.take_along_axis(z_matrix, chosen, axis=1)
        subset_q = np.take_along_axis(quality, chosen, axis=1)
        subset_min_indices = subset.argmin(axis=1)
        subset_min_runs.append(
            metric(
                subset[np.arange(30), subset_min_indices],
                human,
                subset_q[np.arange(30), subset_min_indices],
            )
        )
        selected_q = np.empty(30)
        selected_q_pass = np.empty(30, dtype=bool)
        for paragraph in range(30):
            allowed = np.flatnonzero(subset_q[paragraph])
            pool = allowed if len(allowed) else np.arange(k)
            selected_index = pool[np.argmin(subset[paragraph, pool])]
            selected_q[paragraph] = subset[paragraph, selected_index]
            selected_q_pass[paragraph] = subset_q[paragraph, selected_index]
        subset_quality_runs.append(metric(selected_q, human, selected_q_pass))

    def aggregate(run_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        return {
            name: {
                "mean": float(np.mean([row[name] for row in run_rows])),
                "p025": float(np.quantile([row[name] for row in run_rows], 0.025)),
                "p975": float(np.quantile([row[name] for row in run_rows], 0.975)),
            }
            for name in run_rows[0]
        }

    summary[str(k)] = {
        "prefix_min_z": metric(min_selected, human, min_selected_quality),
        "prefix_quality_then_min_z": metric(quality_selected, human, quality_selected_pass),
        "prefix_quality_available_rate": float(quality_available.mean()),
        "random_single_control": aggregate(random_runs),
        "random_subset_min_z": aggregate(subset_min_runs),
        "random_subset_quality_then_min_z": aggregate(subset_quality_runs),
    }

artifact = {
    "protocol": {
        "paragraphs": 30,
        "candidates_per_paragraph": 16,
        "k_values": K_VALUES,
        "random_subsampling_repeats": REPEATS,
        "seed": SEED,
        "escape_threshold": THRESHOLD,
    },
    "quality": {
        "candidate_pass_rate": float(quality.mean()),
        "all_candidates_amr_nonempty_rate": float(nonempty.mean()),
        "fully_nonempty_document_rate": float(np.mean(nonempty == 1.0)),
        "chunk_human_reference_disagreements_vs_formal": human_reference_disagreements,
    },
    "summary": summary,
}
(exp / "analysis.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

x = np.asarray(K_VALUES)
escape = np.asarray([summary[str(k)]["random_subset_min_z"]["escape_rate"]["mean"] for k in K_VALUES])
quality_escape = np.asarray(
    [summary[str(k)]["random_subset_quality_then_min_z"]["escape_rate"]["mean"] for k in K_VALUES]
)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].plot(x, escape, "o-", label="Min z")
axes[0].plot(x, quality_escape, "o-", label="Quality then min z")
axes[0].set(xlabel="Candidate budget K", ylabel="Escape rate", xticks=x, ylim=(0, 1.03))
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[1].plot(
    x,
    [summary[str(k)]["random_subset_min_z"]["mean_z"]["mean"] for k in K_VALUES],
    "o-",
    label="Min z",
)
axes[1].plot(
    x,
    [summary[str(k)]["random_subset_quality_then_min_z"]["mean_z"]["mean"] for k in K_VALUES],
    "o-",
    label="Quality then min z",
)
axes[1].set(xlabel="Candidate budget K", ylabel="Selected mean z", xticks=x)
axes[1].grid(alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(exp / "candidate_budget_curve.png", dpi=200)
plt.close(fig)

lines = [
    "# 候选预算 K=1/2/4/8/16 先导实验",
    "",
    "同一批30个段落，每段一次API调用生成16个候选；对16候选随机无放回抽取K个并重复1000次。",
    "自动质量约束仅检查数字序列与否定词数量，不能代替盲法人工语义评审。",
    "",
    "| K | 最低z逃逸率 | 最低z自动Valid ASR | 质量优先逃逸率 | 质量优先自动Valid ASR | 最低z均值 | AUROC |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for k in K_VALUES:
    low = summary[str(k)]["random_subset_min_z"]
    constrained = summary[str(k)]["random_subset_quality_then_min_z"]
    lines.append(
        f"| {k} | {low['escape_rate']['mean']:.1%} | {low['automatic_valid_asr']['mean']:.1%} | "
        f"{constrained['escape_rate']['mean']:.1%} | {constrained['automatic_valid_asr']['mean']:.1%} | "
        f"{low['mean_z']['mean']:.3f} | {low['auroc']['mean']:.3f} |"
    )
lines.extend(
    [
        "",
        "## 完整性检查",
        "",
        f"- 数字与否定词自动约束同时通过：{quality.mean():.1%}。",
        f"- 候选AMR平均非空率：{nonempty.mean():.1%}。",
        f"- 所有句子AMR均非空的候选比例：{np.mean(nonempty == 1.0):.1%}。",
        f"- 多进程重算的人类基准与预注册正式基准分别有{human_reference_disagreements}个元素不一致；"
        "AUROC统一采用此前formal_250x5正式基准，未按本次结果挑选。",
        "",
        "## 解释限制",
        "",
        "本实验隔离了候选预算的检测收益，但样本量只有30段。随机子采样区间反映候选抽样波动，"
        "不等价于跨段落总体置信区间。最终Valid ASR需完成盲法人工质量评价后才能报告。",
    ]
)
(exp / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(artifact["quality"], ensure_ascii=False))
for k in K_VALUES:
    print(k, summary[str(k)]["random_subset_min_z"], summary[str(k)]["random_subset_quality_then_min_z"])
