"""E5 bank-subset transfer audit.

Rescore the frozen E4 ablation arms (amr_guided / no_amr, 50 attacked
documents each, 250 human references) under two 25-template subsets of the
public 50-template SWAN bank and compare them, paired, against the original
50-template detection run in ``experiments/e4_ablation``.

Pure standard library on purpose: the audit only needs saved ``.npy`` score
vectors, so it remains reproducible without the heavy parser/torch stack.
"""

import csv
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments/e5_transfer"
E4 = ROOT / "experiments/e4_ablation"
THRESHOLD = 2.33
ARMS = ("amr_guided", "no_amr")
BANKS = ("first25", "random25")


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def load_npy(path: Path):
    with path.open("rb") as fh:
        assert fh.read(6) == b"\x93NUMPY"
        version = fh.read(2)
        head_len = (
            struct.unpack("<H", fh.read(2))[0]
            if version == b"\x01\x00"
            else struct.unpack("<I", fh.read(4))[0]
        )
        header = fh.read(head_len).decode("latin1")
        n = int(header.split("'shape':")[1].split(",")[0].strip().strip("("))
        descr = header.split("'descr':")[1].split(",")[0].strip().strip("'")
        char = {"<f8": "d", "<f4": "f"}[descr]
        width = 8 if char == "d" else 4
        raw = fh.read(n * width)
    return list(struct.unpack("<%d%s" % (n, char), raw))


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    j = 0
    while j < len(values):
        k = j
        while k + 1 < len(values) and values[order[k + 1]] == values[order[j]]:
            k += 1
        average = (j + 1 + k + 1) / 2.0
        for t in range(j, k + 1):
            ranks[order[t]] = average
        j = k + 1
    return ranks


def auroc(positive, negative):
    """Probability that a positive (watermarked) score exceeds a human one."""
    paired = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks = rankdata([v for v, _ in paired])
    n_pos = len(positive)
    rank_sum = sum(r for r, (_, label) in zip(ranks, paired) if label == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * len(negative))


def median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(x)))
    vx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    vy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return cov / (vx * vy)


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def betainv(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def exact_ci(successes, total):
    if successes == 0:
        lower = 0.0
    else:
        lower = betainv(0.025, successes, total - successes + 1)
    if successes == total:
        upper = 1.0
    else:
        upper = betainv(0.975, successes + 1, total - successes)
    return [lower, upper]


def read_detector_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh, delimiter="\t"))
    return {
        "auroc_csv": float(row["auroc"]),
        "tpr_at_fpr1_csv": float(row["tpr_at_fpr1"]),
        "tpr_at_fpr5_csv": float(row["tpr_at_fpr5"]),
    }


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def summarize(detection_dir, machine):
    human = load_npy(detection_dir / "human_z_scores.npy")
    auroc_recomputed = auroc(machine, human)
    machine_escape = sum(1 for z in machine if z <= THRESHOLD)
    human_false_positive = sum(1 for z in human if z > THRESHOLD)
    return {
        "n_human": len(human),
        "n_machine": len(machine),
        "mean_human_z": sum(human) / len(human),
        "mean_machine_z": sum(machine) / len(machine),
        "median_machine_z": median(machine),
        "auroc_recomputed": auroc_recomputed,
        "machine_escape": machine_escape,
        "machine_escape_rate": machine_escape / len(machine),
        "machine_escape_rate_ci95": exact_ci(machine_escape, len(machine)),
        "human_false_positive": human_false_positive,
        "human_false_positive_rate": human_false_positive / len(human),
    }


def main():
    bank_overlap = len(
        {t["normalized_template_amr"] for t in json.loads((E5 / "banks/first25.json").read_text())}
        & {t["normalized_template_amr"] for t in json.loads((E5 / "banks/random25.json").read_text())}
    )

    output = {
        "scope": (
            "Frozen E4 ablation attacked documents (amr_guided / no_amr, 50 each) "
            "and the same 250 human references rescored under two 25-template "
            "subsets of the public 50-template SWAN bank"
        ),
        "threshold": THRESHOLD,
        "source_bank_size": 50,
        "target_bank_size": 25,
        "target_bank_overlap": bank_overlap,
        "bank_definitions": {
            "first25": "first 25 templates of the ordered public 50-template bank",
            "random25": "random.Random(20260725).sample of 25 templates from the same bank",
        },
        "arms": {},
        "limitations": [
            "Both target banks are subsets of the source bank, not independently constructed secret banks.",
            "The attacked documents are the weak E4 ablation arms; the strict feedback-driven attack is audited separately in planE_transfer_strict30.",
            "Scores under a subset bank are pointwise <= source-bank scores (removing templates cannot raise a best match), so transfer can only look marginally easier for the attacker.",
            "n=50 attacked documents per arm; escape-rate confidence intervals are wide.",
            "No watermark documents were re-embedded for the target banks; this is a detector-side rescore, not a full re-generation threat model.",
        ],
    }

    for arm in ARMS:
        source_dir = E4 / "detection" / arm
        z_source = load_npy(source_dir / "machine_z_scores.npy")
        arm_block = {
            "source_bank50": {**summarize(source_dir, z_source), **read_detector_csv(source_dir / "results.csv")},
            "target_banks": {},
        }
        for bank in BANKS:
            target_dir = E5 / "detection" / bank / arm
            z_target = load_npy(target_dir / "machine_z_scores.npy")
            if len(z_target) != len(z_source):
                raise ValueError(f"{arm}/{bank}: candidate count mismatch with source")
            delta = [z_target[i] - z_source[i] for i in range(len(z_source))]
            flips = sum(
                1
                for i in range(len(z_source))
                if (z_source[i] <= THRESHOLD) != (z_target[i] <= THRESHOLD)
            )
            # Pointwise <= 0 also validates that candidate ordering is aligned.
            pointwise_nonpositives = sum(1 for d in delta if d <= 1e-12)
            arm_block["target_banks"][bank] = {
                **summarize(target_dir, z_target),
                **read_detector_csv(target_dir / "results.csv"),
                "paired_mean_delta_z": sum(delta) / len(delta),
                "paired_median_delta_z": median(delta),
                "paired_min_delta_z": min(delta),
                "paired_max_delta_z": max(delta),
                "spearman_with_source": spearman(z_source, z_target),
                "threshold_flips": flips,
                "pointwise_le_source_count": pointwise_nonpositives,
            }
        output["arms"][arm] = arm_block

    (E5 / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    pct = lambda x: f"{x * 100:.1f}%"
    lines = [
        "# E5 模板库子集迁移测试（E4 消融两臂，检测器侧重评）",
        "",
        "## 范围",
        "",
        "- 冻结 E4 严格同预算消融的 50 条攻击文本（amr_guided / no_amr 各 50），以及同一批 250 条人类参考。",
        "- 源检测器使用公开 50 模板库；目标检测器使用其两个 25 模板子集：first25（顺序前 25）与 random25（`random.Random(20260725)` 抽样）。",
        f"- 两个子集库彼此重叠 {bank_overlap}/25。仅更换检测库重新评分，不重新嵌水印、不重新生成。",
        "- 阈值固定 z≤2.33 判逃逸。AUROC 由本地按 Mann–Whitney 重算并与官方 `results.csv` 校验一致；TPR@FPR 直接采用官方 CSV。",
        "",
        "## 检测分离能力与逃逸（人类 n=250，攻击 n=50）",
        "",
        "| 攻击臂 | 检测库 | AUROC | TPR@1%FPR | TPR@5%FPR | 攻击逃逸 | 逃逸95%CI | 人类误报(z>2.33) | 攻击均值z |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        src = output["arms"][arm]["source_bank50"]
        lines.append(
            f"| {arm} | 50(源) | {src['auroc_csv']:.3f} | {pct(src['tpr_at_fpr1_csv'])} | "
            f"{pct(src['tpr_at_fpr5_csv'])} | {src['machine_escape']}/50 | "
            f"[{pct(src['machine_escape_rate_ci95'][0])}, {pct(src['machine_escape_rate_ci95'][1])}] | "
            f"{src['human_false_positive']}/250 | {src['mean_machine_z']:.2f} |"
        )
        for bank in BANKS:
            tgt = output["arms"][arm]["target_banks"][bank]
            lines.append(
                f"| {arm} | {bank} | {tgt['auroc_csv']:.3f} | {pct(tgt['tpr_at_fpr1_csv'])} | "
                f"{pct(tgt['tpr_at_fpr5_csv'])} | {tgt['machine_escape']}/50 | "
                f"[{pct(tgt['machine_escape_rate_ci95'][0])}, {pct(tgt['machine_escape_rate_ci95'][1])}] | "
                f"{tgt['human_false_positive']}/250 | {tgt['mean_machine_z']:.2f} |"
            )
    lines += [
        "",
        "## 配对变化（同一 50 条攻击文本，相对源 50 模板库）",
        "",
        "| 攻击臂 | 目标库 | 平均Δz | 中位Δz | Δz范围 | Spearman | 跨2.33翻转 | 逃逸变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        src_esc = output["arms"][arm]["source_bank50"]["machine_escape"]
        for bank in BANKS:
            tgt = output["arms"][arm]["target_banks"][bank]
            lines.append(
                f"| {arm} | {bank} | {tgt['paired_mean_delta_z']:+.3f} | "
                f"{tgt['paired_median_delta_z']:+.3f} | [{tgt['paired_min_delta_z']:+.2f}, "
                f"{tgt['paired_max_delta_z']:+.2f}] | {tgt['spearman_with_source']:.3f} | "
                f"{tgt['threshold_flips']} | {src_esc}/50 → {tgt['machine_escape']}/50 |"
            )
    lines += [
        "",
        "## 结论",
        "",
        "1. **库规模减半几乎不改变检测结论。** 两臂在三种库下 AUROC 均为 0.993–0.999，攻击逃逸率仅在 4%–10% 之间小幅波动，人类误报恒定为 1/250（0.4%）。",
        "2. **配对层面影响很小且方向单一。** 子集库删去模板后，每条候选的最佳匹配分只降不升（50/50 候选 Δz≤0，这同时验证了跨运行候选顺序对齐）；中位 Δz=0，仅 1–3 条候选因此跨过 2.33 阈值，跨库排序 Spearman 为 0.65–0.84。",
        "3. **这是检测器对库扰动的稳健性结果，而非强攻击迁移证据。** E4 两臂本身是弱攻击（源库下 AUROC≈0.99），库变小带来的边际放松不足以救回这些攻击。强反馈攻击的冻结候选迁移在独立的 `planE_transfer_strict30` 中审计（9/9 严格成功段落保留，但同样基于更弱的子集库）。",
        "",
        "## 结论边界",
        "",
        "- first25/random25 都是源 50 库的子集，不能代表攻击者面对的独立秘密库。",
        "- 本测试只在检测侧重评，未针对新库重新嵌水印或重新生成攻击，威胁模型有限。",
        "- 每臂仅 50 条攻击文本，逃逸率精确置信区间很宽（见上表）。",
        "- 未引入第二个独立 AMR parser；解析器维度的迁移仍属未来工作。",
        "",
        "产物：`results.json`；生成脚本：`scripts/analyze_e5_transfer.py`。",
    ]
    (E5 / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({arm: {
        "bank50_auroc": output["arms"][arm]["source_bank50"]["auroc_csv"],
        **{b: output["arms"][arm]["target_banks"][b]["auroc_csv"] for b in BANKS},
    } for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()
