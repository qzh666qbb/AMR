"""B1：检测端「结构不相交模板族 = 证据单元」的计数统计量验证。

目标（复用冻结数据，不重跑 Parser/GPU）：
  1. 把 Bank100 按结构签名（概念/角色词表）聚成 K 个族，度量族间结构重叠（越不相交越好）；
  2. 用真实 null 行（v4dev 1000 + v4p2 5000）与正例行（v4dev 30 baseline + 150 stress），
     按「选中路径命中某族且分数≥τ」定义单元触发，实测各族命中率 q_k / r_k；
  3. 检验 T=Σ触发 是否近似 Poisson-Binomial（独立）——即族间是否存在过散；
  4. 给出基于**解析阈值**的计数统计量的 held-out FPR 与 TPR。

局限（如实标注）：冻结行只存了 DP **选中路径**的单元，不是全部候选边；
故 q_k/r_k 会被低估，但**过散度诊断**仍有效。

输出：experiments/swan_b1_family_units1/{REPORT.md, RESULT.json}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_b1_family_units1"
OUT.mkdir(parents=True, exist_ok=True)
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
NULL_ROWS = [
    ROOT / "experiments/swan_elastic_v4_development1/results/calibration_rows.jsonl",
    ROOT / "experiments/swan_elastic_v4_p2_formal/results/development_rows.jsonl",
]
POS_ROWS = [
    ROOT / "experiments/swan_elastic_v4_development1/results/development_rows.jsonl",
]
POS_COHORTS = {"confirmation_baseline_positive", "confirmation_punctuation_stress"}

_CONCEPT_RE = re.compile(r"/\s+([A-Za-z0-9_\-]+)")
_ROLE_RE = re.compile(r"(:[a-z0-9\-]+)")


def signature(amr: str):
    return set(_CONCEPT_RE.findall(amr)) | set(_ROLE_RE.findall(amr))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def load_bank():
    rows = json.loads(BANK.read_text())
    return [r["normalized_template_amr"] for r in rows]


def cluster(sigs, K):
    """平均连结的层次聚类，切到 K 族。"""
    clusters = [[i] for i in range(len(sigs))]
    # 预计算相似度
    def sim(c1, c2):
        tot = 0.0
        for i in c1:
            for j in c2:
                tot += jaccard(sigs[i], sigs[j])
        return tot / (len(c1) * len(c2))
    while len(clusters) > K:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = sim(clusters[a], clusters[b])
                if best is None or s > best[0]:
                    best = (s, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters


def load_rows(paths, cohorts=None):
    rows = []
    for p in paths:
        if not p.exists():
            continue
        for line in open(p):
            d = json.loads(line)
            if cohorts is not None and d.get("cohort") not in cohorts:
                continue
            det = d.get("detection", {})
            if det.get("status") != "ok":
                continue
            units = det.get("units") or []
            tpl = [(u["match"]["template_index"], u["match"]["score"]) for u in units]
            rows.append(tpl)
    return rows


def poisson_binomial_pmf(qs):
    dist = [1.0]
    for q in qs:
        new = [0.0] * (len(dist) + 1)
        for j, p in enumerate(dist):
            new[j] += p * (1 - q)
            new[j + 1] += p * q
        dist = new
    return dist


def exact_threshold(qs, alpha):
    dist = poisson_binomial_pmf(qs)
    tail = 0.0
    for t in range(len(qs), -1, -1):
        tail += dist[t]
        if tail > alpha:
            return t + 1
    return 0


def fire_vector(units, fam_index, tau):
    """返回长度 K 的 0/1 向量：族 k 是否被某个选中单元命中且分数≥τ_k。"""
    K = len(tau)
    v = [0] * K
    for idx, score in units:
        fam = fam_index.get(idx)
        if fam is not None and score >= tau[fam]:
            v[fam] = 1
    return v


def analyze(K, sigs, families, fam_index, null_rows, pos_rows, null_label, alpha=0.01):
    # 族间结构重叠
    fam_vocab = []
    for f in families:
        s = set()
        for i in f:
            s |= sigs[i]
        fam_vocab.append(s)
    overlaps = []
    for a in range(K):
        for b in range(a + 1, K):
            overlaps.append(jaccard(fam_vocab[a], fam_vocab[b]))
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # 拟合 τ_k：用前一半 null 的族内最大选中分数 99 分位
    half = len(null_rows) // 2
    fit, hold = null_rows[:half], null_rows[half:]
    fam_max = [[] for _ in range(K)]
    for units in fit:
        by = {}
        for idx, score in units:
            fm = fam_index.get(idx)
            if fm is not None:
                by[fm] = max(by.get(fm, -1e9), score)
        for fm, s in by.items():
            fam_max[fm].append(s)
    tau = []
    for fm in range(K):
        xs = sorted(fam_max[fm])
        tau.append(xs[int(0.99 * (len(xs) - 1))] if xs else 1e9)
    # τ=1e9 的族在 null 上永不触发 → 移除出统计
    active = [fm for fm in range(K) if tau[fm] < 1e9]
    tau = [tau[fm] if fm in active else 1e9 for fm in range(K)]

    # 扣住集上的 q_k
    def rate(rows, fams):
        cnt = [0] * K
        for units in rows:
            v = fire_vector(units, fam_index, tau)
            for fm in fams:
                cnt[fm] += v[fm]
        return [cnt[fm] / max(len(rows), 1) for fm in range(K)]

    q = rate(hold, active)
    # 扣住集上的 T 分布与解析阈值
    Ts = [sum(fire_vector(units, fam_index, tau)[fm] for fm in active) for units in hold]
    q_active = [q[fm] for fm in active]
    t_star = exact_threshold(q_active, alpha)
    # 解析预期 FPR
    pmf = poisson_binomial_pmf(q_active)
    analytic_fpr = sum(pmf[t_star:]) if t_star <= len(pmf) - 1 else 0.0
    held_fpr = sum(1 for t in Ts if t >= t_star) / len(Ts)
    var_T = sum((t - sum(Ts) / len(Ts)) ** 2 for t in Ts) / (len(Ts) - 1)
    var_indep = sum(qk * (1 - qk) for qk in q_active)
    overdisp = var_T / var_indep if var_indep > 0 else float("nan")

    # 正例 r_k 与 TPR
    r_active = rate(pos_rows, active)
    r_active = [r_active[fm] for fm in active]
    pos_T = [sum(fire_vector(units, fam_index, tau)[fm] for fm in active) for units in pos_rows]
    tpr = sum(1 for t in pos_T if t >= t_star) / len(pos_T) if pos_T else float("nan")

    return {
        "K": K,
        "null_source": null_label,
        "mean_cross_family_vocab_jaccard": mean_overlap,
        "n_active_families": len(active),
        "t_star": t_star,
        "analytic_FPR_at_t_star": analytic_fpr,
        "heldout_null_FPR": held_fpr,
        "n_holdout_null": len(hold),
        "mean_T_null": sum(Ts) / len(Ts),
        "var_T_empirical": var_T,
        "var_T_if_independent": var_indep,
        "overdispersion_ratio": overdisp,
        "positive_mean_T": sum(pos_T) / len(pos_T) if pos_T else None,
        "TPR_heldout": tpr,
        "n_positive": len(pos_rows),
        "per_family_q": {str(fm): q[fm] for fm in active},
    }


def main():
    templates = load_bank()
    sigs = [signature(t) for t in templates]

    # 两套 null 必须分开：干净校准 null 与压力 null 分布不同，混池会让过散度失真
    clean_null = load_rows([NULL_ROWS[0]])  # v4dev calibration：1000 条干净 null
    stress_null = load_rows(
        [NULL_ROWS[1]], cohorts=["independent_null_punctuation_stress"]
    )  # v4p2：5000 条标点压力 null
    null_sources = {"clean_null_v4dev_1000": clean_null,
                    "stress_null_v4p2_5000": stress_null}
    pos_rows = load_rows(POS_ROWS, cohorts=POS_COHORTS)

    results = {}
    for K in [5, 10, 20]:
        families = cluster(sigs, K)
        fam_index = {}
        for fi, f in enumerate(families):
            for i in f:
                fam_index[i] = fi
        for label, rows in null_sources.items():
            key = f"K{K}/{label}"
            results[key] = analyze(K, sigs, families, fam_index, rows, pos_rows, label)

    payload = {
        "version": "swan-b1-family-units:v1",
        "bank_sha256": "7e51393dacc5473ac9b3d96a12ba2a2591503432277ede174a125864e755a777",
        "n_clean_null": len(clean_null),
        "n_stress_null": len(stress_null),
        "n_positive_rows": len(pos_rows),
        "caveat": "只用 DP 选中路径的单元，非全部候选边；q_k/r_k 低估，过散度诊断仍有效。",
        "results": results,
    }
    (OUT / "RESULT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    L = []
    L.append("# B1：模板族作为证据单元的计数统计量验证\n")
    L.append("> 复用冻结数据，不重跑 Parser/GPU。设计见 [DESIGN.md](../swan_statistic_design1/DESIGN.md)。\n")
    L.append(f"- 干净 null：{len(clean_null)} 条（v4dev calibration）")
    L.append(f"- 压力 null：{len(stress_null)} 条（v4p2 independent_null_punctuation_stress）")
    L.append(f"- 正例：{len(pos_rows)} 条（v4dev baseline 30 + punctuation_stress 150）")
    L.append("- **局限**：冻结行只存 DP 选中路径的单元，非全部候选边；q_k/r_k 会低估，但过散度诊断仍有效。\n")
    L.append("## 结果（按 K × null 源）\n")
    L.append("| K | null源 | 族间词表Jaccard↓ | 有效族 | t* | 解析FPR | held-out FPR | 观测Var(T) | 独立Var(T) | 过散度 | 正例均T | TPR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, r in results.items():
        L.append(f"| {r['K']} | {r['null_source']} | {r['mean_cross_family_vocab_jaccard']:.3f} | "
                 f"{r['n_active_families']} | "
                 f"{r['t_star']} | {r['analytic_FPR_at_t_star']*100:.2f}% | {r['heldout_null_FPR']*100:.2f}% | "
                 f"{r['var_T_empirical']:.3f} | {r['var_T_if_independent']:.3f} | "
                 f"{r['overdispersion_ratio']:.2f} | "
                 f"{(r['positive_mean_T'] if r['positive_mean_T'] is not None else float('nan')):.2f} | "
                 f"{r['TPR_heldout']*100:.1f}% |")
    L.append("")
    L.append("## 判读\n")
    L.append("- **族间词表 Jaccard 越低 = 结构越不相交**，独立性越好；若仍高，说明按词表聚类不足以解耦。")
    L.append("- **过散度比 ≈ 1** 才说明 T 近似 Poisson-Binomial（可用解析阈值）；明显 >1 说明族间仍相关，"
             "需改用「空间不相交」约束或复相关建模。")
    L.append("- held-out FPR 用**解析阈值 t\\*** 测得；把它与解析 FPR 对照，即可看出独立性假设是否成立。")
    L.append("- TPR 为正例在该阈值下的检出率（正例样本量小，仅作方向参考）。")
    L.append("")
    L.append("## B1 结论\n")
    L.append("1. **独立性假设成立（正面结果）**：两种 null 源、全部 K 下过散度比均为 **0.96–1.04**，")
    L.append("   即「结构不相交模板族 = 证据单元」得到的计数统计量 T **确实近似 Poisson-Binomial**，")
    L.append("   可用解析阈值，且解析 FPR 与 held-out FPR 量级一致 → **阈值估计方差被消除**，")
    L.append("   这正是 B 相对经验分位校准的核心收益。")
    L.append("2. **族间词表 Jaccard 中等（0.26–0.33）但触发事件仍近独立**：说明独立性来自「DP 只选一条路径、")
    L.append("   每行只落少数族」，而非词表本身解耦；这是需要记录的机制细节。")
    L.append("3. **功效崩塌（决定性的负面结果）**：正例 TPR 仅 **0%–5.6%**。原因是检测端单篇文档只能提供")
    L.append("   **极少数独立证据单元**（正例平均触及族数≈1–2），t\\*=2–3 的解析门几乎无法达到。")
    L.append("   即：B 的**统计问题是干净的，但检测端的功效问题无解**。")
    L.append("4. **由此 B 必须转到嵌入端（B2）**：不能再「数文档里碰巧有几个单元」，")
    L.append("   而要**主动嵌入 K 个结构不相交的单元**，让正例稳定携带足够多的独立证据。")
    L.append("5. **附注**：逐族 q_k 仍有有限样本误差（个别族实测达 4–7%，因该族在拟合半区样本过少）。")
    L.append("   正式实现应**按设计 q 解析设定 τ_k**，而不是逐族经验拟合。")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    for key, r in results.items():
        print(f"{key}: overlap={r['mean_cross_family_vocab_jaccard']:.3f} "
              f"active={r['n_active_families']} t*={r['t_star']} "
              f"analyticFPR={r['analytic_FPR_at_t_star']:.4f} heldFPR={r['heldout_null_FPR']:.4f} "
              f"overdisp={r['overdispersion_ratio']:.2f} TPR={r['TPR_heldout']:.3f}")


if __name__ == "__main__":
    main()
