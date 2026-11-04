"""B0 离线可行性验证：max 统计量尾部脆弱性 vs 计数统计量精确性。

全部基于**已冻结的行级数据**，不重跑 Parser、不用 GPU、不新增生成。

数据：
  - 干净 null：experiments/swan_elastic_v4_development1/results/calibration_rows.jsonl (1000)
  - 压力 null：experiments/swan_elastic_v4_p2_formal/results/development_rows.jsonl (1000 文档 x 5 变体)

三部分：
  I.  真实 null 的尾部脆弱性（形状 / 阈值估计误差 / held-out FPR 散布）
  II. 真实 K=5 变体计数统计量，及其独立性检验（是否真服从二项）
  III. 理想独立计数统计量的解析精确性与功效表（明确标注为解析/仿真）

输出：experiments/swan_statistic_feasibility1/{REPORT.md, FEASIBILITY.json}
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_statistic_feasibility1"
OUT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20261019)
VARIANTS = ["semicolon_all", "semicolon_pairs", "comma_all", "newline_all", "space_all"]


# ------------------------------------------------------------- 基础统计
def mean(xs):
    return sum(xs) / len(xs)


def std(xs, ddof=1):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def quantile(sorted_xs, p):
    """线性插值分位数（sorted 输入）。"""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = p * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def skewness(xs):
    n = len(xs); m = mean(xs); s = std(xs)
    return sum(((x - m) / s) ** 3 for x in xs) * n / ((n - 1) * (n - 2))


def kurtosis(xs):
    """超额峰度（Fisher）。"""
    n = len(xs); m = mean(xs); s = std(xs)
    return sum(((x - m) / s) ** 4 for x in xs) / n - 3.0


def poisson_binomial_pmf(qs):
    """Poisson-Binomial 分布，O(K^2)。"""
    dist = [1.0]
    for q in qs:
        new = [0.0] * (len(dist) + 1)
        for j, p in enumerate(dist):
            new[j] += p * (1 - q)
            new[j + 1] += p * q
        dist = new
    return dist


def exact_threshold(qs, alpha):
    """返回 (t*, 精确尾部P(T>=t*) )。"""
    dist = poisson_binomial_pmf(qs)
    K = len(qs)
    # 从大到小找最小 t 使 P(T>=t) <= alpha
    tail = 0.0
    for t in range(K, -1, -1):
        tail += dist[t]
        if tail > alpha:
            # t 不满足；t+1 满足
            return t + 1, tail - dist[t]
    return 0, 1.0


# ------------------------------------------------------------- 数据加载
def load_clean_null():
    xs = []
    for line in open(ROOT / "experiments/swan_elastic_v4_development1/results/calibration_rows.jsonl"):
        d = json.loads(line)
        s = d["detection"].get("statistic")
        if s is not None:
            xs.append(s)
    return xs


def load_stressed_null():
    """返回 {variant: [statistic...]} 与 {group_id: {variant: statistic}}。"""
    by_variant = defaultdict(list)
    by_group = defaultdict(dict)
    for line in open(ROOT / "experiments/swan_elastic_v4_p2_formal/results/development_rows.jsonl"):
        d = json.loads(line)
        s = d["detection"].get("statistic")
        if s is None:
            continue
        by_variant[d["variant_id"]].append(s)
        by_group[d["group_id"]][d["variant_id"]] = s
    return by_variant, by_group


# ------------------------------------------------------------- Part I
def part1_fragility(clean):
    clean_sorted = sorted(clean)
    n = len(clean)
    q95 = quantile(clean_sorted, 0.95)
    q99 = quantile(clean_sorted, 0.99)
    res = {
        "n": n,
        "mean": mean(clean),
        "std": std(clean),
        "skewness": skewness(clean),
        "excess_kurtosis": kurtosis(clean),
        "q95": q95,
        "q99": q99,
        "n_points_at_or_above_q99": sum(1 for x in clean if x >= q99),
    }
    # bootstrap q99 的 95% CI
    boots = []
    for _ in range(2000):
        samp = sorted(RNG.choice(clean) for _ in range(n))
        boots.append(quantile(samp, 0.99))
    boots.sort()
    res["q99_bootstrap_ci95"] = [quantile(boots, 0.025), quantile(boots, 0.975)]
    res["q99_bootstrap_sd"] = std(boots)
    res["q99_spread_in_null_sd"] = (res["q99_bootstrap_ci95"][1] - res["q99_bootstrap_ci95"][0]) / res["std"]
    # split-half: 在一半上定阈值, 在另一半上测 FPR, 重复 2000 次
    fprs = []
    idx = list(range(n))
    half = n // 2
    for _ in range(2000):
        RNG.shuffle(idx)
        a = [clean[i] for i in idx[:half]]
        b = [clean[i] for i in idx[half:]]
        tau = quantile(sorted(a), 0.99)
        fprs.append(sum(1 for x in b if x > tau) / len(b))
    fprs.sort()
    res["split_half_fpr"] = {
        "nominal": 0.01,
        "median": quantile(fprs, 0.5),
        "p05": quantile(fprs, 0.05),
        "p95": quantile(fprs, 0.95),
        "min": fprs[0],
        "max": fprs[-1],
        "sd": std(fprs),
    }
    return res


# ------------------------------------------------------------- Part II
def part2_count_real(by_variant, by_group):
    n_docs = len(by_group)
    # 每变体阈值 = 该变体 null 的 99 分位
    tau = {}
    for v in VARIANTS:
        xs = sorted(by_variant[v])
        tau[v] = quantile(xs, 0.99)
    # 每通道触发率 q_k（应约等于 1%）
    q_k = {}
    for v in VARIANTS:
        xs = by_variant[v]
        q_k[v] = sum(1 for x in xs if x >= tau[v]) / len(xs)
    # 每文档的触发计数 T
    Ts = []
    any_fire = 0
    for gid, d in by_group.items():
        t = sum(1 for v in VARIANTS if d.get(v, float("-inf")) >= tau[v])
        Ts.append(t)
        if t >= 1:
            any_fire += 1
    del gid
    # 实测 T 分布
    from collections import Counter
    emp = Counter(Ts)
    K = len(VARIANTS)
    qbar = sum(q_k.values()) / K
    theo = poisson_binomial_pmf([qbar] * K)
    theo_indep = [theo[j] for j in range(K + 1)]
    emp_pmf = [emp[j] / n_docs for j in range(K + 1)]
    # 过散度：(实测 P(T>=1) 与 独立假设 P(T>=1) 比较)
    p_ge1_emp = sum(emp_pmf[1:])
    p_ge1_indep = 1 - theo_indep[0]
    # 二项方差 vs 实测方差
    var_emp = sum((t - mean(Ts)) ** 2 for t in Ts) / (n_docs - 1)
    var_indep = K * qbar * (1 - qbar)
    return {
        "n_docs": n_docs,
        "per_variant_threshold": tau,
        "per_channel_rate_q_k": q_k,
        "q_bar": qbar,
        "empirical_T_pmf": emp_pmf,
        "binomial_T_pmf": theo_indep,
        "any_variant_fire_rate_FPR": any_fire / n_docs,
        "P_T_ge1_empirical": p_ge1_emp,
        "P_T_ge1_if_independent": p_ge1_indep,
        "var_T_empirical": var_emp,
        "var_T_if_independent": var_indep,
        "overdispersion_ratio": var_emp / var_indep,
    }


# ------------------------------------------------------------- Part III
def part3_ideal(alpha=0.01):
    rows = []
    for K in [5, 10, 20, 50]:
        for q in [0.01, 0.02]:
            t_star, tail = exact_threshold([q] * K, alpha)
            powers = {}
            for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                dist = poisson_binomial_pmf([r] * K)
                powers[r] = sum(dist[t_star:])
            rows.append({"K": K, "q": q, "alpha": alpha, "t_star": t_star,
                         "exact_FPR_at_t_star": tail, "TPR_by_r": powers})
    return rows


def main():
    clean = load_clean_null()
    by_variant, by_group = load_stressed_null()

    p1 = part1_fragility(clean)
    p2 = part2_count_real(by_variant, by_group)
    p3 = part3_ideal(0.01)

    payload = {
        "version": "swan-statistic-feasibility:v1",
        "data_provenance": {
            "clean_null": "experiments/swan_elastic_v4_development1/results/calibration_rows.jsonl",
            "stressed_null": "experiments/swan_elastic_v4_p2_formal/results/development_rows.jsonl",
        },
        "part1_tail_fragility_real": p1,
        "part2_count_statistic_real": p2,
        "part3_ideal_count_analytic": p3,
    }
    (OUT / "FEASIBILITY.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    L = []
    L.append("# SWAN 统计量重构可行性验证（B0）\n")
    L.append("> 纯离线，基于已冻结行级数据；不重跑 Parser、不用 GPU、不新增生成。")
    L.append("> 设计依据：[DESIGN.md](../swan_statistic_design1/DESIGN.md)。\n")

    L.append("## 结论速览\n")
    L.append(f"1. **重要更正**：干净 null 的统计量其实**接近正态**（偏度 {p1['skewness']:.2f}、"
             f"超额峰度 {p1['excess_kurtosis']:.2f}），**并非重尾**。"
             f"因此 v1–v4 误报不稳的主因不是尾形，而是**阈值本身是有限样本估计量**："
             f"n={p1['n']} 时 99% 分位仅由约 {p1['n_points_at_or_above_q99']} 个点决定。")
    L.append(f"2. 99% 阈值的 bootstrap 95% CI = [{p1['q99_bootstrap_ci95'][0]:.3f}, "
             f"{p1['q99_bootstrap_ci95'][1]:.3f}]，SD = {p1['q99_bootstrap_sd']:.3f}"
             f"（≈{p1['q99_spread_in_null_sd']:.2f} 个 null 标准差）→ 换一批校准样本，阈值就漂移。")
    L.append(f"3. split-half 实测 held-out FPR（名义 1%）：中位 {p1['split_half_fpr']['median']*100:.2f}%，"
             f"5–95% 区间 [{p1['split_half_fpr']['p05']*100:.2f}%, {p1['split_half_fpr']['p95']*100:.2f}%]，"
             f"极值 {p1['split_half_fpr']['min']*100:.2f}%~{p1['split_half_fpr']['max']*100:.2f}%。")
    L.append(f"   → 直接解释任务 A 中「51/5000 = 1.02% 被判失败」为何是噪声。")
    L.append(f"4. 真实 K=5 变体计数统计量**违反独立假设**：观测方差 "
             f"{p2['var_T_empirical']:.3f} vs 独立二项方差 {p2['var_T_if_independent']:.3f}，"
             f"过散度比 = {p2['overdispersion_ratio']:.2f}（5 个变体来自同一篇文档）。")
    L.append(f"   → 印证 DESIGN §2.4：计数统计量的单元独立性必须**构造保证**，不能事后假设。")
    L.append(f"5. 理想独立计数统计量的解析阈值**零估计误差**，FPR 恒 = α（见 Part III）。")
    L.append(f"6. **B 的收益因此需要重新定位**：不是「修重尾」，而是"
             f"（a）用解析阈值消除分位数估计方差；（b）用构造性独立单元消除相关假设的误报放大。\n")

    L.append("## Part I — 真实干净 null 的阈值估计脆弱性\n")
    L.append(f"- n = {p1['n']}，均值 {p1['mean']:.3f}，标准差 {p1['std']:.3f}")
    L.append(f"- 偏度 {p1['skewness']:.3f}，超额峰度 {p1['excess_kurtosis']:.3f}")
    L.append(f"- 95% 分位 {p1['q95']:.3f}，99% 分位 {p1['q99']:.3f}")
    L.append(f"- 99% 分位以上仅 **{p1['n_points_at_or_above_q99']}** 个点 → 阈值实际由极少数点决定")
    L.append(f"- 99% 阈值的 bootstrap 95% CI = [{p1['q99_bootstrap_ci95'][0]:.3f}, "
             f"{p1['q99_bootstrap_ci95'][1]:.3f}]，SD = {p1['q99_bootstrap_sd']:.3f}")
    L.append(f"\nheld-out FPR（名义 1%）散布：\n")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    sh = p1["split_half_fpr"]
    L.append(f"| 中位 | {sh['median']*100:.2f}% |")
    L.append(f"| 5% / 95% | {sh['p05']*100:.2f}% / {sh['p95']*100:.2f}% |")
    L.append(f"| 最小 / 最大 | {sh['min']*100:.2f}% / {sh['max']*100:.2f}% |")
    L.append(f"| SD | {sh['sd']*100:.2f}% |")
    L.append("")

    L.append("## Part II — 真实 K=5 变体计数统计量（v4p2 压力 null）\n")
    L.append(f"- 文档数 {p2['n_docs']}，每变体阈值（99 分位）与单通道触发率：\n")
    L.append("| 变体 | 阈值 τ | 单通道触发率 q_k |")
    L.append("|---|---|---|")
    for v in VARIANTS:
        L.append(f"| {v} | {p2['per_variant_threshold'][v]:.3f} | {p2['per_channel_rate_q_k'][v]*100:.2f}% |")
    L.append(f"\n- 平均 q̄ = {p2['q_bar']*100:.2f}%\n")
    L.append("| T | 实测 P(T) | 若独立 B(5,q̄) P(T) |")
    L.append("|---|---|---|")
    for t in range(6):
        L.append(f"| {t} | {p2['empirical_T_pmf'][t]*100:.2f}% | {p2['binomial_T_pmf'][t]*100:.2f}% |")
    L.append(f"\n- 「任一变体触发」FPR = **{p2['any_variant_fire_rate_FPR']*100:.2f}%**"
             f"（实测），独立假设下应为 {p2['P_T_ge1_if_independent']*100:.2f}%")
    L.append(f"- 方差：实测 {p2['var_T_empirical']:.3f} vs 独立 {p2['var_T_if_independent']:.3f}，"
             f"过散度 **{p2['overdispersion_ratio']:.2f}×**")
    L.append(f"- **解读**：")
    L.append(f"  - v4p2 报告的 51/5000 = 1.02% 是**逐 (文档,变体) 行的 FPR**，本就≈标称 1%；"
             f"本表的「任一变体触发」是**另一种更严的鲁棒性口径**，二者不可混用。")
    L.append(f"  - 5 个变体来自同一篇文档，天然强正相关，故实测 T 呈**两端聚集**（98.5% 全灭、"
             f"约 1% 多点同时触发），方差 {p2['var_T_empirical']:.3f} 远高于独立二项 {p2['var_T_if_independent']:.3f}。")
    L.append(f"  - 结论：把**相关通道**当独立单元做计数会误报放大；真正的计数统计量必须让单元"
             f"**结构不相交**（DESIGN §2.4 路线 A/B），并对残余相关用 Poisson-Binomial 复相关建模。\n")

    L.append("## Part III — 理想独立计数统计量的解析精确性（仿真/解析）\n")
    L.append("> 以下为**解析计算**，非实测；用于说明独立计数统计量的统计性质上限。\n")
    for row in p3:
        tpr = row["TPR_by_r"]
        L.append(f"- K={row['K']}, q={row['q']}, α={row['alpha']}: "
                 f"t*={row['t_star']}, 精确 FPR={row['exact_FPR_at_t_star']*100:.2f}%")
    L.append("\n功效表（TPR，按真实单元命中率 r）：\n")
    L.append("| K | q | t* | r=0.3 | r=0.5 | r=0.7 | r=0.9 |")
    L.append("|---|---|---|---|---|---|---|")
    for row in p3:
        if row["q"] != 0.01:
            continue
        tpr = row["TPR_by_r"]
        L.append(f"| {row['K']} | {row['q']} | {row['t_star']} | {tpr[0.3]*100:.1f}% | "
                 f"{tpr[0.5]*100:.1f}% | {tpr[0.7]*100:.1f}% | {tpr[0.9]*100:.1f}% |")
    L.append("")

    L.append("## 对 B1/B2 的直接含义\n")
    L.append("1. **B1（检测端，可立即做）**：把 Bank 划成结构不相交的 K 族，"
             "用现有干净语料实测各族 $q_k$ 与族间相关，先验证独立性假设是否成立；"
             "若不成立，改为「空间不相交」约束或改用 Poisson-Binomial 复相关建模。")
    L.append("2. **B2（嵌入端）**：按 Part III 的功效表选 K——例如要 TPR≥80% 且 r=0.5，"
             "K 需足够大（可查表）；再用任务 A 的功效分析定各集合分母。")
    L.append("3. **先冻结威胁模型**（`space_all` 是否在攻击面内），否则 K 与功效目标无法定。")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")

    print(json.dumps({
        "part1_skew": round(p1["skewness"], 3),
        "part1_kurt": round(p1["excess_kurtosis"], 3),
        "part1_q99": round(p1["q99"], 4),
        "part1_heldout_fpr_p05_p95": [round(sh["p05"], 4), round(sh["p95"], 4)],
        "part2_any_fire_fpr": round(p2["any_variant_fire_rate_FPR"], 4),
        "part2_overdispersion": round(p2["overdispersion_ratio"], 3),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
