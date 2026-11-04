"""按 PROTOCOL.md 生成 P0–P3 与窗口比较的 PROTOCOL_RESULTS.json。

输入（全部为已有实验产物）：
- S12 / Ssub：dataset5_eval/FUSION_RECOVERY.json 与 dataset5_paired_neg_eval/FUSION_RECOVERY.json
- S6：dataset5_eval/window6s1_pos/ 与 dataset5_paired_neg_eval/window6s1_neg/
- S8 / S10：window8s1_* / window10s1_* 分片目录
- P0：dataset5_eval/sentence_pos/ 与 dataset5_paired_neg_eval/sentence_neg/（若存在）

输出：
- P1 单元级：阈值为校准负例 1% 分位；报告测试 FPR、TPR@1/5、AUC
- P2：保留校准端点；另报测试集 ROC 的同预期 FPR 操作点与可达确定性端点
- P3 目标模板诊断：每模板 1% 阈值；AUC、TPR@1%
- P0 官方协议：0.65 green 率、文档 z、经验 5% 阈值
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ed_ablation import auc, tpr  # noqa: E402

EXP = ROOT / "experiments/swan_multiview_development1"
POS12 = EXP / "dataset5_eval/FUSION_RECOVERY.json"
NEG12 = EXP / "dataset5_paired_neg_eval/FUSION_RECOVERY.json"
S6_POS = EXP / "dataset5_eval/window6s1_pos"
S6_NEG = EXP / "dataset5_paired_neg_eval/window6s1_neg"
S8_POS = EXP / "dataset5_eval/window8s1_pos"
S8_NEG = EXP / "dataset5_paired_neg_eval/window8s1_neg"
S10_POS = EXP / "dataset5_eval/window10s1_pos"
S10_NEG = EXP / "dataset5_paired_neg_eval/window10s1_neg"
P0_POS = EXP / "dataset5_eval/sentence_pos"
P0_NEG = EXP / "dataset5_paired_neg_eval/sentence_neg"
OUT = EXP / "dataset5_eval/PROTOCOL_RESULTS.json"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")
INTACT = ("baseline", "semicolon", "newline", "split")


def load(path: Path):
    if path.is_dir():
        rows = []
        files = sorted(path.glob("s*/FUSION_RECOVERY.json"))
        if not files:
            raise ValueError(f"没有完整结果分片：{path}")
        for p in files:
            rows.extend(json.loads(p.read_text())["per_instance"])
    else:
        rows = json.loads(path.read_text())["per_instance"]
    keys = [(r["doc"], r["attack"], r["slot"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"重复的文档/攻击/槽位：{path}")
    return rows


def audit_candidates(rows, dataset, words, stride):
    """用原攻击函数重建候选数；不加载模型、不重算 detector 分数。

    历史产物没有命令元数据，因此这里核验配置兼容性，并记录候选截断和
    零覆盖槽位。函数定义从本仓库评分入口所用的同一源文件读取。
    """
    source = ROOT / "scripts/run_dataset4_attack_recovery.py"
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and (n.name.startswith("attack_") or n.name.startswith("_span_map")
                      or n.name == "_annotate_merge_spans")]
    namespace = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
         namespace)
    from src.core.multiview_candidates import WindowConfig, candidate_windows

    docs = json.loads(dataset.read_text())["documents"]
    actual = {(r["doc"], r["attack"], r["slot"]): r for r in rows}
    expected, uncovered = {}, defaultdict(int)
    truncated, groups_truncated = 0, 0
    for doc in docs:
        for attack in ATTACKS:
            text, spans = namespace[f"attack_{attack}"](doc["slots"])
            if attack == "merge":
                namespace["_annotate_merge_spans"](spans)
            candidates = candidate_windows(
                text, WindowConfig(lengths=(words,), stride=stride,
                                   max_words=24, max_candidates=128))
            truncated += candidates["truncated_count"]
            groups_truncated += candidates["truncated_count"] > 0
            for slot, data in enumerate(doc["slots"]):
                if attack == "merge":
                    span = next((sp["slot_a"] if sp.get("_slot_a_idx") == slot
                                 else sp["slot_b"] for sp in spans
                                 if sp.get("_slot_a_idx") == slot
                                 or sp.get("_slot_b_idx") == slot), None)
                else:
                    span = spans[slot]
                count = sum(span[0] < c["end_char"] and c["start_char"] < span[1]
                            for c in candidates["candidates"]) if span else 0
                key = (doc["document_id"], attack, slot)
                if count:
                    expected[key] = (count, data["template_index"])
                else:
                    uncovered[attack] += 1
    mismatch = sum(k not in expected
                   or r["n_candidates"] != expected[k][0]
                   or r["template_index"] != expected[k][1]
                   for k, r in actual.items())
    missing = set(expected) - set(actual)
    if mismatch or missing:
        raise ValueError(f"候选配置/产物不一致：{dataset.parent.name} "
                         f"{words}w/{stride}, mismatch={mismatch}, missing={len(missing)}")
    return {"configuration_check": "candidate_counts_match",
            "original_command_metadata": "not_recorded",
            "expected_slots": sum(len(d["slots"]) for d in docs) * len(ATTACKS),
            "scored_slots": len(rows), "uncovered_slots": sum(uncovered.values()),
            "uncovered_by_attack": dict(uncovered),
            "truncated_candidates": truncated,
            "doc_attack_groups_with_truncation": groups_truncated,
            "candidate_count_mismatches": mismatch, "unexpected_missing_rows": len(missing)}


def max_of(vec):
    vals = [v for v in (vec or []) if v is not None]
    return max(vals) if vals else None


def split_docs(neg):
    docs = sorted({r["doc"] for r in neg})
    half = len(docs) // 2
    cal, test = set(docs[:half]), set(docs[half:])
    return cal, test


def quantile_threshold(vals, q):
    """返回使 q 比例超过的阈值（高分侧）。"""
    vals = sorted([v for v in vals if v is not None], reverse=True)
    if not vals:
        return None
    k = max(1, math.ceil(q * len(vals)))
    return vals[min(k, len(vals)) - 1]


def unit_table(pos, neg, view, cal_docs, test_docs):
    p = [max_of(r.get(view)) for r in pos]
    n_cal = [max_of(r.get(view)) for r in neg if r["doc"] in cal_docs]
    n_test = [max_of(r.get(view)) for r in neg if r["doc"] in test_docs]
    p = [v for v in p if v is not None]
    n_cal = [v for v in n_cal if v is not None]
    n_test = [v for v in n_test if v is not None]
    tau = quantile_threshold(n_cal, 0.01)
    tau5 = quantile_threshold(n_test, 0.05)
    out = {"tau": None if tau is None else round(tau, 4),
           "n_pos": len(p), "n_neg_cal": len(n_cal), "n_neg_test": len(n_test),
           "auc": round(float(auc(p, n_test)), 4),
           "fpr_test": None if tau is None else round(sum(v >= tau for v in n_test) / len(n_test), 4),
           "tpr_at_cal_tau": None if tau is None else round(sum(v >= tau for v in p) / len(p), 4),
           "tpr5": round(float(tpr(p, n_test, 0.05)), 4),
           "tpr5_realized_fpr": round(sum(v >= tau5 for v in n_test) / len(n_test), 4)}
    return out


def doc_table(pos, neg, view, cal_docs, test_docs, tau):
    def group(rows):
        g = defaultdict(list)
        for r in rows:
            v = max_of(r.get(view))
            if v is not None:
                g[(r["doc"], r["attack"])].append(v)
        return g
    pg, ng = group(pos), group(neg)
    p_r = {k: sum(v >= tau for v in vs) / len(vs) for k, vs in pg.items()}
    n_cal = {k: sum(v >= tau for v in vs) / len(vs)
             for k, vs in ng.items() if k[0] in cal_docs}
    n_test = {k: sum(v >= tau for v in vs) / len(vs)
              for k, vs in ng.items() if k[0] in test_docs}
    tau_r = st.quantiles(list(n_cal.values()), n=20)[-1] if len(n_cal) >= 20 else max(n_cal.values())
    out = {}
    for a in ATTACKS:
        pr = [v for k, v in p_r.items() if k[1] == a]
        nr = [v for k, v in n_test.items() if k[1] == a]
        if not pr or not nr:
            continue
        out[a] = {"n_pos": len(pr), "n_neg": len(nr),
                  "threshold_r": round(tau_r, 4),
                  "fpr_test": round(sum(v >= tau_r for v in nr) / len(nr), 4),
                  "tpr": round(sum(v >= tau_r for v in pr) / len(pr), 4),
                  "mean_r_pos": round(float(np.mean(pr)), 4),
                  "mean_r_neg": round(float(np.mean(nr)), 4)}
    return out, round(tau_r, 4)


def roc_point(pos, neg, target):
    """经验 ROC 操作点；同分随机接受概率仅用于计算预期 FPR/TPR。

    确定性规则为 score > threshold。额外返回 score == threshold 的接受
    概率，使预期 FPR 精确等于目标；不能解释为可部署的校准阈值。
    """
    if not pos or not neg or not 0 <= target < 1:
        raise ValueError("ROC 操作点需要非空正负例及 [0, 1) 内目标 FPR")
    ordered = sorted(neg, reverse=True)
    index = min(math.floor(target * len(neg) + 1e-12), len(neg) - 1)
    cut = ordered[index]
    above_n, equal_n = sum(v > cut for v in neg), sum(v == cut for v in neg)
    above_p, equal_p = sum(v > cut for v in pos), sum(v == cut for v in pos)
    alpha = min(1.0, max(0.0, (target * len(neg) - above_n) / equal_n))
    return {
        "target_fpr": target, "threshold": cut, "tie_accept_probability": alpha,
        "expected_fpr": (above_n + alpha * equal_n) / len(neg),
        "tpr": (above_p + alpha * equal_p) / len(pos),
        "deterministic_fpr": above_n / len(neg),
        "deterministic_tpr": above_p / len(pos),
        "negative_above": above_n, "negative_at_threshold": equal_n,
        "positive_above": above_p, "positive_at_threshold": equal_p,
    }


def matched_doc_table(pos, neg, view, tau, test_docs, targets=(0.0, 1 / 15)):
    """逐攻击的事后 ROC 比较；与校准后独立测试端点分别存储。"""
    def group(rows):
        g = defaultdict(list)
        for r in rows:
            v = max_of(r.get(view))
            if v is not None:
                g[(r["doc"], r["attack"])].append(v)
        return {k: sum(x >= tau for x in vs) / len(vs) for k, vs in g.items()}

    pg, ng = group(pos), group(neg)
    out = {}
    for a in ATTACKS:
        pr = [v for k, v in pg.items() if k[1] == a]
        nr = [v for k, v in ng.items() if k[0] in test_docs and k[1] == a]
        if not pr or not nr:
            continue
        row = {"n_pos": len(pr), "n_neg": len(nr),
               "auc": round(float(auc(pr, nr)), 4), "points": {}}
        for t in targets:
            point = roc_point(pr, nr, t)
            tag = round(t * 100)
            row["points"]["zero" if t == 0 else "one_negative"] = point
            row[f"fpr_{tag}"] = round(point["expected_fpr"], 4)
            row[f"tpr_at_fpr_{tag}"] = round(point["tpr"], 4)
            row[f"deterministic_fpr_{tag}"] = round(point["deterministic_fpr"], 4)
            row[f"deterministic_tpr_{tag}"] = round(point["deterministic_tpr"], 4)
        out[a] = row
    return out


def target_table(pos, neg, view, cal_docs, test_docs):
    cal = [r for r in neg if r["doc"] in cal_docs]
    test = [r for r in neg if r["doc"] in test_docs]

    def per_tpl(rows):
        d = defaultdict(list)
        for r in rows:
            t, v = r.get("template_index"), (r.get(view) or [None])
            if t is not None and t < len(v) and v[t] is not None:
                d[t].append(v[t])
        return {t: sorted(vs, reverse=True) for t, vs in d.items()}

    thr = {t: vs[max(0, math.ceil(0.01 * len(vs)) - 1)]
           for t, vs in per_tpl(cal).items()}
    p, n = [], []
    for r in pos:
        t, v = r.get("template_index"), (r.get(view) or [None])
        if t is not None and t < len(v) and v[t] is not None:
            p.append(v[t])
    for r in test:
        t, v = r.get("template_index"), (r.get(view) or [None])
        if t is not None and t < len(v) and v[t] is not None:
            n.append(v[t])
    hit = 0
    for r in pos:
        t, v = r.get("template_index"), (r.get(view) or [None])
        if t in thr and t < len(v) and v[t] is not None and v[t] >= thr[t]:
            hit += 1
    return {"auc": round(float(auc(p, n)), 4),
            "tpr_at_tpl_1pct": round(hit / len(p), 4) if p else None,
            "n_pos": len(p), "n_neg": len(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--require-sweep", action="store_true",
                    help="缺少 S8/S10 或分片不完整时失败，禁止输出不完整曲线")
    args = ap.parse_args()

    pos12, neg12 = load(POS12), load(NEG12)
    cal, test = split_docs(neg12)
    s6_pos, s6_neg = load(S6_POS), load(S6_NEG)
    # S6 负例的分片划分与主负例一致
    cal6 = {r["doc"] for r in s6_neg if r["doc"] in cal}
    test6 = {r["doc"] for r in s6_neg if r["doc"] in test}

    scopes = [
        ("S12", pos12, neg12, "vec_d1", cal, test),
        ("S6", s6_pos, s6_neg, "vec_d1", cal6, test6),
    ]
    for name, pdir, ndir in (("S8", S8_POS, S8_NEG), ("S10", S10_POS, S10_NEG)):
        if pdir.is_dir() and list(pdir.glob("s*/FUSION_RECOVERY.json")):
            p, n = load(pdir), load(ndir)
            scopes.append((name, p, n, "vec_d1",
                           {r["doc"] for r in n if r["doc"] in cal},
                           {r["doc"] for r in n if r["doc"] in test}))
    scopes.append(("Ssub", pos12, neg12, "vec_d2", cal, test))

    if args.require_sweep and not {"S6", "S8", "S10", "S12"} <= {s[0] for s in scopes}:
        raise ValueError("窗口比较缺少 S8/S10 完整结果")
    result = {"protocol": "P0-P3 v1.1 post-hoc ROC comparison",
              "cal_docs": len(cal), "test_docs": len(test), "P1": {}, "P2": {}, "P3": {},
              "notes": [
                  "所有窗口分数按已知载体槽位聚合，属于 oracle 槽位诊断；不是盲文档检测。",
                  "P1/P3 与 P2 使用全部 30 篇正例；负例校准/测试按源编号分为 15/15。",
                  "P2 matched_fpr 阈值使用测试负例，只是事后 ROC；calibration 端点保留在 per_attack。",
                  "同分插值给出同预期 FPR；同时报告可达确定性 FPR/TPR，未部署随机判定。",
                  "P1 tpr5 沿用历史分位加 >= 规则；并列分数可能令实际 FPR 超过 5%。",
                  "S12 为 12w/6 对照；S6/S8/S10 为步长 1，不能视为纯窗口大小四点消融。",
              ], "candidate_audit": {}, "source_sha256": {}}
    for name, p, n, _, _, _ in scopes:
        if name == "Ssub":
            continue
        words, stride = {"S6": (6, 1), "S8": (8, 1), "S10": (10, 1), "S12": (12, 6)}[name]
        if args.require_sweep:
            if {r["doc"] for r in p} != cal | test or {r["doc"] for r in n} != cal | test:
                raise ValueError(f"{name} 分片未覆盖全部 30 篇文档")
            result["candidate_audit"][name] = {
                "window_words": words, "stride": stride, "max_candidates": 128,
                "positive": audit_candidates(p, EXP / "dataset5/documents.json", words, stride),
                "negative": audit_candidates(n, EXP / "dataset5_paired_neg/documents.json",
                                             words, stride),
            }
    paths = [POS12, NEG12, EXP / "dataset5/documents.json",
             EXP / "dataset5_paired_neg/documents.json",
             ROOT / "src/core/multiview_candidates.py",
             ROOT / "scripts/run_dataset4_attack_recovery.py",
             ROOT / "scripts/plot_protocol_sweep.py", EXP / "PROTOCOL.md", Path(__file__)]
    for directory in (S6_POS, S6_NEG, S8_POS, S8_NEG, S10_POS, S10_NEG, P0_POS, P0_NEG):
        paths.extend(sorted(directory.glob("s*/FUSION_RECOVERY.json")))
    for path in paths:
        result["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # ── P1 单元级 ──
    for name, p, n, view, cd, td in scopes:
        allp = [max_of(r.get(view)) for r in p]
        alln = [max_of(r.get(view)) for r in n if r["doc"] in td]
        allp = [v for v in allp if v is not None]
        alln = [v for v in alln if v is not None]
        tau = quantile_threshold([max_of(r.get(view)) for r in n if r["doc"] in cd], 0.01)
        result["P1"][name] = {
            "overall": unit_table(p, n, view, cd, td),
            "per_attack": {},
        }
        for a in ATTACKS:
            pa = [r for r in p if r["attack"] == a]
            na = [r for r in n if r["attack"] == a]
            result["P1"][name]["per_attack"][a] = unit_table(pa, na, view, cd, td)

    # ── P2 文档级 ──
    for name, p, n, view, cd, td in scopes:
        tau = quantile_threshold([max_of(r.get(view)) for r in n if r["doc"] in cd], 0.01)
        table, tau_r = doc_table(p, n, view, cd, td, tau)
        result["P2"][name] = {"tau_unit": tau, "tau_doc": tau_r, "per_attack": table,
                              "matched_fpr": matched_doc_table(p, n, view, tau, td)}

    # ── P3 目标模板诊断 ──
    for name, p, n, view, cd, td in scopes:
        result["P3"][name] = target_table(p, n, view, cd, td)

    # ── 窗口尺寸曲线（6/8/10/12 词） ──
    # 注意：S6/S8/S10 的步长为 1；S12 是主实验配置（12 词、步长 6），不是同一
    # 步长下的第四点，列在这里仅作参照，曲线解读时须标注这一步长差异。
    SWEEP = (("S6", 6, 1), ("S8", 8, 1), ("S10", 10, 1), ("S12", 12, 6))
    result["sweep"] = []
    for name, words, stride in SWEEP:
        if name not in result["P1"]:
            continue
        o1, o3 = result["P1"][name]["overall"], result["P3"][name]
        m = result["P2"][name].get("matched_fpr", {})
        result["sweep"].append({
            "scope": name, "window_words": words, "stride": stride,
            "p1_auc": o1["auc"], "p1_tau": o1["tau"],
            "p1_tpr_at_cal_tau": o1["tpr_at_cal_tau"], "p1_fpr_test": o1["fpr_test"],
            "p1_tpr5": o1["tpr5"],
            "p3_auc": o3["auc"], "p3_tpr_at_tpl_1pct": o3["tpr_at_tpl_1pct"],
            "p2_tpr_at_fpr0": {a: v.get("tpr_at_fpr_0") for a, v in m.items()},
            "p2_tpr_at_fpr7": {a: v.get("tpr_at_fpr_7") for a, v in m.items()},
        })

    # ── P0 官方协议（若句子级结果存在） ──
    result["P0"] = {"status": "missing"}
    if P0_POS.is_dir() and P0_NEG.is_dir() and list(P0_POS.glob("s*/FUSION_RECOVERY.json")):
        p0p, p0n = load(P0_POS), load(P0_NEG)
        cal0 = {r["doc"] for r in p0n if r["doc"] in cal}
        test0 = {r["doc"] for r in p0n if r["doc"] in test}
        result["P0"] = {"green_rate_065": {}, "doc": {}}
        for a in ATTACKS:
            vals = [max_of(r.get("vec_d1")) for r in p0n if r["attack"] == a]
            vals = [v for v in vals if v is not None]
            result["P0"]["green_rate_065"][a] = round(sum(v >= 0.65 for v in vals) / len(vals), 4) if vals else None
        # 文档 z（λ=0.25）与经验 5% 阈值
        def dz(rows, a):
            g = defaultdict(list)
            for r in rows:
                if r["attack"] != a:
                    continue
                v = max_of(r.get("vec_d1"))
                if v is not None:
                    g[r["doc"]].append(v)
            return {d: (sum(x >= 0.65 for x in vs), len(vs)) for d, vs in g.items()}
        for a in INTACT:
            pz, nz = dz(p0p, a), dz(p0n, a)
            pzs = [(g - 0.25 * n) / math.sqrt(n * 0.25 * 0.75) for g, n in pz.values() if n >= 2]
            nzs = [(g - 0.25 * n) / math.sqrt(n * 0.25 * 0.75) for g, n in nz.values() if n >= 2]
            if not pzs or not nzs:
                continue
            tau_z = sorted(nzs, reverse=True)[max(0, math.ceil(0.05 * len(nzs)) - 1)]
            result["P0"]["doc"][a] = {
                "n_pos": len(pzs), "n_neg": len(nzs),
                "auc": round(float(auc(pzs, nzs)), 4),
                "fpr_test": round(sum(v >= tau_z for v in nzs) / len(nzs), 4),
                "tpr": round(sum(v >= tau_z for v in pzs) / len(pzs), 4),
                "mean_z_pos": round(float(np.mean(pzs)), 3),
                "mean_z_neg": round(float(np.mean(nzs)), 3),
            }
        for a in ("merge", "comma", "space"):
            result["P0"].setdefault("undefined", {})[a] = "no sentence boundary; P0 N/A"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # ── 打印 ──
    names = [s[0] for s in scopes]

    print("=== P1 单元级（阈值=校准负例 1% 分位）===")
    for s in names:
        o = result["P1"][s]["overall"]
        print(f"{s}: tau={o['tau']} AUC={o['auc']} TPR={o['tpr_at_cal_tau']:.1%} "
              f"testFPR={o['fpr_test']:.1%} "
              f"历史5%分位TPR={o['tpr5']:.1%}(实测FPR={o['tpr5_realized_fpr']:.1%})")

    print("\n=== P2 事后 ROC（预期 FPR：0 / 1/15；非部署阈值）===")
    for s in names:
        m = result["P2"][s]["matched_fpr"]
        print(f"{s}: " + " ".join(
            f"{a}={v['tpr_at_fpr_0']:.0%}/{v['tpr_at_fpr_7']:.0%}"
            f"(确定性FPR={v['deterministic_fpr_7']:.1%})"
            for a, v in m.items()))

    print("\n=== P3 目标模板诊断 ===")
    for s in names:
        o = result["P3"][s]
        print(f"{s}: AUC={o['auc']} 校准目标1%下TPR={o['tpr_at_tpl_1pct']:.1%}")

    print("\n=== 窗口尺寸曲线 ===")
    for r in result["sweep"]:
        print(f"{r['scope']}({r['window_words']}w/{r['stride']}): "
              f"AUC={r['p1_auc']} 历史5%分位TPR={r['p1_tpr5']:.1%} "
              f"P3={r['p3_auc']}/{r['p3_tpr_at_tpl_1pct']:.1%} "
              f"merge@{r['p2_tpr_at_fpr7'].get('merge')}")
    print("\n=== P0 官方 ===")
    print(json.dumps(result["P0"], ensure_ascii=False, indent=1)[:1200])
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
