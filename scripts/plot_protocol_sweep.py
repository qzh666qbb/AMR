"""从 PROTOCOL_RESULTS.json 生成窗口比较图；不读取文本、不重新评分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/swan_multiview_development1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=EXP / "dataset5_eval/PROTOCOL_RESULTS.json")
    ap.add_argument("--output", type=Path, default=EXP / "figures/fig4_window_sweep")
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    sweep = {row["scope"]: row for row in data["sweep"]}
    names = ("S6", "S8", "S10", "S12")
    if set(sweep) != set(names):
        raise ValueError("需要 S6/S8/S10/S12 全部结果才能生成窗口图")
    if any(sweep[s]["stride"] != 1 for s in names[:3]) or sweep["S12"]["stride"] != 6:
        raise ValueError("图中的步长标注与数据不一致")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.3))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.79, bottom=0.27, wspace=0.33)
    colors = {"merge": "#245E9E", "comma": "#C06432", "space": "#33846F"}
    for ax in axes:
        ax.axvspan(11.2, 12.7, color="#EEF0F3", zorder=0)
        ax.set_xlim(5.5, 12.7)
        ax.set_xticks([6, 8, 10, 12], ["6 / 1", "8 / 1", "10 / 1", "12 / 6"])
        ax.set_xlabel("Window words / stride")
        ax.grid(axis="y", color="#DDE1E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)

    aucs = [sweep[s]["p1_auc"] for s in names]
    axes[0].plot([6, 8, 10], aucs[:3], color="#293A50", marker="o", linewidth=1.8)
    axes[0].scatter([12], [aucs[3]], marker="D", color="#293A50", s=38, zorder=3)
    for x, value in zip([6, 8, 10, 12], aucs):
        axes[0].annotate(f"{value:.4f}", (x, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_ylim(0.82, 0.95)
    axes[0].set_ylabel("Carrier-level AUROC")
    axes[0].set_title("(a) P1: pooled carrier ranking", loc="left", pad=12)

    for ax, point_name, title in [
        (axes[1], "zero", "(b) P2: observed FPR = 0 / 15"),
        (axes[2], "one_negative", "(c) P2: expected FPR = 1 / 15"),
    ]:
        for attack, color in colors.items():
            values = [
                100 * data["P2"][s]["matched_fpr"][attack]["points"][point_name]["tpr"]
                for s in names
            ]
            for s in names:
                row = data["P2"][s]["matched_fpr"][attack]
                point = row["points"][point_name]
                expected = 0.0 if point_name == "zero" else 1 / 15
                if row["n_pos"] != 30 or row["n_neg"] != 15:
                    raise ValueError("图的 30 正例 / 15 负例标注不适用")
                if abs(point["expected_fpr"] - expected) > 1e-10:
                    raise ValueError("预期 FPR 不等于图中工作点")
            ax.plot([6, 8, 10], values[:3], label=attack, color=color,
                    marker="o", linewidth=1.8, markersize=5)
            ax.scatter([12], [values[3]], marker="D", color=color, s=38, zorder=3)
        ax.set_ylim(-4, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_ylabel("Document-level TPR")
        ax.set_title(title, loc="left", pad=12)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.69, 0.94),
               ncol=3, frameon=False, handlelength=2.3, columnspacing=2)
    fig.text(0.06, 0.94, "Window comparison", fontsize=14, weight="bold", color="#293A50")
    fig.text(0.06, 0.15,
             "Lines: stride 1. Diamonds / shaded area: stride 6 reference; the 10-to-12 transition is not a controlled size comparison.",
             fontsize=8.5, color="#4B5563")
    fig.text(0.06, 0.105,
             "Oracle slot aggregation; 128-candidate cap. P2 uses post-hoc test-set ROC (30 positive / 15 negative documents per attack).",
             fontsize=8.5, color="#4B5563")
    fig.text(0.06, 0.06,
             "Panel (c) randomizes acceptance of tied scores analytically to match expected FPR; these are not calibrated deployment results.",
             fontsize=8.5, color="#4B5563")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = args.output.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
