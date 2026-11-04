"""Plot augmented-negative FPR and reused-positive TPR with explicit scope."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/swan_multiview_development1/newline_negative_expansion1"


def plot(folder):
    folder = Path(folder)
    result = json.loads((folder / "RESULTS.json").read_text())
    old = result["historical_newline_control"]
    fixed = result["fixed_threshold"]
    expanded = result["expanded_calibration_threshold"]
    labels = ["Historical n = 15\noriginal threshold",
              "New n = 100\noriginal threshold",
              "New n = 100\nrecalibrated",
              "Historical n = 15\nrecalibrated"]
    fpr = [old["FPR"], fixed["new_test_FPR"], expanded["new_test_FPR"]]
    supplement = folder / "COMPLETION_SUPPLEMENT.json"
    if supplement.exists():
        extra = json.loads(supplement.read_text())
        fpr.append(extra["historical_negative_FPR_at_expanded_threshold"])
    else:
        labels = labels[:3]
    tpr = [fixed["historical_TPR"], expanded["historical_TPR"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#0f172a",
        "xtick.color": "#334155", "ytick.color": "#334155",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5), layout="constrained",
                             gridspec_kw={"width_ratios": [1.5, 1]})
    colors = ["#94a3b8", "#2563eb", "#0f766e", "#b45309"][:len(fpr)]
    for ax, values, panel_labels, panel_colors, title, ylabel in (
        (axes[0], fpr, labels, colors, "False positive rate", "FPR (%)"),
        (axes[1], tpr, ["Original threshold", "Recalibrated threshold"],
         ["#2563eb", "#0f766e"], "Historical positive control", "TPR (%)"),
    ):
        positions = range(len(values))
        rates = np.array([v["rate"] * 100 for v in values])
        bounds = np.array([v["binomial_interval_95"] for v in values]) * 100
        ax.bar(positions, rates, color=panel_colors, width=.55, zorder=3)
        ax.errorbar(positions, rates, yerr=np.vstack([rates - bounds[:, 0], bounds[:, 1] - rates]),
                    fmt="none", color="#334155", capsize=5, linewidth=1.2, zorder=4)
        for i, v in enumerate(values):
            ax.text(i, max(rates[i] + 1, bounds[i, 1] + 1),
                    f"{v['decisions_positive']}/{v['denominator']}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(positions, panel_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e2e8f0", zorder=0)
        ax.set_ylim(0, min(100, max(20, bounds[:, 1].max() + 7)))
    axes[0].axhline(5, color="#b45309", linewidth=1.2, linestyle="--")
    axes[0].text(.48, 5.8, "5% target", color="#b45309", ha="center", fontsize=9,
                 transform=axes[0].get_yaxis_transform(), zorder=5,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    fig.suptitle(
        "S6 newline: 100 new calibration + 100 new test negatives",
        fontsize=14, fontweight="bold", ha="left", x=.02)
    fig.supxlabel(
        "Generated negative cohorts differ and are reported separately. Historical positives: 5 complete, 10 abstentions.\n"
        "New test negatives: 59 complete, 41 abstentions. Whiskers: two-sided 95% Clopper–Pearson intervals.",
        fontsize=9, color="#475569")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / ("FPR_TPR_comparison." + extension), dpi=200, bbox_inches="tight")
    plt.close(fig)
    hashes = json.loads((folder / "SHA256SUMS.json").read_text())
    for extension in ("png", "pdf", "svg"):
        p = folder / ("FPR_TPR_comparison." + extension)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (folder / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=DEFAULT)
    plot(cli.parse_args().output)
