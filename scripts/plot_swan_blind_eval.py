"""Standalone scientific plots of calibrated whole-document endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    result = json.loads((folder / "RESULTS.json").read_text())
    methods = result["complete_methods"]
    attacks = ["baseline", "semicolon", "newline", "split", "merge", "comma", "space"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for target in (1, 5):
        fig, axes = plt.subplots(1, 3, figsize=(13.4, max(3.8, .43 * len(methods) + 1.6)),
                                 gridspec_kw={"width_ratios": [1, 1, 1]})
        labels = ["Test TPR (%)", "Test FPR (%)", "Test unresolved (%)"]
        for index, ax in enumerate(axes):
            cells = np.full((len(methods), len(attacks)), np.nan)
            annotations = {}
            for i, method in enumerate(methods):
                op = result["methods"][method]["operating_points"][f"document_{target}pct"]
                for j, attack in enumerate(attacks):
                    measures = op["conditions"][attack]
                    if index < 2:
                        metric = measures["TPR" if index == 0 else "FPR"]
                        if metric["rate"] is not None:
                            cells[i, j] = 100 * metric["rate"]
                            annotations[i, j] = f"{100 * metric['rate']:.0f}\n" \
                                f"({metric['decisions_positive']}/{metric['denominator']})"
                    else:
                        numerator = measures["TPR"]["unresolved"] + measures["FPR"]["unresolved"]
                        denominator = measures["TPR"]["denominator"] + measures["FPR"]["denominator"]
                        cells[i, j] = 100 * numerator / denominator
                        annotations[i, j] = f"{100 * numerator / denominator:.0f}"
            cmap = plt.get_cmap("Blues" if index == 0 else "Oranges").copy()
            cmap.set_bad("#e9e9e9")
            image = ax.imshow(cells, vmin=0, vmax=100, cmap=cmap, aspect="auto")
            for i in range(len(methods)):
                for j in range(len(attacks)):
                    text = annotations.get((i, j), "N/A")
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                            color="white" if np.isfinite(cells[i, j])
                            and cells[i, j] > 65 else "#20252b")
            ax.set_xticks(range(len(attacks)), attacks, rotation=45, ha="right")
            ax.set_yticks(range(len(methods)), methods if index == 0 else [])
            ax.set_title(labels[index], fontsize=11)
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        color_axis = fig.add_axes([.925, .30, .012, .47])
        fig.colorbar(image, cax=color_axis, label="%")
        suffix = " (partial; execution halted)" if (folder / "HALT.json").exists() else \
            " (partial: remaining methods running)" \
            if len(methods) < len(result["planned_methods"]) else ""
        fig.suptitle(f"Whole-document evaluation: nominal {target}% calibration target{suffix}",
                     fontsize=13, y=.985)
        fig.text(.03, .02, "15 paired source documents per condition; all denominators include "
                 "unresolved documents. Explored generated cohort; no independent human test.",
                 fontsize=9, color="#454b52")
        fig.subplots_adjust(left=.09, right=.91, top=.83, bottom=.25, wspace=.13)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(folder / f"document_{target}pct.{ext}",
                        dpi=180 if ext == "png" else None)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output)
