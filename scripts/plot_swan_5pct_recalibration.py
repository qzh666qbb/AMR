"""Scientific comparison of original and condition-aware 5% calibration."""
import argparse
import hashlib
import json
from pathlib import Path


def plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    result = json.loads((folder / "RESULTS.json").read_text())
    methods = result["complete_methods"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    x = np.arange(len(methods))
    colors = ("#aeb8c2", "#25688f")
    for ax, label in zip(axes, ("TPR", "FPR")):
        for j, strategy in enumerate(("original_5pct", "condition_guard_5pct")):
            cells = [result["methods"][m][strategy]["conditions"]["pooled"][label]
                     for m in methods]
            values = [100*c["rate"] if c["rate"] is not None else np.nan for c in cells]
            bars = ax.bar(x + (j-.5)*.32, values, .32, color=colors[j],
                          label="Original pooled calibration" if j == 0 else
                          "Maximum of condition thresholds")
            for bar, value, cell in zip(bars, values, cells):
                text = f"{value:.1f}%\n{cell['decisions_positive']}/{cell['denominator']}" \
                    if np.isfinite(value) else "N/A"
                ax.text(bar.get_x()+bar.get_width()/2,
                        max(value, 0)+1 if np.isfinite(value) else 1,
                        text, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, methods)
        ax.set_ylim(0, 70 if label == "TPR" else 24)
        ax.set_ylabel(f"Test {label} (%)")
        ax.spines[["top", "right"]].set_visible(False)
        if label == "FPR":
            ax.axhline(5, color="#9d602a", linestyle="--", linewidth=1)
            ax.text(.99, 5.5, "Nominal 5% target", color="#9d602a", fontsize=8,
                    ha="right", transform=ax.get_yaxis_transform())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("5% calibration: threshold adjustment on completed methods", fontsize=13)
    fig.text(.05, .035, "15 paired sources × 7 correlated conditions; abstentions retained."
             "\nPost hoc development reassessment; no new or independent human test.",
             fontsize=9, color="#48505a")
    fig.subplots_adjust(left=.075, right=.96, top=.79, bottom=.23, wspace=.30)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(folder / f"comparison.{ext}", dpi=180 if ext == "png" else None)
    plt.close(fig)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    (folder / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output)
