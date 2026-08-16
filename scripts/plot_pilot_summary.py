"""Create a static, reproducible visual summary of the SWAN pilot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "baselines" / "local" / "swan_acl2026_realnews" / "runs"
OUT = ROOT / "baselines" / "local" / "swan_acl2026_realnews" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2673B8"
ORANGE = "#E28E2C"
GOLD = "#C5A12F"
INK = "#263238"
GRID = "#D9E0E5"


def metadata(run: str):
    with open(RUNS / run / "sampled_data.json", encoding="utf-8") as handle:
        return json.load(handle)


smoke = metadata("api_smoke_10x5")
pilot = metadata("api_pilot_30x5")
with open(RUNS / "formal_250x5" / "generation_checkpoint.json", encoding="utf-8") as handle:
    formal = json.load(handle)

smoke_accept = np.mean([v for row in smoke["accepted"] for v in row])
pilot_accept = np.mean([v for row in pilot["accepted"] for v in row])
formal_values = [v for row in formal["accepted"] for v in row]
formal_accept = np.mean(formal_values) if formal_values else np.nan

detect = RUNS / "api_pilot_30x5" / "detection"
machine = np.load(detect / "machine_z_scores.npy")
human = np.load(detect / "human_z_scores.npy")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle("SWAN + DeepSeek experiment: current evidence", fontsize=17, color=INK, weight="bold")

# A: injection acceptance
ax = axes[0, 0]
labels = ["10-prompt smoke", "30-prompt pilot", "Formal checkpoint*"]
values = [smoke_accept * 100, pilot_accept * 100, formal_accept * 100]
bars = ax.barh(labels, values, color=[BLUE, BLUE, GOLD], edgecolor=INK, linewidth=0.7)
ax.axvline(100, color=INK, linewidth=1, linestyle="--")
ax.set_xlim(0, 105)
ax.set_xlabel("Accepted target sentences (%)")
ax.set_title("A. Injection acceptance\nFormal checkpoint is partial and will change", loc="left", color=INK, weight="bold", fontsize=13)
for bar, value in zip(bars, values):
    ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", color=INK)

# B: detection metrics vs paper
ax = axes[0, 1]
metrics = ["AUC", "TPR @ 1% FPR", "TPR @ 5% FPR"]
paper = np.array([99.1, 91.6, 97.6])
current = np.array([99.8, 86.7, 93.3])
x = np.arange(len(metrics))
width = 0.36
ax.bar(x - width / 2, paper, width, label="Paper no-attack", color="#DCEAF5", edgecolor=BLUE)
ax.bar(x + width / 2, current, width, label="30-prompt pilot", color=BLUE, edgecolor=INK, linewidth=0.7)
ax.set_ylim(75, 102)
ax.set_xticks(x, metrics)
ax.set_ylabel("Percent")
ax.set_title("B. Detection vs paper reference", loc="left", color=INK, weight="bold")
ax.legend(frameon=False, loc="lower left")
for xpos, value in zip(x - width / 2, paper):
    ax.text(xpos, value + 0.6, f"{value:.1f}", ha="center", fontsize=9)
for xpos, value in zip(x + width / 2, current):
    ax.text(xpos, value + 0.6, f"{value:.1f}", ha="center", fontsize=9, weight="bold")

# C: z-score separation
ax = axes[1, 0]
parts = ax.violinplot([human, machine], positions=[1, 2], showmeans=True, showextrema=True)
for body, color in zip(parts["bodies"], [ORANGE, BLUE]):
    body.set_facecolor(color)
    body.set_edgecolor(INK)
    body.set_alpha(0.75)
for key in ("cmeans", "cmins", "cmaxes", "cbars"):
    parts[key].set_color(INK)
ax.axhline(0, color=INK, linewidth=1, linestyle="--")
ax.set_xticks([1, 2], ["Human (n=30)", "Watermarked (n=30)"])
ax.set_ylabel("Paragraph z-score")
ax.set_title("C. Pilot z-score separation", loc="left", color=INK, weight="bold")
ax.text(1, np.mean(human), f" mean {np.mean(human):.2f}", va="bottom", ha="center")
ax.text(2, np.mean(machine), f" mean {np.mean(machine):.2f}", va="bottom", ha="center")

# D: formal progress
ax = axes[1, 1]
completed = len(formal["texts"])
pending = 250 - completed
ax.barh(["Formal baseline"], [completed], color=BLUE, edgecolor=INK, label="Completed paragraphs")
ax.barh(["Formal baseline"], [pending], left=[completed], color="#E9EEF2", edgecolor=INK, label="Pending")
ax.set_xlim(0, 250)
ax.set_xlabel("Paragraphs")
ax.set_title(
    f"D. Formal progress ({completed}/250)\nSentence acceptance: {formal_accept * 100:.1f}% ({sum(formal_values)}/{len(formal_values)})",
    loc="left", color=INK, weight="bold", fontsize=13,
)
ax.text(completed + 3, 0, f"{completed}/250", va="center", color=INK, weight="bold")
ax.legend(frameon=False, loc="lower right")

for ax in axes.flat:
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.text(
    0.01,
    0.005,
    "Pilot metrics are preliminary. Formal checkpoint values may change as the 250-prompt run progresses.",
    fontsize=9,
    color=INK,
)
fig.savefig(OUT / "pilot_summary.png", dpi=180, facecolor="white")
fig.savefig(OUT / "pilot_summary.svg", facecolor="white")
print(OUT / "pilot_summary.png")
