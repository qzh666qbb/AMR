"""Build figures needed for the paper draft from existing experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _style(ax):
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _box(ax, xy, text, width=0.22, height=0.12, fc="#f8fafc", ec="#475569"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)


def build_pipeline_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, (0.03, 0.4), "Source paragraph")
    _box(ax, (0.28, 0.4), "K candidate rewrites")
    _box(ax, (0.53, 0.4), "Quality filter\n(surface + semantics)")
    _box(ax, (0.78, 0.4), "SWAN / S2Match\nscore z")
    ax.annotate("", xy=(0.28, 0.46), xytext=(0.25, 0.46), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.53, 0.46), xytext=(0.50, 0.46), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.78, 0.46), xytext=(0.75, 0.46), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(
        0.50,
        0.16,
        r"Strict success if $q_i = 1$ and $z_i \leq \tau$; report Strict Valid ASR over all paragraphs.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(0.50, 0.88, "Paper pipeline", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_frontier_figure(out: Path, root: Path) -> None:
    baseline = json.loads((root / "experiments" / "baseline_comparison" / "baseline_comparison.json").read_text(encoding="utf-8"))
    strict = json.loads((root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "paper_final_bundle.json").read_text(encoding="utf-8"))
    frontier = json.loads((root / "experiments" / "best_observed_frontier" / "best_observed_frontier.json").read_text(encoding="utf-8"))

    labels = [
        "Baseline pilot\n30x5",
        "Baseline formal\n250x5",
        "Proposed stable\nstrict",
        "Best observed\nstrict (K=1)",
        "Best raw\nescape",
    ]
    values = [
        baseline["rows"][0]["escape_rate"],
        baseline["rows"][1]["escape_rate"],
        strict["strict_valid_asr"],
        frontier["best_strict_valid_asr_observed"]["valid_asr"],
        frontier["best_raw_escape_prompt_component"]["escape_rate"],
    ]
    colors = ["#94a3b8", "#64748b", "#7c3aed", "#2563eb", "#16a34a"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Observed rate")
    ax.set_ylim(0, 1.05)
    _style(ax)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Frontier of observed success rates")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_robustness_figure(out: Path, root: Path) -> None:
    threshold = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "threshold_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    repeat = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "full_repeatability_50.json").read_text(
            encoding="utf-8"
        )
    )
    xs = [row["threshold"] for row in threshold]
    strict_asr = [row["strict_valid_asr"] for row in threshold]
    candidate_escape = [row["candidate_escape_rate"] for row in threshold]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(xs, strict_asr, "o-", label="Strict Valid ASR")
    axes[0].plot(xs, candidate_escape, "o-", label="Candidate escape rate")
    axes[0].axvspan(0.65, 0.80, color="#c4b5fd", alpha=0.25, label="Plateau")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0])

    labels = ["Stable candidates", "Mixed flips", "Mean paragraph ASR"]
    values = [
        repeat["stable_candidates"] / repeat["total_candidates"],
        repeat["candidate_with_mixed_escape"] / repeat["total_candidates"],
        repeat["mean_paragraph_asr"],
    ]
    bars = axes[1].bar(labels, values, color=["#2563eb", "#ef4444", "#16a34a"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    _style(axes[1])
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("50-repeat stability")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_ablation_failure_figure(out: Path, root: Path) -> None:
    prompt = json.loads((root / "experiments" / "prompt_component_ablation30" / "analysis.json").read_text(encoding="utf-8"))
    methods = ["weak", "no_boundary", "boundary_only", "full_e2"]
    display = ["Weak", "No boundary", "Boundary only", "Full E2"]
    escape_rates = [prompt["summary"][m]["escape_rate"] for m in methods]
    mean_drops = [prompt["summary"][m]["mean_z_drop"] for m in methods]

    quality_summary = {
        "Plain paraphrase": {"mean_quality": 4.43, "quality_ge4": 0.794, "major_error": 0.0},
        "Sentence boundary": {"mean_quality": 4.24, "quality_ge4": 0.727, "major_error": 0.030},
        "AMR guided": {"mean_quality": 3.63, "quality_ge4": 0.576, "major_error": 0.091},
    }
    q_labels = list(quality_summary.keys())
    q_quality = [quality_summary[k]["mean_quality"] for k in q_labels]
    q_major = [quality_summary[k]["major_error"] for k in q_labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(display))
    axes[0].bar(x, escape_rates, color=["#94a3b8", "#64748b", "#16a34a", "#7c3aed"])
    axes[0].set_xticks(x, display, rotation=15)
    axes[0].set_ylabel("Escape rate")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Prompt-component ablation")
    _style(axes[0])
    ax2 = axes[0].twinx()
    ax2.plot(x, mean_drops, "o--", color="#dc2626", linewidth=1.5)
    ax2.set_ylabel("Mean z drop")
    ax2.grid(False)

    x2 = np.arange(len(q_labels))
    axes[1].bar(x2 - 0.15, q_quality, width=0.3, label="Mean quality", color="#2563eb")
    axes[1].bar(x2 + 0.15, q_major, width=0.3, label="Major-error rate", color="#ef4444")
    axes[1].set_xticks(x2, q_labels, rotation=12)
    axes[1].set_ylim(0, 5.0)
    axes[1].set_title("Quality vs failure")
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    build_pipeline_figure(out_dir / "fig1_pipeline.png")
    build_frontier_figure(out_dir / "fig2_frontier.png", root)
    build_robustness_figure(out_dir / "fig3_robustness.png", root)
    build_ablation_failure_figure(out_dir / "fig4_ablation_failure.png", root)


if __name__ == "__main__":
    main()
