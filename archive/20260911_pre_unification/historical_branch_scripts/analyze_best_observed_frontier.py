"""Summarize the strongest observed results across completed experiments.

This is a consolidation pass over existing outputs. It ranks:
1) strict-valid results from the candidate-budget audit;
2) stability-backed strict results from the feedback-driven audit;
3) raw detector escape results from the prompt-component ablation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "best_observed_frontier"
    out_dir.mkdir(parents=True, exist_ok=True)

    judged = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "judged_analysis.json").read_text(encoding="utf-8")
    )
    strict = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "paper_final_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    prompt = json.loads(
        (root / "experiments" / "prompt_component_ablation30" / "analysis.json").read_text(encoding="utf-8")
    )
    baseline = json.loads((root / "experiments" / "baseline_comparison" / "baseline_comparison.json").read_text(encoding="utf-8"))

    judged_rows = judged["summary"]
    judged_best_k = max(judged_rows.items(), key=lambda item: item[1]["valid_asr"])
    judged_best_k_name, judged_best_k_stats = judged_best_k

    prompt_rows = prompt["summary"]
    prompt_best_escape = max(prompt_rows.items(), key=lambda item: item[1]["escape_rate"])
    prompt_best_escape_name, prompt_best_escape_stats = prompt_best_escape
    prompt_best_quality_escape = max(prompt_rows.items(), key=lambda item: item[1]["mean_z_drop"])
    prompt_best_quality_name, prompt_best_quality_stats = prompt_best_quality_escape

    baseline_rows = baseline["rows"]
    baseline_best_escape = max((row for row in baseline_rows if row["method"] != "proposed"), key=lambda row: row["escape_rate"])

    summary = {
        "best_strict_valid_asr_observed": {
            "source": "candidate_budget16_pilot30/judged_analysis",
            "k": int(judged_best_k_name),
            "valid_asr": judged_best_k_stats["valid_asr"],
            "escape_rate": judged_best_k_stats["escape_rate"],
            "surface_valid_rate": judged_best_k_stats["surface_valid_rate"],
            "semantic_valid_rate": judged_best_k_stats["semantic_valid_rate"],
        },
        "best_stability_backed_strict_result": {
            "source": "feedback_driven_strict30/paper_final_bundle",
            "strict_valid_asr": strict["strict_valid_asr"],
            "strict_valid_asr_ci95": strict["strict_valid_asr_ci95"],
            "threshold_plateau": strict["threshold_plateau"],
            "repeatability_50": {
                "repeats": strict["repeatability_50"]["repeats"],
                "selected_paragraphs": strict["repeatability_50"]["selected_paragraphs"],
                "mean_paragraph_asr": strict["repeatability_50"]["mean_paragraph_asr"],
            },
            "full_repeatability_50": {
                "total_candidates": strict["full_repeatability_50"]["total_candidates"],
                "candidate_with_mixed_escape": strict["full_repeatability_50"]["candidate_with_mixed_escape"],
                "stable_candidates": strict["full_repeatability_50"]["stable_candidates"],
            },
        },
        "best_raw_escape_prompt_component": {
            "source": "prompt_component_ablation30/analysis",
            "strategy": prompt_best_escape_name,
            "escape_rate": prompt_best_escape_stats["escape_rate"],
            "mean_attack_z": prompt_best_escape_stats["mean_attack_z"],
            "mean_z_drop": prompt_best_escape_stats["mean_z_drop"],
            "auroc": prompt_best_escape_stats["auroc"],
        },
        "best_quality_escape_prompt_component": {
            "source": "prompt_component_ablation30/analysis",
            "strategy": prompt_best_quality_name,
            "escape_rate": prompt_best_quality_stats["escape_rate"],
            "mean_attack_z": prompt_best_quality_stats["mean_attack_z"],
            "mean_z_drop": prompt_best_quality_stats["mean_z_drop"],
            "auroc": prompt_best_quality_stats["auroc"],
        },
        "best_external_baseline_escape": {
            "source": "baseline_comparison",
            "label": baseline_best_escape["label"],
            "escape_rate": baseline_best_escape["escape_rate"],
            "mean_z": baseline_best_escape["mean_z"],
            "auroc": baseline_best_escape["auroc"],
        },
    }

    (out_dir / "best_observed_frontier.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Best observed frontier",
        "",
        "This consolidates the strongest results already present in the repository.",
        "",
        "## Highest observed strict-valid result",
        "",
        f"- K = {summary['best_strict_valid_asr_observed']['k']}",
        f"- Valid ASR = {summary['best_strict_valid_asr_observed']['valid_asr']:.1%}",
        f"- Escape rate = {summary['best_strict_valid_asr_observed']['escape_rate']:.1%}",
        f"- Surface-valid rate = {summary['best_strict_valid_asr_observed']['surface_valid_rate']:.1%}",
        f"- Semantic-valid rate = {summary['best_strict_valid_asr_observed']['semantic_valid_rate']:.1%}",
        "",
        "## Most stability-backed strict result",
        "",
        f"- Strict Valid ASR = {strict['strict_valid_asr']:.1%}",
        f"- 95% CI = [{strict['strict_valid_asr_ci95'][0]:.3f}, {strict['strict_valid_asr_ci95'][1]:.3f}]",
        f"- Threshold plateau = {strict['threshold_plateau']['start']:.2f} to {strict['threshold_plateau']['end']:.2f}",
        f"- 50-repeat candidate flips = {strict['full_repeatability_50']['candidate_with_mixed_escape']}",
        f"- 50-repeat stable candidates = {strict['full_repeatability_50']['stable_candidates']}/{strict['full_repeatability_50']['total_candidates']}",
        "",
        "## Best raw detector escape",
        "",
        f"- Strategy = {summary['best_raw_escape_prompt_component']['strategy']}",
        f"- Escape rate = {summary['best_raw_escape_prompt_component']['escape_rate']:.1%}",
        f"- Mean z drop = {summary['best_raw_escape_prompt_component']['mean_z_drop']:.3f}",
        "",
        "## Best quality-weighted prompt component escape",
        "",
        f"- Strategy = {summary['best_quality_escape_prompt_component']['strategy']}",
        f"- Escape rate = {summary['best_quality_escape_prompt_component']['escape_rate']:.1%}",
        f"- Mean z drop = {summary['best_quality_escape_prompt_component']['mean_z_drop']:.3f}",
        "",
        "## Strongest external baseline",
        "",
        f"- {summary['best_external_baseline_escape']['label']}",
        f"- Escape rate = {summary['best_external_baseline_escape']['escape_rate']:.1%}",
        f"- Mean z = {summary['best_external_baseline_escape']['mean_z']:.3f}",
        "",
        "## Interpretation",
        "",
        "- If you optimize purely for the highest strict-valid ASR already observed, K=1 is currently the leader at 33.3%.",
        "- If you optimize for paper robustness, the feedback-driven strict result at 30.0% is stronger because it has a threshold plateau and 50-repeat stability.",
        "- If you optimize for raw detector escape, boundary-only is the strongest prompt primitive at 93.3%, but quality must be accounted for separately.",
    ]
    (out_dir / "best_observed_frontier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(4)
    values = [
        summary["best_strict_valid_asr_observed"]["valid_asr"],
        strict["strict_valid_asr"],
        summary["best_raw_escape_prompt_component"]["escape_rate"],
        summary["best_external_baseline_escape"]["escape_rate"],
    ]
    labels = [
        f"Best strict\n(K={summary['best_strict_valid_asr_observed']['k']})",
        "Stability-backed\nstrict",
        "Best raw\nprompt escape",
        "Best external\nbaseline",
    ]
    colors = ["#2563eb", "#7c3aed", "#16a34a", "#64748b"]
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Observed rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "best_observed_frontier.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
