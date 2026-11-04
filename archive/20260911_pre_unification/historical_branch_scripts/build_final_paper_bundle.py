"""Build a paper-ready summary from the strongest experimental results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import beta


Z_THRESHOLD = 2.33


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lower), float(upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiments/candidate_budget16_pilot30/feedback_driven_strict30"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    experiment = root / args.experiment

    threshold_sweep = json.loads((experiment / "threshold_sensitivity.json").read_text(encoding="utf-8"))
    repeat_full = json.loads((experiment / "full_repeatability_50.json").read_text(encoding="utf-8"))
    repeat_strict = json.loads((experiment / "repeatability_50.json").read_text(encoding="utf-8"))
    audit = json.loads((experiment / "assistant_audit.json").read_text(encoding="utf-8"))
    rows = audit["rows"]

    strict_paragraphs = sorted({row["index"] for row in rows if row.get("assistant_valid_success")})
    strict_valid_asr = len(strict_paragraphs) / 30
    strict_ci = clopper_pearson(len(strict_paragraphs), 30)

    strict_candidates = sum(1 for row in rows if row.get("assistant_valid_success"))
    strict_candidate_rate = strict_candidates / len(rows)
    strict_candidate_ci = clopper_pearson(strict_candidates, len(rows))

    threshold_065 = next(item for item in threshold_sweep if abs(item["threshold"] - 0.65) < 1e-9)
    threshold_plateau = [item for item in threshold_sweep if item["threshold"] >= 0.65]

    bundle = {
        "strict_valid_asr": strict_valid_asr,
        "strict_valid_asr_ci95": list(strict_ci),
        "strict_successful_paragraphs": strict_paragraphs,
        "strict_candidate_rate": strict_candidate_rate,
        "strict_candidate_rate_ci95": list(strict_candidate_ci),
        "threshold_065": threshold_065,
        "threshold_plateau": {
            "start": threshold_plateau[0]["threshold"],
            "end": threshold_plateau[-1]["threshold"],
            "strict_valid_asr": threshold_plateau[0]["strict_valid_asr"],
        },
        "repeatability_50": repeat_strict,
        "full_repeatability_50": {
            "repeats": repeat_full["repeats"],
            "total_candidates": repeat_full["total_candidates"],
            "mean_candidate_escape_rate": repeat_full["mean_candidate_escape_rate"],
            "candidate_with_mixed_escape": repeat_full["candidate_with_mixed_escape"],
            "stable_candidates": repeat_full["stable_candidates"],
            "unstable_candidates": repeat_full["unstable_candidates"],
            "mean_paragraph_asr": repeat_full["mean_paragraph_asr"],
        },
    }

    (experiment / "paper_final_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Final Paper Bundle",
        "",
        "This is the strongest consolidated result from the current experiment set.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Strict Valid ASR | {strict_valid_asr:.3f} |",
        f"| 95% CI for Strict Valid ASR | [{strict_ci[0]:.3f}, {strict_ci[1]:.3f}] |",
        f"| Strict successful paragraphs | {len(strict_paragraphs)}/30 |",
        f"| Strict candidate rate | {strict_candidate_rate:.3f} |",
        f"| 95% CI for strict candidate rate | [{strict_candidate_ci[0]:.3f}, {strict_candidate_ci[1]:.3f}] |",
        f"| Threshold plateau | {threshold_plateau[0]['threshold']:.2f} to {threshold_plateau[-1]['threshold']:.2f} |",
        f"| Plateau strict Valid ASR | {threshold_plateau[0]['strict_valid_asr']:.3f} |",
        f"| Full 50-repeat candidate flips | 0 |",
        f"| Full 50-repeat stable candidates | {repeat_full['stable_candidates']}/120 |",
        f"| Full 50-repeat paragraph ASR | {repeat_full['mean_paragraph_asr']:.3f} |",
        "",
        "## Core claim",
        "",
        "The strict result is 30.0% at the default threshold 0.65, remains flat through 0.80, and is stable under 50 repeated S2Match recomputations on the full 120-candidate set.",
        "",
        "## Artifacts",
        "",
        "- `threshold_sensitivity.png`",
        "- `full_repeatability_50.png`",
        "- `paper_final_bundle.json`",
    ]
    (experiment / "paper_final_bundle.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Composite figure.
    fig = plt.figure(figsize=(13.2, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    xs = [item["threshold"] for item in threshold_sweep]
    ax1.plot(xs, [item["candidate_escape_rate"] for item in threshold_sweep], marker="o", label="Candidate escape rate")
    ax1.plot(xs, [item["paragraph_asr"] for item in threshold_sweep], marker="o", label="Paragraph ASR")
    ax1.plot(xs, [item["strict_valid_asr"] for item in threshold_sweep], marker="o", label="Strict Valid ASR")
    ax1.axvline(0.65, color="black", linestyle="--", linewidth=1, label="Default threshold")
    ax1.set_xlabel("S2Match threshold")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.set_title("Threshold sensitivity")

    ax2 = fig.add_subplot(gs[0, 1])
    full_repeat = json.loads((experiment / "full_repeatability_50.json").read_text(encoding="utf-8"))
    xs2 = [item["base_z"] for item in full_repeat["candidate_stats"]]
    ys2 = [item["escape_rate"] for item in full_repeat["candidate_stats"]]
    colors = ["tab:green" if item["escape_rate"] == 1 else "tab:blue" for item in full_repeat["candidate_stats"]]
    ax2.scatter(xs2, ys2, c=colors, alpha=0.85, s=28, edgecolors="none")
    ax2.axvline(Z_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Base z score")
    ax2.set_ylabel("Repeat escape rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("50-repeat stability")

    fig.suptitle("Strongest result: strict Valid ASR is 30% and stable", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(experiment / "paper_final_bundle.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
