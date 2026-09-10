"""Compare the current method against existing baseline runs under the same detector."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


ESCAPE_THRESHOLD = 2.33


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    if successes == 0:
        low = 0.0
    else:
        low = binomtest(successes, total).proportion_ci(confidence_level=1 - alpha, method="exact").low
    if successes == total:
        high = 1.0
    else:
        high = binomtest(successes, total).proportion_ci(confidence_level=1 - alpha, method="exact").high
    return float(low), float(high)


def load_detection(run_dir: Path) -> dict[str, float]:
    det = run_dir / "detection"
    machine = np.load(det / "machine_z_scores.npy")
    human = np.load(det / "human_z_scores.npy")
    return {
        "n": int(len(machine)),
        "mean_z": float(machine.mean()),
        "escape_rate": float(np.mean(machine <= ESCAPE_THRESHOLD)),
        "escape_count": int(np.sum(machine <= ESCAPE_THRESHOLD)),
        "escape_ci_low": clopper_pearson(int(np.sum(machine <= ESCAPE_THRESHOLD)), len(machine))[0],
        "escape_ci_high": clopper_pearson(int(np.sum(machine <= ESCAPE_THRESHOLD)), len(machine))[1],
        "auroc": float(
            roc_auc_score(
                np.r_[np.zeros(len(human)), np.ones(len(machine))],
                np.r_[human, machine],
            )
        ),
        "min_z": float(machine.min()),
        "max_z": float(machine.max()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines_root = root / "baselines" / "local" / "swan_acl2026_realnews" / "runs"

    baseline_runs = {
        "pilot_30x5": {
            "path": baselines_root / "api_pilot_30x5",
            "label": "Baseline pilot 30x5",
            "notes": "Existing 30-prompt pilot under the same detector",
        },
        "formal_250x5": {
            "path": baselines_root / "formal_250x5",
            "label": "Baseline formal 250x5",
            "notes": "Larger formal baseline, different sample size",
        },
    }

    rows = []
    for key, item in baseline_runs.items():
        stats = load_detection(item["path"])
        rows.append(
            {
                "method": key,
                "label": item["label"],
                "notes": item["notes"],
                **stats,
            }
        )

    current = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "full_repeatability_50.json").read_text(
            encoding="utf-8"
        )
    )
    strict = json.loads(
        (root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30" / "paper_final_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    rows.append(
        {
            "method": "proposed",
            "label": "Proposed method",
            "notes": "Quality-checked strict ASR + repeatability",
            "n": 30,
            "mean_z": float("nan"),
            "escape_rate": float(current["mean_paragraph_asr"]),
            "escape_count": int(round(current["mean_paragraph_asr"] * 30)),
            "escape_ci_low": float(strict["strict_valid_asr_ci95"][0]),
            "escape_ci_high": float(strict["strict_valid_asr_ci95"][1]),
            "auroc": float("nan"),
            "min_z": float("nan"),
            "max_z": float("nan"),
            "strict_valid_asr": float(strict["strict_valid_asr"]),
            "strict_candidate_rate": float(strict["strict_candidate_rate"]),
            "threshold_plateau_low": 0.65,
            "threshold_plateau_high": 0.80,
        }
    )

    output = {"threshold": ESCAPE_THRESHOLD, "rows": rows}
    (out_dir / "baseline_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Baseline comparison under the same detector",
        "",
        f"Escape threshold: {ESCAPE_THRESHOLD}",
        "",
        "| Method | n | Mean z | Escape rate | 95% CI | AUROC | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ci = (
            f"[{row['escape_ci_low']:.3f}, {row['escape_ci_high']:.3f}]"
            if np.isfinite(row["escape_ci_low"]) and np.isfinite(row["escape_ci_high"])
            else "-"
        )
        mean_z = f"{row['mean_z']:.3f}" if np.isfinite(row["mean_z"]) else "-"
        auroc = f"{row['auroc']:.3f}" if np.isfinite(row["auroc"]) else "-"
        lines.append(
            f"| {row['label']} | {row['n']} | {mean_z} | {row['escape_rate']:.1%} | {ci} | {auroc} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The pilot baseline is the cleanest apples-to-apples external reference because it shares the same 30-prompt scale.",
            "- The formal 250x5 run gives a larger-sample background estimate under the same detector.",
            "- The proposed method is reported with quality-checked strict ASR; its detector-level repeatability is summarized separately.",
            "",
            "## Strict-method specifics",
            "",
            f"- Strict Valid ASR: {strict['strict_valid_asr']:.1%}",
            f"- 95% CI: [{strict['strict_valid_asr_ci95'][0]:.3f}, {strict['strict_valid_asr_ci95'][1]:.3f}]",
            f"- Threshold plateau: {strict['threshold_plateau']['start']:.2f} to {strict['threshold_plateau']['end']:.2f}",
            f"- Plateau Strict Valid ASR: {strict['threshold_plateau']['strict_valid_asr']:.1%}",
        ]
    )
    (out_dir / "baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    xs = np.arange(len(rows))
    escape_rates = [row["escape_rate"] for row in rows]
    colors = ["#64748b", "#2563eb", "#16a34a"]
    bars = ax.bar(xs, escape_rates, color=colors)
    ax.set_xticks(xs, [row["label"] for row in rows], rotation=12, ha="right")
    ax.set_ylabel("Escape rate / Strict Valid ASR")
    ax.set_ylim(0, max(escape_rates) * 1.25 if escape_rates else 1)
    ax.grid(axis="y", alpha=0.25)
    for bar, rate in zip(bars, escape_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{rate:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
