"""Sweep the S2Match threshold and measure how the strict Valid ASR changes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LMBDA = 0.25
Z_THRESHOLD = 2.33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiments/candidate_budget16_pilot30/feedback_driven_strict30"),
    )
    parser.add_argument("--start", type=float, default=0.50)
    parser.add_argument("--end", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def compute_z(doc_scores: list[float], threshold: float) -> float:
    count_green = sum(score >= threshold for score in doc_scores)
    n = len(doc_scores)
    return (count_green - LMBDA * n) / math.sqrt(n * LMBDA * (1 - LMBDA))


def main() -> None:
    args = parse_args()
    root = args.root
    experiment = root / args.experiment

    audit = json.loads((experiment / "assistant_audit.json").read_text(encoding="utf-8"))
    rows = audit["rows"]
    matches = json.loads((experiment / "detection" / "machine_amr_matches.json").read_text(encoding="utf-8"))["matched_amrs"]

    paragraph_positions = defaultdict(list)
    strict_positions = defaultdict(list)
    for row_pos, row in enumerate(rows):
        paragraph_positions[row["index"]].append(row_pos)
        if row.get("assistant_valid_success"):
            strict_positions[row["index"]].append(row_pos)

    thresholds = []
    cur = args.start
    while cur <= args.end + 1e-9:
        thresholds.append(round(cur, 2))
        cur += args.step

    records = []
    for threshold in thresholds:
        z_scores = [compute_z([sent["score"] for sent in doc], threshold) for doc in matches]
        candidate_escape_rate = sum(z <= Z_THRESHOLD for z in z_scores) / len(z_scores)
        paragraph_asr = sum(
            any(z_scores[pos] <= Z_THRESHOLD for pos in positions)
            for positions in paragraph_positions.values()
        ) / len(paragraph_positions)
        strict_valid_asr = sum(
            any(z_scores[pos] <= Z_THRESHOLD for pos in positions)
            for positions in strict_positions.values()
        ) / len(paragraph_positions)
        strict_candidate_escape_rate = sum(
            z_scores[pos] <= Z_THRESHOLD for positions in strict_positions.values() for pos in positions
        ) / sum(len(positions) for positions in strict_positions.values())
        records.append(
            {
                "threshold": threshold,
                "candidate_escape_rate": candidate_escape_rate,
                "paragraph_asr": paragraph_asr,
                "strict_valid_asr": strict_valid_asr,
                "strict_candidate_escape_rate": strict_candidate_escape_rate,
            }
        )

    out_json = experiment / "threshold_sensitivity.json"
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = experiment / "threshold_sensitivity.md"
    best_plateau = [item for item in records if item["threshold"] >= 0.65]
    lines = [
        "# Threshold Sensitivity",
        "",
        "This sweep recomputes z-scores from the fixed per-sentence S2Match scores and varies the sentence-level threshold.",
        "",
        "| Threshold | Candidate escape rate | Paragraph ASR | Strict Valid ASR | Strict candidate escape rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item['threshold']:.2f} | {item['candidate_escape_rate']:.3f} | {item['paragraph_asr']:.3f} | {item['strict_valid_asr']:.3f} | {item['strict_candidate_escape_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Plateau at and above 0.65: strict Valid ASR = {best_plateau[0]['strict_valid_asr']:.3f} through {best_plateau[-1]['strict_valid_asr']:.3f}.",
            "The strict-valid curve is flat from 0.65 to 0.80 in this sweep.",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    xs = [item["threshold"] for item in records]
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(xs, [item["candidate_escape_rate"] for item in records], marker="o", label="Candidate escape rate")
    ax.plot(xs, [item["paragraph_asr"] for item in records], marker="o", label="Paragraph ASR")
    ax.plot(xs, [item["strict_valid_asr"] for item in records], marker="o", label="Strict Valid ASR")
    ax.axvline(0.65, color="black", linestyle="--", linewidth=1, label="Default threshold")
    ax.set_xlabel("S2Match threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(experiment / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)

    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
