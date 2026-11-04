"""Export blind transfer comparisons from audited, frozen summary metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.font_manager import findfont
    import numpy as np

    assert Path(findfont("DejaVu Sans", fallback_to_default=False)).is_file()
    summary_path = out / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    assert summary["audit_passed"] and summary["metric_audit"]["passed"]
    assert (out / "COMPLETE.json").exists()
    assert not any((out / name).exists() for name in ("HALT.json", "STOP.json"))
    assert summary["freeze_sha256"] == hashlib.sha256((out / "FREEZE.json").read_bytes()).hexdigest()
    assert summary["metric_audit"]["results_sha256"] == hashlib.sha256(
        (out / "RESULTS.json").read_bytes()).hexdigest()
    modes = ("raw_s6", "s6", "mean", "gate", "attention")
    conditions = [condition for condition in ("clean", "paraphrase", "split", "merge", "comma", "space")
                  if condition in summary["conditions"]]
    labels = ["Raw S6", "Normalized\nS6", "Equal\nweights", "Gate\n5 seeds", "Attention\n5 seeds"]
    scores = np.array([[summary["conditions"][condition]["methods"][mode + "/blind"]["auroc"]["mean"]
                        for mode in modes] for condition in conditions])
    differences = scores - scores[conditions.index("clean")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"wspace": 0.3})
    panels = [
        (scores, "Blind document AUROC", "RdBu", TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)),
        (differences, "AUROC change from clean", "RdBu", TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
    ]
    for index, (axis, (values, title, cmap, norm)) in enumerate(zip(axes, panels)):
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        axis.set_xticks(range(len(modes)), labels)
        axis.set_yticks(range(len(conditions)), conditions)
        axis.tick_params(axis="both", length=0, pad=8)
        axis.set_title(title, loc="left", fontweight="bold", pad=16)
        for row in range(len(conditions)):
            for column in range(len(modes)):
                value = values[row, column]
                rgba = image.cmap(image.norm(value))
                luminance = sum(weight * channel for weight, channel in
                                zip((0.2126, 0.7152, 0.0722), rgba[:3]))
                axis.text(column, row, f"{value:.3f}" if index == 0 else f"{value:+.3f}",
                          ha="center", va="center", color="white" if luminance < 0.48 else "#20252b",
                          fontsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.055, pad=0.16)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.2)
    fig.suptitle("Frozen model transfer to new paired text", x=0.08, y=0.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.065,
             f"Same {summary['conditions']['clean']['readout_pairs']} held-out document pairs across conditions. "
             "All windows extracted from tested text.\n"
             "Gate/attention cells average the five fixed seed metrics; no prediction ensemble. "
             "Document 1%/5% FPR calibration is insufficient.",
             ha="left", va="bottom", fontsize=9, color="#454b52")
    source_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    metadata = {"Title": "Audited frozen-model blind transfer",
                "Subject": "SUMMARY.json SHA256 " + source_sha + "; freeze SHA256 " + summary["freeze_sha256"],
                "Creator": "scripts/plot_multiscale_transfer.py"}
    fig.savefig(out / "fig_blind_transfer.pdf", metadata=metadata, bbox_inches="tight")
    fig.savefig(out / "fig_blind_transfer.png", dpi=200, bbox_inches="tight",
                metadata={"SourceSHA256": source_sha, "FreezeSHA256": summary["freeze_sha256"]})
    plt.close(fig)
    print(json.dumps({"plot_written": True, "summary_sha256": source_sha,
                      "conditions": conditions, "no_fitting_or_selection": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
