"""Compare repeated AMR parses and detector scores within and across runs."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "parser_repeatability30"
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
runs = {}
for run_name in ("run1", "run2"):
    parsed = json.loads((out / f"parsed_amrs_{run_name}.json").read_text(encoding="utf-8"))
    scores = np.load(out / f"detection_{run_name}" / "machine_z_scores.npy")
    if len(parsed) != len(manifest) or len(scores) != len(manifest):
        raise RuntimeError(f"Length mismatch in {run_name}")
    runs[run_name] = (parsed, scores)

positions = defaultdict(list)
for position, row in enumerate(manifest):
    positions[int(row["source_index"])].append(position)

per_source = []
for source_index in sorted(positions):
    pos = positions[source_index]
    run_stats = {}
    for run_name, (parsed, scores) in runs.items():
        amr_serialized = [json.dumps(parsed[p], sort_keys=True) for p in pos]
        values = [float(scores[p]) for p in pos]
        run_stats[run_name] = {
            "unique_amr_parses": len(set(amr_serialized)),
            "unique_z_scores": len(set(values)),
            "z_scores": values,
        }
    cross_equal = [runs["run1"][0][p] == runs["run2"][0][p] for p in pos]
    per_source.append(
        {
            "source_index": source_index,
            **run_stats,
            "cross_run_equal_repeats": sum(cross_equal),
            "cross_run_total_repeats": len(cross_equal),
        }
    )

total_pairs = len(manifest)
cross_amr_equal = sum(
    runs["run1"][0][p] == runs["run2"][0][p] for p in range(total_pairs)
)
cross_z_equal = int(np.sum(runs["run1"][1] == runs["run2"][1]))
summary = {
    "protocol": {"unique_texts": 30, "repeats_per_run": 5, "independent_runs": 2},
    "within_run_sources_with_multiple_amrs": {
        run_name: sum(row[run_name]["unique_amr_parses"] > 1 for row in per_source)
        for run_name in runs
    },
    "within_run_sources_with_multiple_z": {
        run_name: sum(row[run_name]["unique_z_scores"] > 1 for row in per_source)
        for run_name in runs
    },
    "cross_run_amr_equal": cross_amr_equal,
    "cross_run_amr_total": total_pairs,
    "cross_run_z_equal": cross_z_equal,
    "cross_run_z_total": total_pairs,
    "per_source": per_source,
}
(out / "results.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# AMR解析重复性实验",
    "",
    "30个相同原文各复制5次并随机打乱，在两个独立模型加载运行中解析和检测。",
    "",
    f"- 运行1中出现多个AMR解析的原文：{summary['within_run_sources_with_multiple_amrs']['run1']}/30。",
    f"- 运行2中出现多个AMR解析的原文：{summary['within_run_sources_with_multiple_amrs']['run2']}/30。",
    f"- 跨运行逐位置AMR完全一致：{cross_amr_equal}/{total_pairs}。",
    f"- 跨运行逐位置z完全一致：{cross_z_equal}/{total_pairs}。",
]
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "per_source"}, indent=2))
