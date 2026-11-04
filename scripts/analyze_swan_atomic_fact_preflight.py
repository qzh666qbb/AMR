"""Post-hoc length diagnostic for saved atomic-fact preflight outputs."""

import json
import math
from pathlib import Path
import statistics

from run_segmentation_intervention import ROOT, sha, write_json

FOLDER = ROOT / "experiments/swan_atomic_fact_preflight1"
BINS = ((8, 10), (11, 13), (14, 18))


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def describe(rows):
    return {
        "facts": len(rows),
        "scale_compatible": sum(row["scale_compatible"] for row in rows),
        "scale_compatible_fraction": (
            sum(row["scale_compatible"] for row in rows) / len(rows) if rows else None
        ),
        "template_triples_mean": (
            statistics.mean(row["template_triples"] for row in rows) if rows else None
        ),
        "compatible_bank_templates_mean": (
            statistics.mean(row["compatible_bank_templates"] for row in rows)
            if rows else None
        ),
    }


def analyze(folder=FOLDER):
    summary_path = folder / "summary.json"
    audit_path = folder / "audit.json"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Preflight must be complete and audited")
    rows = []
    inputs = {
        str(summary_path.relative_to(ROOT)): sha(summary_path),
        str(audit_path.relative_to(ROOT)): sha(audit_path),
    }
    for path in sorted((folder / "parser").glob("*.json")):
        saved = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = sha(path)
        rows.extend({
            "prompt_id": saved["prompt_id"],
            "sentence_index": row["sentence_index"],
            "words": row["words"],
            "template_triples": row["template_triples"],
            "compatible_bank_templates": row["compatible_bank_templates"],
            "scale_compatible": row["scale_compatible"],
        } for row in saved["facts"])
    if len(rows) != 50:
        raise ValueError("Expected 50 frozen fact rows")
    bins = {}
    for lower, upper in BINS:
        selected = [row for row in rows if lower <= row["words"] <= upper]
        bins[f"{lower}_{upper}"] = describe(selected)
    result = {
        "status": "complete_exploratory_posthoc",
        "scope": "Length bins were examined after the confirmatory continuation gate failed",
        "overall": describe(rows),
        "word_bins": bins,
        "pearson_words_template_triples": pearson(
            [row["words"] for row in rows],
            [row["template_triples"] for row in rows],
        ),
        "interpretation": (
            "Compatibility is concentrated in 8-10 word facts. This motivates a fresh, "
            "pre-registered microfact study but does not alter the failed 0/5 document gate."
        ),
        "limits": [
            "Bins are post-hoc and use only 50 facts from five development documents",
            "Word count does not identify a causal graph-size threshold",
            "Count compatibility is necessary but not sufficient for S2Match acceptance",
            "No generation, attack, ASR or population FPR is evaluated",
        ],
        "new_dataset_http_requests": 0,
        "new_generation_api_calls": 0,
        "new_neural_parser_calls": 0,
        "new_s2match_optimization_calls": 0,
        "input_sha256": inputs,
        "script_sha256": sha(Path(__file__)),
        "rows": rows,
    }
    write_json(folder / "length_diagnostic.json", result)
    lines = [
        "# 原子事实长度探索性诊断",
        "",
        "该分析在预注册继续门失败后进行，属于后验开发诊断，不修改 0/5 的正式判断。",
        "",
        "| 词数层 | 事实句 | 规模兼容 | 兼容率 | 模板图三元组均值 | 兼容模板数均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lower, upper in BINS:
        row = bins[f"{lower}_{upper}"]
        lines.append(
            f"| {lower}--{upper} | {row['facts']} | {row['scale_compatible']} | "
            f"{row['scale_compatible_fraction']:.1%} | "
            f"{row['template_triples_mean']:.3f} | "
            f"{row['compatible_bank_templates_mean']:.3f} |"
        )
    lines += [
        "",
        f"词数与模板图三元组数的 Pearson 相关为 "
        f"`{result['pearson_words_template_triples']:.6f}`。",
        "",
        "兼容性集中在 8--10 词层，支持另行预注册 10 词以内的 microfact 方案；"
        "该结果不能当作因果阈值，也不代表注入成功。",
        "",
    ]
    (folder / "length_diagnostic.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    output = analyze()
    print(json.dumps({
        "status": output["status"],
        "overall": output["overall"],
        "word_bins": output["word_bins"],
        "pearson_words_template_triples": output["pearson_words_template_triples"],
        "new_generation_api_calls": output["new_generation_api_calls"],
    }, indent=2))
