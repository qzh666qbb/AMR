"""合并对称口径零假设的多个分片结果。

各分片写入自己的 PARTIAL_WINDOW.json（含 per_unit），本脚本去重合并
（按 cohort + doc + unit 文本哈希）为一个文件，供 D7 分析使用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "experiments/swan_multiview_development1/null_fp_audit1/shards"
DEFAULT_OUT = (ROOT / "experiments/swan_multiview_development1/"
               "null_fp_audit1/PARTIAL_WINDOW_MERGED.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=SHARDS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    seen, merged, calls = set(), [], 0
    shard_files = sorted(args.shards.glob("shard*/PARTIAL_WINDOW.json"))
    print(f"发现分片文件 {len(shard_files)} 个")
    for f in shard_files:
        d = json.loads(f.read_text())
        calls += d.get("s2match_calls", 0)
        for u in d["per_unit"]:
            key = (u["cohort"], u["doc"],
                   hashlib.sha256(u["unit"].encode()).hexdigest()[:16])
            if key in seen:
                continue
            seen.add(key)
            merged.append(u)
        print(f"  {f.parent.name}: {len(d['per_unit'])} 条")

    by_cohort = {}
    for u in merged:
        by_cohort[u["cohort"]] = by_cohort.get(u["cohort"], 0) + 1
    nw = [u["n_windows"] for u in merged]
    payload = {
        "note": "对称口径零例：单元分数 = 与之重叠的所有候选窗口的逐元素最大",
        "unit_definition": "句界单元 span + 窗口聚合",
        "merged_from": [str(f.parent.name) for f in shard_files],
        "cohorts": by_cohort,
        "units": len(merged),
        "windows_per_unit_mean": round(sum(nw) / len(nw), 2) if nw else 0,
        "s2match_calls_total": calls,
        "per_unit": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n合并后：{len(merged)} 个单元 {by_cohort}，"
          f"平均窗口数 {payload['windows_per_unit_mean']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
