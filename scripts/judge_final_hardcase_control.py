"""Blindly judge the final hard-case control candidates."""

import json
import os
from pathlib import Path

from judge_candidate_budget16_blind import judge_source, load_dotenv


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "final_hardcase_control"
load_dotenv(root / ".env")
key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    raise RuntimeError("DEEPSEEK_API_KEY is missing")
rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
groups = []
for index in (6, 16):
    selected = [row for row in rows if row["index"] == index]
    groups.append(
        judge_source(index, selected[0]["source"], [row["attack"] for row in selected], key, "deepseek-v4-flash")
    )
    print(f"judged index={index}", flush=True)
(out / "judgments.json").write_text(
    json.dumps(
        {"protocol": {"blinded_to_strategy_and_detector": True, "temperature": 0.0}, "groups": groups},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
