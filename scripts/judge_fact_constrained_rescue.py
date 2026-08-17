"""Blindly judge all fact-constrained rescue candidates."""

import json
import os
from pathlib import Path

from judge_candidate_budget16_blind import judge_source, load_dotenv


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "fact_constrained_rescue"
load_dotenv(root / ".env")
key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    raise RuntimeError("DEEPSEEK_API_KEY is missing")
payload = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
groups = []
for record in payload["records"]:
    groups.append(
        judge_source(
            int(record["index"]),
            record["source"],
            record["candidates"],
            key,
            "deepseek-v4-flash",
        )
    )
    print(f"judged index={record['index']}", flush=True)
(out / "judgments.json").write_text(
    json.dumps(
        {
            "protocol": {"temperature": 0.0, "blinded_to_detector_scores": True},
            "groups": groups,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
