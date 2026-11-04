"""导出匿名对应盲审候选；保留空标签，不伪造人工金标。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    for path in sorted((args.run / "private/results").glob("*.json")):
        payload = json.loads(path.read_text())
        pooled = {}
        for method, result in payload["variants"].items():
            events = {event["event_id"]: event for event in result["events"]}
            for edge in result["alignment"]["selected"]:
                source = result["observations"][edge["candidate_index"]]
                event = events[edge["event_id"]]
                support = result["observations"][edge["event_candidate_index"]]
                identity = (payload["row_id"], source["start_char"], source["end_char"],
                            support["start_char"], support["end_char"], event["signature"])
                item_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                row = {
                    "item_id": item_id, "group_id": payload["group_id"],
                    "row_id": payload["row_id"], "variant_id": payload["variant_id"],
                    "text_sha256": payload["text_sha256"],
                    "candidate_text": source["text"],
                    "candidate_start": source["start_char"], "candidate_end": source["end_char"],
                    "event_support_text": support["text"],
                    "event_support_start": support["start_char"],
                    "event_support_end": support["end_char"],
                    "event_description": event["description"],
                    "event_amr": support["amr"].replace("\n", " "),
                    "reviewer_id": "", "fact_id": "", "same_fact": "",
                    "event_faithful_to_support": "", "candidate_supports_event": "",
                    "contradiction_type": "", "label_status": "unreviewed", "notes": "",
                }
                entry = pooled.setdefault(item_id, {"row": row, "methods": []})
                entry["methods"].append(method)
        for item_id, entry in pooled.items():
            rows.append(entry["row"])
            key.append({"item_id": item_id, "methods": sorted(set(entry["methods"])),
                        "result_path": str(path), "result_sha256": hashlib.sha256(
                            path.read_bytes()).hexdigest()})
    rows.sort(key=lambda row: row["item_id"])
    if not rows:
        raise ValueError("没有可审核候选")
    for reviewer in ("A", "B"):
        with (args.output / f"reviewer_{reviewer}.tsv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "private").mkdir()
    (args.output / "private/method_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    (args.output / "SUMMARY.json").write_text(json.dumps({
        "items": len(rows), "documents": len({row["group_id"] for row in rows}),
        "rows": len({row["row_id"] for row in rows}), "completed_human_labels": 0,
        "status": "pooled_predicted_pairs_not_full_gold_not_recall_denominator",
        "blind_fields_removed": ["method", "bank_scores", "statistic", "rule_weight"],
    }, indent=2) + "\n")
    print(json.dumps({"items": len(rows), "human_labels": 0}))


if __name__ == "__main__":
    main()
