"""Prepare genuinely blank, independently shuffled human-review packets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_semantic_readout import PRIVATE, OUT, IDS
from run_semantic_sequence_cycle import atomic, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/core"))
from semantic_cycle2 import constructed_text

README = """# 独立人工盲评

当前包内答案全部空白。两位真人独立完成，不用LLM代填、不互看、不查看对照键或实验结果。
填写 reviewer.json 中真实评审者代号、完成时间与 human_attestation=true；不需要提供实名。
CSV里的文本和图仅用于授权研究，不对外转发；复制评审者目录时不要复制admin目录。

1. semantic.csv：匿名图是待实现的意义。检查载体是否实现该意义，给
semantic_preserved=yes/no/unknown、role_direction、unknown_role、name_coreference、
nested_event、speech_act同样取yes/no/unknown，并写依据。仅判断显式目标要求，
不凭出现的关键词判定。空槽也必须评审，按no或unknown保留。
2. paraphrase.csv：只看源载体和复述段落，判断原意及疑问/请求是否保留。
semantic_preserved=yes/no/unknown；如果唯一定位对应主要动词，填写
phrase为全文中唯一精确子串，predicate为其中准确动词，status=located；
否则status=unknown/changed/absent，保留行并说明。不看目标图或符号。
3. blind_events.csv：只看段落，独立列出主事件。events_json填JSON数组，
每项含start、end、predicate、meaning。字符位置从0开始，左闭右开，
predicate必须等于text[start:end]。无需凑足8个，也不要将每个附属动词算主事件。
相同文字在不同位置可以是不同事件；同一位置不得重复。无事件填[]，
无法判断status=unknown并写原因。text_words.txt提供字符索引辅助，只作位置查找。

三个任务分别填写，先完成blind_events再查看其他任务；对应ID在各任务间随机化。
不要查看原模板编号、计划符号、源槽位、文档条件或模型预测。
提交完整CSV与reviewer.json后运行自有回收验证入口。
分歧保留给第三位真人裁决；不自动把unknown转成yes。
"""


def write_csv(path, rows, labels):
    fields = list(rows[0]) + labels
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, **{k: "" for k in labels}} for row in rows)


def prepare():
    review = PRIVATE / "human_review"
    if (review / "admin/MANIFEST.json").exists():
        raise RuntimeError("Review packet already sealed")
    assignments = json.loads((PRIVATE / "ASSIGNMENTS.json").read_text())
    source = {name: [] for name in ("semantic", "paraphrase", "blind_events")}
    for i in IDS:
        doc = json.loads((PRIVATE / "documents" / f"d{i:02d}.json").read_text())
        for slot in assignments[i]["slots"]:
            s = slot["slot"]
            carrier = doc["arms"]["positive"]["slots"][s]
            source["semantic"].append(({"target_meaning_graph": slot["target"], "carrier": carrier},
                                       {"doc_id": i, "slot": s, "symbol": slot["symbol"]}))
            source["paraphrase"].append(({"source_carrier": carrier,
                                         "paraphrased_account": doc["arms"]["positive"]["paraphrase"]},
                                        {"doc_id": i, "slot": s}))
        for arm in ("positive", "unwatermarked"):
            conditions = ("clean", "paraphrase", "split", "merge") if arm == "positive" else ("clean", "paraphrase")
            for condition in conditions:
                data = doc["arms"][arm]
                text = data["paraphrase"] if condition == "paraphrase" else constructed_text(data["slots"], condition)[0]
                source["blind_events"].append(({"text": text}, {"doc_id": i, "arm": arm, "condition": condition}))
    files, keys = {}, {}
    for reviewer in (1, 2):
        folder = review / f"reviewer{reviewer}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "INSTRUCTIONS.md").write_text(README)
        atomic(folder / "reviewer.json", {"reviewer_code": "", "completed_at": "",
                                          "human_attestation": False, "independent_attestation": False})
        keys[str(reviewer)] = {}
        token_indices = []
        for task, entries in source.items():
            rng = random.Random(20261025 + reviewer * 100 + list(source).index(task))
            indices = list(range(len(entries)))
            rng.shuffle(indices)
            rows, mapping = [], {}
            for j, idx in enumerate(indices):
                payload, identity = entries[idx]
                # Different ids in each task prevent cross-task identity lookups.
                item_id = hashlib.sha256(f"reviewer{reviewer}:{task}:{j}:20261025".encode()).hexdigest()[:16]
                rows.append({"item_id": item_id, **payload})
                mapping[item_id] = {**identity, "payload_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()).hexdigest()}
                if task == "blind_events":
                    token_indices.append(item_id + "\n" + "\n".join(
                        f"[{m.start()},{m.end()}) {m.group()}" for m in re.finditer(r"\S+", payload["text"])))
            keys[str(reviewer)][task] = mapping
            labels = {
                "semantic": ["semantic_preserved", "role_direction", "unknown_role", "name_coreference",
                             "nested_event", "speech_act", "notes"],
                "paraphrase": ["semantic_preserved", "status", "phrase", "predicate", "notes"],
                "blind_events": ["status", "events_json", "notes"]}[task]
            write_csv(folder / f"{task}.csv", rows, labels)
        (folder / "text_words.txt").write_text("\n\n".join(token_indices) + "\n")
        for p in folder.iterdir():
            files[str(p.relative_to(review))] = sha(p)
    atomic(review / "admin/KEYS.json", keys)
    manifest = {"at": datetime.now(timezone.utc).isoformat(),
                "reviewers": 2, "counts_per_reviewer": {k: len(v) for k, v in source.items()},
                "blank_labels": True, "human_completed": False, "files": files,
                "private_keys_sha256": sha(review / "admin/KEYS.json"),
                "source_documents_sha256": {f"d{i:02d}": sha(PRIVATE / "documents" / f"d{i:02d}.json") for i in IDS}}
    atomic(review / "admin/MANIFEST.json", manifest)
    atomic(OUT / "HUMAN_REVIEW_STATUS.json", {k: v for k, v in manifest.items() if k != "source_documents_sha256"})
    return manifest


def collect():
    review = PRIVATE / "human_review"
    keys = json.loads((review / "admin/KEYS.json").read_text())
    expected = json.loads((review / "admin/MANIFEST.json").read_text())
    assignments = json.loads((PRIVATE / "ASSIGNMENTS.json").read_text())
    documents = {i: json.loads((PRIVATE / "documents" / f"d{i:02d}.json").read_text()) for i in IDS}
    if any(sha(PRIVATE / "documents" / (doc_id + ".json")) != signature
           for doc_id, signature in expected["source_documents_sha256"].items()):
        raise RuntimeError("Review source document changed")
    errors, completed, validated = [], {}, {}
    for reviewer in (1, 2):
        folder = review / f"reviewer{reviewer}"
        metadata = json.loads((folder / "reviewer.json").read_text())
        if not (metadata.get("reviewer_code") and metadata.get("completed_at") and
                metadata.get("human_attestation") is True and metadata.get("independent_attestation") is True):
            errors.append({"reviewer": reviewer, "reason": "missing_human_identity_or_attestation"})
        completed[str(reviewer)], validated[str(reviewer)] = {}, {}
        for task in ("semantic", "paraphrase", "blind_events"):
            with (folder / f"{task}.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            mapping = keys[str(reviewer)][task]
            ids = [r["item_id"] for r in rows]
            if len(set(ids)) != len(ids) or set(ids) != set(mapping):
                errors.append({"reviewer": reviewer, "task": task, "reason": "planned_items_changed"})
            valid = 0
            for row in rows:
                okay = row.get("item_id") in mapping
                payload_fields = {"semantic": ("target_meaning_graph", "carrier"),
                                  "paraphrase": ("source_carrier", "paraphrased_account"),
                                  "blind_events": ("text",)}[task]
                payload = {k: row.get(k, "") for k in payload_fields}
                if okay:
                    expected_payload = mapping[row["item_id"]].get("payload_sha256")
                    if expected_payload is not None:
                        okay &= hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == expected_payload
                    else:
                        # Earlier sealed blank packets have source identities instead of payload hashes.
                        identity = mapping[row["item_id"]]
                        doc = documents[identity["doc_id"]]
                        if task == "semantic":
                            s = identity["slot"]
                            original = {"target_meaning_graph": assignments[identity["doc_id"]]["slots"][s]["target"],
                                        "carrier": doc["arms"]["positive"]["slots"][s]}
                        elif task == "paraphrase":
                            original = {"source_carrier": doc["arms"]["positive"]["slots"][identity["slot"]],
                                        "paraphrased_account": doc["arms"]["positive"]["paraphrase"]}
                        else:
                            arm, condition = identity["arm"], identity["condition"]
                            data = doc["arms"][arm]
                            original = {"text": data["paraphrase"] if condition == "paraphrase" else
                                        constructed_text(data["slots"], condition)[0]}
                        okay &= payload == original
                if task == "semantic":
                    okay &= all(row.get(k) in ("yes", "no", "unknown") for k in
                                ("semantic_preserved", "role_direction", "unknown_role", "name_coreference",
                                 "nested_event", "speech_act"))
                elif task == "paraphrase":
                    okay &= row.get("semantic_preserved") in ("yes", "no", "unknown")
                    okay &= row.get("status") in ("located", "unknown", "changed", "absent")
                    if row.get("status") == "located":
                        text, phrase, predicate = row["paraphrased_account"], row["phrase"], row["predicate"]
                        okay &= bool(phrase and predicate) and text.count(phrase) == 1 and predicate in phrase
                else:
                    okay &= row.get("status") in ("complete", "unknown")
                    if row.get("status") == "complete":
                        try:
                            events = json.loads(row["events_json"])
                            spans = []
                            if not isinstance(events, list):
                                raise ValueError()
                            for event in events:
                                a, b = event["start"], event["end"]
                                if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(row["text"])
                                        and row["text"][a:b] == event["predicate"] and event.get("meaning")):
                                    raise ValueError()
                                spans.append((a, b))
                            okay &= len(set(spans)) == len(spans)
                        except (ValueError, KeyError, TypeError):
                            okay = False
                valid += int(okay)
            completed[str(reviewer)][task] = {"valid": valid, "planned": len(mapping)}
            validated[str(reviewer)][task] = rows
            if valid != len(mapping):
                errors.append({"reviewer": reviewer, "task": task, "reason": "incomplete_or_invalid_labels",
                               "valid": valid, "planned": len(mapping)})
    codes = [json.loads((review / f"reviewer{i}/reviewer.json").read_text())["reviewer_code"] for i in (1, 2)]
    if codes[0] == codes[1]:
        errors.append({"reason": "two_distinct_reviewers_required"})
    result = {"human_completed": not errors, "counts": completed, "errors": errors,
              "all_planned_items_retained": True, "formal_G1_G2_G3": "NOT_EVALUATED",
              "original_blank_manifest_sha256": sha(review / "admin/MANIFEST.json")}
    # Valid completed rows are retained privately for third-person adjudication.
    if not errors:
        atomic(review / "admin/RETURNED_LABELS.json", validated)
        result["adjudication"] = "REQUIRED_BEFORE_FORMAL_GATES"
    atomic(OUT / "HUMAN_COLLECTION_CHECK.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "collect"))
    args = p.parse_args()
    result = {"prepare": prepare, "collect": collect}[args.action]()
    print(json.dumps({k: v for k, v in result.items() if k not in ("files",)}, ensure_ascii=False))
