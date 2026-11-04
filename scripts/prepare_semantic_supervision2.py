"""Prepare blank fact/background packets; no generation, AMR or model inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src/core")]
from src.core.semantic_supervision import VERSION, split_scenes
import run_semantic_fact_carrier_pilot2 as v2

OUT = ROOT / "experiments/semantic_multiscale_supervision2"
PRIVATE = Path.home() / ".cache/amr/semantic_multiscale_supervision2"
SOURCE_FILES = (
    "src/core/semantic_supervision.py", "src/core/semantic_multiscale_supervised.py",
    "src/core/semantic_multiscale.py", "src/core/multiscale_fusion.py",
    "src/core/semantic_fact_carrier.py",
    "scripts/prepare_semantic_supervision2.py", "scripts/collect_semantic_supervision2.py",
    "scripts/run_semantic_supervision2.py", "tests/test_semantic_supervision2.py",
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def reviewer_metadata():
    return {"reviewer_type": "human", "reviewer_code": "", "completed_at": "",
            "human_attestation": False, "independent_attestation": False,
            "no_model_assistance_attestation": False}


def word_index(rows):
    return "\n\n".join(row["item_id"] + "\n" + "\n".join(
        f"[{m.start()},{m.end()}) {m.group()}" for m in re.finditer(r"\S+", row["text"]))
        for row in rows) + "\n"


def context_candidates(text, nlp):
    doc = nlp(text)
    sentences = list(doc.sents)
    anchors = sorted({(t.idx, t.idx + len(t.text)) for t in doc if t.pos_ == "VERB"})
    if len(anchors) > 32:
        return {"status": "budget_abstain", "anchor_count": len(anchors), "observations": []}
    observations = []
    for a, b in anchors:
        s = next(i for i, sentence in enumerate(sentences)
                 if sentence.start_char <= a < sentence.end_char)
        spans = []
        for width in (1, 2, 3):
            start = min(max(s - (width - 1) // 2, 0), max(len(sentences) - width, 0))
            end = min(start + width, len(sentences))
            lo, hi = sentences[start].start_char, sentences[end - 1].end_char
            spans.append({"span": [lo, hi], "text": text[lo:hi], "requested_sentences": width})
        observations.append({"anchor_span": [a, b], "anchor_text": text[a:b], "contexts": spans})
    return {"status": "ok" if observations else "no_observable_anchor",
            "anchor_count": len(anchors), "observations": observations}


INSTRUCTIONS = """# 独立真人事实、背景与候选标注

这是事实载体 v2 的补充监督包。全部答案空白；由真实评审者独立填写，
不使用模型代填，不互看，不查看模型开发评审或计划答案。正文仅用于授权研究。
公共角色和物品不说明文本已经实现了事实。无需实名，但声明须真实。

先完成 facts.csv 的实际事实和全文事件清单，再填写 candidates.csv，最后 pairs.csv。
facts/pairs 原有字段沿用原冻结规则；新增 events_json 和 inventory_status。
每个文件和字符索引均使用同一份真实完整文本，不能改原文、ID、锚点或上下文。

1. facts.csv：sender、recipient、object 填文本实际实体或 unknown。
transfer_status=completed/planned_not_occurred/other/unknown。completed 必须明确完成并收到；
planned_not_occurred 必须由叙述者明确当前确定计划及尚未发生。时态、转述、预留不足。
readability=clear/unclear/contradictory/unavailable；explicit_status=yes/no/unknown；
quality=1/2/3/4/5/unknown。evidence_json 为四字段字符证据对象，unknown 字段证据 []。
notes 必填。pairs.csv 的三项保留判断为 yes/no/unknown，双侧证据及 notes 必填。
原句写错但被忠实复述可判断保留；不按目标纠正。

2. 在同一 facts 行填 inventory_status=complete/unknown/unavailable，
events_json 为全文实际事件数组。不要只抄候选，名词化和漏候选也必须列出。
没有事件可填 []；无法保证清单完整填 unknown。事件结构见 SCHEMA.json。
每事件有本段局部 event_id、主谓词的逐字 anchor、meaning、kind、
sender/recipient/object/transfer_status、readability、explicit_status、
四字段 evidence、background_reason 和 notes。
kind=carrier/background/unknown。carrier 是实际完整合同事实；全文若另有真实载体仍须列出。
background 必须根据实际不同活动/对象/角色给正面依据，不能因模型未匹配或事实不清楚判背景。
合同有关但缺证据、矛盾或无法定位的事件用 unknown；不强制 ABCD。
将同一转移的计划、完成/尚未发生证据绑定为同一事件，标一个主要谓词 anchor；
确实不同的转移事件分别列出。背景活动也列出。证据字符区间从0开始、左闭右开，
quote 必须逐字等于 text[start:end]。背景未知的转移字段填 unknown，证据 []，readability=unclear。

3. candidates.csv：fact_item_id 对应完整 facts 文本。这里只标预先提取的共同动词锚点。
kind=carrier/background/duplicate/unknown；event_ids_json 填本段清单的 event_id 数组。
carrier 必须与独立事件清单的主 anchor 完全相同，一事件至多一个 carrier 候选；
同一载体的附属/重复观测填 duplicate，仍绑定同一事件。背景链接背景事件。
多个可能事件或无法判定填 unknown，可用 []，不能猜负类。
evidence_json 是含此候选锚点的逐字证据数组；notes 解释对应关系。
事件没有任何候选不要补锚点，它将记为漏提取并保留在分母。

最后填写 reviewer.json 的真实代号、带时区完成时间和三个声明 true。
两名评审的分歧由第三名真人另包裁决，不自动投票或把 unknown 改成通过。
每人192全文事实/事件、96配对，以及本包列明数量的共同候选。admin 文件不在本包内。
"""


def prepare():
    import spacy
    if (OUT / "FREEZE.json").exists():
        verify()
        return {"status": "ALREADY_PREPARED", "new_amr_or_embedding_calls": 0}
    if PRIVATE.exists() or (OUT / "private_review").exists():
        raise RuntimeError("Partial preparation exists; inspect before creating a new version")
    v2.runtime.verify(v2.OUT, v2.PRIVATE)
    if not (OUT / "PROTOCOL.md").is_file():
        raise RuntimeError("Reviewable protocol required")
    review = v2.OUT / "private_review"
    keys = load(review / "admin/KEYS.json")
    source_manifest = load(review / "admin/MANIFEST.json")
    if sha(review / "admin/KEYS.json") != source_manifest["keys_sha256"]:
        raise RuntimeError("Original administrator key seal mismatch")
    plan = {r["row"]["item_id"]: r["row"] for r in load(v2.PRIVATE / "PLAN.json")}
    nlp = spacy.load("en_core_web_sm")
    documents = {}
    original_sha = {}
    for item, row in plan.items():
        path = v2.PRIVATE / "documents" / (item + ".json")
        original_sha[str(path)] = sha(path)
        data = load(path)
        for condition in ("original", "paraphrase"):
            key = hashlib.sha256(f"supervision2:{item}:{condition}".encode()).hexdigest()[:20]
            text = data[condition]["text"]
            extracted = context_candidates(text, nlp)
            documents[(item, condition)] = {
                "key": key, "scene_id": row["scene_id"], "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "contract": row["public_contract"], "candidates": extracted["observations"],
                "candidate_status": extracted["status"],
                "observed_anchor_count": extracted["anchor_count"],
                "planned_symbol": row["intended_symbol"], "condition": condition}
    groups = split_scenes(sorted({r["scene_id"] for r in plan.values()}))
    atomic(OUT / "SPLITS.json", {"schema": VERSION, "seed": 20261026, "groups": groups,
                                "unit": "scene/all four symbols and both variants",
                                "heldout_role": "development_test_not_independent_confirmation"})
    atomic(OUT / "MEMBERSHIP.json", {d["key"]: d["scene_id"] for d in documents.values()})
    packet_keys, files = {}, {}
    facts_immutable = ["item_id", "role_one", "role_two", "public_object", "text"]
    candidate_fields = ["item_id", "fact_item_id", "anchor_start", "anchor_end", "anchor_text",
                        "context_1", "context_2", "context_3",
                        "kind", "event_ids_json", "evidence_json", "notes"]
    for reviewer in (1, 2):
        tag = str(reviewer)
        folder = OUT / "private_review" / f"reviewer{reviewer}"
        folder.mkdir(parents=True, mode=0o700)
        packet_keys[tag] = {"facts": {}, "candidates": {}, "pairs": {}}
        fact_rows, candidate_rows = [], []
        for row in csv_rows(review / f"reviewer{reviewer}/facts.csv"):
            identity = keys[tag]["facts"][row["item_id"]]["identity"]
            doc = documents[(identity["item"], identity["condition"])]
            immutable = {f: row[f] for f in facts_immutable}
            if immutable["text"] != doc["text"]:
                raise RuntimeError("Original text differs from generation seal")
            fresh = {**immutable, **dict.fromkeys(v2.runtime.FACT_LABELS, ""),
                     "inventory_status": "", "events_json": ""}
            fact_rows.append(fresh)
            packet_keys[tag]["facts"][row["item_id"]] = {"key": doc["key"], "immutable": immutable}
            for i, candidate in enumerate(doc["candidates"]):
                cid = hashlib.sha256(f"supervision2:{tag}:{row['item_id']}:{i}".encode()).hexdigest()[:20]
                a, b = candidate["anchor_span"]
                payload = {"item_id": cid, "fact_item_id": row["item_id"],
                           "anchor_start": str(a), "anchor_end": str(b),
                           "anchor_text": candidate["anchor_text"],
                           **{f"context_{s+1}": x["text"] for s, x in enumerate(candidate["contexts"])}}
                candidate_rows.append({**payload, **dict.fromkeys(candidate_fields[-4:], "")})
                packet_keys[tag]["candidates"][cid] = {"key": doc["key"], "index": i, "immutable": payload}
        pairs = []
        for row in csv_rows(review / f"reviewer{reviewer}/pairs.csv"):
            immutable = {k: row[k] for k in ("item_id", "source", "paraphrase")}
            identity = keys[tag]["pairs"][row["item_id"]]["identity"]
            pair_keys = [documents[(identity["item"], c)]["key"] for c in ("original", "paraphrase")]
            packet_keys[tag]["pairs"][row["item_id"]] = {"keys": pair_keys, "immutable": immutable}
            pairs.append({**immutable, **dict.fromkeys(v2.runtime.PAIR_LABELS, "")})
        for offset, rows in enumerate((fact_rows, pairs, candidate_rows)):
            random.Random(20261026 + reviewer * 10 + offset).shuffle(rows)
        write_csv(folder / "facts.csv", fact_rows)
        write_csv(folder / "pairs.csv", pairs)
        write_csv(folder / "candidates.csv", candidate_rows, candidate_fields)
        (folder / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
        atomic(folder / "reviewer.json", reviewer_metadata())
        atomic(folder / "SCHEMA.json", {
            "event_fields": ["event_id", "anchor", "meaning", "kind", *v2.runtime.FIELDS,
                             "readability", "explicit_status", "evidence", "background_reason", "notes"],
            "anchor_and_evidence_span": {"start": "integer", "end": "integer", "quote": "literal text"},
            "inventory_kinds": ["carrier", "background", "unknown"],
            "candidate_kinds": ["carrier", "background", "duplicate", "unknown"],
            "evidence_fields": list(v2.runtime.FIELDS), "no_filled_label_examples": True})
        (folder / "text_words.txt").write_text(word_index(fact_rows))
        for path in folder.iterdir():
            path.chmod(0o600)
            files[str(path.relative_to(OUT / "private_review"))] = sha(path)
        zip_path = folder.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                archive.write(path, path.name)
        zip_path.chmod(0o600)
        files[str(zip_path.relative_to(OUT / "private_review"))] = sha(zip_path)
    atomic(PRIVATE / "DOCUMENTS.json", {d["key"]: d for d in documents.values()})
    atomic(PRIVATE / "KEYS.json", packet_keys)
    # Inference request carries no target, actual facts or candidate gold.
    atomic(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json", {
        "alphabet_version": load(v2.OUT / "ALPHABET.json")["version"],
        "views": ["R1_full", "R2_full", "R3_core"],
        "core_fact_matcher_implemented": False, "parser_calls": 0,
        "documents": [{k: d[k] for k in ("key", "text", "text_sha256", "contract", "candidates",
                                        "candidate_status")} for d in documents.values()]})
    protected = {}
    for folder in (ROOT / "experiments/semantic_multiscale_readback1", v2.OUT):
        for path in folder.rglob("*"):
            if path.is_file() and "private_review" not in path.parts and path.name != "heartbeat.json":
                protected[str(path.relative_to(ROOT))] = sha(path)
    for reviewer in (1, 2):
        for name in ("facts.csv", "pairs.csv", "reviewer.json"):
            path = review / f"reviewer{reviewer}" / name
            protected[str(path.relative_to(ROOT))] = sha(path)
    count = sum(len(d["candidates"]) for d in documents.values())
    atomic(OUT / "FREEZE.json", {
        "at": datetime.now(timezone.utc).isoformat(), "version": VERSION,
        "protocol_sha256": sha(OUT / "PROTOCOL.md"), "split_sha256": sha(OUT / "SPLITS.json"),
        "membership_sha256": sha(OUT / "MEMBERSHIP.json"),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "private_sha256": {name: sha(PRIVATE / name) for name in
                           ("DOCUMENTS.json", "KEYS.json", "FEATURE_REQUEST_NO_TARGET.json")},
        "input_sha256": original_sha, "protected_sha256": protected, "blank_packet_sha256": files,
        "counts": {"scenes": 24, "facts": 192, "pairs": 96, "candidates": count},
        "candidate_extractor": {"spacy": spacy.__version__, "model": nlp.meta.get("version"),
                                "anchor": "non-AUX VERB", "max_anchors": 32,
                                "views": ["R1_full", "R2_full", "R3_core"],
                                "over_budget": "whole_document_abstain"},
        "amr_or_embedding_inference_executed": False,
    })
    atomic(OUT / "STATUS.json", {"status": "PREPARED_PENDING_HUMANS_AND_FACT_FEATURES",
                               "completed_humans": 0, "gold": False, "training_runs": 0,
                               "new_parser_calls": 0, "counts_per_reviewer": load(OUT / "FREEZE.json")["counts"]})
    return {"status": "PREPARED", "facts_per_reviewer": 192, "pairs_per_reviewer": 96,
            "candidates_per_reviewer": count, "human_labels_completed": 0, "parser_calls": 0}


def verify():
    freeze = load(OUT / "FREEZE.json")
    if sha(OUT / "PROTOCOL.md") != freeze["protocol_sha256"] or sha(OUT / "SPLITS.json") != freeze["split_sha256"]:
        raise RuntimeError("Protocol/split drift")
    if sha(OUT / "MEMBERSHIP.json") != freeze["membership_sha256"]:
        raise RuntimeError("Scene membership drift")
    for name, expected in freeze["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Source drift: " + name)
    for name, expected in freeze["private_sha256"].items():
        if sha(PRIVATE / name) != expected:
            raise RuntimeError("Private input/mapping drift")
    for name, expected in freeze["input_sha256"].items():
        if sha(name) != expected:
            raise RuntimeError("Generation input drift")
    for name, expected in freeze["protected_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Protected historical artifact drift: " + name)
    # Original blank ZIPs are archival; returned CSVs are allowed to change.
    for name, expected in freeze["blank_packet_sha256"].items():
        if name.endswith(".zip") and sha(OUT / "private_review" / name) != expected:
            raise RuntimeError("Original blank ZIP changed")
    return freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else
                     {"status": "PASS", "protected_files": len(verify()["protected_sha256"])},
                     ensure_ascii=False))
