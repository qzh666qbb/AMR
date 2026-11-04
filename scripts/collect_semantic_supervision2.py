"""Validate two real humans and blind adjudication; never auto-complete labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from prepare_semantic_supervision2 import (
    OUT, PRIVATE, v2, atomic, csv_rows, digest, load, reviewer_metadata,
    sha, verify, word_index, write_csv, INSTRUCTIONS,
)
from src.core.semantic_supervision import (
    FACT_FIELDS, ROLES, event_symbol, normalize_document, require_human, validate_inventory,
)


def parse_packet(folder, mapping, documents):
    facts, pairs, candidates = {}, {}, {}
    for task in ("facts", "pairs", "candidates"):
        rows = csv_rows(folder / (task + ".csv"))
        if len(rows) != len(mapping[task]):
            raise ValueError("Missing/additional rows in " + task)
        seen = set()
        for row in rows:
            item_id = row["item_id"]
            if item_id not in mapping[task] or item_id in seen:
                raise ValueError("Missing, edited or repeated item id")
            seen.add(item_id)
            ref = mapping[task][item_id]
            if any(row.get(k) != value for k, value in ref["immutable"].items()):
                raise ValueError("Read-only public text or identity edited")
            if task == "facts":
                v2.runtime.validate_fact(row)
                fact_evidence = json.loads(row["evidence_json"])
                if any(row[k] == "unknown" and fact_evidence[k] for k in FACT_FIELDS):
                    raise ValueError("Unknown document fact must have empty evidence")
                events = json.loads(row["events_json"])
                doc = documents[ref["key"]]
                validate_inventory(doc["text"], doc["contract"], events, row["inventory_status"])
                actual = v2.runtime.fact_symbol(row, doc["contract"])
                if actual is not None and not any(
                        event_symbol(e, doc["contract"]) == actual
                        and all(e[k].strip().casefold() == row[k].strip().casefold()
                                for k in FACT_FIELDS) for e in events):
                    raise ValueError("Document facts disagree with independently listed carrier events")
                facts[ref["key"]] = {**row, "events": events}
            elif task == "pairs":
                v2.runtime.validate_pair(row)
                pairs[tuple(ref["keys"])] = row
            else:
                candidates.setdefault(ref["key"], {})[ref["index"]] = {
                    "kind": row["kind"], "event_ids": json.loads(row["event_ids_json"]),
                    "evidence": json.loads(row["evidence_json"]), "notes": row["notes"],
                    "anchor_span": [int(row["anchor_start"]), int(row["anchor_end"])]}
    normalized = {}
    for key, fact in facts.items():
        doc = documents[key]
        indexed = candidates.get(key, {})
        if set(indexed) != set(range(len(doc["candidates"]))):
            raise ValueError("Every shared candidate needs a human decision")
        ordered = [indexed[i] for i in range(len(indexed))]
        normal = normalize_document(doc, fact, ordered)
        normalized[key] = {"fact": fact, "candidates": ordered, "normalized": normal}
    return normalized, pairs


def signature_document(record):
    """Ignore local event-id spelling, evidence wording and free-form notes."""
    fact = record["fact"]
    anchors = {e["event_id"]: [e["anchor"]["start"], e["anchor"]["end"]] for e in fact["events"]}
    inventory = [{k: e[k] for k in (*FACT_FIELDS, "kind", "readability", "explicit_status")}
                 | {"anchor": anchors[e["event_id"]]} for e in fact["events"]]
    inventory.sort(key=lambda e: (e["anchor"], digest(e)))
    candidates = [{"kind": c["kind"], "links": sorted(anchors[e] for e in c["event_ids"])}
                  for c in record["candidates"]]
    return digest({"fact": {k: fact[k].strip().casefold() for k in
                            (*FACT_FIELDS, "readability", "explicit_status", "quality")},
                   "status": fact["inventory_status"], "events": inventory,
                   "candidates": candidates})


def signature_pair(row):
    return tuple(row[k] for k in v2.runtime.PAIR_LABELS[:3])


def make_adjudication(keys, disputed_docs, disputed_pairs):
    folder = OUT / "private_review/adjudicator"
    mapping_path = PRIVATE / "ADJUDICATION_KEYS.json"
    identity = digest({"docs": sorted(disputed_docs), "pairs": sorted(disputed_pairs)})
    if mapping_path.exists():
        mapping = load(mapping_path)
        if mapping["dispute_identity"] != identity:
            raise RuntimeError("Returns changed after adjudication release; use a new collection version")
        return folder, mapping["mapping"]
    mapping = {task: {} for task in ("facts", "pairs", "candidates")}
    rows = {task: [] for task in mapping}
    for task in mapping:
        original = csv_rows(OUT / "private_review/reviewer1" / (task + ".csv"))
        for row in original:
            ref = keys["1"][task][row["item_id"]]
            included = (tuple(ref["keys"]) in disputed_pairs if task == "pairs"
                        else ref["key"] in disputed_docs)
            if not included:
                continue
            # Third human receives the immutable payload only, neither prior answer.
            alias = "a-" + row["item_id"]
            immutable = {**ref["immutable"], "item_id": alias}
            if task == "candidates":
                immutable["fact_item_id"] = "a-" + immutable["fact_item_id"]
            blank = dict.fromkeys(row, "")
            rows[task].append({**blank, **immutable})
            mapping[task][alias] = {**ref, "immutable": immutable}
    folder.mkdir(parents=True, mode=0o700)
    for task, packet in rows.items():
        fields = list(csv_rows(OUT / "private_review/reviewer1" / (task + ".csv"))[0])
        write_csv(folder / (task + ".csv"), packet, fields)
    (folder / "INSTRUCTIONS.md").write_text(INSTRUCTIONS + "\n本包仅含需要第三人独立裁决的全文与配对。\n")
    atomic(folder / "reviewer.json", reviewer_metadata())
    shutil.copy2(OUT / "private_review/reviewer1/SCHEMA.json", folder / "SCHEMA.json")
    (folder / "text_words.txt").write_text(word_index(rows["facts"]))
    atomic(mapping_path, {"dispute_identity": identity, "mapping": mapping})
    return folder, mapping


def gate(records, pairs, documents):
    scene_values = {scene: {"writing": [], "paraphrase": [], "pair": []}
                    for scene in sorted({d["scene_id"] for d in documents.values()})}
    for key, record in records.items():
        doc = documents[key]
        fact = record["fact"]
        actual = v2.runtime.fact_symbol(fact, doc["contract"])
        axis = "writing" if doc["condition"] == "original" else "paraphrase"
        scene_values[doc["scene_id"]][axis].append(int(actual == doc["planned_symbol"]))
    for (source, _), row in pairs.items():
        scene_values[documents[source]["scene_id"]]["pair"].append(
            int(all(row[k] == "yes" for k in v2.runtime.PAIR_LABELS[:3])))
    result = {}
    for axis in ("writing", "paraphrase", "pair"):
        clustered = []
        for row in scene_values.values():
            if len(row[axis]) != 4:
                raise ValueError("Frozen 96/24 gate denominator changed")
            clustered.append(sum(row[axis]) / 4)
        interval = v2.runtime.bootstrap(clustered)
        passed = sum(clustered) / 24 >= .9 and interval[0] >= .8
        result[axis] = {"correct": round(sum(clustered) * 4), "n": 96,
                        "rate": sum(clustered) / 24, "interval": interval, "pass": passed}
    result["status"] = "PASS" if result["writing"]["pass"] and result["pair"]["pass"] else "STOP"
    result["rule"] = "unchanged frozen writing and pair 90% / clustered CI lower 80%"
    return result


def collect():
    freeze = verify()
    keys = load(PRIVATE / "KEYS.json")
    documents = load(PRIVATE / "DOCUMENTS.json")
    seal = PRIVATE / "HUMAN_GOLD_SEAL.json"
    if seal.exists():
        prior = load(seal)
        if sha(PRIVATE / "HUMAN_GOLD.json") != prior["gold_sha256"]:
            raise RuntimeError("Sealed human gold changed")
        for path, expected in prior["return_sha256"].items():
            if sha(path) != expected:
                raise RuntimeError("Returns changed after human gold seal")
        return {"status": "ALREADY_SEALED", "gold_sha256": prior["gold_sha256"]}
    results, pair_results, codes, errors, returned_sha = {}, {}, [], [], {}
    for reviewer in (1, 2):
        folder = OUT / "private_review" / f"reviewer{reviewer}"
        try:
            code = require_human(load(folder / "reviewer.json"), codes)
            normalized, pairs = parse_packet(folder, keys[str(reviewer)], documents)
            results[str(reviewer)], pair_results[str(reviewer)] = normalized, pairs
            codes.append(code)
            for name in ("facts.csv", "pairs.csv", "candidates.csv", "reviewer.json"):
                path = folder / name
                returned_sha[str(path)] = sha(path)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append({"reviewer": reviewer, "reason": str(error)})
    if errors:
        outcome = {"status": "PENDING_HUMAN_RETURNS", "accepted_complete_reviewers": len(results),
                   "gold": False, "errors": errors, "counts": freeze["counts"], "training_runs": 0}
        atomic(OUT / "COLLECTION_STATUS.json", outcome)
        return outcome
    disputes = {k for k in documents if signature_document(results["1"][k])
                != signature_document(results["2"][k])}
    pair_disputes = {k for k in pair_results["1"] if signature_pair(pair_results["1"][k])
                     != signature_pair(pair_results["2"][k])}
    resolved, resolved_pairs = dict(results["1"]), dict(pair_results["1"])
    if disputes or pair_disputes:
        folder, mapping = make_adjudication(keys, disputes, pair_disputes)
        try:
            require_human(load(folder / "reviewer.json"), codes)
            third, third_pairs = parse_packet(folder, mapping,
                                               {k: documents[k] for k in disputes})
            resolved.update(third)
            resolved_pairs.update(third_pairs)
            for name in ("facts.csv", "pairs.csv", "candidates.csv", "reviewer.json"):
                path = folder / name
                returned_sha[str(path)] = sha(path)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            outcome = {"status": "PENDING_HUMAN_ADJUDICATION", "gold": False,
                       "document_disputes": len(disputes), "pair_disputes": len(pair_disputes),
                       "reason": str(error)}
            atomic(OUT / "COLLECTION_STATUS.json", outcome)
            return outcome
    carrier_gate = gate(resolved, resolved_pairs, documents)
    gold = {}
    for key, record in resolved.items():
        doc = documents[key]
        # Target truth is in label storage only; feature files never carry it.
        gold[key] = {**record["normalized"], "planned_symbol": doc["planned_symbol"],
                     "scene_id": doc["scene_id"], "condition": doc["condition"]}
    atomic(PRIVATE / "HUMAN_GOLD.json", gold)
    atomic(PRIVATE / "RESOLVED_HUMAN_EVIDENCE.json", {
        "documents": resolved, "pairs": [{"keys": list(k), "labels": v} for k, v in resolved_pairs.items()]})
    groups = load(OUT / "SPLITS.json")["groups"]
    label_sha = {role: {} for role in ROLES}
    for key, record in gold.items():
        role = next(role for role in ROLES if record["scene_id"] in groups[role])
        path = PRIVATE / "labels" / role / (key + ".json")
        atomic(path, {**record, "role": role})
        label_sha[role][key] = sha(path)
    ready = carrier_gate["status"] == "PASS" and all(
        r["inventory_status"] in ("complete", "unavailable") for r in gold.values())
    atomic(seal, {"schema": freeze["version"], "gold_sha256": sha(PRIVATE / "HUMAN_GOLD.json"),
                  "evidence_sha256": sha(PRIVATE / "RESOLVED_HUMAN_EVIDENCE.json"),
                  "return_sha256": returned_sha, "real_humans": True, "independent_reviewers": 2,
                  "disputes_adjudicated": bool(disputes or pair_disputes),
                  "carrier_gate": carrier_gate, "training_labels_ready": ready,
                  "role_label_sha256": label_sha,
                  "parent_freeze_sha256": sha(OUT / "FREEZE.json")})
    outcome = {"status": "HUMAN_GOLD_SEALED" if ready else "HUMAN_GOLD_SEALED_NOT_READY",
               "gold": True, "carrier_gate": carrier_gate,
               "unresolved_inventories": sum(r["inventory_status"] == "unknown" for r in gold.values()),
               "unknown_candidates": sum(c["kind"] == "unknown" for r in gold.values() for c in r["candidate_labels"]),
               "missing_events": sum(r["missing_events"] for r in gold.values()),
               "gold_seal_sha256": sha(seal), "training_runs": 0}
    atomic(OUT / "COLLECTION_STATUS.json", outcome)
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect",))
    parser.parse_args()
    print(json.dumps(collect(), ensure_ascii=False, indent=2))
