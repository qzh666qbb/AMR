"""Read-only aggregate audit of the completed old S6 candidate parses."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_swan_blind_eval import save_json
from scripts.prepare_swan_blind_definitions import EXP, ATTACKS, load_attack_functions
from src.core.swan_blind_protocol import ProtocolConfig, digest, plan_document
from src.core.swan_blind_runtime import graph_valid, text_sha


def category(message):
    value = message.casefold()
    if "not a new concept" in value:
        return "start_paren_non_new_concept"
    if "initial node" in value and ("ignor" in value or "token" in value):
        return "initial_node_token_ignored"
    if "missing starting paren" in value:
        return "missing_starting_paren"
    if "unhandled token" in value:
        return "unhandled_token"
    if "epigraph" in value:
        return "duplicate_epigraph_data"
    if "destack" in value:
        return "destack_past_top"
    if "replacing attrib" in value:
        return "attribute_replaced"
    return "other_bound_diagnostic"


def main():
    frozen = json.loads((EXP / "blind_eval1/run2/FREEZE.json").read_text())
    config = ProtocolConfig(**frozen["configurations"]["S6"])
    source = {arm: json.loads((EXP / name / "documents.json").read_text())["documents"]
              for arm, name in (("positive", "dataset5"),
                                ("paired_negative", "dataset5_paired_neg"))}
    calibration = set(frozen["calibration_ids"])
    attacks = load_attack_functions()
    db = sqlite3.connect("file:/tmp/swan_blind_eval1_run2/cache.sqlite?mode=ro", uri=True)
    rows, rejected = [], set()
    for arm, docs in source.items():
        for attack in ATTACKS:
            for split in ("calibration", "test"):
                subset = [d for d in docs if
                          (d["document_id"] in calibration) == (split == "calibration")]
                doc_reasons, events, failures, bad_docs = Counter(), Counter(), set(), 0
                normalized_valid = 0
                for d in subset:
                    text, _ = attacks[attack]([{"text": s["text"]} for s in d["slots"]])
                    plan = plan_document(text, config)
                    records = []
                    for c in plan.candidates:
                        key = text_sha(c.text)
                        value = db.execute("SELECT value FROM parses WHERE key=?", (key,)).fetchone()
                        if value is None:
                            raise RuntimeError("Completed S6 cache missing a candidate")
                        r = json.loads(value[0])
                        if not r["strict_ok"]:
                            records.append((key, r))
                    if records:
                        bad_docs += 1
                        doc_reasons.update({reason for _, r in records for reason in r["reasons"]})
                    for key, r in records:
                        rejected.add(key)
                        if key in failures:
                            continue
                        failures.add(key)
                        normalized_valid += int(bool(r.get("graph")) and graph_valid(r["graph"]))
                        selected = [e for e in r.get("events", [])
                                    if e.get("beam") == r.get("selected_beam")]
                        selected += r.get("normalization_events", [])
                        events.update(category(e.get("message", "")) for e in selected)
                rows.append({
                    "arm": arm, "attack": attack, "split": split,
                    "source_documents": len(subset), "parser_rejected_documents": bad_docs,
                    "unique_rejected_candidates": len(failures),
                    "normalized_structurally_valid_candidates": normalized_valid,
                    "document_failure_reasons": dict(doc_reasons),
                    "selected_or_normalization_event_categories": dict(events),
                })
    result = {"schema": "swan-last-round-cache-audit/v1", "read_only_cache": True,
              "old_freeze_sha256": digest(frozen), "method": "S6", "groups": rows,
              "unique_rejected_inputs": len(rejected), "new_detector_calls": 0,
              "parser_policy_changed": False, "raw_tokens_or_graphs_exported": False}
    save_json(EXP / "abstention_audit1/OLD_S6_CACHE_AUDIT.json", result)
    print(json.dumps({"old_positive_newline_test": next(
        r for r in rows if r["arm"] == "positive" and r["attack"] == "newline"
        and r["split"] == "test")}), flush=True)


if __name__ == "__main__":
    main()
