"""New normal-length, score-blind source cohorts for independent-key experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import swan_v4_confirmation1_sources as history
from swan_final1_common import freeze, load, now, sha256
from src.core.length_boundary_validation import hard_units, END_MARK_RE

OUT = ROOT / "experiments/swan_independent_banks1"
ROLES = {"public_bank": 100, "target_a_bank": 100, "target_b_bank": 100,
         "calibration": 400, "null": 400, "pilot": 12, "main": 60}
SHARDS = (420, 421, 422, 423)
CODE = ("scripts/swan_independent_banks1_sources.py",
        "scripts/swan_v4_confirmation1_sources.py", "scripts/swan_final1_sources.py",
        "scripts/swan_final1_common.py", "src/core/length_boundary_validation.py",
        "src/core/sqlite_cache.py")


def window(text):
    units = hard_units(text)
    for start in range(len(units)):
        selected = []
        for sentence in units[start:]:
            if not END_MARK_RE.search(sentence):
                break
            selected.append(sentence)
            n = sum(len(s.split()) for s in selected)
            if n > 240:
                break
            if n >= 150:
                result = " ".join(selected)
                letters = [c for c in result if c.isalpha()]
                if (letters and sum(c.isascii() for c in letters) / len(letters) >= .95
                        and not re.search(r"https?://|www\.", result, re.I)):
                    return result
    return None


def candidate(raw, shard, index, bags):
    text, url = raw.get("text"), raw.get("url")
    if not isinstance(text, str) or not isinstance(url, str) or not url.strip():
        return None
    prompt = history.source.official_prompt(text)
    locator = f"realnewslike/c4-train.{shard:05d}-of-00512.json.gz:{index}"
    text_ids = {history.raw_sha(text), history.source.identity(text),
                history.source.identity(prompt)}
    url_ids = {history.source.url_identity(url), history.source.identity(url),
               history.raw_sha(" ".join(url.split()))}
    if text_ids & bags["texts"] or url_ids & bags["urls"] or locator in bags["ids"]:
        return None
    chosen = window(text)
    if chosen is None or {history.raw_sha(chosen), history.source.identity(chosen)} & bags["texts"]:
        return None
    return {"source_text": text, "source_url": url, "text": chosen,
            "dataset_locator": locator, "dataset_revision": history.source.REVISION,
            "source_sha256": history.raw_sha(text),
            "normalized_source_sha256": history.source.identity(text),
            "normalized_prompt_sha256": history.source.identity(prompt),
            "url_sha256": history.source.url_identity(url),
            "window_sha256": history.raw_sha(chosen),
            "normalized_window_sha256": history.source.identity(chosen)}


def allocate(rows):
    if len(rows) != sum(ROLES.values()):
        raise ValueError("Complete fixed pool required")
    ordered = list(rows)
    random.Random(20261102).shuffle(ordered)
    result, cursor = {}, 0
    for role, count in ROLES.items():
        result[role] = []
        for i, row in enumerate(ordered[cursor:cursor + count]):
            result[role].append({
                **{k: v for k, v in row.items() if k not in ("source_text", "source_url")},
                "role": role, "allocation_id": f"independent-banks1-{role}-{i:04d}",
                "word_count": len(row["text"].split()),
            })
        cursor += count
    return result


def initialize():
    if (OUT / "SOURCE_CONTRACT.json").exists():
        return verify()
    (OUT / "private").mkdir(parents=True, mode=0o700)
    freeze(OUT / "private/HISTORY.json", history.make_history())
    # Private random key selection is committed before any new parsing/scoring.
    freeze(OUT / "private/KEY_SELECTION.json",
           {role: os.urandom(32).hex() for role in ROLES if role.endswith("_bank")})
    freeze(OUT / "SOURCE_CONTRACT.json", {
        "version": "independent-key-normal-length-sources/v1", "at": now(),
        "roles": ROLES, "shards": SHARDS, "rows_per_shard": 100000,
        "word_range": [150, 240], "role_shuffle_seed": 20261102,
        "dataset_revision": history.source.REVISION,
        "history_sha256": sha256(OUT / "private/HISTORY.json"),
        "key_selection_commitment": sha256(OUT / "private/KEY_SELECTION.json"),
        "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
        "code_sha256": {p: sha256(ROOT / p) for p in CODE},
        "score_filtering": False, "replacement": False, "paid_calls": 0,
    })
    return verify()


def verify():
    f = load(OUT / "SOURCE_CONTRACT.json")
    pairs = [(ROOT / p, h) for p, h in f["code_sha256"].items()]
    pairs += [(OUT / "private/HISTORY.json", f["history_sha256"]),
              (OUT / "private/KEY_SELECTION.json", f["key_selection_commitment"]),
              (OUT / "PROTOCOL.md", f["protocol_sha256"])]
    if any(sha256(p) != h for p, h in pairs):
        raise RuntimeError("Frozen source contract changed")
    if f["roles"] != ROLES or f["shards"] != list(SHARDS):
        raise RuntimeError("Frozen cohort changed")
    return f


def acquire():
    verify()
    if (OUT / "INPUT_MANIFEST.json").exists():
        return audit()
    bags = {k: set(v) for k, v in load(OUT / "private/HISTORY.json")["identities"].items()}
    journal = OUT / "private/SOURCE_POOL.jsonl"
    raw = journal.read_bytes() if journal.exists() else b""
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("Partial journal: requires explicit recovery")
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        history.collect(row, bags)
    scans = []
    for shard in SHARDS:
        if len(rows) == sum(ROLES.values()):
            break
        for index, raw in history.source.stream_shard(shard):
            row = candidate(raw, shard, index, bags)
            if row:
                with journal.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                journal.chmod(0o600)
                rows.append(row)
                history.collect(row, bags)
            if index % 100 == 0 or len(rows) == sum(ROLES.values()):
                from run_multiscale_transfer import atomic
                atomic(OUT / "SOURCE_HEARTBEAT.json", {
                    "at": now(), "completed": len(rows), "total": sum(ROLES.values()),
                    "shard": shard, "last_row": index, "detector_calls": 0})
            if len(rows) == sum(ROLES.values()):
                break
        scans.append({"shard": shard, "last_row": index})
    allocated = allocate(rows)
    for role, items in allocated.items():
        freeze(OUT / "private/cohorts" / (role + ".json"), items)
    freeze(OUT / "INPUT_MANIFEST.json", {
        "source_contract_sha256": sha256(OUT / "SOURCE_CONTRACT.json"),
        "source_pool_sha256": sha256(journal), "scans": scans,
        "cohorts": {role: {"count": len(items),
                          "sha256": sha256(OUT / "private/cohorts" / (role + ".json"))}
                    for role, items in allocated.items()},
    })
    return audit()


def audit():
    verify()
    manifest = load(OUT / "INPUT_MANIFEST.json")
    if (manifest["source_contract_sha256"] != sha256(OUT / "SOURCE_CONTRACT.json")
            or manifest["source_pool_sha256"] != sha256(OUT / "private/SOURCE_POOL.jsonl")):
        raise RuntimeError("Input seal changed")
    bags = {k: set(v) for k, v in load(OUT / "private/HISTORY.json")["identities"].items()}
    rows = [json.loads(line) for line in (OUT / "private/SOURCE_POOL.jsonl").read_text().splitlines()]
    for row in rows:
        match = re.fullmatch(r"realnewslike/c4-train\.(\d{5})-of-00512\.json\.gz:(\d+)",
                             row["dataset_locator"])
        if not match or int(match[1]) not in SHARDS or int(match[2]) >= 100000:
            raise RuntimeError("Source outside frozen scan")
        if candidate({"text": row["source_text"], "url": row["source_url"]},
                     int(match[1]), int(match[2]), bags) != row:
            raise RuntimeError("Source exclusion cannot replay")
        history.collect(row, bags)
    allocated = allocate(rows)
    if set(manifest["cohorts"]) != set(ROLES):
        raise RuntimeError("Incomplete cohort set")
    for role, items in allocated.items():
        path = OUT / "private/cohorts" / (role + ".json")
        if (load(path) != items or sha256(path) != manifest["cohorts"][role]["sha256"]
                or len(items) != manifest["cohorts"][role]["count"]):
            raise RuntimeError("Allocation changed")
    result = {"status": "SOURCE_IDENTITIES_VERIFIED", "sources": len(rows),
              "counts": dict(Counter(r["role"] for items in allocated.values() for r in items)),
              "historical_and_cross_cohort_exact_identity_overlaps": 0,
              "semantic_near_duplicate_exclusion_proven": False,
              "source_manifest_sha256": sha256(OUT / "INPUT_MANIFEST.json"),
              "actual_bank_templates": None, "calibration_complete": False,
              "attack_trials_complete": False}
    freeze(OUT / "SOURCE_AUDIT.json", result)
    return result


if __name__ == "__main__":
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("initialize", "acquire", "audit"))
    args = ap.parse_args()
    with (OUT / "source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(globals()[args.action](), default=str))
