"""Score-blind disjoint C4 sources for a new fixed-flow v4 confirmation."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import swan_final1_sources as source
from swan_final1_common import freeze, load, now, sha256
from src.core.length_boundary_validation import END_MARK_RE, hard_units, punctuation_variant

OUT = ROOT / "experiments/swan_v4_independent_confirmation1"
BINS = ((17, 24), (25, 32), (33, 40), (41, 48))
COUNTS = {"calibration": (250,) * 4, "null": (250,) * 4,
          "positive_source": (63, 63, 62, 62)}
SHARDS = (400, 401, 402, 403)
SEED = 20261102
SNAPSHOTS = (
    "experiments/swan_count_neutral_supplement4/prior_identity_snapshot.json",
    "experiments/swan_path0_completion1/LOCAL_HISTORY.json",
    "experiments/swan_path0_completion1/REMOTE_HISTORY.json",
)
CODE = ("scripts/swan_v4_confirmation1_sources.py",
        "scripts/test_swan_v4_confirmation1_sources.py", "scripts/swan_final1_sources.py",
        "scripts/swan_final1_common.py", "src/core/length_boundary_validation.py",
        "src/core/sqlite_cache.py")
HASH_FIELDS = source.HASH_KEYS | {
    "source_sha256", "text_sha256", "window_sha256", "normalized_window_sha256",
    "generation_prompt_sha256", "source_text_sha256",
}


def raw_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def empty_bags():
    return {"texts": set(), "urls": set(), "ids": set()}


def collect(value, bags, key=""):
    if isinstance(value, dict):
        for name, item in value.items():
            collect(item, bags, name.casefold())
    elif isinstance(value, list):
        for item in value:
            collect(item, bags, key)
    elif isinstance(value, str) and value.strip():
        if key in source.TEXT_KEYS:
            bags["texts"].update((source.identity(value), raw_sha(value),
                                  source.identity(source.official_prompt(value))))
        if key in HASH_FIELDS and re.fullmatch(r"[a-f0-9]{64}", value):
            bags["texts"].add(value)
        if key in ("url", "source_url"):
            bags["urls"].update((source.url_identity(value), source.identity(value),
                                raw_sha(" ".join(value.split()))))
        if key == "url_sha256":
            bags["urls"].add(value)
        if key in ("dataset_locator", "locator", "allocation_id"):
            bags["ids"].add(value)


def make_history():
    """Reuse provenance snapshots plus registered current input/source files."""
    bags, inventory = empty_bags(), []
    for name in SNAPSHOTS:
        path = ROOT / name
        data = load(path)
        if "identities" in data:
            identities = data["identities"]
            bags["texts"].update(identities["texts"])
            bags["texts"].update(identities["raw_hashes"])
            bags["urls"].update(identities["urls"])
            bags["ids"].update(identities["ids"])
        else:
            bags["texts"].update(data["normalized_text_sha256"])
            bags["urls"].update(data["url_sha256"])
        inventory.append({"path": name, "sha256": sha256(path), "kind": "existing_history_snapshot"})
    roots = (ROOT / "experiments", ROOT / "archive",
             ROOT / "第二篇_语义消息写入与读回/02_实验",
             ROOT / "第二篇_语义消息写入与读回/08_本地原始资产/cache_amr")
    for root in roots:
        for path in sorted(root.rglob("*")):
            if (not path.is_file() or path.is_symlink() or OUT in path.parents
                    or path.name.startswith("._") or path.suffix not in (".json", ".jsonl")
                    or source.SECRET_PATH.search(str(path))):
                continue
            # Supplement frozen historical identities with source cohorts and
            # target-free input registries; provider responses are unnecessary.
            relevant = (
                "sealed" in path.parts and re.search(r"source|calibration|null|test|pilot|main", path.name, re.I)
                or path.name in ("FEATURE_REQUEST_NO_TARGET.json", "INPUTS_NO_TARGET.json",
                                 "INPUT_NO_TARGET.json", "allocation_journal.json", "DOCUMENTS.json")
            )
            if not relevant:
                continue
            with path.open(encoding="utf-8-sig") as stream:
                if path.suffix == ".jsonl":
                    for line in stream:
                        if line.strip():
                            collect(json.loads(line), bags)
                else:
                    collect(json.load(stream), bags)
            inventory.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                              "kind": "registered_source_or_blind_input"})
    return {
        "at": now(), "identities": {k: sorted(v) for k, v in bags.items()},
        "inventory": inventory,
        "scope": "three frozen local/remote history snapshots plus registered current sources and blind inputs",
        "claim": "exact identity exclusions within listed evidence; not semantic-near-duplicate proof",
        "credentials_or_provider_responses_read": False,
    }


def windows(text):
    sentences = hard_units(text)
    for start in range(len(sentences)):
        words = 0
        for end in range(start, len(sentences)):
            if not END_MARK_RE.search(sentences[end]):
                break
            words += len(sentences[end].split())
            if words > 48:
                break
            if words >= 17:
                window = " ".join(sentences[start:end + 1])
                letters = [c for c in window if c.isalpha()]
                if (re.search(r"https?://|www\.", window, re.I) or not letters
                        or sum(c.isascii() for c in letters) / len(letters) < .95):
                    continue
                yield window, next(i for i, (lo, hi) in enumerate(BINS) if lo <= words <= hi)


def candidate(row, shard, index, bags, available):
    text, url = row.get("text", ""), row.get("url", "")
    if not isinstance(text, str) or not isinstance(url, str) or not text.strip() or not url.strip():
        return None
    locator = f"realnewslike/c4-train.{shard:05d}-of-00512.json.gz:{index}"
    prompt = source.official_prompt(text)
    text_ids = {source.identity(text), raw_sha(text), source.identity(prompt)}
    url_ids = {source.url_identity(url), source.identity(url), raw_sha(" ".join(url.split()))}
    if text_ids & bags["texts"] or url_ids & bags["urls"] or locator in bags["ids"]:
        return None
    for window, bucket in windows(text):
        if not available[bucket]:
            continue
        if {source.identity(window), raw_sha(window)} & bags["texts"]:
            continue
        return {
            "source_text": text, "source_url": url, "text": window, "length_bin": bucket,
            "dataset_locator": locator, "dataset_revision": source.REVISION,
            "normalized_source_sha256": source.identity(text), "source_sha256": raw_sha(text),
            "normalized_prompt_sha256": source.identity(prompt),
            "url_sha256": source.url_identity(url), "window_sha256": raw_sha(window),
            "normalized_window_sha256": source.identity(window),
        }
    return None


def allocations(selected, counts=COUNTS, experiment=OUT.name):
    """Freeze role assignment by within-bin shuffle after score-blind selection."""
    rows = {stage: [] for stage in counts}
    for bucket in range(4):
        pool = [r for r in selected if r["length_bin"] == bucket]
        if len(pool) != sum(values[bucket] for values in counts.values()):
            raise ValueError("Full prespecified source pool required")
        random.Random(SEED + bucket).shuffle(pool)
        cursor = 0
        for stage, bins in counts.items():
            for original in pool[cursor:cursor + bins[bucket]]:
                ordinal = len(rows[stage])
                row = {k: v for k, v in original.items() if k not in ("source_text", "source_url")}
                row.update(stage=stage, ordinal=ordinal,
                           allocation_id=f"{experiment}-{stage}-{ordinal:04d}",
                           word_count=len(row["text"].split()),
                           generation_prompt=row["text"] if stage == "positive_source" else None)
                rows[stage].append(row)
            cursor += bins[bucket]
    return rows


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "SOURCE_CONTRACT.json").exists():
        return verify_contract()
    if not (OUT / "PROTOCOL.md").exists():
        raise RuntimeError("Scientific protocol required before acquisition")
    history_path = OUT / "private/HISTORY.json"
    if not history_path.exists():
        freeze(history_path, make_history())
    freeze(OUT / "SOURCE_CONTRACT.json", {
        "version": "v4-independent-confirmation-source/v1", "at": now(),
        "dataset_revision": source.REVISION, "shards": SHARDS, "rows_per_shard": 100000,
        "word_bins": BINS, "cohort_bin_counts": COUNTS, "role_assignment_seed": SEED,
        "history_sha256": sha256(history_path), "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
        "source_sha256": {p: sha256(ROOT / p) for p in CODE},
        "replacement": False, "parser_detector_calls": 0,
        "role_assignment": "fixed shuffle within bin after complete source acquisition",
    })
    return verify_contract()


def verify_contract():
    f = load(OUT / "SOURCE_CONTRACT.json")
    for path, expected in (
        (OUT / "private/HISTORY.json", f["history_sha256"]),
        (OUT / "PROTOCOL.md", f["protocol_sha256"]),
        *((ROOT / p, v) for p, v in f["source_sha256"].items()),
    ):
        if sha256(path) != expected:
            raise RuntimeError("Source contract drift: " + path.name)
    if (f["cohort_bin_counts"] != {k: list(v) for k, v in COUNTS.items()}
            or f["shards"] != list(SHARDS) or f["dataset_revision"] != source.REVISION):
        raise RuntimeError("Source allocation rules changed")
    return f


def acquire():
    from prepare_semantic_supervision2 import atomic
    verify_contract()
    if (OUT / "INPUT_MANIFEST.json").exists():
        return audit()
    history = load(OUT / "private/HISTORY.json")
    bags = {k: set(v) for k, v in history["identities"].items()}
    journal = OUT / "private/SOURCE_POOL.jsonl"
    selected = []
    if journal.exists():
        raw = journal.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise RuntimeError("Partial source journal needs separate recovery; no silent truncation")
        selected = [json.loads(line) for line in raw.splitlines()]
        for row in selected:
            collect(row, bags)
    totals = tuple(sum(v[bucket] for v in COUNTS.values()) for bucket in range(4))
    done = Counter(row["length_bin"] for row in selected)
    scans = []
    for shard in SHARDS:
        if tuple(done[i] for i in range(4)) == totals:
            break
        last_index = None
        for index, raw in source.stream_shard(shard):
            last_index = index
            row = candidate(raw, shard, index, bags, [done[i] < totals[i] for i in range(4)])
            if row:
                with journal.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                journal.chmod(0o600)
                selected.append(row)
                done[row["length_bin"]] += 1
                collect(row, bags)
            if index % 250 == 0 or tuple(done[i] for i in range(4)) == totals:
                atomic(OUT / "SOURCE_HEARTBEAT.json", {
                    "stage": "source_acquisition", "at": now(), "sources": len(selected),
                    "total_sources": sum(totals), "bins": dict(done), "shard": shard, "row": index})
            if tuple(done[i] for i in range(4)) == totals:
                break
        scans.append({"shard": shard, "last_row": last_index})
    if tuple(done[i] for i in range(4)) != totals:
        raise RuntimeError("Fixed source scan exhausted; no automatic new shards")
    assigned = allocations(selected)
    outputs = {}
    for stage, rows in assigned.items():
        path = OUT / "sealed" / (stage + ".jsonl")
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists() and path.read_text() != payload:
            raise RuntimeError("Allocation checkpoint differs")
        path.write_text(payload)
        path.chmod(0o600)
        outputs[stage] = {"path": str(path.relative_to(OUT)), "sha256": sha256(path), "count": len(rows)}
    freeze(OUT / "INPUT_MANIFEST.json", {
        "source_contract_sha256": sha256(OUT / "SOURCE_CONTRACT.json"),
        "source_pool_sha256": sha256(journal), "outputs": outputs, "scans": scans,
        "counts": {k: len(v) for k, v in assigned.items()}, "parser_detector_calls": 0,
    })
    return audit()


def audit():
    verify_contract()
    manifest = load(OUT / "INPUT_MANIFEST.json")
    if (manifest["source_contract_sha256"] != sha256(OUT / "SOURCE_CONTRACT.json")
            or manifest["source_pool_sha256"] != sha256(OUT / "private/SOURCE_POOL.jsonl")):
        raise RuntimeError("Source freeze identity differs")
    history = load(OUT / "private/HISTORY.json")
    bags = {k: set(v) for k, v in history["identities"].items()}
    selected = [json.loads(line) for line in (OUT / "private/SOURCE_POOL.jsonl").read_text().splitlines()]
    for row in selected:
        match = re.fullmatch(r"realnewslike/c4-train\.(\d{5})-of-00512\.json\.gz:(\d+)",
                             row["dataset_locator"])
        if not match or int(match[1]) not in SHARDS or int(match[2]) >= 100000:
            raise RuntimeError("Source locator outside frozen acquisition scope")
        available = [False] * 4
        available[row["length_bin"]] = True
        replay = candidate({"text": row["source_text"], "url": row["source_url"]},
                           int(match[1]), int(match[2]), bags, available)
        if replay != row:
            raise RuntimeError("Historical exclusion or literal source identity cannot replay")
        collect(row, bags)
        for variant in ("semicolon_all", "semicolon_pairs", "comma_all", "newline_all", "space_all"):
            punctuation_variant(row["text"], variant)
    assigned = allocations(selected)
    for stage, item in manifest["outputs"].items():
        path = OUT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError("Sealed allocation SHA drift")
        observed = [json.loads(line) for line in path.read_text().splitlines()]
        if observed != assigned[stage] or item["count"] != len(observed):
            raise RuntimeError("Role assignment replay differs")
    if set(manifest["outputs"]) != set(COUNTS):
        raise RuntimeError("Missing cohort")
    result = {
        "status": "SOURCES_FROZEN_AND_IDENTITY_REPLAY_PASSED", "counts": manifest["counts"],
        "source_units": len(selected), "cross_role_exact_identity_overlap": 0,
        "historical_exact_identity_overlap": 0,
        "history_scope": history["scope"], "history_input_files": len(history["inventory"]),
        "parser_calls": 0, "detector_calls": 0, "generation_calls": 0,
        "runtime_prepared": False, "formal_result": None,
        "input_manifest_sha256": sha256(OUT / "INPUT_MANIFEST.json"),
    }
    freeze(OUT / "SOURCE_AUDIT.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("initialize", "acquire", "audit"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {"initialize": initialize, "acquire": acquire, "audit": audit}[args.action]()
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
