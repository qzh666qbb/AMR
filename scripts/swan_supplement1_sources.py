"""Detector-blind, fixed-source allocation for the new SWAN supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from swan_final1_common import freeze, load, now, sha256, single_writer
import swan_final1_sources as prior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_count_neutral_supplement1"
COHORTS = (("pilot", 8), ("main", 200), ("calibration", 1000), ("test", 1000))
WORD = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")


def tokenizer():
    import nltk
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    return nltk.sent_tokenize


def control_window(text, tokenize):
    sentences = tokenize(text)
    # First eligible consecutive window, without parsing or detection.
    for start in range(max(0, len(sentences) - 15)):
        window = sentences[start:start + 16]
        if not all(s.endswith(".") and not s.endswith("..")
                   and len(WORD.findall(s)) >= 4 for s in window):
            continue
        words = len(WORD.findall(" ".join(window)))
        if 100 <= words <= 220 and tokenize(" ".join(window)) == window:
            return {"sentences": window, "start_sentence": start, "words": words}
    return None


def allocate():
    if (OUT / "source_manifest.json").exists():
        audit()
        return load(OUT / "source_manifest.json")
    tokenize = tokenizer()
    snapshot_path = OUT / "prior_identity_snapshot.json"
    snapshot = load(snapshot_path) if snapshot_path.exists() else freeze(
        snapshot_path, prior.history_snapshot(root=ROOT, excluded=OUT))
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    plan = [(phase, ordinal) for phase, n in COHORTS for ordinal in range(n)]
    rows, screening = [], []
    for shard in range(20, 24):
        for index, row in prior.stream_shard(shard):
            phase, ordinal = plan[len(rows)]
            screen = prior.screening(row, shard, index, seen_text, seen_url)
            window = control_window(screen["text"], tokenize) if (
                screen["selected"] and phase in ("calibration", "test")) else None
            checks = dict(screen["checks"])
            if phase in ("calibration", "test"):
                checks["eligible_window"] = window is not None
                checks["fresh_window"] = window is not None and prior.identity(
                    " ".join(window["sentences"])) not in seen_text
            screening.append({"locator": screen["locator"], "phase": phase,
                              "checks": checks, "selected": all(checks.values()),
                              "source_sha256": screen["normalized_source_sha256"],
                              "prompt_sha256": screen["normalized_prompt_sha256"],
                              "url_sha256": screen["url_sha256"]})
            if not all(checks.values()):
                continue
            allocation = {**screen, "phase": phase, "ordinal": ordinal,
                          "allocation_id": f"supplement1-{phase}-{ordinal:04d}"}
            if window is not None:
                allocation["control_window"] = window
                seen_text.add(prior.identity(" ".join(window["sentences"])))
            rows.append(allocation)
            seen_text.update((screen["normalized_source_sha256"],
                              screen["normalized_prompt_sha256"]))
            if screen["url"]:
                seen_url.add(screen["url_sha256"])
            if len(rows) == len(plan):
                break
        if len(rows) == len(plan):
            break
    if len(rows) != len(plan):
        freeze(OUT / "SOURCE_STOP.json", {"at": now(), "selected": len(rows),
                                         "expected": len(plan), "replacement": False})
        raise RuntimeError("Fixed source scan exhausted")
    hashes = {}
    for phase, count in COHORTS:
        path = OUT / "private" / f"{phase}_sources.json"
        freeze(path, [r for r in rows if r["phase"] == phase])
        hashes[str(path.relative_to(OUT))] = sha256(path)
    freeze(OUT / "private/source_screening.json", screening)
    manifest = freeze(OUT / "source_manifest.json", {
        "at": now(), "dataset_revision": prior.REVISION, "shards": [20, 21, 22, 23],
        "cohorts": dict(COHORTS), "replacement": False,
        "prior_identity_snapshot_sha256": sha256(snapshot_path),
        "sealed_file_hashes": hashes, "rows_scanned": len(screening),
        "rows_excluded": len(screening) - len(rows),
        "source_screening_sha256": sha256(OUT / "private/source_screening.json"),
        "model_parser_detector_calls": 0,
    })
    freeze(OUT / "SOURCE_AUDIT.json", audit())
    return manifest


def audit():
    manifest = load(OUT / "source_manifest.json")
    assert manifest["cohorts"] == dict(COHORTS)
    assert sha256(OUT / "prior_identity_snapshot.json") == manifest[
        "prior_identity_snapshot_sha256"]
    snapshot = load(OUT / "prior_identity_snapshot.json")
    texts, urls = set(snapshot["normalized_text_sha256"]), set(snapshot["url_sha256"])
    checked = 0
    for phase, count in COHORTS:
        path = OUT / "private" / f"{phase}_sources.json"
        assert sha256(path) == manifest["sealed_file_hashes"][str(path.relative_to(OUT))]
        rows = load(path)
        assert len(rows) == count
        for ordinal, row in enumerate(rows):
            assert row["ordinal"] == ordinal and row["phase"] == phase
            hashes = {prior.identity(row["text"]), prior.identity(row["generation_prompt"])}
            if "control_window" in row:
                hashes.add(prior.identity(" ".join(row["control_window"]["sentences"])))
            assert not hashes & texts
            assert not row["url"] or prior.url_identity(row["url"]) not in urls
            texts.update(hashes)
            if row["url"]:
                urls.add(prior.url_identity(row["url"]))
            checked += 1
    assert sha256(OUT / "private/source_screening.json") == manifest["source_screening_sha256"]
    return {"identity_audit": "Go", "rows": checked, "overlap": 0,
            "cohorts": dict(COHORTS), "main_text_exposed_in_output": False,
            "model_parser_detector_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("allocate", "audit"))
    args = parser.parse_args()
    with single_writer(OUT):
        result = allocate() if args.stage == "allocate" else audit()
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("cohorts", "rows_scanned", "rows_excluded", "identity_audit", "rows", "overlap")}))
