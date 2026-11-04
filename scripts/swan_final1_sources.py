"""Public C4 allocation and complete historical identity exclusion for final1."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from swan_final1_common import ROOT, OUT, digest, freeze, load, now, sha256, single_writer

REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
COHORTS = (("pilot", 8), ("main", 500), ("calibration", 4000), ("test", 4000))
TEXT_KEYS = {
    "article", "article_text", "attacked_text", "baseline_text", "candidate",
    "candidate_text", "context", "generated_text", "generated_texts", "generation_prompt",
    "human_text", "original", "original_passage", "original_text", "passage",
    "prompt", "prompts", "source", "source_text", "text", "texts", "watermarked_text",
    "watermarked_texts", "window", "window_text", "source_window", "prompt_text",
    "prefix", "prefix_text", "sentence", "sentences", "continuation", "output_text",
    "left", "right", "transformed_text", "unit_text", "slot_texts",
}
HASH_KEYS = {
    "identity_sha256", "normalized_passage_sha256", "normalized_sha256",
    "normalized_source_sha256", "normalized_text_sha256", "normalized_prompt_sha256",
}
SECRET_PATH = re.compile(r"(?:^|/)(?:1\.txt|[^/]*(?:credential|secret|api_key)[^/]*)$", re.I)


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def identity(text):
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def url_identity(url):
    parts = urlsplit(url.strip())
    # Fragment is not article identity; retain query because it may identify an article.
    canonical_url = urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(),
                               parts.path.rstrip("/"), parts.query, ""))
    return identity(canonical_url)


def official_prompt(text):
    excerpt = " ".join(text.split(" ")[:32])
    candidates = [excerpt[:excerpt.find(p) + 1] for p in "!.?"
                  if excerpt.find(p) != -1 and
                  len(excerpt[:excerpt.find(p) + 1].split(" ")) > 3]
    return min(candidates, key=len) if candidates else excerpt + "."


def _collect(value, texts, urls, key=""):
    if isinstance(value, dict):
        for field, item in value.items():
            _collect(item, texts, urls, str(field).casefold())
    elif isinstance(value, list):
        for item in value:
            _collect(item, texts, urls, key)
    elif isinstance(value, str) and value.strip():
        if key in TEXT_KEYS:
            texts.add(identity(value))
            texts.add(identity(official_prompt(value)))
        elif key in HASH_KEYS and re.fullmatch(r"[a-f0-9]{64}", value):
            texts.add(value)
        if key in {"url", "source_url"}:
            urls.add(url_identity(value))


def history_snapshot(root=ROOT, excluded=OUT):
    texts, urls, inventory, excluded_paths = set(), set(), [], []
    candidates = [p for name in ("experiments", "archive", "artifacts")
                  for p in (root / name).rglob("*")]
    candidates += list(root.glob("*"))
    scanned = 0
    for path in sorted(set(candidates)):
        if not path.is_file() or path.is_symlink() or excluded in path.parents:
            continue
        if path.suffix not in {".json", ".jsonl", ".csv", ".tsv", ".txt"}:
            continue
        relative = str(path.relative_to(root))
        if SECRET_PATH.search(relative):
            excluded_paths.append({"path": relative, "reason": "credential_path_not_read"})
            continue
        scanned += 1
        raw = path.read_bytes()
        if path.suffix in {".json", ".jsonl"}:
            keys = TEXT_KEYS | HASH_KEYS | {"url", "source_url"}
            if not any(f'"{key}"'.encode() in raw for key in keys):
                # Include bare arrays of document strings as well.
                if not raw.lstrip().startswith(b'["'):
                    continue
            content = raw.decode("utf-8-sig")
            try:
                values = [json.loads(line) for line in content.splitlines() if line.strip()] \
                    if path.suffix == ".jsonl" else [json.loads(content)]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Historical identity file is invalid: {relative}") from exc
            for value in values:
                _collect(value, texts, urls, "texts")
        elif path.suffix in {".csv", ".tsv"}:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for value in csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","):
                    _collect(value, texts, urls)
        else:
            # Plain-text historical notes: extract paragraphs as conservative identities.
            for paragraph in re.split(r"\n\s*\n", raw.decode("utf-8-sig")):
                _collect(paragraph, texts, urls, "text")
        inventory.append({"path": relative, "sha256": sha256(path), "bytes": len(raw)})
    return {
        "created_at": now(), "scope": ["experiments", "archive", "artifacts", "root_data_files"],
        "scanned_files": scanned, "artifact_inventory": inventory,
        "artifact_inventory_digest": digest(inventory), "excluded_paths": excluded_paths,
        "normalized_text_sha256": sorted(texts), "url_sha256": sorted(urls),
        "normalization": "NFKC whitespace collapse casefold; plus official prefix identity",
        "claims": "Exact full/prompt/window identities; not a proof of no semantic near duplicates",
    }


def stream_shard(shard):
    import requests
    url = (f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/"
           f"realnewslike/c4-train.{shard:05d}-of-00512.json.gz")
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as stream:
            for index, line in enumerate(stream):
                if index >= 100000:
                    break
                yield index, json.loads(line)


def screening(row, shard, index, seen_text, seen_url):
    text, url = row.get("text", ""), row.get("url", "")
    text = text if isinstance(text, str) else ""
    url = url if isinstance(url, str) else ""
    prompt = official_prompt(text) if text.strip() else ""
    letters = [c for c in text if c.isalpha()]
    identities = {identity(text), identity(prompt)}
    checks = {
        "nonempty": bool(text.strip() and prompt.strip()),
        "english_characters": bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= .95,
        "new_text_and_prompt": not identities & seen_text,
        "new_url": not url or url_identity(url) not in seen_url,
    }
    return {
        "locator": f"realnewslike/c4-train.{shard:05d}-of-00512.json.gz:{index}",
        "shard": shard, "row": index, "text": text, "url": url,
        "domain": urlsplit(url).hostname or "",
        "generation_prompt": prompt, "normalized_source_sha256": identity(text),
        "normalized_prompt_sha256": identity(prompt), "url_sha256": url_identity(url),
        "checks": checks, "selected": all(checks.values()),
    }


def select(rows, snapshot, cohorts=COHORTS):
    seen_text, seen_url = set(snapshot["normalized_text_sha256"]), set(snapshot["url_sha256"])
    sequence = [(name, index) for name, count in cohorts for index in range(count)]
    selected, screened = [], []
    for shard, index, row in rows:
        screen = screening(row, shard, index, seen_text, seen_url)
        screened.append(screen)
        if not screen["selected"]:
            continue
        phase, ordinal = sequence[len(selected)]
        selected.append({**screen, "phase": phase, "ordinal": ordinal,
                         "allocation_id": f"final1-{phase}-{ordinal:04d}"})
        seen_text.update((screen["normalized_source_sha256"], screen["normalized_prompt_sha256"]))
        if screen["url"]:
            seen_url.add(screen["url_sha256"])
        if len(selected) == len(sequence):
            return selected, screened
    raise RuntimeError(f"Fixed source scan exhausted ({len(selected)}/{len(sequence)})")


def allocate(folder=OUT):
    folder = Path(folder)
    if (folder / "source_manifest.json").exists():
        return check_sources(folder)
    snapshot_path = folder / "prior_identity_snapshot.json"
    # Resume a download by replaying the same fixed scan and frozen history.
    snapshot = load(snapshot_path) if snapshot_path.exists() else \
        freeze(snapshot_path, history_snapshot(excluded=folder))

    def rows():
        for shard in range(13, 17):
            for index, row in stream_shard(shard):
                yield shard, index, row

    selected, screened = select(rows(), snapshot)
    hashes = {}
    for phase, _ in COHORTS:
        path = folder / "sealed" / f"{phase}_sources.json"
        freeze(path, [r for r in selected if r["phase"] == phase])
        hashes[str(path.relative_to(folder))] = sha256(path)
    screening_path = folder / "sealed" / "source_screening.json"
    freeze(screening_path, screened)
    hashes[str(screening_path.relative_to(folder))] = sha256(screening_path)
    return freeze(folder / "source_manifest.json", {
        "created_at": now(), "dataset_revision": REVISION,
        "cohort_counts": dict(COHORTS), "replacement_allowed": False,
        "prior_identity_snapshot_sha256": sha256(snapshot_path),
        "sealed_file_hashes": hashes, "rows_scanned": len(screened),
        "rows_excluded": len(screened) - len(selected),
        "last_locator": screened[-1]["locator"],
        "main_content_access": "after_passed_pilot_only_except_allocator_and_hash_verification",
        "generation_parser_scoring_calls": 0,
    })


def check_sources(folder=OUT):
    folder = Path(folder)
    manifest = load(folder / "source_manifest.json")
    if manifest["cohort_counts"] != dict(COHORTS) or manifest["replacement_allowed"]:
        raise ValueError("Allocation contract changed")
    if sha256(folder / "prior_identity_snapshot.json") != manifest["prior_identity_snapshot_sha256"]:
        raise ValueError("Historical identity snapshot changed")
    # Hash bytes only: does not expose/read parsed main content before the pilot gate.
    for relative, expected in manifest["sealed_file_hashes"].items():
        if sha256(folder / relative) != expected:
            raise ValueError(f"Sealed source changed: {relative}")
    return manifest


def audit_identities(folder=OUT):
    """Replay allocator internally; never print sealed main/control text."""
    folder = Path(folder)
    manifest = check_sources(folder)
    snapshot = load(folder / "prior_identity_snapshot.json")
    screened = load(folder / "sealed" / "source_screening.json")
    inputs = ((r["shard"], r["row"], {"text": r["text"], "url": r["url"]}) for r in screened)
    selected, replay = select(inputs, snapshot)
    if replay != screened:
        raise ValueError("Source screening cannot be reproduced from frozen inputs")
    for phase, count in COHORTS:
        original = load(folder / "sealed" / f"{phase}_sources.json")
        expected = [r for r in selected if r["phase"] == phase]
        if len(original) != count or original != expected:
            raise ValueError(f"Source allocation cannot be reproduced: {phase}")
    return {"identity_replay_pass": True, "cohort_counts": manifest["cohort_counts"],
            "rows_scanned": len(screened), "rows_excluded": len(screened) - len(selected),
            "sealed_content_exposed": False, "model_parser_scorer_calls": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("allocate", "check", "audit-identities"))
    args = parser.parse_args()
    with single_writer():
        result = {"allocate": allocate, "check": check_sources,
                  "audit-identities": audit_identities}[args.stage]()
    print(json.dumps({k: result[k] for k in ("cohort_counts", "rows_scanned", "rows_excluded")}))
