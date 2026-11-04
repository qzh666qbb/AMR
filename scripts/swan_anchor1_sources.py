"""Allocate unseen C4 source windows without reading final1 sealed content."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from swan_anchor1_common import ROOT, OUT, digest, freeze, load, now, sha256
from swan_final1_transform import sentence_spans, unit_texts, word_tokens

REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
COHORTS = (("pilot", 8), ("main", 500), ("calibration", 4000), ("test", 4000))
POSITIVE_RANGE = ("validation", 0, 13_863)
HUMAN_RANGE = ("train", 1_700_000, 1_809_584)
PAGE_SIZE = 100
PAGE_RETRIES = 8
TEXT_KEYS = {
    "article", "article_text", "attacked_text", "baseline_text", "candidate",
    "candidate_text", "context", "generated_text", "generated_texts",
    "generation_prompt", "human_text", "original", "original_passage",
    "original_text", "passage", "prompt", "prompts", "source", "source_text",
    "source_window", "source_anchors", "text", "texts", "watermarked_text",
    "watermarked_texts", "window", "window_text", "prompt_text", "prefix",
    "prefix_text", "sentence", "sentences", "continuation", "output_text",
    "left", "right", "transformed_text", "unit_text", "slot_texts",
}
HASH_KEYS = {
    "identity_sha256", "normalized_passage_sha256", "normalized_sha256",
    "normalized_source_sha256", "normalized_text_sha256",
    "normalized_prompt_sha256", "normalized_window_sha256",
}
SECRET_PATH = re.compile(
    r"(?:^|/)(?:api\.txt|1\.txt|[^/]*(?:credential|secret|api[_-]?key)[^/]*)$",
    re.I,
)


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def identity(text):
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def url_identity(url):
    parts = urlsplit(url.strip())
    value = urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"),
         parts.query, "")
    )
    return identity(value)


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
        elif key in HASH_KEYS and re.fullmatch(r"[a-f0-9]{64}", value):
            texts.add(value)
        if key in {"url", "source_url"}:
            urls.add(url_identity(value))


def _denied(path, root, excluded):
    relative = str(path.relative_to(root))
    final1_sealed = root / "experiments/swan_count_neutral_final1/sealed"
    return (
        path == excluded
        or excluded in path.parents
        or path == final1_sealed
        or final1_sealed in path.parents
        or bool(SECRET_PATH.search(relative))
    )


def history_snapshot(root=ROOT, excluded=OUT):
    """Collect prior exact identities while leaving sealed final1 and secrets unread."""
    root, excluded = Path(root), Path(excluded)
    texts, urls, inventory, excluded_paths = set(), set(), [], []
    candidates = [
        path
        for name in ("experiments", "archive", "artifacts")
        if (root / name).exists()
        for path in (root / name).rglob("*")
    ]
    candidates += list(root.glob("*"))
    scanned = 0
    for path in sorted(set(candidates)):
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to(root))
        if _denied(path, root, excluded):
            excluded_paths.append({"path": relative, "reason": "prohibited_path_not_read"})
            continue
        if path.suffix not in {".json", ".jsonl", ".csv", ".tsv", ".txt"}:
            continue
        scanned += 1
        raw = path.read_bytes()
        if path.suffix in {".json", ".jsonl"}:
            keys = TEXT_KEYS | HASH_KEYS | {"url", "source_url"}
            if not any(f'"{key}"'.encode() in raw for key in keys) \
                    and not raw.lstrip().startswith(b'["'):
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
                delimiter = "\t" if path.suffix == ".tsv" else ","
                for value in csv.DictReader(stream, delimiter=delimiter):
                    _collect(value, texts, urls)
        else:
            for paragraph in re.split(r"\n\s*\n", raw.decode("utf-8-sig")):
                _collect(paragraph, texts, urls, "text")
        inventory.append({"path": relative, "sha256": sha256(path), "bytes": len(raw)})
    return {
        "created_at": now(),
        "scope": ["experiments", "archive", "artifacts", "root_data_files"],
        "scanned_files": scanned,
        "artifact_inventory": inventory,
        "artifact_inventory_digest": digest(inventory),
        "excluded_paths": excluded_paths,
        "excluded_unread_roots": [
            "experiments/swan_count_neutral_final1/sealed",
            "api.txt",
            "1.txt",
            "*credential*/*secret*/*api_key*",
        ],
        "normalized_text_sha256": sorted(texts),
        "url_sha256": sorted(urls),
        "normalization": "NFKC whitespace collapse casefold",
        "positive_split_separation": (
            "anchor1 positives use validation; final1 positives used train shards 13-16"
        ),
    }


def source_window(text):
    """Return the first qualifying contiguous ten-sentence window."""
    spans = sentence_spans(text)
    sentences = unit_texts(text, spans)
    for start in range(max(0, len(sentences) - 9)):
        anchors = sentences[start:start + 10]
        counts = [len(word_tokens(sentence)) for sentence in anchors]
        if len(anchors) == 10 and all(8 <= count <= 35 for count in counts) \
                and 100 <= sum(counts) <= 220:
            first, last = spans[start][0], spans[start + 9][1]
            return {
                "text": text[first:last],
                "anchors": anchors,
                "source_span": [first, last],
                "anchor_word_counts": counts,
                "word_count": sum(counts),
                "num_units": 10,
            }
    return None


def stream_rows(split, start, stop):
    import requests
    if split not in {"train", "validation"} or not 0 <= start <= stop:
        raise ValueError("Invalid frozen dataset range")
    endpoint = "https://datasets-server.huggingface.co/rows"
    for offset in range(start, stop, PAGE_SIZE):
        length = min(PAGE_SIZE, stop - offset)
        parameters = {
            "dataset": "allenai/c4", "config": "realnewslike",
            "split": split, "offset": offset, "length": length,
            "revision": REVISION,
        }
        for retry in range(PAGE_RETRIES + 1):
            response = requests.get(
                endpoint, params=parameters, timeout=(20, 60)
            )
            if response.status_code != 429:
                break
            if retry == PAGE_RETRIES:
                response.raise_for_status()
            header = response.headers.get("retry-after", "")
            delay = int(header) if header.isdigit() else min(60, 2 ** retry)
            time.sleep(max(1, delay))
        response.raise_for_status()
        if response.headers.get("x-revision") != REVISION:
            raise RuntimeError("Dataset server revision header differs")
        payload = response.json()
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != length:
            raise RuntimeError("Dataset server returned an incomplete fixed page")
        for expected_index, item in enumerate(rows, start=offset):
            if item.get("row_idx") != expected_index \
                    or not isinstance(item.get("row"), dict) \
                    or item.get("truncated_cells"):
                raise RuntimeError("Dataset row identity or content is incomplete")
            yield split, expected_index, item["row"]


def screening(row, split, index, seen_text, seen_url):
    text = row.get("text", "") if isinstance(row.get("text", ""), str) else ""
    url = row.get("url", "") if isinstance(row.get("url", ""), str) else ""
    letters = [character for character in text if character.isalpha()]
    window = source_window(text) if text.strip() else None
    candidate_ids = {identity(text)}
    if window:
        candidate_ids.add(identity(window["text"]))
        candidate_ids.update(identity(anchor) for anchor in window["anchors"])
    checks = {
        "nonempty": bool(text.strip()),
        "english_characters": bool(letters)
        and sum(character.isascii() for character in letters) / len(letters) >= .95,
        "first_qualifying_ten_sentence_window": window is not None,
        "new_article_window_and_anchors": not candidate_ids & seen_text,
        "new_url": not url or url_identity(url) not in seen_url,
    }
    return {
        "locator": f"datasets-server/realnewslike/{split}:{index}",
        "split": split,
        "row": index,
        "url": url,
        "domain": urlsplit(url).hostname or "",
        "source_text": text,
        "source_window": window["text"] if window else "",
        "source_anchors": window["anchors"] if window else [],
        "source_span": window["source_span"] if window else None,
        "anchor_word_counts": window["anchor_word_counts"] if window else [],
        "source_word_count": window["word_count"] if window else 0,
        "normalized_source_sha256": identity(text),
        "normalized_window_sha256": identity(window["text"]) if window else None,
        "url_sha256": url_identity(url),
        "checks": checks,
        "selected": all(checks.values()),
    }


def select(rows, snapshot, cohorts=COHORTS):
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    sequence = [(name, ordinal) for name, count in cohorts for ordinal in range(count)]
    selected, screened = [], []
    for split, index, row in rows:
        screen = screening(row, split, index, seen_text, seen_url)
        screened.append(screen)
        if not screen["selected"]:
            continue
        phase, ordinal = sequence[len(selected)]
        selected.append({
            **screen,
            "phase": phase,
            "ordinal": ordinal,
            "allocation_id": f"anchor1-{phase}-{ordinal:04d}",
        })
        seen_text.update({
            screen["normalized_source_sha256"],
            screen["normalized_window_sha256"],
            *(identity(anchor) for anchor in screen["source_anchors"]),
        })
        if screen["url"]:
            seen_url.add(screen["url_sha256"])
        if len(selected) == len(sequence):
            return selected, screened
    raise RuntimeError(f"Fixed source scan exhausted ({len(selected)}/{len(sequence)})")


def control_locators():
    """Freeze human source identities without exposing content before positive gates."""
    split, start, stop = HUMAN_RANGE
    total = sum(count for _, count in COHORTS[2:])
    if start + total > stop:
        raise RuntimeError("Fixed human locator range is too small")
    records = []
    cursor = start
    for phase, count in COHORTS[2:]:
        for ordinal in range(count):
            records.append({
                "allocation_id": f"anchor1-{phase}-{ordinal:04d}",
                "phase": phase,
                "ordinal": ordinal,
                "split": split,
                "row": cursor,
                "locator": f"datasets-server/realnewslike/{split}:{cursor}",
                "dataset_revision": REVISION,
                "content_materialized": False,
            })
            cursor += 1
    return records


def allocate(folder=OUT):
    folder = Path(folder)
    if (folder / "source_manifest.json").exists():
        return check_sources(folder)
    snapshot_path = folder / "prior_identity_snapshot.json"
    snapshot = load(snapshot_path) if snapshot_path.exists() else freeze(
        snapshot_path, history_snapshot(excluded=folder)
    )

    positive, positive_screened = select(
        stream_rows(*POSITIVE_RANGE),
        snapshot,
        COHORTS[:2],
    )
    controls = control_locators()
    selected = positive + controls
    screened = positive_screened
    hashes = {}
    for phase, _ in COHORTS:
        path = folder / "sealed" / f"{phase}_sources.json"
        freeze(path, [record for record in selected if record["phase"] == phase])
        hashes[str(path.relative_to(folder))] = sha256(path)
    screening_path = folder / "sealed" / "source_screening.json"
    freeze(screening_path, screened)
    hashes[str(screening_path.relative_to(folder))] = sha256(screening_path)
    return freeze(folder / "source_manifest.json", {
        "created_at": now(),
        "dataset_revision": REVISION,
        "dataset_endpoint": "https://datasets-server.huggingface.co/rows",
        "positive_range": list(POSITIVE_RANGE),
        "human_range": list(HUMAN_RANGE),
        "page_size": PAGE_SIZE,
        "page_retries_on_429": PAGE_RETRIES,
        "cohort_counts": dict(COHORTS),
        "replacement_allowed": False,
        "window_rule": "first contiguous 10 sentences; each 8-35 words; total 100-220",
        "download_integrity": "TLS plus exact x-revision response header on every page",
        "prior_identity_snapshot_sha256": sha256(snapshot_path),
        "sealed_file_hashes": hashes,
        "rows_scanned": len(screened),
        "rows_excluded": len(screened) - len(positive),
        "last_locator": screened[-1]["locator"],
        "human_content_materialization": (
            "deferred until positive main quality gate; fixed locators fail without replacement"
        ),
        "main_content_access": "after evidence-derived passed pilot only",
        "final1_sealed_content_access": False,
        "credential_file_access": False,
        "generation_parser_scoring_calls": 0,
    })


def check_sources(folder=OUT):
    folder = Path(folder)
    manifest = load(folder / "source_manifest.json")
    if manifest["cohort_counts"] != dict(COHORTS) or manifest["replacement_allowed"]:
        raise ValueError("Source allocation contract changed")
    if manifest["positive_range"] != list(POSITIVE_RANGE) \
            or manifest["human_range"] != list(HUMAN_RANGE) \
            or manifest["final1_sealed_content_access"] \
            or manifest["credential_file_access"]:
        raise ValueError("Source isolation contract changed")
    if sha256(folder / "prior_identity_snapshot.json") != \
            manifest["prior_identity_snapshot_sha256"]:
        raise ValueError("Historical identity snapshot changed")
    for relative, expected in manifest["sealed_file_hashes"].items():
        if sha256(folder / relative) != expected:
            raise ValueError(f"Sealed source changed: {relative}")
    return manifest
