"""Freeze source identities and allocate score-blind, disjoint development data."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata


REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
BINS = ((17, 24), (25, 32), (33, 40), (41, 48))
COUNTS = {"calibration": (250,) * 4, "null": (50,) * 4, "positive_source": (13, 13, 12, 12)}
TEXT_KEYS = {
    "source", "source_text", "original", "original_text", "article", "article_text",
    "text", "context", "prompt", "generation_prompt", "window", "sentence",
    "document", "watermarked_text", "generated_text", "completion", "output_text",
}
HASH_KEYS = {
    "normalized_source_sha256", "normalized_sha256", "identity_sha256",
    "window_sha256", "text_sha256", "source_sha256", "generation_prompt_sha256",
    "source_text_sha256", "normalized_window_sha256",
}
ID_KEYS = {"allocation_id", "dataset_locator", "locator"}
HEX = re.compile(r"^[a-f0-9]{64}$")
SENTENCE = re.compile(r"""[^.!?]+[.!?]+["')\]]*(?=\s|$)""")
VARIANTS = ("semicolon_all", "semicolon_pairs", "comma_all", "newline_all", "space_all")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def identity(text):
    return sha(canonical(text).casefold().encode())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze(path: Path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen artifact differs: {path.name}")
    else:
        write(path, value)
    return value


def collect(value, bags, key=None):
    if isinstance(value, dict):
        for name, item in value.items():
            collect(item, bags, name.casefold())
    elif isinstance(value, list):
        for item in value:
            collect(item, bags, key)
    elif isinstance(value, str):
        if key in TEXT_KEYS and value.strip():
            bags["texts"].add(identity(value))
            bags["raw_hashes"].add(sha(value.encode()))
        if key in HASH_KEYS and HEX.fullmatch(value):
            bags["raw_hashes"].add(value)
            bags["texts"].add(value)
        if key in ("url", "source_url") and value.strip():
            bags["urls"].add(sha(canonical(value).encode()))
        if key == "url_sha256" and HEX.fullmatch(value):
            bags["urls"].add(value)
        if key in ID_KEYS and value:
            bags["ids"].add(value)


def history_snapshot(root: Path, exclude: Path):
    bags = {key: set() for key in ("texts", "raw_hashes", "urls", "ids")}
    inventory = []
    excluded_sidecars = []
    for path in sorted((root / "experiments").rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl") or exclude in path.parents:
            continue
        if path.name.startswith("._"):
            excluded_sidecars.append({"path": str(path.relative_to(root)), "sha256": file_sha(path)})
            continue
        inventory.append({"path": str(path.relative_to(root)), "sha256": file_sha(path)})
        with path.open(encoding="utf-8-sig") as stream:
            if path.suffix == ".jsonl":
                for line in stream:
                    if line.strip():
                        collect(json.loads(line), bags)
            else:
                collect(json.load(stream), bags)
    return {
        "schema_version": 1, "inventory": inventory,
        "excluded_appledouble_metadata": excluded_sidecars,
        "identities": {key: sorted(value) for key, value in bags.items()},
    }


def variants(text):
    from .length_boundary_validation import punctuation_variant

    return {"baseline": text, **{
        variant: punctuation_variant(text, variant) for variant in VARIANTS
    }}


def windows(text):
    from .length_boundary_validation import END_MARK_RE, hard_units

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
                if re.search(r"https?://|www\.", window, re.I) or not re.search(r"[A-Za-z]", window):
                    continue
                yield window, next(i for i, (lo, hi) in enumerate(BINS) if lo <= words <= hi)


def prepare(root: Path, output: Path):
    import requests

    manifest_path = output / "INPUT_MANIFEST.json"
    if manifest_path.exists():
        return verify(output)
    local_history = json.loads((output / "LOCAL_HISTORY.json").read_text())
    remote_path = output / "REMOTE_HISTORY.json"
    if remote_path.exists():
        remote_history = json.loads(remote_path.read_text())
    else:
        remote_history = freeze(remote_path, history_snapshot(root, output))
    bags = {
        key: set(local_history["identities"][key]) | set(remote_history["identities"][key])
        for key in ("texts", "raw_hashes", "urls", "ids")
    }
    rows = {stage: [] for stage in COUNTS}
    completed = {stage: Counter() for stage in COUNTS}
    selected = output / "sealed/source_rows.jsonl"
    if selected.exists():
        for line in selected.read_text().splitlines():
            row = json.loads(line)
            rows[row["stage"]].append(row)
            completed[row["stage"]][row["length_bin"]] += 1
            collect(row, bags)
    scans = []
    for shard in range(20, 24):
        if all(tuple(completed[stage][i] for i in range(4)) == COUNTS[stage] for stage in COUNTS):
            break
        name = f"realnewslike/c4-train.{shard:05d}-of-00512.json.gz"
        url = f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{name}"
        last_error = None
        for attempt in range(1, 4):
            try:
                with requests.get(url, stream=True, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    response.raw.decode_content = True
                    with gzip.GzipFile(fileobj=response.raw) as stream:
                        for index, line in enumerate(stream):
                            if index >= 50000:
                                break
                            source = json.loads(line)
                            text, source_url = source.get("text", ""), source.get("url", "")
                            source_id = identity(text)
                            url_id = sha(canonical(source_url).encode())
                            locator = f"{name}:{index}"
                            if not source_url or source_id in bags["texts"] or url_id in bags["urls"] or locator in bags["ids"]:
                                continue
                            for window, bucket in windows(text):
                                window_id = identity(window)
                                if window_id in bags["texts"] or sha(window.encode()) in bags["raw_hashes"]:
                                    continue
                                stage = next(
                                    (stage for stage in COUNTS if completed[stage][bucket] < COUNTS[stage][bucket]),
                                    None,
                                )
                                if stage is None:
                                    continue
                                ordinal = len(rows[stage])
                                row = {
                                    "stage": stage, "length_bin": bucket, "ordinal": ordinal,
                                    "allocation_id": f"{output.name}-{stage}-{ordinal:04d}",
                                    "dataset_locator": locator, "dataset_revision": REVISION,
                                    "normalized_source_sha256": source_id,
                                    "source_sha256": sha(text.encode()), "url_sha256": url_id,
                                    "window_sha256": sha(window.encode()),
                                    "normalized_window_sha256": window_id,
                                    "text": window, "word_count": len(window.split()),
                                    "generation_prompt": window if stage == "positive_source" else None,
                                }
                                selected.parent.mkdir(parents=True, exist_ok=True)
                                with selected.open("a") as journal:
                                    journal.write(json.dumps(row, sort_keys=True) + "\n")
                                    journal.flush()
                                    os.fsync(journal.fileno())
                                rows[stage].append(row)
                                completed[stage][bucket] += 1
                                bags["texts"].update((source_id, window_id))
                                bags["urls"].add(url_id)
                                bags["ids"].update((locator, row["allocation_id"]))
                                break
                            if index % 100 == 0:
                                write(output / "HEARTBEAT.json", {
                                    "status": "running", "stage": "source_preparation", "updated_at": now(),
                                    "shard": shard, "row": index, "counts": {k: len(v) for k, v in rows.items()},
                                })
                            if all(tuple(completed[stage][i] for i in range(4)) == COUNTS[stage] for stage in COUNTS):
                                break
                        scans.append({"shard": shard, "last_row": index, "attempt": attempt})
                last_error = None
                break
            except (requests.RequestException, OSError, EOFError) as exc:
                last_error = type(exc).__name__
        if last_error:
            raise RuntimeError(f"Source acquisition failed after 3 attempts: {last_error}")
    if not all(tuple(completed[stage][i] for i in range(4)) == COUNTS[stage] for stage in COUNTS):
        raise RuntimeError("Frozen source allocation limit reached before sufficient disjoint inputs")
    outputs = {}
    for stage, values in rows.items():
        path = output / f"sealed/{stage}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in values).encode()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Source checkpoint differs")
        path.write_bytes(payload)
        outputs[stage] = {"path": str(path.relative_to(output)), "sha256": file_sha(path), "count": len(values)}
    manifest = {
        "schema_version": 1, "version": "path0-development-v1", "outputs": outputs,
        "protocol_sha256": file_sha(output / "PROTOCOL.md"),
        "history_sha256": {name: file_sha(output / name) for name in ("LOCAL_HISTORY.json", "REMOTE_HISTORY.json")},
        "counts": {stage: dict(counts) for stage, counts in completed.items()},
        "cross_set_identity_overlap": 0, "scans": scans,
    }
    freeze(manifest_path, manifest)
    return verify(output)


def verify(output):
    manifest = json.loads((output / "INPUT_MANIFEST.json").read_text())
    if manifest["protocol_sha256"] != file_sha(output / "PROTOCOL.md"):
        raise ValueError("Input protocol SHA mismatch")
    for name, digest in manifest["history_sha256"].items():
        if file_sha(output / name) != digest:
            raise ValueError("History snapshot SHA mismatch")
    bags = {key: set() for key in ("normalized_source_sha256", "url_sha256", "dataset_locator", "allocation_id", "normalized_window_sha256")}
    for stage, item in manifest["outputs"].items():
        path = output / item["path"]
        if file_sha(path) != item["sha256"]:
            raise ValueError("Input file SHA mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        counts = Counter(row["length_bin"] for row in rows)
        if len(rows) != sum(COUNTS[stage]) or tuple(counts[i] for i in range(4)) != COUNTS[stage]:
            raise ValueError("Frozen source count mismatch")
        for row in rows:
            if sha(row["text"].encode()) != row["window_sha256"] or identity(row["text"]) != row["normalized_window_sha256"]:
                raise ValueError("Window SHA mismatch")
            variants(row["text"])
            for key, seen in bags.items():
                if row[key] in seen:
                    raise ValueError(f"New source identity overlap: {key}")
                seen.add(row[key])
    return manifest
