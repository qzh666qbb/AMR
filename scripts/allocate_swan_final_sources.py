"""Freeze runtime provenance and allocate final SWAN sources without scoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

from prepare_swan_final_confirmation import check as check_preregistration
from run_confirmation_experiment import digest, freeze
from run_segmentation_intervention import ROOT, sha, text_sha, write_json


OUTPUT = ROOT / "experiments/swan_final_confirmation1"
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SHARD_COUNT = 512
SOURCE_START_SHARD = 6
SOURCE_COUNT = 250
HUMAN_CALIBRATION_COUNT = 250
HUMAN_TEST_COUNT = 250
HUMAN_COUNT = HUMAN_CALIBRATION_COUNT + HUMAN_TEST_COUNT
MAX_ACQUISITION_ATTEMPTS = 3
MIN_WORDS = 100
MAX_WORDS = 220
MIN_UNITS = 8
MAX_CONTROL_UNITS = 20
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 50
LIST_MARKER = re.compile(r"^\s*(?:[-*#]|\d+[.)])\s")
TERMINAL_PUNCTUATION = re.compile(r"[.?!][\"')\]]?$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

HISTORY_NAME_MARKERS = (
    "manifest",
    "source",
    "prompt",
    "record",
    "candidate",
    "preparation",
)
TEXT_IDENTITY_KEYS = {
    "source",
    "source_text",
    "original",
    "original_text",
    "article",
    "article_text",
    "text",
    "context",
    "prompt",
    "generation_prompt",
}
HASH_IDENTITY_KEYS = {
    "normalized_source_sha256",
    "normalized_sha256",
    "identity_sha256",
}
URL_KEYS = {"url", "source_url"}


def configure_environment() -> None:
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_hash(text: str) -> str:
    return text_sha(canonical(text).casefold())


def normalized_url_hash(url: str) -> str:
    return text_sha(canonical(url))


def shard_name(index: int) -> str:
    return (
        f"realnewslike/c4-train.{index:05d}-of-{SHARD_COUNT:05d}.json.gz"
    )


def stream_train_shard(index: int):
    import requests

    if index not in range(SHARD_COUNT):
        raise ValueError(f"Invalid C4 shard index: {index}")
    url = (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        f"{REVISION}/{shard_name(index)}"
    )
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for line in handle:
                yield json.loads(line)


def history_artifact_paths(root: Path = ROOT) -> list[Path]:
    experiments = root / "experiments"
    target = experiments / "swan_final_confirmation1"
    paths = []
    for suffix in ("*.json", "*.jsonl"):
        for path in experiments.rglob(suffix):
            if target in path.parents:
                continue
            name = path.name.casefold()
            if any(marker in name for marker in HISTORY_NAME_MARKERS):
                paths.append(path)
    return sorted(set(paths))


def _collect_identity_values(
    value,
    text_hashes: set[str],
    url_hashes: set[str],
    inherited_key: str | None = None,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_identity_values(
                item,
                text_hashes,
                url_hashes,
                str(key).casefold(),
            )
        return
    if isinstance(value, list):
        for item in value:
            _collect_identity_values(
                item,
                text_hashes,
                url_hashes,
                inherited_key,
            )
        return
    if not isinstance(value, str):
        return
    if inherited_key in HASH_IDENTITY_KEYS and HEX_SHA256.fullmatch(value):
        text_hashes.add(value)
    elif inherited_key in TEXT_IDENTITY_KEYS and canonical(value):
        text_hashes.add(normalized_hash(value))
    if inherited_key in URL_KEYS and canonical(value):
        url_hashes.add(normalized_url_hash(value))


def _load_history_artifact(path: Path) -> list:
    content = path.read_bytes().decode("utf-8-sig")
    if path.suffix == ".jsonl":
        values = []
        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if line.strip():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid history JSONL {path}:{line_number}"
                    ) from exc
        return values
    return [json.loads(content)]


def build_prior_identity_snapshot(
    paths: list[Path],
) -> dict:
    text_hashes: set[str] = set()
    url_hashes: set[str] = set()
    inventory = []
    for path in paths:
        relative = str(path.relative_to(ROOT))
        inventory.append(
            {
                "path": relative,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
        )
        for value in _load_history_artifact(path):
            _collect_identity_values(value, text_hashes, url_hashes)
    return {
        "schema_version": 1,
        "selection": {
            "roots": ["experiments"],
            "extensions": [".json", ".jsonl"],
            "filename_contains_any": list(HISTORY_NAME_MARKERS),
            "excluded_tree": "experiments/swan_final_confirmation1",
            "text_keys": sorted(TEXT_IDENTITY_KEYS),
            "hash_keys": sorted(HASH_IDENTITY_KEYS),
            "url_keys": sorted(URL_KEYS),
            "policy": (
                "Conservative prior-use screen: hash text-bearing source-like "
                "fields and retain declared normalized hashes and URL hashes."
            ),
        },
        "artifact_inventory": inventory,
        "artifact_inventory_digest": digest(inventory),
        "normalized_text_sha256": sorted(text_hashes),
        "url_sha256": sorted(url_hashes),
        "counts": {
            "artifacts": len(inventory),
            "normalized_text_identities": len(text_hashes),
            "url_identities": len(url_hashes),
        },
    }


def runtime_dependency_paths() -> list[Path]:
    paths = [
        Path(__file__),
        ROOT / "scripts/run_swan_final_baselines.py",
        ROOT / "scripts/test_swan_final_runtime.py",
        ROOT / "scripts/prepare_swan_final_confirmation.py",
        ROOT / "scripts/swan_final_structural.py",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/run_segmentation_intervention.py",
        ROOT / "scripts/supervise_confirmation_experiment.py",
        ROOT / "third_party/SWAN/injection/injection_amr_utils.py",
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/text_utils.py",
        ROOT / "third_party/SWAN/amr_bank/banks/amr_bank_50.json",
        ROOT
        / "third_party/SWAN/amr_bank/artifacts/normalized_amr_examples.json",
        ROOT / "vectors/glove.6B.100d.txt",
        ROOT / ".nltk_data/corpora/propbank.zip",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    ]
    paths.extend(
        path
        for directory in (
            ROOT / "models/model_parse_xfm_bart_large-v0_1_0",
            ROOT / "models/bart-large-tokenizer",
        )
        for path in directory.glob("*")
        if path.is_file()
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing runtime dependencies: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def technical_amendment() -> dict:
    return {
        "schema_version": 1,
        "status": "frozen_before_source_acquisition",
        "scope": "implementation_detail_only",
        "claim_or_gate_change": False,
        "details": {
            "watermarked_source_order": (
                "Scan C4 RealNews train shards from shard 6 in increasing "
                "shard and row order; select the first 250 nonempty records "
                "not present in the frozen prior-identity snapshot."
            ),
            "control_start": (
                "Begin at the shard immediately after the final shard used "
                "for watermarked sources, ensuring disjoint train shards."
            ),
            "control_passage": (
                "Use the shortest sentence prefix with 8..20 units and "
                "100..220 whitespace words. Every included sentence must "
                "have 5..50 words, terminal punctuation, no list marker, "
                "and a unique normalized identity."
            ),
            "source_prompt": (
                "Use SWAN utils.text_utils.extract_prompt_from_text(text, 32)."
            ),
            "prior_overlap": (
                "Conservatively scan source-like fields in prior experiment "
                "manifests, source/prompt/record/candidate/preparation files."
            ),
            "network_failure": (
                "No partial allocation is frozen. A retry rescans the same "
                "immutable dataset revision from the same starting shard; "
                "at most three acquisition attempts."
            ),
        },
    }


def freeze_runtime(folder: Path = OUTPUT) -> dict:
    preregistration = check_preregistration(folder)
    source_path = folder / "source_manifest.json"
    controls_path = folder / "human_control_manifest.json"
    prior_path = folder / "prior_identity_snapshot.json"
    amendment_path = (
        folder / "implementation_amendments/pre_source_runtime.json"
    )
    runtime_path = folder / "runtime.json"

    if runtime_path.exists():
        return check_runtime(folder)
    if source_path.exists() or controls_path.exists():
        raise RuntimeError("Source artifacts exist before runtime freeze")

    prior = build_prior_identity_snapshot(history_artifact_paths())
    freeze(prior_path, prior)
    amendment = technical_amendment()
    freeze(amendment_path, amendment)
    dependencies = {
        str(path.relative_to(ROOT)): sha(path)
        for path in runtime_dependency_paths()
    }
    runtime = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_source_acquisition",
        "preregistration_sha256": sha(folder / "manifest.json"),
        "prior_identity_snapshot_sha256": sha(prior_path),
        "technical_amendment_sha256": sha(amendment_path),
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": REVISION,
            "shard_count": SHARD_COUNT,
            "watermarked_source_start_shard": SOURCE_START_SHARD,
        },
        "allocation": {
            "watermarked_sources": SOURCE_COUNT,
            "human_calibration": HUMAN_CALIBRATION_COUNT,
            "human_test": HUMAN_TEST_COUNT,
            "maximum_acquisition_attempts": MAX_ACQUISITION_ATTEMPTS,
        },
        "human_surface_filter": {
            "word_range": [MIN_WORDS, MAX_WORDS],
            "unit_range": [MIN_UNITS, MAX_CONTROL_UNITS],
            "sentence_word_range": [
                MIN_SENTENCE_WORDS,
                MAX_SENTENCE_WORDS,
            ],
            "requires_terminal_punctuation": True,
            "rejects_list_markers": True,
            "requires_unique_sentences": True,
        },
        "remote_model": {
            "requested": "deepseek-v4-flash",
            "returned": None,
            "identity_frozen_by": "model_probe.json before baseline generation",
        },
        "remote_generation_request_cap": preregistration["protocol"][
            "remote_generation_request_cap"
        ],
        "dependencies": dependencies,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "amrlib",
                "penman",
                "nltk",
                "numpy",
                "scipy",
                "requests",
            )
        },
    }
    freeze(runtime_path, runtime)
    return runtime


def check_prior_snapshot(folder: Path, runtime: dict) -> dict:
    path = folder / "prior_identity_snapshot.json"
    if sha(path) != runtime["prior_identity_snapshot_sha256"]:
        raise ValueError("Prior identity snapshot changed")
    snapshot = json.loads(path.read_text())
    inventory = snapshot["artifact_inventory"]
    if digest(inventory) != snapshot["artifact_inventory_digest"]:
        raise ValueError("Prior artifact inventory digest changed")
    if snapshot["counts"] != {
        "artifacts": len(inventory),
        "normalized_text_identities": len(
            snapshot["normalized_text_sha256"]
        ),
        "url_identities": len(snapshot["url_sha256"]),
    }:
        raise ValueError("Prior identity snapshot counts changed")
    return snapshot


def check_runtime(folder: Path = OUTPUT) -> dict:
    check_preregistration(folder)
    runtime = json.loads((folder / "runtime.json").read_text())
    if (
        runtime["status"] != "frozen_before_source_acquisition"
        or runtime["preregistration_sha256"]
        != sha(folder / "manifest.json")
        or runtime["technical_amendment_sha256"]
        != sha(folder / "implementation_amendments/pre_source_runtime.json")
    ):
        raise ValueError("Runtime provenance mismatch")
    for relative, expected in runtime["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen runtime dependency changed: {relative}")
    for name, expected in runtime["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Frozen runtime package changed: {name}")
    check_prior_snapshot(folder, runtime)
    return runtime


def human_control_passage(text: str, tokenize) -> dict | None:
    rows = []
    for sentence in tokenize(canonical(text)):
        value = canonical(sentence)
        words = len(value.split())
        rows.append(
            {
                "text": value,
                "words": words,
                "identity_sha256": normalized_hash(value),
                "surface_pass": bool(
                    MIN_SENTENCE_WORDS <= words <= MAX_SENTENCE_WORDS
                    and TERMINAL_PUNCTUATION.search(value)
                    and not LIST_MARKER.match(value)
                ),
            }
        )
    for stop in range(MIN_UNITS, min(len(rows), MAX_CONTROL_UNITS) + 1):
        window = rows[:stop]
        words = sum(row["words"] for row in window)
        if words < MIN_WORDS:
            continue
        if words > MAX_WORDS:
            return None
        checks = {
            "word_range": MIN_WORDS <= words <= MAX_WORDS,
            "unit_range": MIN_UNITS <= len(window) <= MAX_CONTROL_UNITS,
            "sentence_surface": all(
                row["surface_pass"] for row in window
            ),
            "unique_sentences": len(
                {row["identity_sha256"] for row in window}
            )
            == len(window),
        }
        if not all(checks.values()):
            return None
        passage = " ".join(row["text"] for row in window)
        reparsed = [canonical(item) for item in tokenize(passage)]
        if reparsed != [row["text"] for row in window]:
            return None
        return {
            "passage": passage,
            "passage_sha256": text_sha(passage),
            "normalized_passage_sha256": normalized_hash(passage),
            "words": words,
            "units": len(window),
            "sentences": rows[:stop],
            "checks": checks,
        }
    return None


def _source_record(
    row: dict,
    shard_index: int,
    row_index: int,
    ordinal: int,
) -> dict:
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from utils.text_utils import extract_prompt_from_text

    text = row["text"]
    prompt = extract_prompt_from_text(text, 32)
    return {
        "source_id": f"wm-{ordinal:03d}",
        "shard_index": shard_index,
        "shard_row": row_index,
        "dataset_locator": f"{shard_name(shard_index)}:{row_index}",
        "url": row.get("url", ""),
        "url_sha256": normalized_url_hash(row.get("url", "")),
        "text": text,
        "text_sha256": text_sha(text),
        "normalized_source_sha256": normalized_hash(text),
        "generation_prompt": prompt,
        "generation_prompt_sha256": text_sha(prompt),
        "role": "watermarked_generation_source",
    }


def _control_record(
    row: dict,
    passage: dict,
    shard_index: int,
    row_index: int,
    ordinal: int,
) -> dict:
    split = (
        "calibration"
        if ordinal < HUMAN_CALIBRATION_COUNT
        else "test"
    )
    split_ordinal = (
        ordinal
        if split == "calibration"
        else ordinal - HUMAN_CALIBRATION_COUNT
    )
    text = row["text"]
    return {
        "control_id": f"human-{split}-{split_ordinal:03d}",
        "split": split,
        "shard_index": shard_index,
        "shard_row": row_index,
        "dataset_locator": f"{shard_name(shard_index)}:{row_index}",
        "url": row.get("url", ""),
        "url_sha256": normalized_url_hash(row.get("url", "")),
        "source_text": text,
        "source_text_sha256": text_sha(text),
        "normalized_source_sha256": normalized_hash(text),
        **passage,
        "role": "natural_human_detector_control",
    }


def _scan_watermarked_sources(
    prior_text: set[str],
    prior_url: set[str],
) -> tuple[list[dict], dict, set[str], set[str]]:
    seen_text = set(prior_text)
    seen_url = set(prior_url)
    selected = []
    screening = {
        "rows_scanned": 0,
        "empty_rows": 0,
        "prior_or_duplicate_rows": 0,
        "selected": 0,
        "scanned_through": {},
    }
    for shard_index in range(SOURCE_START_SHARD, SHARD_COUNT):
        stream = stream_train_shard(shard_index)
        try:
            for row_index, row in enumerate(stream):
                screening["rows_scanned"] += 1
                screening["scanned_through"][str(shard_index)] = row_index
                text = row.get("text", "")
                url = row.get("url", "")
                text_identity = normalized_hash(text)
                url_identity = normalized_url_hash(url)
                duplicate = (
                    text_identity in seen_text
                    or bool(url and url_identity in seen_url)
                )
                seen_text.add(text_identity)
                if url:
                    seen_url.add(url_identity)
                if not canonical(text):
                    screening["empty_rows"] += 1
                    continue
                if duplicate:
                    screening["prior_or_duplicate_rows"] += 1
                    continue
                record = _source_record(
                    row,
                    shard_index,
                    row_index,
                    len(selected),
                )
                if not canonical(record["generation_prompt"]):
                    screening["empty_rows"] += 1
                    continue
                selected.append(record)
                if len(selected) == SOURCE_COUNT:
                    screening["selected"] = len(selected)
                    return selected, screening, seen_text, seen_url
        finally:
            stream.close()
    raise RuntimeError("C4 exhausted before 250 watermarked sources")


def _scan_human_controls(
    start_shard: int,
    seen_text: set[str],
    seen_url: set[str],
    tokenize,
) -> tuple[list[dict], dict]:
    selected = []
    screening = {
        "rows_scanned": 0,
        "empty_rows": 0,
        "prior_or_duplicate_rows": 0,
        "surface_filter_failures": 0,
        "selected": 0,
        "scanned_through": {},
    }
    for shard_index in range(start_shard, SHARD_COUNT):
        stream = stream_train_shard(shard_index)
        try:
            for row_index, row in enumerate(stream):
                screening["rows_scanned"] += 1
                screening["scanned_through"][str(shard_index)] = row_index
                text = row.get("text", "")
                url = row.get("url", "")
                text_identity = normalized_hash(text)
                url_identity = normalized_url_hash(url)
                duplicate = (
                    text_identity in seen_text
                    or bool(url and url_identity in seen_url)
                )
                seen_text.add(text_identity)
                if url:
                    seen_url.add(url_identity)
                if not canonical(text):
                    screening["empty_rows"] += 1
                    continue
                if duplicate:
                    screening["prior_or_duplicate_rows"] += 1
                    continue
                passage = human_control_passage(text, tokenize)
                if passage is None:
                    screening["surface_filter_failures"] += 1
                    continue
                selected.append(
                    _control_record(
                        row,
                        passage,
                        shard_index,
                        row_index,
                        len(selected),
                    )
                )
                if len(selected) == HUMAN_COUNT:
                    screening["selected"] = len(selected)
                    return selected, screening
        finally:
            stream.close()
    raise RuntimeError("C4 exhausted before 500 human controls")


def acquire(folder: Path = OUTPUT) -> tuple[dict, dict]:
    configure_environment()
    runtime = check_runtime(folder)
    source_path = folder / "source_manifest.json"
    controls_path = folder / "human_control_manifest.json"
    if source_path.exists() and controls_path.exists():
        return check_allocations(folder)
    if source_path.exists() != controls_path.exists():
        raise RuntimeError("Only one allocation manifest exists; audit manually")

    journal_path = folder / "allocation_journal.json"
    journal = (
        json.loads(journal_path.read_text())
        if journal_path.exists()
        else {"attempts": []}
    )
    if len(journal["attempts"]) >= MAX_ACQUISITION_ATTEMPTS:
        raise RuntimeError("Source acquisition attempt cap reached")
    attempt = {
        "ordinal": len(journal["attempts"]) + 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_flight",
    }
    journal["attempts"].append(attempt)
    write_json(journal_path, journal)

    try:
        import nltk

        snapshot = check_prior_snapshot(folder, runtime)
        sources, source_screening, seen_text, seen_url = (
            _scan_watermarked_sources(
                set(snapshot["normalized_text_sha256"]),
                set(snapshot["url_sha256"]),
            )
        )
        final_source_shard = max(row["shard_index"] for row in sources)
        control_start_shard = final_source_shard + 1
        controls, control_screening = _scan_human_controls(
            control_start_shard,
            seen_text,
            seen_url,
            nltk.sent_tokenize,
        )
        common = {
            "schema_version": 1,
            "status": "frozen_before_generation_or_detection",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": "allenai/c4",
                "configuration": "realnewslike",
                "revision": REVISION,
            },
            "runtime_sha256": sha(folder / "runtime.json"),
            "preregistration_sha256": sha(folder / "manifest.json"),
            "prior_identity_snapshot_sha256": sha(
                folder / "prior_identity_snapshot.json"
            ),
        }
        source_manifest = {
            **common,
            "selection": (
                "first 250 eligible records in increasing shard/row order"
            ),
            "start_shard": SOURCE_START_SHARD,
            "final_shard": final_source_shard,
            "screening": source_screening,
            "sources": sources,
        }
        controls_manifest = {
            **common,
            "selection": (
                "first 500 surface-eligible passages in increasing "
                "shard/row order"
            ),
            "start_shard": control_start_shard,
            "final_shard": max(row["shard_index"] for row in controls),
            "screening": control_screening,
            "calibration_count": HUMAN_CALIBRATION_COUNT,
            "test_count": HUMAN_TEST_COUNT,
            "controls": controls,
        }
        freeze(source_path, source_manifest)
        freeze(controls_path, controls_manifest)
        attempt.update(
            status="complete",
            finished_at=datetime.now(timezone.utc).isoformat(),
            watermarked_sources=len(sources),
            human_controls=len(controls),
            source_manifest_sha256=sha(source_path),
            human_control_manifest_sha256=sha(controls_path),
        )
        write_json(journal_path, journal)
        return source_manifest, controls_manifest
    except Exception as exc:
        attempt.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(exc).__name__,
            allocation_manifests_written=(
                source_path.exists() or controls_path.exists()
            ),
        )
        write_json(journal_path, journal)
        raise


def _validate_source_record(row: dict) -> None:
    sys.path.insert(0, str(ROOT / "third_party/SWAN"))
    from utils.text_utils import extract_prompt_from_text

    if (
        row["role"] != "watermarked_generation_source"
        or row["text_sha256"] != text_sha(row["text"])
        or row["normalized_source_sha256"] != normalized_hash(row["text"])
        or row["url_sha256"] != normalized_url_hash(row["url"])
        or row["generation_prompt"]
        != extract_prompt_from_text(row["text"], 32)
        or row["generation_prompt_sha256"]
        != text_sha(row["generation_prompt"])
    ):
        raise ValueError(f"Watermarked source changed: {row['source_id']}")


def _validate_control_record(row: dict, tokenize) -> None:
    passage = human_control_passage(row["source_text"], tokenize)
    expected = {
        key: passage[key]
        for key in (
            "passage",
            "passage_sha256",
            "normalized_passage_sha256",
            "words",
            "units",
            "sentences",
            "checks",
        )
    } if passage else None
    actual = {key: row[key] for key in expected} if expected else None
    if (
        expected is None
        or actual != expected
        or row["role"] != "natural_human_detector_control"
        or row["source_text_sha256"] != text_sha(row["source_text"])
        or row["normalized_source_sha256"]
        != normalized_hash(row["source_text"])
        or row["url_sha256"] != normalized_url_hash(row["url"])
    ):
        raise ValueError(f"Human control changed: {row['control_id']}")


def check_allocations(folder: Path = OUTPUT) -> tuple[dict, dict]:
    configure_environment()
    import nltk

    runtime = check_runtime(folder)
    snapshot = check_prior_snapshot(folder, runtime)
    source_manifest = json.loads(
        (folder / "source_manifest.json").read_text()
    )
    controls_manifest = json.loads(
        (folder / "human_control_manifest.json").read_text()
    )
    for manifest in (source_manifest, controls_manifest):
        if (
            manifest["status"] != "frozen_before_generation_or_detection"
            or manifest["runtime_sha256"] != sha(folder / "runtime.json")
            or manifest["preregistration_sha256"]
            != sha(folder / "manifest.json")
            or manifest["prior_identity_snapshot_sha256"]
            != sha(folder / "prior_identity_snapshot.json")
        ):
            raise ValueError("Allocation provenance mismatch")

    sources = source_manifest["sources"]
    controls = controls_manifest["controls"]
    if len(sources) != SOURCE_COUNT or len(controls) != HUMAN_COUNT:
        raise ValueError("Allocation count mismatch")
    if [row["source_id"] for row in sources] != [
        f"wm-{index:03d}" for index in range(SOURCE_COUNT)
    ]:
        raise ValueError("Watermarked source order changed")
    if source_manifest["start_shard"] != SOURCE_START_SHARD:
        raise ValueError("Watermarked source shard changed")
    if controls_manifest["start_shard"] <= source_manifest["final_shard"]:
        raise ValueError("Human controls are not on later disjoint shards")

    prior_text = set(snapshot["normalized_text_sha256"])
    prior_url = set(snapshot["url_sha256"])
    all_text = []
    all_url = []
    for row in sources:
        _validate_source_record(row)
        all_text.append(row["normalized_source_sha256"])
        if row["url"]:
            all_url.append(row["url_sha256"])
    for index, row in enumerate(controls):
        expected_split = (
            "calibration"
            if index < HUMAN_CALIBRATION_COUNT
            else "test"
        )
        if row["split"] != expected_split:
            raise ValueError("Human control split changed")
        _validate_control_record(row, nltk.sent_tokenize)
        all_text.append(row["normalized_source_sha256"])
        if row["url"]:
            all_url.append(row["url_sha256"])
    if (
        len(all_text) != len(set(all_text))
        or len(all_url) != len(set(all_url))
        or set(all_text) & prior_text
        or set(all_url) & prior_url
    ):
        raise ValueError("Prior or within-final source overlap detected")
    return source_manifest, controls_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("freeze-runtime", "allocate", "check"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "freeze-runtime":
        runtime = freeze_runtime(args.output)
        payload = {
            "stage": args.stage,
            "status": runtime["status"],
            "prior_identities": json.loads(
                (
                    args.output / "prior_identity_snapshot.json"
                ).read_text()
            )["counts"],
        }
    elif args.stage == "allocate":
        sources, controls = acquire(args.output)
        payload = {
            "stage": args.stage,
            "watermarked_sources": len(sources["sources"]),
            "human_calibration": controls["calibration_count"],
            "human_test": controls["test_count"],
        }
    else:
        sources, controls = check_allocations(args.output)
        payload = {
            "stage": args.stage,
            "status": "verified",
            "watermarked_sources": len(sources["sources"]),
            "human_controls": len(controls["controls"]),
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
