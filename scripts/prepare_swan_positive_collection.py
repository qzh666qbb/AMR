"""Prepare the locked, independent SWAN positive-document collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PROTOCOL = ROOT / "experiments/swan_positive_asset_audit1"
OUTPUT = ROOT / "experiments/swan_positive_collection1"

REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SWAN_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
SHARD_COUNT = 512
SOURCE_START_SHARD = 8
PILOT_COUNT = 5
MAIN_COUNT = 30
SOURCE_COUNT = PILOT_COUNT + MAIN_COUNT

TEXT_IDENTITY_KEYS = {
    "article",
    "article_text",
    "attacked_text",
    "baseline_text",
    "candidate_text",
    "context",
    "generated_text",
    "generated_texts",
    "generation_prompt",
    "human_text",
    "original",
    "original_passage",
    "original_text",
    "passage",
    "prompt",
    "prompts",
    "source",
    "source_text",
    "text",
    "texts",
    "watermarked_text",
    "watermarked_texts",
}
HASH_IDENTITY_KEYS = {
    "identity_sha256",
    "normalized_passage_sha256",
    "normalized_sha256",
    "normalized_source_sha256",
    "normalized_text_sha256",
}
URL_KEYS = {"source_url", "url"}
HEX_DIGITS = set("0123456789abcdef")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_hash(text: str) -> str:
    return text_sha256(canonical(text).casefold())


def normalized_url_hash(url: str) -> str:
    return text_sha256(canonical(url))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def freeze_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def protocol_inputs() -> dict:
    protocol_path = AUDIT_PROTOCOL / "protocol.md"
    audit_manifest_path = AUDIT_PROTOCOL / "manifest.json"
    audit_manifest = load_json(audit_manifest_path)
    actual_protocol_hash = sha256(protocol_path)
    if actual_protocol_hash != audit_manifest["protocol_sha256"]:
        raise ValueError("Positive-asset protocol hash does not match manifest")
    collection = audit_manifest["new_collection"]
    expected = {
        "revision": REVISION,
        "start_train_shard": SOURCE_START_SHARD,
        "pilot": PILOT_COUNT,
        "main": MAIN_COUNT,
        "swan_commit": SWAN_COMMIT,
    }
    actual = {
        "revision": collection["dataset"]["revision"],
        "start_train_shard": collection["dataset"]["start_train_shard"],
        "pilot": collection["allocation"]["pilot"],
        "main": collection["allocation"]["main"],
        "swan_commit": collection["generation"]["swan_commit"],
    }
    if actual != expected or collection["execution_authorized"]:
        raise ValueError("Frozen collection contract does not match allocator")
    return {
        "protocol_sha256": actual_protocol_hash,
        "asset_audit_manifest_sha256": sha256(audit_manifest_path),
        "asset_audit_manifest": audit_manifest,
    }


def _contains_identity_key(content: bytes) -> bool:
    keys = TEXT_IDENTITY_KEYS | HASH_IDENTITY_KEYS | URL_KEYS
    return any(f'"{key}"'.encode("ascii") in content for key in keys)


def discover_history_artifacts(
    root: Path = ROOT,
    excluded_tree: Path = OUTPUT,
) -> tuple[list[Path], int]:
    experiments = root / "experiments"
    paths = []
    scanned = 0
    excluded_tree = excluded_tree.resolve()
    for suffix in ("*.json", "*.jsonl"):
        for path in experiments.rglob(suffix):
            resolved = path.resolve()
            if resolved == excluded_tree or excluded_tree in resolved.parents:
                continue
            scanned += 1
            if _contains_identity_key(path.read_bytes()):
                paths.append(path)
    return sorted(set(paths)), scanned


def _load_history_artifact(path: Path) -> list:
    content = path.read_bytes().decode("utf-8-sig")
    if path.suffix == ".jsonl":
        values = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid history JSONL {path}:{line_number}"
                ) from exc
        return values
    try:
        return [json.loads(content)]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid history JSON {path}") from exc


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value).issubset(HEX_DIGITS)


def _collect_identity_values(
    value,
    text_hashes: set[str],
    url_hashes: set[str],
    field_counts: dict[str, int],
    inherited_key: str = "",
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_identity_values(
                item,
                text_hashes,
                url_hashes,
                field_counts,
                str(key).casefold(),
            )
        return
    if isinstance(value, list):
        for item in value:
            _collect_identity_values(
                item,
                text_hashes,
                url_hashes,
                field_counts,
                inherited_key,
            )
        return
    if not isinstance(value, str):
        return

    if inherited_key in HASH_IDENTITY_KEYS and _is_sha256(value):
        text_hashes.add(value)
        field_counts[inherited_key] = field_counts.get(inherited_key, 0) + 1
    elif inherited_key in TEXT_IDENTITY_KEYS and canonical(value):
        text_hashes.add(normalized_hash(value))
        field_counts[inherited_key] = field_counts.get(inherited_key, 0) + 1
    if inherited_key in URL_KEYS and canonical(value):
        url_hashes.add(normalized_url_hash(value))
        field_counts[inherited_key] = field_counts.get(inherited_key, 0) + 1


def build_prior_identity_snapshot(
    paths: list[Path],
    root: Path = ROOT,
    all_files_scanned: int = 0,
    created_at: str = "",
) -> dict:
    text_hashes: set[str] = set()
    url_hashes: set[str] = set()
    field_counts: dict[str, int] = {}
    inventory = []
    for path in paths:
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        for value in _load_history_artifact(path):
            _collect_identity_values(
                value,
                text_hashes,
                url_hashes,
                field_counts,
            )
    return {
        "schema_version": 1,
        "created_at": created_at or utc_now(),
        "status": "frozen_before_new_source_allocation",
        "selection": {
            "root": "experiments",
            "extensions": [".json", ".jsonl"],
            "excluded_tree": str(OUTPUT.relative_to(ROOT)),
            "candidate_rule": (
                "Parse every JSON/JSONL artifact containing an exact "
                "identity-field key; no prior experiment tree is excluded."
            ),
            "includes_terminated_swan_final_confirmation1": True,
            "text_keys": sorted(TEXT_IDENTITY_KEYS),
            "hash_keys": sorted(HASH_IDENTITY_KEYS),
            "url_keys": sorted(URL_KEYS),
            "normalization": "NFKC, whitespace collapse, casefold for text",
        },
        "scan_counts": {
            "all_json_jsonl_files": all_files_scanned,
            "identity_key_candidate_files": len(paths),
        },
        "artifact_inventory": inventory,
        "artifact_inventory_digest": json_digest(inventory),
        "field_value_counts": dict(sorted(field_counts.items())),
        "normalized_text_sha256": sorted(text_hashes),
        "url_sha256": sorted(url_hashes),
        "counts": {
            "artifacts": len(inventory),
            "normalized_text_identities": len(text_hashes),
            "url_identities": len(url_hashes),
        },
    }


def _required_local_files(asset_manifest: dict) -> dict[str, list[Path]]:
    upstream = [
        ROOT / "third_party/SWAN" / relative
        for relative in asset_manifest["official_snapshot"]["tracked_paths"]
    ]
    preparation = [
        Path(__file__),
        ROOT / "scripts/audit_swan_positive_collection.py",
        ROOT / "scripts/test_swan_positive_collection.py",
    ]
    parser = sorted(
        path
        for path in (
            ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
        ).glob("*")
        if path.is_file()
    )
    tokenizer = sorted(
        path
        for path in (ROOT / "models/bart-large-tokenizer").glob("*")
        if path.is_file()
    )
    lexical_and_segmentation = [
        ROOT / "vectors/glove.6B.100d.txt",
        ROOT / ".nltk_data/corpora/propbank.zip",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
        ROOT / ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    ]
    groups = {
        "upstream_swan_commit_files": upstream,
        "preparation_and_audit_code": preparation,
        "amr_parser_weights": parser,
        "amr_tokenizer": tokenizer,
        "lexical_and_segmentation_resources": lexical_and_segmentation,
    }
    missing = [
        path
        for paths in groups.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing frozen dependency: " + ", ".join(map(str, missing))
        )
    return groups


def _dependency_hashes(asset_manifest: dict) -> dict:
    return {
        group: {
            str(path.relative_to(ROOT)): sha256(path)
            for path in paths
        }
        for group, paths in _required_local_files(asset_manifest).items()
    }


def _package_versions() -> dict[str, str]:
    result = {}
    for name in (
        "amrlib",
        "boto3",
        "datasets",
        "nltk",
        "numpy",
        "penman",
        "requests",
        "scipy",
        "torch",
        "transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def freeze_local_runtime(folder: Path = OUTPUT) -> dict:
    inputs = protocol_inputs()
    folder.mkdir(parents=True, exist_ok=True)
    prior_path = folder / "prior_identity_snapshot.json"
    runtime_path = folder / "runtime.json"
    source_path = folder / "source_manifest.json"

    if prior_path.exists() or runtime_path.exists():
        if not prior_path.exists() or not runtime_path.exists():
            raise RuntimeError("Partial runtime freeze requires manual audit")
        return check_local_runtime(folder)
    if source_path.exists():
        raise RuntimeError("Source manifest exists before runtime freeze")

    paths, scanned = discover_history_artifacts(
        root=ROOT,
        excluded_tree=folder,
    )
    snapshot = build_prior_identity_snapshot(
        paths,
        all_files_scanned=scanned,
    )
    freeze_json(prior_path, snapshot)

    collection = inputs["asset_audit_manifest"]["new_collection"]
    packages = _package_versions()
    runtime = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "local_runtime_frozen_remote_execution_unauthorized",
        "execution_authorized": False,
        "protocol_sha256": inputs["protocol_sha256"],
        "asset_audit_manifest_sha256": inputs[
            "asset_audit_manifest_sha256"
        ],
        "prior_identity_snapshot_sha256": sha256(prior_path),
        "dataset": {
            "name": "allenai/c4",
            "configuration": "realnewslike",
            "revision": REVISION,
            "shard_count": SHARD_COUNT,
            "start_train_shard": SOURCE_START_SHARD,
        },
        "allocation": {
            "pilot": PILOT_COUNT,
            "main": MAIN_COUNT,
            "freeze_all_before_generation": True,
            "replacement_allowed": False,
        },
        "upstream_swan": {
            "commit": SWAN_COMMIT,
            "local_root": "third_party/SWAN",
            "tracked_file_count": len(
                inputs["asset_audit_manifest"]["official_snapshot"][
                    "tracked_paths"
                ]
            ),
        },
        "generation": {
            **collection["generation"],
            "region": "us-east-1",
            "random_seed": 20260721,
            "region_and_seed_source": (
                "Defaults in the frozen upstream watermark_generation.py"
            ),
        },
        "remote_model": {
            "requested": collection["generation"]["requested_model"],
            "returned": None,
            "region": "us-east-1",
            "model_probe_authorized": False,
            "generation_authorized": False,
            "identity_frozen_by": (
                "model_probe.json after explicit paid-budget authorization"
            ),
        },
        "dependencies": _dependency_hashes(
            inputs["asset_audit_manifest"]
        ),
        "packages": packages,
        "resource_policy": {
            "public_c4_download_allowed": True,
            "paid_api_calls_allowed": False,
            "model_loading_allowed": False,
            "parser_calls_allowed": False,
            "s2match_calls_allowed": False,
            "detector_calls_allowed": False,
            "attack_calls_allowed": False,
        },
    }
    freeze_json(runtime_path, runtime)
    return runtime


def check_prior_snapshot(folder: Path, runtime: dict) -> dict:
    path = folder / "prior_identity_snapshot.json"
    if sha256(path) != runtime["prior_identity_snapshot_sha256"]:
        raise ValueError("Prior identity snapshot hash changed")
    snapshot = load_json(path)
    inventory = snapshot["artifact_inventory"]
    if json_digest(inventory) != snapshot["artifact_inventory_digest"]:
        raise ValueError("Prior artifact inventory digest changed")
    for row in inventory:
        artifact = ROOT / row["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row["bytes"]
            or sha256(artifact) != row["sha256"]
        ):
            raise ValueError(f"Prior artifact changed: {row['path']}")
    if snapshot["counts"] != {
        "artifacts": len(inventory),
        "normalized_text_identities": len(
            snapshot["normalized_text_sha256"]
        ),
        "url_identities": len(snapshot["url_sha256"]),
    }:
        raise ValueError("Prior identity snapshot counts changed")
    required_prior = (
        "experiments/swan_final_confirmation1/source_manifest.json"
    )
    if required_prior not in {row["path"] for row in inventory}:
        raise ValueError("Terminated final run is absent from prior snapshot")
    return snapshot


def check_local_runtime(folder: Path = OUTPUT) -> dict:
    inputs = protocol_inputs()
    runtime = load_json(folder / "runtime.json")
    if (
        runtime["status"]
        != "local_runtime_frozen_remote_execution_unauthorized"
        or runtime["execution_authorized"]
        or runtime["protocol_sha256"] != inputs["protocol_sha256"]
        or runtime["asset_audit_manifest_sha256"]
        != inputs["asset_audit_manifest_sha256"]
        or runtime["remote_model"]["returned"] is not None
        or runtime["remote_model"]["model_probe_authorized"]
        or runtime["remote_model"]["generation_authorized"]
    ):
        raise ValueError("Runtime execution lock or provenance changed")
    for group in runtime["dependencies"].values():
        for relative, expected in group.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Frozen dependency changed: {relative}")
    if _package_versions() != runtime["packages"]:
        raise ValueError("Frozen package environment changed")
    check_prior_snapshot(folder, runtime)
    return runtime


def shard_name(index: int) -> str:
    if index not in range(SHARD_COUNT):
        raise ValueError(f"Invalid C4 shard index: {index}")
    return (
        f"realnewslike/c4-train.{index:05d}-of-{SHARD_COUNT:05d}.json.gz"
    )


def shard_url(index: int) -> str:
    return (
        "https://huggingface.co/datasets/allenai/c4/resolve/"
        f"{REVISION}/{shard_name(index)}"
    )


def stream_train_shard(index: int):
    import requests

    url = shard_url(index)
    headers = {"User-Agent": "SWAN-positive-source-audit/1.0"}
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(20, 90),
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for line in handle:
                yield json.loads(line)


def _extract_prompt(text: str) -> str:
    swan_root = str(ROOT / "third_party/SWAN")
    if swan_root not in sys.path:
        sys.path.insert(0, swan_root)
    from utils.text_utils import extract_prompt_from_text

    return extract_prompt_from_text(text, 32)


def _screening_record(
    row: dict,
    shard_index: int,
    row_index: int,
    seen_text: set[str],
    seen_url: set[str],
) -> dict:
    text = row.get("text", "")
    url = row.get("url", "")
    if not isinstance(text, str):
        text = ""
    if not isinstance(url, str):
        url = ""
    normalized_text = normalized_hash(text)
    normalized_url = normalized_url_hash(url)
    prompt = _extract_prompt(text) if canonical(text) else ""
    checks = {
        "nonempty_source": bool(canonical(text)),
        "nonempty_prompt": bool(canonical(prompt)),
        "normalized_text_unique": normalized_text not in seen_text,
        "url_unique": not url or normalized_url not in seen_url,
    }
    selected = all(checks.values())
    return {
        "shard_index": shard_index,
        "shard_row": row_index,
        "dataset_locator": f"{shard_name(shard_index)}:{row_index}",
        "text": text,
        "text_sha256": text_sha256(text),
        "normalized_source_sha256": normalized_text,
        "url": url,
        "url_sha256": normalized_url,
        "generation_prompt": prompt,
        "generation_prompt_sha256": text_sha256(prompt),
        "checks": checks,
        "decision": "selected" if selected else "excluded",
    }


def _source_record(screen: dict, ordinal: int, pilot_count: int) -> dict:
    cohort = "pilot" if ordinal < pilot_count else "main"
    cohort_ordinal = ordinal if cohort == "pilot" else ordinal - pilot_count
    allocation_id = f"{cohort}-{cohort_ordinal:03d}"
    return {
        "allocation_id": allocation_id,
        "allocation_ordinal": ordinal,
        "cohort": cohort,
        "cohort_ordinal": cohort_ordinal,
        **{
            key: screen[key]
            for key in (
                "shard_index",
                "shard_row",
                "dataset_locator",
                "text",
                "text_sha256",
                "normalized_source_sha256",
                "url",
                "url_sha256",
                "generation_prompt",
                "generation_prompt_sha256",
            )
        },
        "role": "watermarked_generation_source",
    }


def scan_sources(
    snapshot: dict,
    source_count: int = SOURCE_COUNT,
    pilot_count: int = PILOT_COUNT,
    start_shard: int = SOURCE_START_SHARD,
) -> tuple[list[dict], list[dict], dict]:
    seen_text = set(snapshot["normalized_text_sha256"])
    seen_url = set(snapshot["url_sha256"])
    selected = []
    screened = []
    exclusion_counts = {
        "empty_source": 0,
        "empty_prompt": 0,
        "normalized_text_collision": 0,
        "url_collision": 0,
    }

    for shard_index in range(start_shard, SHARD_COUNT):
        stream = stream_train_shard(shard_index)
        try:
            for row_index, row in enumerate(stream):
                screen = _screening_record(
                    row,
                    shard_index,
                    row_index,
                    seen_text,
                    seen_url,
                )
                screened.append(screen)
                checks = screen["checks"]
                if not checks["nonempty_source"]:
                    exclusion_counts["empty_source"] += 1
                if not checks["nonempty_prompt"]:
                    exclusion_counts["empty_prompt"] += 1
                if not checks["normalized_text_unique"]:
                    exclusion_counts["normalized_text_collision"] += 1
                if not checks["url_unique"]:
                    exclusion_counts["url_collision"] += 1
                if screen["decision"] == "excluded":
                    continue
                selected.append(
                    _source_record(screen, len(selected), pilot_count)
                )
                seen_text.add(screen["normalized_source_sha256"])
                if screen["url"]:
                    seen_url.add(screen["url_sha256"])
                if len(selected) == source_count:
                    return selected, screened, {
                        "rows_scanned": len(screened),
                        "selected": len(selected),
                        "excluded": len(screened) - len(selected),
                        "exclusion_condition_counts": exclusion_counts,
                        "scanned_through": {
                            "shard_index": shard_index,
                            "shard_row": row_index,
                        },
                    }
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    raise RuntimeError(
        f"C4 exhausted before {source_count} eligible source allocations"
    )


def validate_source_manifest(
    manifest: dict,
    snapshot: dict,
    runtime_sha256: str,
    folder: Path = OUTPUT,
) -> None:
    if (
        manifest["status"]
        != "sources_frozen_remote_generation_unauthorized"
        or manifest["execution_authorized"]
        or manifest["runtime_sha256"] != runtime_sha256
        or manifest["protocol_sha256"]
        != sha256(AUDIT_PROTOCOL / "protocol.md")
        or manifest["prior_identity_snapshot_sha256"]
        != sha256(folder / "prior_identity_snapshot.json")
        or manifest["start_shard"] != SOURCE_START_SHARD
    ):
        raise ValueError("Source manifest lock or provenance changed")

    sources = manifest["sources"]
    screened = manifest["screened_rows"]
    if (
        len(sources) != SOURCE_COUNT
        or manifest["cohort_counts"]
        != {"pilot": PILOT_COUNT, "main": MAIN_COUNT}
        or manifest["screening"]["rows_scanned"] != len(screened)
        or manifest["screening"]["selected"] != len(sources)
        or screened[-1]["decision"] != "selected"
    ):
        raise ValueError("Source allocation counts changed")

    prior_text = set(snapshot["normalized_text_sha256"])
    prior_url = set(snapshot["url_sha256"])
    expected_sources = []
    previous_position = None
    exclusion_counts = {
        "empty_source": 0,
        "empty_prompt": 0,
        "normalized_text_collision": 0,
        "url_collision": 0,
    }
    for saved in screened:
        position = (saved["shard_index"], saved["shard_row"])
        if previous_position is not None and position <= previous_position:
            raise ValueError("Screened C4 rows are not strictly ordered")
        previous_position = position
        rebuilt = _screening_record(
            saved,
            saved["shard_index"],
            saved["shard_row"],
            prior_text,
            prior_url,
        )
        if rebuilt != saved:
            raise ValueError(
                f"Screening evidence changed: {saved['dataset_locator']}"
            )
        checks = rebuilt["checks"]
        if not checks["nonempty_source"]:
            exclusion_counts["empty_source"] += 1
        if not checks["nonempty_prompt"]:
            exclusion_counts["empty_prompt"] += 1
        if not checks["normalized_text_unique"]:
            exclusion_counts["normalized_text_collision"] += 1
        if not checks["url_unique"]:
            exclusion_counts["url_collision"] += 1
        if rebuilt["decision"] == "selected":
            expected_sources.append(
                _source_record(
                    rebuilt,
                    len(expected_sources),
                    PILOT_COUNT,
                )
            )
            prior_text.add(rebuilt["normalized_source_sha256"])
            if rebuilt["url"]:
                prior_url.add(rebuilt["url_sha256"])
    if expected_sources != sources:
        raise ValueError("Saved sources do not replay from screening evidence")
    if (
        manifest["screening"]["excluded"]
        != len(screened) - len(sources)
        or manifest["screening"]["exclusion_condition_counts"]
        != exclusion_counts
        or manifest["screening"]["scanned_through"]
        != {
            "shard_index": screened[-1]["shard_index"],
            "shard_row": screened[-1]["shard_row"],
        }
    ):
        raise ValueError("Screening summary does not match row evidence")


def check_source_allocation(folder: Path = OUTPUT) -> dict:
    runtime = check_local_runtime(folder)
    snapshot = check_prior_snapshot(folder, runtime)
    manifest = load_json(folder / "source_manifest.json")
    validate_source_manifest(
        manifest,
        snapshot,
        sha256(folder / "runtime.json"),
        folder,
    )
    return manifest


def acquire_sources(folder: Path = OUTPUT) -> dict:
    runtime = check_local_runtime(folder)
    source_path = folder / "source_manifest.json"
    if source_path.exists():
        return check_source_allocation(folder)

    journal_path = folder / "allocation_journal.json"
    journal = (
        load_json(journal_path)
        if journal_path.exists()
        else {
            "schema_version": 1,
            "purpose": "Public C4 source acquisition only",
            "paid_or_model_calls": 0,
            "attempts": [],
        }
    )
    attempt = {
        "ordinal": len(journal["attempts"]) + 1,
        "started_at": utc_now(),
        "status": "in_flight",
        "dataset_revision": REVISION,
        "start_shard": SOURCE_START_SHARD,
        "first_request_url": shard_url(SOURCE_START_SHARD),
    }
    journal["attempts"].append(attempt)
    write_json(journal_path, journal)

    try:
        snapshot = check_prior_snapshot(folder, runtime)
        sources, screened, screening = scan_sources(snapshot)
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "sources_frozen_remote_generation_unauthorized",
            "execution_authorized": False,
            "dataset": runtime["dataset"],
            "selection": (
                "First 35 nonempty, prompt-nonempty, prior/new-identity-"
                "unique records in increasing shard and row order"
            ),
            "start_shard": SOURCE_START_SHARD,
            "final_shard": sources[-1]["shard_index"],
            "cohort_counts": {
                "pilot": PILOT_COUNT,
                "main": MAIN_COUNT,
            },
            "replacement_allowed": False,
            "runtime_sha256": sha256(folder / "runtime.json"),
            "protocol_sha256": runtime["protocol_sha256"],
            "prior_identity_snapshot_sha256": sha256(
                folder / "prior_identity_snapshot.json"
            ),
            "screening": screening,
            "screened_rows": screened,
            "sources": sources,
        }
        freeze_json(source_path, manifest)
        attempt.update(
            status="complete",
            finished_at=utc_now(),
            public_dataset_http_requests=len(
                {row["shard_index"] for row in screened}
            ),
            rows_scanned=len(screened),
            sources_frozen=len(sources),
            source_manifest_sha256=sha256(source_path),
        )
        write_json(journal_path, journal)
        return check_source_allocation(folder)
    except Exception as exc:
        attempt.update(
            status="failed",
            finished_at=utc_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
            source_manifest_written=source_path.exists(),
            paid_or_model_calls=0,
        )
        write_json(journal_path, journal)
        raise


def finalize_pre_execution(folder: Path = OUTPUT) -> dict:
    runtime = check_local_runtime(folder)
    sources = check_source_allocation(folder)
    path = folder / "pre_execution_manifest.json"
    if path.exists():
        return check_pre_execution(folder)
    package_blockers = [
        name
        for name, version in runtime["packages"].items()
        if version == "not_installed"
    ]
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "local_preparation_complete_remote_execution_unauthorized",
        "execution_authorized": False,
        "completed": {
            "protocol_frozen": True,
            "prior_identity_snapshot_frozen": True,
            "all_35_sources_frozen": True,
            "pilot_main_membership_frozen": True,
            "local_dependency_hashes_frozen": True,
            "upstream_source_hashes_frozen": True,
            "parser_and_tokenizer_hashes_frozen": True,
        },
        "pending": {
            "explicit_paid_budget_authorization": True,
            "returned_model_identity_probe": True,
            "model_probe_json": True,
            "remote_generation": True,
        },
        "package_state": {
            "not_installed": package_blockers,
            "interpretation": (
                "Recorded environment state, not authorization to install "
                "or invoke any remote model."
            ),
        },
        "cohorts": sources["cohort_counts"],
        "requested_model": runtime["remote_model"]["requested"],
        "returned_model": None,
        "region": runtime["remote_model"]["region"],
        "hashes": {
            "protocol_sha256": runtime["protocol_sha256"],
            "asset_audit_manifest_sha256": runtime[
                "asset_audit_manifest_sha256"
            ],
            "prior_identity_snapshot_sha256": sha256(
                folder / "prior_identity_snapshot.json"
            ),
            "runtime_sha256": sha256(folder / "runtime.json"),
            "source_manifest_sha256": sha256(
                folder / "source_manifest.json"
            ),
            "allocation_journal_sha256": sha256(
                folder / "allocation_journal.json"
            ),
        },
        "resource_counts": {
            "public_dataset_http_requests": sum(
                attempt.get("public_dataset_http_requests", 0)
                for attempt in load_json(
                    folder / "allocation_journal.json"
                )["attempts"]
            ),
            "paid_api_calls": 0,
            "model_probe_calls": 0,
            "generation_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
        "next_valid_transition": (
            "Explicit paid-budget authorization, then one model identity "
            "probe; generation remains blocked until that probe is frozen."
        ),
    }
    freeze_json(path, manifest)
    return manifest


def check_pre_execution(folder: Path = OUTPUT) -> dict:
    runtime = check_local_runtime(folder)
    sources = check_source_allocation(folder)
    manifest = load_json(folder / "pre_execution_manifest.json")
    expected_hashes = {
        "protocol_sha256": runtime["protocol_sha256"],
        "asset_audit_manifest_sha256": runtime[
            "asset_audit_manifest_sha256"
        ],
        "prior_identity_snapshot_sha256": sha256(
            folder / "prior_identity_snapshot.json"
        ),
        "runtime_sha256": sha256(folder / "runtime.json"),
        "source_manifest_sha256": sha256(folder / "source_manifest.json"),
        "allocation_journal_sha256": sha256(
            folder / "allocation_journal.json"
        ),
    }
    if (
        manifest["status"]
        != "local_preparation_complete_remote_execution_unauthorized"
        or manifest["execution_authorized"]
        or manifest["hashes"] != expected_hashes
        or manifest["cohorts"] != sources["cohort_counts"]
        or manifest["returned_model"] is not None
        or manifest["resource_counts"]["paid_api_calls"] != 0
        or manifest["resource_counts"]["model_probe_calls"] != 0
        or manifest["resource_counts"]["generation_calls"] != 0
    ):
        raise ValueError("Pre-execution lock or hashes changed")
    forbidden = [
        folder / "model_probe.json",
        folder / "quality_labels.json",
        folder / "original_detection.json",
    ]
    if any(path.exists() for path in forbidden):
        raise ValueError("Unauthorized downstream artifact exists")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "freeze-runtime",
            "allocate",
            "finalize",
            "all",
            "check",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.stage in {"freeze-runtime", "all"}:
        freeze_local_runtime(args.output)
    if args.stage in {"allocate", "all"}:
        acquire_sources(args.output)
    if args.stage in {"finalize", "all"}:
        finalize_pre_execution(args.output)

    if args.stage == "freeze-runtime":
        runtime = check_local_runtime(args.output)
        snapshot = load_json(args.output / "prior_identity_snapshot.json")
        payload = {
            "stage": args.stage,
            "status": runtime["status"],
            "prior_identity_counts": snapshot["counts"],
        }
    elif args.stage == "allocate":
        sources = check_source_allocation(args.output)
        payload = {
            "stage": args.stage,
            "status": sources["status"],
            "cohort_counts": sources["cohort_counts"],
        }
    else:
        manifest = check_pre_execution(args.output)
        payload = {
            "stage": args.stage,
            "status": manifest["status"],
            "execution_authorized": manifest["execution_authorized"],
            "cohorts": manifest["cohorts"],
            "resource_counts": manifest["resource_counts"],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
