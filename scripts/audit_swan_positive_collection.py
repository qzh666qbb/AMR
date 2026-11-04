"""Independent offline audit of SWAN positive-collection preparation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "experiments/swan_positive_asset_audit1"
OUTPUT = ROOT / "experiments/swan_positive_collection1"
EXPECTED_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
EXPECTED_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
EXPECTED_START_SHARD = 8
EXPECTED_PILOT = 5
EXPECTED_MAIN = 30


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


def write_frozen_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen audit artifact differs: {path}")
    path.write_text(text, encoding="utf-8")


def write_frozen_text(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen audit artifact differs: {path}")
    path.write_text(text, encoding="utf-8")


def extract_prompt(text: str) -> str:
    swan_root = str(ROOT / "third_party/SWAN")
    if swan_root not in sys.path:
        sys.path.insert(0, swan_root)
    from utils.text_utils import extract_prompt_from_text

    return extract_prompt_from_text(text, 32)


def package_versions(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def verify_protocol() -> dict:
    manifest = load_json(PROTOCOL_DIR / "manifest.json")
    protocol_hash = sha256(PROTOCOL_DIR / "protocol.md")
    collection = manifest["new_collection"]
    checks = {
        "protocol_hash_matches_asset_audit": (
            protocol_hash == manifest["protocol_sha256"]
        ),
        "collection_declared_unauthorized": (
            collection["execution_authorized"] is False
        ),
        "dataset_revision_matches": (
            collection["dataset"]["revision"] == EXPECTED_REVISION
        ),
        "starts_at_train_shard_8": (
            collection["dataset"]["start_train_shard"]
            == EXPECTED_START_SHARD
        ),
        "pilot_main_counts_match": (
            collection["allocation"]["pilot"] == EXPECTED_PILOT
            and collection["allocation"]["main"] == EXPECTED_MAIN
        ),
        "upstream_commit_matches": (
            collection["generation"]["swan_commit"] == EXPECTED_COMMIT
        ),
    }
    if not all(checks.values()):
        raise ValueError("Frozen protocol contract failed")
    return {
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "asset_audit_manifest_sha256": sha256(
            PROTOCOL_DIR / "manifest.json"
        ),
    }


def verify_prior_snapshot(folder: Path) -> dict:
    path = folder / "prior_identity_snapshot.json"
    snapshot = load_json(path)
    inventory = snapshot["artifact_inventory"]
    inventory_paths = {row["path"] for row in inventory}
    checks = {
        "status_frozen_before_allocation": (
            snapshot["status"] == "frozen_before_new_source_allocation"
        ),
        "no_prior_experiment_excluded": (
            snapshot["selection"][
                "includes_terminated_swan_final_confirmation1"
            ]
            is True
        ),
        "terminated_source_manifest_in_inventory": (
            "experiments/swan_final_confirmation1/source_manifest.json"
            in inventory_paths
        ),
        "inventory_digest_matches": (
            json_digest(inventory)
            == snapshot["artifact_inventory_digest"]
        ),
        "counts_match": (
            snapshot["counts"]
            == {
                "artifacts": len(inventory),
                "normalized_text_identities": len(
                    snapshot["normalized_text_sha256"]
                ),
                "url_identities": len(snapshot["url_sha256"]),
            }
        ),
        "identity_lists_unique": (
            len(snapshot["normalized_text_sha256"])
            == len(set(snapshot["normalized_text_sha256"]))
            and len(snapshot["url_sha256"])
            == len(set(snapshot["url_sha256"]))
        ),
    }
    changed = []
    for row in inventory:
        artifact = ROOT / row["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row["bytes"]
            or sha256(artifact) != row["sha256"]
        ):
            changed.append(row["path"])
    checks["all_inventory_files_unchanged"] = not changed
    if not all(checks.values()):
        raise ValueError(
            "Prior snapshot audit failed"
            + (f"; changed={changed[:5]}" if changed else "")
        )
    return {
        "checks": checks,
        "sha256": sha256(path),
        "counts": snapshot["counts"],
        "text_identities": set(snapshot["normalized_text_sha256"]),
        "url_identities": set(snapshot["url_sha256"]),
    }


def verify_runtime(folder: Path, protocol: dict, prior: dict) -> dict:
    path = folder / "runtime.json"
    runtime = load_json(path)
    checks = {
        "status_is_remote_locked": (
            runtime["status"]
            == "local_runtime_frozen_remote_execution_unauthorized"
        ),
        "execution_not_authorized": (
            runtime["execution_authorized"] is False
        ),
        "protocol_hash_matches": (
            runtime["protocol_sha256"] == protocol["protocol_sha256"]
        ),
        "asset_manifest_hash_matches": (
            runtime["asset_audit_manifest_sha256"]
            == protocol["asset_audit_manifest_sha256"]
        ),
        "prior_snapshot_hash_matches": (
            runtime["prior_identity_snapshot_sha256"] == prior["sha256"]
        ),
        "dataset_matches": (
            runtime["dataset"]["revision"] == EXPECTED_REVISION
            and runtime["dataset"]["start_train_shard"]
            == EXPECTED_START_SHARD
        ),
        "upstream_commit_matches": (
            runtime["upstream_swan"]["commit"] == EXPECTED_COMMIT
        ),
        "model_identity_unprobed": (
            runtime["remote_model"]["returned"] is None
            and runtime["remote_model"]["model_probe_authorized"] is False
        ),
        "generation_unauthorized": (
            runtime["remote_model"]["generation_authorized"] is False
            and runtime["resource_policy"]["paid_api_calls_allowed"] is False
        ),
    }
    dependency_count = 0
    changed_dependencies = []
    for group in runtime["dependencies"].values():
        for relative, expected in group.items():
            dependency_count += 1
            path_to_check = ROOT / relative
            if (
                not path_to_check.is_file()
                or sha256(path_to_check) != expected
            ):
                changed_dependencies.append(relative)
    checks["all_dependency_hashes_match"] = not changed_dependencies
    checks["package_state_matches"] = (
        package_versions(sorted(runtime["packages"]))
        == runtime["packages"]
    )
    if not all(checks.values()):
        raise ValueError(
            "Runtime audit failed"
            + (
                f"; changed_dependencies={changed_dependencies[:5]}"
                if changed_dependencies
                else ""
            )
        )
    return {
        "checks": checks,
        "sha256": sha256(path),
        "dependency_files": dependency_count,
        "packages": runtime["packages"],
    }


def rebuild_screening(
    saved: dict,
    seen_text: set[str],
    seen_url: set[str],
) -> dict:
    text = saved["text"]
    url = saved["url"]
    prompt = extract_prompt(text) if canonical(text) else ""
    text_identity = normalized_hash(text)
    url_identity = normalized_url_hash(url)
    checks = {
        "nonempty_source": bool(canonical(text)),
        "nonempty_prompt": bool(canonical(prompt)),
        "normalized_text_unique": text_identity not in seen_text,
        "url_unique": not url or url_identity not in seen_url,
    }
    return {
        "shard_index": saved["shard_index"],
        "shard_row": saved["shard_row"],
        "dataset_locator": (
            "realnewslike/c4-train."
            f"{saved['shard_index']:05d}-of-00512.json.gz:"
            f"{saved['shard_row']}"
        ),
        "text": text,
        "text_sha256": text_sha256(text),
        "normalized_source_sha256": text_identity,
        "url": url,
        "url_sha256": url_identity,
        "generation_prompt": prompt,
        "generation_prompt_sha256": text_sha256(prompt),
        "checks": checks,
        "decision": "selected" if all(checks.values()) else "excluded",
    }


def source_from_screen(screen: dict, ordinal: int) -> dict:
    cohort = "pilot" if ordinal < EXPECTED_PILOT else "main"
    cohort_ordinal = (
        ordinal if cohort == "pilot" else ordinal - EXPECTED_PILOT
    )
    return {
        "allocation_id": f"{cohort}-{cohort_ordinal:03d}",
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


def verify_sources(
    folder: Path,
    protocol: dict,
    prior: dict,
    runtime: dict,
) -> dict:
    path = folder / "source_manifest.json"
    manifest = load_json(path)
    screens = manifest["screened_rows"]
    sources = manifest["sources"]
    checks = {
        "status_is_remote_locked": (
            manifest["status"]
            == "sources_frozen_remote_generation_unauthorized"
        ),
        "execution_not_authorized": (
            manifest["execution_authorized"] is False
        ),
        "runtime_hash_matches": (
            manifest["runtime_sha256"] == runtime["sha256"]
        ),
        "protocol_hash_matches": (
            manifest["protocol_sha256"] == protocol["protocol_sha256"]
        ),
        "prior_hash_matches": (
            manifest["prior_identity_snapshot_sha256"] == prior["sha256"]
        ),
        "cohort_counts_match": (
            manifest["cohort_counts"]
            == {"pilot": EXPECTED_PILOT, "main": EXPECTED_MAIN}
            and len(sources) == EXPECTED_PILOT + EXPECTED_MAIN
        ),
        "start_shard_matches": (
            manifest["start_shard"] == EXPECTED_START_SHARD
        ),
        "no_replacement": manifest["replacement_allowed"] is False,
    }

    seen_text = set(prior["text_identities"])
    seen_url = set(prior["url_identities"])
    replayed_sources = []
    previous_position = None
    screen_mismatches = []
    exclusion_counts = {
        "empty_source": 0,
        "empty_prompt": 0,
        "normalized_text_collision": 0,
        "url_collision": 0,
    }
    for saved in screens:
        position = (saved["shard_index"], saved["shard_row"])
        if previous_position is not None and position <= previous_position:
            checks["strict_shard_row_order"] = False
        previous_position = position
        rebuilt = rebuild_screening(saved, seen_text, seen_url)
        if rebuilt != saved:
            screen_mismatches.append(saved["dataset_locator"])
            continue
        row_checks = rebuilt["checks"]
        if not row_checks["nonempty_source"]:
            exclusion_counts["empty_source"] += 1
        if not row_checks["nonempty_prompt"]:
            exclusion_counts["empty_prompt"] += 1
        if not row_checks["normalized_text_unique"]:
            exclusion_counts["normalized_text_collision"] += 1
        if not row_checks["url_unique"]:
            exclusion_counts["url_collision"] += 1
        if rebuilt["decision"] == "selected":
            replayed_sources.append(
                source_from_screen(rebuilt, len(replayed_sources))
            )
            seen_text.add(rebuilt["normalized_source_sha256"])
            if rebuilt["url"]:
                seen_url.add(rebuilt["url_sha256"])
    checks.setdefault("strict_shard_row_order", True)
    checks["all_screen_rows_recompute"] = not screen_mismatches
    checks["first_35_eligible_replay_exactly"] = (
        replayed_sources == sources
        and len(replayed_sources) == EXPECTED_PILOT + EXPECTED_MAIN
        and screens[-1]["decision"] == "selected"
    )
    checks["screening_summary_matches"] = (
        manifest["screening"]
        == {
            "rows_scanned": len(screens),
            "selected": len(replayed_sources),
            "excluded": len(screens) - len(replayed_sources),
            "exclusion_condition_counts": exclusion_counts,
            "scanned_through": {
                "shard_index": screens[-1]["shard_index"],
                "shard_row": screens[-1]["shard_row"],
            },
        }
    )
    if not all(checks.values()):
        raise ValueError(
            "Source audit failed"
            + (
                f"; screen_mismatches={screen_mismatches[:5]}"
                if screen_mismatches
                else ""
            )
        )
    return {
        "checks": checks,
        "sha256": sha256(path),
        "rows_scanned": len(screens),
        "sources": len(sources),
        "pilot": sum(row["cohort"] == "pilot" for row in sources),
        "main": sum(row["cohort"] == "main" for row in sources),
        "final_locator": sources[-1]["dataset_locator"],
    }


def verify_pre_execution(
    folder: Path,
    protocol: dict,
    prior: dict,
    runtime: dict,
    sources: dict,
) -> dict:
    path = folder / "pre_execution_manifest.json"
    manifest = load_json(path)
    expected_hashes = {
        "protocol_sha256": protocol["protocol_sha256"],
        "asset_audit_manifest_sha256": protocol[
            "asset_audit_manifest_sha256"
        ],
        "prior_identity_snapshot_sha256": prior["sha256"],
        "runtime_sha256": runtime["sha256"],
        "source_manifest_sha256": sources["sha256"],
        "allocation_journal_sha256": sha256(
            folder / "allocation_journal.json"
        ),
    }
    checks = {
        "status_is_remote_locked": (
            manifest["status"]
            == "local_preparation_complete_remote_execution_unauthorized"
        ),
        "execution_not_authorized": (
            manifest["execution_authorized"] is False
        ),
        "all_frozen_hashes_match": manifest["hashes"] == expected_hashes,
        "all_35_sources_frozen": (
            manifest["completed"]["all_35_sources_frozen"] is True
        ),
        "model_probe_pending": (
            manifest["pending"]["returned_model_identity_probe"] is True
            and manifest["returned_model"] is None
        ),
        "remote_counts_are_zero": all(
            manifest["resource_counts"][key] == 0
            for key in (
                "paid_api_calls",
                "model_probe_calls",
                "generation_calls",
                "parser_calls",
                "s2match_calls",
                "detector_calls",
                "attack_calls",
            )
        ),
    }
    forbidden_files = [
        folder / "model_probe.json",
        folder / "quality_packet.json",
        folder / "quality_labels.json",
        folder / "original_detection.json",
    ]
    forbidden_directories = [
        folder / "calls",
        folder / "generation_traces",
        folder / "baselines",
    ]
    checks["no_remote_or_downstream_artifacts"] = (
        not any(path.exists() for path in forbidden_files)
        and not any(path.exists() for path in forbidden_directories)
    )
    if not all(checks.values()):
        raise ValueError("Pre-execution audit failed")
    return {
        "checks": checks,
        "sha256": sha256(path),
        "status": manifest["status"],
        "resource_counts": manifest["resource_counts"],
        "pending": manifest["pending"],
    }


def build_audit(folder: Path = OUTPUT) -> dict:
    protocol = verify_protocol()
    prior = verify_prior_snapshot(folder)
    runtime = verify_runtime(folder, protocol, prior)
    sources = verify_sources(folder, protocol, prior, runtime)
    pre_execution = verify_pre_execution(
        folder,
        protocol,
        prior,
        runtime,
        sources,
    )
    sections = {
        "protocol": protocol["checks"],
        "prior_identity_snapshot": prior["checks"],
        "runtime": runtime["checks"],
        "source_allocation": sources["checks"],
        "pre_execution_lock": pre_execution["checks"],
    }
    return {
        "schema_version": 1,
        "experiment": "swan_positive_collection1",
        "status": "verified_local_preparation_remote_execution_unauthorized",
        "decision": "await_explicit_paid_budget_authorization",
        "checks": sections,
        "all_checks_pass": all(
            value
            for section in sections.values()
            for value in section.values()
        ),
        "prior_identity_counts": prior["counts"],
        "runtime_dependency_files": runtime["dependency_files"],
        "allocation": {
            "rows_scanned": sources["rows_scanned"],
            "sources": sources["sources"],
            "pilot": sources["pilot"],
            "main": sources["main"],
            "final_locator": sources["final_locator"],
        },
        "resource_counts": pre_execution["resource_counts"],
        "remaining_gates": [
            "explicit paid-budget authorization",
            "one returned-model identity probe",
            "freeze model_probe.json before generation",
        ],
        "claim_boundary": (
            "This audit verifies deterministic source allocation and local "
            "runtime provenance only. It reports no generated watermarked "
            "documents, detector result, attack result, or ASR."
        ),
    }


def render_report(audit: dict) -> str:
    allocation = audit["allocation"]
    resources = audit["resource_counts"]
    return f"""# SWAN positive collection pre-execution report

## Status

The independent offline audit passed all saved-evidence checks. Local
preparation is complete, but paid remote execution is not authorized.

## Frozen allocation

- Prior identity artifacts: {audit["prior_identity_counts"]["artifacts"]}
- Prior normalized text identities: {audit["prior_identity_counts"]["normalized_text_identities"]}
- Prior URL identities: {audit["prior_identity_counts"]["url_identities"]}
- C4 rows screened: {allocation["rows_scanned"]}
- Pilot sources: {allocation["pilot"]}
- Main sources: {allocation["main"]}
- Final selected locator: `{allocation["final_locator"]}`
- Frozen runtime dependency files: {audit["runtime_dependency_files"]}

The audit replayed every saved screening row in shard/row order and recovered
the exact first 35 eligible identities. The first five are pilot-only and the
next 30 are main-only. Replacement remains forbidden.

## Resource boundary

The only permitted remote operation was public C4 source retrieval
({resources["public_dataset_http_requests"]} HTTP request). Paid API calls,
model probes, generation, parser, S2Match, detector, and attack calls are all
zero.

No `model_probe.json`, generation trace, baseline, quality label, detection
result, or attack artifact exists. The next valid transition requires explicit
paid-budget authorization followed by one model-identity probe. This report
contains no watermarked-document, detector, attack, or ASR result.
"""


def run(folder: Path = OUTPUT) -> dict:
    audit = build_audit(folder)
    if not audit["all_checks_pass"]:
        raise ValueError("Independent audit contains a failed check")
    write_frozen_json(folder / "audit.json", audit)
    write_frozen_text(folder / "report.md", render_report(audit))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit = run(args.output)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "all_checks_pass": audit["all_checks_pass"],
                "allocation": audit["allocation"],
                "resource_counts": audit["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
