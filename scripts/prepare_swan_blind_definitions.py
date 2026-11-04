"""Audit frozen text geometry only; never import AMR/scoring experiment modules."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.swan_blind_protocol import (  # noqa: E402
    METHODS, VERSION, ProtocolConfig, digest, file_sha256, plan_document,
    segmenter_metadata,
)

EXP = ROOT / "experiments/swan_multiview_development1"
OFFICIAL = ROOT / "third_party/SWAN_official_2ca0c207_clean"
ATTACKS = ("baseline", "semicolon", "newline", "split", "merge", "comma", "space")


def load_attack_functions():
    """Extract unchanged text transforms, bypassing model/vector module imports."""
    source = ROOT / "scripts/run_dataset4_attack_recovery.py"
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and (n.name.startswith("attack_") or n.name.startswith("_span_map"))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
         namespace)
    return {name: namespace[f"attack_{name}"] for name in ATTACKS}


def run_audit() -> dict:
    datasets = {
        "positive": EXP / "dataset5/documents.json",
        "paired_negative": EXP / "dataset5_paired_neg/documents.json",
    }
    sources = list(datasets.values()) + [
        EXP / "PROTOCOL.md", EXP / "PAPER_DRAFT.md",
        EXP / "dataset5_eval/PROTOCOL_RESULTS.json",
        ROOT / "src/core/swan_blind_protocol.py",
        ROOT / "src/core/multiscale_blind.py",
        ROOT / "scripts/prepare_swan_blind_definitions.py",
        ROOT / "tests/test_swan_blind_protocol.py",
        ROOT / "scripts/run_dataset4_attack_recovery.py",
        OFFICIAL / "detection/parse_machine_text.py",
        OFFICIAL / "detection/parse_human_text.py",
        OFFICIAL / "detection/detect_from_parsed_amrs.py",
        OFFICIAL / "detection/detection_utils.py",
        OFFICIAL / "utils/amr_utils.py",
        OFFICIAL / "amr_bank/banks/amr_bank_100.json",
    ]
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}
    docs = {arm: json.loads(path.read_text())["documents"]
            for arm, path in datasets.items()}
    ids = {arm: [d["document_id"] for d in rows] for arm, rows in docs.items()}
    if any(len(rows) != 30 for rows in docs.values()) \
            or any(len(set(values)) != 30 for values in ids.values()) \
            or set(ids["positive"]) != set(ids["paired_negative"]):
        raise ValueError("Frozen cohort size or pairing mismatch")
    neg_by_id = {d["document_id"]: d for d in docs["paired_negative"]}
    if any(len(d["slots"]) != len(neg_by_id[d["document_id"]]["slots"])
           for d in docs["positive"]):
        raise ValueError("Frozen paired slot counts mismatch")
    bank_hash = hashes[str((OFFICIAL / "amr_bank/banks/amr_bank_100.json").relative_to(ROOT))]
    configs = {m: ProtocolConfig(m, bank_sha256=bank_hash) for m in METHODS}
    attacks = load_attack_functions()
    groups = []
    configuration_hashes = {m: set() for m in METHODS}
    text_fingerprints = []
    for arm, documents in docs.items():
        for attack in ATTACKS:
            per_method = {m: [] for m in METHODS}
            for document in sorted(documents, key=lambda d: d["document_id"]):
                # Attack construction may know slots; all oracle annotations stop here.
                text, _ = attacks[attack]([{"text": s["text"]} for s in document["slots"]])
                text_fingerprints.append(
                    (arm, document["document_id"], attack, digest(text)))
                for method in METHODS:
                    plan = plan_document(text, configs[method])
                    configuration_hashes[method].add(plan.configuration_sha256)
                    batches = plan.batches()
                    scheduled = [c.candidate_id for batch in batches for c in batch]
                    if plan.status == "geometry_ready" \
                            and scheduled != list(range(len(plan.candidates))):
                        raise ValueError("Batch scheduler dropped or reordered candidates")
                    per_method[method].append({
                        "status": plan.status, "reasons": plan.reasons,
                        "candidates": len(plan.candidates), "batches": len(batches),
                        "missing_characters": plan.uncovered_nonwhitespace,
                        "max_candidate_whitespace_tokens": max(
                            (len(c.text.split()) for c in plan.candidates), default=0),
                    })
            for method, plans in per_method.items():
                counts = [p["candidates"] for p in plans]
                groups.append({
                    "arm": arm, "attack": attack, "method": method,
                    "documents": len(plans),
                    "status_counts": dict(Counter(p["status"] for p in plans)),
                    "abstention_reasons": dict(Counter(
                        reason for p in plans for reason in p["reasons"])),
                    "candidates_total": sum(counts), "candidates_min": min(counts),
                    "candidates_max": max(counts),
                    "documents_exceeding_legacy_128_candidates": sum(c > 128 for c in counts),
                    "batches_total": sum(p["batches"] for p in plans),
                    "uncovered_nonwhitespace_characters": sum(
                        p["missing_characters"] for p in plans),
                    "max_candidate_whitespace_tokens": max(
                        p["max_candidate_whitespace_tokens"] for p in plans),
                })
    if any(len(values) != 1 for values in configuration_hashes.values()):
        raise ValueError("Configuration changed during audit")
    if any(file_sha256(p) != hashes[str(p.relative_to(ROOT))] for p in sources):
        raise ValueError("Frozen inputs or source changed during audit")
    all_ready = all(g["status_counts"] == {"geometry_ready": g["documents"]}
                    for g in groups)
    return {
        "schema": "swan-blind-definition-audit/v1", "protocol": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "geometry_go_execution_not_frozen" if all_ready else "geometry_stop",
        "scope": "definitions_and_text_geometry_only",
        "research_scoring_executed": False, "calibration_executed": False,
        "cohort": {"source_documents": 60, "document_attack_variants": 420,
                   "method_document_plans": sum(g["documents"] for g in groups),
                   "source_slots_by_arm": {
                       arm: sum(len(d["slots"]) for d in rows) for arm, rows in docs.items()},
                   "attacks": list(ATTACKS), "methods": list(METHODS)},
        "configuration_sha256": {m: next(iter(v)) for m, v in configuration_hashes.items()},
        "segmenter": segmenter_metadata(),
        "coverage": {"rule": "all_nonwhitespace_characters_in_candidate_union",
                     "candidate_selection_cap": None, "batch_size": 32,
                     "complete_geometry": all_ready,
                     "uncovered_nonwhitespace_characters": sum(
                         g["uncovered_nonwhitespace_characters"] for g in groups)},
        "future_execution_gates": {
            "native_parser_context_validated": False,
            "parser_and_tokenizer_artifacts_frozen": False,
            "normalizer_and_matcher_runtime_frozen": False,
            "all_candidates_parsed_and_bank_scored": False,
            "method_specific_calibration_frozen": False,
            "independent_human_test_completed": False,
        },
        "historical_results": {
            "scores_modified": False,
            "P0_NA_metadata": "obsolete_under_v1.2_definition_kept_for_history",
            "oracle_caches_valid_for_blind_detection": False,
        },
        "attacked_text_set_sha256": digest(text_fingerprints),
        "source_sha256": hashes, "groups": groups,
        "blind_detection_metrics": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXP / "BLIND_DEFINITION_AUDIT.json")
    args = parser.parse_args()
    audit = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": audit["status"], **audit["cohort"], "coverage": audit["coverage"],
        "output": str(args.output), "sha256": file_sha256(args.output),
    }, ensure_ascii=False))
    if audit["status"] == "geometry_stop":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
