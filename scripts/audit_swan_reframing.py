"""Read-only historical audit; output only a separate reframing audit artifact."""
import argparse
import hashlib
import json
from pathlib import Path

from freeze_swan_reframing import check
from lint_research_claims import evaluate_registry, lint_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_reframing1"
ALLOWED_EDITS = {
    "任务清单.md", "TERMINOLOGY.md", "README.md", "scripts/README.md",
    "experiments/README.md", "CLAIM_EVIDENCE_MATRIX.md", "CURRENT_RESULTS.md",
    "RESEARCH_STATE.md", "叙事.md",
    "experiments/swan_amr_composition_decomposition1/report.md",
    "experiments/swan_amr_precision_recall1/report.md",
    "experiments/swan_length_gate_tradeoff1/report.md",
    "archive/20260918_reframing_cleanup/root_docs/SWAN_RESEARCH_NARRATIVE.md",
    "archive/20260918_reframing_cleanup/root_docs/任务规划.md",
    "archive/20260911_pre_unification/paper_strict30_draft/main.tex",
    "archive/20260911_pre_unification/paper_strict30_draft/draft.md",
    "archive/20260911_pre_unification/paper_strict30_draft/outline.md",
    "archive/20260911_pre_unification/paper_strict30_draft/related_work.md",
}


def audit():
    preserved = check()
    changes = preserved["changes_since_snapshot"]
    unexpected = sorted(set(changes) - ALLOWED_EDITS)
    hashes = json.loads((OUT / "pre_reset_hashes.json").read_text())
    original = {row["path"]: row["sha256"] for row in hashes["files"]}
    scanned = [
        ROOT / name for name in ("叙事.md", "CURRENT_RESULTS.md",
                                "RESEARCH_STATE.md", "CLAIM_EVIDENCE_MATRIX.md")
    ]
    scanned += [ROOT / name for name in sorted(ALLOWED_EDITS)
                if name.startswith("archive/20260918")]
    archive = ROOT / "archive/20260911_pre_unification/paper_strict30_draft"
    scanned += sorted(p for p in archive.rglob("*") if p.suffix in {".md", ".tex"})
    scan = []
    for path in scanned:
        text = path.read_text()
        scan.append({
            "path": str(path.relative_to(ROOT)),
            "bytes_read": len(text.encode()),
            "findings": lint_text(text, "history"),
            "use": "historical_only" if archive in path.parents else "corrected_entry",
        })
    registry = json.loads((OUT / "claims_v2.json").read_text())
    claims = evaluate_registry(registry)
    bad_endpoints = [row["id"] for claim, row in zip(registry["claims"], claims)
                    if claim["ceiling"] not in {"unconfirmed", "withdrawn"}
                    and row["reasons"]]
    return {
        "ok": not unexpected and not bad_endpoints,
        "scope": "P0-P2 and statistical infrastructure; not a final experiment audit",
        "historical_preservation": preserved,
        "unexpected_historical_changes": unexpected,
        "historical_raw_results_manifests_and_scripts_unchanged": not unexpected,
        "approved_document_errata": [
            {"path": name, "before_sha256": original[name],
             "after_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
             "original_byte_copy_preserved": name in hashes["copied_originals"]}
            for name in changes if name in ALLOWED_EDITS
        ],
        "text_scan": scan,
        "endpoint_gates": claims,
        "bad_endpoints": bad_endpoints,
        "new_model_parser_s2match_detector_observations": 0,
        "limitations": [
            "Historical quote lint findings are retained, not hidden.",
            "Archive child sections remain original; draft/main and ERRATUM mark the whole draft superseded.",
            "The current JSON assertions do not substitute for the future independent empirical audit.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "ok": result["ok"],
        "preserved_originals": result["historical_preservation"]["preserved_originals_verified"],
        "changed_documents": len(result["approved_document_errata"]),
        "unexpected_changes": result["unexpected_historical_changes"],
        "scanned_documents": len(result["text_scan"]),
    }, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)
