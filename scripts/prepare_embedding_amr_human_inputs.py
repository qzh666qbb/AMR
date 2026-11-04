"""Export frozen human-origin controls without scoring, filtering or old caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.embedding_amr import digest
from src.core.embedding_amr_runtime import atomic, file_sha
from run_embedding_amr_detection import input_rows


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def prepare(parent, output):
    if output.exists():
        raise ValueError("Use a new directory; do not overwrite frozen inputs")
    freeze = json.loads((parent / "FREEZE.json").read_text())
    exports, identities, seals, summaries = {}, {}, {}, {}
    for split in ("calibration", "test"):
        relative = f"private/sources/{split}.json"
        path = parent / relative
        if file_sha(path) != freeze["files"][relative]:
            raise ValueError("Frozen source file SHA mismatch")
        seals[relative] = file_sha(path)
        sources = json.loads(path.read_text())
        rows, ids, hashes, windows, urls = [], set(), set(), set(), set()
        for source in sources:
            if (not source["selected"] or not all(source["checks"].values())
                    or digest(canonical(source["text"])) != source["normalized_source_sha256"]):
                raise ValueError("Source identity or qualification mismatch; no replacement")
            text = " ".join(source["control_window"]["sentences"])
            if len(source["control_window"]["sentences"]) != 16:
                raise ValueError("Frozen sixteen-sentence window mismatch")
            sid, source_hash = source["allocation_id"], source["normalized_source_sha256"]
            window_hash, url = digest(canonical(text)), source["url_sha256"]
            if sid in ids or source_hash in hashes or window_hash in windows or (url and url in urls):
                raise ValueError("Repeated source, URL or window; no filtering permitted")
            ids.add(sid)
            hashes.add(source_hash)
            windows.add(window_hash)
            if url:
                urls.add(url)
            rows.append({
                "row_id": sid + "::clean", "source_id": sid,
                "source_sha256": digest(source["text"]),
                "condition": "clean", "kind": "human", "split": split, "text": text,
            })
        if len(rows) != 1000:
            raise ValueError("Frozen full denominator must be 1000 per split")
        identities[split] = {"ids": ids, "content": hashes | windows, "urls": urls}
        exports[split] = rows
        summaries[split] = {"sources": len(rows), "rows": len(rows),
                            "unique_windows": len(windows), "unavailable": 0}
    overlap = {name: len(identities["calibration"][name] & identities["test"][name])
               for name in ("ids", "content", "urls")}
    if any(overlap.values()):
        raise ValueError("Cross-split identity/window/URL overlap")
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    (output / ".gitignore").write_text("private/\n*.log\n")
    for split, rows in exports.items():
        path = private / f"human_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        input_rows(path)
    manifest = {
        "version": "embedding-amr-human-inputs/v1",
        "parent_freeze_sha256": file_sha(parent / "FREEZE.json"),
        "source_file_sha256": seals, "preparation_code_sha256": file_sha(__file__),
        "files": {str(p.relative_to(output)): file_sha(p) for p in sorted(private.glob("*.jsonl"))},
        "conditions": ["clean"], "counts": summaries, "cross_split_overlap": overlap,
        "original_window_rule": "frozen control_window sentences joined with single spaces",
        "source_sha_rule": "SHA256 of exact original source text; canonical identity checked separately",
        "qualification": "existing C4 human-origin controls; authorship not individually certified",
        "reuse_scope": "repurposed frozen controls; not newly collected unseen project-level sources",
        "scores_read": False, "model_inference_run": False, "old_cache_imported": False,
        "positive_inputs_prepared": False, "edited_conditions_prepared": False,
        "semantic_truth_labels_present": False,
    }
    atomic(output / "INPUT_MANIFEST.json", manifest)
    readiness = {
        "status": "HUMAN_CLEAN_INPUTS_PREPARED_FULL_METHOD_EVALUATION_NOT_READY",
        "input_manifest_sha256": file_sha(output / "INPUT_MANIFEST.json"),
        "counts": summaries, "cross_split_overlap": overlap,
        "checks": {
            "frozen_source_sha_verified": True,
            "source_and_window_canonical_hashes_verified": True,
            "full_original_denominators_retained": True,
            "same_clean_rendering_across_splits": True,
        },
        "missing_local_positive_evidence": {
            name: not (parent / name).exists()
            for name in ("MAIN_GENERATION_SEAL.json", "INPUT_SEAL.json", "TERMINAL.json")
        },
        "blockers": [
            "No sealed eligible positive input cohort bound to the new detector.",
            "Normal-length writer uses Bank50; older alignment uses Bank100. Bind the actual writing Bank.",
            "No supplied target private Bank with writing provenance for this new run.",
            "No independent human fact/correspondence labels for method-mechanism validation.",
            "Only clean controls exported; edited positive/null conditions require their own frozen preparation.",
            "No fresh full-flow calibration, independent FPR/TPR, or throughput/resource gate yet.",
        ],
        "remote_state": "not_queried; missing local artifacts do not establish remote failure",
        "fusion_benefit_established": False,
    }
    atomic(output / "READINESS.json", readiness)
    atomic(output / "TERMINAL.json", {
        "status": "input_preparation_complete_not_detection",
        "manifest_sha256": file_sha(output / "INPUT_MANIFEST.json"),
        "readiness_sha256": file_sha(output / "READINESS.json"),
    })
    return readiness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "experiments/swan_normal_completion1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output), ensure_ascii=False))
