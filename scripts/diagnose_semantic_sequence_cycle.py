"""Post-gate, read-only diagnosis and annotation packets; no new inference.

Corrects a normalization wrapper error in the initial auxiliary coverage
diagnostic: the official function expects an AMR string without ::snt metadata.
The frozen raw-graph G1e gate never used this normalization and is unaffected.
"""
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys

from run_semantic_sequence_cycle import OUT, PRIVATE, ROOT, atomic, digest, sha
from audit_semantic_sequence_alignment import normalization_namespace, structure, covers


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.events = []

    def emit(self, record):
        self.events.append({"level": record.levelname, "message": record.getMessage()})


def main():
    out, private = OUT, PRIVATE
    initial = json.loads((out / "WRITE_READ_RESULTS.json").read_text())
    targets = json.loads((private / "TARGETS.json").read_text())
    normalizer, _ = normalization_namespace(ROOT, private / "normalizer_runtime")
    logger = logging.getLogger("alignment-normalizer")
    handler = Capture()
    logger.addHandler(handler)
    previous = logger.propagate
    logger.propagate = False
    corrected = []
    private_details = []
    try:
        for target in targets:
            item = json.loads((private / "PARSES" / (target["input_id"] + ".json")).read_text())
            raw = "\n".join(line for line in item["amr"].splitlines()
                            if not line.lstrip().startswith("#"))
            handler.events.clear()
            normalized = normalizer(raw)
            if handler.events:
                atomic(private / "NORMALIZATION_DIAGNOSTICS.json", {
                    "input_id": target["input_id"], "events": handler.events})
                raise RuntimeError("Bound normalization diagnostic; do not use graph")
            coverage = covers(structure(normalized), structure(target["target"]), rooted=False)
            required, actual = next((r["required"], r["actual"]) for r in initial["rows"]
                                    if r["input_id"] == target["input_id"])
            corrected.append({"input_id": target["input_id"], "template_id": target["template_id"],
                              "strict_valid": item["strict_valid"], "whole_target_covered": coverage,
                              "raw_sha256": digest(raw), "normalized_sha256": digest(normalized)})
            private_details.append({"input_id": target["input_id"], "text": item["text"],
                                    "target": target["target"], "normalized": normalized,
                                    "required": required, "actual": actual})
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous
    atomic(private / "CORRECTED_NORMALIZATION_DETAILS.json", private_details)
    atomic(out / "NORMALIZATION_CORRECTION.json", {
        "initial_results_sha256": sha(out / "WRITE_READ_RESULTS.json"),
        "issue": "Initial auxiliary coverage passed metadata-prefixed AMR into the official normalizer; its startswith('(') guard returned unnormalized raw graph.",
        "correction": "Remove comment metadata only, then run the unchanged official normalizer; all diagnostics bound per input.",
        "inference_calls": 0, "gate_affected": False,
        "corrected_whole_target_covered": sum(r["whole_target_covered"] for r in corrected),
        "corrected_strict_whole_target_covered": sum(r["whole_target_covered"] and r["strict_valid"] for r in corrected),
        "denominator": 24, "rows": corrected, "source_sha256": sha(Path(__file__))})
    # Blind annotation stages: all text first, intended labels separately.
    stage1, stage2 = [], []
    for doc_id in (0, 1, 2):
        document = json.loads((private / "documents" / f"d{doc_id:02d}.json").read_text())
        source = document["arms"]["positive"]
        for slot, text in enumerate(source["slots"]):
            identifier = digest(f"cycle1-annotation:{doc_id}:{slot}")[:16]
            stage1.append({"annotation_id": identifier, "source_text": source["text"],
                           "carrier_text": text, "paraphrase": source["paraphrase"],
                           "source_proposition": None, "source_event_anchor_span": None,
                           "paraphrase_event_spans": None, "proposition_preserved": None,
                           "reviewer_id": None})
            target = next(r for r in targets if r["doc_id"] == doc_id and r["slot"] == slot)
            stage2.append({"annotation_id": identifier, "target_amr": target["target"],
                           "realized": None, "evidence": None, "reviewer_id": None})
    for reviewer in ("reviewer_A", "reviewer_B"):
        atomic(private / "annotation" / reviewer / "stage1_text_only.json", stage1)
        atomic(private / "annotation" / reviewer / "stage2_target_realization.json", stage2)
    atomic(out / "ANNOTATION_STATUS.json", {
        "status": "PREPARED_NOT_ANNOTATED", "carrier_mappings": 24,
        "reviewers_required": 2, "labels_completed": 0,
        "truth_source": "independent humans; AI draft must be labelled separately",
        "private_packet_sha256": {
            str(p.relative_to(private)): sha(p) for p in (private / "annotation").rglob("*.json")}})
    failed_ids = {r["input_id"] for r in initial["rows"] if not r["g1e_success"]}
    raw_rows = []
    for item in private_details:
        if item["input_id"] not in failed_ids:
            continue
        r = next(r for r in initial["rows"] if r["input_id"] == item["input_id"])
        raw_rows.append({"input_id": item["input_id"], "template_id": r["template_id"],
                         "symbol": r["symbol"], "failed_checks": [k for k, v in r["checks"].items() if not v],
                         "required": r["required"], "actual": r["actual"]})
    atomic(out / "FAILURE_DIAGNOSIS.json", {
        "scope": "Observed graph constraints, not independent generation/parser causal attribution",
        "failed_carriers": len(raw_rows), "rows": raw_rows,
        "full_graph_coverage_source": "NORMALIZATION_CORRECTION.json",
        "conclusion": "Necessary marker fidelity improved enough to measure but did not pass the new fixed pilot gate; keep current alphabet and batch closed."})
    print(json.dumps({"corrected_whole_target_covered": sum(r["whole_target_covered"] for r in corrected),
                      "gate_unchanged": True, "failed_carriers": len(raw_rows), "annotation_packets": 4}))


if __name__ == "__main__":
    os.umask(0o077)
    main()
