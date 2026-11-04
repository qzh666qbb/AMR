"""Build and detect two accepted-only SWAN baseline documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from swan_feasibility2_common import ROOT, digest, freeze, load, now, sha256
from swan_final1_runtime import official_runtime, score_triple
from swan_final1_transform import sentence_spans, unit_texts

OUT = ROOT / "experiments/swan_accepted_baseline_preflight1"
SLOTS = ROOT / "experiments/swan_count_neutral_final1/private/pilot/generation_slots"
TRACES = ROOT / "experiments/swan_count_neutral_final1/private/pilot/generation_attempts"
BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"
EXPECTED_BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
COUNT = 20
GROUP = 10
GREEN_THRESHOLD = .65
NULL_RATE = .25


def seed(*parts):
    value = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def source_slots():
    rows = []
    for path in sorted(SLOTS.glob("*.json")):
        row = load(path)
        if not row.get("accepted") or row.get("fallback") \
                or row.get("score", 0) < .75:
            continue
        attempt = row["attempts"] - 1
        trace_path = TRACES / (
            f"{row['allocation_id']}-slot-{row['slot']:02d}-attempt-{attempt:02d}.json"
        )
        trace = load(trace_path)
        if trace.get("accepted") is not True or trace.get("candidate") != row["text"] \
                or trace.get("score") != row["score"] \
                or sha256(trace_path) != row["attempt_trace_sha256"][-1]:
            raise ValueError("Accepted slot lacks matching final trace")
        rows.append({
            "allocation_id": row["allocation_id"],
            "slot": row["slot"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "target_template": row["chosen_template"],
            "target_score": row["score"],
            "slot_path": str(path.relative_to(ROOT)),
            "slot_sha256": sha256(path),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "trace_sha256": sha256(trace_path),
        })
    return rows


def select_inputs():
    selected, seen = [], set()
    for row in source_slots():
        if row["text_sha256"] in seen:
            continue
        seen.add(row["text_sha256"])
        selected.append(row)
        if len(selected) == COUNT:
            break
    if len(selected) != COUNT:
        raise ValueError(f"Need {COUNT} unique accepted sentences")
    documents = []
    for start in range(0, COUNT, GROUP):
        rows = selected[start:start + GROUP]
        text = " ".join(row["text"] for row in rows)
        spans = sentence_spans(text)
        if len(spans) != GROUP or unit_texts(text, spans) != [row["text"] for row in rows]:
            raise ValueError("Selected sentences do not preserve ten detector units")
        documents.append({
            "document_id": f"accepted-baseline-{start // GROUP:02d}",
            "text": text,
            "segmentation_spans": spans,
            "sentence_refs": [
                {
                    "allocation_id": row["allocation_id"],
                    "slot": row["slot"],
                    "text_sha256": row["text_sha256"],
                    "target_score": row["target_score"],
                }
                for row in rows
            ],
        })
    return {"selected": selected, "documents": documents}


def dependencies():
    paths = [
        Path(__file__),
        OUT / "protocol.md",
        BANK,
        ROOT / "scripts/swan_final1_runtime.py",
        ROOT / "scripts/swan_final1_transform.py",
        ROOT / "scripts/run_swan_positive_collection.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def prepare():
    if (OUT / "manifest.json").exists():
        return check()
    inputs = select_inputs()
    freeze(OUT / "inputs.json", inputs)
    manifest = freeze(OUT / "manifest.json", {
        "experiment": "swan_accepted_baseline_preflight1",
        "frozen_at": now(),
        "status": "frozen_before_detection",
        "selection": "accepted, nonfallback, score>=.75; lexical order; first unique hash",
        "selected_sentences": COUNT,
        "documents": COUNT // GROUP,
        "sentences_per_document": GROUP,
        "green_threshold": GREEN_THRESHOLD,
        "null_rate": NULL_RATE,
        "bank_sha256": EXPECTED_BANK,
        "inputs_sha256": sha256(OUT / "inputs.json"),
        "dependencies": dependencies(),
        "sealed_source_access": False,
        "new_generation_calls": 0,
    })
    return {"state": manifest["status"], "manifest_sha256": sha256(OUT / "manifest.json")}


def check():
    manifest = load(OUT / "manifest.json")
    if manifest["experiment"] != "swan_accepted_baseline_preflight1" \
            or manifest["sealed_source_access"] \
            or manifest["new_generation_calls"] != 0 \
            or manifest["inputs_sha256"] != sha256(OUT / "inputs.json") \
            or manifest["bank_sha256"] != sha256(BANK):
        raise ValueError("Preflight contract changed")
    for relative, expected in manifest["dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Dependency changed: {relative}")
    for row in load(OUT / "inputs.json")["selected"]:
        if sha256(ROOT / row["slot_path"]) != row["slot_sha256"] \
                or sha256(ROOT / row["trace_path"]) != row["trace_sha256"]:
            raise ValueError("Accepted source evidence changed")
    return {"state": "frozen_hashes_verified", "manifest_sha256": sha256(OUT / "manifest.json")}


class ParserWarning(RuntimeError):
    pass


class FailOnWarning(logging.Handler):
    def emit(self, record):
        raise ParserWarning(record.getMessage())


def score_sentence(module, bank, document_id, index, text):
    raw = module.parse_amr(text)
    candidate = module.normalize_amr_variables(
        module.generate_template_amr(module.normalize_amr_variables(raw))
    )
    scores = []
    for bank_index, graph in enumerate(bank):
        p, r, f1 = [
            float(value)
            for value in score_triple(
                candidate, graph, seed("detect", document_id, index, bank_index)
            )
        ]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (p, r, f1)):
            raise ValueError("Invalid S2Match score")
        scores.append({"bank_index": bank_index, "precision": p, "recall": r, "f1": f1})
    maximum = max(row["f1"] for row in scores)
    return {
        "text": text,
        "text_sha256": digest(text),
        "raw_amr": raw,
        "candidate_template_amr": candidate,
        "scores": scores,
        "max_score": maximum,
        "max_bank_indices": [row["bank_index"] for row in scores if row["f1"] == maximum],
        "green": maximum >= GREEN_THRESHOLD,
    }


def detect():
    check()
    if (OUT / "detection.json").exists():
        return load(OUT / "detection.json")
    inputs = load(OUT / "inputs.json")
    bank = [row["normalized_template_amr"] for row in load(BANK)]
    warning_handler = FailOnWarning(level=logging.WARNING)
    parser_logger = logging.getLogger("amrlib.models.parse_xfm.penman_serializer")
    parser_logger.addHandler(warning_handler)
    rows = []
    try:
        with official_runtime(OUT, error_folder=OUT / "errors") as module:
            for document in inputs["documents"]:
                units = []
                for index, (start, stop) in enumerate(document["segmentation_spans"]):
                    units.append(score_sentence(
                        module, bank, document["document_id"], index,
                        document["text"][start:stop],
                    ))
                green = sum(row["green"] for row in units)
                n = len(units)
                z = (green - NULL_RATE * n) / math.sqrt(
                    n * NULL_RATE * (1 - NULL_RATE)
                )
                rows.append({
                    "document_id": document["document_id"],
                    "N": n,
                    "green": green,
                    "z": z,
                    "detected_at_1pct_theoretical": z > 2.33,
                    "detected_at_5pct_theoretical": z > 1.645,
                    "units": units,
                })
    finally:
        parser_logger.removeHandler(warning_handler)
    result = {
        "completed_at": now(),
        "manifest_sha256": sha256(OUT / "manifest.json"),
        "documents": rows,
        "TPR_at_1pct_theoretical": sum(
            row["detected_at_1pct_theoretical"] for row in rows
        ) / len(rows),
        "TPR_at_5pct_theoretical": sum(
            row["detected_at_5pct_theoretical"] for row in rows
        ) / len(rows),
        "independent_empirical_calibration_pending": True,
    }
    freeze(OUT / "detection.json", result)
    return result


def status():
    return {
        "prepared": (OUT / "manifest.json").exists(),
        "detected": (OUT / "detection.json").exists(),
        "result": load(OUT / "detection.json") if (OUT / "detection.json").exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "detect", "status"))
    args = parser.parse_args()
    result = {
        "prepare": prepare,
        "check": check,
        "detect": detect,
        "status": status,
    }[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
