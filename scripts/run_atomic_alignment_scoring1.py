"""CPU scoring successor over an immutable, independently sealed graph source."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_atomic_alignment_completion2 as base
import run_multiscale_transfer as rt
from semantic_fact_raw_amr import audit_raw


class ReadOnlyGraphs:
    """No parse, write, cache repair, source copying or fallback capability."""

    def __init__(self, reference, rows):
        self.reference = reference
        self.root = Path(reference["root"])
        if not self.root.is_absolute():
            raise ValueError("Absolute source root required")
        self.freeze_sha256 = reference["files"]["FREEZE.json"]
        self.seal_sha256 = reference["files"]["RAW_GRAPH_SEAL.json"]
        self.texts = {rt.digest(c["text"]): c["text"] for r in rows for c in r["candidates"]}
        self.rows = rows

    def checked(self, relative):
        path = self.root / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve())):
            raise ValueError("Unsafe source path")
        payload = path.read_bytes()
        identity = hashlib.sha256(payload).hexdigest()
        if identity != self.reference["files"][relative]:
            raise ValueError("Read-only source SHA changed: " + relative)
        return json.loads(payload), identity

    def load(self, text):
        identity = rt.digest(text)
        if identity not in self.texts or self.texts[identity] != text:
            raise ValueError("Candidate absent from frozen source")
        record, sha256 = self.checked("private/raw_graphs/" + identity + ".json")
        audit_raw(record, base.raw_input(text), self.freeze_sha256)
        if record["amr"]:
            accepted = [a for a in record["attempts"] if a["accepted"]]
            if (record.get("parser_output_version") != "accepted-graph/v2-exact-metadata"
                    or len(accepted) != 1 or accepted[0]["graph_sha256"] != record["amr_sha256"]):
                raise ValueError("Accepted graph identity differs")
        return record, sha256

    def verify(self):
        freeze, _ = self.checked("FREEZE.json")
        seal, _ = self.checked("RAW_GRAPH_SEAL.json")
        if seal["freeze_sha256"] != self.freeze_sha256:
            raise ValueError("Source seal belongs to a different freeze")
        for name, expected in {**freeze["files"], **seal["files"]}.items():
            if self.reference["files"].get(name) != expected:
                raise ValueError("Source manifest differs from original seal")
        expected = {"private/raw_graphs/" + h + ".json" for h in self.texts}
        actual = {str(p.relative_to(self.root)) for p in (self.root / "private/raw_graphs").glob("*")}
        if set(seal["files"]) != expected or actual != expected:
            raise ValueError("Incomplete or extra graph coverage")
        source_rows, _ = self.checked("private/INPUTS.json")
        if source_rows != self.rows:
            raise ValueError("Source candidate cohort changed")
        for name in self.reference["files"]:
            # Model weights and frozen code are not JSON.
            path = self.root / name
            if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Unsafe source asset")
            if rt.sha(path) != self.reference["files"][name]:
                raise ValueError("Source asset changed: " + name)
        eligible, reasons = 0, Counter()
        for text in self.texts.values():
            record, _ = self.load(text)
            audit = audit_raw(record, base.raw_input(text), self.freeze_sha256)
            eligible += audit["structurally_eligible"]
            reasons.update(audit["exclusion_reasons"])
        observed = {"rows": len(self.rows), "graphs": len(self.texts),
                    "candidate_occurrences": sum(len(r["candidates"]) for r in self.rows),
                    "eligible": eligible, "excluded": len(self.texts) - eligible}
        if observed != self.reference["counts"]:
            raise ValueError("Source counts differ")
        return {"status": "PASS", **observed, "exclusion_reason_counts": dict(reasons),
                "source_freeze_sha256": self.freeze_sha256,
                "raw_graph_seal_sha256": self.seal_sha256,
                "source_halt_sha256": self.reference["files"]["HALT.json"],
                "verified_source_files": len(self.reference["files"]),
                "raw_graphs_copied": 0, "new_parser_calls": 0, "new_provider_calls": 0}


def environment(out):
    spec = rt.read(out / "RUNTIME_ENVIRONMENT.json")
    for name, info in spec["packages"].items():
        dist = metadata.distribution(name)
        if dist.version != info["version"]:
            raise RuntimeError("Runtime version changed: " + name)
        if str(Path(dist.locate_file("")).resolve()) != info["location"]:
            raise RuntimeError("Runtime package location changed: " + name)
    for path, expected in spec["files"].items():
        if rt.sha(path) != expected:
            raise RuntimeError("Runtime dependency SHA changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Scoring successor must be CPU-only")
    return {"packages": len(spec["packages"]), "files": len(spec["files"]),
            "environment_sha256": rt.sha(out / "RUNTIME_ENVIRONMENT.json")}


def source(out):
    base.verify(out)
    rows = rt.read(out / "private/INPUTS.json")
    reference = rt.read(out / "SOURCE_REFERENCE.json")
    if Path(reference["root"]).resolve() == out.resolve():
        raise RuntimeError("Source must be separate from scoring output")
    if (out / "private/raw_graphs").exists():
        raise RuntimeError("Successor must not own or copy raw graphs")
    return ReadOnlyGraphs(reference, rows)


def readiness(out):
    import numpy as np
    import spacy
    import torch
    from src.core.bank_scorer import SwanAMRBankScorer
    from src.core.multiview_views import SurfaceAnalyzer, extract_events

    store = source(out)
    before = store.verify()
    runtime = environment(out)
    # Choose by parser eligibility only, before any Bank score or metadata read.
    row = next(r for r in store.rows if any(
        audit_raw(store.load(c["text"])[0], base.raw_input(c["text"]),
                  store.freeze_sha256)["structurally_eligible"]
        and extract_events(store.load(c["text"])[0]["amr"], i)
        for i, c in enumerate(r["candidates"])))
    results, vectors = [], []
    started = time.monotonic()
    for _ in range(2):
        encoder = base.Encoder(out / "private/encoder")
        if encoder.model.max_seq_length != 256:
            raise RuntimeError("Frozen native encoder context changed")
        text = next(c["text"] for c in row["candidates"] if encoder.fits(c["text"]))
        vectors.append(encoder.encode([text]))
        surface = object.__new__(SurfaceAnalyzer)
        surface.nlp = spacy.load(out / "private/surface_model")

        class SealedExtractor:
            fingerprint = "presealed-raw-graphs-only"

            def extract(self, *args, **kwargs):
                raise RuntimeError("New parser calls forbidden")

        scorer = SwanAMRBankScorer(SealedExtractor(), str(out / "private/BANK.json"))
        result = base.observe(row, out, encoder, surface, scorer, store)
        if not result["events"] or not any(o["bank_scores"] for o in result["observations"]):
            raise RuntimeError("Readiness did not exercise real event and Bank scoring")
        results.append(result)
    error = float(np.max(np.abs(vectors[0] - vectors[1])))
    if error > 1e-6 or results[0] != results[1]:
        raise RuntimeError("Independent model/row double replay failed")
    if store.verify() != before:
        raise RuntimeError("Source changed during readiness")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU scoring initialized CUDA")
    ready = out / "readiness"
    base.immutable(ready / "RESULT.json", {
        "status": "PASS", "source_audit": before, "environment": runtime,
        "real_model_loads": 2, "actual_observe_replays": 2, "row_exact_replay": True,
        "encoder_max_abs_error": error, "encoder_context": 256, "dtype": "float32",
        "device": "cpu", "cuda_initialized": False, "new_parser_calls": 0,
        "new_provider_calls": 0, "readiness_elapsed_seconds": time.monotonic() - started,
        "row_id_sha256": rt.digest(row["row_id"]),
        "observation_sha256": rt.digest(json.dumps(results[0], sort_keys=True)),
    })
    base.immutable(ready / "AUDIT.json", {"status": "PASS", "source_unchanged": True,
                                         "runtime_sha_verified": True,
                                         "freeze_sha256": rt.sha(out / "FREEZE.json")})
    base.immutable(ready / "TERMINAL.json", {
        "status": "PASS", "result_sha256": rt.sha(ready / "RESULT.json"),
        "audit_sha256": rt.sha(ready / "AUDIT.json")})


def run(out):
    store = source(out)
    before = store.verify()
    environment(out)
    ready = out / "readiness"
    terminal = rt.read(ready / "TERMINAL.json")
    if (terminal["status"] != "PASS" or terminal["result_sha256"] != rt.sha(ready / "RESULT.json")
            or terminal["audit_sha256"] != rt.sha(ready / "AUDIT.json")
            or rt.read(ready / "RESULT.json")["source_audit"] != before
            or rt.read(ready / "AUDIT.json")["freeze_sha256"] != rt.sha(out / "FREEZE.json")):
        raise RuntimeError("Readiness receipt invalid")
    base.immutable(out / "SOURCE_AUDIT_BEFORE.json", before)
    base.score(out, graph_store=store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("readiness", "run", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "verify":
        print(json.dumps(source(out).verify()))
        return
    if (out / "HALT.json").exists() or (out / "TERMINAL.json").exists():
        raise RuntimeError("Existing HALT/terminal requires separate review")
    with (out / "private/scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        try:
            globals()[args.action](out)
        except Exception as exc:
            rt.atomic(out / "HALT.json", {"phase": args.action, "type": type(exc).__name__,
                                         "resume_allowed": False, "scientific_result": None,
                                         "at": rt.now()})
            raise
        finally:
            rt.atomic(out / "private/sessions" / (str(time.time_ns()) + ".json"),
                      {"phase": args.action, "worker_elapsed_seconds": time.monotonic() - started,
                       "device": "cpu", "new_parser_calls": 0, "new_provider_calls": 0})
    print(json.dumps({"action": args.action, "status": "COMPLETE"}))


if __name__ == "__main__":
    main()
