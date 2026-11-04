"""Score, calibrate and evaluate embedding-assisted AMR detection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bank_scorer import SwanAMRBankScorer
from src.core.embedding_amr import EmbeddingAMRDetector, FusionConfig, digest
from src.core.embedding_amr_evaluation import evaluate, fit, validate_rows
from src.core.embedding_amr_runtime import (
    FreshParser, LocalEncoder, SealedObservationProvider, atomic, file_sha, sealed_scores,
)
from src.core.multiview_candidates import WindowConfig

FIELDS = {"row_id", "source_id", "source_sha256", "condition", "kind", "split", "text"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        if set(row) != FIELDS or any(not isinstance(v, str) or (not v and k != "text")
                                   for k, v in row.items()):
            raise ValueError("Input fields must be exactly " + ", ".join(sorted(FIELDS)))
        if row["text"] and not row["text"].strip():
            raise ValueError("Unavailable input uses empty text, not whitespace")
        if row["split"] not in ("development", "calibration", "test"):
            raise ValueError("Unknown input split")
        if row["kind"] not in ("human", "watermarked", "llm", "synthetic"):
            raise ValueError("Unknown input kind")
        if len(row["source_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["source_sha256"]):
            raise ValueError("Original source SHA256 required")
    # Same identity checks as evaluation, before any inference. Detector never
    # receives kind, split, source identity, or condition.
    validate_rows([{**r, "text_sha256": digest(r["text"]), "detector_fingerprint": "input",
                    "arms": {a: {"statistic": None} for a in (
                        "amr_only", "embedding_amr", "embedding_without_relation")}} for r in rows])
    return rows


def score(args):
    started = time.monotonic()
    rows = input_rows(args.input)
    if args.output.exists():
        raise ValueError("Use a new output directory; completed/frozen runs are immutable")
    if file_sha(args.bank) != args.bank_sha256:
        raise ValueError("Target Bank SHA256 mismatch")
    config = FusionConfig(candidate_budget=args.candidate_budget, top_k=args.top_k,
                          min_similarity=args.min_similarity)
    if args.replay and any(r["split"] != "development" for r in rows):
        raise ValueError("Sealed replay is development-only; calibration/test require fresh parsing")
    if any(r["kind"] == "watermarked" for r in rows) and args.writer_bank_sha256 != args.bank_sha256:
        raise ValueError("Watermarked rows require matching writer Bank SHA256")
    window_config = WindowConfig()
    args.output.mkdir(parents=True)
    private = args.output / "private"
    private.mkdir(mode=0o700)
    observations_dir = private / "observations"
    observations_dir.mkdir(mode=0o700)
    (args.output / ".gitignore").write_text("private/\n*.log\n")
    try:
        encoder = LocalEncoder(args.encoder, args.device)
        if args.replay:
            provider = SealedObservationProvider(args.replay)
        elif args.batched:
            from src.core.embedding_amr_batch import BatchFreshParser
            provider = BatchFreshParser(args.parser_model, args.tokenizer, observations_dir,
                                        args.device, batch_size=args.parser_batch_size)
        else:
            provider = FreshParser(args.parser_model, args.tokenizer, observations_dir, args.device)
        if args.scorer == "official":
            from src.core.embedding_amr_official import OfficialBankScorer
            scorer = OfficialBankScorer(args.bank, args.bank_sha256, args.vectors)
            scoring_mode = "official_normalization_s2match_new_document_statistic"
        else:
            scorer = SwanAMRBankScorer(None, str(args.bank))
            scoring_mode = "structural_feature_bank_development_not_official_s2match"
        detector = EmbeddingAMRDetector(encoder, scorer, provider.fingerprint, config, window_config)
        code_paths = sorted((ROOT / "src/core").glob("*.py")) + [Path(__file__).resolve()]
        code = {str(p.relative_to(ROOT)): file_sha(p) for p in code_paths}
        dependencies = {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "networkx", "penman", "torch", "transformers",
            "sentence-transformers", "amrlib")}
        allocator = {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        }
        detector.fingerprint = digest({"algorithm": detector.fingerprint, "code": code,
                                       "dependencies": dependencies, "cuda_allocator": allocator})
        manifest = {
            "version": "embedding-amr-run/v1", "input_sha256": file_sha(args.input),
            "row_count": len(rows), "config": asdict(config), "windows": asdict(window_config),
            "bank_sha256": scorer.bank_sha256, "bank_scorer_fingerprint": scorer.fingerprint,
            "scoring_mode": scoring_mode,
            "scorer_assets": getattr(scorer, "metadata", None),
            "writer_bank_sha256": args.writer_bank_sha256,
            "parser_fingerprint": provider.fingerprint, "encoder_fingerprint": encoder.fingerprint,
            "encoder_assets": encoder.assets, "code_sha256": code, "dependencies": dependencies,
            "detector_fingerprint": detector.fingerprint, "semantic_truth_confirmed": False,
            "cuda_allocator_policy": (
                "disabled_cache" if allocator["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
                else "release_unused_cache_before_each_document"),
            "cuda_allocator_environment": allocator,
            "replay_parent_manifest_sha256": file_sha(args.replay / "MANIFEST.json") if args.replay else None,
        }
        atomic(args.output / "MANIFEST.json", manifest)
        resume_rows = getattr(args, "resume_rows", 0) or 0
        resume_manifest = None
        if resume_rows:
            if not args.resume_run:
                raise ValueError("Resume requires --resume-run")
            source_heartbeat = json.loads((args.resume_run / "HEARTBEAT.json").read_text())
            if source_heartbeat.get("completed") != resume_rows or resume_rows > len(rows):
                raise ValueError("Resume row count does not match the source heartbeat")
            for path in (args.resume_run / "private/observations").glob("*.json"):
                target = observations_dir / path.name
                if not target.exists():
                    os.link(path, target)
            resume_manifest = {
                "source_run": str(args.resume_run), "rows": resume_rows,
                "rechecked_rows": list(range(max(0, resume_rows - args.resume_recheck), resume_rows)),
                "source_heartbeat_sha256": file_sha(args.resume_run / "HEARTBEAT.json"),
                "source_manifest_sha256": file_sha(args.resume_run / "MANIFEST.json"),
                "carried_observations": len(list(observations_dir.glob("*.json"))),
                "row_files": {},
            }
            manifest["resume_parent"] = {
                k: resume_manifest[k] for k in (
                    "source_run", "rows", "rechecked_rows", "source_heartbeat_sha256",
                    "source_manifest_sha256", "carried_observations")}
            atomic(args.output / "MANIFEST.json", manifest)
            atomic(args.output / "RESUME.json", resume_manifest)
        scores, replay_ok = [], True
        for index, row in enumerate(rows):
            if args.device == "cuda":
                import torch
                # Variable-length generation leaves unused allocator blocks.
                # Release only unused blocks between documents, preserving all
                # model tensors, FP32 settings and the complete inference flow.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            recheck = resume_manifest is not None and index >= resume_rows - args.resume_recheck
            if resume_manifest is not None and index < resume_rows:
                source = args.resume_run / "private" / (digest(row["row_id"]) + ".json")
                result = json.loads(source.read_text())
                if result["text_sha256"] != digest(row["text"]):
                    raise RuntimeError("Resumed row text mismatch")
                if recheck:
                    observations = provider.observe(row["text"], config, window_config)
                    fresh = detector.detect(row["text"], observations)
                    if detector.detect(row["text"], observations) != fresh:
                        replay_ok = False
                        raise RuntimeError("Detector exact replay differs")
                    for key in ("arms", "candidate_count", "usable_candidates", "exclusions",
                                "event_count", "fact_identity_count", "budget"):
                        if result[key] != fresh[key]:
                            raise RuntimeError("Resumed row differs from fresh recomputation: " + key)
                    # GPU embedding similarities are reproducible within a process
                    # but drift at the last bits across process starts. Pair
                    # structure must match exactly; similarity drift is bounded
                    # and recorded.
                    if ([{k: v for k, v in p.items() if k != "similarity"} for p in result["pairs"]]
                            != [{k: v for k, v in p.items() if k != "similarity"} for p in fresh["pairs"]]):
                        raise RuntimeError("Resumed row differs from fresh recomputation: pairs")
                    drift = max((abs(p["similarity"] - q["similarity"])
                                 for p, q in zip(result["pairs"], fresh["pairs"])), default=0.)
                    if drift > 1e-4:
                        raise RuntimeError("Resumed row similarity drift exceeds tolerance")
                    resume_manifest["recheck_max_similarity_drift"] = max(
                        resume_manifest.get("recheck_max_similarity_drift", 0.), drift)
                    result = fresh
                result["detector_fingerprint"] = detector.fingerprint
                resume_manifest["row_files"][row["row_id"]] = file_sha(source)
            else:
                observations = provider.observe(row["text"], config, window_config)
                if args.replay:
                    for obs in observations:
                        atomic(observations_dir / (obs.input_sha256 + ".json"),
                               {"observation": asdict(obs), "audit": {"sealed_parent_replay": True}})
                result = detector.detect(row["text"], observations)
                repeated = detector.detect(row["text"], observations)
                if result != repeated:
                    replay_ok = False
                    raise RuntimeError("Detector exact replay differs")
            atomic(private / (digest(row["row_id"]) + ".json"), result)
            scores.append({
                **{k: v for k, v in row.items() if k != "text"},
                "text_sha256": result["text_sha256"], "detector_fingerprint": detector.fingerprint,
                "arms": {name: {k: arm[k] for k in ("statistic", "evidence_count", "status")}
                         for name, arm in result["arms"].items()},
                "candidate_count": result["candidate_count"],
                "usable_candidates": result["usable_candidates"],
            })
            telemetry = {}
            if args.device == "cuda":
                import torch
                free, total = torch.cuda.mem_get_info()
                telemetry = {"device_used_fraction": 1 - free / total,
                             "process_peak_bytes": torch.cuda.max_memory_allocated(),
                             "process_allocated_bytes": torch.cuda.memory_allocated(),
                             "process_reserved_bytes": torch.cuda.memory_reserved(),
                             "process_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                # PyTorch's caching-allocator counters may be zero when caching
                # is disabled. The device gate always uses cudaMemGetInfo.
                telemetry["allocator_counters_track_uncached_allocations"] = False
                if telemetry["device_used_fraction"] >= .8:
                    atomic(args.output / "MEMORY_GATE_FAILURE.json", {
                        "row_index": index, "elapsed_seconds": time.monotonic() - started,
                        **telemetry})
                    raise RuntimeError("GPU device utilization exceeds frozen memory gate")
            atomic(args.output / "HEARTBEAT.json", {
                "completed": index + 1, "total": len(rows),
                "elapsed_seconds": time.monotonic() - started, **telemetry})
        if (file_sha(args.input) != manifest["input_sha256"]
                or file_sha(args.bank) != manifest["bank_sha256"]
                or any(file_sha(ROOT / p) != h for p, h in code.items())):
            raise RuntimeError("Input/Bank/code drift during run")
        if args.scorer == "official":
            scorer.verify_assets()
            atomic(private / "BANK_AUDIT.json", scorer.audit)
        if resume_manifest is not None:
            atomic(args.output / "RESUME.json", resume_manifest)
        atomic(args.output / "SCORES.json", scores)
        atomic(args.output / "OBSERVATION_SEAL.json", {
            str(p.relative_to(args.output)): file_sha(p) for p in sorted(observations_dir.glob("*.json"))})
        atomic(args.output / "AUDIT.json", {
            "rows_expected": len(rows), "rows_scored": len(scores),
            "detector_exact_replay": replay_ok, "parser_exact_replay": provider.precheck_passed,
            "encoder_replay_max_abs": encoder.max_replay_error,
            "bank_scores_used_in_selection": False, "raw_text_in_public_scores": False,
            "original_boundaries_used": False, "independent_FPR_TPR_measured": False,
            "official_bank_graphs_exact_replayed": getattr(scorer, "exact_replays", None),
        })
        atomic(args.output / "TERMINAL.json", {
            "status": "score_complete", "manifest_sha256": file_sha(args.output / "MANIFEST.json"),
            "scores_sha256": file_sha(args.output / "SCORES.json"),
            "observations_sha256": file_sha(args.output / "OBSERVATION_SEAL.json"),
            "audit_sha256": file_sha(args.output / "AUDIT.json"),
        })
        print(json.dumps({"status": "score_complete", "rows": len(scores), "output": str(args.output)}))
    except Exception as exc:
        atomic(args.output / "HALT.json", {
            "status": "halted", "exception_type": type(exc).__name__,
            "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="Fresh audited AMR + local sentence encoder + target Bank")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--bank", type=Path, required=True)
    s.add_argument("--bank-sha256", required=True)
    s.add_argument("--writer-bank-sha256", help="Verified writing Bank; required for watermarked rows")
    s.add_argument("--scorer", choices=("official", "structural"), default="official")
    s.add_argument("--vectors", type=Path, help="Official S2Match GloVe vector file")
    s.add_argument("--encoder", type=Path, required=True)
    s.add_argument("--parser-model", type=Path, default=ROOT / "models/model_parse_xfm_bart_large-v0_1_0")
    s.add_argument("--tokenizer", type=Path, default=ROOT / "models/bart-large-tokenizer")
    s.add_argument("--replay", type=Path, help="Replay only a completed run made by this runner")
    s.add_argument("--resume-run", type=Path,
                   help="Carry forward scored rows from a halted run of the same frozen input")
    s.add_argument("--resume-rows", type=int, default=0,
                   help="Leading rows to carry forward; must equal the source heartbeat count")
    s.add_argument("--resume-recheck", type=int, default=3,
                   help="Trailing resumed rows recomputed and compared with fresh parsing")
    s.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    s.add_argument("--batched", action="store_true", help="Fresh within-document AMR batching")
    s.add_argument("--parser-batch-size", type=int, default=8)
    s.add_argument("--candidate-budget", type=int, default=24)
    s.add_argument("--top-k", type=int, default=4)
    s.add_argument("--min-similarity", type=float, default=.35)
    c = sub.add_parser("calibrate", help="Fit each arm separately on independent human nulls")
    c.add_argument("--scores", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--alpha", type=float, default=.05)
    e = sub.add_parser("evaluate", help="Report condition-specific human FPR and positive TPR")
    e.add_argument("--scores", type=Path, required=True)
    e.add_argument("--calibration", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "score":
        score(args)
    else:
        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = sealed_scores(args.scores)
        result = fit(rows, args.alpha) if args.command == "calibrate" else evaluate(
            rows, read(args.calibration))
        atomic(args.output, result)
        print(json.dumps({"status": args.command + "_complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
