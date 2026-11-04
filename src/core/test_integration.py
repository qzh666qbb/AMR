from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .amr_extractor import AMRExtractor, AmrlibBackend, HeuristicBackend
from .lattice_builder import LatticeBuilder
from .real_scorer import RealScorer
from .sqlite_cache import AMRCache
from .viterbi_aligner import ViterbiAligner


ROOT = Path(__file__).resolve().parents[2]


def build_backend(args):
    if args.backend == "heuristic":
        return HeuristicBackend()
    return AmrlibBackend(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


def run_integration(args) -> dict:
    targets = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaped over a sleeping dog.",
        "The cat watched them.",
        "The birds flew away.",
    ]
    attacked = (
        "The quick brown fox jumps over the lazy dog and a fast brown fox "
        "leaped over a sleeping dog. The cat watched them; the birds flew away."
    )

    backend = build_backend(args)
    cache = AMRCache(args.cache_db)
    lattice = LatticeBuilder(max_boundary_hops=3).build_lattice(attacked)
    aligner = ViterbiAligner()

    cold_extractor = AMRExtractor(backend=backend, cache=cache)
    cold_scorer = RealScorer(cold_extractor)
    started = time.monotonic()
    cold_result = aligner.align(lattice, targets, cold_scorer)
    cold_seconds = time.monotonic() - started

    warm_extractor = AMRExtractor(backend=backend, cache=cache)
    warm_scorer = RealScorer(warm_extractor)
    started = time.monotonic()
    warm_result = aligner.align(lattice, targets, warm_scorer)
    warm_seconds = time.monotonic() - started

    payload = {
        "backend": args.backend,
        "parser_fingerprint": backend.fingerprint,
        "lattice_nodes": lattice.node_count,
        "lattice_edges": len(lattice.edges),
        "cache_records": cache.count(backend.fingerprint),
        "cold": {
            "seconds": cold_seconds,
            "normalized_score": cold_result.normalized_score,
            "stats": vars(cold_extractor.stats),
        },
        "warm": {
            "seconds": warm_seconds,
            "normalized_score": warm_result.normalized_score,
            "stats": vars(warm_extractor.stats),
        },
        "alignment": [
            {
                "operation": step.operation,
                "target_index": step.target_index,
                "match_score": step.match_score,
                "span": step.edge.span_text if step.edge else None,
            }
            for step in warm_result.steps
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the lattice + AMR + SQLite + DP integration smoke test"
    )
    parser.add_argument(
        "--backend",
        choices=("heuristic", "amrlib"),
        default="heuristic",
    )
    parser.add_argument(
        "--cache-db",
        default=str(ROOT / "build/swan_elastic/integration_cache.db"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=str(ROOT / "models/bart-large-tokenizer"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_integration(parse_args())
