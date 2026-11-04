from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

from .amr_extractor import AMRExtractor, AmrlibBackend, HeuristicBackend
from .bank_scorer import SwanAMRBankScorer
from .lattice_builder import LatticeBuilder
from .models import BankMatch, DetectionResult, DetectionUnit, Lattice, LatticeEdge
from .sqlite_cache import AMRCache


ROOT = Path(__file__).resolve().parents[2]
DETECTOR_VERSION = "swan-bank-lattice-detector:v1"


class SwanBankLatticeDetector:
    """Target-free detector that selects the strongest full-coverage lattice path."""

    def __init__(
        self,
        builder: LatticeBuilder,
        scorer: SwanAMRBankScorer,
        score_center: float = 0.65,
        green_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= score_center <= 1.0:
            raise ValueError("score_center must be between zero and one")
        if not 0.0 <= green_threshold <= 1.0:
            raise ValueError("green_threshold must be between zero and one")
        self.builder = builder
        self.scorer = scorer
        self.score_center = score_center
        self.green_threshold = green_threshold
        config = {
            "detector_version": DETECTOR_VERSION,
            "green_threshold": green_threshold,
            "include_soft_boundaries": builder.include_soft_boundaries,
            "max_boundary_hops": builder.max_boundary_hops,
            "max_span_words": builder.max_span_words,
            "score_center": score_center,
        }
        self.fingerprint = "swan-bank-detector:" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def detect(self, text: str) -> DetectionResult:
        return self.detect_lattice(self.builder.build_lattice(text))

    def detect_lattice(self, lattice: Lattice) -> DetectionResult:
        if lattice.node_count == 1:
            return DetectionResult(
                statistic=0.0,
                raw_utility=0.0,
                unit_count=0,
                green_count=0,
                units=(),
            )

        edge_matches = self.scorer.score_many(
            tuple(edge.span_text for edge in lattice.edges)
        )
        outgoing: List[List[Tuple[LatticeEdge, BankMatch]]] = [
            [] for _ in range(lattice.node_count)
        ]
        for edge, match in zip(lattice.edges, edge_matches):
            outgoing[edge.start].append((edge, match))

        negative_infinity = float("-inf")
        utility = [negative_infinity] * lattice.node_count
        unit_count = [0] * lattice.node_count
        back: List[Optional[Tuple[int, DetectionUnit]]] = [
            None
        ] * lattice.node_count
        utility[0] = 0.0
        epsilon = 1e-12

        for node in range(lattice.node_count):
            if utility[node] == negative_infinity:
                continue
            for edge, match in outgoing[node]:
                edge_utility = match.score - self.score_center
                candidate_utility = utility[node] + edge_utility
                candidate_count = unit_count[node] + 1
                improves = candidate_utility > utility[edge.end] + epsilon
                tie_with_fewer_units = (
                    abs(candidate_utility - utility[edge.end]) <= epsilon
                    and (
                        back[edge.end] is None
                        or candidate_count < unit_count[edge.end]
                    )
                )
                if not improves and not tie_with_fewer_units:
                    continue
                unit = DetectionUnit(
                    edge=edge,
                    match=match,
                    utility=edge_utility,
                )
                utility[edge.end] = candidate_utility
                unit_count[edge.end] = candidate_count
                back[edge.end] = (node, unit)

        final_node = lattice.node_count - 1
        if utility[final_node] == negative_infinity:
            raise RuntimeError("Lattice has no full-coverage path")

        units: List[DetectionUnit] = []
        node = final_node
        while node != 0:
            pointer = back[node]
            if pointer is None:
                raise RuntimeError("Detector backtrace is incomplete")
            node, unit = pointer
            units.append(unit)
        units.reverse()

        count = len(units)
        raw_utility = utility[final_node]
        statistic = raw_utility / math.sqrt(count) if count else 0.0
        green_count = sum(
            unit.match.score >= self.green_threshold for unit in units
        )
        return DetectionResult(
            statistic=statistic,
            raw_utility=raw_utility,
            unit_count=count,
            green_count=green_count,
            units=tuple(units),
        )


def detection_payload(result: DetectionResult) -> dict:
    return {
        "statistic": result.statistic,
        "raw_utility": result.raw_utility,
        "unit_count": result.unit_count,
        "green_count": result.green_count,
        "units": [
            {
                "start": unit.edge.start,
                "end": unit.edge.end,
                "span_text": unit.edge.span_text,
                "boundary_hops": unit.edge.boundary_hops,
                "utility": unit.utility,
                "match": asdict(unit.match),
            }
            for unit in result.units
        ],
    }


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bank",
        default=str(
            ROOT
            / "third_party/SWAN_official_2ca0c207_clean"
            / "amr_bank/banks/amr_bank_100.json"
        ),
    )
    parser.add_argument(
        "--cache-db",
        default=str(ROOT / "build/swan_elastic/bank_detector_cache.db"),
    )
    parser.add_argument(
        "--backend",
        choices=("amrlib", "heuristic"),
        default="amrlib",
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models/model_parse_xfm_bart_large-v0_1_0"),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=str(ROOT / "models/bart-large-tokenizer"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--request-batch-size", type=int, default=128)
    parser.add_argument("--max-boundary-hops", type=int, default=3)
    parser.add_argument("--max-span-words", type=int, default=80)
    parser.add_argument("--score-center", type=float, default=0.65)
    parser.add_argument("--green-threshold", type=float, default=0.75)


def build_detector(args: argparse.Namespace) -> SwanBankLatticeDetector:
    if args.backend == "heuristic":
        backend = HeuristicBackend()
    else:
        backend = AmrlibBackend(
            model_dir=args.model_dir,
            tokenizer_dir=args.tokenizer_dir,
            device=args.device,
            batch_size=args.model_batch_size,
        )
    extractor = AMRExtractor(
        backend=backend,
        cache=AMRCache(args.cache_db),
        request_batch_size=args.request_batch_size,
    )
    scorer = SwanAMRBankScorer(extractor=extractor, bank_path=args.bank)
    builder = LatticeBuilder(
        max_boundary_hops=args.max_boundary_hops,
        max_span_words=args.max_span_words,
    )
    return SwanBankLatticeDetector(
        builder=builder,
        scorer=scorer,
        score_center=args.score_center,
        green_threshold=args.green_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target-free SWAN AMR Bank lattice detection"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    add_runtime_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = (
        args.text
        if args.text is not None
        else args.text_file.read_text(encoding="utf-8")
    )
    detector = build_detector(args)
    result = detector.detect(text)
    payload = {
        "detector_fingerprint": detector.fingerprint,
        "bank_fingerprint": detector.scorer.fingerprint,
        "bank_sha256": detector.scorer.bank_sha256,
        "parser_fingerprint": detector.scorer.parser_fingerprint,
        "detection": detection_payload(result),
    }
    if args.calibration is not None:
        from .null_calibration import EmpiricalNullCalibrator

        calibrator = EmpiricalNullCalibrator.load(args.calibration)
        calibrator.assert_compatible(
            detector_fingerprint=detector.fingerprint,
            bank_fingerprint=detector.scorer.fingerprint,
            parser_fingerprint=detector.scorer.parser_fingerprint,
        )
        payload["calibration"] = asdict(
            calibrator.calibrate(
                statistic=result.statistic,
                word_count=len(text.split()),
                alpha=args.alpha,
            )
        )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
