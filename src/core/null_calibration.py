from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .bank_detector import add_runtime_arguments, build_detector
from .models import CalibrationResult, DetectionResult


CALIBRATION_VERSION = "swan-empirical-null:v1"
DEFAULT_WORD_BINS = (
    ("words-000-080", 0, 80),
    ("words-081-160", 81, 160),
    ("words-161-320", 161, 320),
    ("words-321-plus", 321, None),
)


def _bucket_for(
    word_count: int,
    bins: Sequence[Tuple[str, int, Optional[int]]],
) -> str:
    if word_count < 0:
        raise ValueError("word_count must be non-negative")
    for label, minimum, maximum in bins:
        if word_count >= minimum and (maximum is None or word_count <= maximum):
            return label
    raise ValueError(f"word_count {word_count} is outside the declared bins")


class EmpiricalNullCalibrator:
    """Finite-sample empirical upper-tail calibration for detector statistics."""

    def __init__(self, artifact: dict) -> None:
        if artifact.get("version") != CALIBRATION_VERSION:
            raise ValueError("Unsupported empirical Null artifact version")
        self.artifact = artifact
        self.bins = tuple(
            (row["label"], int(row["min_words"]), row["max_words"])
            for row in artifact["word_bins"]
        )
        if not artifact.get("global_statistics"):
            raise ValueError("Empirical Null artifact has no statistics")

    @classmethod
    def fit(
        cls,
        samples: Iterable[Tuple[str, DetectionResult]],
        detector_fingerprint: str,
        bank_fingerprint: str,
        bank_sha256: str,
        parser_fingerprint: str,
        min_bucket_size: int = 20,
        bins: Sequence[Tuple[str, int, Optional[int]]] = DEFAULT_WORD_BINS,
    ) -> "EmpiricalNullCalibrator":
        if min_bucket_size < 1:
            raise ValueError("min_bucket_size must be positive")
        global_statistics: List[float] = []
        bucket_statistics: Dict[str, List[float]] = {
            label: [] for label, _, _ in bins
        }
        for text, result in samples:
            statistic = float(result.statistic)
            word_count = len(text.split())
            label = _bucket_for(word_count, bins)
            global_statistics.append(statistic)
            bucket_statistics[label].append(statistic)
        if not global_statistics:
            raise ValueError("At least one Null sample is required")

        artifact = {
            "version": CALIBRATION_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detector_fingerprint": detector_fingerprint,
            "bank_fingerprint": bank_fingerprint,
            "bank_sha256": bank_sha256,
            "parser_fingerprint": parser_fingerprint,
            "min_bucket_size": min_bucket_size,
            "sample_count": len(global_statistics),
            "minimum_resolvable_p": 1.0 / (len(global_statistics) + 1),
            "word_bins": [
                {
                    "label": label,
                    "min_words": minimum,
                    "max_words": maximum,
                }
                for label, minimum, maximum in bins
            ],
            "global_statistics": sorted(global_statistics),
            "bucket_statistics": {
                label: sorted(values)
                for label, values in bucket_statistics.items()
            },
        }
        return cls(artifact)

    def assert_compatible(
        self,
        detector_fingerprint: str,
        bank_fingerprint: str,
        parser_fingerprint: str,
    ) -> None:
        expected = {
            "detector_fingerprint": detector_fingerprint,
            "bank_fingerprint": bank_fingerprint,
            "parser_fingerprint": parser_fingerprint,
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.artifact.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Calibration artifact fingerprint mismatch: "
                + ", ".join(mismatches)
            )

    def calibrate(
        self,
        statistic: float,
        word_count: int,
        alpha: float = 0.05,
    ) -> CalibrationResult:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        bucket = _bucket_for(word_count, self.bins)
        bucket_values = self.artifact["bucket_statistics"].get(bucket, [])
        if len(bucket_values) >= self.artifact["min_bucket_size"]:
            null_values = bucket_values
            scope = "bucket"
        else:
            null_values = self.artifact["global_statistics"]
            scope = "global_fallback"
        exceedances = sum(value >= statistic for value in null_values)
        p_value = (1.0 + exceedances) / (len(null_values) + 1.0)
        return CalibrationResult(
            statistic=float(statistic),
            p_value=p_value,
            alpha=alpha,
            is_watermarked=p_value <= alpha,
            null_count=len(null_values),
            scope=scope,
            bucket=bucket,
        )

    def save(self, path: Path) -> None:
        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.artifact,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=destination.name + ".",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary = stream.name
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: Path) -> "EmpiricalNullCalibrator":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def iter_input_texts(
    path: Path,
    text_field: str,
    collection_field: Optional[str],
    limit: Optional[int],
) -> Iterator[str]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if path.suffix.lower() == ".jsonl":
        rows: Iterable[object] = (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if collection_field is not None:
            if not isinstance(payload, dict):
                raise ValueError("collection_field requires a JSON object")
            rows = payload.get(collection_field)
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("JSON calibration input must resolve to a list")

    emitted = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Calibration row {index} is not an object")
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Calibration row {index} has no non-empty {text_field!r}"
            )
        yield text
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a frozen empirical Null for SWAN Bank lattice detection"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--collection-field")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-bucket-size", type=int, default=20)
    add_runtime_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args)
    texts = list(
        iter_input_texts(
            path=args.input,
            text_field=args.text_field,
            collection_field=args.collection_field,
            limit=args.limit,
        )
    )
    completed: List[Tuple[str, DetectionResult]] = []
    for index, text in enumerate(texts, start=1):
        result = detector.detect(text)
        completed.append((text, result))
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(texts),
                    "words": len(text.split()),
                    "statistic": result.statistic,
                    "unit_count": result.unit_count,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    calibrator = EmpiricalNullCalibrator.fit(
        samples=completed,
        detector_fingerprint=detector.fingerprint,
        bank_fingerprint=detector.scorer.fingerprint,
        bank_sha256=detector.scorer.bank_sha256,
        parser_fingerprint=detector.scorer.parser_fingerprint,
        min_bucket_size=args.min_bucket_size,
    )
    calibrator.save(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "sample_count": calibrator.artifact["sample_count"],
                "minimum_resolvable_p": calibrator.artifact[
                    "minimum_resolvable_p"
                ],
                "formal_1pct_ready": calibrator.artifact["sample_count"] >= 99,
                "extractor_stats": asdict(detector.scorer.extractor.stats),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
