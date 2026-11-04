from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Iterator, List, Optional

from .amr_extractor import AMRExtractor, AmrlibBackend
from .lattice_builder import LatticeBuilder
from .sqlite_cache import AMRCache


ROOT = Path(__file__).resolve().parents[2]


def iter_jsonl_texts(
    path: Path,
    text_field: str,
    limit: Optional[int],
) -> Iterator[str]:
    emitted = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row[text_field]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Text field is empty or non-string at {path}:{line_number}"
                )
            yield text
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def chunked(values: Iterable[str], size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute parser-aware AMR records for all lattice spans"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--cache-db", required=True)
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
    parser.add_argument("--document-batch-size", type=int, default=32)
    parser.add_argument("--max-boundary-hops", type=int, default=3)
    parser.add_argument("--max-span-words", type=int, default=80)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    backend = AmrlibBackend(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        batch_size=args.model_batch_size,
    )
    cache = AMRCache(args.cache_db)
    extractor = AMRExtractor(
        backend=backend,
        cache=cache,
        request_batch_size=args.request_batch_size,
    )
    builder = LatticeBuilder(
        max_boundary_hops=args.max_boundary_hops,
        max_span_words=args.max_span_words,
    )

    documents = iter_jsonl_texts(args.input, args.text_field, args.limit)
    completed = 0
    for document_batch in chunked(documents, args.document_batch_size):
        spans = []
        for text in document_batch:
            lattice = builder.build_lattice(text)
            spans.extend(edge.span_text for edge in lattice.edges)
        extractor.extract_many(spans)
        completed += len(document_batch)
        print(
            json.dumps(
                {
                    "documents_completed": completed,
                    "cache_records": cache.count(backend.fingerprint),
                    "stats": vars(extractor.stats),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "complete",
                "documents_completed": completed,
                "cache_records": cache.count(backend.fingerprint),
                "parser_fingerprint": backend.fingerprint,
                "stats": vars(extractor.stats),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"precompute failed: {exc}", file=sys.stderr)
        raise
