from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
import tempfile

from .amr_extractor import AMRExtractor, AmrlibBackend
from .sqlite_cache import AMRCache


ROOT = Path(__file__).resolve().parents[2]


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an AMR parser host before a cache precompute run"
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parse-smoke", action="store_true")
    args = parser.parse_args()

    import torch

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("amrlib", "penman", "torch", "transformers")
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "requested_device": args.device,
    }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    backend = AmrlibBackend(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    report["parser_fingerprint"] = backend.fingerprint

    if args.parse_smoke:
        with tempfile.TemporaryDirectory(prefix="swan-elastic-preflight-") as folder:
            extractor = AMRExtractor(
                backend=backend,
                cache=AMRCache(str(Path(folder) / "cache.db")),
                request_batch_size=1,
            )
            record = extractor.extract("The cat watches the bird.")
            report["parse_smoke"] = {
                "concepts": record.concepts,
                "triples": record.triples,
                "stats": vars(extractor.stats),
            }

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
