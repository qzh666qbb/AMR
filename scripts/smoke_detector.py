"""Run a small, metric-free SWAN detector smoke test on pre-parsed AMRs."""

import argparse
import json
import math
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swan-root", required=True)
    parser.add_argument("--parsed-amrs", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--lmbd", type=float, default=0.25)
    args = parser.parse_args()

    swan_root = os.path.abspath(args.swan_root)
    sys.path.insert(0, swan_root)
    from detection.detect_from_parsed_amrs import compute_zscore_for_doc
    from utils.amr_utils import load_amr_bank

    with open(args.parsed_amrs, encoding="utf-8") as handle:
        documents = json.load(handle)
    bank = load_amr_bank(args.bank)

    started = time.perf_counter()
    rows = []
    for index, document in enumerate(documents):
        _, z_score, matches = compute_zscore_for_doc(
            index, document, bank, args.threshold, args.lmbd, True
        )
        best_scores = [float(score) for score, _ in matches]
        rows.append(
            {
                "index": index,
                "sentence_count": len(document),
                "z_score": None if math.isnan(z_score) else float(z_score),
                "best_s2match_scores": best_scores,
                "green_count": sum(score >= args.threshold for score in best_scores),
            }
        )

    payload = {
        "contract": {
            "threshold": args.threshold,
            "lambda": args.lmbd,
            "bank_size": len(bank),
            "document_count": len(documents),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "documents": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["contract"], ensure_ascii=False))
    print(f"elapsed_seconds={payload['elapsed_seconds']:.3f}")
    print(f"output={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
