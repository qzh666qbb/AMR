"""Run graph-level AMR invariance controls for the D1-A attack study."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import penman


ROOT = Path(__file__).resolve().parents[1]
SWAN = ROOT / "third_party" / "SWAN"
sys.path.insert(0, str(SWAN))

from utils.amr_utils import (  # noqa: E402
    compute_s2match_score,
    load_amr_bank,
    normalize_amr_variables,
)


def rename_variables(amr: str, index: int) -> str:
    graph = penman.decode(amr)
    variables = sorted(graph.variables())
    mapping = {var: f"r{index}_{i}" for i, var in enumerate(variables)}
    triples = [
        (mapping.get(source, source), role, mapping.get(target, target))
        for source, role, target in graph.triples
    ]
    return penman.encode(penman.Graph(triples, top=mapping.get(graph.top, graph.top)))


def reorder_triples(amr: str, rng: random.Random) -> str:
    graph = penman.decode(amr)
    triples = list(graph.triples)
    rng.shuffle(triples)
    return penman.encode(penman.Graph(triples, top=graph.top))


def reformat_amr(amr: str, index: int) -> str:
    graph = penman.decode(amr)
    del index
    return penman.encode(graph, indent=None)


def collect_amrs(path: Path, limit: int) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    selected = []
    for document in documents:
        for amr in document:
            if isinstance(amr, str) and amr.strip().startswith("("):
                try:
                    penman.decode(amr)
                except Exception:
                    continue
                selected.append(amr)
                if len(selected) == limit:
                    return selected
    raise RuntimeError(f"Only found {len(selected)} valid AMRs; need {limit}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--parsed-amrs",
        type=Path,
        default=ROOT
        / "baselines/local/swan_acl2026_realnews/runs/formal_250x5/parsed_amrs.json",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=SWAN / "amr_bank/banks/amr_bank_50.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/e1_invariance/results.json",
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "GLOVE_VECTORS_PATH", str(SWAN / "vectors/glove.6B.100d.txt")
    )
    rng = random.Random(args.seed)
    amrs = collect_amrs(args.parsed_amrs, args.limit)
    bank = load_amr_bank(str(args.bank))
    rows = []

    for index, original in enumerate(amrs):
        transformations = {
            "variable_rename": rename_variables(original, index),
            "triple_reorder": reorder_triples(original, rng),
            "whitespace_format": reformat_amr(original, index),
            "canonical_normalize": normalize_amr_variables(original),
        }
        original_bank_scores = [compute_s2match_score(original, item) for item in bank]
        original_max = max(original_bank_scores)
        for name, transformed in transformations.items():
            pair_score = compute_s2match_score(original, transformed)
            transformed_max = max(
                compute_s2match_score(transformed, item) for item in bank
            )
            rows.append(
                {
                    "sample": index,
                    "transformation": name,
                    "pair_s2match": pair_score,
                    "original_bank_max": original_max,
                    "transformed_bank_max": transformed_max,
                    "bank_max_delta": transformed_max - original_max,
                }
            )

    pair_scores = np.array([row["pair_s2match"] for row in rows])
    bank_deltas = np.array([row["bank_max_delta"] for row in rows])
    summary = {
        "samples": len(amrs),
        "comparisons": len(rows),
        "seed": args.seed,
        "transformations": sorted({row["transformation"] for row in rows}),
        "pair_s2match_min": float(pair_scores.min()),
        "pair_s2match_mean": float(pair_scores.mean()),
        "bank_max_abs_delta_max": float(np.abs(bank_deltas).max()),
        "bank_max_abs_delta_mean": float(np.abs(bank_deltas).mean()),
        "pair_failures_below_0_999": int(np.sum(pair_scores < 0.999)),
        "bank_failures_above_1e_9": int(np.sum(np.abs(bank_deltas) > 1e-9)),
        "gate_pass": bool(
            np.all(pair_scores >= 0.999) and np.all(np.abs(bank_deltas) <= 1e-9)
        ),
    }
    payload = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
