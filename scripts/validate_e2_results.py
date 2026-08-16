"""Apply predeclared surface constraints and summarize E2 pilot results."""

import json
import re
from pathlib import Path

import numpy as np
import spacy

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/e2_attack_pilot"
FAMILIES = ("plain_paraphrase", "sentence_boundary", "amr_guided")


def numbers(text):
    return set(re.findall(r"\b\d[\d,.:/%-]*\b", text))


def entities(text, nlp):
    return {(ent.text, ent.label_) for ent in nlp(text).ents}


def has_negation(text):
    return bool(re.search(r"\b(?:no|not|never|without|n't)\b", text, re.I))


def main():
    with (PILOT / "candidates.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    nlp = spacy.load("en_core_web_sm")
    threshold = float(payload["selection_threshold_1pct_fpr"])
    summaries = {}
    all_rows = []
    for family in FAMILIES:
        z = np.load(PILOT / "detection" / family / "machine_z_scores.npy")
        rows = []
        for idx, record in enumerate(payload["records"]):
            source = record["source"]
            candidate = record["candidates"][family]
            source_entities = entities(source, nlp)
            candidate_entities = entities(candidate, nlp)
            check = {
                "numbers_preserved": numbers(source).issubset(numbers(candidate)),
                "negation_preserved": has_negation(source) == has_negation(candidate),
                "entities_preserved": source_entities.issubset(candidate_entities),
                "length_ratio_ok": 0.45 <= len(candidate) / max(1, len(source)) <= 2.0,
            }
            quality_pass = all(check.values())
            row = {
                "source_index": record["source_index"],
                "z_score": float(z[idx]),
                "baseline_z": float(record["baseline_z"]),
                "z_drop": float(record["baseline_z"] - z[idx]),
                "escaped_at_1pct": bool(z[idx] <= threshold),
                "quality_pass": quality_pass,
                "valid_escape": bool(quality_pass and z[idx] <= threshold),
                "checks": check,
            }
            rows.append(row)
            all_rows.append({"family": family, **row})
        summaries[family] = {
            "n": len(rows),
            "quality_pass": sum(r["quality_pass"] for r in rows),
            "escape_count": sum(r["escaped_at_1pct"] for r in rows),
            "escape_rate": float(np.mean([r["escaped_at_1pct"] for r in rows])),
            "valid_escape_count": sum(r["valid_escape"] for r in rows),
            "valid_asr": float(np.mean([r["valid_escape"] for r in rows])),
            "mean_z": float(np.mean(z)),
            "mean_z_drop": float(np.mean([r["z_drop"] for r in rows])),
            "rows": rows,
        }
    output = {"threshold_1pct_fpr": threshold, "summaries": summaries, "rows": all_rows}
    with (PILOT / "validated_results.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    for family, summary in summaries.items():
        print(family, {k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
