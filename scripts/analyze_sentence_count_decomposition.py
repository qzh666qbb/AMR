"""Offline score decomposition on fixed, quality-aware Plan E selections.

Hybrid scores substitute sentence count or green rate algebraically. They are
not detector reruns, realizable AMRs, or causal effects of text interventions.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_feedback_evidence import group_candidates, read_vector, z_score, THRESHOLD


def decompose(n0, g0, n1, g1):
    before = z_score(n0, g0)
    after = z_score(n1, g1)
    count_only = z_score(n1, (g0 / n0) * n1)
    rate_only = z_score(n0, (g1 / n1) * n0)
    # Average the marginal drop over the two possible substitution orders.
    count_drop = ((before - count_only) + (rate_only - after)) / 2
    rate_drop = ((before - rate_only) + (count_only - after)) / 2
    if not math.isclose(count_drop + rate_drop, before - after, abs_tol=1e-10):
        raise ValueError("Decomposition does not sum to the observed score drop")
    return {
        "baseline": before, "count_only": count_only,
        "rate_only": rate_only, "observed": after,
        "count_drop": count_drop, "rate_drop": rate_drop,
    }


def summarize(rows):
    scenarios = {}
    for name in ("baseline", "count_only", "rate_only", "observed"):
        scenarios[name] = {
            "mean_z": statistics.mean(r[name] for r in rows) if rows else None,
            "low_count": sum(r[name] <= THRESHOLD for r in rows),
            "quality_and_low_count": sum(r["quality_valid"] and r[name] <= THRESHOLD for r in rows),
        }
    return {
        "n": len(rows), "scenarios": scenarios,
        "mean_count_drop": statistics.mean(r["count_drop"] for r in rows) if rows else None,
        "mean_rate_drop": statistics.mean(r["rate_drop"] for r in rows) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    base = root / "experiments/planE_e2_joint250"
    hashes = {}

    def load(path):
        content = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    reviews = load(base / "quality_aware_reviews.json")
    ordered = load(base / "rows.json")
    parsed = load(base / "parsed.json")
    matches = load(base / "detection/machine_amr_matches.json")["matched_amrs"]
    path = base / "detection/machine_z_scores.npy"
    scores = read_vector(path)
    hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = load(root / "experiments/mechanism_e2_amr/results.json")["rows"]
    before = {row["index"]: row for row in metadata}
    groups = group_candidates(reviews, 4)
    if len(before) != len(metadata) or set(before) != set(groups):
        raise ValueError("Missing or duplicate baseline metadata")
    if not len(reviews) == len(ordered) == len(parsed) == len(matches) == len(scores):
        raise ValueError("Detection inputs have different lengths")
    rows = []
    for index, pool in groups.items():
        selected = min([r for r in pool if r["quality_valid"]] or pool,
                       key=lambda r: (r["z_score"], r["candidate"]))
        pos, old = selected["position"], before[index]
        if pos < 0 or pos >= len(ordered):
            raise ValueError("Invalid global position")
        if any(selected[f] != ordered[pos][f] for f in ("index", "candidate", "source", "attack")):
            raise ValueError("Selection and detection input mismatch")
        if old["candidate"] != selected["candidate"] or old["quality_valid"] != selected["quality_valid"]:
            raise ValueError("Prior mechanism report selected a different candidate")
        n1 = len(parsed[pos])
        if len(matches[pos]) != n1:
            raise ValueError("Parsed sentence and match counts disagree")
        g1 = sum(item["score"] >= .65 for item in matches[pos])
        n0, g0 = old["before_sentences"], old["before_green_count"]
        values = decompose(n0, g0, n1, g1)
        for actual, expected in (
            (values["observed"], selected["z_score"]), (values["observed"], scores[pos]),
            (values["observed"], old["attack_z"]), (values["baseline"], old["baseline_z"]),
        ):
            if not math.isclose(actual, expected, abs_tol=1e-10):
                raise ValueError("Score reconstruction disagrees with saved evidence")
        rows.append({
            "index": index, "candidate": selected["candidate"], "position": pos,
            "quality_valid": selected["quality_valid"],
            "before_sentences": n0, "before_green": g0,
            "after_sentences": n1, "after_green": g1, **values,
        })
    successful = [r for r in rows if r["quality_valid"] and r["observed"] <= THRESHOLD]
    detected_before = [r for r in rows if r["baseline"] > THRESHOLD]
    successful_detected = [r for r in successful if r["baseline"] > THRESHOLD]
    categories = Counter(
        "both_individually_low" if r["count_only"] <= THRESHOLD and r["rate_only"] <= THRESHOLD
        else "count_only_low" if r["count_only"] <= THRESHOLD
        else "rate_only_low" if r["rate_only"] <= THRESHOLD
        else "joint_change_required"
        for r in successful_detected
    )
    result = {
        "protocol": {
            "threshold": THRESHOLD, "lambda": .25, "match_threshold": .65,
            "selection": "Frozen quality then minimum z; minimum z fallback for quality failures",
            "baseline_source": "Saved derived mechanism metadata, not a new baseline run",
            "hybrid_definition": "z(N1, p0*N1) and z(N0, p1*N0); fractional virtual green counts allowed",
            "attribution": "Two-feature Shapley decomposition of score, not causal attribution",
            "important": "No candidate reselection, text generation, parser or matcher reruns, or threshold calibration",
        },
        "all_selected": summarize(rows),
        "quality_selected": summarize([r for r in rows if r["quality_valid"]]),
        "successful_selected": summarize(successful),
        "baseline_detected": summarize(detected_before),
        "conditional_valid_success": {
            "successes": len(successful_detected), "denominator": len(detected_before),
            "rate": len(successful_detected) / len(detected_before) if detected_before else None,
            "status": "Derived metadata cross-check only; retain primary 191/250 convention",
        },
        "successful_baseline_detected_hybrid_categories": dict(categories),
        "single_amr_successes": sum(r["after_sentences"] == 1 for r in successful),
        "single_amr_successes_with_green_hit": sum(r["after_sentences"] == 1 and r["after_green"] == 1 for r in successful),
        "single_amr_successes_without_green_hit": sum(r["after_sentences"] == 1 and r["after_green"] == 0 for r in successful),
        "rows": rows, "input_sha256": hashes,
    }
    output = base / "sentence_count_decomposition.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "input_sha256", "protocol")}, indent=2))


if __name__ == "__main__":
    main()
