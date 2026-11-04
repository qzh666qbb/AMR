"""Evaluate retries online, or recheck saved reviews without API calls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re

import penman

from audit_feedback_evidence import read_vector, THRESHOLD


def identity(row):
    return row["index"], row["candidate"]


def gates(row, graphs):
    source, attack = row["source"], row["attack"]
    numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", source))
    attack_numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", attack))
    negation = re.compile(r"\b(?:no|not|never|without|cannot|n't)\b", re.I)
    ratio = len(attack.split()) / max(1, len(source.split()))
    amr_ok = bool(graphs)
    for graph in graphs:
        try:
            penman.decode(graph)
        except (penman.exceptions.PenmanError, ValueError, TypeError):
            amr_ok = False
    return {
        "length_ok": .7 <= ratio <= 1.6,
        "numbers_ok": numbers.issubset(attack_numbers),
        "negation_ok": bool(negation.search(source)) == bool(negation.search(attack)),
        "amr_ok": amr_ok,
    }


def evaluate_rows(rows, scores, parsed, reviews):
    if not len(rows) == len(scores) == len(parsed) == len(reviews):
        raise ValueError("Candidate, score, AMR and review counts must agree")
    lookup = {identity(row): row for row in reviews}
    keys = {identity(row) for row in rows}
    if len(keys) != len(rows) or len(lookup) != len(reviews) or keys != set(lookup):
        raise ValueError("Duplicate or missing candidate identity")
    result = []
    # Position belongs to the global detection input, not a paragraph subgroup.
    for position, (row, score, graphs) in enumerate(zip(rows, scores, parsed)):
        saved = lookup[identity(row)]
        if any(row[field] != saved[field] for field in ("source", "attack")):
            raise ValueError("Review text does not match the detection input")
        if not math.isfinite(score) or (
            "z_score" in saved and not math.isclose(saved["z_score"], score, abs_tol=1e-10)
        ):
            raise ValueError("Nonfinite or mismatched detection score")
        surface = gates(row, graphs)
        review = saved["review"]
        quality = all(surface.values()) and review.get("score", 0) >= 4 and not review.get("major_error", True)
        result.append({**row, "position": position, "z_score": score,
                       "surface": surface, "review": review, "quality_valid": quality})
    return result


def best_valid(rows):
    best = {}
    for row in rows:
        if row["quality_valid"] and (
            row["index"] not in best
            or (row["z_score"], row["candidate"]) < (best[row["index"]]["z_score"], best[row["index"]]["candidate"])
        ):
            best[row["index"]] = row
    return [best[index] for index in sorted(best)]


def request_reviews(rows, key):
    import requests

    prompt = (
        'Evaluate candidates against SOURCE. Return JSON array of four objects '
        '{"candidate":number,"score":1-5,"major_error":boolean,"reason":"short"}. '
        'Major error means changed entity, number, polarity, modality, attribution, event, or relation.\nSOURCE:\n'
        + rows[0]["source"] + "\n"
        + "\n".join(f'\nCANDIDATE {r["candidate"]}:\n{r["attack"]}' for r in rows)
    )
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 700, "thinking": {"type": "disabled"}},
            timeout=240,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        items = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        lookup = {int(item["candidate"]): item for item in items}
    except Exception as exc:
        lookup = {r["candidate"]: {"score": 0, "major_error": True, "reason": type(exc).__name__} for r in rows}
    return [{**r, "review": lookup.get(r["candidate"], {"score": 0, "major_error": True})} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-reviews", action="store_true", help="Offline recheck; preserve historical files")
    args = parser.parse_args()
    base = args.root / "experiments/planE_e2_joint250"
    folder = base / "adaptive_retry"
    hashes = {}

    def load(path):
        content = path.read_bytes()
        hashes[str(path.relative_to(args.root))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    rows = load(folder / "rows.json")
    parsed = load(folder / "parsed.json")
    score_path = folder / "detection/machine_z_scores.npy"
    scores = read_vector(score_path)
    hashes[str(score_path.relative_to(args.root))] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    if args.reuse_reviews:
        reviews = load(folder / "evaluated.json")
    else:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key and (args.root / "1.txt").exists():
            key = (args.root / "1.txt").read_text().strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY missing; use --reuse-reviews for offline validation")
        groups = [[r for r in rows if r["index"] == i] for i in sorted({r["index"] for r in rows})]
        with ThreadPoolExecutor(max_workers=10) as pool:
            reviews = [r for batch in pool.map(lambda group: request_reviews(group, key), groups) for r in batch]
    evaluated = evaluate_rows(rows, scores, parsed, reviews)
    best = best_valid(evaluated)
    summary = {"candidates": len(evaluated), "quality_candidates": sum(r["quality_valid"] for r in evaluated),
               "quality_paragraphs": len(best), "successful_retry_paragraphs": sum(r["z_score"] <= THRESHOLD for r in best)}
    if args.reuse_reviews:
        previous = {identity(r): r for r in reviews}
        changes = [{"index": r["index"], "candidate": r["candidate"],
                    "surface_before": previous[identity(r)]["surface"], "surface_after": r["surface"],
                    "quality_before": previous[identity(r)]["quality_valid"], "quality_after": r["quality_valid"]}
                   for r in evaluated if r["surface"] != previous[identity(r)]["surface"]
                   or r["quality_valid"] != previous[identity(r)]["quality_valid"]]
        initial = load(base / "quality_aware_selected.json")
        initial_ids = {r["index"] for r in initial if r["z_score"] <= THRESHOLD}
        retry_ids = {r["index"] for r in best if r["z_score"] <= THRESHOLD}
        summary.update({"changed_candidates": len(changes), "initial_successes": len(initial_ids),
                        "new_successes": len(retry_ids - initial_ids), "combined_successes": len(initial_ids | retry_ids)})
        payload = {"mode": "offline_saved_semantic_reviews", "penman_version": penman.__version__,
                   "summary": summary, "changes": changes, "rows": evaluated, "input_sha256": hashes}
        (folder / "index_recheck.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    else:
        (folder / "evaluated.json").write_text(json.dumps(evaluated, indent=2) + "\n")
        (folder / "quality_valid_best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
