"""Audit saved Plan E evidence without model inference or external dependencies."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import struct

THRESHOLD = 2.33
LAMBDA = 0.25
MATCH_THRESHOLD = 0.65


def read_vector(path):
    """Read only one-dimensional little-endian float NPY files, never pickle."""
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("Invalid NPY magic")
        version = handle.read(2)
        if version not in (b"\x01\x00", b"\x02\x00"):
            raise ValueError("Unsupported NPY version")
        width = 2 if version == b"\x01\x00" else 4
        header_size = int.from_bytes(handle.read(width), "little")
        header = ast.literal_eval(handle.read(header_size).decode("latin1").strip())
        shape, dtype = header["shape"], header["descr"]
        if len(shape) != 1 or dtype not in ("<f8", "<f4"):
            raise ValueError("Expected a float vector")
        fmt = "d" if dtype == "<f8" else "f"
        values = list(struct.unpack("<" + str(shape[0]) + fmt, handle.read()))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Empty or nonfinite score vector")
    return values


def z_score(n, green):
    if n < 1 or not 0 <= green <= n:
        raise ValueError("Invalid sentence/green count")
    return (green - LAMBDA * n) / math.sqrt(n * LAMBDA * (1 - LAMBDA))


def valid(row):
    return row["quality_valid"] and row["z_score"] <= THRESHOLD


def group_candidates(rows, k=None):
    groups = defaultdict(list)
    keys = set()
    for row in rows:
        key = (row["index"], row["candidate"])
        if key in keys:
            raise ValueError("Duplicate candidate identity")
        keys.add(key)
        if type(row["quality_valid"]) is not bool or not math.isfinite(row["z_score"]):
            raise ValueError("Invalid quality flag or score")
        groups[row["index"]].append(row)
    for items in groups.values():
        if len({row["source"] for row in items}) != 1:
            raise ValueError("Inconsistent source within paragraph")
        if k is not None and {row["candidate"] for row in items} != set(range(k)):
            raise ValueError("Incomplete candidate group")
        items.sort(key=lambda row: row["candidate"])
    return dict(sorted(groups.items()))


def policy_success(pool, policy):
    if policy == "random":
        return sum(valid(row) for row in pool) / len(pool)
    quality = [row for row in pool if row["quality_valid"]]
    if policy == "quality_only":
        return sum(valid(row) for row in quality) / len(quality) if quality else 0.0
    eligible = pool if policy == "min_z" else quality or pool
    if policy not in ("min_z", "quality_then_min_z"):
        raise ValueError("Unknown policy")
    return float(valid(min(eligible, key=lambda row: (row["z_score"], row["candidate"]))))


def exact_budget(groups):
    output = {}
    for k in range(1, 5):
        output[str(k)] = {}
        for policy in ("random", "min_z", "quality_only", "quality_then_min_z"):
            values = []
            for pool in groups.values():
                subsets = list(combinations(pool, k))
                values.append(sum(policy_success(s, policy) for s in subsets) / len(subsets))
            output[str(k)][policy] = sum(values) / len(values)
    return output


def feedback_interval(groups, repeats=2000):
    deltas = [
        policy_success(pool, "quality_then_min_z") - policy_success(pool, "quality_only")
        for pool in groups.values()
    ]
    rng = random.Random(20260911)
    samples = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(repeats)
    )
    return {
        "mean": sum(deltas) / len(deltas),
        "paired_paragraph_bootstrap_ci95": [samples[int(repeats * .025)], samples[int(repeats * .975)]],
        "seed": 20260911,
        "repeats": repeats,
        "scope": "Sampling uncertainty across saved paragraphs; not new-generation uncertainty",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    hashes = {}

    def load(relative):
        path = root / relative
        content = path.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def vector(relative):
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_vector(path)

    base = "experiments/planE_e2_joint250/"
    reviews = load(base + "quality_aware_reviews.json")
    candidates = load(base + "candidates.json")["rows"]
    ordered = load(base + "rows.json")
    parsed = load(base + "parsed.json")
    matches = load(base + "detection/machine_amr_matches.json")["matched_amrs"]
    scores = vector(base + "detection/machine_z_scores.npy")
    groups = group_candidates(reviews, k=4)
    if len(groups) != 250 or set(groups) != set(range(250)):
        raise ValueError("Expected 250 indexed paragraphs")
    if not len(reviews) == len(candidates) == len(ordered) == len(parsed) == len(matches) == len(scores):
        raise ValueError("Candidate/detection lengths disagree")
    by_key = {(row["index"], row["candidate"]): row for row in candidates}
    if len(by_key) != len(candidates):
        raise ValueError("Duplicate generated candidate")
    sentence_counts = {}
    for row in reviews:
        position = row["position"]
        for record in (by_key[(row["index"], row["candidate"])], ordered[position]):
            if any(row[field] != record[field] for field in ("index", "candidate", "source", "attack")):
                raise ValueError("Candidate alignment mismatch")
        n = len(parsed[position])
        if len(matches[position]) != n or not all(parsed[position]):
            raise ValueError("AMR/match count mismatch or empty graph")
        green = sum(item["score"] >= MATCH_THRESHOLD for item in matches[position])
        if not math.isclose(z_score(n, green), scores[position], abs_tol=1e-10):
            raise ValueError("Cannot reconstruct official z")
        if not math.isclose(scores[position], row["z_score"], abs_tol=1e-10):
            raise ValueError("Review score mismatch")
        gate = all(row["surface"].values()) and row["review"]["score"] >= 4 and not row["review"]["major_error"]
        if gate != row["quality_valid"]:
            raise ValueError("Saved quality flag disagrees with gate")
        sentence_counts[position] = n

    selected = [
        min([r for r in pool if r["quality_valid"]] or pool,
            key=lambda r: (r["z_score"], r["candidate"]))
        for pool in groups.values()
    ]
    quality_selected = load(base + "quality_aware_selected.json")
    expected = {(r["index"], r["candidate"]) for r in selected if r["quality_valid"]}
    if expected != {(r["index"], r["candidate"]) for r in quality_selected}:
        raise ValueError("Quality-selected file mismatch")
    initial_success = {row["index"] for row in selected if valid(row)}
    retry = load(base + "adaptive_retry/evaluated.json")
    retry_groups = group_candidates(retry, k=4)
    if set(retry_groups) != set(groups) - initial_success:
        raise ValueError("Retry indices are not exactly the failures")
    retry_success = {r["index"] for r in retry if valid(r)}

    inventory = []
    for name, path, unit, status in [
        ("roundtrip_zh", "planE_e1_roundtrip/zh/records.json", "2 API stages per candidate", "pilot"),
        ("roundtrip_de", "planE_e1_roundtrip/de/records.json", "2 API stages per candidate", "pilot"),
        ("joint", "planE_e2_joint250/candidates.json", "1 API call per candidate; K=4", "full_candidate_quality_labels"),
        ("combo_prompt", "planE_e3_combo30/candidates.json", "1 API call simulating a chain; K=4", "not_multistep_execution"),
        ("targeted", "targeted_sentence_controlled30/candidates.json", "1 call per paragraph/strategy", "sentence_level_proxy"),
    ]:
        payload = load("experiments/" + path)
        rows = payload["rows"] if isinstance(payload, dict) else payload
        inventory.append({
            "method": name, "file": "experiments/" + path,
            "rows": len(rows), "paragraphs": len({r["index"] for r in rows}),
            "rows_by_strategy": dict(Counter(r.get("strategy", name) for r in rows)),
            "nonempty_outputs": sum(bool(r["attack"].strip()) for r in rows),
            "budget_from_code": unit, "status": status,
            "full_usage_records": all(bool(r.get("usage")) for r in rows),
        })
    overlap = {}
    for size in (30, 100):
        rows = load(f"experiments/planE_e2_joint{size}/candidates.json")["rows"]
        overlap[str(size)] = {
            "rows": len(rows),
            "same_source_and_attack_as_250": sum(
                all(r[f] == by_key[(r["index"], r["candidate"])][f] for f in ("source", "attack"))
                for r in rows
            ),
        }
    counts = Counter(sentence_counts[r["position"]] for r in selected)
    output = {
        "threshold": THRESHOLD,
        "checks": {"candidate_alignment": True, "z_reconstructed_from_matches": True,
                   "saved_quality_gate_consistent": True, "retry_indices_are_failures": True},
        "counts": {
            "candidates": len(reviews), "raw_low_candidates": sum(r["z_score"] <= THRESHOLD for r in reviews),
            "quality_candidates": sum(r["quality_valid"] for r in reviews),
            "valid_candidates": sum(valid(r) for r in reviews),
            "raw_paragraphs": sum(any(r["z_score"] <= THRESHOLD for r in pool) for pool in groups.values()),
            "quality_paragraphs": len(quality_selected), "valid_paragraphs": len(initial_success),
            "retry_candidates": len(retry), "retry_quality_candidates": sum(r["quality_valid"] for r in retry),
            "retry_new_successes": len(retry_success - initial_success),
            "combined_successes_using_saved_flags": len(initial_success | retry_success),
        },
        "exact_expected_valid_asr": exact_budget(groups),
        "feedback_gain_over_quality_random_k4": feedback_interval(groups),
        "sentence_mechanism": {
            "maximum_z_by_n": {str(n): z_score(n, n) for n in range(1, 7)},
            "selected_sentence_count_histogram": dict(sorted(counts.items())),
            "valid_successes_with_single_amr": sum(valid(r) and sentence_counts[r["position"]] == 1 for r in selected),
            "all_candidates_single_amr": sum(n == 1 for n in sentence_counts.values()),
            "interpretation": "At N=1, even G=N gives z<2.33. A threshold ceiling, not proof of preserved template matches.",
        },
        "inventory": inventory, "nested_pool_overlap": overlap,
        "unverified": [
            "Independent parser-disagreement arm not located in current project artifacts",
            "Raw formal_250x5 baseline directory absent; no fresh baseline or conditional ASR recomputation",
            "Retry evaluator uses within-group AMR index instead of global position; parse validity needs recheck",
            "Semantic review labels reused, not reassessed",
        ],
    }
    for relative in [
        "scripts/run_planE_e2_candidates30.py", "scripts/extend_planE_e2_to250.py",
        "scripts/quality_aware_select_e2_250.py", "scripts/evaluate_adaptive_retry.py",
        "scripts/run_planE_e3_combo30.py", "scripts/run_plane_e1_roundtrip30.py",
        "scripts/generate_targeted_sentence_controlled.py",
    ]:
        hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    output["input_sha256"] = hashes
    destination = args.output or root / "experiments/planE_e2_joint250/feedback_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in ("counts", "exact_expected_valid_asr", "sentence_mechanism")}, indent=2))


if __name__ == "__main__":
    main()
