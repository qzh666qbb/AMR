"""CPU-only combinatorial audit; no tokenizer, detector or model dependencies."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def pair_spans(atom_count, boundaries):
    """Represent each adjacent pair by its three endpoints on a common tape."""
    integer(atom_count, 2, "atom_count")
    boundaries = tuple(boundaries)
    if (
        not boundaries
        or any(type(b) is not int or not 0 < b < atom_count for b in boundaries)
        or tuple(sorted(set(boundaries))) != boundaries
    ):
        raise ValueError("boundaries must be sorted distinct internal positions")
    points = (0,) + boundaries + (atom_count,)
    return frozenset(zip(points, points[1:], points[2:]))


def relocation_distance(left, right):
    left, right = frozenset(left), frozenset(right)
    if len(left) != len(right):
        raise ValueError("relocation distance requires equal boundary counts")
    return len(left - right)


def local_hit_lower_bound(original_hits, edge_count, radius):
    integer(edge_count, 1, "edge_count")
    integer(original_hits, 0, "original_hits")
    integer(radius, 0, "radius")
    if original_hits > edge_count:
        raise ValueError("original_hits exceeds edge_count")
    return max(0, original_hits - 5 * radius)


def local_score_lower_bound(original_hits, edge_count, radius):
    hits = local_hit_lower_bound(original_hits, edge_count, radius)
    return (hits - 0.25 * edge_count) / math.sqrt(edge_count * 0.25 * 0.75)


def toy_hit(pair):
    """A deterministic local binary function, unrelated to SemStamp's model."""
    return hashlib.sha256(canonical_bytes(pair)).digest()[0] % 2


def check_pair(atom_count, left, right):
    left_pairs = pair_spans(atom_count, left)
    right_pairs = pair_spans(atom_count, right)
    radius = relocation_distance(left, right)
    lost = len(left_pairs - right_pairs)
    if lost > min(len(left_pairs), 5 * radius):
        raise AssertionError("5k structural bound violated")
    left_hits = sum(map(toy_hit, left_pairs))
    right_hits = sum(map(toy_hit, right_pairs))
    if right_hits < local_hit_lower_bound(left_hits, len(left_pairs), radius):
        raise AssertionError("binary lower bound violated")
    return [atom_count, list(left), list(right), radius, lost, left_hits, right_hits]


def recentering_counterexample():
    atom_count = 9
    partitions = [
        b for b in itertools.combinations(range(1, atom_count), 2)
        if all(2 <= v - u <= 4 for u, v in zip((0,) + b, b + (atom_count,)))
    ]
    low = (2, 5)
    original, attacked = (3, 6), (2, 6)

    def score(partition):
        # First pair contributes 0 only at low, otherwise 1; last pair is 0.
        return sum(
            int(a == 0 and (b, c) != low)
            for a, b, c in pair_spans(atom_count, partition)
        )

    def certificate(center, radius):
        return min(
            score(p) for p in partitions
            if relocation_distance(center, p) <= radius
        )

    if (
        relocation_distance(original, attacked) != 1
        or certificate(original, 1) != 1
        or certificate(attacked, 1) != 0
        or certificate(original, 2) != 0
    ):
        raise AssertionError("counterexample changed")
    # Verify the triangle-inequality implication on this common legal space.
    triangle_checks = 0
    for center in partitions:
        for other in partitions:
            for radius in (0, 1, 2):
                if relocation_distance(center, other) <= radius:
                    if certificate(other, radius) < certificate(center, 2 * radius):
                        raise AssertionError("recentered 2k bound violated")
                    triangle_checks += 1
    return {
        "atom_count": atom_count,
        "unit_count": 3,
        "legal_unit_length": [2, 4],
        "legal_partitions": partitions,
        "original": original,
        "attacked": attacked,
        "low_score_partition": low,
        "threshold": 0.5,
        "c1_original": certificate(original, 1),
        "c1_attacked": certificate(attacked, 1),
        "c2_original": certificate(original, 2),
        "common_space_triangle_checks": triangle_checks,
        "interpretation": "C1 > threshold does not certify the recentered C1 classifier",
    }


def run_audit():
    digest = hashlib.sha256()
    exhaustive_count = 0
    for atom_count in range(3, 11):
        for units in range(2, min(atom_count, 6) + 1):
            partitions = list(itertools.combinations(range(1, atom_count), units - 1))
            for left, right in itertools.combinations_with_replacement(partitions, 2):
                if relocation_distance(left, right) > 3:
                    continue
                digest.update(canonical_bytes(check_pair(atom_count, left, right)) + b"\n")
                exhaustive_count += 1
    rng = random.Random(20260927)
    random_count = 2000
    for _ in range(random_count):
        atom_count = rng.randint(12, 120)
        units = rng.randint(2, min(atom_count, 30))
        left = tuple(sorted(rng.sample(range(1, atom_count), units - 1)))
        radius = rng.randint(0, min(3, units - 1, atom_count - units))
        removed = set(rng.sample(left, radius))
        added = rng.sample(sorted(set(range(1, atom_count)) - set(left)), radius)
        right = tuple(sorted((set(left) - removed) | set(added)))
        digest.update(canonical_bytes(check_pair(atom_count, left, right)) + b"\n")
    tight = []
    for radius in (1, 2, 3):
        atom_count = 12 * radius
        left = tuple(range(2, atom_count, 2))
        right = set(left)
        for block in range(radius):
            right.remove(12 * block + 4)
            right.add(12 * block + 9)
        row = check_pair(atom_count, left, tuple(sorted(right)))
        if row[3] != radius or row[4] != 5 * radius:
            raise AssertionError("structural tightness witness changed")
        tight.append({
            "radius": radius, "atom_count": atom_count,
            "original": left, "alternative": sorted(right),
            "lost_original_pairs": row[4],
        })
    return {
        "schema_version": 1,
        "scope": "offline_combinatorics_and_synthetic_local_scores_only",
        "status": "pass",
        "exhaustive": {
            "atom_counts": [3, 10], "max_units": 6, "max_radius": 3,
            "unordered_partition_pairs_including_self": exhaustive_count,
        },
        "random": {"seed": 20260927, "pairs": random_count, "max_atoms": 120},
        "enumerated_evidence_sha256": digest.hexdigest(),
        "structural_tightness": tight,
        "recentered_counterexample": recentering_counterexample(),
        "attestation": {
            "model_imports": 0, "detector_calls": 0, "real_certificate_calls": 0,
            "sealed_inputs_read": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
