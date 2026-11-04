"""Correctness tests for the compiled SemStamp v3 variable-N core."""

from itertools import combinations
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from run_semstamp_certified_defense_v3_variable_n_profile import (
    read_mode5_codes,
    write_mode5_union_sidecar,
)
from semstamp_certified_defense_v1_variable_n import (
    ExactScore,
    VariableNPairwiseProblem,
    compare_exact_scores,
    exact_variable_n_pairwise_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    ROOT
    / "build/semstamp_v3/semstamp_certified_defense_v3_variable_n_core"
)
V6_CORE = (
    ROOT / "build/wp2_v6/certified_resegmentation_wp2_v6_core"
)


def score_pair(left, middle, right):
    return (
        (left + 3) * 31
        + (middle + 5) * 17
        + (right + 7) * 13
    ) % 97


def span_bits(token_count, allowed):
    width = token_count + 1
    bits = bytearray(b"0" * (width * width))
    for start, end in allowed:
        bits[start * width + end] = ord("1")
    return bits.decode("ascii")


def parse_histogram(value):
    return {
        int(key): int(count)
        for item in value.split(",")
        for key, count in (item.split(":", 1),)
    }


def parse_output(payload):
    fields = payload.decode("ascii").strip().split("\t")
    if len(fields) != 12 or fields[0] != "0":
        raise AssertionError("compiled core returned malformed output")
    return {
        "original_sum": float(fields[1]),
        "minimum_sum": float(fields[2]),
        "threat_set_size": int(fields[3]),
        "insertion_histogram": parse_histogram(fields[4]),
        "witness": tuple(map(int, fields[5].split(","))),
        "reachable_states": int(fields[6]),
        "transitions": int(fields[7]),
        "peak_layer_states": int(fields[8]),
        "candidate_positions": int(fields[9]),
        "unique_spans": int(fields[10]),
        "unique_pairs": int(fields[11]),
    }


def run_core(path, payload):
    completed = subprocess.run(
        [str(path)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed


def v3_payload(token_count, original, target_count, radius, allowed):
    fields = [
        token_count,
        len(original),
        target_count,
        radius,
        2,
        0,
        1,
        0,
        0,
        *original,
        span_bits(token_count, allowed),
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def v6_payload(token_count, original, radius, allowed):
    fields = [
        token_count,
        len(original),
        radius,
        2,
        0,
        1,
        0,
        0,
        *original,
        span_bits(token_count, allowed),
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def sidecar_dump_payload(token_count, original, target_count, radius):
    fields = [
        token_count,
        len(original),
        target_count,
        radius,
        3,
        1,
        1,
        0,
        0,
        *original,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def mode5_payload(
    token_count,
    original,
    target_count,
    radius,
    sidecar_path,
):
    fields = [
        token_count,
        len(original),
        target_count,
        radius,
        5,
        2,
        1,
        0,
        0,
        *original,
        sidecar_path,
    ]
    return ("1\n" + " ".join(map(str, fields)) + "\n").encode("ascii")


def synthetic_relation(previous, current):
    return int(((previous + 1) * 5 + (current + 3) * 3) % 4 == 0)


def synthetic_region(start, end):
    return ((start + 3) * 5 + (end + 7) * 3) % 8


def semstamp_hit(left, middle, right):
    return synthetic_relation(
        synthetic_region(left, middle),
        synthetic_region(middle, right),
    )


def always_allowed(start, end):
    return start < end


def brute_target(
    token_count,
    original,
    target_count,
    radius,
    allowed,
    scorer=score_pair,
):
    original_set = set(original)
    histogram = {index: 0 for index in range(4)}
    candidates = []
    for boundaries in combinations(range(1, token_count), target_count):
        selected = set(boundaries)
        insertions = len(selected - original_set)
        deletions = len(original_set - selected)
        if max(insertions, deletions) > radius:
            continue
        endpoints = (0, *boundaries, token_count)
        spans = tuple(zip(endpoints, endpoints[1:]))
        if any(span not in allowed for span in spans):
            continue
        total = sum(
            scorer(
                endpoints[index - 1],
                endpoints[index],
                endpoints[index + 1],
            )
            for index in range(1, len(endpoints) - 1)
        )
        histogram[insertions] += 1
        candidates.append((total, insertions, boundaries))
    if not candidates:
        return None
    candidates.sort()
    best = candidates[0]
    original_endpoints = (0, *original, token_count)
    original_sum = sum(
        scorer(
            original_endpoints[index - 1],
            original_endpoints[index],
            original_endpoints[index + 1],
        )
        for index in range(1, len(original_endpoints) - 1)
    )
    return {
        "original_sum": float(original_sum),
        "minimum_sum": float(best[0]),
        "threat_set_size": len(candidates),
        "insertion_histogram": histogram,
        "witness": best[2],
    }


def random_case(rng):
    token_count = rng.randint(3, 10)
    original_count = rng.randint(1, token_count - 1)
    original = tuple(sorted(
        rng.sample(range(1, token_count), original_count)
    ))
    radius = rng.randint(0, min(3, original_count))
    lower = max(1, original_count - radius)
    upper = min(token_count - 1, original_count + radius)
    target_count = rng.randint(lower, upper)
    allowed = {
        (start, end)
        for start in range(token_count)
        for end in range(start + 1, token_count + 1)
        if rng.random() < 0.76
    }
    endpoints = (0, *original, token_count)
    allowed.update(zip(endpoints, endpoints[1:]))
    return token_count, original, target_count, radius, allowed


class VariableNCompiledCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CORE.is_file():
            raise unittest.SkipTest(f"compiled v3 core is absent: {CORE}")

    def assert_case(self, case):
        token_count, original, target_count, radius, allowed = case
        expected = brute_target(
            token_count,
            original,
            target_count,
            radius,
            allowed,
        )
        completed = run_core(
            CORE,
            v3_payload(
                token_count,
                original,
                target_count,
                radius,
                allowed,
            ),
        )
        if expected is None:
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"empty threat set", completed.stderr)
            return
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        observed = parse_output(completed.stdout)
        for key in (
            "original_sum",
            "minimum_sum",
            "threat_set_size",
            "insertion_histogram",
            "witness",
        ):
            self.assertEqual(observed[key], expected[key], (case, key))

    def test_systematic_small_cases(self):
        cases = 0
        for token_count in range(3, 8):
            internal = range(1, token_count)
            all_allowed = {
                (start, end)
                for start in range(token_count)
                for end in range(start + 1, token_count + 1)
            }
            for original_count in range(1, token_count):
                for original in combinations(internal, original_count):
                    for radius in range(min(3, original_count) + 1):
                        lower = max(1, original_count - radius)
                        upper = min(
                            token_count - 1,
                            original_count + radius,
                        )
                        for target_count in range(lower, upper + 1):
                            self.assert_case((
                                token_count,
                                original,
                                target_count,
                                radius,
                                all_allowed,
                            ))
                            cases += 1
        self.assertGreaterEqual(cases, 500)

    def test_random_legality_cases(self):
        rng = random.Random(20261008)
        for _ in range(1000):
            self.assert_case(random_case(rng))

    def test_fixed_n_reduces_to_v6(self):
        if not V6_CORE.is_file():
            self.skipTest(f"compiled v6 core is absent: {V6_CORE}")
        rng = random.Random(20261009)
        compared = 0
        while compared < 500:
            case = random_case(rng)
            token_count = case[0]
            original = case[1]
            radius = case[3]
            allowed = case[4]
            target_count = len(original)
            current = run_core(
                CORE,
                v3_payload(
                    token_count,
                    original,
                    target_count,
                    radius,
                    allowed,
                ),
            )
            legacy = run_core(
                V6_CORE,
                v6_payload(token_count, original, radius, allowed),
            )
            self.assertEqual(current.returncode, legacy.returncode)
            if current.returncode != 0:
                continue
            current_row = parse_output(current.stdout)
            legacy_row = parse_output(legacy.stdout)
            self.assertEqual(current_row, legacy_row)
            compared += 1

    def test_target_outside_radius_fails_closed(self):
        payload = v3_payload(
            9,
            (2, 5, 7),
            5,
            1,
            {
                (start, end)
                for start in range(9)
                for end in range(start + 1, 10)
            },
        )
        completed = run_core(CORE, payload)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            b"target edge count is outside variable-N radius",
            completed.stderr,
        )

    def test_semstamp_mode5_cross_m_matches_variable_n_oracle(self):
        rng = random.Random(20261010)
        for _ in range(40):
            token_count = rng.randint(4, 10)
            original_count = rng.randint(1, token_count - 1)
            original = tuple(sorted(
                rng.sample(range(1, token_count), original_count)
            ))
            radius = rng.randint(0, min(3, original_count))
            allowed = {
                (start, end)
                for start in range(token_count)
                for end in range(start + 1, token_count + 1)
            }
            target_rows = []
            with tempfile.TemporaryDirectory() as directory:
                for target_count in range(
                    max(1, original_count - radius),
                    min(token_count - 1, original_count + radius) + 1,
                ):
                    sidecar = Path(directory) / (
                        f"features_m{target_count}.bin"
                    )
                    dumped = subprocess.run(
                        [
                            str(CORE),
                            "--dump-synthetic-sidecar",
                            str(sidecar),
                            "5",
                        ],
                        input=sidecar_dump_payload(
                            token_count,
                            original,
                            target_count,
                            radius,
                        ),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertEqual(
                        dumped.returncode,
                        0,
                        dumped.stderr.decode("utf-8", errors="replace"),
                    )
                    observed_process = run_core(
                        CORE,
                        mode5_payload(
                            token_count,
                            original,
                            target_count,
                            radius,
                            sidecar,
                        ),
                    )
                    self.assertEqual(
                        observed_process.returncode,
                        0,
                        observed_process.stderr.decode(
                            "utf-8", errors="replace"
                        ),
                    )
                    observed = parse_output(observed_process.stdout)
                    expected = brute_target(
                        token_count,
                        original,
                        target_count,
                        radius,
                        allowed,
                        scorer=semstamp_hit,
                    )
                    self.assertIsNotNone(expected)
                    for key in (
                        "original_sum",
                        "minimum_sum",
                        "threat_set_size",
                        "insertion_histogram",
                        "witness",
                    ):
                        self.assertEqual(
                            observed[key],
                            expected[key],
                            (
                                token_count,
                                original,
                                radius,
                                target_count,
                                key,
                            ),
                        )
                    target_rows.append((target_count, observed))

            reference = exact_variable_n_pairwise_certificate(
                VariableNPairwiseProblem(
                    token_count=token_count,
                    original_boundaries=original,
                    radius=radius,
                    span_allowed=always_allowed,
                    pair_hit=semstamp_hit,
                )
            )
            best = None
            threat_set_size = 0
            operation_histogram = {}
            for target_count, row in target_rows:
                threat_set_size += row["threat_set_size"]
                for insertions, count in row[
                    "insertion_histogram"
                ].items():
                    if not count:
                        continue
                    deletions = original_count - target_count + insertions
                    operation_histogram[(insertions, deletions)] = count
                candidate = {
                    "score": ExactScore(
                        int(row["minimum_sum"]),
                        target_count,
                    ),
                    "boundaries": row["witness"],
                    "insertions": len(
                        set(row["witness"]) - set(original)
                    ),
                    "deletions": len(
                        set(original) - set(row["witness"])
                    ),
                }
                candidate["operation_cost"] = max(
                    candidate["insertions"],
                    candidate["deletions"],
                )
                if best is None:
                    best = candidate
                    continue
                comparison = compare_exact_scores(
                    candidate["score"], best["score"]
                )
                candidate_key = (
                    candidate["operation_cost"],
                    candidate["insertions"],
                    candidate["deletions"],
                    target_count,
                    candidate["boundaries"],
                )
                best_key = (
                    best["operation_cost"],
                    best["insertions"],
                    best["deletions"],
                    best["score"].edge_count,
                    best["boundaries"],
                )
                if comparison < 0 or (
                    comparison == 0 and candidate_key < best_key
                ):
                    best = candidate

            self.assertEqual(best["score"], reference.exact_score)
            self.assertEqual(
                best["boundaries"], reference.worst_boundaries
            )
            self.assertEqual(threat_set_size, reference.threat_set_size)
            self.assertEqual(
                operation_histogram,
                {
                    (insertions, deletions): count
                    for insertions, deletions, count
                    in reference.histogram_by_insertions_deletions
                },
            )

    def test_mode5_radius_union_requires_explicit_superset_flag(self):
        token_count = 9
        original = (2, 5, 7)
        radius = 1
        targets = (2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            exact_outputs = {}
            code_sets = []
            for target_count in targets:
                sidecar = directory / f"exact-m{target_count}.bin"
                dumped = subprocess.run(
                    [
                        str(CORE),
                        "--dump-synthetic-sidecar",
                        str(sidecar),
                        "5",
                    ],
                    input=sidecar_dump_payload(
                        token_count,
                        original,
                        target_count,
                        radius,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(dumped.returncode, 0)
                code_sets.append(read_mode5_codes(sidecar, token_count))
                exact = run_core(
                    CORE,
                    mode5_payload(
                        token_count,
                        original,
                        target_count,
                        radius,
                        sidecar,
                    ),
                )
                self.assertEqual(exact.returncode, 0)
                exact_outputs[target_count] = exact.stdout

            union_codes = sorted({
                int(code)
                for codes in code_sets
                for code in codes
            })
            union_sidecar = directory / "union.bin"
            write_mode5_union_sidecar(
                union_sidecar,
                token_count,
                union_codes,
                [True] * len(union_codes),
            )
            strict = run_core(
                CORE,
                mode5_payload(
                    token_count,
                    original,
                    targets[0],
                    radius,
                    union_sidecar,
                ),
            )
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn(b"unused external span record", strict.stderr)

            for target_count in targets:
                relaxed = subprocess.run(
                    [str(CORE), "--allow-unused-feature-records"],
                    input=mode5_payload(
                        token_count,
                        original,
                        target_count,
                        radius,
                        union_sidecar,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(
                    relaxed.returncode,
                    0,
                    relaxed.stderr.decode("utf-8", errors="replace"),
                )
                self.assertEqual(
                    relaxed.stdout,
                    exact_outputs[target_count],
                )


if __name__ == "__main__":
    unittest.main()
