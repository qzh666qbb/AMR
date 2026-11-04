"""Regression tests for WP2-v6 factorized feature sidecars."""

import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from audit_certified_resegmentation_wp2_v4_correctness import (
    parse_core_output,
)
from certified_resegmentation_wp2_v3 import SparsePairwiseProblem
from certified_resegmentation_wp2_v4 import (
    exact_grouped_sparse_pairwise_certificate,
)
from certified_resegmentation_wp2_v6_features import (
    SEMSTAMP_MODE,
    SIMMARK_DIMENSION,
    SIMMARK_MODE,
    encode_sidecar,
    feature_pair_score,
    serialize_core_problem,
    synthetic_embedding,
    synthetic_region,
    synthetic_relation,
)
from test_certified_resegmentation_wp2_v5 import (
    make_reference as make_external_reference,
    serialize as serialize_external_problem,
)


CORE = (
    Path(__file__).resolve().parents[1]
    / "build"
    / "wp2_v6"
    / "certified_resegmentation_wp2_v6_core"
)


def run_core(payload):
    return subprocess.run(
        [str(CORE)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def make_case(mode, *, all_allowed=False, radius=3):
    token_count = 9
    original = (2, 5, 7)
    original_spans = set(zip(
        (0, *original),
        (*original, token_count),
    ))
    seen = set()
    relation = synthetic_relation()
    features = {}

    def allowed(start, end):
        seen.add((start, end))
        return (
            all_allowed
            or (start, end) in original_spans
            or (start * 7 + end * 11) % 5 != 0
        )

    def feature(start, end):
        key = (start, end)
        if key not in features:
            features[key] = (
                synthetic_region(start, end)
                if mode == SEMSTAMP_MODE
                else synthetic_embedding(start, end)
            )
        return features[key]

    def pair_score(left, middle, right):
        feature(left, middle)
        feature(middle, right)
        return feature_pair_score(
            mode,
            features,
            relation,
            left,
            middle,
            right,
        )

    problem = SparsePairwiseProblem(
        token_count=token_count,
        original_boundaries=original,
        max_boundary_moves=radius,
        span_allowed=allowed,
        pair_scorer=pair_score,
        normalization=3.0,
        bias=0.125,
        threshold=0.25,
    )
    reference = exact_grouped_sparse_pairwise_certificate(problem)
    records = []
    for start, end in sorted(seen):
        is_allowed = allowed(start, end)
        records.append((
            start,
            end,
            is_allowed,
            feature(start, end) if is_allowed else None,
        ))
    return problem, reference, records, relation


def compare_result(test, observed, reference, tolerance):
    test.assertLessEqual(
        abs(observed["original_score"] - reference.original_score),
        tolerance,
    )
    test.assertLessEqual(
        abs(observed["certified_score"] - reference.certified_score),
        tolerance,
    )
    fields = (
        "worst_boundaries",
        "exact_move_histogram",
        "threat_set_size",
        "reachable_state_count",
        "transitions_evaluated",
        "peak_layer_states",
        "candidate_boundary_positions",
        "unique_span_checks",
        "unique_pair_scores",
    )
    for field in fields:
        test.assertEqual(observed[field], getattr(reference, field))


class FactorizedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CORE.is_file():
            raise unittest.SkipTest(f"compiled v6 core is absent: {CORE}")

    def run_case(self, mode):
        problem, reference, records, relation = make_case(mode)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.bin"
            encoded = encode_sidecar(
                mode,
                problem.token_count,
                records,
                relation=relation,
            )
            path.write_bytes(encoded)
            completed = run_core(
                serialize_core_problem(problem, mode, path)
            )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8"),
        )
        observed = parse_core_output(completed.stdout, 1)[0]
        compare_result(
            self,
            observed,
            reference,
            1e-12 if mode == SEMSTAMP_MODE else 1e-10,
        )
        return encoded

    def test_semstamp_factorized_matches_reference(self):
        self.run_case(SEMSTAMP_MODE)

    def test_simmark_factorized_matches_reference(self):
        self.run_case(SIMMARK_MODE)

    def test_sidecar_generation_is_byte_identical(self):
        problem, _reference, records, relation = make_case(
            SIMMARK_MODE
        )
        first = encode_sidecar(
            SIMMARK_MODE,
            problem.token_count,
            records,
            relation=relation,
        )
        second = encode_sidecar(
            SIMMARK_MODE,
            problem.token_count,
            records,
            relation=relation,
        )
        self.assertEqual(first, second)

    def test_v5_external_pair_mode_is_preserved(self):
        problem, reference, scores = make_external_reference()
        rows = [
            (*key, format(value, ".17g"))
            for key, value in sorted(scores.items())
        ]
        completed = run_core(serialize_external_problem(problem, rows))
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8"),
        )
        observed = parse_core_output(completed.stdout, 1)[0]
        compare_result(self, observed, reference, 1e-12)

    def assert_bad_sidecar(
        self,
        problem,
        mode,
        encoded,
        expected_message,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(encoded)
            completed = run_core(
                serialize_core_problem(problem, mode, path)
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(
            expected_message,
            completed.stderr.decode("utf-8"),
        )

    def test_missing_and_extra_span_records_fail_closed(self):
        problem, _reference, records, relation = make_case(
            SEMSTAMP_MODE,
            all_allowed=True,
            radius=0,
        )
        missing = encode_sidecar(
            SEMSTAMP_MODE,
            problem.token_count,
            records[1:],
            relation=relation,
        )
        self.assert_bad_sidecar(
            problem,
            SEMSTAMP_MODE,
            missing,
            "missing external span record",
        )

        observed = {(row[0], row[1]) for row in records}
        extra_span = next(
            (start, end)
            for start in range(problem.token_count)
            for end in range(start + 1, problem.token_count + 1)
            if (start, end) not in observed
        )
        extra_records = sorted([
            *records,
            (
                *extra_span,
                True,
                synthetic_region(*extra_span),
            ),
        ])
        extra = encode_sidecar(
            SEMSTAMP_MODE,
            problem.token_count,
            extra_records,
            relation=relation,
        )
        self.assert_bad_sidecar(
            problem,
            SEMSTAMP_MODE,
            extra,
            "unused external span record",
        )

    def test_bad_identity_and_layout_fail_closed(self):
        problem, _reference, records, relation = make_case(
            SEMSTAMP_MODE,
            all_allowed=True,
            radius=0,
        )
        valid = bytearray(encode_sidecar(
            SEMSTAMP_MODE,
            problem.token_count,
            records,
            relation=relation,
        ))
        mutations = []

        bad_magic = bytearray(valid)
        bad_magic[0] = ord("X")
        mutations.append((bad_magic, "invalid feature sidecar magic"))

        bad_mode = bytearray(valid)
        struct.pack_into("<I", bad_mode, 8, SIMMARK_MODE)
        mutations.append((bad_mode, "feature sidecar identity mismatch"))

        bad_n = bytearray(valid)
        struct.pack_into("<I", bad_n, 12, problem.token_count + 1)
        mutations.append((bad_n, "feature sidecar identity mismatch"))

        bad_dim = bytearray(valid)
        struct.pack_into("<I", bad_dim, 24, 2)
        mutations.append((bad_dim, "invalid feature dimension"))

        bad_relation = bytearray(valid)
        bad_relation[28] = 2
        mutations.append((
            bad_relation,
            "invalid SemStamp relation value",
        ))

        first_record = 28 + 64
        bad_endpoint = bytearray(valid)
        struct.pack_into("<I", bad_endpoint, first_record, 1)
        struct.pack_into("<I", bad_endpoint, first_record + 4, 1)
        mutations.append((bad_endpoint, "invalid feature span record"))

        bad_region = bytearray(valid)
        bad_region[first_record + 9] = 8
        mutations.append((bad_region, "invalid SemStamp region"))

        first = bytes(valid[first_record:first_record + 10])
        duplicate = bytearray(valid)
        count = struct.unpack_from("<Q", duplicate, 16)[0]
        struct.pack_into("<Q", duplicate, 16, count + 1)
        duplicate.extend(first)
        mutations.append((
            duplicate,
            "feature span records are not strictly sorted",
        ))

        unsorted = bytearray(valid)
        first = bytes(unsorted[first_record:first_record + 10])
        second = bytes(unsorted[first_record + 10:first_record + 20])
        unsorted[first_record:first_record + 10] = second
        unsorted[first_record + 10:first_record + 20] = first
        mutations.append((
            unsorted,
            "feature span records are not strictly sorted",
        ))

        mutations.append((valid[:-1], "truncated feature sidecar"))
        mutations.append((
            valid + b"x",
            "trailing feature sidecar bytes",
        ))
        for encoded, message in mutations:
            with self.subTest(message=message):
                self.assert_bad_sidecar(
                    problem,
                    SEMSTAMP_MODE,
                    bytes(encoded),
                    message,
                )

    def test_nonfinite_and_zero_embeddings_fail_closed(self):
        problem, _reference, records, relation = make_case(
            SIMMARK_MODE,
            all_allowed=True,
            radius=0,
        )
        valid = bytearray(encode_sidecar(
            SIMMARK_MODE,
            problem.token_count,
            records,
            relation=relation,
        ))
        first_embedding = 28 + 9
        mutations = []
        for value, label in (
            (math.nan, "nan"),
            (math.inf, "inf"),
        ):
            mutated = bytearray(valid)
            struct.pack_into("<f", mutated, first_embedding, value)
            mutations.append((
                label,
                mutated,
                "non-finite SimMark embedding",
            ))
        zero = bytearray(valid)
        zero[first_embedding:first_embedding + 4 * SIMMARK_DIMENSION] = (
            b"\0" * (4 * SIMMARK_DIMENSION)
        )
        mutations.append((
            "zero",
            zero,
            "zero-norm SimMark embedding",
        ))
        for label, encoded, message in mutations:
            with self.subTest(label=label):
                self.assert_bad_sidecar(
                    problem,
                    SIMMARK_MODE,
                    bytes(encoded),
                    message,
                )


if __name__ == "__main__":
    unittest.main()
