"""Offline tests for the model-free WP2 adapter and cohort contract."""

import hashlib
import math
import unittest

from certified_resegmentation import (
    brute_force_pairwise_certificate,
    exact_pairwise_certificate,
)
from certified_resegmentation_wp2_adapter import (
    build_document_pair_scores,
    build_document_pairwise_problem,
    canonicalize,
    cohort_sort_hash,
    hash_raw_value,
    make_source_record,
    split_null_cohort,
)


class CanonicalAdapterTests(unittest.TestCase):
    def test_string_is_split_exactly_once_and_p0_round_trips(self):
        text = "Alpha\tbeta. Gamma \u4e16\u754c!"
        calls = []

        def splitter(value):
            calls.append(value)
            return ["Alpha\tbeta.", "Gamma \u4e16\u754c!"]

        document = canonicalize(text, splitter)
        self.assertEqual(calls, [text])
        self.assertEqual(document.raw_field_type, "string")
        self.assertEqual(
            document.raw_text_sha256,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            document.canonical_tape,
            "Alpha\tbeta. Gamma \u4e16\u754c!",
        )
        self.assertEqual(document.atom_count, 4)
        self.assertEqual(document.original_boundaries, (2,))
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            document.original_units,
        )

    def test_list_bypasses_splitter_and_has_a_type_domain(self):
        units = ["One two.", "Three four."]

        def forbidden_splitter(_):
            raise AssertionError("list fields must bypass the splitter")

        document = canonicalize(units, forbidden_splitter)
        self.assertEqual(document.raw_field_type, "list")
        self.assertEqual(document.original_units, tuple(units))
        self.assertNotEqual(hash_raw_value(units), hash_raw_value('["One two."]'))
        self.assertEqual(document.render_partition((2,)), tuple(units))

    def test_arbitrary_partition_preserves_tape_substrings(self):
        document = canonicalize(
            ["Alpha\tbeta.", "Gamma \u4e16\u754c!", "Last unit."]
        )
        self.assertEqual(
            document.render_partition((1, 3)),
            ("Alpha", "beta. Gamma", "\u4e16\u754c! Last unit."),
        )
        self.assertEqual(document.render_unit(1, 3), "beta. Gamma")

    def test_system_preprocessing_is_exact(self):
        document = canonicalize(["aa bb", "cc dd", "ee ff"])
        self.assertEqual(
            document.preprocess_partition("semstamp"),
            ("aa bb", "cc dd", "ee ff"),
        )
        self.assertEqual(
            document.preprocess_partition("simmark"),
            ("aa bb", " cc dd", " ee ff"),
        )
        self.assertEqual(
            document.preprocess_partition("simmark", (1, 3)),
            ("aa", " bb cc", " dd ee ff"),
        )

    def test_invalid_splitter_outputs_fail_closed(self):
        invalid = (
            lambda _: ("aa", "bb"),
            lambda _: [],
            lambda _: [" aa", "bb"],
            lambda _: ["aa", "bb "],
            lambda _: ["aa", ""],
            lambda _: ["aa", 3],
        )
        for splitter in invalid:
            with self.subTest(splitter=splitter):
                with self.assertRaises((TypeError, ValueError)):
                    canonicalize("raw", splitter)
        with self.assertRaises(ValueError):
            canonicalize("raw")
        with self.assertRaises(TypeError):
            canonicalize(("aa", "bb"))

    def test_invalid_boundaries_and_systems_fail_closed(self):
        document = canonicalize(["aa bb", "cc dd"])
        for boundaries in ((0,), (4,), (2, 2), (3, 2), (True,)):
            with self.subTest(boundaries=boundaries):
                with self.assertRaises((TypeError, ValueError)):
                    document.render_partition(boundaries)
        with self.assertRaises(ValueError):
            document.preprocess_partition("SemStamp")


class CohortSplitTests(unittest.TestCase):
    def make_record(self, row_index, text, system_id="semstamp"):
        return make_source_record(
            system_id,
            "data-revision",
            row_index,
            text,
            source_repository="example/data",
            relative_path="rows.arrow",
        )

    def test_dedup_overlap_and_stable_split(self):
        unique_null = [
            self.make_record(index, f"null-{index}") for index in range(10)
        ]
        duplicate = self.make_record(99, "null-0")
        null_input = unique_null + [duplicate]
        positive = [
            self.make_record(100, "positive-only"),
            self.make_record(101, "null-3"),
        ]

        forward = split_null_cohort(
            "semstamp", "data-revision", null_input, positive
        )
        reverse = split_null_cohort(
            "semstamp",
            "data-revision",
            reversed(null_input),
            reversed(positive),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward.detector_fit), 2)
        self.assertEqual(len(forward.certificate_calibration), 4)
        self.assertEqual(len(forward.independent_test), 4)
        self.assertEqual(forward.attrition.null_input_count, 11)
        self.assertEqual(forward.attrition.null_unique_count, 10)
        self.assertEqual(forward.attrition.null_duplicate_count, 1)
        self.assertEqual(forward.attrition.positive_overlap_count, 1)
        self.assertEqual(forward.attrition.positive_retained_count, 1)
        self.assertEqual(
            forward.positive_overlap_excluded[0].raw_text_sha256,
            hash_raw_value("null-3"),
        )
        self.assertEqual(
            forward.positive_retained[0].raw_text_sha256,
            hash_raw_value("positive-only"),
        )

        all_null = (
            forward.detector_fit
            + forward.certificate_calibration
            + forward.independent_test
        )
        self.assertEqual(
            all_null,
            tuple(sorted(
                unique_null,
                key=lambda record: (
                    cohort_sort_hash(record),
                    record.row_index,
                    (
                        record.row_index,
                        record.source_repository,
                        record.relative_path,
                        record.raw_field_type,
                        record.raw_text_sha256,
                    ),
                ),
            )),
        )

    def test_rounding_remainder_all_goes_to_test(self):
        records = [self.make_record(index, str(index)) for index in range(7)]
        split = split_null_cohort(
            "semstamp", "data-revision", records, []
        )
        self.assertEqual(len(split.detector_fit), 1)
        self.assertEqual(len(split.certificate_calibration), 2)
        self.assertEqual(len(split.independent_test), 4)

    def test_invalid_identity_and_fractions_fail_closed(self):
        record = self.make_record(0, "null")
        invalid_fractions = (
            (-0.1, 0.4),
            (0.2, 1.1),
            (0.7, 0.4),
            (float("nan"), 0.4),
            (True, 0.4),
        )
        for fit, calibration in invalid_fractions:
            with self.subTest(fit=fit, calibration=calibration):
                with self.assertRaises(ValueError):
                    split_null_cohort(
                        "semstamp",
                        "data-revision",
                        [record],
                        [],
                        fit,
                        calibration,
                    )
        with self.assertRaises(ValueError):
            split_null_cohort("unknown", "data-revision", [], [])
        mismatched = self.make_record(1, "other", system_id="simmark")
        with self.assertRaises(ValueError):
            split_null_cohort(
                "semstamp", "data-revision", [mismatched], []
            )
        with self.assertRaises(ValueError):
            make_source_record("semstamp", "data-revision", -1, "raw")


class PairwiseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.document = canonicalize(["aa bb", "cc dd", "ee ff"])

    @staticmethod
    def scorer(previous, current):
        payload = previous + "\0" + current
        return float(
            sum((index + 1) * ord(character)
                for index, character in enumerate(payload))
            % 41
            - 20
        )

    def make_problem(self, system_id, radius):
        return build_document_pairwise_problem(
            self.document,
            system_id=system_id,
            min_unit_atoms=1,
            max_unit_atoms=3,
            max_boundary_moves=radius,
            scorer=self.scorer,
            threshold=0.0,
        )

    def test_adapter_scores_feed_exact_dp_and_independent_oracle(self):
        for system_id in ("semstamp", "simmark"):
            with self.subTest(system_id=system_id):
                problem = self.make_problem(system_id, radius=1)
                dynamic = exact_pairwise_certificate(problem)
                oracle = brute_force_pairwise_certificate(problem)
                self.assertTrue(math.isclose(
                    dynamic.certified_score,
                    oracle.certified_score,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ))
                self.assertEqual(
                    dynamic.exact_move_histogram,
                    oracle.exact_move_histogram,
                )
                self.assertEqual(
                    dynamic.threat_set_size,
                    oracle.threat_set_size,
                )
                rendered = self.document.preprocess_partition(
                    system_id, dynamic.worst_boundaries
                )
                witness_score = sum(
                    self.scorer(rendered[index - 1], rendered[index])
                    for index in range(1, len(rendered))
                ) / (len(rendered) - 1)
                self.assertTrue(math.isclose(
                    witness_score,
                    dynamic.certified_score,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ))

    def test_zero_radius_equals_direct_p0_score(self):
        for system_id in ("semstamp", "simmark"):
            with self.subTest(system_id=system_id):
                result = exact_pairwise_certificate(
                    self.make_problem(system_id, radius=0)
                )
                units = self.document.preprocess_partition(system_id)
                expected = sum(
                    self.scorer(units[index - 1], units[index])
                    for index in range(1, len(units))
                ) / (len(units) - 1)
                self.assertEqual(result.worst_boundaries, (2, 4))
                self.assertEqual(result.threat_set_size, 1)
                self.assertTrue(math.isclose(
                    result.original_score,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ))
                self.assertEqual(result.certified_score, result.original_score)

    def test_pair_preprocessing_matches_score_key(self):
        semstamp = build_document_pair_scores(
            self.document, "semstamp", 1, 3, self.scorer
        )
        simmark = build_document_pair_scores(
            self.document, "simmark", 1, 3, self.scorer
        )
        self.assertEqual(
            semstamp[(0, 2, 4)],
            self.scorer("aa bb", "cc dd"),
        )
        self.assertEqual(
            simmark[(0, 2, 4)],
            self.scorer("aa bb", " cc dd"),
        )
        self.assertEqual(
            simmark[(2, 4, 6)],
            self.scorer(" cc dd", " ee ff"),
        )

    def test_scorer_failure_and_nonfinite_values_fail_closed(self):
        def failed(*_):
            raise RuntimeError("synthetic scorer failure")

        with self.assertRaises(RuntimeError):
            build_document_pair_scores(
                self.document, "semstamp", 1, 3, failed
            )
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_document_pair_scores(
                        self.document,
                        "semstamp",
                        1,
                        3,
                        lambda *_: value,
                    )
        with self.assertRaises(TypeError):
            build_document_pair_scores(
                self.document, "semstamp", 1, 3, None
            )


if __name__ == "__main__":
    unittest.main()
