"""Tests for the WP2-v18 B1 runner and independent auditor."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_certified_resegmentation_wp2_v18_b1 as audit
from certified_resegmentation_wp2_v3 import canonicalize_structured
import run_certified_resegmentation_wp2_v18_b1 as runner


class B1Tests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "plan_index": 0,
            "cohort": "positive",
            "cohort_index": 0,
            "raw_text_sha256": "a" * 64,
            "unit_count": 3,
        }
        self.workload = {
            "schema_version": 1,
            "protocol_sha256": runner.PROTOCOL_SHA256,
            "plan_index": 0,
            "cohort": "positive",
            "cohort_index": 0,
            "raw_text_sha256": "a" * 64,
            "tape_sha256": "b" * 64,
            "original_units_sha256": "c" * 64,
            "atom_count": 9,
            "unit_count": 3,
            "edge_count": 2,
            "original_boundaries": [2, 6],
            "radii": {},
            "resource_counts": {
                "maximum_legal_input_tokens": 12,
                "maximum_sidecar_bytes": 100,
            },
            "estimate": {},
            "contains_raw_text": False,
        }
        for radius in runner.RADII:
            effective = min(radius, 2)
            self.workload["radii"][str(radius)] = {
                "requested_radius": radius,
                "effective_radius": effective,
                "all_allowed_unique_spans": 10 + effective,
                "all_allowed_unique_pairs": 20 + effective,
                "unique_span_count": 10 + effective,
                "legal_span_count": 10 + effective,
                "unique_pair_count": 20 + effective,
                "candidate_boundary_positions": 8,
                "reachable_state_count": 9,
                "transitions_evaluated": 10,
                "peak_layer_states": 4,
                "threat_set_size": 1 + effective,
            }
        self.b0 = {
            "hit_count": 2,
            "direct_score": runner.v10.score_from_raw(2, 2),
            "regions_sha256": "e" * 64,
        }

    def certificate_row(self):
        certificates = {}
        hits = (2, 1, 0, 0)
        for radius, hit_count in enumerate(hits):
            effective = min(radius, 2)
            cheap_hits = max(0, 2 - 5 * effective)
            value = {
                "requested_radius": radius,
                "effective_radius": effective,
                "exact_min_hits": hit_count,
                "exact_score": runner.v10.score_from_raw(hit_count, 2),
                "cheap_min_hits": cheap_hits,
                "cheap_score": runner.v10.score_from_raw(cheap_hits, 2),
                "threat_set_size": 1 + effective,
                "reachable_state_count": 9,
                "unique_span_count": 10 + effective,
                "unique_pair_count": 20 + effective,
                "witness_boundaries": [2, 6],
                "witness_score": runner.v10.score_from_raw(hit_count, 2),
                "feature_cache_sha256": "d" * 64,
            }
            value["certificate_sha256"] = runner.certificate_digest(value)
            certificates[str(radius)] = value
        return {
            "schema_version": 1,
            "protocol_sha256": runner.PROTOCOL_SHA256,
            "plan_index": 0,
            "cohort": "positive",
            "cohort_index": 0,
            "raw_text_sha256": "a" * 64,
            "tape_sha256": "b" * 64,
            "original_units_sha256": "c" * 64,
            "atom_count": 9,
            "unit_count": 3,
            "edge_count": 2,
            "original_hits": 2,
            "original_score": self.b0["direct_score"],
            "original_regions_sha256": "e" * 64,
            "certificates": certificates,
            "contains_raw_text": False,
        }

    def test_protocol_digest_is_frozen(self):
        protocol = (
            Path(__file__).resolve().parents[1]
            / "experiments/certified_resegmentation_wp2_v18_b1_exact_certificate1"
            / "PROTOCOL.md"
        )
        self.assertEqual(
            runner.sha256_file(protocol), runner.PROTOCOL_SHA256
        )
        self.assertEqual(runner.PROTOCOL_SHA256, audit.PROTOCOL_SHA256)

    def test_renderer_restores_original_units(self):
        document = canonicalize_structured(
            ["One  two.", " Three four! ", "Five six?"]
        )
        checks = runner.renderer_checks(document)
        self.assertTrue(checks["original_partition_exact"])
        self.assertEqual(
            document.render_partition(document.original_boundaries),
            document.original_units,
        )

    def test_adjacent_profiles_are_conservative_candidates(self):
        profiles = [{"atom_count": value} for value in (10, 20, 30)]
        self.assertEqual(
            runner.adjacent_profile_indices(profiles, 5), (0, 1)
        )
        self.assertEqual(
            runner.adjacent_profile_indices(profiles, 20), (0, 1, 2)
        )
        self.assertEqual(
            runner.adjacent_profile_indices(profiles, 40), (1, 2)
        )

    def test_paired_counts_enforce_subset_direction(self):
        self.assertEqual(
            runner.paired_counts([2.0, 2.0, 0.0], [2.0, 0.0, 0.0], 1.0),
            {
                "both_certify": 1,
                "exact_only": 1,
                "cheap_only": 0,
                "neither": 1,
            },
        )

    def test_valid_workload_row_passes_both_validators(self):
        runner.validate_workload_row(self.workload, 0, self.plan)
        audit.validate_workload_row(self.workload, 0, self.plan)

    def test_certificate_row_passes_both_validators(self):
        row = self.certificate_row()
        runner.validate_certificate_row(
            row, 0, self.plan, self.workload, self.b0
        )
        audit.validate_certificate_row(
            row, 0, self.plan, self.b0, self.workload
        )

    def test_certificate_tamper_is_rejected(self):
        row = self.certificate_row()
        row["certificates"]["1"]["exact_min_hits"] = 2
        with self.assertRaises(ValueError):
            runner.validate_certificate_row(
                row, 0, self.plan, self.workload, self.b0
            )
        with self.assertRaises(ValueError):
            audit.validate_certificate_row(
                row, 0, self.plan, self.b0, self.workload
            )

    def test_nonmonotone_certificate_is_rejected(self):
        row = self.certificate_row()
        changed = row["certificates"]["2"]
        changed["exact_min_hits"] = 2
        changed["exact_score"] = runner.v10.score_from_raw(2, 2)
        changed["witness_score"] = changed["exact_score"]
        changed["certificate_sha256"] = runner.certificate_digest(changed)
        with self.assertRaises(ValueError):
            runner.validate_certificate_row(
                row, 0, self.plan, self.workload, self.b0
            )

    def test_cheap_only_is_visible(self):
        counts = audit.pair([0.0], [2.0], 1.0)
        self.assertEqual(counts["cheap_only"], 1)

    def test_public_raw_content_keys_are_rejected(self):
        row = self.certificate_row()
        row["generated_text"] = "not public"
        self.assertFalse(audit.no_forbidden_keys(row))

    def test_certificate_digest_ignores_stored_digest_only(self):
        row = self.certificate_row()
        value = row["certificates"]["1"]
        expected = value["certificate_sha256"]
        value["certificate_sha256"] = "f" * 64
        self.assertEqual(runner.certificate_digest(value), expected)
        value["exact_min_hits"] = 0
        self.assertNotEqual(runner.certificate_digest(value), expected)

    def test_workload_native_legality_ordering_rejects_tamper(self):
        row = copy.deepcopy(self.workload)
        row["radii"]["2"]["legal_span_count"] = 999
        with self.assertRaises(ValueError):
            runner.validate_workload_row(row, 0, self.plan)
        with self.assertRaises(ValueError):
            audit.validate_workload_row(row, 0, self.plan)


if __name__ == "__main__":
    unittest.main()
