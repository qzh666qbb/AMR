"""Offline regression tests: synthetic structures and already sealed history."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_certified_resegmentation_local_bound as bound
import certified_resegmentation_wp2_v13_b0_support as b0


ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


def synthetic_relation():
    return bytes(int(b in (a, (a + 3) % 8)) for a in range(8) for b in range(8))


def manifests():
    sources = []
    for index in range(3500):
        cohort = (
            "calibration_null" if index < 1000
            else "independent_test_null" if index < 2000 else "generation_primary"
        )
        sources.append({
            "cohort": cohort, "cohort_index": index % 1000,
            "raw_text_sha256": identity(index),
            "semstamp_unit_count": 4, "semstamp_max_native_tokens": 12,
            "contains_raw_text": False,
        })
    positives = [{
        "positive_index": i, "output_raw_text_sha256": identity(i + 3500),
        "unit_count": 4, "max_native_tokens": 12, "contains_raw_text": False,
    } for i in range(1000)]
    return sources, positives


def synthetic_h0():
    return {
        "schema_version": 1, "stage": "WP2-v13-H0-identity-freeze", "status": "go",
        "scope": "identity_only_no_detector_scores", "contains_raw_text": False,
        "runner_sha256": b0.H0_RUNNER_SHA256,
        "decision": {
            "h0_identity_freeze": "go",
            "b0_detector_health": "authorized_to_freeze_protocol",
            "b1_exact_certificate": "stop",
        },
        "resource_gate": {"below_0_80": True},
        "counts": {
            "positive": 1000, "positive_unique": 1000,
            "v12_calibration_null": 1000, "v12_independent_test_null": 1000,
            "g1_attempts": 1120, "g1_failures": 120,
        },
        "overlaps": {label: {"count": 0} for label in b0.OVERLAP_LABELS},
        "execution_attestation": {
            **{label: 0 for label in (
                "detector_imports", "detector_model_loads", "detector_score_reads",
                "detector_scores_computed", "resegmentation_calls", "certificate_calls")},
            "sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
    }


class LocalBoundTests(unittest.TestCase):
    def test_recentered_classifier_counterexample(self):
        result = bound.recentering_counterexample()
        self.assertEqual(result["c1_original"], 1)
        self.assertEqual(result["c1_attacked"], 0)
        self.assertEqual(result["c2_original"], 0)

    def test_five_edges_can_change_at_radius_one(self):
        row = bound.check_pair(12, (2, 4, 6, 8, 10), (2, 6, 8, 9, 10))
        self.assertEqual(row[3:5], [1, 5])
        self.assertEqual(bound.local_hit_lower_bound(9, 10, 1), 4)
        self.assertEqual(bound.local_hit_lower_bound(3, 10, 1), 0)

    def test_radius_zero_and_monotonicity(self):
        for edges in range(1, 30):
            for hits in range(edges + 1):
                scores = [bound.local_score_lower_bound(hits, edges, k) for k in range(4)]
                self.assertEqual(scores[0], b0.score_from_counts(hits, edges))
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_invalid_domains_rejected(self):
        for args in ((True, 3, 1), (2, 0, 1), (4, 3, 1), (2, 3, -1)):
            with self.assertRaises(ValueError):
                bound.local_hit_lower_bound(*args)
        with self.assertRaises(ValueError):
            bound.pair_spans(8, (3, 3))


class IdentityTests(unittest.TestCase):
    def test_h0_gate_rejects_partial_or_unbound_evidence(self):
        b0.validate_h0(synthetic_h0())
        for field, value in (("status", "running"), ("runner_sha256", identity("wrong")),
                             ("overlaps", {}), ("contains_raw_text", True)):
            altered = synthetic_h0()
            altered[field] = value
            with self.assertRaises(ValueError):
                b0.validate_h0(altered)
        altered = synthetic_h0()
        altered["execution_attestation"]["detector_scores_computed"] = 1
        with self.assertRaises(ValueError):
            b0.validate_h0(altered)

    def test_manifest_order_uniqueness_and_native_limits(self):
        sources, positives = manifests()
        plan = b0.make_public_plan(sources, positives)
        self.assertEqual([plan[i]["cohort"] for i in (0, 1000, 2000)], list(b0.COHORTS))
        for field, value in (
            ("positive_index", 1), ("output_raw_text_sha256", sources[0]["raw_text_sha256"]),
            ("max_native_tokens", 513), ("unit_count", 1), ("unit_count", True),
        ):
            altered = copy.deepcopy(positives)
            altered[0][field] = value
            with self.assertRaises(ValueError):
                b0.make_public_plan(sources, altered)

    def test_terminal_public_file_binding_without_sealed_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h0 = synthetic_h0()
            sources, positives = manifests()
            values = {
                "g1_result": {
                    "stage": "WP2-v13-G1-SemStamp-generation", "status": "go",
                    "terminal_reason": "target_eligible_reached",
                    "counts": {"eligible_count": 1000, "attempt_count": 1120},
                },
                "v12_source_identities": sources, "positive_identities": positives,
            }
            h0["inputs"] = {}
            for label, value in values.items():
                encoded = (b0.canonical_bytes(value) if isinstance(value, dict)
                           else b"\n".join(map(b0.canonical_bytes, value))) + b"\n"
                (root / label).write_bytes(encoded)
                h0["inputs"][label] = {
                    "relative_path": label, "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            h0_path = root / "h0.json"
            h0_path.write_bytes(b0.canonical_bytes(h0))
            frozen = b0.file_sha256(h0_path)
            self.assertEqual(len(b0.load_public_plan(root, h0_path, frozen)), 3000)
            with self.assertRaises(ValueError):
                b0.load_public_plan(root, h0_path, identity("wrong"))
            (root / "positive_identities").write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                b0.load_public_plan(root, h0_path, frozen)


class ScoreAndReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = b0.make_public_plan(*manifests())
        cls.relation = synthetic_relation()
        cls.relation_sha = hashlib.sha256(cls.relation).hexdigest()

    def setUp(self):
        # Explicit synthetic-only pin; the production function has no override.
        self.pin = patch.object(b0, "RELATION_SHA256", self.relation_sha)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def entry(self, planned):
        regions = [0, 0, 0, 0] if planned["cohort"] == "positive" else [0, 1, 2, 3]
        score = math.sqrt(9) if planned["cohort"] == "positive" else -1.0
        return {
            "regions": regions,
            "public": b0.make_public_score(planned, regions, self.relation, score),
        }

    def checkpoint(self, complete=False):
        entries = [self.entry(row) for row in self.plan] if complete else [
            self.entry(self.plan[0])]
        return {
            "schema_version": 1, "contract_sha256": identity("contract"),
            "plan_sha256": b0.digest(self.plan),
            "replays": {"0": entries, "1": copy.deepcopy(entries) if complete else []},
        }

    def test_directed_relation_and_exact_score_checks(self):
        score = b0.score_from_counts(1, 2)
        planned = {**self.plan[0], "unit_count": 3}
        row = b0.make_public_score(planned, [0, 3, 0], self.relation, score)
        self.assertEqual(row["hit_count"], 1)
        for regions, official in (([0, 3, 0], score + 1e-12),
                                  ([0, 3, 8], score), ([0, True, 0], score),
                                  ([0, 3], score), ([0, 3, 0], float("nan"))):
            with self.assertRaises(ValueError):
                b0.make_public_score(planned, regions, self.relation, official)
        wrong_relation = bytes(reversed(self.relation))
        with self.assertRaises(ValueError):
            b0.validate_relation(wrong_relation)

    def test_public_row_cannot_hide_raw_or_change_counts(self):
        row = self.entry(self.plan[0])["public"]
        for key, value in (("hit_count", 1), ("edge_count", 4),
                           ("raw_text", "synthetic"), ("regions_sha256", "broken")):
            with self.assertRaises(ValueError):
                b0.validate_public_score({**row, key: value}, self.plan[0])

    def test_partial_checkpoint_resumes_but_cannot_summarize(self):
        state = self.checkpoint()
        result = b0.validate_checkpoint(state, self.plan, self.relation, identity("contract"))
        self.assertEqual(result["completed"], [1, 0])
        with self.assertRaises(ValueError):
            b0.summarize_checkpoint(state, self.plan, self.relation, identity("contract"))
        state["replays"]["1"] = copy.deepcopy(state["replays"]["0"])
        with self.assertRaises(ValueError):
            b0.validate_checkpoint(state, self.plan, self.relation, identity("contract"))

    def test_replay_and_region_corruption_rejected(self):
        state = self.checkpoint(True)
        result = b0.summarize_checkpoint(state, self.plan, self.relation, identity("contract"))
        self.assertTrue(result["statistics_pass"])
        self.assertFalse(result["b1_authorized"])
        altered = copy.deepcopy(state)
        altered["replays"]["1"][0]["regions"][0] = 1
        with self.assertRaises(ValueError):
            b0.validate_checkpoint(altered, self.plan, self.relation, identity("contract"))
        altered = copy.deepcopy(state)
        altered["replays"]["1"][0], altered["replays"]["1"][1] = (
            altered["replays"]["1"][1], altered["replays"]["1"][0])
        with self.assertRaises(ValueError):
            b0.validate_checkpoint(altered, self.plan, self.relation, identity("contract"))
        with self.assertRaises(ValueError):
            b0.validate_checkpoint(state, self.plan, self.relation, identity("wrong-contract"))

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            b0.atomic_write_checkpoint(path, {"previous": 1})
            with patch.object(b0.os, "replace", side_effect=OSError("injected interruption")):
                with self.assertRaises(OSError):
                    b0.atomic_write_checkpoint(path, {"next": 2})
            self.assertEqual(json.loads(path.read_text()), {"previous": 1})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class StatisticsTests(unittest.TestCase):
    def test_rank_ties_and_cp_boundary(self):
        summary = b0.summarize_scores([0] * 1000, [0] * 935 + [1] * 65,
                                      [0] * 200 + [1] * 800)
        self.assertEqual(summary["calibration"]["order_rank"], 951)
        self.assertEqual(summary["calibration"]["strict_exceedances"], 0)
        self.assertLessEqual(summary["test"]["one_sided_cp_upper"], 0.08)
        self.assertGreater(b0.clopper_pearson_upper(66, 1000), 0.08)
        self.assertGreater(summary["positive"]["one_sided_cp_lower"], 0.75)

    def test_nonfinite_and_boolean_scores_rejected(self):
        for invalid in (True, float("nan"), float("inf"), "1.0"):
            with self.assertRaises(ValueError):
                b0.summarize_scores([0, invalid], [0], [1])

    def test_historical_v10_v11_exact_reconstruction(self):
        base = ROOT / "experiments"
        r1 = json.loads((base / "certified_resegmentation_wp2_v11_detector_reproduction1"
                        / "r1_semstamp.json").read_text())
        rows = r1["execution"]["rows"]
        self.assertEqual(len(rows), 1999)
        for row in rows:
            self.assertEqual(b0.score_from_counts(row["hit_count"], row["edge_count"]),
                             row["direct_score"])
            self.assertEqual(row["direct_score"], row["official_score"])
        summary = b0.summarize_scores(*[
            [r["direct_score"] for r in rows if r["cohort"] == cohort]
            for cohort in ("certificate_calibration", "independent_test", "positive")
        ])
        old = json.loads((base / "certified_resegmentation_wp2_v10_semstamp1"
                         / "p0_summary.json").read_text())
        self.assertEqual(summary["calibration"], old["calibration"])
        for cohort in ("test", "positive"):
            for field, expected in old[cohort].items():
                if field in ("one_sided_cp_lower", "one_sided_cp_upper"):
                    # Historical Linux vs local macOS libm rounding only.
                    # Count, score, threshold and replay checks stay exact.
                    self.assertLessEqual(abs(summary[cohort][field] - expected), 1e-12)
                else:
                    self.assertEqual(summary[cohort][field], expected)


if __name__ == "__main__":
    unittest.main()
