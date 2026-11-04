import json
from pathlib import Path
import tempfile
import unittest

import run_certified_resegmentation_wp2_v15_generation as runner


def metrics(**updates):
    value = {
        "patch_id": runner.PATCH_ID,
        "max_trials": 100,
        "total_trials": 14,
        "accepted_segments": 4,
        "max_trials_reached_segments": 0,
        "fallback_acceptances": 0,
        "max_current_trials_observed": 6,
        "terminated_with_empty_generation": False,
    }
    value.update(updates)
    return value


def attempt(index=0, eligible=True):
    codes = [] if eligible else ["invalid_or_short_p0"]
    output_identity = f"{index + 1:064x}"
    return {
        "attempt_index": index,
        "source_raw_text_sha256": f"{index + 1000:064x}",
        "seed": index,
        "prompt_sha256": f"{index + 2000:064x}",
        "output": {
            "raw_text_sha256": output_identity,
            "unit_count": 2,
            "max_native_tokens": 10,
            "output_equals_prompt": False,
        },
        "exclusion_codes": codes,
        "eligible": eligible,
        "elapsed_seconds": 1.0,
        "sampling": metrics(),
        "hidden_log_sha256": f"{index + 3000:064x}",
        "source_text": "source",
        "prompt": "prompt",
        "generated_text": "generated",
    }


def expected(row, eligible_count=1, **updates):
    value = {
        "attempt_index": row["attempt_index"],
        "output_raw_text_sha256": row["output"]["raw_text_sha256"],
        "eligible": row["eligible"],
        "exclusion_codes": row["exclusion_codes"],
        "eligible_count": eligible_count,
    }
    value.update(updates)
    return value


class SamplingMetricTests(unittest.TestCase):
    def test_valid_metrics(self):
        self.assertEqual(
            runner.validate_sampling_metrics(metrics())["total_trials"],
            14,
        )
        self.assertEqual(
            runner.validate_sampling_metrics(metrics(
                total_trials=100,
                max_trials_reached_segments=1,
                fallback_acceptances=1,
                max_current_trials_observed=100,
            ))["fallback_acceptances"],
            1,
        )

    def test_rejects_invalid_metrics(self):
        changes = (
            {"patch_id": "wrong"},
            {"max_trials": 99},
            {"total_trials": True},
            {"max_current_trials_observed": 101},
            {"fallback_acceptances": 2, "max_trials_reached_segments": 1},
            {"max_trials_reached_segments": 5, "accepted_segments": 4},
            {"total_trials": 501},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_sampling_metrics(metrics(**change))


class PrefixTests(unittest.TestCase):
    def test_exact_prefix_match(self):
        row = attempt()
        comparison = runner.prefix_comparison(row, expected(row), 1)
        self.assertTrue(comparison["acceptable"])
        self.assertEqual(comparison["classification"], "exact")
        self.assertTrue(comparison["semantic_match"])
        self.assertTrue(comparison["cumulative_eligible_match"])
        self.assertEqual(
            comparison["actual_sha256"],
            comparison["expected_sha256"],
        )

    def test_cumulative_shift_only_is_accepted(self):
        row = attempt()
        comparison = runner.prefix_comparison(
            row, expected(row, eligible_count=0), 1
        )
        self.assertTrue(comparison["acceptable"])
        self.assertEqual(
            comparison["classification"], "cumulative_shift_only"
        )
        self.assertTrue(comparison["semantic_match"])
        self.assertFalse(comparison["cumulative_eligible_match"])

    def test_no_fallback_semantic_divergence_stops(self):
        row = attempt()
        comparison = runner.prefix_comparison(
            row,
            expected(row, output_raw_text_sha256="f" * 64),
            1,
        )
        self.assertFalse(comparison["acceptable"])
        self.assertEqual(
            comparison["classification"], "unexplained_divergence"
        )
        self.assertFalse(comparison["fallback_activated"])

    def test_fallback_semantic_divergence_is_explained(self):
        row = attempt()
        row["sampling"] = metrics(
            total_trials=100,
            max_trials_reached_segments=1,
            fallback_acceptances=1,
            max_current_trials_observed=100,
        )
        comparison = runner.prefix_comparison(
            row,
            expected(row, output_raw_text_sha256="f" * 64),
            1,
        )
        self.assertTrue(comparison["acceptable"])
        self.assertEqual(
            comparison["classification"], "patch_explained_divergence"
        )
        self.assertTrue(comparison["fallback_activated"])

    def test_fallback_never_exempts_index_mismatch(self):
        row = attempt()
        row["sampling"] = metrics(
            total_trials=100,
            max_trials_reached_segments=1,
            fallback_acceptances=1,
            max_current_trials_observed=100,
        )
        comparison = runner.prefix_comparison(
            row, expected(row, attempt_index=1), 1
        )
        self.assertFalse(comparison["acceptable"])
        self.assertFalse(comparison["index_match"])

    def test_fallback_never_exempts_metadata_only_divergence(self):
        row = attempt()
        row["sampling"] = metrics(
            total_trials=100,
            max_trials_reached_segments=1,
            fallback_acceptances=1,
            max_current_trials_observed=100,
        )
        comparison = runner.prefix_comparison(
            row,
            expected(
                row,
                eligible=False,
                exclusion_codes=["invalid_or_short_p0"],
            ),
            1,
        )
        self.assertFalse(comparison["acceptable"])
        self.assertTrue(comparison["output_identity_match"])
        self.assertFalse(comparison["eligibility_match"])

    def test_v14_observed_rows_have_expected_causal_classes(self):
        root = Path(__file__).resolve().parents[1]
        result = json.loads(
            (
                root
                / "experiments"
                / "certified_resegmentation_wp2_v14_bounded_generation1"
                / "G1_RESULT.json"
            ).read_text(encoding="utf-8")
        )
        prefix = runner.parent.read_jsonl(
            root
            / "experiments"
            / "certified_resegmentation_wp2_v14_bounded_generation1"
            / "V13_COMMITTED_PREFIX.jsonl"
        )
        eligible_count = 0
        observed = []
        for row in result["attempts"]:
            eligible_count += int(row["eligible"])
            observed.append(
                runner.prefix_comparison(
                    row, prefix[row["attempt_index"]], eligible_count
                )["classification"]
            )
        self.assertEqual(
            observed,
            ["exact", "exact", "patch_explained_divergence"],
        )

    def test_nonprefix_row_has_no_comparison(self):
        self.assertIsNone(runner.prefix_comparison(attempt(), None, 1))


class CheckpointTests(unittest.TestCase):
    def test_new_checkpoint_is_bound_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed" / "checkpoint.json"
            checkpoint = runner.load_or_create_checkpoint(
                path, "a" * 64
            )
            self.assertEqual(checkpoint["stage"], runner.STAGE)
            self.assertEqual(checkpoint["attempts"], [])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_checkpoint_rejects_runner_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            runner.load_or_create_checkpoint(path, "a" * 64)
            with self.assertRaises(ValueError):
                runner.load_or_create_checkpoint(path, "b" * 64)

    def test_public_redaction_keeps_sampler_evidence(self):
        public = runner.parent.public_attempt(attempt())
        for field in ("source_text", "prompt", "generated_text"):
            self.assertNotIn(field, public)
        self.assertEqual(public["sampling"]["patch_id"], runner.PATCH_ID)
        json.dumps(public, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
