import hashlib
import json
import unittest

import audit_certified_resegmentation_wp2_v15_h0 as audit


def identity(prefix, index):
    return hashlib.sha256(f"{prefix}-{index}".encode("ascii")).hexdigest()


def positive_bytes(rows):
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def sampling(**updates):
    value = {
        "patch_id": audit.PATCH_ID,
        "max_trials": 100,
        "total_trials": 4,
        "accepted_segments": 2,
        "max_trials_reached_segments": 0,
        "fallback_acceptances": 0,
        "max_current_trials_observed": 2,
        "terminated_with_empty_generation": False,
    }
    value.update(updates)
    return value


def prefix_evidence(row, expected, eligible_count):
    expected_semantic = {
        "output_raw_text_sha256": expected["output_raw_text_sha256"],
        "eligible": expected["eligible"],
        "exclusion_codes": expected["exclusion_codes"],
    }
    actual_semantic = {
        "output_raw_text_sha256": row["output"]["raw_text_sha256"],
        "eligible": row["eligible"],
        "exclusion_codes": row["exclusion_codes"],
    }
    actual = {
        "attempt_index": row["attempt_index"],
        **actual_semantic,
        "eligible_count": eligible_count,
    }
    index_match = actual["attempt_index"] == expected["attempt_index"]
    output_identity_match = (
        actual_semantic["output_raw_text_sha256"]
        == expected_semantic["output_raw_text_sha256"]
    )
    eligibility_match = (
        actual_semantic["eligible"] == expected_semantic["eligible"]
        and actual_semantic["exclusion_codes"]
        == expected_semantic["exclusion_codes"]
    )
    semantic_match = output_identity_match and eligibility_match
    cumulative_match = eligible_count == expected["eligible_count"]
    fallback_activated = row["sampling"]["fallback_acceptances"] > 0
    if not index_match:
        classification = "unexplained_divergence"
    elif semantic_match and cumulative_match:
        classification = "exact"
    elif semantic_match:
        classification = "cumulative_shift_only"
    elif fallback_activated and not output_identity_match:
        classification = "patch_explained_divergence"
    else:
        classification = "unexplained_divergence"
    return {
        "expected_sha256": audit.sha256_value(expected),
        "actual_sha256": audit.sha256_value(actual),
        "expected_semantic_sha256": audit.sha256_value(expected_semantic),
        "actual_semantic_sha256": audit.sha256_value(actual_semantic),
        "index_match": index_match,
        "output_identity_match": output_identity_match,
        "eligibility_match": eligibility_match,
        "semantic_match": semantic_match,
        "cumulative_eligible_match": cumulative_match,
        "fallback_activated": fallback_activated,
        "classification": classification,
        "acceptable": classification != "unexplained_divergence",
    }


def valid_terminal_fixture():
    v15_sources = [
        identity("source", index)
        for index in range(audit.MAX_ATTEMPTS)
    ]
    attempts = []
    positive_rows = []
    prefix_rows = []
    for index in range(audit.TARGET_ELIGIBLE):
        output_identity = identity("output", index)
        output = {
            "raw_text_sha256": output_identity,
            "unit_count": 2,
            "max_native_tokens": 10,
            "output_equals_prompt": False,
        }
        attempts.append({
            "attempt_index": index,
            "source_raw_text_sha256": v15_sources[index],
            "seed": index,
            "prompt_sha256": identity("prompt", index),
            "output": output,
            "exclusion_codes": [],
            "eligible": True,
            "elapsed_seconds": 1.0,
            "sampling": sampling(),
            "hidden_log_sha256": identity("log", index),
            "v13_prefix_replay": None,
        })
        positive_rows.append({
            "positive_index": index,
            "attempt_index": index,
            "source_raw_text_sha256": v15_sources[index],
            "output_raw_text_sha256": output_identity,
            "seed": index,
            "unit_count": 2,
            "max_native_tokens": 10,
            "contains_raw_text": False,
        })
        if index < audit.PREFIX_ATTEMPTS:
            expected = {
                "attempt_index": index,
                "output_raw_text_sha256": output_identity,
                "eligible": True,
                "exclusion_codes": [],
                "eligible_count": index + 1,
            }
            prefix_rows.append(expected)
            attempts[-1]["v13_prefix_replay"] = prefix_evidence(
                attempts[-1], expected, index + 1
            )
    encoded = positive_bytes(positive_rows)
    positive_metadata = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
    sealed_bytes = b"sealed-terminal-checkpoint"
    sealed_metadata = {
        "sha256": hashlib.sha256(sealed_bytes).hexdigest(),
        "bytes": len(sealed_bytes),
    }
    result = {
        "schema_version": 1,
        "stage": "WP2-v15-G1-SemStamp-causal-prefix-generation",
        "status": "go",
        "terminal_reason": "target_eligible_reached",
        "contains_raw_text": False,
        "scope": (
            "formal_causal_prefix_bounded_generation_only_no_detector_scores"
        ),
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "contract_sha256": audit.CONTRACT_SHA256,
        "g0_sha256": audit.G0_SHA256,
        "identity_manifest_sha256": audit.V15_SOURCE_SHA256,
        "sealed_source_sha256": audit.SEALED_SOURCE_SHA256,
        "runner_sha256": audit.GENERATION_RUNNER_SHA256,
        "parent_runner_sha256": audit.PARENT_GENERATION_RUNNER_SHA256,
        "bounded_sampler_sha256": audit.BOUNDED_SAMPLER_SHA256,
        "v13_runtime_stop_sha256": audit.V13_RUNTIME_STOP_SHA256,
        "v13_prefix_sha256": audit.V13_PREFIX_SHA256,
        "v14_runtime_stop_sha256": audit.V14_RUNTIME_STOP_SHA256,
        "v14_g1_result_sha256": audit.V14_G1_RESULT_SHA256,
        "v14_positive_identities_sha256": audit.V14_POSITIVE_SHA256,
        "v12_smoke_runner_sha256": audit.V12_SMOKE_RUNNER_SHA256,
        "counts": {
            "attempt_count": audit.TARGET_ELIGIBLE,
            "eligible_count": audit.TARGET_ELIGIBLE,
            "failure_count": 0,
            "target_eligible": audit.TARGET_ELIGIBLE,
            "max_attempts": audit.MAX_ATTEMPTS,
            "failure_codes": {},
        },
        "positive_manifest": {
            **positive_metadata,
            "rows": audit.TARGET_ELIGIBLE,
            "relative_path": "positive.jsonl",
            "contains_raw_text": False,
        },
        "bounded_sampling": {
            "patch_id": audit.PATCH_ID,
            "maximum_trials_per_segment": 100,
            "total_trials": 4 * audit.TARGET_ELIGIBLE,
            "fallback_attempts": 0,
            "fallback_acceptances": 0,
        },
        "prefix_replay": {
            "expected": audit.PREFIX_ATTEMPTS,
            "checked": audit.PREFIX_ATTEMPTS,
            "acceptable": audit.PREFIX_ATTEMPTS,
            "classifications": {
                "cumulative_shift_only": 0,
                "exact": audit.PREFIX_ATTEMPTS,
                "patch_explained_divergence": 0,
                "unexplained_divergence": 0,
            },
        },
        "attempts": attempts,
        "sealed_checkpoint": {
            "relative_path": "sealed/checkpoint.json",
            **sealed_metadata,
            "contains_raw_text": True,
            "public_copy_allowed": False,
        },
        "runtime": {
            "max_cuda_memory_allocated_bytes": 4,
            "total_cuda_memory_bytes": 10,
            "process_loads": [{"model_load_seconds": 1.0}],
        },
        "execution_attestation": {
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_score_reads": 0,
            "detector_scores_computed": 0,
            "resegmentation_calls": 0,
            "certificate_calls": 0,
            "v13_sealed_checkpoint_content_parsed": False,
            "v14_sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": "freeze_and_run_v15_H0_identity_audit",
    }
    return (
        result,
        positive_rows,
        positive_metadata,
        v15_sources,
        prefix_rows,
        sealed_metadata,
    )


class SemStampV15H0Tests(unittest.TestCase):
    def validate(
        self, result, positive_rows, metadata, sources, prefix, sealed
    ):
        return audit.validate_g1_result(
            result,
            positive_rows,
            metadata,
            "positive.jsonl",
            sources,
            prefix,
            sealed["sha256"],
            sealed,
            "sealed/checkpoint.json",
        )

    def test_valid_terminal_fixture_passes(self):
        fixture = valid_terminal_fixture()
        result = self.validate(*fixture)
        self.assertEqual(result["attempt_count"], 1000)
        self.assertEqual(len(result["positive_identities"]), 1000)
        self.assertEqual(result["peak_vram_fraction"], 0.4)

    def test_nonterminal_result_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "authorized Go"):
            self.validate(*fixture)

    def test_duplicate_positive_identity_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["attempts"][500]["output"]["raw_text_sha256"] = (
            fixture[0]["attempts"][499]["output"]["raw_text_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "not identity-unique"):
            self.validate(*fixture)

    def test_source_order_mismatch_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["attempts"][0]["source_raw_text_sha256"] = identity(
            "wrong", 0
        )
        with self.assertRaisesRegex(ValueError, "source order"):
            self.validate(*fixture)

    def test_public_raw_key_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["attempts"][0]["generated_text"] = "not public"
        with self.assertRaisesRegex(ValueError, "raw public key"):
            self.validate(*fixture)

    def test_overlap_is_a_stop_condition(self):
        fixture = valid_terminal_fixture()
        positive = fixture[1][0]["output_raw_text_sha256"]
        v12_rows = [
            {
                "raw_text_sha256": positive,
                "cohort": "calibration_null",
            },
            {
                "raw_text_sha256": identity("test", 0),
                "cohort": "independent_test_null",
            },
        ]
        overlaps = audit.compute_overlaps(
            [positive],
            [identity("v15", 0)],
            v12_rows,
            [],
            [],
            [],
        )
        self.assertEqual(overlaps["v12_all_sources"]["count"], 1)
        self.assertEqual(overlaps["v12_calibration_null"]["count"], 1)
        self.assertTrue(
            any(item["count"] for item in overlaps.values())
        )

    def test_causal_evidence_digest_drift_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["attempts"][0]["v13_prefix_replay"][
            "actual_semantic_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "causal-prefix evidence"):
            self.validate(*fixture)

    def test_unexplained_prefix_summary_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["prefix_replay"]["classifications"]["exact"] -= 1
        fixture[0]["prefix_replay"]["classifications"][
            "unexplained_divergence"
        ] = 1
        with self.assertRaisesRegex(ValueError, "terminal summary"):
            self.validate(*fixture)

    def test_sampling_cap_violation_is_rejected(self):
        fixture = list(valid_terminal_fixture())
        fixture[0]["attempts"][0]["sampling"][
            "max_current_trials_observed"
        ] = 101
        with self.assertRaisesRegex(ValueError, "bounded sampling"):
            self.validate(*fixture)

    def test_patch_explained_prefix_divergence_passes(self):
        fixture = list(valid_terminal_fixture())
        result, positive_rows, metadata, _, prefix, _ = fixture
        row = result["attempts"][0]
        changed_identity = identity("bounded-output", 0)
        row["output"]["raw_text_sha256"] = changed_identity
        row["sampling"] = sampling(
            total_trials=100,
            max_trials_reached_segments=1,
            fallback_acceptances=1,
            max_current_trials_observed=100,
        )
        row["v13_prefix_replay"] = prefix_evidence(
            row, prefix[0], 1
        )
        positive_rows[0]["output_raw_text_sha256"] = changed_identity
        encoded = positive_bytes(positive_rows)
        metadata["sha256"] = hashlib.sha256(encoded).hexdigest()
        metadata["bytes"] = len(encoded)
        result["positive_manifest"]["sha256"] = metadata["sha256"]
        result["positive_manifest"]["bytes"] = metadata["bytes"]
        result["prefix_replay"]["classifications"]["exact"] -= 1
        result["prefix_replay"]["classifications"][
            "patch_explained_divergence"
        ] = 1
        result["bounded_sampling"]["total_trials"] += 96
        result["bounded_sampling"]["fallback_attempts"] = 1
        result["bounded_sampling"]["fallback_acceptances"] = 1
        observed = self.validate(*fixture)
        self.assertEqual(
            observed["prefix_classifications"][
                "patch_explained_divergence"
            ],
            1,
        )

    def test_sha256_validation_is_canonical(self):
        self.assertTrue(audit.is_sha256("a" * 64))
        self.assertFalse(audit.is_sha256("A" * 64))
        self.assertFalse(audit.is_sha256("not-a-hash"))


if __name__ == "__main__":
    unittest.main()
