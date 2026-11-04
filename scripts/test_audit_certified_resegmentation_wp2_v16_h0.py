import copy
import hashlib
import json
from pathlib import Path
import unittest

import audit_certified_resegmentation_wp2_v16_h0 as audit


ROOT = Path(__file__).resolve().parents[1]
V15 = (
    ROOT
    / "experiments"
    / "certified_resegmentation_wp2_v15_causal_prefix1"
)


def identity(label, index):
    return hashlib.sha256(
        f"{label}-{index}".encode("ascii")
    ).hexdigest()


def jsonl_metadata(rows):
    encoded = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def supplement_fixture():
    sources = [
        identity("source", index)
        for index in range(audit.MAX_ATTEMPTS)
    ]
    attempts = []
    positives = []
    for supplement_index in range(audit.TARGET_SUPPLEMENT):
        attempt_index = audit.START_ATTEMPT + supplement_index
        output_identity = identity("output", attempt_index)
        seed = audit.generation_seed(sources[attempt_index])
        attempts.append({
            "attempt_index": attempt_index,
            "source_raw_text_sha256": sources[attempt_index],
            "seed": seed,
            "prompt_sha256": identity("prompt", attempt_index),
            "output": {
                "raw_text_sha256": output_identity,
                "unit_count": 2,
                "max_native_tokens": 10,
                "output_equals_prompt": False,
            },
            "exclusion_codes": [],
            "eligible": True,
            "elapsed_seconds": 1.0,
            "sampling": {
                "patch_id": audit.common.PATCH_ID,
                "max_trials": 100,
                "total_trials": 4,
                "accepted_segments": 2,
                "max_trials_reached_segments": 0,
                "fallback_acceptances": 0,
                "max_current_trials_observed": 2,
                "terminated_with_empty_generation": False,
            },
            "hidden_log_sha256": identity("log", attempt_index),
        })
        positives.append({
            "supplement_index": supplement_index,
            "attempt_index": attempt_index,
            "source_raw_text_sha256": sources[attempt_index],
            "output_raw_text_sha256": output_identity,
            "seed": seed,
            "unit_count": 2,
            "max_native_tokens": 10,
            "contains_raw_text": False,
        })
    metadata = jsonl_metadata(positives)
    sealed = {
        "sha256": identity("sealed", 0),
        "bytes": 1234,
    }
    result = {
        "schema_version": 1,
        "stage": "WP2-v16-G1R-prompt-only-rescue",
        "status": "go",
        "terminal_reason": "target_supplement_reached",
        "contains_raw_text": False,
        "scope": "fixed_queue_prompt_only_rescue_no_detector_scores",
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "contract_sha256": audit.CONTRACT_SHA256,
        "runner_sha256": audit.GENERATION_RUNNER_SHA256,
        "bounded_sampler_sha256": audit.BOUNDED_SAMPLER_SHA256,
        "generation_parameters": audit.GENERATION_PARAMETERS,
        "v15_parent": {
            "protocol_sha256": audit.common.PROTOCOL_SHA256,
            "contract_sha256": audit.common.CONTRACT_SHA256,
            "runner_sha256": audit.common.GENERATION_RUNNER_SHA256,
            "g1_result_sha256": audit.V15_G1_RESULT_SHA256,
            "positive_identities_sha256": audit.V15_POSITIVE_SHA256,
            "h0_stop_sha256": audit.V15_H0_STOP_SHA256,
            "sealed_checkpoint_sha256": audit.V15_SEALED_SHA256,
            "sealed_checkpoint_bytes": audit.V15_SEALED_BYTES,
            "sealed_checkpoint_content_parsed": False,
        },
        "parent_reclassification": {
            "claimed_positive": 1000,
            "excluded_prompt_only": 5,
            "prompt_only_attempts": list(
                audit.PROMPT_ONLY_ATTEMPTS
            ),
            "retained_positive": audit.RETAINED_PARENT,
            "rule": (
                "output_equals_prompt_true_to_prompt_only_output"
            ),
            "score_conditioned": False,
        },
        "counts": {
            "retained_parent_count": audit.RETAINED_PARENT,
            "supplement_attempt_count": audit.TARGET_SUPPLEMENT,
            "supplement_eligible_count": audit.TARGET_SUPPLEMENT,
            "supplement_failure_count": 0,
            "combined_eligible_count": audit.TARGET_COMBINED,
            "target_supplement": audit.TARGET_SUPPLEMENT,
            "target_combined": audit.TARGET_COMBINED,
            "failure_codes": {},
        },
        "continuation": {
            "start_attempt": audit.START_ATTEMPT,
            "end_attempt_inclusive": (
                audit.START_ATTEMPT + audit.TARGET_SUPPLEMENT - 1
            ),
            "strictly_sequential": True,
            "v15_output_identity_exclusion_count": 10,
        },
        "bounded_sampling": {
            "patch_id": audit.common.PATCH_ID,
            "maximum_trials_per_segment": 100,
            "total_trials": 4 * audit.TARGET_SUPPLEMENT,
            "fallback_attempts": 0,
            "fallback_acceptances": 0,
        },
        "supplement_manifest": {
            **metadata,
            "rows": audit.TARGET_SUPPLEMENT,
            "relative_path": "supplement.jsonl",
            "contains_raw_text": False,
        },
        "attempts": attempts,
        "sealed_checkpoint": {
            **sealed,
            "relative_path": "sealed/checkpoint.json",
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
            "v15_sealed_checkpoint_content_parsed": False,
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": "run_independent_v16_H0_identity_audit",
    }
    return result, positives, metadata, sources, sealed


class ParentTests(unittest.TestCase):
    def test_actual_parent_reclassifies_to_995(self):
        result = json.loads(
            (V15 / "G1_RESULT.json").read_text(encoding="utf-8")
        )
        positives = audit.common.read_jsonl(
            V15 / "POSITIVE_IDENTITIES.jsonl"
        )
        h0_stop = json.loads(
            (V15 / "H0_STOP.json").read_text(encoding="utf-8")
        )
        retained, outputs, _failed = audit.validate_parent(
            result, positives, h0_stop
        )
        self.assertEqual(len(retained), audit.RETAINED_PARENT)
        self.assertGreaterEqual(len(outputs), audit.RETAINED_PARENT)


class SupplementTests(unittest.TestCase):
    def validate(self, fixture):
        result, positives, metadata, sources, sealed = fixture
        return audit.validate_supplement(
            result,
            positives,
            metadata,
            "supplement.jsonl",
            sources,
            {identity("v15", index) for index in range(10)},
            sealed["sha256"],
            sealed,
            "sealed/checkpoint.json",
        )

    def test_valid_supplement_passes(self):
        result = self.validate(supplement_fixture())
        self.assertEqual(
            len(result["eligible_attempts"]),
            audit.TARGET_SUPPLEMENT,
        )

    def test_prompt_only_requires_exclusion(self):
        fixture = list(supplement_fixture())
        fixture[0] = copy.deepcopy(fixture[0])
        fixture[0]["attempts"][0]["output"][
            "output_equals_prompt"
        ] = True
        with self.assertRaisesRegex(
            ValueError, "prompt-only exclusion mismatch"
        ):
            self.validate(fixture)

    def test_combined_rejects_duplicate_output(self):
        retained = [
            {
                "attempt_index": index,
                "source_raw_text_sha256": identity("source", index),
                "output_raw_text_sha256": identity("output", index),
                "seed": index,
                "unit_count": 2,
                "max_native_tokens": 10,
            }
            for index in range(audit.RETAINED_PARENT)
        ]
        supplement = [
            {
                "attempt_index": audit.START_ATTEMPT + index,
                "source_raw_text_sha256": identity(
                    "source", audit.START_ATTEMPT + index
                ),
                "output_raw_text_sha256": identity(
                    "supplement", index
                ),
                "seed": index,
                "unit_count": 2,
                "max_native_tokens": 10,
            }
            for index in range(audit.TARGET_SUPPLEMENT)
        ]
        supplement[0]["output_raw_text_sha256"] = retained[0][
            "output_raw_text_sha256"
        ]
        with self.assertRaisesRegex(
            ValueError, "identities are not unique"
        ):
            audit.build_combined(retained, supplement)


if __name__ == "__main__":
    unittest.main()
