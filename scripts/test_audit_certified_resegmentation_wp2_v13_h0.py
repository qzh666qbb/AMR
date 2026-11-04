import hashlib
import json
import unittest

import audit_certified_resegmentation_wp2_v13_h0 as audit


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


def valid_terminal_fixture():
    v13_sources = [
        identity("source", index)
        for index in range(audit.MAX_ATTEMPTS)
    ]
    attempts = []
    positive_rows = []
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
            "source_raw_text_sha256": v13_sources[index],
            "seed": index,
            "prompt_sha256": identity("prompt", index),
            "output": output,
            "exclusion_codes": [],
            "eligible": True,
            "elapsed_seconds": 1.0,
            "hidden_log_sha256": identity("log", index),
        })
        positive_rows.append({
            "positive_index": index,
            "attempt_index": index,
            "source_raw_text_sha256": v13_sources[index],
            "output_raw_text_sha256": output_identity,
            "seed": index,
            "unit_count": 2,
            "max_native_tokens": 10,
            "contains_raw_text": False,
        })
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
        "stage": "WP2-v13-G1-SemStamp-generation",
        "status": "go",
        "terminal_reason": "target_eligible_reached",
        "contains_raw_text": False,
        "scope": "formal_generation_only_no_detector_scores",
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "contract_sha256": audit.CONTRACT_SHA256,
        "g0_sha256": audit.G0_SHA256,
        "identity_manifest_sha256": audit.V13_SOURCE_SHA256,
        "sealed_source_sha256": audit.SEALED_SOURCE_SHA256,
        "runner_sha256": audit.GENERATION_RUNNER_SHA256,
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
            "raw_text_emitted_to_public_artifacts": False,
        },
        "next_gate": "run_H0_identity_freeze",
    }
    return (
        result,
        positive_rows,
        positive_metadata,
        v13_sources,
        sealed_metadata,
    )


class SemStampV13H0Tests(unittest.TestCase):
    def validate(self, result, positive_rows, metadata, sources, sealed):
        return audit.validate_g1_result(
            result,
            positive_rows,
            metadata,
            "positive.jsonl",
            sources,
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
        fixture[0]["attempts"][1]["output"]["raw_text_sha256"] = (
            fixture[0]["attempts"][0]["output"]["raw_text_sha256"]
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
            [identity("v13", 0)],
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

    def test_sha256_validation_is_canonical(self):
        self.assertTrue(audit.is_sha256("a" * 64))
        self.assertFalse(audit.is_sha256("A" * 64))
        self.assertFalse(audit.is_sha256("not-a-hash"))


if __name__ == "__main__":
    unittest.main()
