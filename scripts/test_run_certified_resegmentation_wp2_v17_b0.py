"""Offline tests for the frozen WP2-v17 B0 runner."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import certified_resegmentation_wp2_v13_b0_support as support
import run_certified_resegmentation_wp2_v17_b0 as b0


ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def attempt(index, text, eligible=True):
    return {
        "attempt_index": index,
        "source_raw_text_sha256": identity(f"source-{index}"),
        "seed": index + 17,
        "eligible": eligible,
        "exclusion_codes": [] if eligible else ["prompt_only_output"],
        "output": {
            "raw_text_sha256": identity(text),
            "unit_count": 2,
            "max_native_tokens": 7,
            "output_equals_prompt": not eligible,
        },
        "prompt": text if not eligible else "Prompt.",
        "generated_text": text,
    }


def positive_manifest():
    v15_attempts = [
        attempt(index, f"V15 output {index}. Second sentence.")
        if index < 995
        else {"attempt_index": index}
        for index in range(1111)
    ]
    v16_attempts = [
        attempt(1111, "Prompt only. Second sentence.", eligible=False)
    ] + [
        attempt(index, f"V16 output {index}. Second sentence.")
        for index in range(1112, 1117)
    ]
    combined = []
    for index in range(1000):
        if index < 995:
            current = v15_attempts[index]
            origin = "v15_retained"
            origin_index = index
        else:
            current = v16_attempts[index - 994]
            origin = "v16_supplement"
            origin_index = index - 995
        combined.append({
            "positive_index": index,
            "origin": origin,
            "origin_index": origin_index,
            "attempt_index": current["attempt_index"],
            "source_raw_text_sha256": current["source_raw_text_sha256"],
            "output_raw_text_sha256": current["output"]["raw_text_sha256"],
            "seed": current["seed"],
            "unit_count": 2,
            "max_native_tokens": 7,
            "contains_raw_text": False,
        })
    v15 = {
        "stage": "WP2-v15-G1-SemStamp-causal-prefix-generation",
        "status": "go",
        "terminal_reason": "target_eligible_reached",
        "attempts": v15_attempts,
    }
    v16 = {
        "stage": "WP2-v16-G1R-prompt-only-rescue",
        "status": "go",
        "terminal_reason": "target_supplement_reached",
        "attempts": v16_attempts,
    }
    return combined, v15, v16


class ParentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.h0 = json.loads(
            (
                ROOT
                / "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1"
                / "H0_RESULT.json"
            ).read_text(encoding="utf-8")
        )

    def test_frozen_v16_h0_is_accepted(self):
        self.assertIs(b0.validate_h0(self.h0), self.h0)

    def test_h0_drift_is_rejected(self):
        for mutation in ("status", "overlap", "detector"):
            altered = copy.deepcopy(self.h0)
            if mutation == "status":
                altered["status"] = "running"
            elif mutation == "overlap":
                altered["overlaps"]["v12_all_sources"]["count"] = 1
            else:
                altered["execution_attestation"]["detector_model_loads"] = 1
            with self.assertRaises(ValueError):
                b0.validate_h0(altered)

    def test_real_public_manifests_are_exact(self):
        v12 = b0.read_jsonl(
            ROOT
            / "experiments/certified_resegmentation_wp2_v12_fresh_assets1"
            / "SOURCE_IDENTITIES.jsonl"
        )
        combined = b0.read_jsonl(
            ROOT
            / "experiments/certified_resegmentation_wp2_v16_prompt_only_rescue1"
            / "COMBINED_POSITIVE_IDENTITIES.jsonl"
        )
        b0.validate_v12_public(v12)
        b0.validate_combined(combined)
        plan = support.make_public_plan(v12, combined)
        self.assertEqual(len(plan), 3000)
        self.assertEqual(
            [plan[index]["cohort"] for index in (0, 1000, 2000)],
            list(support.COHORTS),
        )


class SealedMappingTests(unittest.TestCase):
    def test_null_mapping_recomputes_text_identity(self):
        public = []
        sealed = []
        for index in range(2000):
            cohort = (
                "calibration_null"
                if index < 1000
                else "independent_test_null"
            )
            cohort_index = index % 1000
            text = f"Null document {index}. Second sentence."
            public.append({
                "allocation_index": index,
                "cohort": cohort,
                "cohort_index": cohort_index,
                "source_row_index": index + 10,
                "raw_text_sha256": identity(text),
            })
            sealed.append({
                **public[-1],
                "text": text,
            })
        mapped = b0.map_null_texts(public, sealed)
        self.assertEqual(len(mapped), 2000)
        self.assertEqual(
            mapped[("independent_test_null", 999)],
            "Null document 1999. Second sentence.",
        )
        sealed[0]["text"] = "Changed."
        with self.assertRaises(ValueError):
            b0.map_null_texts(public, sealed)

    def test_positive_mapping_preserves_995_plus_5_provenance(self):
        combined, v15, v16 = positive_manifest()
        b0.validate_combined(combined)
        texts = b0.map_positive_texts(combined, v15, v16)
        self.assertEqual(len(texts), 1000)
        self.assertEqual(texts[994], "V15 output 994. Second sentence.")
        self.assertEqual(texts[995], "V16 output 1112. Second sentence.")
        self.assertEqual(texts[-1], "V16 output 1116. Second sentence.")

    def test_positive_mapping_rejects_prompt_only_or_origin_drift(self):
        combined, v15, v16 = positive_manifest()
        v15["attempts"][0]["output"]["output_equals_prompt"] = True
        with self.assertRaises(ValueError):
            b0.map_positive_texts(combined, v15, v16)
        combined, v15, v16 = positive_manifest()
        combined[995]["origin_index"] = 1
        with self.assertRaises(ValueError):
            b0.validate_combined(combined)


class ExecutionPrimitiveTests(unittest.TestCase):
    def test_import_is_model_free(self):
        self.assertNotIn("torch", vars(b0))
        self.assertNotIn("numpy", vars(b0))

    def test_official_replay_model_requires_exact_consumption(self):
        model = b0.ReplayLSHModel([1, 2])
        self.assertEqual(model.get_hash(["a"]), [1])
        with self.assertRaises(AssertionError):
            model.assert_exhausted()
        self.assertEqual(model.get_hash(["b"]), [2])
        model.assert_exhausted()
        with self.assertRaises(AssertionError):
            model.get_hash(["c"])
        with self.assertRaises(ValueError):
            b0.ReplayLSHModel([1]).get_hash(["a", "b"])

    def test_atomic_public_write_is_private_and_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            b0.atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_frozen_statistical_boundary(self):
        summary = support.summarize_scores(
            [0.0] * 1000,
            [0.0] * 935 + [1.0] * 65,
            [0.0] * 200 + [1.0] * 800,
        )
        self.assertEqual(summary["calibration"]["order_rank"], 951)
        self.assertLessEqual(summary["test"]["one_sided_cp_upper"], 0.08)
        self.assertGreater(summary["positive"]["one_sided_cp_lower"], 0.75)


if __name__ == "__main__":
    unittest.main()
