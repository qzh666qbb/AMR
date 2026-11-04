"""Tests for text-preserving segmentation and frozen cohort integrity."""

import json
import math
import os
from pathlib import Path
import re
import tempfile
import unittest

from analyze_segmentation_intervention import calibrate, paired_success, rate_summary, summarize
from run_segmentation_intervention import (
    ROOT, PRE_METADATA_FIX_SHA, amr_body, compatible_runtime, partition, segment, sha, text_sha, write_json,
)

os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))


class PartitionTests(unittest.TestCase):
    def test_multiline_metadata(self):
        self.assertEqual(amr_body("# ::snt A story.\nMore story text.\n(a / story)\n"), "(a / story)")
        with self.assertRaises(ValueError):
            amr_body("# ::snt Only text\nNo graph")

    def test_old_runtime_only_accepted_when_extraction_unchanged(self):
        runtime = {"script_sha256": PRE_METADATA_FIX_SHA, "device": "cpu"}
        saved = {"runtime": runtime, "runtime_signature": text_sha(json.dumps(runtime, sort_keys=True)),
                 "variants": {"sentence": {"details": [{"raw_amr": "# ::snt text\n(a / story)"}]}}}
        self.assertEqual(compatible_runtime(saved), {"device": "cpu"})
        saved["variants"]["sentence"]["details"][0]["raw_amr"] = "# ::snt text\nextra text\n(a / story)"
        with self.assertRaises(ValueError):
            compatible_runtime(saved)

    def test_no_split(self):
        self.assertEqual(partition("The company announced a server.", []),
                         ["The company announced a server."])

    def test_preserve_semicolon(self):
        text = "The company announced a server; the research team evaluated its performance."
        parts = partition(text, [text.index(";") + 1])
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].endswith(";"))
        self.assertEqual(re.sub(r"\s+", "", "".join(parts)), re.sub(r"\s+", "", text))

    def test_short_tail_is_not_split(self):
        text = "The company announced a server; yesterday."
        self.assertEqual(partition(text, [text.index(";") + 1]), [text])

    def test_invalid_and_duplicate_boundaries(self):
        text = "The company announced a server; the research team evaluated its performance."
        boundary = text.index(";") + 1
        self.assertEqual(partition(text, [-1, 0, boundary, boundary, len(text), len(text) + 1]),
                         partition(text, [boundary]))


class ModelSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import nltk
        import spacy
        cls.nlp = spacy.load("en_core_web_sm")
        cls.tokenize = staticmethod(nltk.sent_tokenize)

    def test_explicit_subject_conjunction(self):
        text = "The company announced a new server, and the research team evaluated its performance."
        pieces = segment(text, self.tokenize, self.nlp)
        self.assertEqual(len(pieces["sentence"]), 1)
        self.assertEqual(len(pieces["clause"]), 2)

    def test_shared_subject_stays_together(self):
        text = "The company announced a new server and evaluated its performance."
        pieces = segment(text, self.tokenize, self.nlp)
        self.assertEqual(pieces["sentence"], pieces["clause"])

    def test_abbreviation_not_a_boundary(self):
        text = "Dr. Smith announced a new server. The research team evaluated its performance."
        pieces = segment(text, self.tokenize, self.nlp)
        self.assertEqual(len(pieces["sentence"]), 2)
        self.assertEqual(len(pieces["clause"]), 2)


class FrozenManifestTests(unittest.TestCase):
    def test_frozen_inputs(self):
        path = ROOT / "experiments/segmentation_intervention/manifest.json"
        if not path.exists():
            self.skipTest("Manifest has not been prepared")
        manifest = json.loads(path.read_text())
        records = manifest["records"]
        ids = [r["id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(r["cohort"] == "attack" for r in records), 30)
        humans = [r for r in records if r["cohort"] != "attack"]
        self.assertEqual(len(humans), 500)
        self.assertEqual(len({r["text_sha256"] for r in humans}), 500)
        for record in records:
            self.assertEqual(text_sha(record["text"]), record["text_sha256"])
            for pieces in record["segments"].values():
                self.assertEqual(re.sub(r"\s+", "", "".join(pieces)), re.sub(r"\s+", "", record["text"]))
        for cohort in ("human_calibration", "human_test"):
            self.assertEqual(sum(r["cohort"] == cohort for r in records), 250)

    def test_partial_results_do_not_report_calibrated_metrics(self):
        folder = ROOT / "experiments/segmentation_intervention"
        if not (folder / "manifest.json").exists():
            self.skipTest("Manifest has not been prepared")
        result = summarize(folder)
        incomplete = any(v["completed"] < v["planned"] for v in result["coverage"].values())
        if incomplete:
            self.assertEqual(result["status"], "paired_pilot_uncalibrated")
            self.assertIsNone(result["calibrated"])


class CalibrationTests(unittest.TestCase):
    def test_strict_threshold(self):
        scores = list(range(250))
        threshold = calibrate(scores)
        self.assertEqual(threshold, 247)
        self.assertEqual(sum(s > threshold for s in scores), 2)

    def test_ties_are_conservative(self):
        scores = [0.] * 240 + [3.] * 10
        self.assertEqual(calibrate(scores), 3.)
        self.assertEqual(sum(s > calibrate(scores) for s in scores), 0)

    def test_invalid_inputs(self):
        for scores, alpha in (([], .01), ([float("nan")], .01), ([1.], 0), ([1.], 1)):
            with self.assertRaises(ValueError):
                calibrate(scores, alpha)

    def test_paired_exact_test(self):
        result = paired_success([True] * 3 + [False] * 27, [False] * 30)
        self.assertEqual(result["sentence_only_success"], 3)
        self.assertEqual(result["clause_only_success"], 0)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], .25)
        self.assertEqual(result["valid_asr_delta_clause_minus_sentence"], -.1)
        self.assertEqual(paired_success([True], [True])["mcnemar_exact_two_sided_p"], 1.)
        with self.assertRaises(ValueError):
            paired_success([True], [])

    def test_zero_false_positives_is_not_certainty(self):
        interval = rate_summary(0, 250)
        self.assertEqual(interval["rate"], 0)
        self.assertEqual(interval["clopper_pearson_95"][0], 0)
        self.assertGreater(interval["clopper_pearson_95"][1], .01)
        self.assertLess(interval["clopper_pearson_95"][1], .02)
        with self.assertRaises(ValueError):
            rate_summary(1, 0)


class CalibrationIntegrationTests(unittest.TestCase):
    def test_test_set_cannot_move_threshold_and_incomplete_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = []
            for cohort, count in (("attack", 1), ("human_calibration", 100), ("human_test", 100)):
                for i in range(count):
                    records.append({
                        "id": f"{cohort}-{i}", "cohort": cohort,
                        "segments": {"sentence": ["a"], "clause": ["a"]},
                        "quality_valid": True, "historical_z": -1 / math.sqrt(3),
                    })
            manifest = folder / "manifest.json"
            write_json(manifest, {"records": records})
            for record in records:
                # Calibration and attack have zero hits; every test text has a hit.
                green = int(record["cohort"] == "human_test")
                variant = {"n": 1, "green": green, "green_rate": green,
                           "z": (green - .25) / math.sqrt(.25 * .75),
                           "details": [{"text": "a", "score": float(green)}]}
                runtime = {"manifest_sha256": sha(manifest), "script_sha256": sha(ROOT / "scripts/run_segmentation_intervention.py")}
                write_json(folder / "detected" / f'{record["id"]}.json', {
                    "record": record, "runtime": runtime,
                    "runtime_signature": text_sha(json.dumps(runtime, sort_keys=True)),
                    "variants": {"sentence": variant, "clause": variant},
                })
            result = summarize(folder)
            self.assertEqual(result["status"], "complete_calibrated")
            for mode in ("sentence", "clause"):
                summary = result["calibrated"][mode]
                self.assertAlmostEqual(summary["threshold"], -1 / math.sqrt(3))
                self.assertEqual(summary["test_fpr"], 1.)
                self.assertFalse(summary["test_fpr_point_estimate_within_1pct"])
                self.assertEqual(summary["valid_successes"], 1)
            (folder / "detected" / "human_test-99.json").unlink()
            self.assertIsNone(summarize(folder)["calibrated"])


if __name__ == "__main__":
    unittest.main()
