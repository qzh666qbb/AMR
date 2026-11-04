"""Independent denominator, original-control equivalence and resume checks."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_swan_v4_confirmation1 as run
import swan_v4_confirmation1_scoring as score


class ConfirmationTests(unittest.TestCase):
    def rows(self, n, stage):
        return [score.miss_row({"allocation_id": f"{stage}-{i}", "length_bin": 0},
                               v, {}, "failed") for i in range(n) for v in score.VARIANTS]

    def hit(self, row):
        row["word_count"] = 20
        row["arms"]["control"].update(status="ok", statistic=1)

    def summary(self, null, positive):
        cal = SimpleNamespace(calibrate=lambda *a: SimpleNamespace(is_watermarked=True))
        return score.summarize(null, positive, cal, [f"null-{i}" for i in range(1000)],
                               [f"positive-{i}" for i in range(250)])

    def test_null_source_union_and_full_positive_denominator(self):
        null, positive = self.rows(1000, "null"), self.rows(250, "positive")
        self.hit(null[1])
        self.hit(null[2])  # Same source hit twice, union counts once.
        self.hit(null[6 + 3])
        for row in positive[:12]:
            self.hit(row)
        positive[1]["arms"]["control"]["status"] = "abstain"
        positive[9]["arms"]["control"]["status"] = "abstain"
        result = self.summary(null, positive)
        self.assertEqual(result["null"]["any_stress_false_positive"]["successes"], 2)
        self.assertEqual(result["null"]["any_stress_false_positive"]["trials"], 1000)
        self.assertEqual(result["positive"]["full_denominator_TPR"]["baseline"]["trials"], 250)
        self.assertEqual(result["positive"]["conditional_survival"]["semicolon_all"]["rate"], .5)
        self.assertEqual(result["positive"]["all_five_survival"]["rate"], 0)
        self.assertFalse(result["computational_gates_pass"])

    def test_duplicate_rows_cannot_hide_missing_conditions(self):
        null, positive = self.rows(1000, "null"), self.rows(250, "positive")
        null[-1] = null[-2]
        with self.assertRaises(ValueError):
            self.summary(null, positive)

    def test_no_baseline_and_nonfinite_or_invalid_edges(self):
        null, positive = self.rows(1000, "null"), self.rows(250, "positive")
        self.assertIsNone(self.summary(null, positive)["positive"]["all_five_survival"]["rate"])
        self.hit(positive[0])
        for bad in (float("nan"), float("inf"), None):
            positive[0]["arms"]["control"]["statistic"] = bad
            with self.assertRaises(ValueError):
                self.summary(null, positive)
        positive[0]["arms"]["control"].update(statistic=1, selected_invalid_edges=1)
        with self.assertRaises(ValueError):
            self.summary(null, positive)

    def test_joint_gate_uses_baseline_conditional_and_reports_full_rate(self):
        null, positive = self.rows(1000, "null"), self.rows(250, "positive")
        for row in positive[:200 * 6]:
            self.hit(row)
        result = self.summary(null, positive)
        self.assertTrue(result["computational_gates_pass"])
        self.assertEqual(result["positive"]["all_five_survival"]["rate"], 1)
        self.assertEqual(result["positive"]["all_five_full_denominator"]["rate"], .8)
        self.assertAlmostEqual(result["null"]["conditions"]["baseline"]["upper95_one_sided"],
                               1 - .05 ** (1 / 1000))

    def test_original_control_single_arm_scoring_equivalence(self):
        dev = score.core()
        bank = run.ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
        if not bank.exists():
            bank = Path("/Users/bytedance/AMR") / bank.relative_to(run.ROOT)
        builder = dev.ElasticBoundaryProposerV3()
        comparisons = 0
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_root=Path(tmp))
            for text in ("Alice ran.", "Alice ran. Bob stayed.", "Alice ran; Bob stayed.",
                         "Alice ran and Bob stayed, then both left."):
                lattice = builder.build_lattice(text)
                spans = sorted({dev.normalize_text(e.span_text) for e in lattice.edges})
                for invalid in (False, True):
                    records = {s: dev.InvalidTolerantAMRExtractor._record(
                        s, "test", "(a / run-01 :ARG0 (b / person))") for s in spans}
                    audits = {dev.text_sha256(s): {"selected_beam": 0, "events": (
                        [{"level": "ERROR", "beam_index": 0}] if invalid and s == spans[0] else [])}
                              for s in spans}
                    recovery = SimpleNamespace(records=audits, assert_record=lambda r: None)
                    arms = {}
                    for name, cls in (("control", dev.AllBankScorer), ("path0", dev.EmptyPathZeroScorer)):
                        scorer = cls(None, str(bank))
                        scorer.prepare = lambda xs, obj=scorer: setattr(obj, "_records", dict(records))
                        detector = dev.SwanBankLatticeDetectorV4(builder, scorer)
                        detector.optimizer = dev.CheckedHopOptimizer()
                        arms[name] = detector
                    source = {"allocation_id": f"equivalence-{comparisons}", "stage": "null", "length_bin": 0}
                    original = dev.score_document(args, arms, recovery, source, "baseline", text, {})
                    current = score.score_document(args, arms["control"], recovery, source, "baseline", text, {})
                    self.assertEqual(original["arms"]["control"], current["arms"]["control"])
                    self.assertEqual(original["candidate_edges"], current["candidate_edges"])
                    comparisons += 1
            self.assertEqual(comparisons, 8)

    def test_calibration_replay_checks_arrays_and_input_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "INPUT_MANIFEST.json").write_text("{}")
            args = SimpleNamespace(output_root=folder)
            detector = SimpleNamespace(fingerprint="v4", scorer=SimpleNamespace(
                fingerprint="bank", bank_sha256="bank-sha", parser_fingerprint="parser"))
            sources = [{"allocation_id": f"c-{i}", "text": "word " * 20} for i in range(1000)]
            rows = [{"allocation_id": s["allocation_id"], "stage": "calibration", "variant": "baseline",
                     "arms": {"control": {"status": "ok", "statistic": i / 1000,
                                          "unit_count": 2, "green_count": 0}}}
                    for i, s in enumerate(sources)]
            first = score.fit(args, detector, rows, sources, "runtime")
            second = score.fit(args, detector, rows, sources, "runtime")
            self.assertEqual(first.artifact, second.artifact)
            path = folder / "results/CALIBRATION_control.json"
            clean = path.read_text()
            artifact = json.loads(clean)
            artifact["global_statistics"][-1] = 100
            path.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                score.fit(args, detector, rows, sources, "runtime")
            path.write_text(clean)
            (folder / "INPUT_MANIFEST.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                score.fit(args, detector, rows, sources, "runtime")

    def test_budget_pause_does_not_halt_or_block_independent_jobs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "OUT", Path(tmp)), \
                patch.object(run, "bind"), patch.object(run, "generate", side_effect=run.BUDGET.Pause("phase_money_cap")), \
                patch.object(run, "journal", return_value=SimpleNamespace(totals=lambda: {"calls": 1})):
            self.assertEqual(run.execute("generate"), 76)
            self.assertFalse((Path(tmp) / "HALT.json").exists())
            self.assertEqual(run.execute("score"), 76)
            self.assertFalse((Path(tmp) / "HALT.json").exists())

    def test_empty_paid_response_is_replayed_without_another_request(self):
        def payload(body):
            return {"id": "test-response", "model": run.BUDGET.MODEL,
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
                    "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(run.BUDGET.CAPS, {"pilot": (100, 1000000000)}):
                log = run.Journal(Path(tmp), "pilot", 100, payload, "contract", parents=[])
                result = log.request("source-slot-0", "user", "system", 50, .7, .9)
                self.assertEqual(result["text"], "")
                self.assertEqual(log.totals()["calls"], 1)
                self.assertEqual(log.totals()["uncertain_or_invalid_calls"], 0)

    def test_full_250_source_generation_assignment_and_budget(self):
        records = [{"allocation_id": f"b-{b}-{i}", "length_bin": b}
                   for b, count in enumerate((63, 63, 62, 62)) for i in range(count)]
        assignments = run.shared.assignment(records, list(range(100)))
        self.assertEqual(len(assignments), 250)
        self.assertEqual(sum(len(a["slots"]) for a in assignments.values()), 811)
        self.assertEqual(run.CALL_CAP, 811 * 50)
        self.assertGreater(run.CALL_CAP * run.BUDGET.MAX_RESERVE, run.COST_CAP)

    def test_human_collection_requires_two_distinct_complete_reviewers(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            review = folder / "private/human_review"
            review.mkdir(parents=True)
            items = [{"item_id": str(i)} for i in range(250)]
            (review / "PACKET.json").write_text(json.dumps({"items": items}))
            fields = ["item_id", "grammar_pass", "coherence_pass", "no_AMR_leak", "reviewer_id", "is_human"]
            def write(name, author, completed):
                with (review / (name + ".csv")).open("w", newline="") as stream:
                    w = csv.DictWriter(stream, fieldnames=fields)
                    w.writeheader()
                    w.writerows({"item_id": str(i), "grammar_pass": "1" if i < completed else "",
                                 "coherence_pass": "1", "no_AMR_leak": "1",
                                 "reviewer_id": author, "is_human": "true"} for i in range(250))
            with patch.object(run, "OUT", folder), patch.object(run, "verify"), \
                    patch.object(run, "verify_generation"), patch.object(run, "review_packet"):
                write("reviewer1", "human-A", 250)
                write("reviewer2", "human-A", 250)
                with self.assertRaises(ValueError):
                    run.collect_reviews()
                write("reviewer2", "human-B", 249)
                result = run.collect_reviews()
                self.assertEqual(result["complete_sources"], 249)
                self.assertFalse(result["quality_pass"])
                write("reviewer2", "human-B", 250)
                self.assertTrue(run.collect_reviews()["quality_pass"])


if __name__ == "__main__":
    unittest.main()
