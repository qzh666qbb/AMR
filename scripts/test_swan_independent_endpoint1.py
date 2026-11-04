import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swan_independent_endpoint1_core as core
import swan_independent_endpoint1_budget as budget
import run_swan_independent_endpoint1 as run
import collect_swan_independent_endpoint1_quality as human
from swan_final1_common import freeze
from swan_independent_calibration1_core import document_statistics, FLOWS
from swan_restricted_feedback1_server import session


def response(text="A researcher explained the result.", tokens=10):
    return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": tokens},
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def make_labels(path, items, reviewer, values=None):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "item_id", "packet_sha256", "reviewer_id", *human.FIELDS])
        writer.writeheader()
        for item in items:
            row = {"item_id": item["item_id"], "packet_sha256": "frozen", "reviewer_id": reviewer,
                   **dict.fromkeys(human.FIELDS, "true")}
            row.update((values or {}).get(item["item_id"], {}))
            writer.writerow(row)


class EndpointTests(unittest.TestCase):
    def test_assignments_use_all_sources_and_equal_slots(self):
        sources = [{"source_id": str(i)} for i in range(72)]
        ordered = core.order(sources)
        self.assertEqual(ordered, core.order(sources))
        self.assertEqual(len(ordered), 4608)
        for arm in core.ARMS:
            self.assertEqual(sum(r["arm"] == arm for r in ordered), 1152)

    def test_joint_balances_and_pooled_keeps_duplicate_capacity(self):
        banks = {"a": list(range(50)), "b": list(range(50))}
        keys = [core.template_schedule(banks, "s", "joint", slot)[0][0] for slot in range(16)]
        self.assertEqual(keys.count("a"), 8)
        self.assertEqual(keys.count("b"), 8)
        self.assertEqual(len(core.template_schedule(banks, "s", "pooled", 0)), 10)
        banks["b"] = list(range(49))
        with self.assertRaises(ValueError):
            core.template_schedule(banks, "s", "pooled", 0)

    def test_attack_prompts_are_sixteen_distinct_score_free_requests(self):
        texts = [json.dumps(core.attack_messages("A source paragraph.", i)) for i in range(16)]
        self.assertEqual(len(set(texts)), 16)
        self.assertTrue(all("A source paragraph." in x for x in texts))
        self.assertTrue(all("threshold" not in x and "AMR" not in x for x in texts))

    def test_abstention_and_disabled_threshold_labels(self):
        candidates = [{"candidate_id": str(i), "text": "text " + str(i)} for i in range(4)]
        scores = {str(i): core.client_score(None if i == 1 else float(i)) for i in range(4)}
        result = session(candidates, scores, .5, "label", 4)
        self.assertEqual(result["selection"]["selected"], "0")
        self.assertTrue(all(set(r["response"]) == {"label"} for r in result["transcript"]))
        self.assertEqual(session(candidates, scores, .5, "none", 4)["selection"]["queries"], 0)
        self.assertFalse(session(candidates, scores, core.client_threshold({"disabled": True}),
                                 "score", 4)["evaluation"]["detected"])

    def test_profiles_keep_request_tokens_and_costs_separate(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            for phase, expected in (("writer", 50), ("attack", 1600)):
                j = budget.journal(d, "contract", phase, lambda b: calls.append(b) or response(""))
                self.assertEqual(j.request("first", [])["text"], "")
                self.assertEqual(calls[-1]["max_tokens"], expected)
                restored = budget.journal(d, "contract", phase, lambda b: self.fail("redispatched"))
                restored.request("first", [])
                self.assertEqual(restored.totals()["calls"], 1)
            self.assertEqual(len(calls), 2)

    def test_unknown_cost_or_changed_response_blocks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            j = budget.journal(d, "contract", "writer", lambda b: response(tokens=51))
            with self.assertRaisesRegex(RuntimeError, "usage_exceeds"):
                j.request("bad", [])
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                budget.journal(d, "contract", "writer")

    def test_money_guard_precedes_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(budget.PROFILES, {"writer": (core.WRITER_PARAMETERS, 1, 1)}):
                j = budget.journal(d, "contract", "writer", lambda b: self.fail("money overrun"))
                with self.assertRaisesRegex(RuntimeError, "cost_cap"):
                    j.request("bad", [])
                self.assertEqual(j.totals()["calls"], 0)

    def test_missing_human_label_and_same_reviewer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.csv", Path(d) / "b.csv"
            items = [{"item_id": "x", "kind": "rewrite"}]
            make_labels(a, items, "R1")
            make_labels(b, items, "R1")
            with self.assertRaisesRegex(ValueError, "distinct"):
                human.resolve(items, "frozen", a, b)
            make_labels(b, items, "R2", {"x": {"preserves_meaning": ""}})
            with self.assertRaisesRegex(ValueError, "Missing"):
                human.resolve(items, "frozen", a, b)

    def test_disagreement_requires_third_person_and_preserves_false(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = [Path(d) / (x + ".csv") for x in ("a", "b", "c")]
            items = [{"item_id": "x", "kind": "rewrite"}]
            make_labels(a, items, "R1")
            make_labels(b, items, "R2", {"x": {"preserves_meaning": "false"}})
            labels, audit = human.resolve(items, "frozen", a, b)
            self.assertIsNone(labels)
            self.assertEqual(audit["status"], "ADJUDICATION_REQUIRED")
            make_labels(c, items, "R3", {"x": {"preserves_meaning": "false"}})
            labels, audit = human.resolve(items, "frozen", a, b, c)
            self.assertFalse(labels["x"])

    def test_asr_and_atr_keep_distinct_conditional_denominators(self):
        selections, mapping, quality = [], {}, {}
        for i, (clean, proxy_clean, attack, proxy_attack, valid) in enumerate((
                (True, True, False, False, True),
                (True, False, False, False, True),
                (True, True, False, False, False),
                (False, True, False, False, True))):
            key = str(i)
            selections.append({"role": "main", "arm": "a", "oracle": "public", "mode": "score",
                "budget": 16, "doc_id": key, "selection": {"selected": "0"},
                "clean_detected": {"a": clean, "public": proxy_clean},
                "attack_detected": {"a": attack, "public": proxy_attack}})
            for cid in (None, "0"):
                item = key + str(cid)
                mapping[item] = {"doc_id": key, "candidate_id": cid}
                quality[item] = valid if cid is not None else True
        rows = human.qualified_metrics(selections, mapping, quality)
        row = next(r for r in rows if (r["role"], r["arm"], r["oracle"], r["mode"], r["budget"])
                   == ("main", "a", "public", "score", 16))
        self.assertEqual((row["ASR"]["numerator"], row["ASR"]["denominator"]), (2, 3))
        self.assertEqual((row["ATR"]["numerator"], row["ATR"]["denominator"]), (1, 1))

    def test_row_checksum_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "row.json"
            run.save_row(path, {"source_id": "a", "value": 1})
            value = run.rt.read(path)
            value["row"]["value"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                run.read_row(path, {"source_id": "a"})

    def test_shared_positive_stats_match_calibration_erasures(self):
        rows = [{"public": .65, "a": .75, "b": .64}, {"public": None, "a": None, "b": None}]
        self.assertEqual(run.document_statistics(rows), document_statistics(rows))
        self.assertEqual(run.document_statistics(rows)["sentences"], 2)

    def test_full_writer_and_attack_resume_without_live_calls(self):
        import nltk
        nltk.data.path.insert(0, str(Path(__file__).resolve().parents[1] / ".nltk_data"))
        tok = nltk.tokenize.PunktTokenizer("english")
        class FakeParser:
            def __init__(self, *args):
                self.freeze_sha = run.rt.sha(args[1] / "FREEZE.json")
            def fresh(self, text):
                return dict.fromkeys(("text_sha256", "amr", "amr_sha256", "valid", "events", "attempts",
                                      "decode_events", "beam_lengths", "clipped", "token_length",
                                      "freeze_sha256"), "fixed")
        class FakeMatcher:
            def __init__(self, *args):
                self.db = SimpleNamespace(close=lambda: None)
            def fresh(self, *args):
                return 1.
            pair = fresh
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            freeze(out / "FREEZE.json", {"test_only": True})
            sources = [{"source_id": "pilot-0", "role": "pilot", "prompt": "A researcher worked."}]
            freeze(out / "private/SOURCES.json", sources)
            freeze(out / "private/ORDER.json", core.order(sources))
            root = Path(__file__).resolve().parents[1]
            for name in (run.OFFICIAL, run.EXAMPLES):
                target = out / "private/code" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((root / name).read_bytes())
            sent = []
            def send(body):
                sent.append(body)
                return response()
            banks = {"a": ["(x / V)"] * 50, "b": ["(x / V)"] * 50}
            with patch.object(run.rt, "Parser", FakeParser), \
                    patch.object(run.calibration, "Matcher", FakeMatcher), \
                    patch.object(run.construction, "Normalizer"), \
                    patch.object(run, "raw_template", return_value="(x / V)"), \
                    patch.object(run, "tokenizer", return_value=tok), \
                    patch.object(run, "sender", return_value=send):
                run.write_documents(out, out, out, banks, out / "unused", {})
                self.assertEqual(len(sent), 64)
                run.write_documents(out, out, out, banks, out / "unused", {})
                self.assertEqual(len(sent), 64)
                run.generate_attacks(out, out / "unused")
                self.assertEqual(len(sent), 128)
                run.generate_attacks(out, out / "unused")
                self.assertEqual(len(sent), 128)
                inputs = run.evaluation_inputs(out, out)
                self.assertEqual(len(inputs), 4 * 21)
                self.assertEqual(sum(r["condition"] == "paraphrase" for r in inputs), 64)
            thresholds = {condition: {flow: {"disabled": False, "threshold": 0.}
                                      for flow in FLOWS} for condition in run.CONDITIONS}
            freeze(out / "private/DETECTOR_BINDING.json", {"thresholds": thresholds})
            for doc in inputs:
                ident = run.rt.digest(doc["doc_id"] + "/" + doc["condition"] + "/" + str(doc["candidate_id"]))
                values = [{key: .9 if doc["condition"] == "clean" else .1 for key in run.BANK_KEYS}] * 2
                binding = {k: doc[k] for k in ("source_id", "role", "arm", "doc_id", "condition",
                                             "candidate_id", "text_sha256")}
                run.save_row(out / "private/rows" / (ident + ".json"), {
                    **binding, **document_statistics(values),
                    "logical_comparisons": dict.fromkeys(FLOWS, 100)})
            run.seal(out, "SCORING_COMPLETE.json", list((out / "private/rows").glob("*.json")))
            rows = run.select_attacks(out)
            self.assertEqual(run.check_seal(out, "SELECTIONS_COMPLETE.json")["sessions"], 72)
            run.review_packet(out)
            report = run.summarize(out, rows)
            self.assertEqual(report["writer_costs"]["joint"]["accepted_slots"], 16)
            self.assertIsNone(report["ASR"])
            items = run.rt.read(out / "private/quality/BLIND_ITEMS.json")
            forms = []
            for i in (1, 2):
                path = out / "private/quality" / ("filled" + str(i) + ".csv")
                make_labels(path, items, "human_" + str(i))
                path.write_text(path.read_text().replace("frozen", run.rt.sha(out / "private/quality/BLIND_ITEMS.json")))
                forms.append(path)
            result = human.collect(out, *forms, out / "human_result")
            self.assertEqual(result["status"], "HUMAN_QUALITY_COMPLETE")


if __name__ == "__main__":
    unittest.main()
