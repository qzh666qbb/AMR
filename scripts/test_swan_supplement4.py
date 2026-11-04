"""No-transport regression tests for the successor's costly failure boundaries."""
import json
import logging
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import run_swan_supplement4 as run
from swan_supplement4_runtime import CandidateReject, ResourcePause, observed
from swan_final1_common import freeze, sha256


def payload():
    return {"id": "test-response", "model": run.budget.MODEL,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [{"message": {"content": "A scientist carefully tested the small sample."},
                         "finish_reason": "stop"}]}


class Supplement4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        freeze(self.out / "FREEZE.json", {"test": "no paid calls"})
        self.bindings = [patch.object(run, "OUT", self.out),
                         patch.object(run.pilot, "OUT", self.out)]
        for p in self.bindings:
            p.start()
    def tearDown(self):
        for p in reversed(self.bindings):
            p.stop()
        self.temp.cleanup()
    def test_paid_replay_and_request_mutation(self):
        previous = self.out / "parent"
        freeze(previous / "FREEZE.json", {"parent": True})
        sent = []
        j = run.budget.Journal(previous, "pilot", 6400,
                               lambda body: sent.append(body) or payload(),
                               sha256(previous / "FREEZE.json"))
        expected = j.request("call-1", "prompt", "system", 50, .7, .9)
        replay = run.ReplayJournal(self.out, "pilot", 6400,
                                   lambda body: self.fail("Unexpected transport"),
                                   sha256(self.out / "FREEZE.json"), parents=[previous])
        for _ in range(2):
            self.assertEqual(replay.request("call-1", "prompt", "system", 50, .7, .9), expected)
        self.assertEqual(replay.totals()["actual_peak_nanodollars"], 15000)
        self.assertEqual(replay.totals()["calls"], 1)
        self.assertEqual(len(sent), 1)
        with self.assertRaises(run.IntegrityHalt):
            replay.request("call-1", "changed", "system", 50, .7, .9)
    def test_uncertain_request_never_redispatched(self):
        j = run.ReplayJournal(self.out, "pilot", 6400,
                               lambda body: (_ for _ in ()).throw(TimeoutError()),
                               sha256(self.out / "FREEZE.json"), parents=[])
        with self.assertRaises(run.budget.DocumentFailure):
            j.request("call-2", "prompt", "system", 50, .7, .9)
        with self.assertRaises(run.IntegrityHalt):
            j.request("call-2", "prompt", "system", 50, .7, .9)
        self.assertEqual(j.totals()["calls"], 1)
        self.assertGreater(j.totals()["exposure_nanodollars"], 15000)
    def test_scoped_warning_vs_unbound_score(self):
        def parser(text):
            logging.warning("gid=x Start paren present but page is not a new concept")
        with self.assertRaises(CandidateReject):
            observed(parser, self.out, "call-3")("private text")
        with self.assertRaises(run.IntegrityHalt):
            observed(parser, self.out)("private text")
    def test_resume_after_paid_response_before_parse(self):
        sent, evaluated = [], []
        j = run.ReplayJournal(self.out, "pilot", 6400,
                               lambda body: sent.append(body) or payload(),
                               sha256(self.out / "FREEZE.json"), parents=[])
        module = run.pilot.swan.official_function_module_for_test({})
        def evaluate(text, target, call_id):
            evaluated.append(call_id)
            if len(evaluated) == 1:
                raise ResourcePause("simulated_cuda_pressure")
            if len(evaluated) == 2:
                raise CandidateReject("invalid_parse")
            return {"accepted": True, "score": .8, "rejection": None}
        module.evaluate = evaluate
        source = {"allocation_id": "unit-test"}
        with self.assertRaises(ResourcePause):
            run.run_slot(module, source, 0, "context", ["(x / test)"], j)
        resumed = run.ReplayJournal(self.out, "pilot", 6400,
                                     lambda body: sent.append(body) or payload(),
                                     sha256(self.out / "FREEZE.json"), parents=[])
        result = run.run_slot(module, source, 0, "context", ["(x / test)"], resumed)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(sent), 2)
        self.assertEqual(resumed.totals()["calls"], 2)
        self.assertEqual(len(list((self.out / "private/attempts").glob("*.json"))), 2)
    def test_request_sequence_matches_official_loop(self):
        sent = []
        j = run.ReplayJournal(self.out, "pilot", 6400,
                               lambda body: sent.append(body) or payload(),
                               sha256(self.out / "FREEZE.json"), parents=[])
        module = run.pilot.swan.official_function_module_for_test({})
        module.evaluate = lambda *args: {"accepted": False, "score": .2, "rejection": None}
        bank = ["(x / alpha)", "(x / beta)", "(x / gamma)"]
        run.run_slot(module, {"allocation_id": "parity"}, 0, "context", bank, j)
        original = []
        class Manager:
            def __init__(self, **kwargs):
                pass
            def generate(self, user_text, system_text, max_tokens, temperature, top_p):
                original.append(run.budget.body_for(user_text, system_text, max_tokens, temperature, top_p))
                return payload()["choices"][0]["message"]["content"]
        official = run.pilot.swan.official_function_module_for_test({
            "BedrockManager": Manager, "parse_amr": lambda text: "(x / text)",
            "normalize_amr_variables": lambda graph: graph,
            "generate_template_amr": lambda graph: graph,
            "decide_s2match_match": lambda *args, **kwargs: (False, .2)})
        build = official.build_user_prompt
        official.build_user_prompt = lambda *a, **k: build(*a, **k) + run.pilot.LENGTH
        official.random = random.Random(run.content_seed("template", "parity", 0))
        official.amr_instruction_completion(
            context="context", amr_bank=bank, model_id=run.budget.MODEL, region=run.budget.URL,
            max_new_tokens=50, temperature=.7, top_p=.9, max_trials=50,
            amr_examples_path=str(run.pilot.swan.OFFICIAL_ROOT /
                                 "amr_bank/artifacts/normalized_amr_examples.json"),
            s2match_threshold=.75, max_templates_to_try=10, max_retries_per_template=5)
        self.assertEqual(sent, original)
        self.assertEqual(len(sent), 50)


if __name__ == "__main__":
    unittest.main()
