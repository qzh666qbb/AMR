"""Leakage, human provenance and full-denominator checks; fixtures are not results."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src/core")]
from src.core.semantic_supervision import (
    calibrate_rejection, choose_epoch, identity_metrics, normalize_document,
    require_human, split_scenes, validate_candidate, validate_inventory, validate_split,
)
from semantic_multiscale_supervised import SupervisedSymbolReadout, emit
from semantic_multiscale import scores_from_observation
import prepare_semantic_supervision2 as preparation
import collect_semantic_supervision2 as collection
import run_semantic_supervision2 as runner

CONTRACT = {"role_one": "Mina", "role_two": "Theo", "object": "box"}
TEXT = "Mina gave Theo the box. Theo received it. Mina walked."


def span(quote):
    start = TEXT.index(quote)
    return {"start": start, "end": start + len(quote), "quote": quote}


def event():
    return {"event_id": "e1", "kind": "carrier", "anchor": span("gave"),
            "meaning": "actual transfer", "sender": "Mina", "recipient": "Theo", "object": "box",
            "transfer_status": "completed", "readability": "clear", "explicit_status": "yes",
            "evidence": {"sender": [span("Mina")], "recipient": [span("Theo")],
                         "object": [span("box")], "transfer_status": [span("gave"), span("received")]},
            "background_reason": "", "notes": "Synthetic unit-test event, not human gold"}


def label(kind="carrier", anchor="gave"):
    s = span(anchor)
    return {"kind": kind, "event_ids": ["e1"], "evidence": [s],
            "anchor_span": [s["start"], s["end"]], "notes": "Synthetic unit-test linkage"}


def metric_doc(predictions, *, missing=0, truth="A"):
    return {"planned_symbol": truth, "predictions": predictions,
            "candidate_labels": [{"kind": "carrier"}, {"kind": "background"},
                                 {"kind": "duplicate"}, {"kind": "unknown"}][:len(predictions)],
            "references": [{"symbol": "A", "candidate_index": None if missing else 0}],
            "missing_events": missing}


class SupervisionTests(unittest.TestCase):
    def packet_fixture(self, folder):
        """Build a tiny explicit synthetic annotation fixture outside the workspace."""
        facts = {"item_id": "f1", "role_one": "Mina", "role_two": "Theo",
                 "public_object": "box", "text": TEXT, "sender": "Mina", "recipient": "Theo",
                 "object": "box", "transfer_status": "completed", "readability": "clear",
                 "explicit_status": "yes", "quality": "4", "notes": "Synthetic unit test only",
                 "evidence_json": json.dumps(event()["evidence"]),
                 "inventory_status": "complete", "events_json": json.dumps([event()])}
        candidate = {"item_id": "c1", "fact_item_id": "f1", "anchor_start": str(span("gave")["start"]),
                     "anchor_end": str(span("gave")["end"]), "anchor_text": "gave",
                     "context_1": TEXT, "context_2": TEXT, "context_3": TEXT,
                     "kind": "carrier", "event_ids_json": '["e1"]',
                     "evidence_json": json.dumps([span("gave")]), "notes": "Synthetic unit test"}
        pair = {"item_id": "p1", "source": TEXT, "paraphrase": TEXT,
                "facts_preserved": "yes", "explicit_status_preserved": "yes",
                "event_order_preserved": "yes",
                "evidence_json": json.dumps({"source": [span("gave")], "paraphrase": [span("gave")]}),
                "notes": "Synthetic unchanged pair"}
        preparation.write_csv(folder / "facts.csv", [facts])
        preparation.write_csv(folder / "candidates.csv", [candidate])
        preparation.write_csv(folder / "pairs.csv", [pair])
        mapping = {
            "facts": {"f1": {"key": "d1", "immutable": {k: facts[k] for k in
                                                       ("item_id", "role_one", "role_two", "public_object", "text")}}},
            "candidates": {"c1": {"key": "d1", "index": 0,
                                  "immutable": {k: candidate[k] for k in list(candidate)[:-4]}}},
            "pairs": {"p1": {"keys": ["d1", "d2"],
                             "immutable": {k: pair[k] for k in ("item_id", "source", "paraphrase")}}}}
        documents = {"d1": {"text": TEXT, "contract": CONTRACT,
                             "candidates": [{"anchor_span": label()["anchor_span"]}]}}
        return mapping, documents

    def test_collector_rejects_edited_text_and_inconsistent_event_inventory(self):
        with tempfile.TemporaryDirectory(prefix="amr-human-import-test-") as directory:
            folder = Path(directory)
            mapping, documents = self.packet_fixture(folder)
            normalized, pairs = collection.parse_packet(folder, mapping, documents)
            self.assertEqual(normalized["d1"]["normalized"]["references"][0]["symbol"], "A")
            self.assertEqual(len(pairs), 1)
            facts = preparation.csv_rows(folder / "facts.csv")
            facts[0]["events_json"] = "[]"
            preparation.write_csv(folder / "facts.csv", facts)
            with self.assertRaises(ValueError):
                collection.parse_packet(folder, mapping, documents)
            facts[0]["events_json"] = json.dumps([event()])
            facts[0]["text"] += " Edited."
            preparation.write_csv(folder / "facts.csv", facts)
            with self.assertRaises(ValueError):
                collection.parse_packet(folder, mapping, documents)

    def test_adjudication_contains_no_prior_answers_or_administrator_keys(self):
        with tempfile.TemporaryDirectory(prefix="amr-adjudication-test-") as directory:
            base = Path(directory)
            out, private = base / "public", base / "private"
            folder = out / "private_review/reviewer1"
            mapping, _ = self.packet_fixture(folder)
            preparation.atomic(folder / "SCHEMA.json", {"synthetic_unit_test": True})
            with patch.object(collection, "OUT", out), patch.object(collection, "PRIVATE", private):
                target, released = collection.make_adjudication({"1": mapping}, {"d1"}, set())
                self.assertTrue((private / "ADJUDICATION_KEYS.json").is_file())
                self.assertFalse((target / "KEYS.json").exists())
                self.assertEqual(preparation.csv_rows(target / "facts.csv")[0]["events_json"], "")
                self.assertEqual(preparation.csv_rows(target / "candidates.csv")[0]["kind"], "")
                self.assertEqual(preparation.csv_rows(target / "pairs.csv"), [])
                self.assertIn("a-f1", (target / "text_words.txt").read_text())
                self.assertEqual(released["facts"]["a-f1"]["key"], "d1")

    def test_scene_split_is_complete_disjoint_and_reproducible(self):
        first = split_scenes(range(24))
        self.assertEqual([len(first[k]) for k in ("train", "validation", "calibration", "test")],
                         [12, 4, 4, 4])
        self.assertEqual(first, split_scenes(reversed(range(24))))
        leaked = copy.deepcopy(first)
        leaked["calibration"][0] = leaked["validation"][0]
        with self.assertRaises(ValueError):
            validate_split(leaked, range(24))

    def test_human_attestation_does_not_accept_models_or_same_reviewer(self):
        meta = {"reviewer_type": "human", "reviewer_code": "unit-test-fixture",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "human_attestation": True, "independent_attestation": True,
                "no_model_assistance_attestation": True}
        self.assertEqual(require_human(meta), "unit-test-fixture")
        for changed in ({"reviewer_type": "model"}, {"human_attestation": False},
                        {"no_model_assistance_attestation": False}, {"completed_at": "invalid"}):
            with self.assertRaises(ValueError):
                require_human({**meta, **changed})
        with self.assertRaises(ValueError):
            require_human(meta, ["unit-test-fixture"])

    def test_actual_fact_needs_literal_evidence_and_explicit_state(self):
        validate_inventory(TEXT, CONTRACT, [event()], "complete")
        bad = event()
        bad["evidence"]["object"][0]["quote"] = "invented"
        with self.assertRaises(ValueError):
            validate_inventory(TEXT, CONTRACT, [bad], "complete")
        bad = {**event(), "explicit_status": "no"}
        with self.assertRaises(ValueError):
            validate_inventory(TEXT, CONTRACT, [bad], "complete")

    def test_unknown_and_unmatched_are_not_background(self):
        with self.assertRaises(ValueError):
            validate_candidate({**label(), "kind": "background"}, TEXT, [event()])
        with self.assertRaises(ValueError):
            validate_inventory(TEXT, CONTRACT, [{**event(), "kind": "background",
                                                 "background_reason": "unmatched"}], "complete")
        unknown = {**label(), "kind": "unknown", "event_ids": []}
        self.assertEqual(validate_candidate(unknown, TEXT, [event()])["kind"], "unknown")

    def test_primary_anchor_duplicate_and_missing_inventory_events(self):
        payload = {"text": TEXT, "contract": CONTRACT,
                   "candidates": [{"anchor_span": label()["anchor_span"]},
                                  {"anchor_span": label(anchor="received")["anchor_span"]}]}
        fact = {"events": [event()], "inventory_status": "complete"}
        normal = normalize_document(payload, fact, [label(), label("duplicate", "received")])
        self.assertEqual(normal["missing_events"], 0)
        self.assertEqual(normal["candidate_labels"][1]["symbol"], "A")
        payload["candidates"] = [payload["candidates"][1]]
        normal = normalize_document(payload, fact, [label("duplicate", "received")])
        self.assertEqual(normal["missing_events"], 1)
        self.assertIsNone(normal["references"][0]["candidate_index"])
        with self.assertRaises(ValueError):
            validate_candidate(label("carrier", "received"), TEXT, [event()])

    def test_all_candidates_require_labels(self):
        payload = {"text": TEXT, "contract": CONTRACT,
                   "candidates": [{"anchor_span": label()["anchor_span"]}]}
        with self.assertRaises(ValueError):
            normalize_document(payload, {"events": [event()], "inventory_status": "complete"}, [])

    def test_unknown_fact_cannot_have_asserted_evidence(self):
        unknown = event()
        unknown.update(kind="unknown", sender="unknown", readability="unclear")
        with self.assertRaises(ValueError):
            validate_inventory(TEXT, CONTRACT, [unknown], "complete")
        unknown["evidence"]["sender"] = []
        validate_inventory(TEXT, CONTRACT, [unknown], "complete")

    def test_rejection_missing_and_extra_outputs_stay_in_denominator(self):
        good = identity_metrics([metric_doc(["A", "A", "A", "A"])])
        self.assertEqual(good["tp"], [1, 0, 0, 0])
        self.assertEqual(good["fp"], [3, 0, 0, 0])
        self.assertEqual(good["background_emissions"], 1)
        self.assertEqual(good["unknown_emissions"], 1)
        self.assertEqual(good["duplicate_emissions"], 1)
        erased = identity_metrics([metric_doc([None, None, None, None], missing=1)])
        self.assertEqual(erased["fn"], [1, 0, 0, 0])
        self.assertEqual(erased["assigned_slots"], 1)
        self.assertEqual(erased["missing_events"], 1)
        wrong_write = identity_metrics([metric_doc(["A"], truth="B")])
        self.assertEqual(wrong_write["tp"], [0] * 4)
        self.assertEqual(wrong_write["fn"][1], 1)

    def test_epoch_selection_uses_full_identity_then_false_outputs(self):
        a = {"epoch": 10, "metrics": {"macro_f1": .3, "fp": [1, 0, 0, 0]}}
        b = {"epoch": 20, "metrics": {"macro_f1": .3, "fp": [0, 0, 0, 0]}}
        self.assertEqual(choose_epoch([a, b]), b)
        self.assertEqual(choose_epoch([b, {**b, "epoch": 40}]), b)

    def test_rejection_cannot_use_validation_or_test(self):
        record = {**metric_doc(["A"]), "role": "calibration"}
        outcome = calibrate_rejection([record], lambda r, p, c: ["A"], [(0., 0.)])
        self.assertFalse(outcome["checkpoint_updated"])
        self.assertFalse(outcome["error_rate_guarantee"])
        for role in ("train", "validation", "test"):
            with self.assertRaises(ValueError):
                calibrate_rejection([{**record, "role": role}], lambda r, p, c: ["A"], [(0., 0.)])

    def test_candidates_use_text_only_context_and_whole_document_budget(self):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        source = preparation.context_candidates(TEXT, nlp)
        self.assertGreater(len(source["observations"]), 0)
        for observation in source["observations"]:
            a, b = observation["anchor_span"]
            self.assertEqual(observation["anchor_text"], TEXT[a:b])
            for window in observation["contexts"]:
                lo, hi = window["span"]
                self.assertEqual(window["text"], TEXT[lo:hi])
                self.assertTrue(lo <= a < b <= hi)
        # 33 separately repeated finite verbs trigger whole-document abstention.
        over = preparation.context_candidates(" ".join(["Mina walked."] * 33), nlp)
        self.assertEqual(over["status"], "budget_abstain")
        self.assertEqual(over["observations"], [])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        data = json.loads((Path.home() / ".cache/amr/semantic_channel_audit1/predictions"
                           / "d12-positive-clean.json").read_text())
        cls.x = torch.tensor([scores_from_observation(o) for o in data["observations"]])
        cls.groups = [list(range(c * 6, (c + 1) * 6)) for c in range(4)]

    def test_real_cache_backward_and_unknown_supervision(self):
        for mode in runner.MODES:
            model = SupervisedSymbolReadout(self.groups, mode)
            model.fit_normalizer(self.x)
            kinds = torch.tensor([0, 1, 2, 3])
            symbols = torch.tensor([0, -1, 1, -1])
            loss = model.loss(self.x[:4], kinds, symbols)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(model.configuration()["event_parameters"], 40)
            # Unknown/background symbol annotations cannot influence the loss.
            changed = torch.tensor([0, 3, 1, 2])
            self.assertTrue(torch.equal(model.loss(self.x[:4], kinds, symbols),
                                        model.loss(self.x[:4], kinds, changed)))

    def test_no_target_input_all_missing_and_checkpoint_replay(self):
        model = SupervisedSymbolReadout(self.groups, "attention")
        model.fit_normalizer(self.x)
        with self.assertRaises(TypeError):
            model(self.x, gold_symbol=0)
        missing = torch.full_like(self.x[:1], float("nan"))
        self.assertEqual(emit(model(missing)), [None])
        clone = SupervisedSymbolReadout(self.groups, "attention")
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(model(self.x)["q"], clone(self.x)["q"]))

    def test_selection_and_rejection_ties_never_guess_a_symbol(self):
        output = {"q": torch.tensor([[.7, .1, .1, .1], [.25] * 4]),
                  "event_q": torch.tensor([[.9, .04, .03, .03], [.9, .04, .03, .03]]),
                  "abstain": torch.tensor([False, False])}
        self.assertEqual(emit(output), ["A", None])
        self.assertEqual(emit(output, 1.01), [None, None])
        output["event_q"][0] = torch.tensor([.5, .5, 0., 0.])
        self.assertEqual(emit(output), [None, None])

    def test_four_phase_label_access_and_seals_in_temporary_fixture(self):
        """Synthetic engineering fixture in /tmp, never an experimental dataset."""
        with tempfile.TemporaryDirectory(prefix="amr-supervision-stage-test-") as directory:
            base = Path(directory)
            out, private = base / "public", base / "private"
            preparation.atomic(out / "FREEZE.json", {"engineering_fixture_only": True})
            groups = split_scenes(range(24))
            preparation.atomic(out / "SPLITS.json", {"groups": groups})
            membership = {f"d{scene:02d}": scene for scene in range(24)}
            preparation.atomic(out / "MEMBERSHIP.json", membership)
            preparation.atomic(private / "FEATURE_REQUEST_NO_TARGET.json",
                               {"documents": [{"key": key} for key in membership]})
            labels_sha = {role: {} for role in groups}
            features_sha = {}
            rng = np.random.default_rng(20261026)
            for key, scene in membership.items():
                role = next(role for role, scenes in groups.items() if scene in scenes)
                row = metric_doc(["A", None])
                row.pop("predictions")
                row.update(role=role, scene_id=scene, condition="original")
                row["candidate_labels"] = [{"kind": "carrier", "symbol": "A"},
                                           {"kind": "background", "symbol": None}]
                path = private / "labels" / role / (key + ".json")
                preparation.atomic(path, row)
                labels_sha[role][key] = preparation.sha(path)
                path = private / "features" / (key + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, x=rng.uniform(0, 1, (2, 3, 24)).astype(np.float32))
                features_sha[key] = preparation.sha(path)
            preparation.atomic(private / "HUMAN_GOLD_SEAL.json", {
                "engineering_fixture_only": True, "real_humans": True,
                "training_labels_ready": True, "role_label_sha256": labels_sha,
                "parent_freeze_sha256": preparation.sha(out / "FREEZE.json")})
            preparation.atomic(out / "FEATURE_SEAL.json", {
                "engineering_fixture_only": True, "feature_sha256": features_sha,
                "candidate_counts": dict.fromkeys(membership, 2), "groups": self.groups,
                "metadata_sha256": {}, "parent_freeze_sha256": preparation.sha(out / "FREEZE.json")})
            accessed = set()
            original_load = runner.load

            def tracked_load(path):
                path = Path(path)
                if "labels" in path.parts:
                    accessed.add(path.parent.name)
                return original_load(path)

            with patch.object(runner, "OUT", out), patch.object(runner, "PRIVATE", private), \
                 patch.object(runner, "verify", lambda: {}), patch.object(runner, "EPOCHS", (1, 2)), \
                 patch.object(runner, "GRID", [(0., 0.)]), patch.object(runner, "load", tracked_load):
                runner.select()
                self.assertEqual(accessed, {"train", "validation"})
                models_before = {m: preparation.sha(private / "models" / (m + ".pt")) for m in runner.MODES}
                accessed.clear()
                runner.calibrate()
                self.assertEqual(accessed, {"calibration"})
                self.assertEqual(models_before, {m: preparation.sha(private / "models" / (m + ".pt"))
                                                for m in runner.MODES})
                accessed.clear()
                runner.predict()
                self.assertEqual(accessed, set())
                runner.evaluate()
                self.assertEqual(accessed, {"test"})
                with self.assertRaises(RuntimeError):
                    runner.select()
                # A post-seal checkpoint modification must fail before reusing it.
                with (private / "models/gate.pt").open("ab") as stream:
                    stream.write(b"drift")
                with self.assertRaises(RuntimeError):
                    runner.selected_model("gate")


if __name__ == "__main__":
    unittest.main()
