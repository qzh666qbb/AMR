"""Independent integrity and fixed-sample replay audit for staged transfer."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time


def load_transfer(out):
    spec = importlib.util.spec_from_file_location(
        "frozen_transfer_audit", out / "private/code/scripts/run_multiscale_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(root, out):
    m = load_transfer(out)
    freeze = m.read(out / "FREEZE.json")
    docid = freeze["split"]["precheck"][0]
    m.atomic(out / "AUDIT_PLAN.json", {
        "at": m.now(), "freeze_sha256": m.sha(out / "FREEZE.json"),
        "audit_source_sha256": m.sha(root / "scripts/audit_multiscale_transfer.py"),
        "samples": [
            {"doc_id": docid, "arm": "unwatermarked", "kind": "original_sentence", "index": 0},
            {"doc_id": docid, "arm": "positive", "kind": "S6", "index": 0},
            {"doc_id": docid, "arm": "positive", "kind": "S12", "index": 0}],
        "max_score_error": 1e-6, "require_exact_parse_sha": True,
        "all_input_and_model_sha": True, "all_features_and_predictions": True,
        "no_test_tuning": True})
    print("Audit plan frozen.")


def audit(root, out):
    import numpy as np
    import torch
    m = load_transfer(out)
    sys.path.insert(0, str(out / "private/code"))
    from src.core.multiscale_blind import assemble_features, blind_geometry
    from src.core.multiscale_fusion import MultiScaleAttentionFusion

    started = time.monotonic()
    audit_plan = m.read(out / "AUDIT_PLAN.json")
    freeze = m.read(out / "FREEZE.json")
    assert m.sha(root / "scripts/audit_multiscale_transfer.py") == audit_plan["audit_source_sha256"]
    assert audit_plan["freeze_sha256"] == m.sha(out / "FREEZE.json")
    m.verify(out, freeze)
    assert (out / "COMPLETE.json").exists() and not (out / "HALT.json").exists()
    assert not (out / "STOP.json").exists()
    completion = m.read(out / "COMPLETE.json")
    gate = m.read(out / "CLEAN_GATE.json")
    assert completion["attack_stage_executed"] == gate["passed"]
    documents = m.read(out / "private/DOCUMENTS.json")
    docs = {d["doc_id"]: d for d in documents}
    records = [m.read(p) for p in sorted((out / "private/records").glob("*.json"))]
    conditions = freeze["conditions"] if gate["passed"] else ["clean"]
    assert len(records) == len(docs) * len(conditions)
    assert {(r["doc_id"], r["condition"]) for r in records} == {
        (d, c) for d in docs for c in conditions}
    assert len(list((out / "private/signal").glob("*.json"))) == 24
    assert m.read(out / "SIGNAL_COMPLETE.json")["sentences"] == 384
    signal_time = m.read(out / "SIGNAL_COMPLETE.json")["at"]
    assert signal_time <= m.read(out / "CLEAN_COMPLETE.json")["at"] <= completion["at"]
    # The worker guards every feature call; also verify the on-disk chronology.
    signal_mtime = (out / "SIGNAL_COMPLETE.json").stat().st_mtime
    assert all(p.stat().st_mtime >= signal_mtime for p in (out / "private/records").glob("*.json"))
    tensor_file = torch.load(out / "private/CLEAN_FEATURES.pt", map_location="cpu", weights_only=False)
    tensor = tensor_file["scores"]
    assert tuple(tensor.shape) == (384, 3, 100)
    assert tensor_file["freeze_sha256"] == m.sha(out / "FREEZE.json")
    tensor_lookup = {(row["doc_id"], row["arm"], row["slot"]): i
                     for i, row in enumerate(tensor_file["metadata"])}
    assert len(tensor_lookup) == 384
    blind_file = torch.load(out / "private/BLIND_FEATURES.pt", map_location="cpu", weights_only=False)
    blind_lookup = {(row["doc_id"], row["condition"], row["arm"], row["anchor"]): i
                    for i, row in enumerate(blind_file["metadata"])}
    assert len(blind_lookup) == len(blind_file["scores"])
    expected_predictions = {
        (row["doc_id"], row["condition"], row["arm"], row["view"], row["method"]): row["statistics"]
        for row in m.read(out / "private/PREDICTIONS.json")}
    models = {}
    for row in freeze["checkpoints"]:
        checkpoint = torch.load(out / row["file"], map_location="cpu", weights_only=False)
        model = MultiScaleAttentionFusion(**checkpoint["config"]).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[row["name"]] = model
    torch.set_num_threads(2)
    max_prediction_error, cells_checked, anchors_checked = 0.0, 0, 0
    for record in records:
        docid, condition = record["doc_id"], record["condition"]
        assert record["freeze_sha256"] == m.sha(out / "FREEZE.json")
        for name in ("positive", "unwatermarked"):
            arm = record["arms"][name]
            assert arm["text_sha256"] == docs[docid]["arms"][name]["text_sha256"][condition]
            assert arm["geometry"] == blind_geometry(docs[docid]["arms"][name]["texts"][condition])
            cs = arm["candidate_scores"]
            blind = assemble_features(arm["geometry"], cs["S6"], cs["S12"], cs["Ssub"])
            assert blind == arm["blind_features"]
            for i, feature in enumerate(blind):
                index = blind_lookup[(docid, condition, name, i)]
                assert np.allclose(np.asarray(feature, dtype=float), blind_file["scores"][index].numpy(),
                                   rtol=0, atol=1e-7, equal_nan=True)
            anchors_checked += len(blind)
            if condition == "clean":
                signal = m.read(out / "private/signal" / f"d{docid:02d}.json")
                assert arm["official_sentence_scores"] == signal["arms"][name]["sentence_scores"]
                carrier = assemble_features(arm["geometry"], cs["S6"], cs["S12"], cs["Ssub"],
                                            docs[docid]["arms"][name]["carriers"])
                assert carrier == arm["carrier_features"] and len(carrier) == 8
                for i, feature in enumerate(carrier):
                    index = tensor_lookup[(docid, name, i)]
                    assert np.allclose(np.asarray(feature, dtype=float), tensor[index].numpy(),
                                       rtol=0, atol=1e-7, equal_nan=True)
            for scale, rows in cs.items():
                for row in rows:
                    assert len(row) == 100
                    assert all(v is None or (math.isfinite(v) and 0 <= v <= 1) for v in row)
                    cells_checked += len(row)
            for view in ("blind", "carrier"):
                if view + "_features" not in arm:
                    continue
                x = torch.tensor(np.asarray(arm[view + "_features"], dtype=float), dtype=torch.float32)
                with torch.no_grad():
                    for method, model in models.items():
                        values = model(x)["statistic"].tolist()
                        if view == "blind":
                            values = [m.mean_complete(values)]
                        expected = expected_predictions[(docid, condition, name, view, method)]
                        assert len(values) == len(expected)
                        for observed, prior in zip(values, expected):
                            if prior is None:
                                assert observed == -math.inf
                            else:
                                error = abs(observed - prior)
                                max_prediction_error = max(max_prediction_error, error)
                                assert error <= 1e-6
    parser, scorer = m.Parser(root, out, freeze), m.Scorer(root, out)
    replay = []
    for sample in audit_plan["samples"]:
        arm = docs[sample["doc_id"]]["arms"][sample["arm"]]
        if sample["kind"] == "original_sentence":
            text = arm["carriers"][sample["index"]]["text"]
            normalization = "official"
        else:
            text = blind_geometry(arm["texts"]["clean"])[sample["kind"]][sample["index"]]["text"]
            normalization = "project"
        prior = m.read(out / "private/parses" / (m.digest(text) + ".json"))
        fresh = parser.fresh(text)
        assert fresh["amr_sha256"] == prior["amr_sha256"] and fresh["valid"] == prior["valid"]
        score_error = None
        if fresh["valid"]:
            normalized = scorer.normalize(fresh["amr"], normalization)
            old = scorer.scores(prior["amr"], normalization)
            new = [scorer.pair(normalized, template) for template in scorer.bank]
            assert all((a is None) == (b is None) for a, b in zip(old, new))
            errors = [abs(a - b) for a, b in zip(old, new) if a is not None]
            score_error = max(errors, default=0.0)
            assert score_error <= 1e-6
        replay.append({**sample, "exact_parse_sha": True, "valid": fresh["valid"],
                       "max_score_error": score_error})
    parses = [m.read(p) for p in (out / "private/parses").glob("*.json")]
    assert all(p["freeze_sha256"] == m.sha(out / "FREEZE.json") for p in parses)
    assert not any(e.get("span_index") is None for p in parses for e in p["events"])
    result = {"passed": True, "at": m.now(), "freeze_sha256": m.sha(out / "FREEZE.json"),
              "records": len(records), "conditions": conditions, "clean_feature_shape": list(tensor.shape),
              "blind_feature_shape": list(blind_file["scores"].shape),
              "anchors_checked": anchors_checked, "candidate_score_cells_checked": cells_checked,
              "model_checkpoints_verified": len(models), "max_prediction_error": max_prediction_error,
              "fixed_sample_replays": replay, "unique_parses": len(parses),
              "invalid_parse_spans": sum(not p["valid"] for p in parses),
              "parse_spans_with_bound_events": sum(bool(p["events"]) for p in parses),
              "unbound_events": 0, "stage_order_verified": True,
              "seconds": time.monotonic() - started, "no_model_or_threshold_changes": True,
              "independent_human_test_fpr": None}
    m.atomic(out / "FINAL_AUDIT.json", result)
    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("plan", "audit"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "plan":
        plan(args.root.resolve(), args.out.resolve())
    else:
        audit(args.root.resolve(), args.out.resolve())
