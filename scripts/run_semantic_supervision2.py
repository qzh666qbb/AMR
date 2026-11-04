"""Separate select/calibrate/predict/evaluate processes on a newly sealed fact task."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from prepare_semantic_supervision2 import OUT, PRIVATE, atomic, digest, load, sha, verify
from src.core.semantic_supervision import (
    KINDS, ROLES, calibrate_rejection, choose_epoch, identity_metrics, validate_split,
)
from semantic_multiscale_supervised import SupervisedSymbolReadout, emit

MODES = ("mean", "gate", "gate_wide", "attention")
EPOCHS = (10, 20, 40)
GRID = [(p, c) for p in (0., .25, .5, .75, 1.01) for c in (0., .5, .75, 1.01)]


def role_keys(role):
    """Labels are physically separated: fit never opens calibration/test label files."""
    groups = load(OUT / "SPLITS.json")["groups"]
    documents = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")["documents"]
    # Scene membership is sealed separately from feature text and labels.
    membership = load(OUT / "MEMBERSHIP.json")
    return [d["key"] for d in documents if membership[d["key"]] in groups[role]]


def gold_seal():
    seal = load(PRIVATE / "HUMAN_GOLD_SEAL.json")
    if not seal.get("real_humans") or not seal.get("training_labels_ready"):
        raise RuntimeError("Independent complete human labels and original carrier gate required")
    if seal["parent_freeze_sha256"] != sha(OUT / "FREEZE.json"):
        raise RuntimeError("Human gold belongs to another input freeze")
    # Verify only the requested role's labels below, never open the aggregate
    # administrator gold (which also contains calibration/test truth).
    return seal


def role_labels(role):
    if role not in ROLES:
        raise ValueError("Unknown data role")
    seal = gold_seal()
    expected = set(role_keys(role))
    if set(seal["role_label_sha256"][role]) != expected:
        raise RuntimeError("Label membership mismatch")
    rows = {}
    for key in sorted(expected):
        path = PRIVATE / "labels" / role / (key + ".json")
        if sha(path) != seal["role_label_sha256"][role][key]:
            raise RuntimeError("Role label SHA drift")
        row = load(path)
        if row["role"] != role:
            raise RuntimeError("Label-role contamination")
        rows[key] = row
    return rows


def feature_seal():
    seal = load(OUT / "FEATURE_SEAL.json")
    if seal["parent_freeze_sha256"] != sha(OUT / "FREEZE.json"):
        raise RuntimeError("Feature freeze does not belong to this protocol")
    for name, expected in seal["metadata_sha256"].items():
        if sha(PRIVATE / name) != expected:
            raise RuntimeError("Feature dictionary/evidence metadata drift")
    return seal


def feature(key, seal):
    path = PRIVATE / "features" / (key + ".npz")
    if sha(path) != seal["feature_sha256"][key]:
        raise RuntimeError("Feature SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["x"]:
            raise ValueError("Feature arrays may contain only score x; labels forbidden")
        array = archive["x"].copy()
    if array.dtype != np.float32 or array.shape != (seal["candidate_counts"][key], 3, 24):
        raise ValueError("Shared candidate/three-view dictionary shape mismatch")
    if np.isinf(array).any() or ((array[np.isfinite(array)] < 0) | (array[np.isfinite(array)] > 1)).any():
        raise ValueError("Finite scores must lie in [0,1]; absent evidence is NaN")
    return torch.from_numpy(array)


def readiness():
    verify()
    reasons = []
    if not (PRIVATE / "HUMAN_GOLD_SEAL.json").exists():
        reasons.append("independent_human_facts_background_duplicate_unknown_not_returned")
    else:
        try:
            gold_seal()
        except RuntimeError as error:
            reasons.append(str(error))
    if not (OUT / "FEATURE_SEAL.json").exists():
        reasons.append("new_fact_templates_core_matcher_and_R_context_features_not_sealed")
    else:
        feature_seal()
    outcome = {"status": "READY_FOR_MODEL_SELECTION" if not reasons else "NOT_READY",
               "reasons": reasons, "model_selection_uses": ["train", "validation"],
               "rejection_uses": ["calibration"], "prediction_uses_test_labels": False,
               "training_runs": int((OUT / "MODEL_SELECTION_SEAL.json").exists()) * len(MODES)}
    atomic(OUT / "READINESS.json", outcome)
    return outcome


def seal_features():
    verify()
    gold_seal()
    if (OUT / "FEATURE_SEAL.json").exists():
        raise RuntimeError("Feature release immutable; use a new version")
    metadata = load(PRIVATE / "FEATURE_METADATA.json")
    request = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")
    if (metadata["alphabet_version"] != request["alphabet_version"]
            or metadata["views"] != request["views"]
            or metadata["template_semantics"] != "new_transfer_direction_status"
            or metadata["candidate_source_sha256"] != sha(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")
            or metadata["core_matcher_ready"] is not True):
        raise RuntimeError("Old structure alphabet and R-short cache cannot be relabelled as facts")
    train = load(OUT / "SPLITS.json")["groups"]["train"]
    bank = load(PRIVATE / "TEMPLATE_BANK.json")
    if sorted(bank["source_scene_ids"]) != sorted(train):
        raise RuntimeError("Templates must come exclusively from the frozen training scenes")
    if len(bank["templates"]) != 24 or metadata["groups"] != bank["groups"]:
        raise RuntimeError("Four equally-sized groups must partition 24 new templates")
    # Constructor validates exact group partition; no model is fitted here.
    SupervisedSymbolReadout(bank["groups"], "mean")
    evidence = load(PRIVATE / "MATCH_EVIDENCE.json")
    if set(evidence) != {d["key"] for d in request["documents"]}:
        raise RuntimeError("All text versions require common graph/mapping evidence")
    counts = {d["key"]: len(d["candidates"]) for d in request["documents"]}
    for key, count in counts.items():
        if len(evidence[key]) != count:
            raise RuntimeError("Missing candidate match evidence")
        for row in evidence[key]:
            if (row.get("shared_core_policy") is not True
                    or row.get("status") not in ("complete", "invalid", "budget_abstain")
                    or not row.get("graph_sha256") or not row.get("mapping_bank_sha256")):
                raise RuntimeError("Original graph/common core-validity and search evidence required")
    trial = {"candidate_counts": counts,
             "feature_sha256": {key: sha(PRIVATE / "features" / (key + ".npz")) for key in counts},
             "groups": metadata["groups"],
             "metadata_sha256": {name: sha(PRIVATE / name) for name in
                                 ("FEATURE_METADATA.json", "TEMPLATE_BANK.json", "MATCH_EVIDENCE.json")},
             "parent_freeze_sha256": sha(OUT / "FREEZE.json"),
             "template_bank_sha256": sha(PRIVATE / "TEMPLATE_BANK.json"),
             "new_fact_task": True}
    for key in counts:
        feature(key, trial)
    atomic(OUT / "FEATURE_SEAL.json", trial)
    return {"status": "FEATURES_SEALED", "documents": len(counts), "inference_executed_here": False}


def predict_records(model, keys, features):
    model.eval()
    result = {}
    with torch.no_grad():
        for key in keys:
            x = features[key]
            if len(x) == 0:
                result[key] = {"q": torch.zeros((0, 4)), "event_q": torch.zeros((0, 4)),
                               "abstain": torch.zeros(0, dtype=torch.bool)}
            else:
                out = model(x)
                result[key] = {k: out[k].detach() for k in ("q", "event_q", "abstain")}
    return result


def select():
    if readiness()["status"] != "READY_FOR_MODEL_SELECTION":
        raise RuntimeError("Human/feature prerequisites not satisfied; no training")
    if (OUT / "MODEL_SELECTION_SEAL.json").exists():
        raise RuntimeError("Selected models immutable; use a new experiment")
    torch.set_num_threads(1)
    training, validation = role_labels("train"), role_labels("validation")
    seal = feature_seal()
    features = {key: feature(key, seal) for key in (*training, *validation)}
    rows = [(key, value) for key, value in training.items() if len(features[key])]
    if not rows:
        raise RuntimeError("No training observations; full cohort remains unscored")
    all_x = torch.cat([features[key] for key, _ in rows])
    supervision = {}
    for key, value in rows:
        kinds = torch.tensor([KINDS.index(r["kind"]) for r in value["candidate_labels"]])
        symbols = torch.tensor(["ABCD".index(r["symbol"]) if r["symbol"] else -1
                                for r in value["candidate_labels"]])
        supervision[key] = (kinds, symbols)
    counts = {kind: sum(r["kind"] == kind for v in training.values()
                       for r in v["candidate_labels"]) for kind in KINDS}
    if not counts["carrier"] or not counts["background"]:
        raise RuntimeError("Both human-confirmed carriers and backgrounds required in training")
    selections, hashes = {}, {}
    for mode in MODES:
        torch.manual_seed(20261026)
        model = SupervisedSymbolReadout(seal["groups"], mode)
        model.fit_normalizer(all_x)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=.01, weight_decay=.01)
        curve, best_state, best = [], None, None
        for epoch in range(1, max(EPOCHS) + 1):
            model.train()
            optimizer.zero_grad()
            losses = [model.loss(features[key], *supervision[key]) for key, _ in rows]
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; no checkpoint published")
            loss.backward()
            optimizer.step()
            if epoch not in EPOCHS:
                continue
            output = predict_records(model, validation, features)
            docs = [{**v, "predictions": emit(output[key])} for key, v in validation.items()]
            entry = {"epoch": epoch, "metrics": identity_metrics(docs)}
            curve.append(entry)
            chosen = choose_epoch(curve)
            if best != chosen:
                best = copy.deepcopy(chosen)
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        path = PRIVATE / "models" / (mode + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        torch.save({"state_dict": best_state, "configuration": model.configuration()}, path)
        path.chmod(0o600)
        hashes[mode] = sha(path)
        selections[mode] = {"chosen": best, "validation_curve": curve,
                            "loss": "equal_document_mean(event_CE + conditional_carrier_duplicate_CE)",
                            "supervision_counts": counts, "configuration": model.configuration()}
    atomic(OUT / "MODEL_SELECTION.json", selections)
    atomic(OUT / "MODEL_SELECTION_SEAL.json", {
        "models_sha256": hashes, "selection_sha256": sha(OUT / "MODEL_SELECTION.json"),
        "feature_seal_sha256": sha(OUT / "FEATURE_SEAL.json"),
        "gold_seal_sha256": sha(PRIVATE / "HUMAN_GOLD_SEAL.json"),
        "split_sha256": sha(OUT / "SPLITS.json"),
        "label_roles_read": ["train", "validation"],
        "calibration_labels_read": False, "test_labels_read": False,
        "epochs": EPOCHS, "configurations_per_mode": len(EPOCHS)})
    return {"status": "MODELS_SELECTED_AND_SEALED", "models": len(MODES),
            "calibration_used": False, "test_used": False}


def selected_model(mode):
    selection = load(OUT / "MODEL_SELECTION_SEAL.json")
    for name, expected in (("MODEL_SELECTION.json", selection["selection_sha256"]),
                           ("FEATURE_SEAL.json", selection["feature_seal_sha256"]),
                           ("SPLITS.json", selection["split_sha256"])):
        if sha(OUT / name) != expected:
            raise RuntimeError("Selection input/configuration drift")
    if sha(PRIVATE / "HUMAN_GOLD_SEAL.json") != selection["gold_seal_sha256"]:
        raise RuntimeError("Selection gold seal drift")
    path = PRIVATE / "models" / (mode + ".pt")
    if sha(path) != selection["models_sha256"][mode]:
        raise RuntimeError("Selected checkpoint changed")
    data = torch.load(path, weights_only=True)
    model = SupervisedSymbolReadout(data["configuration"]["groups"], mode)
    model.load_state_dict(data["state_dict"])
    return model


def calibrate():
    verify()
    if (OUT / "REJECTION_SEAL.json").exists():
        raise RuntimeError("Rejection calibration immutable")
    selection_sha = sha(OUT / "MODEL_SELECTION_SEAL.json")
    calibration = role_labels("calibration")
    features = {key: feature(key, feature_seal()) for key in calibration}
    result = {}
    for mode in MODES:
        model = selected_model(mode)
        before = digest({k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})
        outputs = predict_records(model, calibration, features)
        records = [{**v, "key": key} for key, v in calibration.items()]
        result[mode] = calibrate_rejection(records,
            lambda r, p, c: emit(outputs[r["key"]], p, c), GRID)
        after = digest({k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})
        if before != after:
            raise RuntimeError("Calibration changed model weights")
    atomic(OUT / "REJECTION.json", result)
    atomic(OUT / "REJECTION_SEAL.json", {
        "selection_seal_sha256": selection_sha, "rejection_sha256": sha(OUT / "REJECTION.json"),
        "label_roles_read": ["calibration"], "models_updated": False,
        "model_configurations_plus_rejection_rules_per_mode": len(EPOCHS) + len(GRID),
        "test_labels_read": False})
    return {"status": "REJECTION_SEALED", "models_changed": False, "rules_per_mode": len(GRID)}


def predict():
    verify()
    if (OUT / "PREDICTION_SEAL.json").exists():
        raise RuntimeError("Test predictions immutable")
    reject_seal = load(OUT / "REJECTION_SEAL.json")
    if (sha(OUT / "REJECTION.json") != reject_seal["rejection_sha256"]
            or sha(OUT / "MODEL_SELECTION_SEAL.json") != reject_seal["selection_seal_sha256"]):
        raise RuntimeError("Rejection/selection seal drift")
    rejection = load(OUT / "REJECTION.json")
    keys = role_keys("test")  # Opens scene membership and target-free inputs only.
    features = {key: feature(key, feature_seal()) for key in keys}
    hashes = {}
    for mode in MODES:
        outputs = predict_records(selected_model(mode), keys, features)
        rule = rejection[mode]["selection"]
        records = {key: {"raw": emit(out),
                         "calibrated": emit(out, rule["presence"], rule["confidence"])}
                   for key, out in outputs.items()}
        path = PRIVATE / "predictions" / (mode + ".json")
        atomic(path, records)
        hashes[mode] = sha(path)
    atomic(OUT / "PREDICTION_SEAL.json", {
        "sha256": hashes, "rejection_seal_sha256": sha(OUT / "REJECTION_SEAL.json"),
        "test_labels_read": False, "documents": len(keys),
        "all_shared_candidates_retained": True, "message_recovery_evaluated": False})
    return {"status": "TEST_PREDICTIONS_SEALED", "documents": len(keys), "test_labels_read": False}


def evaluate():
    verify()
    if (OUT / "RESULT.json").exists():
        raise RuntimeError("Evaluation immutable")
    seal = load(OUT / "PREDICTION_SEAL.json")
    if seal["test_labels_read"] or sha(OUT / "REJECTION_SEAL.json") != seal["rejection_seal_sha256"]:
        raise RuntimeError("Invalid prediction seal")
    labels = role_labels("test")
    results = {}
    for mode in MODES:
        path = PRIVATE / "predictions" / (mode + ".json")
        if sha(path) != seal["sha256"][mode]:
            raise RuntimeError("Test prediction changed")
        predictions = load(path)
        if set(predictions) != set(labels):
            raise RuntimeError("Test documents dropped or added")
        results[mode] = {}
        for condition in ("original", "paraphrase"):
            results[mode][condition] = {}
            for rule in ("raw", "calibrated"):
                rows = [{**v, "predictions": predictions[key][rule]}
                        for key, v in labels.items() if v["condition"] == condition]
                results[mode][condition][rule] = identity_metrics(rows)
    atomic(OUT / "RESULT.json", {"results": results,
                               "scope": "four previously analysed development scenes",
                               "independent_confirmation": False, "message_recovery_evaluated": False})
    return {"status": "IDENTITY_EVALUATED", "message_recovery_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "seal-features", "select", "calibrate", "predict", "evaluate"))
    args = parser.parse_args()
    commands = {"check": readiness, "seal-features": seal_features, "select": select,
                "calibrate": calibrate, "predict": predict, "evaluate": evaluate}
    print(json.dumps(commands[args.command](), ensure_ascii=False, indent=2))
