"""Reproducible, grouped post-hoc readback evaluation on sealed real caches.

Phases are separate processes. Predict reads feature arrays and checkpoints,
never reference/assignment files. Every held-out model excludes its test group.
This is development cross-validation with constructed/model-assisted labels.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semantic_multiscale_readback1"
PRIVATE = Path.home() / ".cache/amr/semantic_multiscale_readback1"
PRIOR = Path.home() / ".cache/amr/semantic_sequence_readout1"
CACHE = Path.home() / ".cache/amr/semantic_channel_audit1"
OLD = ROOT / "experiments/semantic_sequence_readout1"
sys.path[:0] = [str(ROOT / "src/core"), str(ROOT / "scripts")]
from semantic_multiscale import MultiScaleSymbolReadout, MODES, scores_from_observation
from semantic_sequence import exact_costs, rank_truth
from semantic_cycle2 import diagnostic, identity, match_positions
from audit_semantic_readout import q_from_raw, full_dp, independent_diagnostic

SEED = 20261026
ALL_MODES = ("legacy_W", "legacy_E1", "legacy_C", *MODES)
CONDITIONS = ("clean", "paraphrase", "split", "merge")
SOURCE_FILES = (
    "scripts/run_semantic_multiscale_readback.py", "src/core/semantic_multiscale.py",
    "src/core/multiscale_fusion.py", "src/core/semantic_cycle2.py",
    "src/core/semantic_sequence.py", "scripts/audit_semantic_readout.py",
    "scripts/semantic_cycle2_gpu.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def heartbeat(phase, completed, total, started):
    elapsed = time.monotonic() - started
    atomic(OUT / "heartbeat.json", {
        "phase": phase, "completed": completed, "total": total,
        "elapsed_seconds": elapsed, "at": now()})
    print(json.dumps({"phase": phase, "completed": completed, "total": total,
                      "elapsed_seconds": round(elapsed, 2)}), flush=True)
    if elapsed > 3600:
        raise RuntimeError("Phase CPU budget exceeded")


def candidates():
    ids = np.arange(4 ** 8, dtype=np.int32)
    return ((ids[:, None] // (4 ** np.arange(7, -1, -1))) % 4).astype(np.int8)


def retained(q, threshold):
    return q if identity(q, threshold) is not None else None


def q_list(array):
    values = []
    for row in array:
        if not np.isfinite(row).all():
            values.append(None)
        else:
            v = np.asarray(row, dtype=np.float64)
            values.append((v / v.sum()).tolist())
    return values


def verify():
    frozen = read(OUT / "FREEZE.json")
    for name, expected in frozen["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Source SHA changed: " + name)
    if sha(OUT / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("Protocol changed after freeze")
    for name, expected in frozen["old_public_sha256"].items():
        if sha(OLD / name) != expected:
            raise RuntimeError("Original public artifact changed: " + name)
    for key, expected in frozen["old_prediction_sha256"].items():
        if sha(CACHE / "predictions" / (key + ".json")) != expected:
            raise RuntimeError("Original sealed prediction changed")
    for key, expected in frozen["feature_sha256"].items():
        if sha(PRIVATE / "features" / (key + ".npz")) != expected:
            raise RuntimeError("Sanitized feature cache changed")
    if sha(OUT / "SPLITS.json") != frozen["splits_sha256"]:
        raise RuntimeError("Split manifest changed")
    return frozen


def labels(key, frozen):
    expected = frozen["reference_sha256"][key]
    path = PRIVATE / "labels" / (key + ".json")
    if sha(path) != expected:
        raise RuntimeError("Reference label cache changed")
    return read(path)


def feature(key):
    with np.load(PRIVATE / "features" / (key + ".npz")) as data:
        return {k: data[k].copy() for k in data.files}


def model_q(model, x):
    if len(x) == 0:
        return []
    with torch.no_grad():
        result = model(torch.from_numpy(x))
    array = result["q"].cpu().numpy().astype(np.float64)
    array[result["abstain"].cpu().numpy()] = np.nan
    return q_list(array)


def macro_f1(observations, refs, predictions):
    pairs = match_positions(observations, refs)
    scores = []
    for c in range(4):
        tp = sum(predictions[p] == c and refs[r]["symbol"] == c for p, r in pairs)
        predicted = sum(p == c for p in predictions)
        expected = sum(r["symbol"] == c for r in refs)
        scores.append(2 * tp / (predicted + expected) if predicted + expected else 0.0)
    return float(np.mean(scores))


def prepare():
    import spacy
    from semantic_cycle2_gpu import references
    if (OUT / "FREEZE.json").exists() or PRIVATE.exists():
        raise RuntimeError("Fresh experiment/cache required")
    started = time.monotonic()
    PRIVATE.mkdir(parents=True, mode=0o700)
    groups_map = read(OLD / "SYMBOL_MAP.json")["template_to_symbol"]
    ids = sorted(map(int, groups_map))
    groups = [[j for j, t in enumerate(ids) if groups_map[str(t)] == c] for c in "ABCD"]
    old_seal = read(OLD / "PREDICTION_SEAL.json")["sha256"]
    if len(old_seal) != 72:
        raise RuntimeError("All 72 original predictions required")
    locked = read(OLD / "LOCKED_SELECTION.json")["selection"]
    assignments = read(PRIOR / "ASSIGNMENTS.json")
    nlp = spacy.load("en_core_web_sm")
    features_sha, reference_sha, manifest = {}, {}, {}
    total_observations = 0
    for index, (key, expected) in enumerate(sorted(old_seal.items()), 1):
        source = CACHE / "predictions" / (key + ".json")
        if sha(source) != expected:
            raise RuntimeError("Original prediction seal mismatch")
        row = read(source)
        refs = references(CACHE / "reference_mirror", row, nlp, assignments)
        observations = [{"anchor_span": o["anchor_span"]} for o in row["observations"]]
        x = np.asarray([scores_from_observation(o) for o in row["observations"]],
                       dtype=np.float32).reshape(-1, 3, 24)
        arrays = {"x": x}
        for rep in ("W", "E1", "C"):
            qs = [q_from_raw(o["raw_scores"][rep], rep, groups, locked[rep]["temperature"])
                  for o in row["observations"]]
            arrays["legacy_" + rep] = np.asarray(
                [[np.nan] * 4 if q is None else q for q in qs], dtype=np.float64).reshape(-1, 4)
        path = PRIVATE / "features" / (key + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        features_sha[key] = sha(path)
        truth = [r["symbol"] for r in refs] if row["arm"] == "positive" else None
        atomic(PRIVATE / "labels" / (key + ".json"), {
            "references": refs, "truth": truth,
            "label_source": "constructed positions/model-assisted paraphrase references"})
        reference_sha[key] = sha(PRIVATE / "labels" / (key + ".json"))
        manifest[key] = {
            "doc_id": row["doc_id"], "arm": row["arm"], "condition": row["condition"],
            "observations": observations, "candidate_status": row["candidates"]["status"],
            "candidate_count": len(observations)}
        total_observations += len(observations)
    docs = sorted({r["doc_id"] for r in manifest.values()})
    if docs != list(range(12, 24)) or total_observations != 1523:
        raise RuntimeError("Original cohort identity mismatch")
    splits = []
    for i, test in enumerate(docs):
        calibration = [docs[(i + j) % len(docs)] for j in (1, 2, 3)]
        train = [d for d in docs if d != test and d not in calibration]
        if len(train) != 8 or set(train) & set(calibration) or test in train + calibration:
            raise RuntimeError("Group isolation failed")
        splits.append({"test": test, "train": train, "calibration": calibration})
    atomic(OUT / "SPLITS.json", {"unit": "scenario/all variants", "folds": splits})
    # Engineering checks use actual cached scores, not simulated performance.
    trial_key = next(k for k, r in manifest.items()
                     if r["arm"] == "positive" and r["condition"] == "clean")
    trial = feature(trial_key)["x"]
    torch.manual_seed(SEED)
    model = MultiScaleSymbolReadout(groups)
    model.fit_normalizer(torch.from_numpy(trial))
    output = model(torch.from_numpy(trial))
    valid = ~output["abstain"]
    loss = -output["q"][valid].clamp_min(1e-9).log().mean()
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Real-input backward produced nonfinite gradients")
    q = model_q(model, trial)
    small = candidates()[[0, 137, 2520, 65535]]
    error = float(np.max(np.abs(exact_costs(q, small) - full_dp(q, small))))
    if error > 1e-10:
        raise RuntimeError("Independent exact-DP precheck failed")
    config = {
        "schema": "semantic-score-msa-readback/v1", "seed": SEED,
        "at": now(), "authorization": "User requested integration into paper two and reevaluation",
        "groups": groups, "modes": list(ALL_MODES), "legacy_selection": locked,
        "epochs": 40, "lr": 0.01, "weight_decay": 0.01,
        "threshold_grid": [0.25 + 0.05 * j for j in range(12)],
        "bootstrap_draws": 10000, "candidate_count": 65536, "message_length": 8,
        "cpu_budget_seconds": 3600, "posthoc_development_only": True,
        "human_gold_available": False, "new_provider_calls": 0,
        "new_amr_or_embedding_calls": 0, "precision": "CPU FP32",
        "source_sha256": {n: sha(ROOT / n) for n in SOURCE_FILES},
        "protocol_sha256": sha(OUT / "PROTOCOL.md"),
        "splits_sha256": sha(OUT / "SPLITS.json"),
        "old_public_sha256": {n: sha(OLD / n) for n in (
            "FREEZE.json", "SYMBOL_MAP.json", "LOCKED_SELECTION.json", "PREDICTION_SEAL.json",
            "RESULT.json", "ORDER_RESULT.json", "STATUS.json")},
        "old_prediction_sha256": old_seal, "feature_sha256": features_sha,
        "reference_sha256": reference_sha, "manifest": manifest,
        "original_reference_sha256": {
            "ASSIGNMENTS.json": sha(PRIOR / "ASSIGNMENTS.json"),
            "documents": {p.name: sha(p) for p in sorted((PRIOR / "documents").glob("*.json"))},
            "references": {p.name: sha(p) for p in sorted((PRIOR / "references").glob("*.json"))}},
        "dependencies": {n: metadata.version(n) for n in (
            "torch", "numpy", "spacy", "penman", "en-core-web-sm")},
    }
    atomic(OUT / "FREEZE.json", config)
    atomic(OUT / "READINESS.json", {
        "Go": True, "scenario_count": len(docs), "versions": len(manifest),
        "candidate_observations": total_observations, "original_prediction_sha_verified": 72,
        "group_isolation_passed": True, "real_input_gradient_finite": True,
        "dp_oracle_max_error": error, "new_upstream_model_calls": 0,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_counts": {m: MultiScaleSymbolReadout(groups, m).configuration()["trainable_parameters"]
                             for m in MODES}})
    return read(OUT / "READINESS.json")


def supervised(keys, frozen):
    rows, targets, weights = [], [], []
    for key in keys:
        meta = frozen["manifest"][key]
        ref = labels(key, frozen)["references"]
        pairs = match_positions(meta["observations"], ref)
        if not pairs:
            continue
        x = feature(key)["x"]
        for p, r in pairs:
            rows.append(x[p])
            targets.append(ref[r]["symbol"])
            weights.append(1.0 / (len(keys) * len(ref)))
    if not rows:
        raise RuntimeError("No training/calibration reference carriers")
    return (torch.from_numpy(np.asarray(rows, dtype=np.float32)),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32))


def carrier_loss(model, data):
    x, y, weight = data
    output = model(x)
    valid = ~output["abstain"] & output["class_valid"].gather(1, y[:, None]).squeeze(1)
    if not valid.any():
        raise RuntimeError("No valid supervised evidence")
    # Missing-evidence carriers contribute a fixed uniform-class loss to
    # selection; they do not fabricate a label-dependent gradient.
    losses = torch.full_like(weight, math.log(4))
    losses[valid] = F.cross_entropy(output["logits"][valid], y[valid], reduction="none")
    return (losses * weight).sum() / weight.sum()


def fit():
    frozen = verify()
    if (OUT / "MODEL_SEAL.json").exists():
        raise RuntimeError("Models already sealed")
    started = time.monotonic()
    checkpoints, selections = {}, []
    folds = read(OUT / "SPLITS.json")["folds"]
    for fold in folds:
        keys_for = lambda docs: [k for k, r in frozen["manifest"].items()
                                if r["doc_id"] in docs and r["arm"] == "positive"
                                and r["condition"] in ("clean", "paraphrase")]
        train_keys, val_keys = keys_for(fold["train"]), keys_for(fold["calibration"])
        train, val = supervised(train_keys, frozen), supervised(val_keys, frozen)
        normalization = torch.from_numpy(np.concatenate([feature(k)["x"] for k in train_keys]))
        for mode in MODES:
            torch.manual_seed(SEED)
            model = MultiScaleSymbolReadout(frozen["groups"], mode)
            model.fit_normalizer(normalization)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=frozen["lr"], weight_decay=frozen["weight_decay"])
            best_loss, best_epoch, best_state = math.inf, None, None
            history = []
            for epoch in range(1, frozen["epochs"] + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = carrier_loss(model, train)
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation = float(carrier_loss(model, val))
                history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                                "calibration_carrier_loss": validation})
                if validation < best_loss:
                    best_loss, best_epoch = validation, epoch
                    best_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            val_q = {k: model_q(model, feature(k)["x"]) for k in val_keys}
            threshold_scores = []
            for threshold in frozen["threshold_grid"]:
                f1 = []
                for key in val_keys:
                    predictions = [identity(q, threshold) for q in val_q[key]]
                    f1.append(macro_f1(frozen["manifest"][key]["observations"],
                                      labels(key, frozen)["references"], predictions))
                threshold_scores.append({"threshold": threshold, "macro_f1": float(np.mean(f1))})
            selection = min(threshold_scores, key=lambda v: (-v["macro_f1"], v["threshold"]))
            identifier = f"d{fold['test']:02d}-{mode}"
            model_path = PRIVATE / "models" / (identifier + ".pt")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "groups": frozen["groups"], "mode": mode}, model_path)
            reloaded = MultiScaleSymbolReadout(frozen["groups"], mode)
            reloaded.load_state_dict(torch.load(model_path, weights_only=True)["state_dict"])
            reloaded.eval()
            original, replay = model_q(model, feature(val_keys[0])["x"]), model_q(
                reloaded, feature(val_keys[0])["x"])
            if original != replay:
                raise RuntimeError("Checkpoint replay mismatch")
            checkpoints[identifier] = sha(model_path)
            record = {**fold, "mode": mode, **selection, "best_epoch": best_epoch,
                      "calibration_carrier_loss": best_loss, "training_seed": SEED,
                      "model_sha256": checkpoints[identifier], "configuration": model.configuration(),
                      "normalizer_training_keys": train_keys,
                      "supervision_training_keys": train_keys, "selection_calibration_keys": val_keys,
                      "checkpoint_replay_exact": True, "threshold_scores": threshold_scores,
                      "history": history}
            selections.append(record)
            atomic(OUT / "TRAINING_RESULTS.json", {"scope": "grouped development cross-validation",
                   "selections": selections, "expected_models": 84})
            heartbeat("FIT", len(selections), len(folds) * len(MODES), started)
    atomic(OUT / "MODEL_SEAL.json", {
        "at": now(), "sha256": checkpoints, "models": len(checkpoints),
        "training_results_sha256": sha(OUT / "TRAINING_RESULTS.json"),
        "held_out_test_labels_used_for_each_model": False,
        "all_scenarios_used_for_training_in_other_folds": True})
    return {"models": len(checkpoints), "elapsed_seconds": time.monotonic() - started}


def prediction_summary(costs, cs):
    minimum = float(costs.min())
    ties = np.flatnonzero(np.abs(costs - minimum) <= 1e-10)
    second = float(np.partition(costs, 1)[1])
    return {
        "minimum_cost": minimum, "second_message_cost": second,
        "different_message_margin": second - minimum,
        "minimum_ties": len(ties), "unique_prediction": len(ties) == 1,
        "predicted_message": cs[ties[0]].tolist() if len(ties) == 1 else None}


def predict():
    # This phase does not call labels() or read original references/assignments.
    frozen = verify()
    seal = read(OUT / "MODEL_SEAL.json")
    if sha(OUT / "TRAINING_RESULTS.json") != seal["training_results_sha256"]:
        raise RuntimeError("Model selection changed after seal")
    if (OUT / "PREDICTION_SEAL.json").exists():
        raise RuntimeError("Predictions already sealed")
    selections = {(r["test"], r["mode"]): r for r in read(OUT / "TRAINING_RESULTS.json")["selections"]}
    started = time.monotonic()
    cs = candidates()
    prediction_sha, costs_sha = {}, {}
    count = 0
    for key, meta in sorted(frozen["manifest"].items()):
        data = feature(key)
        records, costs_arrays = {}, {}
        for mode in ALL_MODES:
            if mode.startswith("legacy_"):
                rep = mode.removeprefix("legacy_")
                qs = q_list(data[mode])
                threshold = frozen["legacy_selection"][rep]["threshold"]
            else:
                identifier = f"d{meta['doc_id']:02d}-{mode}"
                path = PRIVATE / "models" / (identifier + ".pt")
                if sha(path) != seal["sha256"][identifier]:
                    raise RuntimeError("Model seal mismatch")
                model = MultiScaleSymbolReadout(frozen["groups"], mode)
                model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
                model.eval()
                qs = model_q(model, data["x"])
                threshold = selections[(meta["doc_id"], mode)]["threshold"]
            predictions = [identity(q, threshold) for q in qs]
            retained_qs = [q if p is not None else None for q, p in zip(qs, predictions)]
            record = {"q": qs, "predictions": predictions, "threshold": threshold,
                      "emitted": sum(p is not None for p in predictions),
                      "observations": len(predictions)}
            if meta["arm"] == "positive":
                costs = exact_costs(retained_qs, cs)
                costs_arrays[mode] = costs
                record["message"] = prediction_summary(costs, cs)
            records[mode] = record
            count += 1
        path = PRIVATE / "predictions" / (key + ".json")
        atomic(path, {"identity": {k: meta[k] for k in ("doc_id", "arm", "condition")},
                      "test_labels_loaded": False, "records": records})
        prediction_sha[key] = sha(path)
        if costs_arrays:
            cost_path = PRIVATE / "costs" / (key + ".npz")
            cost_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cost_path, **costs_arrays)
            costs_sha[key] = sha(cost_path)
        heartbeat("PREDICT_AND_EXACT_DECODE", len(prediction_sha), 72, started)
    atomic(OUT / "PREDICTION_SEAL.json", {
        "at": now(), "sha256": prediction_sha, "costs_sha256": costs_sha,
        "versions": 72, "positive_cost_archives": 48, "method_versions": count,
        "candidate_count": len(cs), "test_labels_loaded_in_prediction_process": False,
        "model_seal_sha256": sha(OUT / "MODEL_SEAL.json"),
        "elapsed_seconds": time.monotonic() - started})
    return {"sealed_versions": 72, "method_versions": count,
            "candidate_count": len(cs), "elapsed_seconds": time.monotonic() - started}


def interval(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(SEED).integers(0, len(values), (10000, len(values)))
    return {"mean": float(values.mean()),
            "CI95": np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
            "n": len(values), "unit": "scenario", "descriptive_development_interval": True}


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def evaluate():
    frozen = verify()
    seal = read(OUT / "PREDICTION_SEAL.json")
    if seal["test_labels_loaded_in_prediction_process"] or len(seal["sha256"]) != 72:
        raise RuntimeError("Invalid prediction seal")
    started = time.monotonic()
    cs, rows, negatives = candidates(), [], []
    max_error, cost_entries, legacy_checks, metric_checks = 0.0, 0, 0, 0
    old_rows = {(r["doc_id"], r["arm"], r["condition"]): r for r in read(OLD / "RESULT.json")["rows"]}
    for index, (key, expected) in enumerate(sorted(seal["sha256"].items()), 1):
        path = PRIVATE / "predictions" / (key + ".json")
        if sha(path) != expected:
            raise RuntimeError("Prediction seal mismatch")
        prediction, meta = read(path), frozen["manifest"][key]
        ref = labels(key, frozen)
        if meta["arm"] == "unwatermarked":
            for mode, record in prediction["records"].items():
                negatives.append({"doc_id": meta["doc_id"], "condition": meta["condition"],
                                  "method": mode, "observations": record["observations"],
                                  "emitted": record["emitted"], "not_detector_FPR": True})
            continue
        cost_path = PRIVATE / "costs" / (key + ".npz")
        if sha(cost_path) != seal["costs_sha256"][key]:
            raise RuntimeError("Cost archive seal mismatch")
        with np.load(cost_path) as archive:
            for mode, record in prediction["records"].items():
                diag = diagnostic(meta["observations"], ref["references"], record["predictions"])
                independent = independent_diagnostic(
                    meta["observations"], ref["references"], record["predictions"])
                if diag != independent:
                    raise RuntimeError("Independent event diagnostic mismatch")
                metric_checks += 1
                if mode.startswith("legacy_"):
                    rep = mode.removeprefix("legacy_")
                    historical = old_rows[(meta["doc_id"], "positive", meta["condition"])]["diagnostics"][rep]
                    if diag != historical:
                        raise RuntimeError("Legacy baseline does not reproduce original")
                    legacy_checks += 1
                qs = [q if p is not None else None
                      for q, p in zip(record["q"], record["predictions"])]
                independent_costs = full_dp(qs, cs)
                error = float(np.max(np.abs(independent_costs - archive[mode])))
                max_error = max(max_error, error)
                cost_entries += len(cs)
                if error > 1e-10:
                    raise RuntimeError("Independent full-space DP audit failed")
                ranking = rank_truth(archive[mode], cs, ref["truth"])
                predicted = record["message"]["predicted_message"]
                correct = predicted == ref["truth"]
                matches = match_positions(meta["observations"], ref["references"])
                q_correct = sum(identity(record["q"][p], .25) == ref["references"][r]["symbol"]
                                for p, r in matches)
                rows.append({
                    "doc_id": meta["doc_id"], "condition": meta["condition"], "method": mode,
                    **diag, "macro_f1": macro_f1(meta["observations"], ref["references"],
                                               record["predictions"]),
                    "blind_carrier_accuracy_full_denominator": diag["identity_correct"] / 8,
                    "located_no_reject_correct_full_denominator": q_correct / 8,
                    **ranking, **record["message"], "complete_correct": int(correct),
                    "wrong_unique_message": int(predicted is not None and not correct),
                    "message_erasure": int(predicted is None),
                    "symbol_edit_error_or_erasure_rate": (
                        edit_distance(predicted, ref["truth"]) / 8 if predicted is not None else 1.0)})
        heartbeat("EVALUATE_AND_INDEPENDENT_AUDIT", index, 72, started)
    summaries = []
    for condition in CONDITIONS:
        for mode in ALL_MODES:
            subset = sorted([r for r in rows if r["condition"] == condition and r["method"] == mode],
                            key=lambda r: r["doc_id"])
            if len(subset) != 12:
                raise RuntimeError("Incomplete condition/method denominator")
            summaries.append({
                "condition": condition, "method": mode, "scenarios": 12,
                "planned_carriers": 96, "correct_carriers": sum(r["identity_correct"] for r in subset),
                "complete_correct": sum(r["complete_correct"] for r in subset),
                "wrong_unique_messages": sum(r["wrong_unique_message"] for r in subset),
                "message_erasures": sum(r["message_erasure"] for r in subset),
                "emitted": sum(r["emitted"] for r in subset),
                "observations": sum(r["observations"] for r in subset),
                **{metric: interval([r[metric] for r in subset]) for metric in (
                    "macro_f1", "identity_f1", "blind_carrier_accuracy_full_denominator",
                    "located_no_reject_correct_full_denominator", "complete_correct",
                    "average_rank", "pairwise_win_rate", "symbol_edit_error_or_erasure_rate")}})
    contrasts = []
    for condition in CONDITIONS:
        for baseline in ("gate", "gate_wide", "mean", "legacy_W", "legacy_E1"):
            a = sorted([r for r in rows if r["condition"] == condition and r["method"] == "attention"],
                       key=lambda r: r["doc_id"])
            b = sorted([r for r in rows if r["condition"] == condition and r["method"] == baseline],
                       key=lambda r: r["doc_id"])
            for metric in ("macro_f1", "identity_f1", "complete_correct"):
                contrasts.append({
                    "condition": condition, "contrast": "attention-" + baseline, "metric": metric,
                    **interval([x[metric] - y[metric] for x, y in zip(a, b)])})
    primary = next(c for c in contrasts if c["condition"] == "paraphrase"
                   and c["contrast"] == "attention-gate" and c["metric"] == "macro_f1")
    development_gain = primary["mean"] >= .05 and primary["CI95"][0] > 0
    result = {
        "schema": "semantic-score-msa-development-results/v1", "at": now(),
        "scope": "posthoc old-data grouped cross-validation; constructed/model-assisted references",
        "new_independent_scenarios": 0, "scenarios": 12, "versions": 72,
        "methods": list(ALL_MODES), "trained_models": 84, "candidate_messages": 65536,
        "summaries": summaries, "paired_contrasts": contrasts, "rows": rows,
        "unwatermarked_output_diagnostics": negatives,
        "primary": primary, "development_gain_established": development_gain,
        "formal_new_fact_symbol_validation": "NOT_RUN", "human_gold": False,
        "new_data_complete_message_validation": "NOT_RUN",
        "attention_robustness_confirmed": False, "detector_FPR_TPR": None,
        "new_provider_calls": 0, "new_AMR_embedding_calls": 0,
        "prediction_seal_sha256": sha(OUT / "PREDICTION_SEAL.json"),
        "evaluation_seconds": time.monotonic() - started}
    atomic(OUT / "RESULT.json", result)
    atomic(OUT / "AUDIT.json", {
        "Go": True, "original_prediction_sha_checks": 72, "legacy_event_metric_reproductions": legacy_checks,
        "independent_event_metric_checks": metric_checks, "independent_full_space_cost_entries": cost_entries,
        "max_dp_error": max_error, "prediction_process_test_labels_loaded": False,
        "group_isolation_and_checkpoint_replay": True, "result_sha256": sha(OUT / "RESULT.json"),
        "at": now()})
    return {"primary_attention_gate_macro_f1_difference": primary,
            "development_gain_established": development_gain,
            "independent_full_space_cost_entries": cost_entries,
            "max_dp_error": max_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit", "predict", "evaluate"))
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if (OUT / "HALT.json").exists():
        raise RuntimeError("Permanent experiment halt")
    try:
        result = {"prepare": prepare, "fit": fit, "predict": predict, "evaluate": evaluate}[args.action]()
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        atomic(OUT / "HALT.json", {"at": now(), "permanent": True, "phase": args.action,
                                  "reason_class": type(exc).__name__, "reason": str(exc)})
        raise


if __name__ == "__main__":
    main()
