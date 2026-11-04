"""Exploratory attribution from frozen readback checkpoints and cached scores.

No training, upstream inference, new text, or parameter selection. Threshold
swaps and block bypasses are labeled diagnostic sensitivities, not new methods.
All event labels are the old constructed/model-assisted references.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src/core")]
import run_semantic_multiscale_readback as run
from semantic_multiscale import MultiScaleSymbolReadout
from semantic_cycle2 import identity, match_positions
from audit_semantic_readout import independent_diagnostic

OUT = ROOT / "report_assets/semantic_multiscale_failure_analysis_20261026"
MODES = ("gate", "attention", "gate_wide", "mean")


def model_load(doc, mode, frozen):
    checkpoint = run.PRIVATE / "models" / f"d{doc:02d}-{mode}.pt"
    seal = run.read(run.OUT / "MODEL_SEAL.json")["sha256"]
    if run.sha(checkpoint) != seal[checkpoint.stem]:
        raise RuntimeError("Checkpoint SHA mismatch")
    model = MultiScaleSymbolReadout(frozen["groups"], mode)
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    model.eval()
    return model


def symbol_q(model, fused, template_valid):
    group_logits, class_valid = [], []
    for group in model.groups:
        valid = template_valid[:, group].any(1)
        values = fused[:, group].masked_fill(~valid[:, None], 0)
        group_logits.append(torch.logsumexp(values, 1) - math.log(len(group)))
        class_valid.append(valid)
    valid = torch.stack(class_valid, 1)
    logits = F.softplus(model.log_scale) * torch.stack(group_logits, 1) + model.class_bias
    logits = logits.masked_fill(~valid, -torch.inf).masked_fill(~valid.any(1)[:, None], 0)
    q = torch.softmax(logits, 1)
    array = q.numpy().astype(np.float64)
    array[~valid.any(1).numpy()] = np.nan
    return run.q_list(array)


def attention_bypass(model, x, kind):
    """Frozen weights, changed forward path; no fitting after intervention."""
    fusion = model.fusion
    valid = fusion._validate(x)
    z = torch.where(valid, (x - fusion.center) / fusion.spread, 0)
    strongest = z.masked_fill(~valid, -torch.inf).amax(2, keepdim=True)
    strongest = torch.where(torch.isfinite(strongest), strongest, 0)
    features = torch.stack((z, z - strongest), -1)
    features = torch.where(valid.unsqueeze(-1), features, 0)
    tokens = fusion.project(features) + fusion.scale_embedding[None, :, None, :]
    batch, _, templates, hidden = tokens.shape
    tokens = tokens.permute(0, 2, 1, 3).reshape(-1, 3, hidden)
    present = valid.permute(0, 2, 1).reshape(-1, 3)
    tokens = tokens.masked_fill(~present.unsqueeze(-1), 0)
    if kind in ("original", "no_layernorm"):
        padding = ~present.clone()
        padding[~present.any(1), 0] = False
        exchanged, _ = fusion.cross_attention(
            tokens, tokens, tokens, key_padding_mask=padding, need_weights=False)
    else:
        exchanged = torch.zeros_like(tokens)
    if kind in ("original", "no_qkv"):
        tokens = fusion.norm(tokens + exchanged)
    else:
        tokens = tokens + exchanged
    logits = fusion.gate(tokens).squeeze(-1).reshape(batch, templates, 3).permute(0, 2, 1)
    template_valid = valid.any(1)
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = torch.where(template_valid.unsqueeze(1), logits, 0)
    weights = torch.softmax(logits, 1).masked_fill(~valid, 0)
    fused = (weights * z).sum(1).masked_fill(~template_valid, -torch.inf)
    return symbol_q(model, fused, template_valid)


def evaluate_q(qs, threshold, observations, refs):
    preds = [identity(q, threshold) for q in qs]
    pairs = match_positions(observations, refs)
    diag = independent_diagnostic(observations, refs, preds)
    raw_preds = [identity(q, .25) for q in qs]
    raw_correct = sum(raw_preds[p] == refs[r]["symbol"] for p, r in pairs)
    correct_rejected = sum(raw_preds[p] == refs[r]["symbol"] and preds[p] is None
                           for p, r in pairs)
    confusion = [[0] * 5 for _ in range(4)]
    raw_confusion = [[0] * 5 for _ in range(4)]
    for p, r in pairs:
        c = refs[r]["symbol"]
        confusion[c][4 if preds[p] is None else preds[p]] += 1
        raw_confusion[c][4 if raw_preds[p] is None else raw_preds[p]] += 1
    for r in {r for _, r in pairs} ^ set(range(len(refs))):
        confusion[refs[r]["symbol"]][4] += 1
        raw_confusion[refs[r]["symbol"]][4] += 1
    nll, brier, ptrue, confidence = [], [], [], []
    true_correct_confidence, true_wrong_confidence = [], []
    for p, r in pairs:
        q = qs[p]
        if q is None:
            nll.append(math.log(4)); brier.append(.75); ptrue.append(.25)
        else:
            c = refs[r]["symbol"]
            nll.append(-math.log(max(q[c], 1e-9)))
            brier.append(float(np.sum((np.asarray(q) - np.eye(4)[c]) ** 2)))
            ptrue.append(q[c]); confidence.append(max(q))
            if raw_preds[p] == c:
                true_correct_confidence.append(max(q))
            else:
                true_wrong_confidence.append(max(q))
    class_f1 = []
    for c in range(4):
        tp = diag["per_class"][c]["correct"]
        predicted = sum(p == c for p in preds)
        expected = diag["per_class"][c]["planned"]
        class_f1.append(2 * tp / (predicted + expected) if predicted + expected else 0)
    return {
        "macro_f1": float(np.mean(class_f1)), "class_f1": class_f1,
        "identity_f1": diag["identity_f1"], "correct": diag["identity_correct"],
        "raw_correct": raw_correct, "correct_rejected": correct_rejected,
        "emitted": diag["emitted"], "observations": len(observations),
        "located_references": len(pairs), "planned": len(refs),
        "matched_nll": float(np.mean(nll)), "matched_brier": float(np.mean(brier)),
        "matched_true_probability": float(np.mean(ptrue)),
        "mean_max_probability": float(np.mean(confidence)),
        "mean_correct_confidence": float(np.mean(true_correct_confidence)) if true_correct_confidence else None,
        "mean_wrong_confidence": float(np.mean(true_wrong_confidence)) if true_wrong_confidence else None,
        "confusion": confusion, "raw_confusion": raw_confusion,
        "prediction_counts": [sum(p == c for p in preds) for c in range(4)],
        "unmatched_emitted": sum(preds[p] is not None
                                 for p in set(range(len(preds))) - {p for p, _ in pairs}),
        "no_background_gold": True,
    }


def aggregate(rows):
    scalar = ("macro_f1", "identity_f1", "matched_nll", "matched_brier", "matched_true_probability",
              "mean_max_probability")
    summed = ("correct", "raw_correct", "correct_rejected", "emitted", "observations",
              "located_references", "planned", "unmatched_emitted")
    return {
        **{name: float(np.mean([r[name] for r in rows])) for name in scalar},
        **{name: sum(r[name] for r in rows) for name in summed},
        "class_f1": np.mean([r["class_f1"] for r in rows], 0).tolist(),
        "confusion": np.sum([r["confusion"] for r in rows], 0).tolist(),
        "raw_confusion": np.sum([r["raw_confusion"] for r in rows], 0).tolist(),
        "prediction_counts": np.sum([r["prediction_counts"] for r in rows], 0).tolist(),
        "scenarios": len(rows),
    }


def weight_stats(model, x, condition):
    if len(x) == 0:
        return None
    output = model.fusion(x, return_attention=model.mode == "attention")
    weights = output["scale_weights"].permute(0, 2, 1).numpy()
    present = output["valid_templates"].numpy()
    flat = weights[present]
    valid_views = torch.isfinite(x).permute(0, 2, 1).numpy()[present].sum(1)
    entropy = -np.sum(flat * np.log(np.maximum(flat, 1e-30)), 1)
    normalized_entropy = entropy[valid_views > 1] / np.log(valid_views[valid_views > 1])
    result = {
        "condition": condition, "mean_scale_weight": np.mean(flat, 0).tolist(),
        "normalized_weight_entropy": float(np.mean(normalized_entropy)),
        "one_scale_over_90_percent": float(np.mean(flat.max(1) > .9)),
    }
    attention = output["cross_attention"]
    if attention is not None:
        matrices = attention.numpy()[present]
        result["mean_qkv_matrix"] = np.mean(matrices, 0).tolist()
        result["qkv_uniform_distance_mean"] = float(np.mean(np.abs(matrices - 1/3)))
    return result


def analyze():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    frozen = run.verify()
    if (OUT / "RESULT.json").exists():
        raise RuntimeError("Do not overwrite a completed exploratory diagnostic")
    OUT.mkdir(parents=True, exist_ok=True)
    selections = run.read(run.OUT / "TRAINING_RESULTS.json")["selections"]
    indexed = {(r["test"], r["mode"]): r for r in selections}
    training, metrics, weights, cases = [], [], [], []
    bypass_checks = []
    raw_scores, score_metadata = [], []
    for key, meta in frozen["manifest"].items():
        if meta["arm"] == "positive" and meta["condition"] == "paraphrase":
            raw_scores.extend(run.feature(key)["x"].tolist())
            score_metadata.extend([meta["doc_id"]] * meta["candidate_count"])
    for doc in range(12, 24):
        for mode in MODES:
            selection = indexed[(doc, mode)]
            model = model_load(doc, mode, frozen)
            tr = run.supervised(selection["supervision_training_keys"], frozen)
            va = run.supervised(selection["selection_calibration_keys"], frozen)
            with torch.no_grad():
                tr_loss = float(run.carrier_loss(model, tr))
                va_loss = float(run.carrier_loss(model, va))
            training.append({
                "doc_id": doc, "mode": mode, "train_carriers": len(tr[1]),
                "calibration_carriers": len(va[1]), "checkpoint_train_nll": tr_loss,
                "checkpoint_calibration_nll": va_loss, "selected_epoch": selection["best_epoch"],
                "threshold": selection["threshold"], "history": selection["history"],
                "class_bias": model.class_bias.detach().numpy().tolist(),
                "positive_logit_scale": float(F.softplus(model.log_scale).detach()),
            })
            for condition in run.CONDITIONS:
                key = f"d{doc:02d}-positive-{condition}"
                meta = frozen["manifest"][key]
                ref = run.labels(key, frozen)["references"]
                record = run.read(run.PRIVATE / "predictions" / (key + ".json"))["records"][mode]
                qs = record["q"]
                own = evaluate_q(qs, selection["threshold"], meta["observations"], ref)
                metrics.append({"doc_id": doc, "condition": condition, "mode": mode,
                                "diagnostic": "own_threshold", **own})
                if condition == "paraphrase":
                    for threshold in (.25, .30, .35, .40):
                        metrics.append({
                            "doc_id": doc, "condition": condition, "mode": mode,
                            "diagnostic": "fixed_threshold_" + str(threshold),
                            **evaluate_q(qs, threshold, meta["observations"], ref)})
                    if mode in ("attention", "gate"):
                        other = "gate" if mode == "attention" else "attention"
                        metrics.append({
                            "doc_id": doc, "condition": condition, "mode": mode,
                            "diagnostic": "other_model_threshold",
                            **evaluate_q(qs, indexed[(doc, other)]["threshold"], meta["observations"], ref)})
                x = torch.from_numpy(run.feature(key)["x"])
                with torch.no_grad():
                    weight = weight_stats(model, x, condition) if mode != "mean" else None
                    if weight:
                        weights.append({"doc_id": doc, "mode": mode, **weight})
                    if mode == "attention" and condition == "paraphrase":
                        original = attention_bypass(model, x, "original")
                        error = max(np.max(np.abs(np.asarray(a) - np.asarray(b)))
                                    for a, b in zip(original, qs) if a is not None and b is not None)
                        bypass_checks.append(float(error))
                        if error > 1e-6:
                            raise RuntimeError("Attention probe failed to reproduce frozen forward")
                        for kind in ("no_qkv", "no_layernorm", "no_qkv_and_layernorm"):
                            changed = attention_bypass(model, x, kind)
                            metrics.append({
                                "doc_id": doc, "condition": condition, "mode": "attention",
                                "diagnostic": kind + "_frozen_weights",
                                **evaluate_q(changed, selection["threshold"], meta["observations"], ref)})
        key = f"d{doc:02d}-positive-paraphrase"
        meta = frozen["manifest"][key]
        ref = run.labels(key, frozen)["references"]
        record = run.read(run.PRIVATE / "predictions" / (key + ".json"))["records"]
        for p, rr in match_positions(meta["observations"], ref):
            truth = ref[rr]["symbol"]
            item = {
                "doc_id": doc, "observation_index": p, "reference_slot": ref[rr]["slot"],
                "reference_label": truth,
                "gate_raw_prediction": identity(record["gate"]["q"][p], .25),
                "attention_raw_prediction": identity(record["attention"]["q"][p], .25),
                "gate_prediction": record["gate"]["predictions"][p],
                "attention_prediction": record["attention"]["predictions"][p],
                "gate_q": record["gate"]["q"][p], "attention_q": record["attention"]["q"][p],
                "gate_threshold": indexed[(doc, "gate")]["threshold"],
                "attention_threshold": indexed[(doc, "attention")]["threshold"],
            }
            item["gate_raw_correct"] = item["gate_raw_prediction"] == truth
            item["attention_raw_correct"] = item["attention_raw_prediction"] == truth
            item["gate_correct"] = item["gate_prediction"] == truth
            item["attention_correct"] = item["attention_prediction"] == truth
            cases.append(item)
        print(json.dumps({"completed_scenarios": doc-11, "total": 12}), flush=True)
    keys = sorted({(r["condition"], r["mode"], r["diagnostic"]) for r in metrics})
    summaries = [{"condition": c, "mode": m, "diagnostic": d,
                  **aggregate([r for r in metrics if (r["condition"], r["mode"], r["diagnostic"]) == (c,m,d)])}
                 for c,m,d in keys]
    train_summaries = []
    for mode in MODES:
        rows = [r for r in training if r["mode"] == mode]
        train_summaries.append({
            "mode": mode,
            **{name: float(np.mean([r[name] for r in rows])) for name in (
                "checkpoint_train_nll", "checkpoint_calibration_nll",
                "train_carriers", "calibration_carriers", "threshold", "selected_epoch",
                "positive_logit_scale")},
            "thresholds": [r["threshold"] for r in rows],
            "selected_epochs": [r["selected_epoch"] for r in rows],
            "class_bias_mean": np.mean([r["class_bias"] for r in rows], 0).tolist(),
            "first_train_loss": float(np.mean([r["history"][0]["train_loss"] for r in rows])),
            "last_train_loss": float(np.mean([r["history"][-1]["train_loss"] for r in rows])),
            "last_calibration_loss": float(np.mean([r["history"][-1]["calibration_carrier_loss"] for r in rows])),
        })
    correlations = []
    x = np.asarray(raw_scores)
    for a,b in ((0,1),(0,2),(1,2)):
        per_template = []
        for t in range(24):
            valid = np.isfinite(x[:,a,t]) & np.isfinite(x[:,b,t])
            p,q = x[valid,a,t], x[valid,b,t]
            if np.std(p)>0 and np.std(q)>0:
                per_template.append(float(np.corrcoef(p,q)[0,1]))
        correlations.append({"scales":[("W6","W12","E1_12")[a],("W6","W12","E1_12")[b]],
                             "mean_template_pearson":float(np.mean(per_template)),
                             "median_template_pearson":float(np.median(per_template)),
                             "template_correlations":per_template, "templates":len(per_template),
                             "scope":"same 253 old paraphrase candidate observations; descriptive only"})
    # Algebraic two-factor accounting, not a causal Shapley experiment.
    shapley = []
    for doc in range(12,24):
        val = lambda mode,diag: next(r for r in metrics if r["doc_id"]==doc and
                                    r["condition"]=="paraphrase" and r["mode"]==mode and r["diagnostic"]==diag)["macro_f1"]
        gg=val("gate","own_threshold"); aa=val("attention","own_threshold")
        ag=val("attention","other_model_threshold"); ga=val("gate","other_model_threshold")
        shapley.append({"doc_id":doc, "total":aa-gg,
                        "q_change_accounting": .5*((ag-gg)+(aa-ga)),
                        "threshold_change_accounting": .5*((ga-gg)+(aa-ag))})
    counts = {
        "matched_references":len(cases),
        "gate_correct_attention_wrong_raw":sum(c["gate_raw_correct"] and not c["attention_raw_correct"] for c in cases),
        "attention_correct_gate_wrong_raw":sum(c["attention_raw_correct"] and not c["gate_raw_correct"] for c in cases),
        "gate_correct_attention_wrong_after_reject":sum(c["gate_correct"] and not c["attention_correct"] for c in cases),
        "attention_correct_gate_wrong_after_reject":sum(c["attention_correct"] and not c["gate_correct"] for c in cases),
        "gate_correct_rejected":sum(c["gate_raw_correct"] and c["gate_prediction"] is None for c in cases),
        "attention_correct_rejected":sum(c["attention_raw_correct"] and c["attention_prediction"] is None for c in cases),
        "joint_raw_correct":sum(c["gate_raw_correct"] and c["attention_raw_correct"] for c in cases),
        "joint_after_reject_correct":sum(c["gate_correct"] and c["attention_correct"] for c in cases),
    }
    weight_summaries=[]
    for mode in ("gate","attention","gate_wide"):
        for condition in run.CONDITIONS:
            rows=[r for r in weights if r["mode"]==mode and r["condition"]==condition]
            weight_summaries.append({"mode":mode,"condition":condition,
                "mean_scale_weight":np.mean([r["mean_scale_weight"] for r in rows],0).tolist(),
                "normalized_weight_entropy":float(np.mean([r["normalized_weight_entropy"] for r in rows])),
                "one_scale_over_90_percent":float(np.mean([r["one_scale_over_90_percent"] for r in rows])),
                **({"mean_qkv_matrix":np.mean([r["mean_qkv_matrix"] for r in rows],0).tolist(),
                    "qkv_uniform_distance_mean":float(np.mean([r["qkv_uniform_distance_mean"] for r in rows]))}
                   if mode=="attention" else {})})
    result = {
        "schema":"semantic-score-msa-failure-exploratory/v1", "at":run.now(),
        "scope":"posthoc label-aware diagnostic on existing held-out checkpoints; no new method results",
        "human_gold":False, "new_training_runs":0, "new_AMR_or_embedding_calls":0,
        "parent_result_sha256":run.sha(run.OUT/"RESULT.json"),
        "script_sha256":run.sha(Path(__file__)), "summaries":summaries,
        "training_summary":train_summaries, "training":training,
        "weight_summaries":weight_summaries, "weight_rows":weights,
        "raw_view_correlations":correlations,"case_counts":counts,
        "threshold_q_accounting": {
            "is_causal":False, "total":float(np.mean([r["total"] for r in shapley])),
            "q_change":float(np.mean([r["q_change_accounting"] for r in shapley])),
            "threshold_change":float(np.mean([r["threshold_change_accounting"] for r in shapley])),
            "rows":shapley},
        "diagnostic_rows":metrics, "counterfactual_forward_max_error":max(bypass_checks),
        "elapsed_seconds":time.monotonic()-started,
    }
    # Supporting per-observation labels stay in private cache.
    case_path=Path.home()/".cache/amr/semantic_multiscale_failure_analysis_20261026/CASES.json"
    run.atomic(case_path, {"rows":cases,"model_assisted_not_human_gold":True})
    run.atomic(OUT/"RESULT.json",result)
    run.atomic(OUT/"SOURCE_SHA.json",{
        "analysis_script":run.sha(Path(__file__)),
        "parent_files":{n:run.sha(run.OUT/n) for n in
                        ("FREEZE.json","MODEL_SEAL.json","PREDICTION_SEAL.json","TRAINING_RESULTS.json",
                         "RESULT.json","INDEPENDENT_AUDIT.json","STATUS.json")},
        "private_case_numeric_record_sha256":run.sha(case_path)})
    return {k:result[k] for k in ("training_summary","case_counts","threshold_q_accounting",
                                "weight_summaries","raw_view_correlations")}


if __name__ == "__main__":
    result=analyze()
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("weight_summaries","raw_view_correlations","threshold_q_accounting")},ensure_ascii=False))
