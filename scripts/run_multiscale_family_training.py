"""Frozen family-hard development: train -> select on validation -> evaluate."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiscale_ablation import load_data
from scripts.run_multiscale_fusion import infer, partition, write_json
from scripts.run_multiscale_validation import sha, verify_inputs
from src.core.multiscale_ablation import ConfidenceFusion, hard_negative_weights
from src.core.multiscale_data import validate_split
from src.core.multiscale_family_training import (
    VERSION, family_groups, family_negative_weights, guarded_epoch,
    max_format_loss, select_coefficient,
)
from src.core.multiscale_fusion import EmpiricalThreshold, MultiScaleAttentionFusion

DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/family_hard1"


def model_for(stage):
    if stage.startswith("U"):
        return MultiScaleAttentionFusion(mode="gate")
    return ConfidenceFusion(mode="attention" if stage.startswith("X") else "gate")


def key_partition(positive, negative, docs):
    return ([positive.keys[i] for i in positive.indices(docs)]
            + [negative.keys[i] for i in negative.indices(docs)])


def validation_metrics(model, tx, ty, vx, vy, vw, vk, batch):
    tn = infer(model, tx[ty == 0], batch)
    vp = infer(model, vx, batch)
    if not torch.isfinite(tn).all() or not torch.isfinite(vp).all():
        raise ValueError("Missing training/validation evidence")
    threshold = EmpiricalThreshold.fit(tn.tolist(), 0.05)
    pred = np.array(threshold.predict(vp.tolist()))
    negative = vy.numpy() == 0
    loss = F.binary_cross_entropy_with_logits(vp, vy, reduction="none")
    per = {a: float(pred[negative & np.array([k[1] == a for k in vk])].mean())
           for a in sorted({k[1] for k in vk})}
    return {
        "validation_bce": float((loss * vw).sum()),
        "validation_positive_bce": float((loss[vy == 1] * vw[vy == 1]).sum()
                                         / vw[vy == 1].sum()),
        "validation_negative_bce": float((loss[vy == 0] * vw[vy == 0]).sum()
                                         / vw[vy == 0].sum()),
        "validation_tpr": float(pred[~negative].mean()),
        "validation_fpr": float(pred[negative].mean()),
        "validation_worst_fpr": max(per.values()),
        "validation_condition_fpr": per,
        "training_proxy_threshold": threshold.to_dict(),
    }, tn, vp


def train_one(stage, coefficient_name, seed, out, protocol):
    verify_inputs(protocol)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    positive, negative = load_data()
    split = validate_split(json.loads((ROOT / protocol["split_path"]).read_text()),
                           positive, negative)
    directory = out / "runs" / f"{stage}_{coefficient_name}_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    coefficient = protocol["coefficients"][coefficient_name]
    if not stage.endswith("3") and coefficient:
        raise ValueError("Non-loss stage with nonzero coefficient")
    model = model_for(stage)
    model.fit_normalizer(negative.scores[negative.indices(split["train"])])
    tx, ty, tw = partition(positive, negative, split["train"])
    vx, vy, vw = partition(positive, negative, split["validation"])
    tk = key_partition(positive, negative, split["train"])
    vk = key_partition(positive, negative, split["validation"])
    ni = (ty == 0).nonzero().flatten()
    groups = family_groups([tk[i] for i in ni.tolist()])
    nf_doc = {}
    for (doc, _), _rows in groups:
        nf_doc[doc] = nf_doc.get(doc, 0) + 1
    family_weights = torch.tensor([1 / (len(nf_doc) * nf_doc[key[0]])
                                   for key, _ in groups])
    optimizer = torch.optim.AdamW(model.parameters(), **protocol["optimizer"])
    row_rng = torch.Generator().manual_seed(seed)
    family_rng = torch.Generator().manual_seed(seed + protocol["family_rng_offset"])
    states, history, mining = {}, [], []
    trace = {"base_weights": tw.numpy(), "labels": ty.numpy(),
             "keys": np.array(tk, dtype=str), "negative_indices": ni.numpy()}
    vtrace = {"validation_keys": np.array(vk, dtype=str), "validation_labels": vy.numpy()}
    hard = protocol["hard_negative"]
    for epoch in range(1, protocol["epochs"] + 1):
        weights = tw
        if epoch > hard["warmup_epochs"] and stage != "U0":
            scores = infer(model, tx[ni], protocol["batch_size"])
            fn = hard_negative_weights if stage.endswith("1") else family_negative_weights
            weights, selected, audit = fn(tw, ty, tk, scores,
                                         hard["fraction"], hard["multiplier"])
            audit.update(epoch=epoch, partition="training_negatives_only",
                         method="row" if stage.endswith("1") else "family")
            mining.append(audit)
            trace[f"statistics_e{epoch:02}"] = scores.numpy()
            trace[f"weights_e{epoch:02}"] = weights.numpy()
            trace[f"selected_e{epoch:02}"] = selected.numpy()
        batches = torch.randperm(len(tx), generator=row_rng).split(protocol["batch_size"])
        family_batches = torch.tensor_split(
            torch.randperm(len(groups), generator=family_rng), len(batches))
        model.train()
        base_losses, worst_losses, positive_losses = [], [], []
        for indices, fid in zip(batches, family_batches):
            prediction = model(tx[indices])
            if prediction["abstain"].any():
                raise ValueError("All-missing training row")
            loss_rows = F.binary_cross_entropy_with_logits(
                prediction["statistic"], ty[indices], reduction="none")
            weight = weights[indices]
            base_loss = (loss_rows * weight).sum() / weight.sum()
            pos = ty[indices] == 1
            if pos.any():
                positive_losses.append(float((loss_rows[pos] * weight[pos]).sum().detach()
                                              / weight[pos].sum()))
            loss = base_loss
            if coefficient and epoch > hard["warmup_epochs"]:
                source_rows, local_groups = [], []
                for f in fid.tolist():
                    rows = ni[groups[f][1]].tolist()
                    local_groups.append(list(range(len(source_rows), len(source_rows) + len(rows))))
                    source_rows.extend(rows)
                output = model(tx[source_rows])
                penalty = max_format_loss(output["statistic"], local_groups, family_weights[fid])
                loss = loss + coefficient * penalty
                worst_losses.append(float(penalty.detach()))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            base_losses.append(float(base_loss.detach()))
        row, tn, vp = validation_metrics(model, tx, ty, vx, vy, vw, vk, protocol["batch_size"])
        row.update(epoch=epoch, train_batch_mean_bce=float(np.mean(base_losses)),
                   train_batch_mean_positive_bce=float(np.mean(positive_losses)),
                   train_batch_mean_max_format_loss=float(np.mean(worst_losses)) if worst_losses else None,
                   max_format_coefficient=coefficient if worst_losses else 0.0,
                   max_format_families_processed=len(groups) if worst_losses else 0)
        history.append(row)
        states[epoch] = copy.deepcopy(model.state_dict())
        vtrace[f"training_negative_e{epoch:02}"] = tn.numpy()
        vtrace[f"validation_e{epoch:02}"] = vp.numpy()
        print(f"epoch={epoch} val_bce={row['validation_bce']:.6f} "
              f"val_tpr={row['validation_tpr']:.4f} "
              f"val_worst_fpr={row['validation_worst_fpr']:.4f}", flush=True)
    selection = guarded_epoch(history, **protocol["selection_guards"])
    checkpoint = directory / "EPOCHS.pt"
    torch.save({"version": VERSION, "stage": stage, "seed": seed, "split": split,
                "config": model.config, "states": states}, checkpoint)
    for name, epoch_key in (("bce", "bce_epoch"), ("guarded", "guarded_epoch")):
        torch.save({"version": VERSION, "stage": stage, "seed": seed, "split": split,
                    "config": model.config, "epoch": selection[epoch_key],
                    "state_dict": states[selection[epoch_key]]}, directory / f"MODEL_{name}.pt")
    for filename, arrays in (("MINING_TRACE.npz", trace), ("VALIDATION_TRACE.npz", vtrace)):
        with (directory / filename).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    replay = None
    old_stage = {"U0": "A0", "E1": "A2", "X1": "A3"}.get(stage)
    if old_stage:
        old = ROOT / protocol["old_ablation_directory"] / "runs" / f"{old_stage}_{seed}"
        saved = torch.load(old / "MODEL.pt", map_location="cpu", weights_only=True)
        previous = json.loads((old / "RESULT.json").read_text())["history"]
        replay = {
            "state_dict_exact": all(torch.equal(v, saved["state_dict"][k])
                                    for k, v in states[selection["bce_epoch"]].items()),
            "history_bce_exact": all(
                h["validation_bce"] == p["validation_document_weighted_bce"]
                and h["train_batch_mean_bce"] == p["train_batch_mean_bce"]
                for h, p in zip(history, previous)),
        }
        if not all(replay.values()):
            raise RuntimeError("Frozen baseline trajectory replay failed")
    write_json(directory / "MINING_AUDIT.json", mining)
    write_json(directory / "TRAINING.json", {
        "version": VERSION, "stage": stage, "seed": seed,
        "coefficient_name": coefficient_name, "coefficient": coefficient,
        "history": history, "epoch_selection": selection,
        "parameters": sum(p.numel() for p in model.parameters()),
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "checkpoint_sha256": sha(checkpoint),
        "selected_checkpoint_sha256": {s: sha(directory / f"MODEL_{s}.pt")
                                       for s in ("bce", "guarded")},
        "training_only": True, "calibration_or_test_inference_calls": 0,
        "positive_cache_audit": positive.audit(), "negative_cache_audit": negative.audit(),
        "frozen_baseline_replay": replay,
        "max_format_family_weights": family_weights.tolist(),
        "max_format_family_keys": [list(key) for key, _ in groups],
    })


def matrix(protocol):
    return [(s, c, seed) for s in protocol["stages"]
            for c in (list(protocol["coefficients"])[1:] if s.endswith("3") else ["L000"])
            for seed in protocol["seeds"]]


def select(out, protocol):
    if not (out / "TRAIN_EXECUTION.json").exists():
        raise RuntimeError("Complete frozen training matrix before selecting")
    choices = {}
    for stage, reference in protocol["coefficient_references"].items():
        candidates = {}
        for coefficient in protocol["coefficients"]:
            source = reference if coefficient == "L000" else stage
            candidates[coefficient] = [
                json.loads((out / "runs" / f"{source}_{coefficient}_{seed}" /
                            "TRAINING.json").read_text())["epoch_selection"]["guarded"]
                for seed in protocol["seeds"]]
        choices[stage] = select_coefficient(candidates, **protocol["selection_guards"])
    methods = {s: {"source_stage": s, "coefficient": "L000"}
               for s in protocol["stages"] if not s.endswith("3")}
    methods.update({s: {"source_stage": protocol["coefficient_references"][s]
                        if r["selected"] == "L000" else s, "coefficient": r["selected"]}
                    for s, r in choices.items()})
    write_json(out / "SELECTION.json", {
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "training_execution_sha256": sha(out / "TRAIN_EXECUTION.json"),
        "coefficient_choices": choices, "evaluation_methods": methods,
        "no_calibration_or_test_selection": True,
        "shared_coefficient_all_seeds_and_epoch_selectors": True,
    })
    print(json.dumps(choices, indent=2), flush=True)


def evaluation(model, positive, negative, split, batch):
    pi, ni, ci = (positive.indices(split["test"]), negative.indices(split["test"]),
                  negative.indices(split["calibration"]))
    p = infer(model, positive.scores[pi], batch).numpy()
    n = infer(model, negative.scores[ni], batch).numpy()
    c = infer(model, negative.scores[ci], batch).numpy()
    pk, nk, ck = ([positive.keys[i] for i in pi], [negative.keys[i] for i in ni],
                  [negative.keys[i] for i in ci])
    rows = {}
    for alpha in (0.01, 0.05):
        threshold = EmpiricalThreshold.fit(c, alpha)
        pp, nn, cc = (np.array(threshold.predict(v)) for v in (p, n, c))
        per = {a: {"tpr": float(pp[np.array([k[1] == a for k in pk])].mean()),
                   "fpr": float(nn[np.array([k[1] == a for k in nk])].mean())}
               for a in sorted({k[1] for k in pk})}
        rows[str(alpha)] = {
            "calibration": threshold.to_dict(), "calibration_fpr": float(cc.mean()),
            "tpr": float(pp.mean()), "fpr": float(nn.mean()), "per_attack": per,
            "worst_condition_fpr": max(r["fpr"] for r in per.values()),
        }
    return {"working_points": rows, "positive_abstain": int((~np.isfinite(p)).sum()),
            "negative_abstain": int((~np.isfinite(n)).sum()),
            "independent_human_test_fpr": None, "document_level_tpr_fpr": None}, {
                "positive": p, "negative": n, "calibration_negative": c,
                "keys": np.array(pk, dtype=str), "calibration_keys": np.array(ck, dtype=str)}


def evaluate_selected(out, protocol):
    selection = json.loads((out / "SELECTION.json").read_text())
    if selection["protocol_sha256"] != sha(out / "RUN_PROTOCOL.json"):
        raise RuntimeError("Protocol changed after selection")
    positive, negative = load_data()
    split = validate_split(json.loads((ROOT / protocol["split_path"]).read_text()),
                           positive, negative)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    completed = []
    for stage, choice in selection["evaluation_methods"].items():
        for seed in protocol["seeds"]:
            source = out / "runs" / f"{choice['source_stage']}_{choice['coefficient']}_{seed}"
            training = json.loads((source / "TRAINING.json").read_text())
            for selector in ("bce", "guarded"):
                checkpoint = source / f"MODEL_{selector}.pt"
                if sha(checkpoint) != training["selected_checkpoint_sha256"][selector]:
                    raise RuntimeError("Selected checkpoint changed")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = model_for(choice["source_stage"])
                model.load_state_dict(saved["state_dict"], strict=True)
                directory = out / "evaluation" / f"{stage}_{selector}_{seed}"
                directory.mkdir(parents=True, exist_ok=False)
                result, arrays = evaluation(model, positive, negative, split, protocol["batch_size"])
                with (directory / "PREDICTIONS.npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                write_json(directory / "RESULT.json", {
                    "stage": stage, "selector": selector, "seed": seed,
                    "source_run": source.name, "epoch": saved["epoch"],
                    "parameters": training["parameters"],
                    "checkpoint_sha256": sha(checkpoint),
                    "training_sha256": sha(source / "TRAINING.json"),
                    "selection_sha256": sha(out / "SELECTION.json"),
                    "predictions_sha256": sha(directory / "PREDICTIONS.npz"),
                    "evaluation": result,
                })
                completed.append(directory.name)
                print(f"evaluated={directory.name}", flush=True)
    write_json(out / "EVALUATION_EXECUTION.json", {
        "completed_endpoints": len(completed), "endpoints": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_sha256": sha(out / "SELECTION.json"),
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--phase", choices=("train", "select", "evaluate"), required=True)
    parser.add_argument("--one-stage")
    parser.add_argument("--coefficient", default="L000")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = json.loads((args.output / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    if args.phase == "select":
        select(args.output, protocol)
        return
    if args.phase == "evaluate":
        evaluate_selected(args.output, protocol)
        return
    if args.one_stage:
        if (args.one_stage, args.coefficient, args.seed) not in matrix(protocol):
            parser.error("Run outside frozen matrix")
        train_one(args.one_stage, args.coefficient, args.seed, args.output, protocol)
        return
    if (args.output / "runs").exists():
        raise FileExistsError("Training directory exists; no implicit resume/overwrite")
    started = time.perf_counter()
    rows = []
    for stage, coefficient, seed in matrix(protocol):
        name = f"{stage}_{coefficient}_{seed}"
        log = args.output / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", "train",
                   "--one-stage", stage, "--coefficient", coefficient, "--seed", str(seed),
                   "--output", str(args.output.resolve())]
        begin = time.perf_counter()
        with log.open("x") as stream:
            run = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        metadata = {"run": name, "returncode": run.returncode,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - begin, "command": command,
                    "log_sha256": sha(log), "protocol_sha256": sha(args.output / "RUN_PROTOCOL.json")}
        write_json(log.with_suffix(".metadata.json"), metadata)
        rows.append(metadata)
        print(f"{len(rows)}/{len(matrix(protocol))} {name} exit={run.returncode} "
              f"seconds={metadata['elapsed_seconds']:.2f}", flush=True)
        if run.returncode:
            raise RuntimeError(f"Run failed; inspect {log}")
    write_json(args.output / "TRAIN_EXECUTION.json", {
        "completed_runs": len(rows), "runs": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "max_child_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "protocol_sha256": sha(args.output / "RUN_PROTOCOL.json"),
        "calibration_or_test_inference_calls": 0,
    })


if __name__ == "__main__":
    main()
