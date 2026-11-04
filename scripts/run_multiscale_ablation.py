"""Run the four frozen score-fusion ablations, in order, on existing caches."""
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

from scripts.run_multiscale_fusion import BASE, evaluate, infer, partition, train, write_json
from scripts.run_multiscale_validation import sha, verify_inputs
from src.core.multiscale_ablation import ConfidenceFusion, FEATURES, VERSION, hard_negative_weights
from src.core.multiscale_data import load_cached_scales, validate_split
from src.core.multiscale_fusion import MultiScaleAttentionFusion

DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/ablation1"
STAGES = ("A0", "A1", "A2", "A3")


def load_data():
    return (
        load_cached_scales(BASE / "dataset5_eval/FUSION_RECOVERY.json",
                           BASE / "dataset5_eval/window6s1_pos"),
        load_cached_scales(BASE / "dataset5_paired_neg_eval/FUSION_RECOVERY.json",
                           BASE / "dataset5_paired_neg_eval/window6s1_neg"),
    )


def make_model(stage):
    if stage == "A0":
        return MultiScaleAttentionFusion(mode="gate")
    return ConfidenceFusion(mode="attention" if stage == "A3" else "gate")


def train_hard(model, positive, negative, split, protocol, seed, output):
    tx, ty, tw = partition(positive, negative, split["train"])
    vx, vy, vw = partition(positive, negative, split["validation"])
    keys = ([positive.keys[i] for i in positive.indices(split["train"])]
            + [negative.keys[i] for i in negative.indices(split["train"])])
    neg_idx = (ty == 0).nonzero().flatten()
    optimizer = torch.optim.AdamW(model.parameters(), **protocol["optimizer"])
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, history, mining_history = float("inf"), None, [], []
    traces = {"base_weights": tw.numpy(), "labels": ty.numpy(),
              "keys": np.array(keys, dtype=str), "negative_indices": neg_idx.numpy()}
    hard = protocol["hard_negative"]
    for epoch in range(1, protocol["epochs"] + 1):
        weights = tw
        if epoch > hard["warmup_epochs"]:
            scores = infer(model, tx[neg_idx], protocol["batch_size"])
            weights, selected, audit = hard_negative_weights(
                tw, ty, keys, scores, hard["fraction"], hard["multiplier"])
            audit["epoch"] = epoch
            audit["partition"] = "training_negatives_only"
            mining_history.append(audit)
            traces[f"statistics_e{epoch:02}"] = scores.numpy()
            traces[f"weights_e{epoch:02}"] = weights.numpy()
            traces[f"selected_e{epoch:02}"] = selected.numpy()
        model.train()
        losses = []
        for indices in torch.randperm(len(tx), generator=generator).split(protocol["batch_size"]):
            prediction = model(tx[indices])
            if prediction["abstain"].any():
                raise ValueError("All-missing training row")
            loss = F.binary_cross_entropy_with_logits(
                prediction["statistic"], ty[indices], reduction="none")
            weight = weights[indices]
            loss = (loss * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction = infer(model, vx, protocol["batch_size"])
        if not torch.isfinite(prediction).all():
            raise ValueError("All-missing validation row")
        val_loss = float((F.binary_cross_entropy_with_logits(
            prediction, vy, reduction="none") * vw).sum())
        history.append({"epoch": epoch, "train_batch_mean_bce": float(np.mean(losses)),
                        "validation_document_weighted_bce": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        print(f"epoch={epoch} validation_bce={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    with (output / "MINING_TRACE.npz").open("xb") as stream:
        np.savez_compressed(stream, **traces)
    write_json(output / "MINING_AUDIT.json", mining_history)
    return history


def run_one(stage, seed, out, protocol):
    verify_inputs(protocol)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    positive, negative = load_data()
    split = validate_split(json.loads((ROOT / protocol["split_path"]).read_text()),
                           positive, negative)
    output = out / "runs" / f"{stage}_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "version": VERSION, "stage": stage, "seed": seed, "split": split,
        "split_sha256": protocol["split_sha256"],
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "positive": positive.audit(), "negative": negative.audit(),
        "new_parser_calls": 0, "new_s2match_calls": 0,
        "formal_detection_validated": False, "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__), "python_version": sys.version,
    }
    write_json(output / "CACHE_AUDIT.json", audit)
    model = make_model(stage)
    model.fit_normalizer(negative.scores[negative.indices(split["train"])])
    if stage in ("A0", "A1"):
        history = train(model, positive, negative, split, protocol["epochs"],
                        protocol["batch_size"], seed)
    else:
        history = train_hard(model, positive, negative, split, protocol, seed, output)
    baseline_check = None
    if stage == "A0":
        old = ROOT / protocol["baseline_directory"] / f"gate_{seed}"
        saved = torch.load(old / "MODEL.pt", map_location="cpu", weights_only=True)
        baseline_check = {
            "state_dict_tensor_exact": all(torch.equal(value, saved["state_dict"][name])
                                           for name, value in model.state_dict().items()),
            "history_exact": history == json.loads((old / "RESULT.json").read_text())["history"],
        }
        if not all(baseline_check.values()):
            raise RuntimeError("A0 failed frozen v1 baseline replay")
    checkpoint = output / "MODEL.pt"
    torch.save({"version": VERSION, "stage": stage, "config": model.config,
                "state_dict": model.state_dict(), "split": split, "seed": seed}, checkpoint)
    evaluation = evaluate(model, positive, negative, split, protocol["batch_size"],
                          seed, protocol["run_bootstrap_repetitions"])
    if stage == "A0":
        old_evaluation = json.loads((old / "RESULT.json").read_text())["evaluation"]
        baseline_check["evaluation_exact"] = evaluation == old_evaluation
        if not baseline_check["evaluation_exact"]:
            raise RuntimeError("A0 evaluation differs from v1")
    result = {
        "version": VERSION, "stage": stage, "seed": seed,
        "status": "adaptive_development_oracle_slot_ablation",
        "config": model.config, "features": list(FEATURES) if stage != "A0" else list(FEATURES[:2]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "evaluation": evaluation,
        "baseline_v1_check": baseline_check,
        "checkpoint_sha256": sha(checkpoint),
        "cache_audit_sha256": sha(output / "CACHE_AUDIT.json"),
        "mining_trace_sha256": sha(output / "MINING_TRACE.npz") if stage in ("A2", "A3") else None,
        "mining_audit_sha256": sha(output / "MINING_AUDIT.json") if stage in ("A2", "A3") else None,
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
    }
    write_json(output / "RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--one-stage", choices=STAGES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = json.loads((args.output / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    if args.one_stage:
        if args.seed not in protocol["seeds"]:
            parser.error("Seed not in frozen protocol")
        run_one(args.one_stage, args.seed, args.output, protocol)
        return
    if (args.output / "runs").exists():
        parser.error("Runs directory exists; use a new protocol/output directory")
    begin = time.perf_counter()
    for stage in STAGES:
        stage_begin = time.perf_counter()
        rows = []
        for seed in protocol["seeds"]:
            name = f"{stage}_{seed}"
            log = args.output / "logs" / f"{name}.log"
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--one-stage", stage,
                       "--seed", str(seed), "--output", str(args.output.resolve())]
            started = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            with log.open("x") as stream:
                run = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            metadata = {
                "run": name, "stage": stage, "seed": seed,
                "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - start, "returncode": run.returncode,
                "command": command, "log_sha256": sha(log),
                "protocol_sha256": sha(args.output / "RUN_PROTOCOL.json"),
            }
            write_json(log.with_suffix(".metadata.json"), metadata)
            print(f"{name} exit={run.returncode} elapsed={metadata['elapsed_seconds']:.2f}s", flush=True)
            if run.returncode:
                raise RuntimeError(f"Run failed; inspect {log}")
            rows.append(metadata)
        write_json(args.output / f"{stage}_EXECUTION.json", {
            "stage": stage, "runs": rows, "elapsed_seconds": time.perf_counter() - stage_begin,
        })
        print(f"Stage {stage} completed", flush=True)
    write_json(args.output / "EXECUTION.json", {
        "completed_runs": 20, "elapsed_seconds": time.perf_counter() - begin,
        "maximum_child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "rss_platform": sys.platform, "device": "CPU", "torch_threads": 2,
    })


if __name__ == "__main__":
    main()
