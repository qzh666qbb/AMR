"""Audit caches or fit/evaluate experimental score-level MSA on explicit splits.

No parser, provider API, remote process or historical-result mutation.
All output is exploratory carrier-slot analysis, not a blind SWAN endpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.multiscale_data import (  # noqa: E402
    load_cached_scales, sha256_file, validate_split,
)
from src.core.multiscale_fusion import (  # noqa: E402
    EmpiricalThreshold, MODES, MultiScaleAttentionFusion, VERSION,
)

BASE = ROOT / "experiments/swan_multiview_development1"


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def partition(positive, negative, documents):
    pi, ni = positive.indices(documents), negative.indices(documents)
    scores = torch.cat((positive.scores[pi], negative.scores[ni]))
    labels = torch.cat((torch.ones(len(pi)), torch.zeros(len(ni))))
    keys = [positive.keys[i] for i in pi] + [negative.keys[i] for i in ni]
    # Equal weight per class and source document, not per perturbation slot.
    counts = {}
    for label, key in zip(labels.tolist(), keys):
        group = (label, key[0])
        counts[group] = counts.get(group, 0) + 1
    weights = torch.tensor([1 / counts[(label, key[0])]
                            for label, key in zip(labels.tolist(), keys)])
    return scores, labels, weights / weights.sum()


@torch.no_grad()
def infer(model, scores, batch_size=128):
    model.eval()
    outputs = [model(chunk)["statistic"] for chunk in scores.split(batch_size)]
    return torch.cat(outputs)


def train(model, positive, negative, split, epochs, batch_size, seed):
    train_x, train_y, train_w = partition(positive, negative, split["train"])
    val_x, val_y, val_w = partition(positive, negative, split["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in torch.randperm(len(train_x), generator=generator).split(batch_size):
            prediction = model(train_x[indices])
            if prediction["abstain"].any():
                raise ValueError("All-missing training row; audit before fitting")
            loss = F.binary_cross_entropy_with_logits(
                prediction["statistic"], train_y[indices], reduction="none")
            weight = train_w[indices]
            loss = (loss * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction = infer(model, val_x, batch_size)
        if not torch.isfinite(prediction).all():
            raise ValueError("All-missing validation row; audit before model selection")
        val_loss = float((F.binary_cross_entropy_with_logits(
            prediction, val_y, reduction="none") * val_w).sum())
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
        history.append({"epoch": epoch + 1, "train_batch_mean_bce": float(np.mean(losses)),
                        "validation_document_weighted_bce": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch={epoch + 1} validation_bce={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    return history


def cluster_interval(flags, keys, seed, repetitions):
    docs = sorted({key[0] for key in keys})
    if len(docs) < 2:
        return None
    counts = np.array([[sum(flag for flag, key in zip(flags, keys) if key[0] == doc),
                        sum(key[0] == doc for key in keys)] for doc in docs])
    rng = np.random.default_rng(seed)
    draws = counts[rng.integers(0, len(docs), (repetitions, len(docs)))].sum(1)
    return np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975]).tolist()


def evaluate(model, positive, negative, split, batch_size, seed, boot):
    # Freeze model first, then use only held-out negatives for thresholds.
    ni_cal = negative.indices(split["calibration"])
    pi_test = positive.indices(split["test"])
    ni_test = negative.indices(split["test"])
    calibration = infer(model, negative.scores[ni_cal], batch_size).tolist()
    pos = infer(model, positive.scores[pi_test], batch_size).tolist()
    neg = infer(model, negative.scores[ni_test], batch_size).tolist()
    pk = [positive.keys[i] for i in pi_test]
    nk = [negative.keys[i] for i in ni_test]
    result = {
        "test_positive_n": len(pos), "test_negative_n": len(neg),
        "test_documents": len(split["test"]),
        "positive_abstain": sum(not np.isfinite(x) for x in pos),
        "negative_abstain": sum(not np.isfinite(x) for x in neg),
        "working_points": {},
        "ci_method": "source-document bootstrap; model and threshold held fixed",
        "bootstrap_repetitions": boot,
        "independent_human_test_fpr": None,
        "document_level_tpr_fpr": None,
    }
    for alpha in (0.01, 0.05):
        threshold = EmpiricalThreshold.fit(calibration, alpha)
        row = {"calibration": threshold.to_dict()}
        if threshold.threshold is not None:
            pp, npred = threshold.predict(pos), threshold.predict(neg)
            row.update({
                "calibration_fpr": float(np.mean(threshold.predict(calibration))),
                "tpr": float(np.mean(pp)), "fpr": float(np.mean(npred)),
                "tpr_ci": cluster_interval(pp, pk, seed, boot),
                "fpr_ci": cluster_interval(npred, nk, seed, boot),
                "per_attack": {},
            })
            for attack in sorted({key[1] for key in pk} | {key[1] for key in nk}):
                p = [flag for flag, key in zip(pp, pk) if key[1] == attack]
                n = [flag for flag, key in zip(npred, nk) if key[1] == attack]
                row["per_attack"][attack] = {
                    "positive_n": len(p), "negative_n": len(n),
                    "tpr": float(np.mean(p)) if p else None,
                    "fpr": float(np.mean(n)) if n else None,
                }
        result["working_points"][str(alpha)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "fit"), default="audit")
    parser.add_argument("--positive-main", type=Path,
                        default=BASE / "dataset5_eval/FUSION_RECOVERY.json")
    parser.add_argument("--negative-main", type=Path,
                        default=BASE / "dataset5_paired_neg_eval/FUSION_RECOVERY.json")
    parser.add_argument("--positive-short", type=Path,
                        default=BASE / "dataset5_eval/window6s1_pos")
    parser.add_argument("--negative-short", type=Path,
                        default=BASE / "dataset5_paired_neg_eval/window6s1_neg")
    parser.add_argument("--split", type=Path,
                        help="JSON with train/validation/calibration/test document lists")
    parser.add_argument("--fusion", choices=MODES, default="attention")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--boot", type=int, default=1500)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.boot < 1500:
        parser.error("epochs/batch-size must be positive; boot must be >=1500")
    if args.mode == "fit" and args.split is None:
        parser.error("fit requires --split; no automatic repartitioning")
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    positive = load_cached_scales(args.positive_main, args.positive_short)
    negative = load_cached_scales(args.negative_main, args.negative_short)
    audit = {
        "version": VERSION, "mode": args.mode, "seed": args.seed,
        "torch_version": str(torch.__version__), "python_version": sys.version,
        "positive": positive.audit(), "negative": negative.audit(),
        "positive_only_union_keys": len(set(positive.keys) - set(negative.keys)),
        "negative_only_union_keys": len(set(negative.keys) - set(positive.keys)),
        "new_parser_calls": 0, "new_s2match_calls": 0,
        "formal_detection_validated": False,
    }
    split = None
    if args.split:
        split = validate_split(json.loads(args.split.read_text()), positive, negative)
        audit.update(split=split, split_sha256=sha256_file(args.split))
    args.output.mkdir(parents=True)
    write_json(args.output / "CACHE_AUDIT.json", audit)
    if args.mode == "audit":
        print(json.dumps({side: {k: v for k, v in audit[side].items() if k != "sources"}
                          for side in ("positive", "negative")}, ensure_ascii=False, indent=2))
        return
    model = MultiScaleAttentionFusion(mode=args.fusion)
    model.fit_normalizer(negative.scores[negative.indices(split["train"])])
    history = []
    if args.fusion in ("attention", "gate"):
        history = train(model, positive, negative, split, args.epochs,
                        args.batch_size, args.seed)
    checkpoint = args.output / "MODEL.pt"
    torch.save({"version": VERSION, "config": model.config, "state_dict": model.state_dict(),
                "split": split, "seed": args.seed}, checkpoint)
    report = {
        "version": VERSION, "status": "exploratory_oracle_slot_evaluation",
        "config": model.config, "parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "checkpoint_sha256": sha256_file(checkpoint),
        "cache_audit_sha256": sha256_file(args.output / "CACHE_AUDIT.json"),
        "limitations": [
            "Historical carrier slots use known source spans; not blind detection.",
            "Previously inspected data are development evidence, not fresh confirmation.",
            "Bank ordering inherited from legacy caches; no independent historical proof.",
            "No independent human-test FPR or document-level detection result.",
            "Candidate count/compute confounding remains; fusion does not solve it.",
        ],
        "evaluation": evaluate(model, positive, negative, split, args.batch_size,
                               args.seed, args.boot),
    }
    write_json(args.output / "RESULT.json", report)
    print(f"Saved exploratory {args.fusion} result to {args.output}", flush=True)


if __name__ == "__main__":
    main()
