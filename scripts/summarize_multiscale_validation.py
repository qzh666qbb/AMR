"""Replay frozen checkpoints and compare cached MSA runs by source document.

No fitting or threshold tuning. Bootstrap intervals condition on the existing
models, thresholds and five fixed seeds; they are exploratory, unadjusted CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiscale_fusion import BASE, infer, write_json  # noqa: E402
from scripts.run_multiscale_validation import DEFAULT, sha, verify_inputs  # noqa: E402
from src.core.multiscale_data import load_cached_scales, validate_split  # noqa: E402
from src.core.multiscale_fusion import EmpiricalThreshold, MultiScaleAttentionFusion  # noqa: E402


def auroc(positive, negative):
    """Tie-corrected Mann-Whitney AUC; abstentions (-inf) rank at the bottom."""
    values = np.concatenate((positive, negative))
    _, inverse = np.unique(values, return_inverse=True)
    p = np.bincount(inverse[:len(positive)], minlength=inverse.max() + 1)
    n = np.bincount(inverse[len(positive):], minlength=inverse.max() + 1)
    return float(np.sum(p * (np.cumsum(n) - 0.5 * n)) / (len(positive) * len(negative)))


def spread(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "by_seed": values.tolist()}


def source_counts(flags, keys, docs, mask):
    flags = np.atleast_2d(flags)
    indices = np.array([docs.index(key[0]) for key in keys])
    count = np.bincount(indices[mask], minlength=len(docs)).astype(float)
    success = np.array([
        np.bincount(indices[mask], weights=row[mask], minlength=len(docs))
        for row in flags
    ])
    return success, count


def bootstrap_rates(success, count, draws):
    """Return [seed, bootstrap] micro rates under one common document draw."""
    denominator = count[draws].sum(1)
    if (denominator <= 0).any():
        raise ValueError("Empty bootstrap denominator")
    return success[:, draws].sum(2) / denominator[None, :]


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def metric_summary(flags, keys, docs, mask, draws):
    success, count = source_counts(flags, keys, docs, mask)
    micro = success.sum(1) / count.sum()
    present = count > 0
    macro = (success[:, present] / count[present]).mean(1)
    boot = bootstrap_rates(success, count, draws)
    return {
        "n": int(count.sum()), "source_documents": int(present.sum()),
        "micro": spread(micro), "macro": spread(macro),
        "mean_micro_source_bootstrap_ci95": interval(boot.mean(0)),
    }


def delta_summary(left, right, keys, docs, mask, draws):
    # A deterministic right-hand baseline broadcasts to all learned seeds.
    difference = np.atleast_2d(left) - np.atleast_2d(right)
    success, count = source_counts(difference, keys, docs, mask)
    value = success.sum(1) / count.sum()
    boot = bootstrap_rates(success, count, draws)
    return {
        "micro_delta": spread(value),
        "mean_micro_delta_ci95": interval(boot.mean(0)),
        "per_seed_micro_delta_ci95": [interval(row) for row in boot],
        "positive_delta_seed_count": int((value > 0).sum()),
        "zero_delta_seed_count": int((value == 0).sum()),
        "negative_delta_seed_count": int((value < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.input
    for name in ("SUMMARY.json", "PREDICTIONS.npz"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}")
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    split = json.loads((ROOT / protocol["split_path"]).read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    positive = load_cached_scales(BASE / "dataset5_eval/FUSION_RECOVERY.json",
                                  BASE / "dataset5_eval/window6s1_pos")
    negative = load_cached_scales(BASE / "dataset5_paired_neg_eval/FUSION_RECOVERY.json",
                                  BASE / "dataset5_paired_neg_eval/window6s1_neg")
    validate_split(split, positive, negative)
    pidx, nidx = positive.indices(split["test"]), negative.indices(split["test"])
    cidx = negative.indices(split["calibration"])
    pk = [positive.keys[i] for i in pidx]
    nk = [negative.keys[i] for i in nidx]
    assert pk == nk
    test_p, test_n, cal_n = positive.scores[pidx], negative.scores[nidx], negative.scores[cidx]
    reference = MultiScaleAttentionFusion(mode="mean")
    reference.fit_normalizer(negative.scores[negative.indices(split["train"])])
    docs = split["test"]
    draws = np.random.default_rng(protocol["paired_bootstrap_seed"]).integers(
        0, len(docs), (protocol["paired_bootstrap_repetitions"], len(docs)))
    alphas = [str(protocol["primary_alpha"]), str(protocol["secondary_alpha"])]
    modes = protocol["deterministic_modes"] + protocol["learned_modes"]
    masks = {"all": np.ones(len(pk), dtype=bool)}
    masks.update({a: np.array([k[1] == a for k in pk]) for a in sorted({k[1] for k in pk})})
    run_info, arrays, model_scores, flags = {}, {}, {}, {}
    expected_runs = []
    for mode in modes:
        seeds = protocol["seeds"] if mode in protocol["learned_modes"] else protocol["seeds"][:1]
        model_scores[mode] = []
        flags[mode] = {alpha: {"positive": [], "negative": []} for alpha in alphas}
        for seed in seeds:
            name = f"{mode}_{seed}"
            expected_runs.append(name)
            directory = out / "runs" / name
            result = json.loads((directory / "RESULT.json").read_text())
            checkpoint = directory / "MODEL.pt"
            assert sha(checkpoint) == result["checkpoint_sha256"]
            assert sha(directory / "CACHE_AUDIT.json") == result["cache_audit_sha256"]
            audit = json.loads((directory / "CACHE_AUDIT.json").read_text())
            assert audit["split_sha256"] == protocol["split_sha256"]
            assert audit["positive"] == positive.audit()
            assert audit["negative"] == negative.audit()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["split"] == split and saved["seed"] == seed
            assert saved["config"]["mode"] == mode
            model = MultiScaleAttentionFusion(**saved["config"])
            model.load_state_dict(saved["state_dict"], strict=True)
            for attr in ("center", "spread", "normalizer_count", "normalizer_ready"):
                assert torch.equal(getattr(model, attr), getattr(reference, attr))
            p = infer(model, test_p).numpy()
            n = infer(model, test_n).numpy()
            c = infer(model, cal_n).numpy()
            model_scores[mode].append((p, n))
            for label, values in (("positive", p), ("negative", n), ("calibration_negative", c)):
                arrays[f"{name}__{label}"] = values
            history = result["history"]
            best = min(history, key=lambda r: r["validation_document_weighted_bce"]) if history else None
            metadata = json.loads((out / "logs" / f"{name}.metadata.json").read_text())
            assert metadata["returncode"] == 0
            assert metadata["protocol_sha256"] == sha(out / "RUN_PROTOCOL.json")
            run_info[name] = {
                "mode": mode, "seed": seed, "parameters": result["parameters"],
                "checkpoint_sha256": sha(checkpoint), "result_sha256": sha(directory / "RESULT.json"),
                "elapsed_seconds": metadata["elapsed_seconds"],
                "best_validation_epoch": best["epoch"] if best else None,
                "best_validation_bce": best["validation_document_weighted_bce"] if best else None,
                "positive_abstain": int((~np.isfinite(p)).sum()),
                "negative_abstain": int((~np.isfinite(n)).sum()),
                "calibration_negative_abstain": int((~np.isfinite(c)).sum()),
                "auroc": auroc(p, n), "working_points": {},
            }
            for alpha in alphas:
                row = result["evaluation"]["working_points"][alpha]
                threshold = EmpiricalThreshold(**row["calibration"])
                assert EmpiricalThreshold.fit(c, float(alpha)) == threshold
                pp = np.array(threshold.predict(p), dtype=float)
                nn = np.array(threshold.predict(n), dtype=float)
                assert abs(pp.mean() - row["tpr"]) < 1e-12
                assert abs(nn.mean() - row["fpr"]) < 1e-12
                assert abs(np.mean(threshold.predict(c)) - row["calibration_fpr"]) < 1e-12
                for attack, subset in row["per_attack"].items():
                    assert abs(pp[masks[attack]].mean() - subset["tpr"]) < 1e-12
                    assert abs(nn[masks[attack]].mean() - subset["fpr"]) < 1e-12
                flags[mode][alpha]["positive"].append(pp)
                flags[mode][alpha]["negative"].append(nn)
                run_info[name]["working_points"][alpha] = {
                    "tpr": row["tpr"], "fpr": row["fpr"],
                    "calibration_fpr": row["calibration_fpr"],
                    "calibration_threshold": threshold.threshold,
                    "calibration_n": threshold.n,
                }
    assert sorted(p.name for p in (out / "runs").iterdir()) == sorted(expected_runs)
    summary = {
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "analysis_script_sha256": sha(__file__),
        "evidence_scope": protocol["evidence_scope"],
        "verified_runs": len(run_info), "runs": run_info,
        "data": {
            name: {"source_documents": len(part), "positive_n": len(positive.indices(part)),
                   "negative_n": len(negative.indices(part))}
            for name, part in split.items()
        },
        "analysis": {
            "bootstrap_repetitions": len(draws), "bootstrap_seed": protocol["paired_bootstrap_seed"],
            "ci": "95% paired source-document percentile bootstrap; fixed models, thresholds and seeds",
            "multiplicity": "unadjusted exploratory intervals; not confirmatory hypothesis tests",
            "seed_average_is_ensemble": False, "abstentions_in_denominator": True,
            "auroc": "tie-corrected pooled slots; abstentions rank at negative infinity",
            "independent_human_test_fpr": None, "document_level_tpr_fpr": None,
            "checkpoint_replay_tpr_fpr_matches_original": True,
            "all_normalizers_equal_training_negatives_recomputation": True,
        },
        "methods": {}, "attention_minus": {},
    }
    for mode in modes:
        group = {"seed_count": len(model_scores[mode]), "working_points": {},
                 "auroc": spread([auroc(p, n) for p, n in model_scores[mode]])}
        for alpha in alphas:
            for side in flags[mode][alpha]:
                flags[mode][alpha][side] = np.stack(flags[mode][alpha][side])
            group["working_points"][alpha] = {}
            for attack, mask in masks.items():
                group["working_points"][alpha][attack] = {
                    "tpr": metric_summary(flags[mode][alpha]["positive"], pk, docs, mask, draws),
                    "fpr": metric_summary(flags[mode][alpha]["negative"], nk, docs, mask, draws),
                }
        group["per_attack_auroc"] = {
            attack: spread([auroc(p[mask], n[mask]) for p, n in model_scores[mode]])
            for attack, mask in masks.items() if attack != "all"
        }
        summary["methods"][mode] = group
    for mode in (m for m in modes if m != "attention"):
        summary["attention_minus"][mode] = {}
        for alpha in alphas:
            summary["attention_minus"][mode][alpha] = {}
            for attack, mask in masks.items():
                summary["attention_minus"][mode][alpha][attack] = {
                    metric: delta_summary(flags["attention"][alpha][side], flags[mode][alpha][side],
                                          keys, docs, mask, draws)
                    for metric, side, keys in (("tpr", "positive", pk), ("fpr", "negative", nk))
                }
    primary = str(protocol["primary_alpha"])
    decisions = {}
    for mode in protocol["primary_comparators"]:
        row = summary["attention_minus"][mode][primary]["all"]
        decisions[mode] = {
            "tpr_mean_ci_lower_positive": row["tpr"]["mean_micro_delta_ci95"][0] > 0,
            "fpr_mean_ci_upper_nonpositive": row["fpr"]["mean_micro_delta_ci95"][1] <= 0,
            "every_seed_tpr_delta_positive": row["tpr"]["positive_delta_seed_count"] == len(protocol["seeds"]),
        }
    attention_fpr = summary["methods"]["attention"]["working_points"][primary]["all"]["fpr"]["micro"]
    nominal_ok = attention_fpr["max"] <= float(primary)
    passed = nominal_ok and all(all(check.values()) for check in decisions.values())
    summary["engineering_gain_rule"] = {
        "status": "gain_established_within_development_scope" if passed else "gain_not_established",
        "comparators": decisions, "every_attention_seed_test_fpr_le_nominal": nominal_ok,
    }
    summary["execution"] = {
        group: json.loads((out / f"{group}_EXECUTION.json").read_text())
        for group in ("attention", "gate", "baselines")
    }
    # Identifiers and predictions only: no source/generated text is exported.
    arrays["keys"] = np.array(pk, dtype=str)
    arrays["calibration_keys"] = np.array([negative.keys[i] for i in cidx], dtype=str)
    with (out / "PREDICTIONS.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    summary["predictions_sha256"] = sha(out / "PREDICTIONS.npz")
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps({
        "runs_verified": summary["verified_runs"], "data": summary["data"],
        "decision": summary["engineering_gain_rule"],
        "methods": {
            mode: {alpha: {
                metric: summary["methods"][mode]["working_points"][alpha]["all"][metric]["micro"]
                for metric in ("tpr", "fpr")
            } for alpha in alphas}
            for mode in modes
        },
    }, indent=2))


if __name__ == "__main__":
    main()
