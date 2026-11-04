"""Replay the four frozen ablations and calculate paired development changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiscale_ablation import DEFAULT, STAGES, load_data, make_model
from scripts.run_multiscale_fusion import infer, partition, write_json
from scripts.run_multiscale_validation import sha, verify_inputs
from scripts.summarize_multiscale_validation import auroc, delta_summary, metric_summary, spread
from src.core.multiscale_data import validate_split
from src.core.multiscale_fusion import EmpiricalThreshold


def verify_mining(directory, positive, negative, split):
    """Independent NumPy audit: training keys, cutoffs, ties and weight mass."""
    _, labels, base = partition(positive, negative, split["train"])
    keys = ([positive.keys[i] for i in positive.indices(split["train"])]
            + [negative.keys[i] for i in negative.indices(split["train"])])
    trace = np.load(directory / "MINING_TRACE.npz", allow_pickle=False)
    assert np.array_equal(trace["keys"], np.array(keys))
    assert np.array_equal(trace["labels"], labels.numpy())
    assert np.array_equal(trace["base_weights"], base.numpy())
    neg_idx = np.flatnonzero(labels.numpy() == 0)
    assert np.array_equal(trace["negative_indices"], neg_idx)
    logs = json.loads((directory / "MINING_AUDIT.json").read_text())
    assert [a["epoch"] for a in logs] == list(range(6, 31))
    checked_groups = 0
    for audit in logs:
        epoch = audit["epoch"]
        scores = trace[f"statistics_e{epoch:02}"]
        actual_weights = trace[f"weights_e{epoch:02}"]
        actual_selected = trace[f"selected_e{epoch:02}"]
        expected_weights = base.numpy().copy()
        expected_selected = np.zeros(len(labels), dtype=bool)
        assert np.isfinite(scores).all()
        for doc in split["train"]:
            local = np.array([i for i, row in enumerate(neg_idx) if keys[row][0] == doc])
            rows = neg_idx[local]
            cutoff = np.sort(scores[local])[len(rows) - math.ceil(len(rows) * .1)]
            selected = rows[scores[local] >= cutoff]
            expected_selected[selected] = True
            expected_weights[selected] *= 3
            expected_weights[rows] *= base.numpy()[rows].sum() / expected_weights[rows].sum()
            row = audit["per_document"][doc]
            assert row["selected_with_ties"] == len(selected)
            assert row["cutoff"] == float(cutoff)
            assert abs(actual_weights[rows].sum() - base.numpy()[rows].sum()) < 1e-7
            checked_groups += 1
        assert np.array_equal(actual_selected, expected_selected)
        assert np.allclose(actual_weights, expected_weights, rtol=1e-6, atol=1e-8)
        assert np.array_equal(actual_weights[labels.numpy() == 1], base.numpy()[labels.numpy() == 1])
        selected_keys = [list(k) for k, selected in zip(keys, expected_selected) if selected]
        digest = hashlib.sha256(json.dumps(selected_keys, separators=(",", ":")).encode()).hexdigest()
        assert digest == audit["selected_keys_sha256"]
    return {"epochs_checked": len(logs), "source_document_epoch_groups_checked": checked_groups,
            "training_only_keys_match": True, "selection_and_mass_match": True,
            "positive_weights_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.input
    for name in ("SUMMARY.json", "PREDICTIONS.npz"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    positive, negative = load_data()
    split = validate_split(json.loads((ROOT / protocol["split_path"]).read_text()), positive, negative)
    pidx, nidx = positive.indices(split["test"]), negative.indices(split["test"])
    cidx = negative.indices(split["calibration"])
    keys = [positive.keys[i] for i in pidx]
    assert keys == [negative.keys[i] for i in nidx]
    reference = make_model("A0")
    reference.fit_normalizer(negative.scores[negative.indices(split["train"])])
    masks = {"all": np.ones(len(keys), dtype=bool)}
    masks.update({a: np.array([k[1] == a for k in keys]) for a in sorted({k[1] for k in keys})})
    draws = np.random.default_rng(protocol["paired_bootstrap_seed"]).integers(
        0, len(split["test"]), (protocol["paired_bootstrap_repetitions"], len(split["test"])))
    arrays, flags, scores_by_stage, run_info = {}, {}, {}, {}
    mining_checks = {}
    for stage in STAGES:
        flags[stage] = {alpha: {"positive": [], "negative": []} for alpha in ("0.01", "0.05")}
        scores_by_stage[stage] = []
        for seed in protocol["seeds"]:
            name = f"{stage}_{seed}"
            directory = out / "runs" / name
            result = json.loads((directory / "RESULT.json").read_text())
            assert sha(directory / "MODEL.pt") == result["checkpoint_sha256"]
            assert sha(directory / "CACHE_AUDIT.json") == result["cache_audit_sha256"]
            assert result["protocol_sha256"] == sha(out / "RUN_PROTOCOL.json")
            saved = torch.load(directory / "MODEL.pt", weights_only=True, map_location="cpu")
            assert saved["split"] == split and saved["stage"] == stage and saved["seed"] == seed
            torch.manual_seed(seed)
            model = make_model(stage)
            assert saved["config"] == model.config
            model.load_state_dict(saved["state_dict"], strict=True)
            for attr in ("center", "spread", "normalizer_count", "normalizer_ready"):
                assert torch.equal(getattr(model, attr), getattr(reference, attr))
            audit = json.loads((directory / "CACHE_AUDIT.json").read_text())
            assert audit["positive"] == positive.audit() and audit["negative"] == negative.audit()
            if stage in ("A2", "A3"):
                assert sha(directory / "MINING_TRACE.npz") == result["mining_trace_sha256"]
                assert sha(directory / "MINING_AUDIT.json") == result["mining_audit_sha256"]
                mining_checks[name] = verify_mining(directory, positive, negative, split)
            p = infer(model, positive.scores[pidx]).numpy()
            n = infer(model, negative.scores[nidx]).numpy()
            c = infer(model, negative.scores[cidx]).numpy()
            scores_by_stage[stage].append((p, n))
            for side, values in (("positive", p), ("negative", n), ("calibration_negative", c)):
                arrays[f"{name}__{side}"] = values
            metadata = json.loads((out / "logs" / f"{name}.metadata.json").read_text())
            assert metadata["returncode"] == 0
            assert sha(out / "logs" / f"{name}.log") == metadata["log_sha256"]
            best = min(result["history"], key=lambda r: r["validation_document_weighted_bce"])
            info = {
                "stage": stage, "seed": seed, "parameters": result["parameters"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "checkpoint_sha256": result["checkpoint_sha256"], "result_sha256": sha(directory / "RESULT.json"),
                "best_validation_epoch": best["epoch"],
                "best_validation_bce": best["validation_document_weighted_bce"],
                "baseline_v1_check": result["baseline_v1_check"],
                "positive_abstain": int((~np.isfinite(p)).sum()),
                "negative_abstain": int((~np.isfinite(n)).sum()),
                "auroc": auroc(p, n), "working_points": {},
            }
            for alpha in flags[stage]:
                row = result["evaluation"]["working_points"][alpha]
                threshold = EmpiricalThreshold(**row["calibration"])
                assert EmpiricalThreshold.fit(c, float(alpha)) == threshold
                pp, nn = np.array(threshold.predict(p), dtype=float), np.array(threshold.predict(n), dtype=float)
                assert abs(pp.mean() - row["tpr"]) < 1e-12 and abs(nn.mean() - row["fpr"]) < 1e-12
                for attack, subset in row["per_attack"].items():
                    assert abs(pp[masks[attack]].mean() - subset["tpr"]) < 1e-12
                    assert abs(nn[masks[attack]].mean() - subset["fpr"]) < 1e-12
                flags[stage][alpha]["positive"].append(pp)
                flags[stage][alpha]["negative"].append(nn)
                info["working_points"][alpha] = {
                    "tpr": row["tpr"], "fpr": row["fpr"], "calibration_fpr": row["calibration_fpr"],
                    "calibration_threshold": threshold.threshold, "calibration_n": threshold.n,
                }
            run_info[name] = info
    assert sorted(p.name for p in (out / "runs").iterdir()) == sorted(run_info)
    summary = {
        "version": protocol["protocol_version"], "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "analysis_script_sha256": sha(__file__), "evidence_scope": protocol["evidence_scope"],
        "verified_runs": len(run_info), "runs": run_info, "mining_checks": mining_checks,
        "data": {name: {"source_documents": len(part), "positive_n": len(positive.indices(part)),
                        "negative_n": len(negative.indices(part))} for name, part in split.items()},
        "analysis": {
            "checkpoint_replay_tpr_fpr_matches_original": True,
            "all_normalizers_equal_training_negatives_recomputation": True,
            "bootstrap_repetitions": len(draws), "bootstrap_seed": protocol["paired_bootstrap_seed"],
            "intervals": "95% paired source-document percentile; conditional on fixed models, thresholds and seeds",
            "multiplicity": "unadjusted adaptive development intervals",
            "seed_average_is_ensemble": False, "abstentions_in_denominator": True,
            "independent_human_test_fpr": None, "document_level_tpr_fpr": None,
        },
        "methods": {}, "comparisons": {}, "engineering_gain_rule": {},
        "execution": json.loads((out / "EXECUTION.json").read_text()),
    }
    for stage in STAGES:
        group = {"seed_count": 5, "auroc": spread([auroc(p, n) for p, n in scores_by_stage[stage]]),
                 "parameters": run_info[f"{stage}_{protocol['seeds'][0]}"]["parameters"], "working_points": {}}
        for alpha in flags[stage]:
            flags[stage][alpha] = {side: np.stack(rows) for side, rows in flags[stage][alpha].items()}
            group["working_points"][alpha] = {
                attack: {metric: metric_summary(flags[stage][alpha][side], keys, split["test"], mask, draws)
                         for metric, side in (("tpr", "positive"), ("fpr", "negative"))}
                for attack, mask in masks.items()
            }
        summary["methods"][stage] = group
    for left, right in (("A1", "A0"), ("A2", "A1"), ("A3", "A2"), ("A2", "A0"), ("A3", "A0")):
        name = f"{left}_minus_{right}"
        comparison = {
            alpha: {attack: {
                metric: delta_summary(flags[left][alpha][side], flags[right][alpha][side],
                                      keys, split["test"], mask, draws)
                for metric, side in (("tpr", "positive"), ("fpr", "negative"))
            } for attack, mask in masks.items()} for alpha in flags[left]
        }
        summary["comparisons"][name] = comparison
        row = comparison["0.01"]["all"]
        checks = {
            "tpr_delta_ci_lower_positive": row["tpr"]["mean_micro_delta_ci95"][0] > 0,
            "fpr_delta_ci_upper_nonpositive": row["fpr"]["mean_micro_delta_ci95"][1] <= 0,
            "every_seed_tpr_delta_positive": row["tpr"]["positive_delta_seed_count"] == 5,
            "every_new_stage_seed_fpr_le_nominal": summary["methods"][left]["working_points"]["0.01"]["all"]["fpr"]["micro"]["max"] <= .01,
        }
        summary["engineering_gain_rule"][name] = {
            "status": "stable_development_gain" if all(checks.values()) else "stable_gain_not_established",
            "checks": checks,
        }
    arrays["keys"] = np.array(keys, dtype=str)
    arrays["calibration_keys"] = np.array([negative.keys[i] for i in cidx], dtype=str)
    with (out / "PREDICTIONS.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    summary["predictions_sha256"] = sha(out / "PREDICTIONS.npz")
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps({
        stage: {alpha: {
            metric: summary["methods"][stage]["working_points"][alpha]["all"][metric]["micro"]
            for metric in ("tpr", "fpr")} for alpha in ("0.01", "0.05")}
        for stage in STAGES
    }, indent=2))


if __name__ == "__main__":
    main()
