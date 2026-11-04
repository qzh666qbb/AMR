"""Validate conservative operating points for frozen A2/A3 score-fusion models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiscale_ablation import load_data, make_model
from scripts.run_multiscale_fusion import infer, write_json
from scripts.run_multiscale_validation import sha, verify_inputs
from scripts.summarize_multiscale_validation import delta_summary, metric_summary, spread
from src.core.multiscale_data import validate_split
from src.core.multiscale_fusion import EmpiricalThreshold

DEFAULT = ROOT / "experiments/swan_multiscale_attention_development1/operating_points1"
POLICIES = ("pooled_05", "pooled_03", "pooled_02", "condition_max_05")


def thresholds(scores, keys, policy):
    alpha = {"pooled_05": .05, "pooled_03": .03, "pooled_02": .02,
             "condition_max_05": .05}[policy]
    pooled = EmpiricalThreshold.fit(scores, alpha)
    sources = {"pooled": pooled.to_dict()}
    if policy == "condition_max_05":
        for attack in sorted({key[1] for key in keys}):
            sub = scores[np.array([key[1] == attack for key in keys])]
            sources[attack] = EmpiricalThreshold.fit(sub, .05).to_dict()
    available = all(s["threshold"] is not None for s in sources.values())
    return {"available": available,
            "threshold": max(s["threshold"] for s in sources.values()) if available else None,
            "sources": sources, "unified_inference_threshold": True}


def masks_for(keys):
    return {"all": np.ones(len(keys), dtype=bool),
            **{a: np.array([k[1] == a for k in keys]) for a in sorted({k[1] for k in keys})}}


def point_metrics(positive, negative, keys):
    """Scalar partition rates for a single model; no bootstrap selection."""
    return {
        attack: {"tpr": float(positive[mask].mean()) if positive is not None else None,
                 "fpr": float(negative[mask].mean()), "negative_n": int(mask.sum())}
        for attack, mask in masks_for(keys).items()
    }


def mining_diagnostics(base, stages, seeds):
    output = {}
    for stage in stages:
        per_epoch = []
        for seed in seeds:
            directory = base / "runs" / f"{stage}_{seed}"
            trace = np.load(directory / "MINING_TRACE.npz", allow_pickle=False)
            keys, labels = trace["keys"], trace["labels"]
            negative = labels == 0
            original_mass = trace["base_weights"][negative].sum()
            for epoch in range(26, 31):
                selected = trace[f"selected_e{epoch}"]
                weights = trace[f"weights_e{epoch}"]
                assert not selected[~negative].any()
                assert abs(weights[negative].sum() - original_mass) < 1e-7
                families = {}
                for key in keys[selected]:
                    family = (key[0], key[2])
                    families[family] = families.get(family, 0) + 1
                count = int(selected.sum())
                row = {
                    "seed": seed, "epoch": epoch, "selected_rows": count,
                    "selected_carrier_families": len(families),
                    "redundant_row_selection_fraction": (count - len(families)) / count,
                    "rows_in_multiply_selected_families_fraction":
                        sum(n for n in families.values() if n > 1) / count,
                    "per_condition": {},
                }
                for attack in sorted(set(keys[negative, 1])):
                    mask = negative & (keys[:, 1] == attack)
                    row["per_condition"][attack] = {
                        "original_row_share": float(mask.sum() / negative.sum()),
                        "selected_row_share": float((selected & mask).sum() / count),
                        "negative_weight_share": float(weights[mask].sum() / weights[negative].sum()),
                    }
                per_epoch.append(row)
        output[stage] = {
            "scope": "training_trace_epochs26_to30_only", "epochs_by_seed": per_epoch,
            "redundant_row_selection_fraction": spread([x["redundant_row_selection_fraction"] for x in per_epoch]),
            "rows_in_multiply_selected_families_fraction": spread([
                x["rows_in_multiply_selected_families_fraction"] for x in per_epoch]),
            "per_condition": {
                attack: {metric: spread([x["per_condition"][attack][metric] for x in per_epoch])
                         for metric in ("original_row_share", "selected_row_share", "negative_weight_share")}
                for attack in per_epoch[0]["per_condition"]
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output
    for name in ("SUMMARY.json", "PREDICTIONS.npz", "SELECTION.json", "MINING_DIAGNOSTICS.json"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    positive, negative = load_data()
    split = validate_split(json.loads((ROOT / protocol["split_path"]).read_text()), positive, negative)
    old = ROOT / protocol["ablation_directory"]
    previous = np.load(old / "PREDICTIONS.npz", allow_pickle=False)
    keys = {part: [negative.keys[i] for i in negative.indices(docs)] for part, docs in split.items()}
    for part in ("train", "validation", "test"):
        assert keys[part] == [positive.keys[i] for i in positive.indices(split[part])]
    arrays, runs, flags, selection = {}, {}, {}, {}
    for stage in ("A2", "A3"):
        flags[stage] = {policy: {part: {"positive": [], "negative": []} for part in split}
                        for policy in POLICIES}
        for seed in protocol["seeds"]:
            name = f"{stage}_{seed}"
            directory = old / "runs" / name
            saved = torch.load(directory / "MODEL.pt", weights_only=True, map_location="cpu")
            assert saved["stage"] == stage and saved["split"] == split and saved["seed"] == seed
            model = make_model(stage)
            model.load_state_dict(saved["state_dict"], strict=True)
            scores = {}
            for part, docs in split.items():
                scores[part] = {"negative": infer(model, negative.scores[negative.indices(docs)]).numpy()}
                if part != "calibration":
                    scores[part]["positive"] = infer(model, positive.scores[positive.indices(docs)]).numpy()
                for side, values in scores[part].items():
                    arrays[f"{name}__{part}_{side}"] = values
            for side in ("positive", "negative"):
                assert np.array_equal(scores["test"][side], previous[f"{name}__{side}"])
            assert np.array_equal(scores["calibration"]["negative"], previous[f"{name}__calibration_negative"])
            record = {"stage": stage, "seed": seed, "checkpoint_sha256": sha(directory / "MODEL.pt"), "policies": {}}
            for policy in POLICIES:
                threshold = thresholds(scores["calibration"]["negative"], keys["calibration"], policy)
                assert threshold["available"], "Required condition lacks calibration resolution"
                tau = threshold["threshold"]
                by_part = {}
                for part, sides in scores.items():
                    pred = {side: np.isfinite(value) & (value > tau) for side, value in sides.items()}
                    pp = pred.get("positive")
                    by_part[part] = point_metrics(pp, pred["negative"], keys[part])
                    flags[stage][policy][part]["negative"].append(pred["negative"].astype(float))
                    if pp is not None:
                        flags[stage][policy][part]["positive"].append(pp.astype(float))
                record["policies"][policy] = {"calibration": threshold, "partition_metrics": by_part}
                if policy == "pooled_05":
                    result = json.loads((directory / "RESULT.json").read_text())
                    prior = result["evaluation"]["working_points"]["0.05"]
                    assert tau == prior["calibration"]["threshold"]
                    assert by_part["test"]["all"]["tpr"] == prior["tpr"]
                    assert by_part["test"]["all"]["fpr"] == prior["fpr"]
            runs[name] = record
        # Selection reads only validation and calibration rates, never test.
        eligible = []
        status = {}
        for policy in POLICIES:
            allowed = True
            utility = []
            reasons = []
            for seed in protocol["seeds"]:
                row = runs[f"{stage}_{seed}"]["policies"][policy]
                utility.append(row["partition_metrics"]["validation"]["all"]["tpr"])
                for part in ("calibration", "validation"):
                    for attack, point in row["partition_metrics"][part].items():
                        if point["fpr"] > .05:
                            allowed = False
                            reasons.append({"seed": seed, "partition": part,
                                            "condition": attack, "fpr": point["fpr"]})
            status[policy] = {"eligible": allowed, "validation_mean_tpr": float(np.mean(utility)),
                              "violations": reasons}
            if allowed:
                eligible.append(policy)
        chosen = max(eligible, key=lambda p: status[p]["validation_mean_tpr"]) if eligible else None
        selection[stage] = {"chosen_policy": chosen, "candidates": status,
                            "selection_used_test": False, "same_policy_for_all_seeds": True}
    write_json(out / "SELECTION.json", selection)
    summary = {
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "analysis_script_sha256": sha(__file__),
        "evidence_scope": protocol["evidence_scope"], "runs": runs, "selection": selection,
        "selection_sha256": sha(out / "SELECTION.json"),
        "data": {part: {"documents": len(split[part]), "negative_rows": len(k),
                        "carrier_families": len({(x[0], x[2]) for x in k}),
                        "rows_per_condition": {a: int(m.sum()) for a, m in masks_for(k).items() if a != "all"}}
                 for part, k in keys.items()},
        "analysis": {
            "bootstrap_repetitions": 5000, "bootstrap_seed": 20261030,
            "intervals": "paired source-document percentile; fixed models, thresholds, policies and seeds",
            "adaptive_development": True, "iid_fpr_guarantee": False,
            "independent_human_test_fpr": None, "blind_document_tpr_fpr": None,
            "baseline_and_prediction_replay_exact": True,
        },
        "candidates": {}, "comparisons": {}, "calibration_resolution": {},
    }
    draws = np.random.default_rng(20261030).integers(0, len(split["test"]), (5000, len(split["test"])))
    for stage in flags:
        summary["candidates"][stage] = {}
        for policy in POLICIES:
            by_part = {}
            for part in split:
                f = flags[stage][policy][part]
                f["negative"] = np.stack(f["negative"])
                if f["positive"]:
                    f["positive"] = np.stack(f["positive"])
                else:
                    f["positive"] = None
                rates = {}
                for attack, mask in masks_for(keys[part]).items():
                    rates[attack] = {"fpr": spread(f["negative"][:, mask].mean(1))}
                    if f["positive"] is not None:
                        rates[attack]["tpr"] = spread(f["positive"][:, mask].mean(1))
                if part == "test":
                    rates = {attack: {
                        metric: metric_summary(f[side], keys[part], split[part], mask, draws)
                        for metric, side in (("tpr", "positive"), ("fpr", "negative"))
                    } for attack, mask in masks_for(keys[part]).items()}
                by_part[part] = rates
            summary["candidates"][stage][policy] = {
                "partition_metrics": by_part, "eligible": selection[stage]["candidates"][policy]["eligible"],
                "selected": selection[stage]["chosen_policy"] == policy,
                "thresholds_by_seed": [runs[f"{stage}_{seed}"]["policies"][policy]["calibration"]
                                       for seed in protocol["seeds"]],
            }
    comparisons = [("A3", p, "A3", "pooled_05") for p in POLICIES if p != "pooled_05"]
    if selection["A3"]["chosen_policy"] and selection["A2"]["chosen_policy"]:
        comparisons.append(("A3", selection["A3"]["chosen_policy"], "A2", selection["A2"]["chosen_policy"]))
    for left_stage, left_policy, right_stage, right_policy in comparisons:
        name = f"{left_stage}_{left_policy}_minus_{right_stage}_{right_policy}"
        left, right = flags[left_stage][left_policy]["test"], flags[right_stage][right_policy]["test"]
        summary["comparisons"][name] = {
            attack: {metric: delta_summary(left[side], right[side], keys["test"], split["test"], mask, draws)
                     for metric, side in (("tpr", "positive"), ("fpr", "negative"))}
            for attack, mask in masks_for(keys["test"]).items()
        }
    for attack, mask in masks_for(keys["calibration"]).items():
        summary["calibration_resolution"][attack] = {
            "rows": int(mask.sum()), "documents": len({keys["calibration"][i][0] for i in np.flatnonzero(mask)}),
            "one_exceedance_fraction": 1 / int(mask.sum()),
            "alpha_01_rank": int(np.ceil((mask.sum() + 1) * .99)),
            "alpha_01_available_by_rank": bool(np.ceil((mask.sum() + 1) * .99) <= mask.sum()),
        }
    for part, k in keys.items():
        arrays[f"keys__{part}"] = np.array(k, dtype=str)
    with (out / "PREDICTIONS.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    summary["predictions_sha256"] = sha(out / "PREDICTIONS.npz")
    write_json(out / "MINING_DIAGNOSTICS.json", mining_diagnostics(old, ("A2", "A3"), protocol["seeds"]))
    summary["mining_diagnostics_sha256"] = sha(out / "MINING_DIAGNOSTICS.json")
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps({"selection": selection, "test": {
        stage: {policy: {metric: row["partition_metrics"]["test"]["all"][metric]["micro"]
                        for metric in ("tpr", "fpr")}
                for policy, row in policies.items()}
        for stage, policies in summary["candidates"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
