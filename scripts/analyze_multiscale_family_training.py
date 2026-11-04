"""Independently audit frozen training, replay endpoints and report paired CIs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multiscale_ablation import load_data
from scripts.run_multiscale_family_training import DEFAULT, matrix, model_for
from scripts.run_multiscale_fusion import infer, write_json
from scripts.run_multiscale_validation import sha, verify_inputs
from scripts.summarize_multiscale_validation import (
    auroc, bootstrap_rates, delta_summary, interval, metric_summary, source_counts, spread,
)


def threshold_from(values, alpha):
    # Deliberately independent of EmpiricalThreshold.fit.
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError("Insufficient calibration")
    return float(np.sort(values)[rank - 1])


def choose_epoch_independently(history, guard):
    reference = min(history, key=lambda h: (h["validation_bce"], h["epoch"]))
    candidates = [h for h in history
                  if h["validation_tpr"] >= reference["validation_tpr"] - guard["tpr_tolerance"]
                  and h["validation_positive_bce"] <=
                  reference["validation_positive_bce"] + guard["positive_bce_tolerance"]]
    feasible = [h for h in candidates if h["validation_worst_fpr"] <= guard["alpha"]]
    def rank(row):
        if feasible:
            return (-row["validation_tpr"], row["validation_worst_fpr"],
                    row["validation_bce"], row["epoch"])
        return (row["validation_worst_fpr"], -row["validation_tpr"],
                row["validation_bce"], row["epoch"])
    return reference["epoch"], min(feasible or candidates, key=rank)["epoch"]


def audit_training(out, protocol):
    count, doc_epochs, val_epochs, historical = 0, 0, 0, 0
    diagnostic = {}
    train_data = {}
    for stage, coefficient, seed in matrix(protocol):
        name = f"{stage}_{coefficient}_{seed}"
        directory = out / "runs" / name
        row = json.loads((directory / "TRAINING.json").read_text())
        train_data[name] = row
        assert row["calibration_or_test_inference_calls"] == 0
        assert row["protocol_sha256"] == sha(out / "RUN_PROTOCOL.json")
        assert sha(directory / "EPOCHS.pt") == row["checkpoint_sha256"]
        meta = json.loads((out / "logs" / f"{name}.metadata.json").read_text())
        assert meta["returncode"] == 0 and meta["log_sha256"] == sha(
            out / "logs" / f"{name}.log")
        trace = np.load(directory / "MINING_TRACE.npz", allow_pickle=False)
        vt = np.load(directory / "VALIDATION_TRACE.npz", allow_pickle=False)
        labels, base, keys = trace["labels"], trace["base_weights"], trace["keys"]
        negi = trace["negative_indices"]
        nk = keys[negi]
        docs = sorted(set(nk[:, 0]))
        family_keys = sorted({(k[0], int(k[2])) for k in nk})
        assert all(sum(k[0] == d and int(k[2]) == s for k in nk) == 7
                   for d, s in family_keys)
        assert [list(k) for k in family_keys] == row["max_format_family_keys"]
        saved_epochs = torch.load(directory / "EPOCHS.pt", map_location="cpu", weights_only=True)
        for selector in ("bce", "guarded"):
            selected_path = directory / f"MODEL_{selector}.pt"
            assert sha(selected_path) == row["selected_checkpoint_sha256"][selector]
            selected_model = torch.load(selected_path, map_location="cpu", weights_only=True)
            assert all(torch.equal(value, saved_epochs["states"][selected_model["epoch"]][key])
                       for key, value in selected_model["state_dict"].items())
        for (doc, _), weight in zip(family_keys, row["max_format_family_weights"]):
            assert abs(weight - 1 / (len(docs) * sum(d == doc for d, _ in family_keys))) < 1e-8
        reference, selected = choose_epoch_independently(row["history"], protocol["selection_guards"])
        assert reference == row["epoch_selection"]["bce_epoch"]
        assert selected == row["epoch_selection"]["guarded_epoch"]
        # Recompute every validation TPR and condition FPR from saved scores,
        # and independently reproduce the train-only temporary threshold.
        vk, vl = vt["validation_keys"], vt["validation_labels"]
        validation_counts = {}
        for label, key in zip(vl, vk):
            group = (label, key[0])
            validation_counts[group] = validation_counts.get(group, 0) + 1
        validation_weights = np.array([1 / validation_counts[(label, key[0])]
                                       for label, key in zip(vl, vk)])
        validation_weights /= validation_weights.sum()
        for h in row["history"]:
            epoch = h["epoch"]
            values = vt[f"validation_e{epoch:02}"]
            loss = np.logaddexp(0., values.astype(float)) - vl * values
            assert abs(float((loss * validation_weights).sum()) - h["validation_bce"]) < 2e-7
            for label, field in ((1, "validation_positive_bce"), (0, "validation_negative_bce")):
                active = vl == label
                expected_loss = (loss[active] * validation_weights[active]).sum() / validation_weights[active].sum()
                assert abs(float(expected_loss) - h[field]) < 2e-7
            threshold = threshold_from(vt[f"training_negative_e{epoch:02}"], .05)
            assert threshold == h["training_proxy_threshold"]["threshold"]
            flags = values > threshold
            assert abs(float(flags[vl == 1].mean()) - h["validation_tpr"]) < 1e-12
            assert abs(float(flags[vl == 0].mean()) - h["validation_fpr"]) < 1e-12
            conditions = {a: float(flags[(vl == 0) & (vk[:, 1] == a)].mean())
                          for a in sorted(set(vk[:, 1]))}
            assert conditions == h["validation_condition_fpr"]
            assert max(conditions.values()) == h["validation_worst_fpr"]
            expected_families = len(family_keys) if protocol["coefficients"][coefficient] and epoch > 5 else 0
            assert h["max_format_families_processed"] == expected_families
            val_epochs += 1
        if row["frozen_baseline_replay"]:
            assert all(row["frozen_baseline_replay"].values())
            historical += 1
        summaries = []
        for epoch in range(6, protocol["epochs"] + 1):
            if stage == "U0":
                break
            scores = trace[f"statistics_e{epoch:02}"]
            weights = trace[f"weights_e{epoch:02}"]
            selected_mask = trace[f"selected_e{epoch:02}"]
            expected_mask = np.zeros(len(base), dtype=bool)
            expected = base.astype(np.float64).copy()
            expansions = 0
            for doc in docs:
                local = np.flatnonzero(nk[:, 0] == doc)
                budget = math.ceil(len(local) * protocol["hard_negative"]["fraction"])
                if stage.endswith("1"):
                    cutoff = sorted(scores[local], reverse=True)[budget - 1]
                    chosen = local[scores[local] >= cutoff]
                    expansions += len(chosen) - budget
                else:
                    representatives = []
                    for slot in sorted(set(nk[local, 2].astype(int))):
                        rows = [i for i in local if int(nk[i, 2]) == slot]
                        representatives.append(min(rows, key=lambda i: (-float(scores[i]), nk[i, 1])))
                    representatives.sort(key=lambda i: (-float(scores[i]), int(nk[i, 2])))
                    chosen = np.array(representatives[:budget])
                    assert len({int(nk[i, 2]) for i in chosen}) == budget
                expected_mask[negi[chosen]] = True
                rows = negi[local]
                expected[negi[chosen]] *= protocol["hard_negative"]["multiplier"]
                expected[rows] *= base[rows].astype(np.float64).sum() / expected[rows].sum()
                assert abs(float(weights[rows].sum() - base[rows].sum())) < 1e-7
                doc_epochs += 1
            assert np.array_equal(selected_mask, expected_mask)
            assert np.max(np.abs(weights.astype(float) - expected)) < 2e-9
            assert np.array_equal(weights[labels == 1], base[labels == 1])
            assert (weights > 0).all()
            chosen_keys = keys[selected_mask]
            unique = len({(k[0], k[2]) for k in chosen_keys})
            split_mask = (labels == 0) & (keys[:, 1] == "split")
            summaries.append({
                "epoch": epoch, "selected_rows": len(chosen_keys),
                "distinct_families": unique,
                "duplicate_extra_format_fraction": 1 - unique / len(chosen_keys),
                "row_tie_budget_expansion": expansions,
                "split_selected_fraction": float(np.mean(chosen_keys[:, 1] == "split")),
                "split_negative_training_mass_fraction": float(
                    weights[split_mask].sum() / weights[labels == 0].sum()),
                "minimum_nonselected_negative_weight": float(
                    weights[(labels == 0) & ~selected_mask].min()),
            })
        tail = summaries[-5:]
        diagnostic[name] = {
            "stage": stage, "coefficient": coefficient, "seed": seed,
            "epochs": summaries,
            "last5": {field: float(np.mean([h[field] for h in tail])) for field in (
                "selected_rows", "distinct_families", "duplicate_extra_format_fraction",
                "row_tie_budget_expansion", "split_selected_fraction",
                "split_negative_training_mass_fraction")} if tail else None,
        }
        count += 1
    assert len(train_data) == 80
    # Independently reproduce cross-seed coefficient selection from training histories.
    selection = json.loads((out / "SELECTION.json").read_text())
    for stage, reference in protocol["coefficient_references"].items():
        candidates = {}
        for coefficient in protocol["coefficients"]:
            src = reference if coefficient == "L000" else stage
            rows = [train_data[f"{src}_{coefficient}_{seed}"]["epoch_selection"]["guarded"]
                    for seed in protocol["seeds"]]
            candidates[coefficient] = {field: float(np.mean([r[field] for r in rows]))
                                       for field in ("validation_tpr", "validation_bce",
                                                     "validation_positive_bce", "validation_worst_fpr")}
            candidates[coefficient]["max_worst_fpr"] = max(r["validation_worst_fpr"] for r in rows)
        zero, guard = candidates["L000"], protocol["selection_guards"]
        eligible = [c for c, r in candidates.items()
                    if r["validation_tpr"] >= zero["validation_tpr"] - guard["tpr_tolerance"]
                    and r["validation_positive_bce"] <=
                    zero["validation_positive_bce"] + guard["positive_bce_tolerance"]]
        feasible = [c for c in eligible if candidates[c]["max_worst_fpr"] <= guard["alpha"]]
        def rank(c):
            r = candidates[c]
            return ((-r["validation_tpr"], r["validation_worst_fpr"], r["validation_bce"], c)
                    if feasible else
                    (r["validation_worst_fpr"], -r["validation_tpr"], r["validation_bce"], c))
        assert min(feasible or eligible, key=rank) == selection["coefficient_choices"][stage]["selected"]
    return diagnostic, train_data, {
        "training_runs_verified": count, "negative_document_epoch_groups_verified": doc_epochs,
        "validation_epoch_metrics_recomputed": val_epochs,
        "old_baseline_exact_replay_runs": historical,
        "family_representative_budget_mass_and_retained_rows_verified": True,
        "epoch_and_coefficient_selection_independently_reproduced": True,
        "coefficient_selection_used_only_train_validation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.input
    for name in ("SUMMARY.json", "ANALYSIS_CHECKS.json", "MINING_DIAGNOSTICS.json"):
        if (out / name).exists():
            raise FileExistsError(name)
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    verify_inputs(protocol)
    selection = json.loads((out / "SELECTION.json").read_text())
    diagnostic, training, checks = audit_training(out, protocol)
    write_json(out / "MINING_DIAGNOSTICS.json", diagnostic)
    positive, negative = load_data()
    split = json.loads((ROOT / protocol["split_path"]).read_text())
    pi, ni, ci = (positive.indices(split["test"]), negative.indices(split["test"]),
                  negative.indices(split["calibration"]))
    keys = [positive.keys[i] for i in pi]
    assert keys == [negative.keys[i] for i in ni]
    docs = split["test"]
    masks = {"all": np.ones(len(keys), dtype=bool)}
    masks.update({a: np.array([k[1] == a for k in keys]) for a in sorted({k[1] for k in keys})})
    draws = np.random.default_rng(protocol["paired_bootstrap_seed"]).integers(
        0, len(docs), (protocol["paired_bootstrap_repetitions"], len(docs)))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reference_model = model_for("U0")
    reference_model.fit_normalizer(negative.scores[negative.indices(split["train"])])
    flags, runs, replay_count, threshold_count, rate_count = {}, {}, 0, 0, 0
    for stage, choice in selection["evaluation_methods"].items():
        for selector in ("bce", "guarded"):
            method = f"{stage}_{selector}"
            flags[method] = {a: {"positive": [], "negative": []} for a in ("0.01", "0.05")}
            for seed in protocol["seeds"]:
                endpoint = f"{method}_{seed}"
                directory = out / "evaluation" / endpoint
                result = json.loads((directory / "RESULT.json").read_text())
                prediction = np.load(directory / "PREDICTIONS.npz", allow_pickle=False)
                assert sha(directory / "PREDICTIONS.npz") == result["predictions_sha256"]
                source = out / "runs" / result["source_run"]
                checkpoint = source / f"MODEL_{selector}.pt"
                assert sha(checkpoint) == result["checkpoint_sha256"]
                assert result["selection_sha256"] == sha(out / "SELECTION.json")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert saved["split"] == split and saved["seed"] == seed
                model = model_for(choice["source_stage"])
                model.load_state_dict(saved["state_dict"], strict=True)
                for attr in ("center", "spread", "normalizer_count", "normalizer_ready"):
                    assert torch.equal(getattr(model, attr), getattr(reference_model, attr))
                for side, tensor in (("positive", positive.scores[pi]),
                                     ("negative", negative.scores[ni]),
                                     ("calibration_negative", negative.scores[ci])):
                    replay = infer(model, tensor, protocol["batch_size"]).numpy()
                    assert np.array_equal(replay, prediction[side])
                p, n, c = (prediction[s] for s in ("positive", "negative", "calibration_negative"))
                # Independent pair-count AUC verifies the rank-based implementation.
                direct_auc = float(((p[:, None] > n[None, :]).sum()
                                    + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))
                assert abs(direct_auc - auroc(p, n)) < 1e-12
                training_row = training[source.name]
                chosen_epoch = training_row["history"][saved["epoch"] - 1]
                runs[endpoint] = {
                    "stage": stage, "selector": selector, "seed": seed,
                    "source_run": source.name, "epoch": saved["epoch"],
                    "auroc": direct_auc, "parameters": result["parameters"],
                    "validation": chosen_epoch,
                    "checkpoint_sha256": sha(checkpoint), "working_points": {},
                }
                for alpha in ("0.01", "0.05"):
                    threshold = threshold_from(c, float(alpha))
                    original = result["evaluation"]["working_points"][alpha]
                    assert threshold == original["calibration"]["threshold"]
                    pp, nn, cc = (v > threshold for v in (p, n, c))
                    assert abs(float(pp.mean()) - original["tpr"]) < 1e-12
                    assert abs(float(nn.mean()) - original["fpr"]) < 1e-12
                    assert abs(float(cc.mean()) - original["calibration_fpr"]) < 1e-12
                    for attack, mask in masks.items():
                        if attack != "all":
                            assert float(pp[mask].mean()) == original["per_attack"][attack]["tpr"]
                            assert float(nn[mask].mean()) == original["per_attack"][attack]["fpr"]
                            rate_count += 2
                    worst = max(float(nn[m].mean()) for a, m in masks.items() if a != "all")
                    assert worst == original["worst_condition_fpr"]
                    flags[method][alpha]["positive"].append(pp.astype(float))
                    flags[method][alpha]["negative"].append(nn.astype(float))
                    runs[endpoint]["working_points"][alpha] = {
                        "tpr": float(pp.mean()), "fpr": float(nn.mean()),
                        "split_fpr": float(nn[masks["split"]].mean()),
                        "worst_fpr": worst, "threshold": threshold,
                    }
                    threshold_count += 1
                    rate_count += 3
                replay_count += 1
    methods = {}
    worst_boot = {}
    for method, points in flags.items():
        methods[method] = {"working_points": {}, "auroc": spread([
            runs[f"{method}_{seed}"]["auroc"] for seed in protocol["seeds"]])}
        worst_boot[method] = {}
        for alpha, sides in points.items():
            sides.update({s: np.stack(v) for s, v in sides.items()})
            table = {a: {metric: metric_summary(sides[side], keys, docs, mask, draws)
                         for metric, side in (("tpr", "positive"), ("fpr", "negative"))}
                     for a, mask in masks.items()}
            per_worst = np.max(np.array([
                table[a]["fpr"]["micro"]["by_seed"] for a in masks if a != "all"]), axis=0)
            b = []
            for a, mask in masks.items():
                if a != "all":
                    success, count = source_counts(sides["negative"], keys, docs, mask)
                    b.append(bootstrap_rates(success, count, draws))
            worst_boot[method][alpha] = np.max(b, axis=0)
            table["worst_condition"] = {
                "fpr": spread(per_worst), "mean_worst_fpr_source_bootstrap_ci95":
                interval(worst_boot[method][alpha].mean(0))}
            methods[method]["working_points"][alpha] = table
    pairs = [(f"{a}_{s}", f"{b}_{s}") for a, b in protocol["comparisons"]
             for s in ("bce", "guarded")]
    pairs += [(f"{s}_guarded", f"{s}_bce") for s in protocol["stages"]]
    comparisons = {}
    for left, right in pairs:
        name = f"{left}_minus_{right}"
        comparisons[name] = {}
        for alpha in ("0.01", "0.05"):
            table = {a: {metric: delta_summary(flags[left][alpha][side],
                                              flags[right][alpha][side], keys, docs, mask, draws)
                         for metric, side in (("tpr", "positive"), ("fpr", "negative"))}
                     for a, mask in masks.items()}
            table["worst_condition"] = {
                "mean_delta_ci95": interval((worst_boot[left][alpha] -
                                            worst_boot[right][alpha]).mean(0)),
                "micro_delta": spread(np.array(methods[left]["working_points"][alpha][
                    "worst_condition"]["fpr"]["by_seed"]) - np.array(
                    methods[right]["working_points"][alpha]["worst_condition"]["fpr"]["by_seed"]))}
            comparisons[name][alpha] = table
        primary = comparisons[name]["0.05"]
        rules = {
            "split_fpr_delta_ci_upper_negative": primary["split"]["fpr"]["mean_micro_delta_ci95"][1] < 0,
            "tpr_delta_ci_lower_ge_minus_2pp": primary["all"]["tpr"]["mean_micro_delta_ci95"][0] >= -.02,
            "every_seed_split_fpr_nonincrease": primary["split"]["fpr"]["micro_delta"]["max"] <= 0,
            "every_seed_worst_condition_fpr_le_05": methods[left]["working_points"]["0.05"][
                "worst_condition"]["fpr"]["max"] <= .05,
        }
        comparisons[name]["engineering_rule"] = {
            "checks": rules, "passed": all(rules.values())}
    summary = {
        "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
        "analysis_script_sha256": sha(__file__), "evidence_scope": protocol["evidence_scope"],
        "selected_coefficients": selection["coefficient_choices"],
        "methods": methods, "comparisons": comparisons, "runs": runs,
        "data": {part: {"documents": len(d), "positive_n": len(positive.indices(d)),
                       "negative_n": len(negative.indices(d))}
                 for part, d in split.items()},
        "analysis": {"bootstrap_repetitions": len(draws), "bootstrap_seed": protocol["paired_bootstrap_seed"],
                     "ci": "unadjusted paired source-document percentile; fixed models thresholds seeds",
                     "seed_means_are_not_ensembles": True, "independent_human_test_fpr": None,
                     "document_level_tpr_fpr": None, "all_missing_in_denominator": True},
        "execution": json.loads((out / "TRAIN_EXECUTION.json").read_text()),
    }
    checks.update(endpoints_checkpoint_replayed=replay_count,
                  independent_thresholds_recomputed=threshold_count,
                  independent_pooled_and_condition_rates_recomputed=rate_count,
                  direct_pair_auroc_verified=replay_count,
                  checkpoint_predictions_bit_exact=True,
                  old_frozen_source_and_input_sha_verified=True,
                  analysis_script_sha256=sha(__file__))
    assert replay_count == 100 and threshold_count == 200
    write_json(out / "ANALYSIS_CHECKS.json", checks)
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps({"checks": checks, "coefficients": {
        s: r["selected"] for s, r in selection["coefficient_choices"].items()},
        "5percent_guarded": {s: {
            "tpr": methods[f"{s}_guarded"]["working_points"]["0.05"]["all"]["tpr"]["micro"]["mean"],
            "fpr": methods[f"{s}_guarded"]["working_points"]["0.05"]["all"]["fpr"]["micro"]["mean"],
            "split_fpr": methods[f"{s}_guarded"]["working_points"]["0.05"]["split"]["fpr"]["micro"]["mean"],
            "worst_fpr": methods[f"{s}_guarded"]["working_points"]["0.05"]["worst_condition"]["fpr"]["mean"],
        } for s in protocol["stages"]}}, indent=2))


if __name__ == "__main__":
    main()
