"""Post-hoc exact full-message audit of sealed real readout1 predictions.

prepare/blind never read source messages or event references. evaluate joins
them only after the entire blind cost archive is sealed. No provider/model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semantic_channel_audit1"
OLD = ROOT / "experiments/semantic_sequence_readout1"
PRIVATE = Path.home() / ".cache/amr/semantic_channel_audit1"
PRIOR = Path.home() / ".cache/amr/semantic_sequence_readout1"
sys.path[:0] = [str(ROOT / "src/core"), str(ROOT / "scripts")]
from semantic_sequence import exact_costs, rank_truth
from audit_semantic_readout import q_from_raw, classify, full_dp, scalar_dp

REPS = ("W", "E1", "C")
CONDITIONS = ("clean", "paraphrase", "split", "merge")
SEED = 20261025


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def candidates():
    indices = np.arange(4 ** 8, dtype=np.int32)
    return ((indices[:, None] // (4 ** np.arange(7, -1, -1))) % 4).astype(np.int8)


def groups():
    mapping = read(OLD / "SYMBOL_MAP.json")["template_to_symbol"]
    ids = sorted(map(int, mapping))
    return [[j for j, t in enumerate(ids) if mapping[str(t)] == c] for c in "ABCD"]


def qs_for(row, rep):
    params = read(OLD / "LOCKED_SELECTION.json")["selection"][rep]
    qs = [q_from_raw(o["raw_scores"][rep], rep, groups(), params["temperature"])
          for o in row["observations"]]
    return [q if classify(q, params["threshold"]) is not None else None for q in qs]


def freeze_check():
    frozen = read(OUT / "FREEZE.json")
    for rel, expected in frozen["sources_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError("Frozen source changed")
    if sha(OUT / "PROTOCOL.md") != frozen["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for rel, expected in frozen["public_inputs_sha256"].items():
        if sha(OLD / rel) != expected:
            raise RuntimeError("Original public artifact changed")
    for name, expected in frozen["prediction_sha256"].items():
        if sha(PRIVATE / "predictions" / (name + ".json")) != expected:
            raise RuntimeError("Original prediction SHA mismatch")
    return frozen


def prepare():
    if (OUT / "FREEZE.json").exists():
        raise RuntimeError("Use a fresh experiment for a new freeze")
    source_seal = read(OLD / "PREDICTION_SEAL.json")
    files = list((PRIVATE / "predictions").glob("*.json"))
    if len(files) != 72 or {p.stem for p in files} != set(source_seal["sha256"]):
        raise RuntimeError("All original 72 predictions are required")
    for path in files:
        if sha(path) != source_seal["sha256"][path.stem]:
            raise RuntimeError("Prediction seal mismatch")
    source_names = (
        "scripts/run_semantic_channel_audit.py", "src/core/semantic_sequence.py",
        "src/core/semantic_cycle2.py", "scripts/audit_semantic_readout.py",
        "scripts/semantic_sequence_stage1_common.py", "scripts/semantic_cycle2_gpu.py",
    )
    public = ("FREEZE.json", "PREDICTION_SEAL.json", "ORDER_SEAL.json", "RESULT.json",
              "ORDER_RESULT.json", "SYMBOL_MAP.json", "LOCKED_SELECTION.json", "PERMUTATIONS.json")
    frozen = {
        "schema": "semantic-channel-posthoc-audit/v1", "at": now(), "seed": SEED,
        "authorization": "User requested semantic writing/readback experiments in current turn",
        "protocol_sha256": sha(OUT / "PROTOCOL.md"),
        "sources_sha256": {n: sha(ROOT / n) for n in source_names},
        "public_inputs_sha256": {n: sha(OLD / n) for n in public},
        "prediction_sha256": source_seal["sha256"],
        "future_reference_inputs_sha256": {
            "ASSIGNMENTS.json": sha(PRIOR / "ASSIGNMENTS.json"),
            "BANK.json": sha(PRIOR / "BANK.json"),
            "documents": {p.name: sha(p) for p in sorted((PRIOR / "documents").glob("*.json"))},
            "references": {p.name: sha(p) for p in sorted((PRIOR / "references").glob("*.json"))},
        },
        "candidate_count": 4 ** 8, "public_message_length": 8,
        "class_composition_disclosed_to_decoder": False,
        "old_data_still_equal_composition": True, "posthoc": True,
        "primary": "W/soft/paraphrase/unique_complete_message",
        "no_new_provider_or_neural_inference": True,
        "dependencies": {n: metadata.version(n) for n in ("numpy", "penman", "spacy", "en-core-web-sm")},
    }
    atomic(OUT / "FREEZE.json", frozen)
    # Check exact recurrence on real observations using differently laid out DP.
    trial = next(read(p) for p in sorted(files) if read(p)["observations"])
    qs = qs_for(trial, "W")
    small = candidates()[:64]
    errors = {}
    for mode in ("soft", "hard"):
        error = float(np.max(np.abs(exact_costs(qs, small, mode) -
                                    full_dp(qs, small, hard=mode == "hard"))))
        if error > 1e-10:
            raise RuntimeError("Independent DP oracle failed")
        errors[mode] = error
    before = time.monotonic()
    exact_costs(qs, candidates(), "soft")
    elapsed = time.monotonic() - before
    projected = elapsed * 72 * 6 * 2.0
    precheck = {"Go": projected < 3600, "candidate_count": 65536,
                "one_real_row_seconds": elapsed, "projected_blind_seconds": projected,
                "cpu_budget_seconds": 3600, "independent_dp_max_errors": errors,
                "new_model_calls": 0, "new_provider_calls": 0, "truth_loaded": False}
    atomic(OUT / "READINESS.json", precheck)
    if not precheck["Go"]:
        raise RuntimeError("CPU resource gate")
    return precheck


def blind():
    frozen = freeze_check()
    if (OUT / "BLIND_SEAL.json").exists():
        raise RuntimeError("Blind archive already sealed")
    if not read(OUT / "READINESS.json")["Go"]:
        raise RuntimeError("Readiness gate")
    cs = candidates()
    seals, count = {}, 0
    started = time.monotonic()
    for key in sorted(frozen["prediction_sha256"]):
        row = read(PRIVATE / "predictions" / (key + ".json"))
        arrays = {}
        for rep in REPS:
            qs = qs_for(row, rep)
            for mode in ("soft", "hard"):
                arrays[rep + "_" + mode] = exact_costs(qs, cs, mode)
                count += len(cs)
        path = PRIVATE / "blind_costs" / (key + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        seals[key] = sha(path)
        atomic(OUT / "heartbeat.json", {"phase": "FULL_SPACE_BLIND", "at": now(),
               "completed": len(seals), "total": 72, "elapsed_seconds": time.monotonic() - started})
        if time.monotonic() - started > 3600:
            raise RuntimeError("CPU budget exceeded")
    result = {"at": now(), "sha256": seals, "versions": 72, "candidate_count": len(cs),
              "cost_entries": count, "truth_loaded": False, "reference_positions_loaded": False,
              "class_composition_loaded": False, "freeze_sha256": sha(OUT / "FREEZE.json"),
              "elapsed_seconds": time.monotonic() - started}
    atomic(OUT / "BLIND_SEAL.json", result)
    return {k: v for k, v in result.items() if k != "sha256"}


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(values), (5000, len(values)))
    return {"mean": float(values.mean()), "CI95": np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
            "n": len(values), "unit": "prompt_pair"}


def metrics(costs, cs, truth):
    result = rank_truth(costs, cs, truth)
    minima = np.flatnonzero(np.abs(costs - costs.min()) <= 1e-10)
    result["global_minimum_ties"] = int(len(minima))
    result["unique_complete_message"] = int(len(minima) == 1 and np.array_equal(cs[minima[0]], truth))
    result["unique_prediction"] = int(len(minima) == 1)
    # All ties count as 16 bit erasures, rather than arbitrary tie breaking.
    if len(minima) == 1:
        difference = np.bitwise_xor(cs[minima[0]], truth)
        result["bit_error_or_erasure_rate"] = float(((difference & 1) + (difference >> 1)).sum() / 16)
    else:
        result["bit_error_or_erasure_rate"] = 1.0
    return result


def check_references(frozen):
    reference = frozen["future_reference_inputs_sha256"]
    for name in ("ASSIGNMENTS.json", "BANK.json"):
        if sha(PRIOR / name) != reference[name]:
            raise RuntimeError("Reference identity changed")
    for sub in ("documents", "references"):
        for name, expected in reference[sub].items():
            if sha(PRIOR / sub / name) != expected:
                raise RuntimeError("Reference source changed")


def target_audit():
    from semantic_sequence_stage1_common import structural_features, jaccard_distance, feature_signature
    bank = read(PRIOR / "BANK.json")
    mapping = read(OLD / "SYMBOL_MAP.json")["template_to_symbol"]
    ids = sorted(map(int, mapping))
    features = [structural_features(b["normalized_template_amr"]) for b in bank]
    signatures = [feature_signature(f) for f in features]
    if len(features) != 24:
        raise RuntimeError("Full target dictionary required")
    result = []
    for i, t in enumerate(ids):
        distances = [(jaccard_distance(features[i], f), j) for j, f in enumerate(features) if j != i]
        nearest = min(d for d, j in distances)
        neighbor_symbols = {mapping[str(ids[j])] for d, j in distances if d == nearest}
        predicted = next(iter(neighbor_symbols)) if len(neighbor_symbols) == 1 else None
        result.append({"template_id": t, "correct": predicted == mapping[str(t)],
                       "erasure": predicted is None, "nearest_distance": float(nearest),
                       "nearest_distinct_symbols": len(neighbor_symbols)})
    collisions = []
    for signature in set(signatures):
        ix = [i for i, s in enumerate(signatures) if s == signature]
        if len({mapping[str(ids[i])] for i in ix}) > 1:
            collisions.append([ids[i] for i in ix])
    return {"scope": "ideal target graphs; groups were originally built from all 24 targets",
            "not_independent_generalization": True, "templates": 24,
            "distinct_signatures": len(set(signatures)), "cross_symbol_collisions": collisions,
            "leave_one_template_correct": sum(r["correct"] for r in result),
            "leave_one_template_erased": sum(r["erasure"] for r in result), "rows": result,
            "human_semantics_confirmed": False, "parser_accuracy": "NOT_EVALUATED"}


def evaluate():
    frozen = freeze_check()
    seal = read(OUT / "BLIND_SEAL.json")
    if seal["truth_loaded"] or len(seal["sha256"]) != 72:
        raise RuntimeError("Incomplete/contaminated blind seal")
    for key, expected in seal["sha256"].items():
        if sha(PRIVATE / "blind_costs" / (key + ".npz")) != expected:
            raise RuntimeError("Blind cost seal changed")
    check_references(frozen)
    from semantic_cycle2 import match_positions
    from semantic_cycle2_gpu import references
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # references() is a pure post-seal helper; it never loads an AMR model.
    mirror = PRIVATE / "reference_mirror/private"
    mirror.mkdir(parents=True, exist_ok=True)
    for name in ("documents", "references"):
        target = mirror / name
        if not target.exists():
            target.symlink_to(PRIOR / name, target_is_directory=True)
    assignments = read(PRIOR / "ASSIGNMENTS.json")
    old_results = {(r["doc_id"], r["arm"], r["condition"]): r for r in read(OLD / "RESULT.json")["rows"]}
    cs = candidates()
    rows, oracle_rows, negative_rows = [], [], []
    max_error, audited_costs, scalar_checks, counterfactual_checks = 0.0, 0, 0, 0
    onehot = np.eye(4, dtype=float)
    started = time.monotonic()
    for key in sorted(seal["sha256"]):
        row = read(PRIVATE / "predictions" / (key + ".json"))
        with np.load(PRIVATE / "blind_costs" / (key + ".npz")) as costs:
            truth = None
            if row["arm"] == "positive":
                truth = np.asarray(["ABCD".index(s["symbol"]) for s in assignments[row["doc_id"]]["slots"]])
            refs = references(PRIVATE / "reference_mirror", row, nlp, assignments) if truth is not None else []
            pairs = match_positions(row["observations"], refs) if refs else []
            row_identity = {k: row[k] for k in ("doc_id", "arm", "condition")}
            for rep in REPS:
                qs = qs_for(row, rep)
                for mode in ("soft", "hard"):
                    stored = costs[rep + "_" + mode]
                    # Independently laid out DP checks all 65,536 scores.
                    independent = full_dp(qs, cs, hard=mode == "hard")
                    error = float(np.max(np.abs(stored - independent)))
                    if error > 1e-10:
                        raise RuntimeError("Full-space independent audit failed")
                    max_error = max(max_error, error)
                    audited_costs += len(cs)
                    if mode == "soft":
                        for ci in (0, 137, 2520, 65535):
                            if abs(scalar_dp(qs, cs[ci]) - stored[ci]) > 1e-10:
                                raise RuntimeError("Scalar alignment oracle failed")
                            scalar_checks += 1
                    if truth is not None:
                        rows.append({**row_identity, "representation": rep, "mode": mode,
                                     **metrics(stored, cs, truth)})
                    else:
                        minimum_ties = int(np.sum(np.abs(stored - stored.min()) <= 1e-10))
                        negative_rows.append({**row_identity, "representation": rep, "mode": mode,
                                              "minimum_cost": float(stored.min()), "minimum_ties": minimum_ties})
                if truth is None:
                    continue
                ordered_pairs = sorted(pairs)
                corrected = list(qs)
                for observation, reference in pairs:
                    corrected[observation] = onehot[refs[reference]["symbol"]]
                variants = {
                    "matched_positions_model_soft": ([qs[p] for p, r in ordered_pairs], "soft"),
                    "matched_symbols_corrected_distractors_retained": (corrected, "soft"),
                }
                if row["condition"] in ("clean", "paraphrase"):
                    known = old_results[(row["doc_id"], row["arm"], row["condition"])]["known_position"][rep]
                    ordered_refs = sorted((r for r in range(8) if refs[r].get("span")),
                                          key=lambda r: refs[r]["span"])
                    variants["reference_positions_original_hard_outputs"] = (
                        [onehot[known[r]["predicted"]] if known[r]["predicted"] is not None else None
                         for r in ordered_refs], "hard")
                for variant, (vqs, mode) in variants.items():
                    vs = exact_costs(vqs, cs, mode)
                    # Independent audits for the actual source and fixed extreme candidates.
                    tis = int(sum(int(t) * 4 ** (7 - j) for j, t in enumerate(truth)))
                    audit_ix = np.unique([0, 137, tis, 65535])
                    oracle = full_dp(vqs, cs[audit_ix], hard=mode == "hard")
                    if np.max(np.abs(oracle - vs[audit_ix])) > 1e-10:
                        raise RuntimeError("Counterfactual DP oracle failed")
                    counterfactual_checks += len(audit_ix)
                    oracle_rows.append({**row_identity, "representation": rep, "variant": variant,
                                        "mode": mode, "true_positions_or_labels_used": True,
                                        **metrics(vs, cs, truth)})
            if truth is not None:
                ordered = sorted((r for r in refs if r.get("span")), key=lambda r: r["span"])
                ideal = [onehot[r["symbol"]] for r in ordered]
                vs = exact_costs(ideal, cs, "hard")
                oracle_rows.append({**row_identity, "representation": "PLAN_ORACLE",
                    "variant": "reference_text_order_ideal_labels", "mode": "hard",
                    "reference_coverage": len(ordered),
                    "source_slot_order_changed": [r["slot"] for r in ordered] != sorted(r["slot"] for r in ordered),
                    "true_positions_or_labels_used": True, **metrics(vs, cs, truth)})
            atomic(OUT / "heartbeat.json", {"phase": "POST_SEAL_AUDIT_AND_REPLACEMENTS", "at": now(),
                   "completed": len({r["doc_id"] for r in rows}), "current_version": key,
                   "elapsed_seconds": time.monotonic() - started})
    summaries = []
    for condition in CONDITIONS:
        for rep in REPS:
            for mode in ("soft", "hard"):
                subset = [r for r in rows if (r["condition"], r["representation"], r["mode"]) == (condition, rep, mode)]
                summaries.append({"condition": condition, "representation": rep, "mode": mode,
                    "complete_message_correct": sum(r["unique_complete_message"] for r in subset), "planned_messages": 12,
                    **{m: interval([r[m] for r in subset]) for m in
                       ("unique_complete_message", "unique_prediction", "bit_error_or_erasure_rate", "average_rank", "top1_credit")}})
    oracle_summaries = []
    keys = sorted({(r["condition"], r["representation"], r["variant"]) for r in oracle_rows})
    for condition, rep, variant in keys:
        subset = [r for r in oracle_rows if (r["condition"], r["representation"], r["variant"]) == (condition, rep, variant)]
        oracle_summaries.append({"condition": condition, "representation": rep, "variant": variant,
             "planned_messages": 12, "complete_message_correct": sum(r["unique_complete_message"] for r in subset),
             "unique_complete_message": interval([r["unique_complete_message"] for r in subset]),
             "mean_rank": float(np.mean([r["average_rank"] for r in subset])),
             "reference_missing_events": sum(8-r["reference_coverage"] for r in subset) if rep == "PLAN_ORACLE" else None,
             "source_slot_order_changed_docs": sum(r["source_slot_order_changed"] for r in subset) if rep == "PLAN_ORACLE" else None})
    target = target_audit()
    primary = next(s for s in summaries if (s["condition"], s["representation"], s["mode"]) == ("paraphrase", "W", "soft"))
    result = {"schema": "semantic-full-space-posthoc-results/v1", "at": now(),
        "scope": "same real readout1 data; posthoc; constructed/model-assisted references",
        "candidate_count": 65536, "source_data_pairs": 12, "new_independent_test_pairs": 0,
        "new_provider_calls": 0, "new_neural_inference_calls": 0,
        "primary_correct": primary["complete_message_correct"], "primary_planned": 12,
        "summaries": summaries, "oracle_summaries": oracle_summaries, "rows": rows,
        "oracle_rows": oracle_rows, "unwatermarked_without_message_truth": negative_rows,
        "target_dictionary_audit": target,
        "human_writing_gate": "NOT_EVALUATED", "parser_semantic_loss": "NOT_EVALUATED",
        "model_comparison": "NOT_TRAINED_HUMAN_GATE_PENDING",
        "new_data_full_message_validation": "NOT_RUN",
        "old_NoGo_preserved": True, "blind_seal_sha256": sha(OUT / "BLIND_SEAL.json")}
    atomic(OUT / "RESULT.json", result)
    atomic(OUT / "AUDIT.json", {"Go": True, "all_72_source_sha_verified": True,
        "all_full_space_blind_costs_independently_checked": audited_costs,
        "scalar_alignment_checks": scalar_checks, "counterfactual_candidate_checks": counterfactual_checks,
        "max_independent_dp_error": max_error, "blind_truth_loaded": False,
        "planned_messages_per_positive_condition": 12, "source_freeze_preserved": True,
        "result_sha256": sha(OUT / "RESULT.json")})
    return {"primary_full_message_correct": result["primary_correct"], "planned": 12,
            "audited_cost_entries": audited_costs, "human_gate": "NOT_EVALUATED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "blind", "evaluate"))
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if (OUT / "HALT.json").exists():
            raise RuntimeError("Permanent audit halt")
        output = {"prepare": prepare, "blind": blind, "evaluate": evaluate}[args.action]()
        print(json.dumps(output, ensure_ascii=False))
    except Exception as exc:
        atomic(OUT / "HALT.json", {"at": now(), "permanent": True, "reason_class": type(exc).__name__,
                                 "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
