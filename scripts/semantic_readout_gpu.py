"""Frozen-parameter CUDA readout; seal every order cost before joining truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

import semantic_cycle2_gpu as base

REPS = ("W", "E1", "C")


def setup(out):
    frozen = base.verify(out)
    sys.path.insert(0, str(out / "private/code/src/core"))
    for name, field in [("PROTOTYPES.json", "prototypes_sha256")]:
        if base.sha(out / "private" / name) != frozen[field]:
            raise RuntimeError("Inherited prototype SHA mismatch")
    for name, field in [("LOCKED_SELECTION.json", "selection_sha256"),
                        ("PERMUTATIONS.json", "permutations_sha256"),
                        ("SYMBOL_MAP.json", "symbol_map_sha256")]:
        if base.sha(out / name) != frozen[field]:
            raise RuntimeError("Locked public artifact SHA mismatch")
    return frozen


def groups_for(out):
    mapping = base.read(out / "SYMBOL_MAP.json")
    # The bank is already frozen in original template-id order.
    ids = sorted(int(k) for k in mapping["template_to_symbol"])
    if len(ids) != 24:
        raise RuntimeError("Full 24-template dictionary required")
    return [[j for j, t in enumerate(ids) if mapping["template_to_symbol"][str(t)] == c]
            for c in "ABCD"]


def bootstrap(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (5000, len(values)))
    return {"mean": float(values.mean()),
            "CI95": np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
            "n": len(values), "unit": "prompt_pair"}


def fresh_scores(scorer, graph):
    if graph is None:
        return [None] * len(scorer.bank)
    scorer.norms.clear()
    try:
        normalized = scorer.normalize(graph, "official")
    except Exception:
        return [None] * len(scorer.bank)
    return [scorer.pair(normalized, template) for template in scorer.bank]


def retained_weights(row, rep, groups, locked):
    from semantic_cycle2 import weights, identity
    params = locked[rep]
    qs = [weights(o["raw_scores"][rep], rep, groups, params["temperature"])
          for o in row["observations"]]
    return [q if identity(q, params["threshold"]) is not None else None for q in qs]


def order_prediction(row, candidates, groups, locked):
    from semantic_sequence import exact_costs, unordered_costs
    scores = {}
    for rep in REPS:
        qs = retained_weights(row, rep, groups, locked)
        unordered = unordered_costs(qs, candidates)
        if not np.all(unordered == unordered[0]):
            raise RuntimeError("Equal-composition frequency baseline must tie")
        scores[rep] = {
            "soft": exact_costs(qs, candidates, "soft").tolist(),
            "hard": exact_costs(qs, candidates, "hard").tolist(),
            "unordered": unordered.tolist(),
            "retained": sum(q is not None for q in qs), "observations": len(qs)}
    return {"doc_id": row["doc_id"], "arm": row["arm"], "condition": row["condition"],
            "scores": scores, "truth_loaded": False}


def blind(out, frozen):
    from semantic_cycle2 import event_view
    started = time.monotonic()
    if (out / "private/ASSIGNMENTS.json").exists() or (out / "private/references").exists():
        raise RuntimeError("Truth/reference present before blind seal")
    parser, scorer, nlp = base.runtime(out, frozen)
    encoder = base.Encoder(out)
    prototypes = base.read(out / "private/PROTOTYPES.json")
    locked = base.read(out / "LOCKED_SELECTION.json")["selection"]
    candidates = np.asarray(base.read(out / "PERMUTATIONS.json"), dtype=np.int8)
    groups = groups_for(out)
    rows, sources = base.dataset(out, nlp)
    if sources or len(rows) != 72 or any(r["split"] != "readout" for r in rows):
        raise RuntimeError("Only twelve new readout pairs are allowed")
    work = [(w["text"], o["lemma"]) for r in rows for o in r["candidates"]["observations"]
            for w in o["windows"]]
    if not work:
        raise RuntimeError("No resource precheck workload")
    measured, replay, peak = [], [], base.memory()
    for text, lemma in work[:8]:
        before = time.monotonic()
        obs = {"lemma": lemma, "windows": [{"text": text}, {"text": text}]}
        base.raw_windows(obs, parser, scorer, encoder, prototypes)
        measured.append(time.monotonic() - before)
        one, two = parser.fresh(text), parser.fresh(text)
        if any(one.get(k) != two.get(k) for k in ("valid", "amr", "events", "decode_events")):
            raise RuntimeError("Fresh parser replay mismatch")
        graphs = [one["amr"] if one["valid"] else None]
        graphs.append(event_view(graphs[0], lemma) if graphs[0] else None)
        errors = []
        for graph in graphs:
            first, second = fresh_scores(scorer, graph), fresh_scores(scorer, graph)
            if first == second:
                errors.append(0.0)
            elif None not in first and None not in second:
                errors.append(float(np.max(np.abs(np.asarray(first) - second))))
            else:
                raise RuntimeError("Scorer replay validity mismatch")
        first, second = encoder.fresh(text), encoder.fresh(text)
        if first is None or second is None:
            if (first is None) != (second is None):
                raise RuntimeError("Encoder replay validity mismatch")
            errors.append(0.0)
        else:
            errors.append(float(np.max(np.abs(first - second))))
        if max(errors) > 1e-6:
            raise RuntimeError("Numerical replay gate")
        peak = max(peak, base.memory())
        replay.append({"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "errors": errors})
    reserve = 12 * 8 * 2 * 2 + 12 * 8
    seconds_per = max(float(np.mean(measured)), float(np.percentile(measured, 75)))
    projected = (len(work) + reserve) * seconds_per * frozen["safety_factor"]
    resource = {"blind_window_occurrences": len(work), "unique_windows": len({t for t, _ in work}),
                "reserved_known_and_sentence_windows": reserve, "measured_seconds": measured,
                "projected_gpu_seconds": projected, "gpu_cap_seconds": frozen["gpu_budget_seconds"],
                "Go": bool(projected < frozen["gpu_budget_seconds"]), "all_conditions_retained": True}
    base.atomic(out / "CUDA_PRECHECK.json", {"Go": True, "replay": replay,
                                            "peak_global_vram_fraction": peak, "FP32": True})
    base.atomic(out / "RESOURCE_PRECHECK.json", resource)
    if not resource["Go"]:
        raise RuntimeError("Frozen resource projection gate")
    prediction_seal, order_seal = {}, {}
    for row in rows:
        key = f"d{row['doc_id']:02d}-{row['arm']}-{row['condition']}"
        path = out / "private/predictions" / f"{key}.json"
        if path.exists():
            prediction = base.read(path)
            if prediction["freeze_sha256"] != base.sha(out / "FREEZE.json"):
                raise RuntimeError("Prediction resume identity mismatch")
        else:
            observations = []
            for obs in row["candidates"]["observations"]:
                observations.append({**obs, "raw_scores": base.raw_windows(
                    obs, parser, scorer, encoder, prototypes)})
                peak = max(peak, base.memory())
                if time.monotonic() - started > frozen["gpu_budget_seconds"]:
                    raise RuntimeError("Actual GPU budget")
            prediction = {**row, "observations": observations,
                          "freeze_sha256": base.sha(out / "FREEZE.json")}
            base.atomic(path, prediction)
        order_path = out / "private/order_costs" / f"{key}.json"
        value = order_prediction(prediction, candidates, groups, locked)
        value.update(prediction_sha256=base.sha(path), freeze_sha256=base.sha(out / "FREEZE.json"))
        base.atomic(order_path, value)
        prediction_seal[key], order_seal[key] = base.sha(path), base.sha(order_path)
        base.progress(out, "blind_and_order_costs", completed=len(prediction_seal), total=72,
                      elapsed_seconds=time.monotonic() - started)
    seal = {"sha256": prediction_seal, "versions": 72, "reference_labels_loaded": False,
            "elapsed_seconds": time.monotonic() - started, "peak_global_vram_fraction": peak}
    base.atomic(out / "ORDER_SEAL.json", {"sha256": order_seal, "versions": 72,
                                         "truth_loaded": False, "candidate_count": 2520})
    base.atomic(out / "PREDICTION_SEAL.json", seal)


def summarize(output, frozen):
    summaries, contrasts = [], []
    for condition in ("clean", "paraphrase", "split", "merge"):
        subset = [r for r in output if r["arm"] == "positive" and r["condition"] == condition]
        for rep in REPS:
            ds = [r["diagnostics"][rep] for r in subset]
            item = {"condition": condition, "representation": rep, "pairs": len(subset),
                    "planned": sum(d["planned"] for d in ds),
                    "reference_coverage": sum(d["reference_coverage"] for d in ds),
                    "observations": sum(d["observations"] for d in ds),
                    "erasures": sum(d["erasures"] for d in ds),
                    "per_class": [{"planned": sum(d["per_class"][c]["planned"] for d in ds),
                                   "correct": sum(d["per_class"][c]["correct"] for d in ds)}
                                  for c in range(4)]}
            for metric in ("identity_f1", "localization_f1", "identity_precision", "identity_recall"):
                item[metric] = bootstrap([d[metric] for d in ds], frozen["seed"])
            if condition in ("clean", "paraphrase"):
                values = [r["known_position"][rep] for r in subset]
                item["known_position_BA"] = bootstrap(
                    [sum(x["predicted"] == x["true"] for x in v) / 8 for v in values], frozen["seed"])
                item["known_position_confusion_including_erasure"] = [
                    [sum(x["true"] == c and x["predicted"] == p for v in values for x in v)
                     for p in (0, 1, 2, 3, None)] for c in range(4)]
            summaries.append(item)
        if condition == "paraphrase":
            for rep in ("E1", "C"):
                contrasts.append({"contrast": rep + "-W", "condition": condition,
                                  "metric": "identity_f1", **bootstrap(
                                      [r["diagnostics"][rep]["identity_f1"] -
                                       r["diagnostics"]["W"]["identity_f1"] for r in subset], frozen["seed"])})
    return summaries, contrasts


def evaluate(out, frozen):
    from semantic_cycle2 import weights, identity, diagnostic
    from semantic_sequence import centered_window, symbol_weights, rank_truth
    started = time.monotonic()
    seal, order_seal = base.read(out / "PREDICTION_SEAL.json"), base.read(out / "ORDER_SEAL.json")
    if len(seal["sha256"]) != 72 or set(seal["sha256"]) != set(order_seal["sha256"]):
        raise RuntimeError("Incomplete prediction seals")
    for key in seal["sha256"]:
        if (base.sha(out / "private/predictions" / f"{key}.json") != seal["sha256"][key] or
                base.sha(out / "private/order_costs" / f"{key}.json") != order_seal["sha256"][key]):
            raise RuntimeError("Prediction/order seal mismatch")
    parser, scorer, nlp = base.runtime(out, frozen)
    encoder = base.Encoder(out)
    prototypes = base.read(out / "private/PROTOTYPES.json")
    locked = base.read(out / "LOCKED_SELECTION.json")["selection"]
    assignments = base.read(out / "private/ASSIGNMENTS.json")
    groups = groups_for(out)
    candidates = np.asarray(base.read(out / "PERMUTATIONS.json"), dtype=np.int8)
    output, order_rows = [], []
    for key in sorted(seal["sha256"]):
        row = base.read(out / "private/predictions" / f"{key}.json")
        refs = base.references(out, row, nlp, assignments)
        result = {k: row[k] for k in ("doc_id", "split", "arm", "condition")}
        result["candidate_status"] = row["candidates"]["status"]
        doc = base.read(out / "private/documents" / f"d{row['doc_id']:02d}.json")
        result["surface"] = {"words": len(row["text"].split()),
                             "inert_paraphrase": row["condition"] == "paraphrase" and
                             row["text"] == doc["arms"][row["arm"]]["text"]}
        result["diagnostics"] = {}
        for rep in REPS:
            params = locked[rep]
            predictions = [identity(weights(o["raw_scores"][rep], rep, groups, params["temperature"]),
                                    params["threshold"]) for o in row["observations"]]
            result["diagnostics"][rep] = diagnostic(row["observations"], refs, predictions)
        if row["arm"] == "positive" and row["condition"] in ("clean", "paraphrase"):
            known = {rep: [] for rep in REPS}
            tokens = list(re.finditer(r"\S+", row["text"]))
            for ref in refs:
                raw = None
                if ref.get("span"):
                    obs = {"lemma": ref["lemma"], "windows": [
                        centered_window(row["text"], tokens, ref["span"][0], w) for w in (6, 12)]}
                    raw = base.raw_windows(obs, parser, scorer, encoder, prototypes)
                for rep in REPS:
                    q = weights(raw[rep], rep, groups, locked[rep]["temperature"]) if raw else None
                    known[rep].append({"true": ref["symbol"], "predicted": identity(q, locked[rep]["threshold"])})
                base.memory()
            result["known_position"] = known
        if row["arm"] == "positive" and row["condition"] == "clean":
            full = []
            for s, text in enumerate(doc["arms"]["positive"]["slots"]):
                scores = scorer.scores(parser.parse(text))
                q = symbol_weights(scores, groups, locked["W"]["temperature"])
                full.append({"true": refs[s]["symbol"], "predicted": identity(
                    q.tolist() if q is not None else None, locked["W"]["threshold"])})
            result["whole_sentence_W"] = full
        costs = base.read(out / "private/order_costs" / f"{key}.json")
        order = {k: row[k] for k in ("doc_id", "arm", "condition")}
        if row["arm"] == "positive":
            truth = [s["symbol"] for s in assignments[row["doc_id"]]["slots"]]
            order["rankings"] = {rep: {mode: rank_truth(costs["scores"][rep][mode], candidates, truth)
                                      for mode in ("soft", "hard", "unordered")} for rep in REPS}
        else:
            order["cost_ranges_without_truth"] = {
                rep: {mode: [min(costs["scores"][rep][mode]), max(costs["scores"][rep][mode])]
                      for mode in ("soft", "hard", "unordered")} for rep in REPS}
        output.append(result)
        order_rows.append(order)
        base.progress(out, "fixed_readout_evaluation", completed=len(output), total=72,
                      elapsed_seconds=time.monotonic() - started)
        if time.monotonic() - started + seal["elapsed_seconds"] > frozen["gpu_budget_seconds"]:
            raise RuntimeError("Cumulative GPU budget")
    summaries, contrasts = summarize(output, frozen)
    order_summaries, order_contrasts = [], []
    for condition in ("clean", "paraphrase", "split", "merge"):
        subset = [r for r in order_rows if r["arm"] == "positive" and r["condition"] == condition]
        for rep in REPS:
            for mode in ("soft", "hard", "unordered"):
                item = {"condition": condition, "representation": rep, "mode": mode}
                for metric in ("pairwise_win_rate", "average_rank", "top1_credit"):
                    item[metric] = bootstrap([r["rankings"][rep][mode][metric] for r in subset], frozen["seed"])
                item["gain_over_frequency"] = bootstrap(
                    [r["rankings"][rep][mode]["pairwise_win_rate"] - .5 for r in subset], frozen["seed"])
                order_summaries.append(item)
        for rep in ("E1", "C"):
            order_contrasts.append({"condition": condition, "contrast": rep + "-W", "mode": "soft",
                                    **bootstrap([r["rankings"][rep]["soft"]["pairwise_win_rate"] -
                                                 r["rankings"]["W"]["soft"]["pairwise_win_rate"]
                                                 for r in subset], frozen["seed"])})
    common = {"scope": "post-No-Go independent constructed/model-assisted diagnostic",
              "formal_G1_G2_G3_G4": "NOT_EVALUATED", "freeze_sha256": base.sha(out / "FREEZE.json"),
              "prediction_seal_sha256": base.sha(out / "PREDICTION_SEAL.json"),
              "locked_selection_sha256": base.sha(out / "LOCKED_SELECTION.json"),
              "selection_retuned": False}
    base.atomic(out / "ORDER_RESULT.json", {**common, "candidate_count": 2520,
                                           "order_seal_sha256": base.sha(out / "ORDER_SEAL.json"),
                                           "summaries": order_summaries, "paired_contrasts": order_contrasts,
                                           "rows": order_rows})
    base.atomic(out / "RESULT.json", {**common, "summaries": summaries, "paired_contrasts": contrasts,
                                      "rows": output, "elapsed_seconds": time.monotonic() - started})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("blind", "evaluate"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    try:
        if (args.out / "HALT.json").exists():
            raise RuntimeError("Permanent halt")
        frozen = setup(args.out)
        {"blind": blind, "evaluate": evaluate}[args.action](args.out, frozen)
    except BaseException as exc:
        base.atomic(args.out / "HALT.json", {"permanent": True, "reason_class": type(exc).__name__,
                                            "reason": str(exc) if isinstance(exc, RuntimeError) else "execution_failure",
                                            "phase": args.action})
        raise


if __name__ == "__main__":
    main()
