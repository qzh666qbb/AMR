"""CPU-only independent arithmetic and sealed-source audit; no model scoring."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q_from_raw(raw, rep, groups, temperature):
    rows = []
    for vector in raw:
        if vector is None or any(v is None for v in vector):
            return None
        v = np.asarray(vector, dtype=float)
        if not np.isfinite(v).all():
            return None
        v = np.exp((v - v.max()) / temperature)
        q = v / v.sum() if rep == "C" else np.asarray([v[g].sum() for g in groups]) / v.sum()
        rows.append(q)
    if len(rows) != 2:
        raise AssertionError("two common windows required")
    return np.mean(rows, axis=0)


def classify(q, threshold):
    if q is None or max(q) < threshold:
        return None
    winners = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
    return int(winners[0]) if len(winners) == 1 else None


def full_dp(qs, candidates, hard=False):
    """Different DP layout: [candidate position, candidate] for each observation."""
    n, length = candidates.shape
    prior = np.arange(length + 1, dtype=float)[:, None].repeat(n, axis=1)
    for i, q in enumerate(qs, 1):
        if q is None:
            cost = np.ones(4)
        elif hard:
            cost = np.ones(4)
            winners = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
            if len(winners) == 1:
                cost[winners[0]] = 0
        else:
            cost = -np.log(np.maximum(q, 1e-6)) / math.log(4)
        current = np.zeros_like(prior)
        current[0] = i
        for j in range(1, length + 1):
            current[j] = np.minimum(np.minimum(prior[j] + 1, current[j - 1] + 1),
                                    prior[j - 1] + cost[candidates[:, j - 1]])
        prior = current
    return prior[-1]


def scalar_dp(qs, candidate):
    previous = list(range(len(candidate) + 1))
    for i, q in enumerate(qs, 1):
        current = [float(i)]
        for j, c in enumerate(candidate, 1):
            cost = 1 if q is None else -math.log(max(float(q[c]), 1e-6)) / math.log(4)
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def independent_diagnostic(observations, references, predictions):
    """Backward matching oracle with the production cardinality/distance/tie rule."""
    @lru_cache(None)
    def solve(position, used):
        if position == len(observations):
            return 0, 0, ()
        choices = [solve(position + 1, used)]
        a, b = observations[position]["anchor_span"]
        for index, reference in enumerate(references):
            span = reference.get("span")
            if used & (1 << index) or not span:
                continue
            c, d = span
            if max(a, c) >= min(b, d):
                continue
            count, distance, pairs = solve(position + 1, used | (1 << index))
            choices.append((count + 1, distance + abs(a + b - c - d),
                            ((position, index),) + pairs))
        return min(choices, key=lambda value: (-value[0], value[1], value[2]))

    located, _, pairs = solve(0, 0)
    emitted = sum(p is not None for p in predictions)
    correct_pairs = [(p, r) for p, r in pairs
                     if predictions[p] is not None and predictions[p] == references[r]["symbol"]]
    correct, planned, count = len(correct_pairs), len(references), len(observations)
    return {
        "planned": planned, "reference_coverage": sum(bool(r.get("span")) for r in references),
        "observations": count, "located": located, "emitted": emitted,
        "identity_correct": correct, "erasures": count - emitted,
        "localization_precision": located / count if count else 0.0,
        "localization_recall": located / planned if planned else 0.0,
        "localization_f1": 2 * located / (count + planned) if count + planned else 0.0,
        "identity_precision": correct / emitted if emitted else 0.0,
        "identity_recall": correct / planned if planned else 0.0,
        "identity_f1": 2 * correct / (emitted + planned) if emitted + planned else 0.0,
        "per_class": [{"planned": sum(r["symbol"] == c for r in references),
                       "correct": sum(references[r]["symbol"] == c for p, r in correct_pairs)}
                      for c in range(4)],
    }


def independent_rank(costs, truth_index):
    v = costs[truth_index]
    equal = np.abs(costs - v) <= 1e-10
    better = int(((costs < v) & ~equal).sum())
    worse = int(((costs > v) & ~equal).sum())
    ties = int(equal.sum())
    return {"average_rank": better + (ties + 1) / 2,
            "pairwise_win_rate": (worse + (ties - 1) / 2) / (len(costs) - 1),
            "top1_credit": 1 / ties if better == 0 else 0.0,
            "tie_count_including_truth": ties}


def interval(values, seed):
    values = np.asarray(values)
    draws = np.random.default_rng(seed).integers(0, len(values), (5000, len(values)))
    return float(values.mean()), np.quantile(values[draws].mean(1), [.025, .975])


def verify_stat(stat, values, seed):
    mean, ci = interval(values, seed)
    assert np.isclose(stat["mean"], mean, atol=1e-12, rtol=0)
    assert np.allclose(stat["CI95"], ci, atol=1e-12, rtol=0)
    assert stat["n"] == len(values)


def audit(out):
    sys.path[:0] = [str(out / "private/code/src/core"), str(out / "private/code/scripts")]
    import spacy
    import semantic_cycle2_gpu as base
    from semantic_sequence import equal_composition_sequences
    frozen = base.verify(out)
    for name, field in (("LOCKED_SELECTION.json", "selection_sha256"),
                        ("PERMUTATIONS.json", "permutations_sha256"),
                        ("SYMBOL_MAP.json", "symbol_map_sha256")):
        assert sha(out / name) == frozen[field]
    assert sha(out / "private/PROTOTYPES.json") == frozen["prototypes_sha256"]
    selection = read(out / "LOCKED_SELECTION.json")["selection"]
    nlp = spacy.load(out / "private/spacy_model")
    assignments = read(out / "private/ASSIGNMENTS.json")
    mapping = read(out / "SYMBOL_MAP.json")
    ids = sorted(int(k) for k in mapping["template_to_symbol"])
    groups = [[j for j, t in enumerate(ids) if mapping["template_to_symbol"][str(t)] == c] for c in "ABCD"]
    candidates = np.asarray(read(out / "PERMUTATIONS.json"), dtype=np.int8)
    assert np.array_equal(candidates, equal_composition_sequences())
    assert candidates.shape == (2520, 8)
    assert all(list(np.bincount(c, minlength=4)) == [2] * 4 for c in candidates)
    seal, order_seal = read(out / "PREDICTION_SEAL.json"), read(out / "ORDER_SEAL.json")
    result, order_result = read(out / "RESULT.json"), read(out / "ORDER_RESULT.json")
    assert result["selection_retuned"] is False and order_result["selection_retuned"] is False
    assert seal["reference_labels_loaded"] is False and order_seal["truth_loaded"] is False
    assert len(seal["sha256"]) == len(order_seal["sha256"]) == 72
    indexed = {(r["doc_id"], r["arm"], r["condition"]): r for r in result["rows"]}
    order_indexed = {(r["doc_id"], r["arm"], r["condition"]): r for r in order_result["rows"]}
    identities = {(i, arm, condition) for i in frozen["readout_doc_ids"]
                  for arm, conditions in (("positive", ("clean", "paraphrase", "split", "merge")),
                                          ("unwatermarked", ("clean", "paraphrase")))
                  for condition in conditions}
    assert len(result["rows"]) == len(order_result["rows"]) == 72
    assert set(indexed) == set(order_indexed) == identities
    for value in (result, order_result):
        assert value["freeze_sha256"] == sha(out / "FREEZE.json")
        assert value["prediction_seal_sha256"] == sha(out / "PREDICTION_SEAL.json")
        assert value["locked_selection_sha256"] == frozen["selection_sha256"]
    checks, max_cost_error, known_count, dp_entries = 0, 0.0, 0, 0
    for key, expected in seal["sha256"].items():
        path = out / "private/predictions" / f"{key}.json"
        cost_path = out / "private/order_costs" / f"{key}.json"
        assert sha(path) == expected and sha(cost_path) == order_seal["sha256"][key]
        row, saved = read(path), read(cost_path)
        assert saved["truth_loaded"] is False and saved["prediction_sha256"] == expected
        assert row["freeze_sha256"] == sha(out / "FREEZE.json")
        identity = (row["doc_id"], row["arm"], row["condition"])
        aggregate, order_aggregate = indexed[identity], order_indexed[identity]
        refs = base.references(out, row, nlp, assignments)
        for rep in ("W", "E1", "C"):
            params = selection[rep]
            raw_qs = [q_from_raw(o["raw_scores"][rep], rep, groups, params["temperature"])
                      for o in row["observations"]]
            predictions = [classify(q, params["threshold"]) for q in raw_qs]
            diag = independent_diagnostic(row["observations"], refs, predictions)
            assert diag == aggregate["diagnostics"][rep]
            qs = [q if p is not None else None for q, p in zip(raw_qs, predictions)]
            for mode in ("soft", "hard"):
                recomputed = full_dp(qs, candidates, mode == "hard")
                stored = np.asarray(saved["scores"][rep][mode])
                error = float(np.max(np.abs(recomputed - stored)))
                max_cost_error = max(max_cost_error, error)
                assert error <= 1e-10
                dp_entries += len(stored)
                if row["arm"] == "positive":
                    truth = ["ABCD".index(s["symbol"]) for s in assignments[row["doc_id"]]["slots"]]
                    ti = np.flatnonzero(np.all(candidates == truth, axis=1))
                    assert len(ti) == 1
                    ranks = independent_rank(stored, ti[0])
                    for field, v in ranks.items():
                        assert np.isclose(order_aggregate["rankings"][rep][mode][field], v, atol=1e-12)
            for ci in (0, 71, 2519):
                assert np.isclose(scalar_dp(qs, candidates[ci]), saved["scores"][rep]["soft"][ci], atol=1e-10)
            unordered = np.asarray(saved["scores"][rep]["unordered"])
            counts = np.sum([q for q in qs if q is not None], axis=0) if any(q is not None for q in qs) else np.zeros(4)
            assert np.all(unordered == unordered[0])
            assert np.isclose(unordered[0], np.abs(counts - 2).sum(), atol=1e-12)
            for item in aggregate.get("known_position", {}).get(rep, []):
                assert item["true"] in range(4) and item["predicted"] in (0, 1, 2, 3, None)
                known_count += 1
            checks += 1
    seed = frozen["seed"]
    for item in result["summaries"]:
        condition, rep = item["condition"], item["representation"]
        subset = [r for r in result["rows"] if r["arm"] == "positive" and r["condition"] == condition]
        assert len(subset) == 12 and item["planned"] == 96
        ds = [r["diagnostics"][rep] for r in subset]
        for metric in ("identity_f1", "localization_f1", "identity_precision", "identity_recall"):
            verify_stat(item[metric], [d[metric] for d in ds], seed)
        assert item["reference_coverage"] == sum(d["reference_coverage"] for d in ds)
        assert item["erasures"] == sum(d["erasures"] for d in ds)
        if "known_position_BA" in item:
            values = [r["known_position"][rep] for r in subset]
            verify_stat(item["known_position_BA"],
                        [sum(x["true"] == x["predicted"] for x in v) / 8 for v in values], seed)
            confusion = [[sum(x["true"] == c and x["predicted"] == p for v in values for x in v)
                          for p in (0, 1, 2, 3, None)] for c in range(4)]
            assert confusion == item["known_position_confusion_including_erasure"]
            assert sum(map(sum, confusion)) == 96
    for item in order_result["summaries"]:
        condition, rep, mode = item["condition"], item["representation"], item["mode"]
        subset = [r for r in order_result["rows"] if r["arm"] == "positive" and r["condition"] == condition]
        for metric in ("pairwise_win_rate", "average_rank", "top1_credit"):
            verify_stat(item[metric], [r["rankings"][rep][mode][metric] for r in subset], seed)
        verify_stat(item["gain_over_frequency"],
                    [r["rankings"][rep][mode]["pairwise_win_rate"] - .5 for r in subset], seed)
    for data, metric in ((result, "identity_f1"), (order_result, "pairwise_win_rate")):
        for item in data["paired_contrasts"]:
            rep = item["contrast"].split("-")[0]
            subset = [r for r in data["rows"] if r["arm"] == "positive" and r["condition"] == item["condition"]]
            values = ([r["diagnostics"][rep][metric] - r["diagnostics"]["W"][metric] for r in subset]
                      if metric == "identity_f1" else
                      [r["rankings"][rep]["soft"][metric] - r["rankings"]["W"]["soft"][metric] for r in subset])
            verify_stat(item, values, seed)
    # Reuse saved whole-sentence parser records for a necessary-contract diagnostic.
    # This neither calls a parser nor upgrades the failed original writing gate.
    import penman
    from penman.models.amr import model as AMR_MODEL
    marker_rows = []
    for doc_id in frozen["readout_doc_ids"]:
        doc = read(out / "private/documents" / f"d{doc_id:02d}.json")
        for slot in assignments[doc_id]["slots"]:
            text = doc["arms"]["positive"]["slots"][slot["slot"]]
            path = out / "private/parses" / (hashlib.sha256(text.encode()).hexdigest() + ".json")
            record = read(path)
            assert record["freeze_sha256"] == sha(out / "FREEZE.json")
            observed = {"unknown_nodes": 0, "name_sizes": [], "imperative": False, "polite": False}
            if record["valid"]:
                graph = penman.decode(record["amr"], model=AMR_MODEL)
                names = {t for s, r, t in graph.triples if r == ":name"}
                observed = {
                    "unknown_nodes": sum(r == ":instance" and t == "amr-unknown" for s, r, t in graph.triples),
                    "name_sizes": sorted(sum(s == n and bool(re.fullmatch(r":op\d+", r))
                                             for s, r, t in graph.triples) for n in names),
                    "imperative": any(r == ":mode" and t == "imperative" for s, r, t in graph.triples),
                    "polite": any(r == ":polite" and t == "+" for s, r, t in graph.triples)}
            required = slot["markers"]
            passed = bool(record["valid"] and text and observed["unknown_nodes"] >= required["unknown_nodes"]
                          and all(observed["name_sizes"].count(x) >= required["name_sizes"].count(x)
                                  for x in required["name_sizes"])
                          and (not required["imperative"] or observed["imperative"])
                          and (not required["polite"] or observed["polite"]))
            marker_rows.append({"doc_id": doc_id, "slot": slot["slot"], "strict_valid": record["valid"],
                                "necessary_contract_pass": passed, "required": required, "observed": observed})
    marker_result = {"scope": "automatic necessary-marker diagnostic, not human semantic correctness",
                     "successes": sum(r["necessary_contract_pass"] for r in marker_rows),
                     "denominator": 96, "old_cycle2_NoGo_unchanged": True, "rows": marker_rows}
    (out / "WRITE_CONTRACT_DIAGNOSTIC.json").write_text(json.dumps(marker_result, sort_keys=True, indent=2) + "\n")
    value = {"Go": True, "neural_inference_called": False, "params_retuned": False,
             "independent_backward_position_matching": True, "frozen_sources_and_assets_verified": True,
             "audit_script_sha256": sha(Path(__file__)),
             "sealed_versions": 72, "representation_rows_recomputed": checks,
             "exact_order_cost_entries_recomputed": dp_entries, "scalar_DP_real_candidate_checks": checks * 3,
             "maximum_DP_error": max_cost_error, "known_predictions_arithmetic_checked": known_count,
             "known_neural_outputs_recomputed": False, "full_denominators_preserved": True,
             "result_sha256": sha(out / "RESULT.json"), "order_result_sha256": sha(out / "ORDER_RESULT.json")}
    (out / "RESULT_AUDIT.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    audit(p.parse_args().out)
