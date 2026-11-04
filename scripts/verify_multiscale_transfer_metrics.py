"""Recalculate frozen transfer metrics from saved predictions without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_auc(positive, negative):
    """Mann–Whitney rank statistic, independent of the scorer's pair loop."""
    rows = sorted([(value, 1) for value in positive] +
                  [(value, 0) for value in negative])
    rank_sum, start = 0.0, 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(label for _, label in rows[start:end])
        start = end
    n_positive, n_negative = len(positive), len(negative)
    return ((rank_sum - n_positive * (n_positive + 1) / 2) /
            (n_positive * n_negative))


def verify(out):
    assert read(out / "FINAL_AUDIT.json")["passed"]
    assert (out / "COMPLETE.json").exists()
    assert not any((out / name).exists() for name in ("HALT.json", "STOP.json"))
    freeze_sha = sha(out / "FREEZE.json")
    freeze, result = read(out / "FREEZE.json"), read(out / "RESULTS.json")
    assert result["freeze_sha256"] == freeze_sha and result["no_refit_no_test_tuning"]
    documents = read(out / "private/DOCUMENTS.json")
    signals = {row["doc_id"]: row for row in
               (read(path) for path in (out / "private/signal").glob("*.json"))}
    assert set(signals) == set(freeze["split"]["development"] + freeze["split"]["readout"])
    assert result["official_signal"] == read(out / "SIGNAL_RESULTS.json")
    assert result["clean_gate"] == read(out / "CLEAN_GATE.json")
    signal_sentences, signal_cells = 0, 0
    for record in signals.values():
        assert record["freeze_sha256"] == freeze_sha
        for arm in record["arms"].values():
            assert len(arm["sentence_scores"]) == 8
            for row in arm["sentence_scores"]:
                assert len(row) == 100
                assert all(value is None or (math.isfinite(value) and 0 <= value <= 1)
                           for value in row)
                signal_sentences += 1
                signal_cells += len(row)
    assert signal_sentences == 384 and signal_cells == 38400

    def sentence_values(docid, arm):
        return [max((value for value in row if value is not None), default=-math.inf)
                for row in signals[docid]["arms"][arm]["sentence_scores"]]

    def complete_mean(values):
        return sum(values) / len(values) if all(map(math.isfinite, values)) else -math.inf

    signal_populations_checked = 0
    for population, metric in result["official_signal"]["populations"].items():
        ids = freeze["split"][population]
        assert metric["pairs"] == len(ids)
        carrier, document = {}, {}
        for arm in ("positive", "unwatermarked"):
            per_document = [sentence_values(docid, arm) for docid in ids]
            carrier[arm] = [value for row in per_document for value in row]
            document[arm] = [complete_mean(row) for row in per_document]
        for view, values in (("carrier", carrier), ("document", document)):
            positive, negative = values["positive"], values["unwatermarked"]
            assert abs(rank_auc(positive, negative) - metric[view + "_auroc"]) < 1e-12
            observed = sum((p > n) + 0.5 * (p == n) for p, n in zip(positive, negative))
            assert abs(observed / len(positive) - metric[view + "_paired_win"]) < 1e-12
        for arm, prefix in (("positive", "positive"), ("unwatermarked", "negative")):
            values = carrier[arm]
            assert sum(not math.isfinite(value) for value in values) == metric[prefix + "_abstain"]
            mean = complete_mean(values)
            assert (metric[prefix + "_mean"] is None if mean == -math.inf else
                    abs(mean - metric[prefix + "_mean"]) < 1e-12)
        signal_populations_checked += 1
    decision = result["official_signal"]["populations"]["signal_decision"]
    assert result["clean_gate"]["passed"] == (
        decision["carrier_auroc"] >= freeze["clean_gate"]["carrier_auroc_min"] and
        decision["document_paired_win"] >= freeze["clean_gate"]["document_paired_win_min"])
    assert not result["clean_gate"]["readout_used"]
    conditions = freeze["conditions"] if result["clean_gate"]["passed"] else ["clean"]
    # Rebuild every parser input key from the frozen text, not from score records.
    import importlib.util
    geometry_path = out / "private/code/src/core/multiscale_blind.py"
    spec = importlib.util.spec_from_file_location("verified_transfer_geometry", geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    keys = set()
    for document in documents:
        for arm in document["arms"].values():
            keys.update(digest(carrier["text"]) for carrier in arm["carriers"])
            for condition in conditions:
                keys.update(digest(window["text"])
                            for windows in geometry.blind_geometry(arm["texts"][condition]).values()
                            for window in windows)
    parses = {path.stem: read(path) for path in (out / "private/parses").glob("*.json")}
    assert keys <= set(parses)
    for key, parsed in parses.items():
        assert parsed["freeze_sha256"] == freeze_sha and parsed["text_sha256"] == key
        assert parsed["amr_sha256"] == (digest(parsed["amr"]) if parsed["amr"] else None)
    if conditions == freeze["conditions"]:
        workload = read(out / "WORKLOAD.json")
        assert len(keys) == workload["unique_all_parse_inputs"]
        assert digest("\n".join(sorted(keys))) == workload["all_parse_key_manifest_sha256"]
        assert set(parses) == keys
    predictions = {}
    for row in read(out / "private/PREDICTIONS.json"):
        key = (row["doc_id"], row["condition"], row["arm"], row["view"], row["method"])
        assert key not in predictions
        predictions[key] = [-math.inf if value is None else value
                            for value in row["statistics"]]

    def scores(ids, condition, arm, view, method):
        return [value for docid in ids
                for value in predictions[(docid, condition, arm, view, method)]]

    metrics_checked, working_points_checked = 0, 0
    for condition, populations in result["conditions"].items():
        for population, group in populations.items():
            ids = freeze["split"][population]
            assert group["pairs"] == group["expected_pairs"] == len(ids)
            for name, metric in group["methods"].items():
                method, view = name.split("/")
                positive, negative = [scores(ids, condition, arm, view, method)
                                      for arm in ("positive", "unwatermarked")]
                assert len(positive) == metric["positive_n"]
                assert len(negative) == metric["negative_n"]
                assert abs(rank_auc(positive, negative) - metric["auroc"]) < 1e-12
                wins = [(p > n) + 0.5 * (p == n) for p, n in zip(positive, negative)]
                assert abs(sum(wins) / len(wins) - metric["paired_win"]) < 1e-12
                assert sum(not math.isfinite(v) for v in positive) == metric["positive_abstain"]
                assert sum(not math.isfinite(v) for v in negative) == metric["negative_abstain"]
                calibration = scores(freeze["split"]["calibration"], "clean",
                                     "unwatermarked", view, method)
                for alpha, point in metric["working_points"].items():
                    rank = math.ceil((len(calibration) + 1) * (1 - float(alpha)))
                    if rank > len(calibration) or all(v == -math.inf for v in calibration):
                        assert point["status"] == "insufficient_calibration"
                        assert point["threshold"] is None
                    else:
                        threshold = sorted(calibration)[rank - 1]
                        if threshold == -math.inf:
                            assert point["status"] == "insufficient_valid_calibration"
                            assert point["threshold"] is None
                        else:
                            assert point["threshold"] == threshold
                            for values, field in ((positive, "tpr"), (negative, "fpr")):
                                observed = sum(math.isfinite(v) and v > threshold for v in values)
                                assert abs(observed / len(values) - point[field]) < 1e-12
                    working_points_checked += 1
                metrics_checked += 1
    audit = {
        "passed": True, "freeze_sha256": freeze_sha,
        "verifier_source_sha256": sha(Path(__file__)),
        "results_sha256": sha(out / "RESULTS.json"),
        "predictions_sha256": sha(out / "private/PREDICTIONS.json"),
        "metrics_checked": metrics_checked, "working_points_checked": working_points_checked,
        "signal_populations_checked": signal_populations_checked,
        "original_signal_sentences_checked": signal_sentences,
        "original_signal_score_cells_checked": signal_cells,
        "all_expected_parse_keys_present": True, "expected_unique_parse_inputs": len(keys),
        "cached_parse_sha_checked": len(parses), "rank_based_auroc_verified": True,
        "thresholds_recomputed_from_clean_calibration_only": True,
        "no_model_training_or_selection": True,
    }
    path = out / "METRIC_AUDIT.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    verify(parser.parse_args().out.resolve())
