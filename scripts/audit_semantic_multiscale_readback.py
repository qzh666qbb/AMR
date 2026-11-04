"""Independent post-seal group/normalizer/replay/statistic audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/core"), str(ROOT / "scripts")]
import run_semantic_multiscale_readback as run
from semantic_multiscale import MultiScaleSymbolReadout
from audit_semantic_readout import independent_diagnostic


def assert_statistic(statistic, values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(run.SEED).integers(0, len(values), (10000, len(values)))
    assert abs(statistic["mean"] - float(np.mean(values))) < 1e-12
    assert np.allclose(statistic["CI95"], np.percentile(np.mean(values[draws], axis=1), [2.5, 97.5]),
                       atol=1e-12, rtol=0)
    assert statistic["n"] == len(values) == 12


def audit():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = run.verify()
    result = run.read(run.OUT / "RESULT.json")
    seal = run.read(run.OUT / "MODEL_SEAL.json")
    prediction_seal = run.read(run.OUT / "PREDICTION_SEAL.json")
    assert run.sha(run.OUT / "TRAINING_RESULTS.json") == seal["training_results_sha256"]
    assert run.sha(run.OUT / "PREDICTION_SEAL.json") == result["prediction_seal_sha256"]
    source = frozen["original_reference_sha256"]
    assert run.sha(run.PRIOR / "ASSIGNMENTS.json") == source["ASSIGNMENTS.json"]
    for sub in ("documents", "references"):
        for name, expected in source[sub].items():
            assert run.sha(run.PRIOR / sub / name) == expected
    selections = run.read(run.OUT / "TRAINING_RESULTS.json")["selections"]
    assert len(selections) == 84
    replay_count, normalizer_error = 0, 0.0
    for selection in selections:
        train, val, test = set(selection["train"]), set(selection["calibration"]), selection["test"]
        assert len(train) == 8 and len(val) == 3
        assert not train & val and test not in train | val
        assert train | val | {test} == set(range(12, 24))
        for name, expected in (
                ("normalizer_training_keys", train), ("supervision_training_keys", train),
                ("selection_calibration_keys", val)):
            keys = selection[name]
            assert {frozen["manifest"][key]["doc_id"] for key in keys} == expected
            assert all(frozen["manifest"][key]["arm"] == "positive"
                       and frozen["manifest"][key]["condition"] in ("clean", "paraphrase")
                       for key in keys)
        checkpoint = run.PRIVATE / "models" / f"d{test:02d}-{selection['mode']}.pt"
        assert run.sha(checkpoint) == selection["model_sha256"]
        assert selection["model_sha256"] == seal["sha256"][checkpoint.stem]
        model = MultiScaleSymbolReadout(frozen["groups"], selection["mode"])
        model.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
        model.eval()
        x = np.concatenate([run.feature(k)["x"] for k in selection["normalizer_training_keys"]])
        count = np.isfinite(x).sum(0)
        mean = np.nansum(x.astype(np.float64), axis=0) / count
        variance = np.nansum((x.astype(np.float64) - mean) ** 2, axis=0) / count
        spread = np.maximum(np.sqrt(variance), .05)
        center_error = float(np.max(np.abs(mean - model.fusion.center.numpy())))
        spread_error = float(np.max(np.abs(spread - model.fusion.spread.numpy())))
        normalizer_error = max(normalizer_error, center_error, spread_error)
        assert center_error < 1e-6 and spread_error < 1e-6
        assert np.array_equal(count, model.fusion.normalizer_count.numpy())
        for key, meta in frozen["manifest"].items():
            if meta["doc_id"] != test:
                continue
            prediction = run.read(run.PRIVATE / "predictions" / (key + ".json"))
            q = run.model_q(model, run.feature(key)["x"])
            assert q == prediction["records"][selection["mode"]]["q"]
            replay_count += 1
    metric_checks = 0
    for row in result["rows"]:
        key = f"d{row['doc_id']:02d}-positive-{row['condition']}"
        meta = frozen["manifest"][key]
        refs = run.labels(key, frozen)["references"]
        prediction_path = run.PRIVATE / "predictions" / (key + ".json")
        assert run.sha(prediction_path) == prediction_seal["sha256"][key]
        record = run.read(prediction_path)["records"][row["method"]]
        independent = independent_diagnostic(meta["observations"], refs, record["predictions"])
        counts = [sum(p == c for p in record["predictions"]) for c in range(4)]
        f1 = []
        for c in range(4):
            planned = independent["per_class"][c]["planned"]
            correct = independent["per_class"][c]["correct"]
            f1.append(2 * correct / (planned + counts[c]) if planned + counts[c] else 0.0)
        assert abs(row["macro_f1"] - np.mean(f1)) < 1e-12
        assert abs(row["identity_f1"] - independent["identity_f1"]) < 1e-12
        assert row["identity_correct"] == independent["identity_correct"]
        assert row["complete_correct"] + row["wrong_unique_message"] + row["message_erasure"] == 1
        metric_checks += 1
    interval_checks = 0
    statistics = (
        "macro_f1", "identity_f1", "blind_carrier_accuracy_full_denominator",
        "located_no_reject_correct_full_denominator", "complete_correct",
        "average_rank", "pairwise_win_rate", "symbol_edit_error_or_erasure_rate")
    for summary in result["summaries"]:
        subset = sorted([r for r in result["rows"]
                         if r["method"] == summary["method"] and r["condition"] == summary["condition"]],
                        key=lambda r: r["doc_id"])
        assert len(subset) == 12
        # In the frozen runner, the complete_correct summary field is a rate
        # interval. The per-row integer is intact; counts are derived from rows.
        assert abs(summary["complete_correct"]["mean"] * 12
                   - sum(r["complete_correct"] for r in subset)) < 1e-12
        for name in ("wrong_unique_messages", "message_erasures", "correct_carriers"):
            metric = {"wrong_unique_messages": "wrong_unique_message", "message_erasures": "message_erasure",
                      "correct_carriers": "identity_correct"}.get(name, name)
            assert summary[name] == sum(r[metric] for r in subset)
        for metric in statistics:
            assert_statistic(summary[metric], [r[metric] for r in subset])
            interval_checks += 1
    for contrast in result["paired_contrasts"]:
        baseline = contrast["contrast"].removeprefix("attention-")
        a = sorted([r for r in result["rows"] if r["method"] == "attention"
                    and r["condition"] == contrast["condition"]], key=lambda r: r["doc_id"])
        b = sorted([r for r in result["rows"] if r["method"] == baseline
                    and r["condition"] == contrast["condition"]], key=lambda r: r["doc_id"])
        assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]
        assert_statistic(contrast, [x[contrast["metric"]] - y[contrast["metric"]] for x, y in zip(a, b)])
        interval_checks += 1
    outcome = {
        "Go": True, "model_group_checks": 84, "independent_normalizer_checks": 84,
        "max_normalizer_error": normalizer_error, "held_out_model_version_replays": replay_count,
        "independent_macro_and_identity_metric_checks": metric_checks,
        "independent_interval_checks": interval_checks, "original_references_SHA_unchanged": True,
        "no_test_group_in_its_model_training_or_selection": True,
        "result_sha256": run.sha(run.OUT / "RESULT.json"),
        "audit_script_sha256": run.sha(Path(__file__)), "elapsed_seconds": time.monotonic() - started}
    run.atomic(run.OUT / "INDEPENDENT_AUDIT.json", outcome)
    return outcome


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False))
