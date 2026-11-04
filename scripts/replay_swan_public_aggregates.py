"""Replay released numerical aggregates without text, models, or API access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source"))
from scripts.analyze_swan_blind_eval import save_json
from scripts.analyze_swan_last_round import analyze
from scripts.recalibrate_swan_5pct import evaluate, threshold_from_calibration
from src.core.swan_blind_protocol import digest


def load(path):
    return json.loads(path.read_text())


def compare(expected, actual, ignored=()):
    maximum = 0.
    def visit(a, b):
        nonlocal maximum
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a)-set(ignored) != set(b)-set(ignored):
                raise ValueError("Replay dictionary fields differ")
            for key in a:
                if key not in ignored:
                    visit(a[key], b[key])
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise ValueError("Replay array shape differs")
            for x, y in zip(a, b):
                visit(x, y)
        elif isinstance(a, float) and isinstance(b, (int, float)):
            delta = abs(a-b)
            maximum = max(maximum, delta)
            if delta > 1e-12:
                raise ValueError("Replay numerical difference exceeds 1e-12")
        elif type(a) is not type(b) or a != b:
            raise ValueError("Replay decisions or counts differ")
    visit(expected, actual)
    return maximum


def replay(folder):
    manifest = load(folder / "SHA256SUMS.json")
    for name, expected in manifest.items():
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Public package SHA mismatch: " + name)
    development = load(folder / "DEVELOPMENT_5PCT_RESULTS.json")
    scores = load(folder / "DEVELOPMENT_DOCUMENT_SCORES.json")
    if scores["freeze_sha256"] != development["source_freeze_sha256"]:
        raise ValueError("Development aggregate provenance differs")
    error = 0.
    rows = scores["documents"]
    methods = development["complete_methods"]
    if len(rows) != 420*len(methods) or len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Development aggregate denominator differs")
    for method in methods:
        measured = development["methods"][method]
        data = [r for r in rows if r["method"] == method]
        negatives = [r for r in data if r["arm"] == "paired_negative"
                     and r["split"] == "calibration"]
        threshold, groups = threshold_from_calibration(negatives, measured["unit_threshold"])
        error = max(error, compare(measured["calibration_by_condition"], groups))
        available = measured["availability"] == "available"
        recomputed = evaluate(data, measured["unit_threshold"], threshold, available)
        error = max(error, compare(measured["condition_guard_5pct"], recomputed))
        old = measured["original_5pct"]
        earlier = evaluate(data, measured["unit_threshold"], old["document_threshold"], available)
        for condition, group in old["conditions"].items():
            # The earlier result schema contains additional reporting fields.
            actual = earlier["conditions"][condition]
            for label in ("TPR", "FPR"):
                for field in ("decisions_positive", "denominator", "rate", "unresolved"):
                    error = max(error, compare(group[label][field], actual[label][field]))
    confirmation = False
    if (folder / "RESULTS.json").exists():
        expected = load(folder / "RESULTS.json")
        frozen_scores = load(folder / "DOCUMENT_SCORES.json")
        if frozen_scores["freeze_sha256"] != expected["freeze_sha256"]:
            raise ValueError("Confirmation aggregate provenance differs")
        # Only candidate_method is used by the numerical analyzer. This temporary
        # wrapper is a replay fixture, not the original experiment freeze.
        fixture = {"candidate_method": expected["candidate_method"]}
        with tempfile.TemporaryDirectory(prefix="swan-public-replay-") as temporary:
            p = Path(temporary)
            save_json(p / "EVAL_FREEZE.json", fixture)
            save_json(p / "DOCUMENT_SCORES.json",
                      {**frozen_scores, "freeze_sha256": digest(fixture)})
            recomputed = analyze(p)
            error = max(error, compare(expected, recomputed, ignored=("freeze_sha256",)))
            error = max(error, compare(load(folder / "CALIBRATION.json"),
                                       load(p / "CALIBRATION.json")))
        confirmation = True
    return {
        "package_sha_verified": True,
        "development_thresholds_counts_and_intervals_replayed": True,
        "development_methods": methods,
        "confirmation_replayed": confirmation,
        "maximum_absolute_difference": error,
        "tolerance": 1e-12,
        "detector_calls": 0,
        "generation_calls": 0,
        "original_experiment_freeze_reconstructed": False,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--package", type=Path, required=True)
    print(json.dumps(replay(cli.parse_args().package), indent=2))
