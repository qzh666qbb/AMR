"""Audit completed signal and partial features without resuming a halted run."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from verify_multiscale_transfer_metrics import rank_auc


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".halt-audit.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def complete_mean(values):
    return sum(values) / len(values) if all(map(math.isfinite, values)) else -math.inf


def score_rows(rows, expected):
    assert len(rows) == expected
    for row in rows:
        assert len(row) == 100
        assert all(value is None or
                   (isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1)
                   for value in row)
    return len(rows) * 100


def audit(out, diagnosis):
    assert (out / "HALT.json").exists()
    assert not any((out / name).exists() for name in
                   ("CLEAN_COMPLETE.json", "COMPLETE.json", "FINAL_AUDIT.json", "RESULTS.json"))
    freeze, freeze_sha = read(out / "FREEZE.json"), sha(out / "FREEZE.json")
    assert diagnosis["freeze_sha256"] == freeze_sha
    assert diagnosis["HALT.json"] == read(out / "HALT.json")
    assert all(state in diagnosis["supervisor_status"] for state in ("STOPPED", "EXITED"))
    assert "RUNNING" not in diagnosis["supervisor_status"]
    assert any("RuntimeError: Global GPU memory gate failed" in lines
               for lines in diagnosis["exception_lines"].values())
    assert not diagnosis.get("unbound_event_metadata")
    assert freeze["no_generation"] and freeze["no_score_filtering"]
    assert not freeze["calibration"]["test_tuning"]
    assert not freeze["calibration"]["refit_or_refit_normalizer"]
    frozen_files = {
        **freeze["sources"], **freeze["normalizer_assets"],
        "private/DOCUMENTS.json": freeze["documents_sha256"],
        "private/BANK.json": freeze["bank_sha256"],
        "private/propbank_verbs.json": freeze["propbank_verbs_sha256"],
        "WORKLOAD.json": freeze["resource_gate"]["workload_sha256"],
        **{row["file"]: row["sha256"] for row in freeze["checkpoints"]},
    }
    for name, expected in frozen_files.items():
        assert sha(out / name) == expected, name
    assert sha(out.parent / "MANIFEST_GENERATION.sha256") == freeze["original_generation_manifest_sha256"]
    assert len(read(out / "private/BANK.json")) == 100

    populations = ("calibration", "signal_decision", "readout")
    groups = [set(freeze["split"][name]) for name in populations]
    assert [len(group) for group in groups] == [6, 6, 12]
    assert all(not left & right for i, left in enumerate(groups) for right in groups[i + 1:])
    documents = {row["doc_id"]: row for row in read(out / "private/DOCUMENTS.json")}
    assert set(documents) == set.union(*groups) and len(documents) == 24
    signals = {}
    signal_sentences, signal_cells = 0, 0
    for path in sorted((out / "private/signal").glob("*.json")):
        row = read(path)
        docid = row["doc_id"]
        assert docid not in signals and row["freeze_sha256"] == freeze_sha
        assert set(row["arms"]) == {"positive", "unwatermarked"}
        for arm, values in row["arms"].items():
            carriers = documents[docid]["arms"][arm]["carriers"]
            assert len(carriers) == 8
            assert values["sentence_sha256"] == [text_sha(carrier["text"]) for carrier in carriers]
            signal_cells += score_rows(values["sentence_scores"], 8)
            signal_sentences += 8
        signals[docid] = row
    assert set(signals) == set(documents)
    assert signal_sentences == 384 and signal_cells == 38400
    result, marker, gate = [read(out / name) for name in
                            ("SIGNAL_RESULTS.json", "SIGNAL_COMPLETE.json", "CLEAN_GATE.json")]
    assert all(row["freeze_sha256"] == freeze_sha for row in (result, marker, gate))
    assert marker["signal_results_sha256"] == sha(out / "SIGNAL_RESULTS.json")
    assert marker["sentences"] == 384 and marker["pairs"] == 24

    def sentence_values(docid, arm):
        return [max((value for value in row if value is not None), default=-math.inf)
                for row in signals[docid]["arms"][arm]["sentence_scores"]]

    for population in populations:
        metric = result["populations"][population]
        ids = freeze["split"][population]
        assert metric["pairs"] == len(ids)
        carrier, document = {}, {}
        for arm in ("positive", "unwatermarked"):
            values = [sentence_values(docid, arm) for docid in ids]
            carrier[arm] = [value for row in values for value in row]
            document[arm] = [complete_mean(row) for row in values]
        for view, values in (("carrier", carrier), ("document", document)):
            positive, negative = values["positive"], values["unwatermarked"]
            assert abs(rank_auc(positive, negative) - metric[view + "_auroc"]) < 1e-12
            paired_win = sum((p > n) + 0.5 * (p == n)
                             for p, n in zip(positive, negative)) / len(positive)
            assert abs(paired_win - metric[view + "_paired_win"]) < 1e-12
        for arm, prefix in (("positive", "positive"), ("unwatermarked", "negative")):
            values = carrier[arm]
            assert sum(not math.isfinite(value) for value in values) == metric[prefix + "_abstain"]
            mean = complete_mean(values)
            assert (metric[prefix + "_mean"] is None if mean == -math.inf else
                    abs(mean - metric[prefix + "_mean"]) < 1e-12)
    decision = result["populations"]["signal_decision"]
    assert gate["passed"] == (
        decision["carrier_auroc"] >= freeze["clean_gate"]["carrier_auroc_min"] and
        decision["document_paired_win"] >= freeze["clean_gate"]["document_paired_win_min"])
    assert gate["carrier_auroc"] == decision["carrier_auroc"]
    assert gate["document_paired_win"] == decision["document_paired_win"]
    assert not gate["readout_used"]

    geometry_path = out / "private/code/src/core/multiscale_blind.py"
    spec = importlib.util.spec_from_file_location("halt_verified_geometry", geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    expected_parse_keys = set()
    for document in documents.values():
        for arm in document["arms"].values():
            expected_parse_keys.update(text_sha(carrier["text"]) for carrier in arm["carriers"])
            for condition in freeze["conditions"]:
                expected_parse_keys.update(text_sha(window["text"])
                                           for windows in geometry.blind_geometry(arm["texts"][condition]).values()
                                           for window in windows)
    workload = read(out / "WORKLOAD.json")
    assert len(expected_parse_keys) == workload["unique_all_parse_inputs"]
    assert text_sha("\n".join(sorted(expected_parse_keys))) == workload["all_parse_key_manifest_sha256"]
    parses = {}
    invalid_parses = 0
    for path in (out / "private/parses").glob("*.json"):
        parsed = read(path)
        assert path.stem in expected_parse_keys
        assert parsed["freeze_sha256"] == freeze_sha and parsed["text_sha256"] == path.stem
        assert parsed["amr_sha256"] == (text_sha(parsed["amr"]) if parsed["amr"] else None)
        invalid_parses += not parsed["valid"]
        parses[path.stem] = parsed
    for row in signals.values():
        for arm in row["arms"].values():
            assert set(arm["sentence_sha256"]) <= set(parses)

    records, candidate_cells, carrier_anchors, blind_anchors = [], 0, 0, 0
    extractor_errors = 0
    for path in sorted((out / "private/records").glob("*.json")):
        row = read(path)
        assert row["freeze_sha256"] == freeze_sha and row["condition"] == "clean"
        assert row["doc_id"] in documents
        assert set(row["arms"]) == {"positive", "unwatermarked"}
        for name, arm in row["arms"].items():
            original = documents[row["doc_id"]]["arms"][name]
            assert arm["text_sha256"] == text_sha(original["texts"]["clean"])
            assert arm["geometry"] == geometry.blind_geometry(original["texts"]["clean"])
            assert arm["official_sentence_scores"] == signals[row["doc_id"]]["arms"][name]["sentence_scores"]
            for scale, rows in arm["candidate_scores"].items():
                windows = arm["geometry"]["S12" if scale == "Ssub" else scale]
                candidate_cells += score_rows(rows, len(windows))
                assert all(text_sha(window["text"]) in parses for window in windows)
            candidates = arm["candidate_scores"]
            args = (arm["geometry"], candidates["S6"], candidates["S12"], candidates["Ssub"])
            assert arm["blind_features"] == geometry.assemble_features(*args)
            assert arm["carrier_features"] == geometry.assemble_features(*args, original["carriers"])
            carrier_anchors += len(arm["carrier_features"])
            blind_anchors += len(arm["blind_features"])
            extractor_errors += len(arm["extractor_errors"])
        records.append(row)
    assert len(records) == diagnosis["records_by_condition"]["clean"] == 5
    assert len(parses) == diagnosis["parse_records"] == 2395
    completed_ids = {row["doc_id"] for row in records}
    assert completed_ids <= groups[0]
    assert len(completed_ids) == len(records)
    output = {
        "passed": True, "scope": "Completed clean signal and partial saved artifacts only",
        "at": datetime.now(timezone.utc).isoformat(), "freeze_sha256": freeze_sha,
        "audit_source_sha256": sha(Path(__file__)),
        "rank_verifier_source_sha256": sha(Path(__file__).with_name("verify_multiscale_transfer_metrics.py")),
        "signal_results_sha256": sha(out / "SIGNAL_RESULTS.json"),
        "full_experiment_completed": False, "halt_retained": True,
        "halt_cause": "Global GPU memory gate failed",
        "exact_crossing_memory_mib_not_recorded": True,
        "frozen_assets_sha_checked": len(frozen_files),
        "frozen_checkpoint_files_checked": len(freeze["checkpoints"]),
        "original_signal_pairs_checked": 24, "original_signal_sentences_checked": signal_sentences,
        "original_signal_score_cells_checked": signal_cells,
        "rank_statistics_checked_populations": list(populations),
        "all_signal_original_sentence_parse_keys_present": True,
        "partial_feature_pairs": len(records), "expected_clean_feature_pairs": 24,
        "partial_features_population": "calibration only",
        "partial_carrier_feature_shape": [carrier_anchors, 3, 100],
        "partial_blind_feature_shape": [blind_anchors, 3, 100],
        "candidate_score_cells_checked": candidate_cells,
        "cached_parse_sha_checked": len(parses), "invalid_cached_parses": invalid_parses,
        "expected_unique_parse_inputs": len(expected_parse_keys),
        "missing_expected_parse_inputs": len(expected_parse_keys - set(parses)),
        "partial_extractor_errors": extractor_errors,
        "model_comparison_available": False, "attack_evaluation_available": False,
        "no_parser_detector_or_model_calls": True, "no_resume_or_successor": True,
        "no_model_training_selection_or_threshold_tuning": True,
        "no_original_stage1b_gate_rewrite": True,
    }
    write(out / "HALT_DIAGNOSIS.json", diagnosis)
    write(out / "HALT_AUDIT.json", output)
    write(out / "private/HALT_ARTIFACT_SHA256SUMS.json", {
        str(path.relative_to(out)): sha(path)
        for path in sorted((out / "private").rglob("*"))
        if path.is_file() and path.name != "HALT_ARTIFACT_SHA256SUMS.json"
        and "__pycache__" not in path.parts and path.suffix in (".json", ".jsonl", ".py", ".pt")
    })
    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    args = parser.parse_args()
    audit(args.out.resolve(), read(args.diagnosis))
