"""Normal-length atomic arms: shared observations, pretest baseline, real Oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_atomic_alignment_completion2 as old
import run_multiscale_transfer as rt
from swan_final1_common import freeze, load, sha256
from swan_atomic_calibration1 import calibrate, evaluate, threshold, reject
from src.core.multiview_candidates import candidate_windows, select_budget
from run_swan_normal_completion1 import (
    CONDITIONS, FAMILIES, verify, verify_seal, assemble, pair_rows, tokenizer, pilot)


def prepare(parent, assets):
    f = verify(parent)
    verify_seal(parent, parent / "MAIN_GENERATION_SEAL.json")
    verify_seal(parent, parent / "LLM_NULL_GENERATION_SEAL.json")
    docs = assemble(parent)
    out = parent / "alignment"
    if (out / "FREEZE.json").exists():
        old.verify(out)
        if load(out / "FREEZE.json")["parent_freeze_sha256"] != sha256(parent / "FREEZE.json"):
            raise RuntimeError("Wrong normal-length parent")
        return out
    af = load(assets / "FREEZE.json")
    if sha256(assets / "FREEZE.json") != f["alignment_parent_freeze_sha256"]:
        raise RuntimeError("Encoder parent contract changed")
    for name, expected in {**f["encoder_files"], **f["surface_files"]}.items():
        if sha256(assets / name) != expected:
            raise RuntimeError("Encoder or surface model drift")
    design = load(parent / "private/ALIGNMENT_DESIGN.json")
    roles = {"positive_development": ("development", "watermarked"),
             "positive_test": ("positive_test", "watermarked"),
             "human_calibration": ("calibration", "human"),
             "human_test": ("null_test", "human"),
             "llm_calibration": ("calibration", "llm"),
             "llm_test": ("null_test", "llm")}
    expected = {s: {"role": roles[k][0], "kind": roles[k][1]}
                for k, ids in design.items() for s in ids}
    source_meta = {r["allocation_id"]: r for phase in ("main", "calibration", "test")
                   for r in load(parent / "private/sources" / (phase + ".json"))}
    for sid, meta in expected.items():
        meta["source_sha256"] = source_meta[sid]["normalized_source_sha256"]
        meta["url_sha256"] = source_meta[sid]["url_sha256"] if source_meta[sid]["url"] else None
    selected = [d for d in docs if d["source_id"] in expected
                and expected[d["source_id"]]["kind"] != "llm"]
    syntax, tokenize = pilot.FrozenSyntax(), tokenizer()
    for sid in design["llm_calibration"] + design["llm_test"]:
        doc = load(parent / "private/null_documents" / (sid + ".json"))
        clean = doc["text"] if doc["eligible"] else None
        pairs, _ = pair_rows(sid + "-llm", doc["sentences"], syntax, tokenize) if clean else (
            {"A": None, "B": None}, {})
        for c, text in {"clean": clean, **pairs}.items():
            selected.append({"source_id": sid, "condition": c, "text": text})
    order = {"calibration": 0, "development": 1, "null_test": 2, "positive_test": 3}
    selected.sort(key=lambda r: (order[expected[r["source_id"]]["role"]], r["source_id"],
                                 CONDITIONS.index(r["condition"])))
    rows, metadata = [], []
    for d in selected:
        text = d["text"] or ""
        pool = candidate_windows(text) if text else {"candidates": []}
        rid = d["source_id"] + "::" + d["condition"]
        rows.append({"row_id": rid, "text": text, "text_sha256": rt.digest(text),
                     "candidates": select_budget(pool["candidates"], 24) if text else []})
        metadata.append({"row_id": rid, "source_id": d["source_id"], "condition": d["condition"],
                         **expected[d["source_id"]], "unavailable": not bool(text)})
    if len(rows) != 5400 or len({r["row_id"] for r in rows}) != 5400:
        raise RuntimeError("Expected 1800 sources x 3 conditions")
    freeze(out / "private/INPUTS.json", rows)
    freeze(out / "private/METADATA.json", metadata)
    freeze(out / "private/EXPECTED.json", expected)
    for name, target in (("encoder", assets / "private/encoder"),
                         ("surface_model", assets / "private/surface_model"),
                         ("code", parent / "private/code")):
        link = out / "private" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError("Shared dependency link changed")
        elif link.exists():
            raise RuntimeError("Unexpected dependency path")
        else:
            link.symlink_to(target, target_is_directory=True)
    shutil.copyfile(parent / "private/amr_bank_100.json", out / "private/BANK.json")
    freeze(out / "FREEZE.json", {
        "version": "normal-length-atomic/v1", "parent_freeze_sha256": sha256(parent / "FREEZE.json"),
        "parser": f["parser"], "execution": f["execution"], "resource_gate": f["resource_gate"],
        "code_sha256": {p.removeprefix("private/code/"): h for p, h in f["files"].items()
                        if p.startswith("private/code/")},
        "files": {str(p.relative_to(out)): sha256(p) for p in (
            out / "private/INPUTS.json", out / "private/METADATA.json",
            out / "private/EXPECTED.json", out / "private/BANK.json")},
        "counts": {"sources": 1800, "rows": 5400, "conditions": 3},
        "input_seals": {p.name: sha256(p) for p in (
            parent / "INPUT_SEAL.json", parent / "MAIN_GENERATION_SEAL.json",
            parent / "LLM_NULL_GENERATION_SEAL.json")},
        "encoder_files": f["encoder_files"], "surface_files": f["surface_files"],
        "Bank": "fixed Bank100 structural-feature scorer; not official z detector",
        "power_plan": f["power_plan"], "primary_condition": "B"})
    return out


def value(arm):
    return arm["statistic"] if arm["unique_graphs"] else None


def baseline_and_cuts(predictions, metadata, expected):
    mapping = {m["row_id"]: m for m in metadata}
    cal_ids = sorted(s for s, m in expected.items() if m["role"] == "calibration")
    dev_ids = sorted(s for s, m in expected.items() if m["role"] == "development")
    rows = {}
    for p in predictions:
        m = mapping[p["row_id"]]
        if m["role"] not in ("calibration", "development"):
            raise ValueError("Test data cannot enter single-view selection")
        key = (m["source_id"], m["condition"])
        if key in rows:
            raise ValueError("Duplicate calibration/development prediction")
        rows[key] = p
    if set(rows) != {(s, c) for s in cal_ids + dev_ids for c in CONDITIONS}:
        raise ValueError("All calibration and development conditions required")
    names = ("A0", "A1", *FAMILIES)
    def score(p, name):
        return value(p[name] if name in ("A0", "A1") else p["single_view_families"][name])
    cuts = {}
    for name in names:
        scores = [[score(rows[s, c], name) for c in CONDITIONS] for s in cal_ids]
        cuts[name] = threshold([max(v for v in group if v is not None)
                               if any(v is not None for v in group) else None for group in scores], .01)
    successes = {name: sum(reject(score(rows[s, "B"], name), cuts[name]) for s in dev_ids)
                 for name in FAMILIES}
    choice = max(FAMILIES, key=lambda name: successes[name])
    return {"selected_family": choice, "calibration_sources": cal_ids,
            "development_sources": dev_ids, "development_hits": successes,
            "family_order_tie_break": list(FAMILIES), "thresholds": cuts,
            "test_rows_read": False, "selection_criterion": "development_B_detection_rate",
            "source_max_condition_calibration": True}


def export(out, results, rows):
    blind, scores, predictions = [], [], []
    for row, result in zip(rows, results):
        blind.append({
            "row_id": row["row_id"], "text": row["text"], "text_sha256": row["text_sha256"],
            **{k: result[k] for k in ("candidates", "events", "surfaces",
                                     "all_extracted_events", "event_exclusions")},
            "candidate_texts": [c["text"] for c in row["candidates"]],
            "candidate_amrs": [o["amr"] for o in result["observations"]]})
        scores.append({k: result[k] for k in ("row_id", "universe", "observations")})
        predictions.append({k: result[k] for k in ("row_id", "A0", "A1", "single_view_families")})
    for name, val in (("BLIND_INPUTS.json", blind), ("POST_SELECTION_SCORES.json", scores),
                      ("PREDICTIONS.json", predictions)):
        freeze(out / "private" / name, val)
    freeze(out / "OBSERVATION_SEAL.json", {
        "freeze_sha256": sha256(out / "FREEZE.json"),
        "blind_inputs_sha256": sha256(out / "private/BLIND_INPUTS.json"),
        "post_selection_scores_sha256": sha256(out / "private/POST_SELECTION_SCORES.json")})
    freeze(out / "PREDICTION_SEAL.json", {
        "freeze_sha256": sha256(out / "FREEZE.json"),
        "predictions_sha256": sha256(out / "private/PREDICTIONS.json"),
        "observation_seal_sha256": sha256(out / "OBSERVATION_SEAL.json"),
        "single_view_selection_sha256": sha256(out / "BASELINE_SELECTION.json"),
        "human_answers_read": False, "Bank_scores_in_alignment": False})
    for i in (1, 2):
        freeze(out / "private/human_review" / f"reviewer{i}_UNFILLED.json", {
            "reviewer_code": "", "human_attestation": False, "independent_attestation": False,
            "no_model_or_target_answers_used": False, "completed_at": "",
            "freeze_sha256": sha256(out / "FREEZE.json"),
            "prediction_seal_sha256": sha256(out / "PREDICTION_SEAL.json"),
            "rows": [{"row_id": r["row_id"], "pairs": [
                {**p, "same_fact_supported": None, "rationale": ""} for p in r["universe"]]}
                for r in results]})
    return predictions


def endpoint_rows(predictions, metadata, selected, oracle=None):
    mapping = {p["row_id"]: p for p in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r["row_id"] for r in metadata}:
        raise ValueError("Prediction denominator changed")
    if oracle is not None and set(oracle) != set(mapping):
        raise ValueError("Complete independent Oracle is required")
    rows = []
    for m in metadata:
        p = mapping[m["row_id"]]
        scores = {"A0": value(p["A0"]), "A1": value(p["A1"]),
                  "single_view": value(p["single_view_families"][selected])}
        if oracle is not None:
            if oracle[m["row_id"]] is None:
                raise ValueError("Missing human labels are not abstentions")
            scores["A2"] = value(oracle[m["row_id"]])
        rows.append({**{k: m[k] for k in ("source_id", "condition", "role", "kind")},
                     "scores": scores, "abstentions": {k: "no_usable_observation"
                                                     for k, v in scores.items() if v is None}})
    return rows


def run(parent, assets, alignment_assets):
    out = prepare(parent, alignment_assets)
    if (out / "TERMINAL.json").exists():
        t = load(out / "TERMINAL.json")
        if any(t[k] != sha256(out / p) for k, p in (
                ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
            raise RuntimeError("Atomic terminal changed")
        return
    f = old.verify(out)
    for name, expected_hash in {**f["encoder_files"], **f["surface_files"]}.items():
        if sha256(out / name) != expected_hash:
            raise RuntimeError("Atomic model drift")
    rows = load(out / "private/INPUTS.json")
    metadata = load(out / "private/METADATA.json")
    expected = load(out / "private/EXPECTED.json")
    old.raw_graphs(out, assets, f, rows)
    from src.core.bank_scorer import SwanAMRBankScorer
    from src.core.multiview_views import SurfaceAnalyzer
    import spacy
    encoder = old.Encoder(out / "private/encoder")
    surface = object.__new__(SurfaceAnalyzer)
    surface.nlp = spacy.load(out / "private/surface_model")
    class Extractor:
        fingerprint = "new-presealed-normal-length-graphs"
    scorer = SwanAMRBankScorer(Extractor(), str(out / "private/BANK.json"))
    results, selection = [], None
    for index, (row, meta) in enumerate(zip(rows, metadata)):
        if row["row_id"] != meta["row_id"]:
            raise RuntimeError("Row order changed")
        if meta["role"] in ("null_test", "positive_test") and selection is None:
            selection = baseline_and_cuts(results, metadata, expected)
            freeze(out / "BASELINE_SELECTION.json", selection)
        path = out / "private/rows" / (rt.digest(row["row_id"]) + ".json")
        receipt = path.with_suffix(".sha.json")
        if path.exists():
            result = load(path)
            if result["freeze_sha256"] != sha256(out / "FREEZE.json") or result["row_id"] != row["row_id"]:
                raise RuntimeError("Atomic checkpoint identity mismatch")
            if receipt.exists():
                if load(receipt)["sha256"] != sha256(path):
                    raise RuntimeError("Atomic row checkpoint changed")
            else:
                if old.observe(row, out, encoder, surface, scorer) != result:
                    raise RuntimeError("Unsealed checkpoint replay differs")
                freeze(receipt, {"sha256": sha256(path)})
        else:
            result = old.observe(row, out, encoder, surface, scorer)
            freeze(path, result)
            freeze(receipt, {"sha256": sha256(path)})
        results.append(result)
        rt.atomic(out / "HEARTBEAT.json", {"phase": "atomic_scoring", "complete": index + 1,
                                         "total": len(rows), "at": rt.now()})
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise rt.ResourcePause("Atomic disk headroom")
    predictions = export(out, results, rows)
    machine_rows = endpoint_rows(predictions, metadata, selection["selected_family"])
    names = ("A0", "A1", "single_view")
    fitted = calibrate(machine_rows, expected, CONDITIONS, names)
    for name in names:
        key = selection["selected_family"] if name == "single_view" else name
        if fitted["thresholds"][name] != selection["thresholds"][key]:
            raise RuntimeError("Pretest calibration replay differs")
    freeze(out / "MACHINE_CALIBRATION.json", fitted)
    result = {"status": "NORMAL_LENGTH_A0_A1_COMPLETE_A2_AND_QUALITY_PENDING",
              **evaluate(machine_rows, expected, fitted, "B"),
              "independent_human_reviews": 0, "completed_rows": len(rows),
              "selected_single_view": selection["selected_family"],
              "power_plan": f["power_plan"], "three_arm_confirmed": False}
    freeze(out / "RESULT.json", result)
    freeze(out / "AUDIT.json", {
        "freeze_sha256": sha256(out / "FREEZE.json"),
        "prediction_seal_sha256": sha256(out / "PREDICTION_SEAL.json"),
        "raw_graph_seal_sha256": sha256(out / "RAW_GRAPH_SEAL.json"),
        "calibration_sha256": sha256(out / "MACHINE_CALIBRATION.json"),
        "single_view_pretest_seal_sha256": sha256(out / "BASELINE_SELECTION.json"),
        "full_denominator": 1800, "source_condition_rows": 5400,
        "independent_quality_complete": False, "full_fact_inventory_complete": False})
    freeze(out / "TERMINAL.json", {"status": result["status"],
                                   "result_sha256": sha256(out / "RESULT.json"),
                                   "audit_sha256": sha256(out / "AUDIT.json")})


def evaluate_oracle(out, collection, destination):
    old.verify(out)
    seal = load(out / "PREDICTION_SEAL.json")
    if seal["predictions_sha256"] != sha256(out / "private/PREDICTIONS.json"):
        raise RuntimeError("Predictions changed")
    audit, result = load(collection / "AUDIT.json"), load(collection / "RESULT.json")
    if audit["prediction_seal_sha256"] != sha256(out / "PREDICTION_SEAL.json") or audit[
            "freeze_sha256"] != sha256(out / "FREEZE.json"):
        raise RuntimeError("Oracle belongs to another prediction set")
    if result["status"] != "COMPLETE_SUPPORT":
        raise ValueError("Complete two-person support with adjudication required")
    oracle = {r["row_id"]: r["A2"] for r in result["rows"]}
    if len(oracle) != len(result["rows"]):
        raise ValueError("Duplicate Oracle row")
    predictions, meta, expected = [load(out / "private" / p) for p in
                                   ("PREDICTIONS.json", "METADATA.json", "EXPECTED.json")]
    selected = load(out / "BASELINE_SELECTION.json")["selected_family"]
    rows = endpoint_rows(predictions, meta, selected, oracle)
    fitted = calibrate(rows, expected, CONDITIONS)
    machine = load(out / "MACHINE_CALIBRATION.json")
    if any(fitted["thresholds"][k] != machine["thresholds"][k] for k in machine["arms"]):
        raise RuntimeError("Human collection changed machine calibration")
    return freeze(destination, {
        "status": "THREE_ARM_SUPPORT_ORACLE_DIAGNOSTICS_COMPLETE",
        "calibration": fitted, **evaluate(rows, expected, fitted, "B"),
        "oracle_audit_sha256": sha256(collection / "AUDIT.json"),
        "oracle_result_sha256": sha256(collection / "RESULT.json"),
        "A2_calibration_timing": "after frozen predictions and complete independent support labels",
        "A2_inference": "oracle conditional; not a prospective deployed detector",
        "full_fact_inventory_required_for_identity_F1": True,
        "independent_quality_required_for_formal_claim": True})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate_oracle(args.out, args.oracle, args.result)))
