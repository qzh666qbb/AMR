"""Quality-frozen, ordered final1 detection with immutable per-operation recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from swan_final1_analysis import (
    CONDITIONS, aggregate_vectors, asr_denominators, calibrate, match_humans,
    missing_effect_bounds, primary_analysis, rates,
)
from swan_final1_common import append_event, assert_hashes, digest, freeze, load, now, seed, sha256
from swan_final1_contract import BANK, EXPECTED_BANK
from swan_final1_generation import load_slots
from swan_final1_quality import parse_utc
from swan_final1_transform import text_hash
from swan_final1_workflow import (
    baseline_cohort, cohort_gates, decision_for, documents_for, pair_records, verified_transforms,
)

STAGES = ("gate_b_calibration", "gate_b_positive", "calibration", "positive", "test")


def evidence_files(folder):
    files = [folder / "manifest.json", folder / "source_manifest.json"]
    for relative in ("sealed", "documents", "human", "quality", "transforms",
                     "private/pilot", "private/main", "private/local_errors"):
        files.extend(p for p in (folder / relative).rglob("*") if p.is_file())
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(set(files))}


def condition_input(base, row, condition, targets):
    spans = row["segmentation_spans"]
    text = row["restored_text"] if condition == "C4" else row["text"]
    units = []
    for a, b in spans:
        # Character substitution preserves offsets; all overlaps retained, no best target.
        overlap = [{**target, "overlap_chars": min(b, target["span"][1])-max(a, target["span"][0])}
                   for target in targets if max(a, target["span"][0]) < min(b, target["span"][1])]
        units.append({"span": [a, b], "text": text[a:b], "target_overlaps": overlap})
    return {"allocation_id": base["allocation_id"], "condition": condition,
            "text": text, "document_sha256": text_hash(text), "units": units,
            "N": len(units), "domain": base["domain"], "ordinal": base["ordinal"],
            "baseline_word_count": base["word_count"],
            "baseline_N": base["num_detection_units"]}


def phase_inputs(folder, phase):
    documents = documents_for(folder, phase)
    cohort = baseline_cohort(folder, phase, documents)
    transformed = verified_transforms(folder, phase, cohort)
    labels = decision_for(folder, phase, "pair", pair_records(transformed))
    passed = {(r["allocation_id"], r["condition_id"]) for r in labels["decisions"]
              if r["quality_pass"]}
    by_condition = {c: [] for c in CONDITIONS}
    attrition = []
    baseline_ids = {r["allocation_id"] for r in cohort}
    transform_map = {r["allocation_id"]: r for r in transformed}
    baseline_map = {r["allocation_id"]: r for r in cohort}
    bank = [r["normalized_template_amr"] for r in load(BANK)]
    for base in documents:
        allocation = base["allocation_id"]
        row = transform_map.get(allocation)
        attrition.append({
            "allocation_id": allocation, "generation_complete": base.get("generation_complete"),
            "termination": base["termination"], "identity_pass": base["identity_pass"],
            "structure_pass": base["structural_pass"], "baseline_quality_pass": allocation in baseline_ids,
            "baseline_review_eligible": base["structural_pass"],
            "construction": {c: bool(row and c in row["conditions"]) for c in CONDITIONS[1:]},
            "pair_quality_pass": {c: (allocation, c) in passed for c in ("C1", "C2", "C3")},
            "construction_failures": row["failures"] if row else None,
        })
        if row is None:
            continue
        targets = []
        if phase == "main":
            slots, _ = load_slots(folder, phase, allocation)
            for slot, span in zip(slots, base["segmentation_spans"]):
                matches = [i for i, graph in enumerate(bank) if graph == slot["chosen_template"]]
                if not matches:
                    raise ValueError("Injection target absent from frozen bank")
                targets.append({"slot": slot["slot"], "span": span,
                                "bank_indices": matches, "accepted": slot["accepted"]})
        for c in CONDITIONS:
            if c not in row["conditions"] or (c in ("C1", "C2", "C3") and (allocation, c) not in passed):
                continue
            by_condition[c].append(condition_input(base, row["conditions"][c], c, targets))
    return {"inputs": by_condition, "baseline": list(baseline_map.values()),
            "attrition": attrition, "quality_agreement": labels["agreement"]}


def build_plan(folder):
    """All labels are verified before any score is available; no inference here."""
    gates = cohort_gates(folder, "main")
    if not gates["passed"]:
        raise PermissionError("Main A/C/D failed; confirmatory detection prohibited")
    phases = {p: phase_inputs(folder, p) for p in ("main", "calibration", "test")}
    main = phases["main"]
    baseline = main["baseline"]
    common_ids = set(gates["common_ids"])
    common = [r for r in baseline if r["allocation_id"] in common_ids]
    matches = {"gate_b_calibration": match_humans(baseline, phases["calibration"]["baseline"])}
    for p in ("calibration", "test"):
        matches[p] = {}
        for c in CONDITIONS:
            qualified = {r["allocation_id"] for r in phases[p]["inputs"][c]}
            candidates = [r for r in phases[p]["baseline"] if r["allocation_id"] in qualified]
            matches[p][c] = match_humans(common, candidates)
    all_matches = [matches["gate_b_calibration"], *matches["calibration"].values(),
                   *matches["test"].values()]
    jobs = {}
    def select(p, c, ids):
        selected = set(ids)
        return [r for r in phases[p]["inputs"][c] if r["allocation_id"] in selected]
    def human_ids(match):
        return [a for r in match["pairs"] for a in r["human_ids"]]
    jobs["gate_b_calibration"] = select("calibration", "C0", human_ids(matches["gate_b_calibration"]))
    jobs["gate_b_positive"] = main["inputs"]["C0"]
    for p in ("calibration", "test"):
        jobs[p] = [r for c in CONDITIONS for r in select(p, c, human_ids(matches[p][c]))]
    # Score all condition-qualified positives for ITT/ASR, not only complete pairs.
    jobs["positive"] = [r for c in CONDITIONS for r in main["inputs"][c]]
    return {
        "schema_version": 1, "manifest_sha256": sha256(folder / "manifest.json"),
        "evidence_hashes": evidence_files(folder), "cohort_gates": gates,
        "matches": matches, "matching_pass": all(r["passed"] for r in all_matches),
        "common_ids": [r["allocation_id"] for r in common],
        "baseline_ids": [r["allocation_id"] for r in baseline],
        "attrition": {p: phases[p]["attrition"] for p in phases},
        "quality_agreement": {p: phases[p]["quality_agreement"] for p in phases},
        "jobs": jobs, "quality_frozen_before_any_detection": True,
    }


def freeze_plan(folder):
    path = folder / "detection_plan.json"
    if path.exists():
        return verify_plan(folder)
    if (folder / "detection").exists():
        raise PermissionError("Cannot freeze/reselect matching after detection starts")
    value = build_plan(folder)
    value["frozen_at"] = now()
    for decision in (folder / "quality").glob("*/*/decision.json"):
        if parse_utc(load(decision)["frozen_at"]) > parse_utc(value["frozen_at"]):
            raise ValueError("Quality freeze postdates detection plan")
    return freeze(path, value)


def verify_plan(folder):
    saved = load(folder / "detection_plan.json")
    assert_hashes(saved["evidence_hashes"], folder)
    actual = build_plan(folder)
    if {k: v for k, v in saved.items() if k != "frozen_at"} != actual:
        raise ValueError("Detection plan cannot be rebuilt from frozen labels and texts")
    return saved


def revision_identity(folder):
    manifest = load(folder / "manifest.json")
    return {
        "experiment_manifest_sha256": sha256(folder / "manifest.json"),
        "environment_lock_sha256": sha256(folder / "environment_lock.json"),
        "parser": "amrlib-0.8.0/BART-large-v0_1_0/cpu/batch1",
        "template_code_sha256": sha256(BANK.parents[2] / "utils/amr_utils.py"),
        "SWAN_commit": "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a",
        "s2match_upstream": "711a231d", "bank_sha256": manifest["bank_sha256"],
        "direction": "T(R(s))_then_bank", "score_threshold": .65, "lambda": .25,
    }


def graph_properties(text):
    import penman
    graph = penman.decode(text)
    nodes = set(graph.variables())
    if not nodes or graph.top not in nodes:
        raise ValueError("Empty graph")
    adjacency = {v: set() for v in nodes}
    for edge in graph.edges():
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    remaining, components = set(nodes), 0
    while remaining:
        todo = [remaining.pop()]
        components += 1
        while todo:
            for other in adjacency[todo.pop()] & remaining:
                remaining.remove(other)
                todo.append(other)
    return {"nodes": len(nodes), "edges": len(graph.edges()),
            "attributes": len(graph.attributes()), "triples": len(graph.triples),
            "components": components, "connected": components == 1}


class MissingOperation(RuntimeError):
    pass


class OperationStore:
    """One hash-chained journal per unit, avoiding millions of tiny score files."""
    def __init__(self, directory):
        self.path = directory / "operations.jsonl"
        self.previous, self.rows = "0"*64, {}
        if self.path.exists():
            raw = self.path.read_bytes()
            if not raw.endswith(b"\n"):
                raise ValueError("Truncated operation journal; no automatic repair")
            for line in raw.splitlines():
                event = json.loads(line)
                checksum = event.pop("sha256")
                if event["previous"] != self.previous or digest(event) != checksum:
                    raise ValueError("Operation journal chain changed")
                self.previous = checksum
                name, kind = event["name"], event["kind"]
                if kind == "start":
                    if name in self.rows:
                        raise ValueError("Duplicate operation start")
                    self.rows[name] = {"start": event, "terminal": None}
                elif kind in ("complete", "failure"):
                    if name not in self.rows or self.rows[name]["terminal"] is not None:
                        raise ValueError("Invalid operation terminal event")
                    self.rows[name]["terminal"] = event
                else:
                    raise ValueError("Unknown operation event")

    def event(self, name, kind, **fields):
        event = {"previous": self.previous, "name": name, "kind": kind, "at": now(), **fields}
        checksum = digest(event)
        append_event(self.path, {**event, "sha256": checksum})
        self.previous = checksum
        if kind == "start":
            self.rows[name] = {"start": event, "terminal": None}
        else:
            self.rows[name]["terminal"] = event

    def operation(self, name, identity, callback=None):
        saved = self.rows.get(name)
        if saved is not None:
            if saved["start"]["identity"] != identity:
                raise ValueError("Operation identity changed")
            terminal = saved["terminal"]
            if terminal and terminal["kind"] == "complete":
                return terminal["payload"]
            raise MissingOperation("Operation interrupted/failed; cannot retry")
        if callback is None:
            raise MissingOperation("Saved complete operation missing")
        self.event(name, "start", identity=identity)
        began = time.monotonic()
        try:
            payload = callback()
            self.event(name, "complete", payload=payload,
                       elapsed_seconds=time.monotonic()-began)
            return payload
        except Exception as exc:
            self.event(name, "failure", reason=type(exc).__name__)
            raise MissingOperation(f"{name} failed; saved error, no retry") from exc


def validate_score(score):
    if any(not isinstance(score[k], (int, float)) or not math.isfinite(score[k])
           or not 0 <= score[k] <= 1 for k in ("precision", "recall", "f1")):
        raise ValueError("Nonfinite/invalid S2Match; cannot turn into red")
    p, r = score["precision"], score["recall"]
    expected = 2*p*r/(p+r) if p+r else 0
    if not math.isclose(score["f1"], expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("F1 differs from saved P/R")
    return score


def content_identity(unit, revision):
    span = unit["span"]
    if len(span) != 2 or span[1]-span[0] != len(unit["text"]):
        raise ValueError("Unit text and absolute segmentation span differ")
    return {"text_sha256": text_hash(unit["text"]), "span": span,
            "revision": revision}


def unit_score(folder, unit, revision, bank, module=None):
    from swan_final1_runtime import score_triple, swan
    identity = content_identity(unit, revision)
    key = digest(identity)
    directory = folder / "detection" / "cache" / key
    store = OperationStore(directory)
    parser_seed = seed("detection-parser", key)
    def parse():
        with swan.fixed_score_seed(parser_seed):
            return module.parse_amr(unit["text"])
    raw = store.operation("raw", {**identity, "seed": parser_seed}, parse if module else None)
    candidate = store.operation("template", {**identity, "raw_sha256": text_hash(raw)},
                                (lambda: module.normalize_amr_variables(
                                    module.generate_template_amr(
                                        module.normalize_amr_variables(raw)))) if module else None)
    raw_features, candidate_features = graph_properties(raw), graph_properties(candidate)
    scores = []
    for index, graph in enumerate(bank):
        score_seed = seed("detection-score", key, index)
        score_id = {**identity, "candidate_sha256": text_hash(candidate),
                    "bank_index": index, "bank_graph_sha256": text_hash(graph), "seed": score_seed}
        def compute():
            p, r, f = [float(v) for v in score_triple(candidate, graph, score_seed)]
            return validate_score({"precision": p, "recall": r, "f1": f})
        score = store.operation(f"bank_{index:02d}", score_id, compute if module else None)
        validate_score(score)
        scores.append({**score, "bank_index": index, "score_seed": score_seed})
    maximum = max(r["f1"] for r in scores)
    return {"cache_key": key, "input_sha256": identity["text_sha256"],
            "operations_sha256": sha256(store.path),
            "content_span": identity["span"], "parser_seed": parser_seed, "revision": revision,
            "raw_amr": raw, "candidate_template_amr": candidate,
            "raw_graph": raw_features, "candidate_graph": candidate_features,
            "bank_scores": scores, "max_bank_score": maximum,
            "max_bank_indices": [r["bank_index"] for r in scores if r["f1"] == maximum],
            "green": maximum >= .65, "errors": []}


def score_job(folder, job, revision, bank, module=None):
    units = []
    for unit in job["units"]:
        value = unit_score(folder, unit, revision, bank, module)
        targets = [{**r, "scores": [value["bank_scores"][i] for i in r["bank_indices"]]}
                   for r in unit["target_overlaps"]]
        units.append({**value, "text": unit["text"], "document_span": unit["span"],
                      "target_overlaps": targets,
                      "provenance_cache_key": digest({
                          "document_sha256": job["document_sha256"], "span": unit["span"],
                          "content_cache_key": value["cache_key"]})})
    summary = aggregate_vectors([[r["f1"] for r in u["bank_scores"]] for u in units])
    return {"allocation_id": job["allocation_id"], "condition": job["condition"],
            "input_sha256": digest(job), "units": units, **summary}


def stage_path(folder, stage):
    return folder / "detection" / "stages" / f"{stage}.json"


def stage_rows(folder, stage, plan, revision, bank):
    saved = load(stage_path(folder, stage))
    if saved["plan_sha256"] != sha256(folder / "detection_plan.json"):
        raise ValueError("Stage uses a different plan")
    rows, missing = [], []
    for job in plan["jobs"][stage]:
        try:
            rows.append(score_job(folder, job, revision, bank))
        except MissingOperation:
            missing.append({"allocation_id": job["allocation_id"], "condition": job["condition"],
                            "reason": "missing_saved_operation_no_retry"})
    expected = {"plan_sha256": sha256(folder / "detection_plan.json"),
                "planned": len(plan["jobs"][stage]), "rows": rows, "missing": missing,
                "complete": not missing}
    if saved != expected:
        raise ValueError("Stage summary cannot be reproduced from saved vectors")
    return saved


def require_stage(folder, stage, plan, revision, bank):
    row = stage_rows(folder, stage, plan, revision, bank)
    if not row["complete"]:
        raise PermissionError(f"{stage} has missing scores; branch C, later detection prohibited")
    return row["rows"]


def threshold_table(folder, stage, plan, revision, bank, create=False):
    human = require_stage(folder, stage, plan, revision, bank)
    values = ({"C0": calibrate(human)} if stage == "gate_b_calibration" else
              {c: calibrate(rows) for c, rows in group_conditions(human).items()})
    record = {"plan_sha256": sha256(folder / "detection_plan.json"),
              "calibration_stage_sha256": sha256(stage_path(folder, stage)), "values": values}
    path = folder / "detection" / "thresholds" / f"{stage}.json"
    if create:
        freeze(path, record)
    if load(path) != record:
        raise ValueError("Thresholds differ from frozen calibration")
    return values


def baseline_gate(folder, plan, revision, bank):
    threshold = threshold_table(folder, "gate_b_calibration", plan, revision, bank)["C0"]
    positive = require_stage(folder, "gate_b_positive", plan, revision, bank)
    rate = rates(positive, threshold["threshold"])
    return {**threshold, "baseline_TPR": rate,
            "passed": rate["successes"]*5 >= rate["n"]*4}


def stage_prerequisites(folder, stage, plan, revision, bank):
    if stage not in STAGES:
        raise ValueError("Explicit --stage required")
    if not plan["matching_pass"]:
        raise PermissionError("Frozen human matching failed; branch C")
    if stage == "gate_b_positive":
        threshold_table(folder, "gate_b_calibration", plan, revision, bank, create=True)
    if stage in ("calibration", "positive", "test"):
        if not baseline_gate(folder, plan, revision, bank)["passed"]:
            raise PermissionError("Baseline Gate B failed; branch C")
    if stage in ("positive", "test"):
        threshold_table(folder, "calibration", plan, revision, bank, create=stage == "positive")
    if stage == "test":
        require_stage(folder, "positive", plan, revision, bank)


def execute_stage(folder, stage, plan, revision, bank):
    stage_prerequisites(folder, stage, plan, revision, bank)
    if stage_path(folder, stage).exists():
        saved = stage_rows(folder, stage, plan, revision, bank)
        return {"stage": stage, "complete": saved["complete"], "missing": len(saved["missing"])}
    from swan_final1_runtime import official_runtime
    rows, missing = [], []
    # The context loads local weights only after all evidence/gates are verified.
    with official_runtime(folder, error_folder=folder / "detection/errors") as module:
        for job in plan["jobs"][stage]:
            try:
                rows.append(score_job(folder, job, revision, bank, module))
            except MissingOperation:
                missing.append({"allocation_id": job["allocation_id"], "condition": job["condition"],
                                "reason": "missing_saved_operation_no_retry"})
    saved = {"plan_sha256": sha256(folder / "detection_plan.json"),
             "planned": len(plan["jobs"][stage]), "rows": rows, "missing": missing,
             "complete": not missing}
    freeze(stage_path(folder, stage), saved)
    return {"stage": stage, "complete": not missing, "planned": saved["planned"],
            "missing": len(missing), "branch": None if not missing else "C"}


def group_conditions(rows):
    return {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}


def failure_report(folder, plan, revision, bank):
    stages = {}
    for stage in STAGES:
        if stage_path(folder, stage).exists():
            row = stage_rows(folder, stage, plan, revision, bank)
            stages[stage] = {"planned": row["planned"], "completed": len(row["rows"]),
                             "failures": row["missing"],
                             "failure_rate": len(row["missing"])/row["planned"] if row["planned"] else None,
                             "by_condition": stage_counts(plan["jobs"][stage], row)}
    report = {"matching_pass": plan["matching_pass"], "stages": stages,
              "attrition": plan["attrition"], "confirmatory_tests_performed": False}
    if not plan["matching_pass"] or any(r["failures"] for r in stages.values()):
        report["branch"] = "C"
    else:
        report["state"] = "pending_stages"
    if stage_path(folder, "positive").exists():
        positive = stage_rows(folder, "positive", plan, revision, bank)["rows"]
        if stage_path(folder, "calibration").exists():
            table = threshold_table(folder, "calibration", plan, revision, bank)
            thresholds = {c: table[c]["threshold"] for c in CONDITIONS}
            by_key = {(r["allocation_id"], r["condition"]): r for r in positive}
            vectors = {c: [by_key[a, c]["z"] > thresholds[c] if (a, c) in by_key else None
                           for a in plan["common_ids"]] for c in CONDITIONS}
            report["missing_sensitivity"] = {
                "H1": missing_effect_bounds(vectors["C0"], vectors["C1"]),
                "H2": missing_effect_bounds(vectors["C0"], vectors["C2"]),
                "H3": missing_effect_bounds(vectors["C4"], vectors["C2"]),
            }
            groups = group_conditions(positive)
            report["ASR_with_unknown_detection"] = {
                c: asr_denominators(groups["C0"],
                                   [r["allocation_id"] for r in plan["jobs"]["positive"]
                                    if r["condition"] == c],
                                   groups[c], thresholds["C0"], thresholds[c])
                for c in ("C1", "C2", "C3")}
    return report


def stage_counts(jobs, row):
    result = {}
    for c in CONDITIONS:
        planned = sum(r["condition"] == c for r in jobs)
        failed = sum(r["condition"] == c for r in row["missing"])
        completed = sum(r["condition"] == c for r in row["rows"])
        result[c] = {"planned": planned, "completed": completed, "failed": failed,
                     "pending": planned-completed-failed,
                     "failure_rate": failed/planned if planned else None}
    return result


def analyze(folder, plan, revision, bank):
    if not plan["matching_pass"] or not all(stage_path(folder, s).exists() for s in STAGES):
        report = failure_report(folder, plan, revision, bank)
        if stage_path(folder, "gate_b_positive").exists():
            try:
                gate = baseline_gate(folder, plan, revision, bank)
                report["baseline_gate"] = gate
                if not gate["passed"]:
                    report.update(branch="C", state="baseline_gate_failed")
            except PermissionError:
                report["branch"] = "C"
        return report
    saved = {s: stage_rows(folder, s, plan, revision, bank) for s in STAGES}
    if any(not r["complete"] for r in saved.values()):
        return failure_report(folder, plan, revision, bank)
    gate = baseline_gate(folder, plan, revision, bank)
    if not gate["passed"]:
        return {"branch": "C", "baseline_gate": gate, "confirmatory_tests_performed": False}
    calibration_summary = threshold_table(folder, "calibration", plan, revision, bank)
    thresholds = {c: calibration_summary[c]["threshold"] for c in CONDITIONS}
    positive = group_conditions(saved["positive"]["rows"])
    test = group_conditions(saved["test"]["rows"])
    index = {c: {r["allocation_id"]: r for r in positive[c]} for c in CONDITIONS}
    common = {c: [index[c][a] for a in plan["common_ids"]] for c in CONDITIONS}
    report = primary_analysis(common, thresholds, test)
    report.update(baseline_gate=gate, calibration=calibration_summary,
                  matching=plan["matches"], attrition=plan["attrition"],
                  detection_counts={s: stage_counts(plan["jobs"][s], saved[s]) for s in STAGES},
                  quality_agreement=plan["quality_agreement"],
                  full_condition_TPR={c: rates(positive[c], thresholds[c]) for c in CONDITIONS},
                  ASR={c: asr_denominators(
                      positive["C0"], index[c], positive[c], thresholds["C0"], thresholds[c])
                       for c in ("C1", "C2", "C3")})
    report["stratified_acceptance_descriptive"] = {}
    for name, predicate in (("below_half", lambda n, d: n*2 < d),
                            ("at_least_half", lambda n, d: n*2 >= d)):
        ids = []
        for allocation in plan["common_ids"]:
            slots, _ = load_slots(folder, "main", allocation)
            if predicate(sum(r["accepted"] for r in slots), len(slots)):
                ids.append(allocation)
        report["stratified_acceptance_descriptive"][name] = {
            c: rates([index[c][a] for a in ids], thresholds[c]) for c in CONDITIONS}
    report.update(confirmatory_tests_performed=True,
                  plan_sha256=sha256(folder / "detection_plan.json"),
                  stage_hashes={s: sha256(stage_path(folder, s)) for s in STAGES})
    return report


def detection_command(folder, command, stage=None):
    folder = Path(folder)
    if command == "audit" and not (folder / "detection_plan.json").exists():
        from swan_final1_sources import audit_identities
        from swan_reframing_statistics import verify_official
        return {"state": "preexecution_only", "sources": audit_identities(folder),
                "formula": verify_official(), "final_experiment_not_performed": True}
    if command == "freeze-detection-plan":
        plan = freeze_plan(folder)
        return {"matching_pass": plan["matching_pass"], "common_pairs": len(plan["common_ids"]),
                "plan_sha256": sha256(folder / "detection_plan.json")}
    plan = verify_plan(folder)
    revision = revision_identity(folder)
    if revision["bank_sha256"] != EXPECTED_BANK or sha256(BANK) != EXPECTED_BANK:
        raise ValueError("Frozen bank differs")
    bank = [r["normalized_template_amr"] for r in load(BANK)]
    if command == "detect":
        return execute_stage(folder, stage, plan, revision, bank)
    report = analyze(folder, plan, revision, bank)
    if report.get("confirmatory_tests_performed"):
        freeze(folder / "results.json", report)
        freeze(folder / "attrition.json", plan["attrition"])
    if command == "audit":
        result = {"state": "saved_artifacts_replayed_without_inference", "result": report,
                  "parser_scorer_generation_calls": 0, "independent_human_audit": False}
        if report.get("confirmatory_tests_performed"):
            freeze(folder / "audit.json", result)
            freeze(folder / "recovery_verification.json", {
                "results_sha256": sha256(folder / "results.json"),
                "audit_sha256": sha256(folder / "audit.json"),
                "saved_vectors_and_original_labels_replayed": True,
                "parser_scorer_generation_calls": 0, "independent_human_audit": False})
        return result
    return report
