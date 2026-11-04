"""Supplementary saved-artifact audit; no generation, neural inference or tuning.

The coverage checker deliberately uses its own PENMAN adjacency construction and
injective backtracking, rather than the original NetworkX monomorphism matcher.
Human truth and external data independence are outside this audit's scope.
"""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import penman
from penman.models.amr import model as amr_model
import spacy

from run_semantic_sequence_cycle import OUT, PRIVATE, ROOT, atomic, digest, sha
from audit_semantic_sequence_alignment import normalization_namespace


class BoundDiagnostics(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.events = []

    def emit(self, record):
        self.events.append({"level": record.levelname,
                            "message_sha256": digest(record.getMessage())})


def adjacency(text):
    graph = penman.decode(text, model=amr_model)
    variables = {t.source: t.target for t in graph.instances()}
    nodes = {v: ("variable", c) for v, c in variables.items()}
    edges = Counter()
    for index, triple in enumerate(graph.triples):
        source, role, target = triple
        if role in (":instance", ":wiki"):
            continue
        if target not in variables:
            literal = ("literal", index)
            nodes[literal] = ("literal", target)
            target = literal
        edges[source, target, role] += 1
    return nodes, edges


def independent_coverage(observed_text, target_text):
    observed, oe = adjacency(observed_text)
    target, te = adjacency(target_text)
    if len(target) > len(observed):
        return False
    incoming, outgoing = {}, {}
    for nodes, edges, destination in ((observed, oe, outgoing), (target, te, incoming)):
        for node in nodes:
            destination[node] = (
                Counter({role: sum(n for (s, t, r), n in edges.items()
                                   if s == node and r == role)
                         for role in {r for s, t, r in edges if s == node}}),
                Counter({role: sum(n for (s, t, r), n in edges.items()
                                   if t == node and r == role)
                         for role in {r for s, t, r in edges if t == node}}))
    options = {
        node: [v for v, label in observed.items() if label == expected
               and all(outgoing[v][direction][role] >= count
                       for direction in (0, 1)
                       for role, count in incoming[node][direction].items())]
        for node, expected in target.items()}
    if any(not choices for choices in options.values()):
        return False
    mapping, used = {}, set()

    def compatible(node, value):
        mapping[node] = value
        valid = all(
            oe[mapping[s], mapping[t], role] >= count
            for (s, t, role), count in te.items() if s in mapping and t in mapping)
        del mapping[node]
        return valid

    def search():
        if len(mapping) == len(target):
            return True
        remaining = {
            node: [v for v in choices if v not in used and compatible(node, v)]
            for node, choices in options.items() if node not in mapping}
        node = min(remaining, key=lambda v: len(remaining[v]))
        for value in remaining[node]:
            mapping[node] = value
            used.add(value)
            if search():
                return True
            used.remove(value)
            del mapping[node]
        return False

    return search()


def main():
    os.umask(0o077)
    checks = {}
    freeze = json.loads((OUT / "FREEZE.json").read_text())
    parser = json.loads((OUT / "PARSER_FREEZE.json").read_text())
    correction = json.loads((OUT / "NORMALIZATION_CORRECTION.json").read_text())
    original_sha = sha(OUT / "WRITE_READ_RESULTS.json")
    initial_audit_sha = sha(OUT / "INDEPENDENT_CHECKS.json")
    result = json.loads((OUT / "WRITE_READ_RESULTS.json").read_text())
    targets = json.loads((PRIVATE / "TARGETS.json").read_text())
    inputs = json.loads((PRIVATE / "INPUTS.json").read_text())
    details = json.loads((PRIVATE / "CORRECTED_NORMALIZATION_DETAILS.json").read_text())
    bank = json.loads((ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json").read_text())
    assignments = json.loads((PRIVATE / "ASSIGNMENTS.json").read_text())
    checks["frozen_generation_sources"] = all(sha(ROOT / p) == expected
                                             for p, expected in freeze["source_sha256"].items())
    checks["frozen_parser_snapshots"] = all(sha(PRIVATE / "snapshots" / p) == expected
                                          for p, expected in parser["snapshots_sha256"].items())
    checks["parser_input_target_and_propbank_sha"] = (
        sha(PRIVATE / "INPUTS.json") == parser["input_sha256"]
        and sha(PRIVATE / "TARGETS.json") == parser["targets_sha256"]
        and sha(PRIVATE / "propbank_verbs.json") == parser["propbank_verbs_sha256"]
        and sha(PRIVATE / "normalizer_runtime/private/propbank_verbs.json") == parser["propbank_verbs_sha256"])
    checks["generation_and_parser_freeze_links"] = (
        parser["generation_freeze_sha256"] == sha(OUT / "FREEZE.json")
        and json.loads((OUT / "GENERATION_COMPLETE.json").read_text())["freeze_sha256"] == sha(OUT / "FREEZE.json")
        and all(json.loads((OUT / name).read_text())["freeze_sha256"] == sha(OUT / "PARSER_FREEZE.json")
                for name in ("CUDA_PRECHECK.json", "PARSE_COMPLETE.json")))
    checks["correction_source_and_original_sha"] = (
        correction["source_sha256"] == sha(ROOT / "scripts/diagnose_semantic_sequence_cycle.py")
        and correction["initial_results_sha256"] == original_sha
        and correction["inference_calls"] == 0 and correction["gate_affected"] is False)
    checks["unchanged_official_normalizer"] = (
        sha(ROOT / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py") == parser["normalizer_source_sha256"])
    old_alignment = ROOT / "experiments/semantic_sequence_stage1_run2/alignment_review1"
    old_freeze = json.loads((old_alignment / "FREEZE.json").read_text())
    model = spacy.load("en_core_web_sm")
    assets = json.loads((old_alignment / "NORMALIZER_ASSETS.json").read_text())
    checks["inherited_spacy_normalizer_assets"] = (
        sha(old_alignment / "NORMALIZER_ASSETS.json") == old_freeze["normalizer"]["spacy_assets_sha256"]
        and all(sha(Path(model.path) / a["path"]) == a["sha256"] for a in assets))
    checks["all_24_ids_and_template_coverage"] = (
        len(targets) == len(inputs) == len(details) == len(correction["rows"]) == 24
        and {r["input_id"] for r in targets} == {r["input_id"] for r in inputs}
        == {r["input_id"] for r in details} == {r["input_id"] for r in correction["rows"]}
        and len({r["template_id"] for r in targets}) == 24)
    normalize, _ = normalization_namespace(ROOT, PRIVATE / "normalizer_runtime")
    logger = logging.getLogger("alignment-normalizer")
    handler = BoundDiagnostics()
    previous = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    verified = []
    try:
        for target in targets:
            identifier = target["input_id"]
            record = json.loads((PRIVATE / "PARSES" / (identifier + ".json")).read_text())
            row = next(r for r in correction["rows"] if r["input_id"] == identifier)
            detail = next(r for r in details if r["input_id"] == identifier)
            planned = assignments[target["doc_id"]]["slots"][target["slot"]]
            assert target["template_id"] == planned["template_id"]
            assert target["target"] == bank[target["template_id"]]["normalized_template_amr"] == detail["target"]
            assert target["required_markers"] == planned["markers"]
            assert record["text"] == detail["text"]
            assert record["text_sha256"] == digest(record["text"])
            assert record["amr_sha256"] == digest(record["amr"])
            raw = "\n".join(line for line in record["amr"].splitlines()
                            if not line.lstrip().startswith("#"))
            handler.events.clear()
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                normalized = normalize(raw)
            assert not captured.getvalue() and not handler.events, "Bound normalization diagnostic"
            assert raw.startswith("(")
            assert digest(raw) == row["raw_sha256"]
            assert normalized == detail["normalized"]
            assert digest(normalized) == row["normalized_sha256"]
            coverage = independent_coverage(normalized, target["target"])
            assert coverage == row["whole_target_covered"]
            assert row["strict_valid"] == record["strict_valid"]
            verified.append({"input_id": identifier, "whole_target_covered": coverage,
                             "strict_valid": record["strict_valid"]})
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous
    checks["all_corrected_normalized_sha"] = True
    checks["independent_injective_coverage_agreement"] = True
    checks["corrected_aggregate"] = (
        sum(r["whole_target_covered"] for r in verified) == correction["corrected_whole_target_covered"] == 1
        and sum(r["whole_target_covered"] and r["strict_valid"] for r in verified)
        == correction["corrected_strict_whole_target_covered"] == 1)
    calls = PRIVATE / "calls"
    expected_ids = {f"d{i:02d}-{arm}-{part}" for i in (0, 1, 2)
                    for arm in ("positive", "unwatermarked")
                    for part in [*(f"s{s}" for s in range(8)), "paraphrase"]}
    for kind in ("intent", "response", "provider"):
        assert {p.name.removesuffix(f".{kind}.json") for p in calls.glob(f"*.{kind}.json")} == expected_ids
    receipts = {}
    for identifier in sorted(expected_ids):
        intent = json.loads((calls / (identifier + ".intent.json")).read_text())
        response = json.loads((calls / (identifier + ".response.json")).read_text())
        provider = json.loads((calls / (identifier + ".provider.json")).read_text())
        body = (json.dumps(intent["body"], ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        assert digest(body) == intent["request_sha256"] == response["request_sha256"]
        assert len(provider["choices"]) == 1
        choice = provider["choices"][0]
        assert response["text"] == choice["message"]["content"]
        assert response["finish_reason"] == choice["finish_reason"] == "stop" and response["complete"]
        assert intent["body"]["n"] == 1
        receipts[identifier] = response
    checks["all_54_unique_single_provider_receipts"] = True
    lengths = {arm: [] for arm in ("positive", "unwatermarked")}
    for doc_id in (0, 1, 2):
        doc = json.loads((PRIVATE / "documents" / f"d{doc_id:02d}.json").read_text())
        assert doc["split"] == "development"
        for arm, values in doc["arms"].items():
            assert len(values["slots"]) == 8
            assert values["text"] == " ".join(values["slots"])
            assert digest(values["text"]) == values["text_sha256"]
            assert digest(values["paraphrase"]) == values["paraphrase_sha256"]
            assert values["paraphrase"] == receipts[f"d{doc_id:02d}-{arm}-paraphrase"]["text"]
            for slot, text in enumerate(values["slots"]):
                assert text == receipts[f"d{doc_id:02d}-{arm}-s{slot}"]["text"].strip()
                lengths[arm].append(len(text.split()))
    checks["documents_receipts_and_lengths"] = all(
        len(v) == 24 and sum(v) / 24 == result["carrier_mean_words"][arm]
        for arm, v in lengths.items())
    annotation = json.loads((OUT / "ANNOTATION_STATUS.json").read_text())
    packets = annotation["private_packet_sha256"]
    assert len(packets) == 4
    for path, expected in packets.items():
        assert sha(PRIVATE / path) == expected
        rows = json.loads((PRIVATE / path).read_text())
        assert len(rows) == 24 and len({r["annotation_id"] for r in rows}) == 24
        fields = (("source_proposition", "source_event_anchor_span", "paraphrase_event_spans",
                   "proposition_preserved", "reviewer_id") if "stage1" in path
                  else ("realized", "evidence", "reviewer_id"))
        assert all(r[f] is None for r in rows for f in fields)
    checks["four_blank_human_packets_and_zero_labels"] = (
        annotation["labels_completed"] == 0 and annotation["reviewers_required"] == 2
        and annotation["status"] == "PREPARED_NOT_ANNOTATED")
    checks["readout_unreleased"] = (
        {p.stem for p in (PRIVATE / "documents").glob("*.json")} == {"d00", "d01", "d02"})
    checks["old_audit_and_results_preserved"] = (
        sha(OUT / "INDEPENDENT_CHECKS.json") == initial_audit_sha
        and sha(OUT / "WRITE_READ_RESULTS.json") == original_sha
        and result["g1e_successes"] == 16 and result["minimum_required"] == 20
        and result["status"] == "NO_GO_WRITE_READ_PILOT")
    checks["no_downstream_truth_claimed"] = all(
        result[k] == "NOT_EVALUATED" for k in ("formal_human_G1", "symbol_G2", "blind_G3", "order_G4"))
    assert all(checks.values()), "Supplementary integrity audit failed"
    historical = {
        str(p.relative_to(ROOT)): sha(p) for p in [
            old_alignment / "REVIEW_REPORT.md", old_alignment / "ALIGNMENT_SUMMARY.json",
            ROOT / "experiments/semantic_sequence_stage1_run2/STAGE1A_REPORT.md",
            ROOT / "experiments/semantic_sequence_stage1_run2/SEQUENCE_DIAGNOSTIC_REPORT.md"]}
    frozen = {str(p.relative_to(OUT)): sha(p) for p in OUT.glob("*.json")
              if p.name not in ("SUPPLEMENTARY_AUDIT.json", "ENGINEERING_CHECKS.json",
                                "STATUS.json", "CLOSEOUT.json", "PROCESS_CLOSURE.json",
                                "SHA256SUMS.json", "PACKAGE_VALIDATION.json")}
    atomic(OUT / "SUPPLEMENTARY_AUDIT.json", {
        "at": datetime.now(timezone.utc).isoformat(), "passed": True, "checks": checks,
        "coverage_checker": "PENMAN adjacency + injective backtracking; no NetworkX matcher",
        "coverage_successes": 1, "strict_coverage_successes": 1, "denominator": 24,
        "rows": verified, "checker_sha256": sha(Path(__file__)),
        "saved_evidence_sha256": frozen, "historical_reference_sha256": historical,
        "scope": "Separate implementation on the same saved pilot; not independent data, human truth or detection performance",
        "new_generation_calls": 0, "neural_inference_calls": 0, "detector_calls": 0,
        "normalizer_asset_provenance": "Cycle1 package versions + current SHA agreement with inherited pre-cycle alignment asset manifest"})
    command = [sys.executable, str(ROOT / "scripts/test_semantic_sequence_cycle.py")]
    completed = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    assert completed.returncode == 0, "Engineering regression failed"
    assert "Ran 7 tests" in completed.stderr and completed.stderr.rstrip().endswith("OK")
    atomic(OUT / "ENGINEERING_CHECKS.json", {
        "at": datetime.now(timezone.utc).isoformat(), "passed": True, "test_methods": 7,
        "exhaustive_small_edit_cases": 560, "equal_frequency_candidates": 2520,
        "exact_repeat_hard_soft": True, "unordered_all_equal": True,
        "occurrence_dedup_and_window_identity": True, "local_erasure": True,
        "budget_abstain_not_prefix": True, "prompt_contract": True,
        "tests_sha256": sha(ROOT / "scripts/test_semantic_sequence_cycle.py"),
        "core_sha256": sha(ROOT / "src/core/semantic_sequence.py"),
        "runner_python": sys.version, "stdout_sha256": digest(completed.stdout),
        "stderr_sha256": digest(completed.stderr),
        "scope": "Engineering fixtures only; no measured AMR/Embedding order gain"})
    print(json.dumps({"passed": True, "checks": len(checks), "coverage": "1/24",
                      "g1e": "16/24", "engineering_tests": 7}))


if __name__ == "__main__":
    main()
