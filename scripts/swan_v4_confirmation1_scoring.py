"""One fixed v4 control, complete denominators, and replayable calibration."""
from collections import Counter
from dataclasses import asdict
import gzip
import json
import math
import os


def core():
    from src.core import elastic_v4_path0_development
    return elastic_v4_path0_development


VARIANTS = ("baseline", "semicolon_all", "semicolon_pairs", "comma_all",
            "newline_all", "space_all")
ALPHA = .01


def build_control(args):
    dev = core()
    base = dev.build_detector(args)
    extractor = dev.InvalidTolerantAMRExtractor(
        base.scorer.extractor.backend, base.scorer.extractor.cache, args.request_batch_size)
    builder = dev.ElasticBoundaryProposerV3(
        args.max_boundary_hops, args.max_span_words,
        args.capitalization_gap_words, args.long_gap_stride_words)
    detector = dev.SwanBankLatticeDetectorV4(
        builder, dev.AllBankScorer(extractor, args.bank), args.score_center, args.green_threshold)
    detector.optimizer = dev.CheckedHopOptimizer(args.score_center, args.green_threshold, .18, .18)
    recovery = dev.attach_recovery_audit(detector, args.output_root / "sealed/parser_recovery.jsonl")
    return detector, recovery


def score_document(args, detector, recovery, source, variant, text, binding):
    dev = core()
    lattice = detector.builder.build_lattice(text)
    spans = sorted({dev.normalize_text(edge.span_text) for edge in lattice.edges})
    detector.scorer.prepare(spans)
    records = detector.scorer._records
    evidence = dev.classify_edge_evidence(lattice, recovery.records, records)
    details, matches = {}, {}
    for span in spans:
        record = records[span]
        digest = dev.text_sha256(span)
        recovery.assert_record(record)
        detail = {"record": asdict(record), "amr_sha256": dev.sha(record.amr.encode()),
                  "recovery": recovery.records[digest]}
        if not record.graph_valid or not record.roundtrip_valid:
            matches[span] = None
            detail["control"] = {"all_bank_matches": [], "best": None}
        else:
            bank_matches, summary = detector.scorer.audited_matches(record.amr)
            matches[span] = max(bank_matches, key=lambda item: item.score)
            detail["control"] = {**summary, "all_bank_matches": [asdict(x) for x in bank_matches]}
        details[digest] = detail
    result = detector.optimizer.optimize(
        lattice, tuple(matches[dev.normalize_text(e.span_text)] for e in lattice.edges),
        tuple(item.valid for item in evidence))
    valid = {edge for edge, item in zip(lattice.edges, evidence) if item.valid}
    if any(unit.edge not in valid for unit in result.units):
        raise ValueError("Selected invalid candidate edge")
    control = {
        "status": result.status, "statistic": result.statistic, "raw_utility": result.raw_utility,
        "unit_count": result.unit_count, "green_count": result.green_count,
        "invalid_edge_count": result.invalid_edge_count, "selected_invalid_edges": 0,
        "abstain_reason": result.abstain_reason,
        "units": [{"start": u.edge.start, "end": u.edge.end,
                   "span_sha256": dev.text_sha256(u.edge.span_text),
                   "boundary_hops": u.edge.boundary_hops, "utility": u.utility,
                   "match": asdict(u.match)} for u in result.units],
    }
    row_id = source["allocation_id"] + "__" + variant
    path = args.output_root / "sealed/candidates" / (row_id + ".json.gz")
    payload = {
        "binding": binding, "text_sha256": dev.sha(text.encode()),
        "boundaries": [asdict(b) for b in lattice.boundaries],
        "edges": [{"start": e.start, "end": e.end, "boundary_hops": e.boundary_hops,
                   "span_sha256": dev.text_sha256(e.span_text), "evidence": asdict(v)}
                  for e, v in zip(lattice.edges, evidence)],
        "spans": details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(gzip.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    detector.scorer.clear_prepared()
    return {
        "binding": binding, "stage": source["stage"], "row_id": row_id,
        "allocation_id": source["allocation_id"], "variant": variant,
        "word_count": len(text.split()), "length_bin": source["length_bin"],
        "arms": {"control": control}, "candidate_edges": len(lattice.edges), "unique_spans": len(spans),
        "evidence_path": str(path.relative_to(args.output_root)), "evidence_sha256": dev.file_sha(path),
    }


def miss_row(source, variant, binding, reason):
    return {
        "binding": binding, "allocation_id": source["allocation_id"],
        "row_id": source["allocation_id"] + "__" + variant, "variant": variant,
        "stage": "positive_source", "word_count": 0, "length_bin": source["length_bin"],
        "generation_failure": reason,
        "arms": {"control": {"status": "abstain", "statistic": None,
                             "selected_invalid_edges": 0, "abstain_reason": reason}},
    }


def fit(args, detector, rows, sources, runtime_sha):
    """Reconstruct calibration from sealed rows; compare every reusable value."""
    dev = core()
    if len(rows) != 1000 or len(sources) != 1000:
        raise ValueError("Expected all 1000 calibration sources")
    samples = []
    for row, source in zip(rows, sources):
        if (row["allocation_id"] != source["allocation_id"] or row["variant"] != "baseline"
                or row["stage"] != "calibration"):
            raise ValueError("Calibration ordering or source identity differs")
        result = row["arms"]["control"]
        statistic = dev.ABSTAIN_STATISTIC if result["status"] == "abstain" else result["statistic"]
        if not math.isfinite(statistic):
            raise ValueError("Non-finite calibration statistic")
        samples.append((source["text"], dev.DetectionResult(
            statistic, statistic, result["unit_count"], result["green_count"], ())))
    expected = dev.EmpiricalNullCalibrator.fit(
        samples, detector.fingerprint, detector.scorer.fingerprint,
        detector.scorer.bank_sha256, detector.scorer.parser_fingerprint)
    expected.artifact.update(
        binding_runtime_sha256=runtime_sha,
        binding_input_sha256=dev.file_sha(args.output_root / "INPUT_MANIFEST.json"),
        binding_rows_sha256=dev.sha(json.dumps(rows, sort_keys=True, allow_nan=False).encode()))
    path = args.output_root / "results/CALIBRATION_control.json"
    if path.exists():
        saved = dev.EmpiricalNullCalibrator.load(path)
        ignore = {"created_at"}
        if ({k: v for k, v in saved.artifact.items() if k not in ignore}
                != {k: v for k, v in expected.artifact.items() if k not in ignore}):
            raise ValueError("Calibration values or identity changed")
        return saved
    expected.save(path)
    return expected


def hit_sets(rows, calibrator, expected_ids):
    ids = set(expected_ids)
    if len(ids) != len(expected_ids):
        raise ValueError("Duplicate expected source")
    expected = {(i, v) for i in ids for v in VARIANTS}
    observed = [(r["allocation_id"], r["variant"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated condition denominator")
    hits, abstains = {v: set() for v in VARIANTS}, Counter()
    for row in rows:
        result = row["arms"]["control"]
        if result["selected_invalid_edges"]:
            raise ValueError("Invalid selected edge in summary")
        if result["status"] == "abstain":
            abstains[row["variant"]] += 1
        else:
            if result["statistic"] is None or not math.isfinite(result["statistic"]):
                raise ValueError("Non-finite scored row")
            if calibrator.calibrate(result["statistic"], row["word_count"], ALPHA).is_watermarked:
                hits[row["variant"]].add(row["allocation_id"])
    return hits, {v: abstains[v] for v in VARIANTS}


def summarize(null_rows, positive_rows, calibrator, null_ids, positive_ids):
    if len(null_ids) != 1000 or len(positive_ids) != 250:
        raise ValueError("Prespecified confirmation denominator differs")
    rate = core().rate
    null_hits, null_abstains = hit_sets(null_rows, calibrator, null_ids)
    positive_hits, positive_abstains = hit_sets(positive_rows, calibrator, positive_ids)
    union = set.union(*(null_hits[v] for v in VARIANTS[1:]))
    joint = set.intersection(*(positive_hits[v] for v in VARIANTS))
    baseline = positive_hits["baseline"]
    null = {
        "conditions": {v: rate(len(null_hits[v]), 1000) for v in VARIANTS},
        "any_stress_false_positive": rate(len(union), 1000), "abstain": null_abstains,
    }
    positive = {
        "full_denominator_TPR": {v: rate(len(positive_hits[v]), 250) for v in VARIANTS},
        "conditional_survival": {v: rate(len(baseline & positive_hits[v]), len(baseline))
                                 for v in VARIANTS[1:]},
        "all_five_survival": rate(len(joint), len(baseline)),
        "all_five_full_denominator": rate(len(joint), 250),
        "abstain_full_denominator": positive_abstains,
    }
    gates = {
        "clean_null_upper95_le_1pct": null["conditions"]["baseline"]["upper95_one_sided"] <= .01,
        "any_stress_null_upper95_le_1pct": null["any_stress_false_positive"]["upper95_one_sided"] <= .01,
        "baseline_TPR_ge_80pct": len(baseline) / 250 >= .8,
        "joint_survival_ge_80pct": bool(baseline) and len(joint) / len(baseline) >= .8,
        "selected_invalid_edges_zero": True,
    }
    return {"alpha": ALPHA, "null": null, "positive": positive, "computational_gates": gates,
            "computational_gates_pass": all(gates.values())}
