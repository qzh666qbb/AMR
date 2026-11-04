"""Blind review workflow and evidence-derived pilot/main cohort gates."""
from __future__ import annotations

from pathlib import Path

from swan_final1_common import digest, freeze, load, sha256
from swan_final1_generation import verify_generation
from swan_final1_quality import export_packets, finalize, verify_decision
from swan_final1_sources import COHORTS, identity
from swan_final1_transform import (
    FrozenSyntax, LEAK_RE, sentence_spans, transform, unit_texts, word_tokens,
)


def sources_for(folder, phase):
    if phase not in dict(COHORTS):
        raise ValueError("Unknown phase")
    if phase != "pilot":
        require_pilot(folder)
    return load(Path(folder) / "sealed" / f"{phase}_sources.json")


def documents_for(folder, phase):
    folder = Path(folder)
    sources = sources_for(folder, phase)
    if phase in ("pilot", "main"):
        return verify_generation(folder, phase, sources)
    records = load(folder / "human" / f"{phase}.json")
    if len(records) != len(sources) or any(
            r["source_sha256"] != digest(s) for r, s in zip(records, sources)):
        raise ValueError("Human window/source identities differ")
    historical = human_history(folder, phase)
    for row, source in zip(records, sources):
        evidence = row["syntax_evidence"]
        text, termination = human_window(source["text"], lambda s: evidence[s])
        expected = human_record(folder, phase, source, text, termination, historical, evidence)
        if row != expected:
            raise ValueError("Human identity, structure or window cannot be reproduced")
        if text:
            historical.add(identity(text))
    return records


def quality_folder(folder, phase, kind):
    return Path(folder) / "quality" / phase / kind


def baseline_records(documents):
    return [{"allocation_id": r["allocation_id"], "source_excerpt": r["source_excerpt"],
             "text": r["text"]} for r in documents if r["structural_pass"]]


def verify_inputs(folder, records):
    """Rebind review packets to actual frozen texts, not hand-entered decisions."""
    index = load(Path(folder) / "private_index.json")
    expected = {(r["allocation_id"], r.get("condition_id")): r for r in records}
    for slot in ("A", "B"):
        mappings = [r for r in index["items"] if r["reviewer_slot"] == slot]
        if len(mappings) != len(expected) or {
                (r["allocation_id"], r["condition_id"]) for r in mappings} != set(expected):
            raise ValueError("Quality index does not cover the expected cohort")
        rows = []
        for mapping in mappings:
            source = expected[(mapping["allocation_id"], mapping["condition_id"])]
            if mapping["input_sha256"] != digest(source):
                raise ValueError("Reviewed text differs from frozen source/transform")
            key = [source["allocation_id"], source.get("condition_id")]
            blind_id = digest(["final1-quality", index["kind"], key])[:24]
            if mapping["blind_id"] != blind_id:
                raise ValueError("Blind identity changed")
            row = {"blind_id": blind_id}
            if index["kind"] == "baseline":
                row.update(source_excerpt=source["source_excerpt"], text=source["text"])
            else:
                swap = int(digest(["final1-side", slot, key])[-1], 16) % 2 == 1
                if mapping["original_on_right"] != swap:
                    raise ValueError("Review left/right assignment changed")
                texts = [source["original_text"], source["transformed_text"]]
                if swap:
                    texts.reverse()
                row.update(left=texts[0], right=texts[1])
            rows.append(row)
        rows.sort(key=lambda r: digest(["final1-order", slot, r["blind_id"]]))
        if load(Path(folder) / f"packet_{slot}.json")["items"] != rows:
            raise ValueError("Review packet no longer matches the expected blind text export")


def decision_for(folder, phase, kind, records):
    path = quality_folder(folder, phase, kind)
    verify_inputs(path, records)
    return verify_decision(path, sha256(Path(folder) / "manifest.json"))


def baseline_cohort(folder, phase, documents=None):
    documents = documents if documents is not None else documents_for(folder, phase)
    decision = decision_for(folder, phase, "baseline", baseline_records(documents))
    passed = {r["allocation_id"] for r in decision["decisions"] if r["quality_pass"]}
    return [r for r in documents if r["structural_pass"] and r["allocation_id"] in passed]


def transform_path(folder, phase):
    return Path(folder) / "transforms" / f"{phase}.json"


def create_transforms(folder, phase):
    cohort = baseline_cohort(folder, phase)
    path = transform_path(folder, phase)
    if path.exists():
        return verified_transforms(folder, phase, cohort)
    syntax = FrozenSyntax()
    result = []
    for r in cohort:
        evidence = {}
        def recorded(text):
            evidence[text] = syntax(text)
            return evidence[text]
        result.append({"allocation_id": r["allocation_id"], "baseline_sha256": digest(r),
                       **transform(r["text"], recorded), "syntax_evidence": evidence})
    freeze(path, result)
    return result


def verified_transforms(folder, phase, cohort):
    """No reparsing: offsets, exact edit records and restoration are verified."""
    from swan_final1_transform import apply_edits, punctuation, restore_original, text_hash
    rows = load(transform_path(folder, phase))
    if len(rows) != len(cohort):
        raise ValueError("Transformation cohort differs")
    for row, base in zip(rows, cohort):
        evidence = row["syntax_evidence"]
        expected = {"allocation_id": base["allocation_id"], "baseline_sha256": digest(base),
                    **transform(base["text"], lambda s: evidence[s]), "syntax_evidence": evidence}
        if row != expected:
            raise ValueError("Candidate enumeration/first selection cannot be reproduced")
        if row["allocation_id"] != base["allocation_id"] or row["baseline_sha256"] != digest(base) \
                or row["input_sha256"] != text_hash(base["text"]) or row["detector_feedback"]:
            raise ValueError("Transformation provenance differs")
        c0 = row["conditions"]["C0"]
        if c0["text"] != base["text"] or c0["segmentation_spans"] != sentence_spans(base["text"]):
            raise ValueError("C0 offsets differ")
        for condition in ("C1", "C2", "C3"):
            if condition not in row["conditions"]:
                if condition not in row["failures"]:
                    raise ValueError("Transformation failure unaccounted")
                continue
            changed = row["conditions"][condition]
            edits = changed["punctuation_edits"]
            kinds = sorted(e["kind"] for e in edits)
            if kinds != (["merge"] if condition == "C1" else ["merge", "split"]):
                raise ValueError("Wrong intervention")
            if apply_edits(base["text"], edits) != changed["text"] \
                    or word_tokens(changed["text"]) != word_tokens(base["text"]) \
                    or sentence_spans(changed["text"]) != changed["segmentation_spans"] \
                    or changed["num_detection_units"] != len(changed["segmentation_spans"]) \
                    or changed["num_detection_units"] != c0["num_detection_units"] - (condition == "C1"):
                raise ValueError("Boundary/token invariants differ")
        if "C3" in row["conditions"] and "C2" in row["conditions"]:
            if punctuation(row["conditions"]["C3"]["text"]) != punctuation(row["conditions"]["C2"]["text"]):
                raise ValueError("Punctuation control differs")
        if "C4" in row["conditions"]:
            restored = restore_original(row["conditions"]["C2"], c0["segmentation_spans"],
                                        c0["document_sha256"])
            if restored != row["conditions"]["C4"] or restored["unit_texts"] != c0["unit_texts"]:
                raise ValueError("Oracle restoration differs")
    return rows


def pair_records(transforms):
    return [{"allocation_id": r["allocation_id"], "condition_id": condition,
             "original_text": r["conditions"]["C0"]["text"],
             "transformed_text": r["conditions"][condition]["text"]}
            for r in transforms for condition in ("C1", "C2", "C3") if condition in r["conditions"]]


def quality_action(folder, phase, kind, finalize_labels=False):
    documents = documents_for(folder, phase)
    if kind == "baseline":
        records = baseline_records(documents)
    else:
        cohort = baseline_cohort(folder, phase, documents)
        records = pair_records(verified_transforms(folder, phase, cohort))
    path = quality_folder(folder, phase, kind)
    contract = sha256(Path(folder) / "manifest.json")
    if finalize_labels:
        verify_inputs(path, records)
        result = finalize(path, contract, folder)
        return {"quality_pass_count": result["quality_pass_count"], "items": len(records)}
    if (path / "private_index.json").exists():
        verify_inputs(path, records)
        return {"items_per_reviewer": len(records), "state": "already_exported"}
    return export_packets(records, path, kind, contract)


def cohort_gates(folder, phase):
    if phase not in ("pilot", "main"):
        raise ValueError("Only positive phases have feasibility gates")
    docs = documents_for(folder, phase)
    structural = sum(r["structural_pass"] for r in docs)
    minimum = 6 if phase == "pilot" else 320
    if structural < minimum:
        return {
            "counts": {"allocated": len(docs), "structure_pass": structural,
                       "baseline_quality_pass": None, "C2_constructed": None,
                       "C2_quality_pass": None, "common_quality_pairs": None},
            "gates": {"structural_upper_bound": False}, "passed": False,
            "reason": "Too few structurally eligible documents; quality cannot repair this gate",
            "baseline_ids": [], "common_ids": [], "branch": "C",
            "unperformed_quality_is_pending_not_failed": True,
            "manifest_sha256": sha256(Path(folder) / "manifest.json"),
            "evidence_hashes": {str(p.relative_to(folder)): sha256(p)
                               for p in sorted((Path(folder) / "documents" / phase).glob("*.json"))},
        }
    cohort = baseline_cohort(folder, phase, docs)
    transforms = verified_transforms(folder, phase, cohort)
    pairs = decision_for(folder, phase, "pair", pair_records(transforms))
    passing = {(r["allocation_id"], r["condition_id"]) for r in pairs["decisions"]
               if r["quality_pass"]}
    constructed = sum("C2" in r["conditions"] for r in transforms)
    quality_c2 = sum((r["allocation_id"], "C2") in passing for r in transforms)
    common = [r["allocation_id"] for r in transforms if "C4" in r["conditions"]
              and all((r["allocation_id"], c) in passing for c in ("C1", "C2", "C3"))]
    counts = {"allocated": len(docs), "structure_pass": sum(r["structural_pass"] for r in docs),
              "baseline_quality_pass": len(cohort), "C2_constructed": constructed,
              "C2_quality_pass": quality_c2, "common_quality_pairs": len(common)}
    if phase == "pilot":
        gates = {"pilot": len(docs) == 8 and counts["structure_pass"] >= 6
                 and len(cohort) >= 6 and quality_c2 >= 1}
    else:
        gates = {"A": len(docs) == 500 and len(cohort) >= 320,
                 "C": bool(cohort) and constructed * 5 >= len(cohort) * 4,
                 "D": constructed > 0 and quality_c2 * 5 >= constructed * 4 and len(common) >= 200}
    evidence = {
        str(p.relative_to(folder)): sha256(p) for p in [
            transform_path(folder, phase),
            quality_folder(folder, phase, "baseline") / "decision.json",
            quality_folder(folder, phase, "pair") / "decision.json",
            *sorted((Path(folder) / "documents" / phase).glob("*.json")),
        ]
    }
    return {"counts": counts, "gates": gates, "passed": all(gates.values()),
            "baseline_ids": [r["allocation_id"] for r in cohort], "common_ids": common,
            "manifest_sha256": sha256(Path(folder) / "manifest.json"),
            "evidence_hashes": evidence}


def require_pilot(folder):
    # Recompute labels and saved-generation provenance; never trust a gate=true file.
    result = cohort_gates(Path(folder), "pilot")
    if not result["passed"]:
        raise PermissionError("Pilot failed; main/human sealed content remains unavailable")
    return result


def human_window(text, syntax):
    spans = sentence_spans(text)
    first = next((i for i, (a, b) in enumerate(spans) if syntax(text[a:b])["complete"]), None)
    if first is None:
        return "", "no_complete_initial_sentence"
    selected = []
    for span in spans[first:first + 24]:
        selected.append(span)
        window = text[selected[0][0]:span[1]]
        if len(word_tokens(window)) >= 100 and len(selected) >= 8:
            return window, "first_100_words_and_8_units"
    return text[selected[0][0]:selected[-1][1]], "source_exhausted_or_24_units"


def prepare_human(folder, phase):
    if phase not in ("calibration", "test"):
        raise ValueError("Expected a human phase")
    folder = Path(folder)
    sources = sources_for(folder, phase)
    # Do not build control windows before positive generation and baseline labels finish.
    baseline_cohort(folder, "main")
    path = folder / "human" / f"{phase}.json"
    if path.exists():
        return {"allocated": len(documents_for(folder, phase)), "state": "already_prepared"}
    historical = human_history(folder, phase)
    syntax, rows = FrozenSyntax(), []
    for source in sources:
        evidence = {}
        def recorded(text):
            evidence[text] = syntax(text)
            return evidence[text]
        text, termination = human_window(source["text"], recorded)
        rows.append(human_record(folder, phase, source, text, termination, historical, evidence))
        if text:
            historical.add(identity(text))
    freeze(path, rows)
    return {"allocated": len(rows), "structural_pass": sum(r["structural_pass"] for r in rows)}


def human_history(folder, phase):
    folder = Path(folder)
    historical = set(load(folder / "prior_identity_snapshot.json")["normalized_text_sha256"])
    for p in sorted((folder / "documents").glob("*/*.json")):
        row = load(p)
        if row["text"]:
            historical.add(identity(row["text"]))
    if phase == "test":
        prior = documents_for(folder, "calibration")
        historical.update(identity(r["text"]) for r in prior if r["text"])
    return historical


def human_record(folder, phase, source, text, termination, historical, evidence):
    spans = sentence_spans(text)
    words, units = len(word_tokens(text)), len(spans)
    normalized = identity(text)
    independent = normalized not in historical
    return {
        "allocation_id": source["allocation_id"], "phase": phase, "ordinal": source["ordinal"],
        "source_sha256": digest(source), "source_excerpt": source["generation_prompt"],
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "domain": source["domain"], "text": text, "segmentation_spans": spans,
        "unit_texts": unit_texts(text, spans), "word_count": words, "num_detection_units": units,
        "termination": termination, "normalized_text_sha256": normalized,
        "syntax_evidence": evidence, "identity_pass": independent,
        "structural_pass": bool(100 <= words <= 220 and 8 <= units <= 24
                                and independent and not LEAK_RE.search(text)),
    }
