"""Freeze blind lemmas, training-only fact templates, and all three-view scores."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import fcntl
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from semantic_supervision_completion1_paths import (
    OUT, PRIVATE, COMPLETION, bind, verify as verify_parent,
)
from prepare_semantic_supervision2 import atomic, digest, load, sha, context_candidates
from run_semantic_supervision_graph_completion1 import workload
from semantic_fact_raw_amr import audit_raw
from src.core.semantic_fact_core_matcher import representation, score_view
from src.core.semantic_graph_fact_reader_v2 import read_graph_facts

GRAPH = OUT / "graph_completion1"
CODE = (
    "scripts/semantic_supervision_completion1_paths.py",
    "scripts/run_semantic_supervision_features_completion1.py",
    "scripts/test_semantic_supervision_features_completion1.py",
    "scripts/semantic_fact_raw_amr.py",
    "scripts/run_semantic_supervision_graph_completion1.py",
    "scripts/run_multiscale_transfer.py",
    "src/core/semantic_fact_core_matcher.py",
    "src/core/semantic_graph_fact_reader.py",
    "src/core/semantic_graph_fact_reader_v2.py",
    "src/core/semantic_fact_carrier.py",
)


def immutable(path, value):
    if path.exists():
        if load(path) != value:
            raise RuntimeError("Immutable artifact changed: " + path.name)
    else:
        atomic(path, value)


def lexical_rows(request, nlp):
    """Replay exact original extraction; no target-bearing document file."""
    rows = {}
    for doc in request["documents"]:
        replay = context_candidates(doc["text"], nlp)
        if replay["observations"] != doc["candidates"] or replay["status"] != doc["candidate_status"]:
            raise RuntimeError("Frozen lexical candidates do not replay")
        token_map = {(t.idx, t.idx + len(t.text)): t for t in nlp(doc["text"])}
        candidates = []
        for candidate in doc["candidates"]:
            token = token_map[tuple(candidate["anchor_span"])]
            if token.pos_ != "VERB" or not token.lemma_:
                raise RuntimeError("Frozen anchor has no observable verb lemma")
            lemma = token.lemma_.lower()
            contexts = []
            for context in candidate["contexts"]:
                # Count all surface tokens, not only verbs. A repeated lexical
                # anchor of any POS makes graph-node anchoring ambiguous.
                parsed = nlp(context["text"])
                contexts.append([t.lemma_.lower() for t in parsed])
            candidates.append({"anchor_lemma": lemma, "context_lemmas": contexts})
        rows[doc["key"]] = candidates
    return rows


def model_identity(nlp):
    return {
        "spacy": importlib.metadata.version("spacy"),
        "model": nlp.meta["version"],
        "files": {str(p.relative_to(nlp.path)): sha(p) for p in sorted(nlp.path.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
    }


def prepare():
    import spacy
    parent = verify_parent()
    if (COMPLETION / "FREEZE.json").exists():
        return verify()
    nlp = spacy.load("en_core_web_sm")
    model = model_identity(nlp)
    if any(model[k] != parent["candidate_extractor"][k] for k in ("spacy", "model")):
        raise RuntimeError("Original lexical model/version required")
    request = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")
    workload(request)
    lemmas = lexical_rows(request, nlp)
    immutable(COMPLETION / "private/BLIND_LEMMAS.json", lemmas)
    immutable(COMPLETION / "LEXICAL_MODEL.json", model)
    immutable(COMPLETION / "FREEZE.json", {
        "version": "semantic-supervision-features-completion/v1",
        "parent_freeze_sha256": sha(OUT / "FREEZE.json"),
        "graph_freeze_sha256": sha(GRAPH / "FREEZE.json"),
        "request_sha256": sha(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json"),
        "lemma_sha256": sha(COMPLETION / "private/BLIND_LEMMAS.json"),
        "lexical_model_sha256": sha(COMPLETION / "LEXICAL_MODEL.json"),
        "source_sha256": {k: sha(ROOT / k) for k in CODE},
        "packages": {p: importlib.metadata.version(p) for p in ("penman", "numpy", "torch")},
        "template_rule": {
            "roles_read": ["train"], "conditions": ["original"],
            "candidate_kind": "human_confirmed_canonical_carrier",
            "class_count": 4, "per_class": 6, "per_scene": 2,
            "scene_class_capacity": 1, "graph_context_index": 2,
            "eligibility": "complete_rule_core_and_reader_agrees_with_actual_human_symbol",
            "tie_break": "SHA256(key,candidate_index); deterministic max-flow",
            "no_evaluation_score_selection": True,
        },
        "matcher": {"max_states": 200000, "views": ["full", "full", "core"],
                    "partial_columns_released": False},
        "scope": "development; previously exposed test; not independent confirmation",
    })
    return verify()


def verify():
    verify_parent()
    f = load(COMPLETION / "FREEZE.json")
    checks = {
        OUT / "FREEZE.json": f["parent_freeze_sha256"],
        GRAPH / "FREEZE.json": f["graph_freeze_sha256"],
        PRIVATE / "FEATURE_REQUEST_NO_TARGET.json": f["request_sha256"],
        COMPLETION / "private/BLIND_LEMMAS.json": f["lemma_sha256"],
        COMPLETION / "LEXICAL_MODEL.json": f["lexical_model_sha256"],
        **{ROOT / k: v for k, v in f["source_sha256"].items()},
    }
    for path, expected in checks.items():
        if sha(path) != expected:
            raise RuntimeError("Feature completion input/code drift: " + path.name)
    for name, expected in f["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError("Feature dependency version drift: " + name)
    return f


def graph_seal():
    seal = load(GRAPH / "RAW_GRAPH_SEAL.json")
    if (seal["freeze_sha256"] != sha(GRAPH / "FREEZE.json")
            or seal["target_access"] or seal["human_label_access"]):
        raise RuntimeError("Raw graph provenance differs")
    request = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")
    expected = {"private/raw_graphs/" + r["case_id"] + ".json"
                for r in workload(request)["unique_texts"]}
    if set(seal["files"]) != expected:
        raise RuntimeError("Full unique graph denominator required")
    # Hash all outputs for integrity. Semantic reads during template selection
    # are restricted to training originals below.
    for name, expected_sha in seal["files"].items():
        if sha(GRAPH / name) != expected_sha:
            raise RuntimeError("Raw graph output drift")
    return seal


def observation(doc, candidate_index, view_index, lex):
    context = doc["candidates"][candidate_index]["contexts"][view_index]
    import hashlib
    text_sha = hashlib.sha256(context["text"].encode()).hexdigest()
    path = GRAPH / "private/raw_graphs" / (text_sha + ".json")
    record = load(path)
    audit = audit_raw(record, {"case_id": text_sha, "input_text_sha256": text_sha},
                      sha(GRAPH / "FREEZE.json"))
    if not audit["structurally_eligible"]:
        rep = {"status": "invalid", "reason": "raw_graph_ineligible",
               "exclusion_reasons": audit["exclusion_reasons"]}
    else:
        rep = representation(record["amr"], doc["contract"], lex["anchor_lemma"],
                             lex["context_lemmas"][view_index])
    return rep, record


def choose_templates(options, train_scenes):
    """Exact feasible 2-per-scene, 6-per-class assignment, no score ranking."""
    scenes = sorted(train_scenes)
    if len(scenes) != 12 or len(set(scenes)) != 12:
        raise ValueError("Exactly twelve frozen training scenes required")
    by_cell = {}
    for row in options:
        if row["scene_id"] not in scenes or row["symbol"] not in "ABCD":
            raise ValueError("Nontraining scene or unknown actual symbol")
        by_cell.setdefault((row["scene_id"], row["symbol"]), []).append(row)
    capacities, adjacent = {}, {}

    def edge(a, b, capacity):
        capacities[a, b], capacities[b, a] = capacity, 0
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)

    for scene in scenes:
        edge("source", ("scene", scene), 2)
        for symbol in "ABCD":
            if (scene, symbol) in by_cell:
                edge(("scene", scene), ("symbol", symbol), 1)
    for symbol in "ABCD":
        edge(("symbol", symbol), "sink", 6)
    flow = 0
    while True:
        parents, queue = {"source": None}, deque(["source"])
        while queue and "sink" not in parents:
            a = queue.popleft()
            for b in adjacent[a]:
                if capacities[a, b] and b not in parents:
                    parents[b] = a
                    queue.append(b)
        if "sink" not in parents:
            break
        node = "sink"
        while parents[node] is not None:
            prev = parents[node]
            capacities[prev, node] -= 1
            capacities[node, prev] += 1
            node = prev
        flow += 1
    if flow != 24:
        raise RuntimeError("Training human/graph coverage cannot supply balanced 24 templates")
    chosen = []
    for (scene, symbol), rows in sorted(by_cell.items()):
        if capacities[("symbol", symbol), ("scene", scene)]:
            chosen.append(min(rows, key=lambda r: digest([r["key"], r["candidate_index"]])))
    return sorted(chosen, key=lambda r: (r["symbol"], r["scene_id"], r["key"], r["candidate_index"]))


def templates():
    verify()
    _, runner = bind()
    graph_seal()
    bank_path = PRIVATE / "TEMPLATE_BANK.json"
    seal_path = COMPLETION / "TEMPLATE_SEAL.json"
    if seal_path.exists():
        return verify_templates()
    if (OUT / "FEATURE_SEAL.json").exists():
        raise RuntimeError("Existing feature release requires a new successor")
    labels = runner.role_labels("train")  # Never open aggregate or held-out gold.
    docs = {d["key"]: d for d in load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")["documents"]}
    lemmas = load(COMPLETION / "private/BLIND_LEMMAS.json")
    membership = load(OUT / "MEMBERSHIP.json")
    options, reasons = [], Counter()
    for key in sorted(labels):
        label, doc = labels[key], docs[key]
        if label["condition"] != "original":
            continue
        if label["scene_id"] != membership[key] or len(label["candidate_labels"]) != len(doc["candidates"]):
            raise RuntimeError("Training membership/candidate mismatch")
        for ci, candidate in enumerate(label["candidate_labels"]):
            if candidate["index"] != ci:
                raise RuntimeError("Training candidate order mismatch")
            if candidate["kind"] != "carrier":
                continue
            rep, raw = observation(doc, ci, 2, lemmas[key][ci])
            if rep["status"] != "complete":
                reasons[rep["reason"]] += 1
                continue
            actual = candidate["symbol"]  # Never read planned_symbol.
            if read_graph_facts(raw["amr"], doc["contract"])["prediction"] != actual:
                reasons["template_graph_reader_disagrees_with_actual_training_fact"] += 1
                continue
            options.append({"key": key, "candidate_index": ci, "scene_id": label["scene_id"],
                            "symbol": actual, "representation": rep,
                            "raw_graph_sha256": raw["amr_sha256"]})
    try:
        selected = choose_templates(options, load(OUT / "SPLITS.json")["groups"]["train"])
    except RuntimeError:
        atomic(COMPLETION / "TEMPLATE_READINESS.json", {
            "status": "PENDING_BALANCED_TRAINING_FACT_COVERAGE",
            "eligible_candidates": len(options), "excluded": dict(reasons),
            "heldout_labels_read": False, "training_runs": 0})
        raise
    bank = {
        "templates": [r["representation"] for r in selected],
        "provenance": [{k: v for k, v in r.items() if k != "representation"} for r in selected],
        "groups": [list(range(i * 6, (i + 1) * 6)) for i in range(4)],
        "source_scene_ids": sorted({r["scene_id"] for r in selected}),
        "semantic_labels": "actual_independent_human_training_text_facts",
        "graph_semantics": "v2_development_rule_agreement; not independent graph truth",
    }
    immutable(bank_path, bank)
    immutable(seal_path, {
        "bank_sha256": sha(bank_path), "completion_freeze_sha256": sha(COMPLETION / "FREEZE.json"),
        "raw_graph_seal_sha256": sha(GRAPH / "RAW_GRAPH_SEAL.json"),
        "human_gold_seal_sha256": sha(PRIVATE / "HUMAN_GOLD_SEAL.json"),
        "training_label_sha256": runner.gold_seal()["role_label_sha256"]["train"],
        "label_roles_read": ["train"], "features_before_template_freeze": False,
        "template_actual_symbol_counts": dict(Counter(r["symbol"] for r in selected)),
    })
    return verify_templates()


def verify_templates():
    seal = load(COMPLETION / "TEMPLATE_SEAL.json")
    for path, expected in (
        (PRIVATE / "TEMPLATE_BANK.json", seal["bank_sha256"]),
        (COMPLETION / "FREEZE.json", seal["completion_freeze_sha256"]),
        (GRAPH / "RAW_GRAPH_SEAL.json", seal["raw_graph_seal_sha256"]),
        (PRIVATE / "HUMAN_GOLD_SEAL.json", seal["human_gold_seal_sha256"]),
    ):
        if sha(path) != expected:
            raise RuntimeError("Template seal drift")
    for key, expected in seal["training_label_sha256"].items():
        if sha(PRIVATE / "labels/train" / (key + ".json")) != expected:
            raise RuntimeError("Template training label drift")
    return seal


def score_document(doc, lemmas, bank, max_states=200000, observe=observation):
    import numpy as np
    x = np.full((len(doc["candidates"]), 3, 24), np.nan, dtype=np.float32)
    evidence = []
    for ci in range(len(doc["candidates"])):
        views, hashes = [], []
        for vi, kind in enumerate(("full", "full", "core")):
            rep, raw = observe(doc, ci, vi, lemmas[ci])
            scored = score_view(rep, bank["templates"], kind, max_states)
            if scored["status"] == "complete":
                x[ci, vi] = scored.pop("scores")
            hashes.append(raw["amr_sha256"])
            views.append({"view_index": vi, "representation": rep, "match": scored,
                          "raw_record_sha256": digest(raw)})
        states = [v["match"]["status"] for v in views]
        status = ("complete" if all(s == "complete" for s in states)
                  else "budget_abstain" if "budget_abstain" in states else "invalid")
        evidence.append({"status": status, "shared_core_policy": True,
                         "graph_sha256": digest(hashes), "original_graph_sha256": hashes,
                         "mapping_bank_sha256": digest(bank), "views": views})
    return x, evidence


def features():
    import numpy as np
    f = verify()
    graph_seal()
    verify_templates()
    _, runner = bind()
    if (OUT / "FEATURE_SEAL.json").exists():
        return runner.feature_seal()
    bank = load(PRIVATE / "TEMPLATE_BANK.json")
    docs = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")["documents"]
    lemmas = load(COMPLETION / "private/BLIND_LEMMAS.json")
    evidence = {}
    for doc in docs:
        key = doc["key"]
        path = PRIVATE / "features" / (key + ".npz")
        receipt = COMPLETION / "private/feature_receipts" / (key + ".json")
        proof = COMPLETION / "private/match_evidence" / (key + ".json")
        if receipt.exists():
            saved = load(receipt)
            if (saved["feature_sha256"] != sha(path) or saved["evidence_sha256"] != sha(proof)
                    or saved["template_seal_sha256"] != sha(COMPLETION / "TEMPLATE_SEAL.json")
                    or saved["completion_freeze_sha256"] != sha(COMPLETION / "FREEZE.json")):
                raise RuntimeError("Feature checkpoint input/output drift")
        else:
            x, matches = score_document(doc, lemmas[key], bank, f["matcher"]["max_states"])
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if saved.files != ["x"] or not np.array_equal(saved["x"], x, equal_nan=True):
                        raise RuntimeError("Unsealed feature checkpoint differs from recomputation")
            else:
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, x=x)
                temp.chmod(0o600)
                temp.replace(path)
            immutable(proof, matches)
            immutable(receipt, {"feature_sha256": sha(path), "evidence_sha256": sha(proof),
                               "template_seal_sha256": sha(COMPLETION / "TEMPLATE_SEAL.json"),
                               "completion_freeze_sha256": sha(COMPLETION / "FREEZE.json")})
        evidence[key] = load(proof)
        atomic(COMPLETION / "HEARTBEAT.json", {"stage": "features", "documents": len(evidence),
                                              "total_documents": len(docs), "labels_read": False})
    immutable(PRIVATE / "MATCH_EVIDENCE.json", evidence)
    request = load(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json")
    immutable(PRIVATE / "FEATURE_METADATA.json", {
        "alphabet_version": request["alphabet_version"], "views": request["views"],
        "template_semantics": "new_transfer_direction_status",
        "candidate_source_sha256": sha(PRIVATE / "FEATURE_REQUEST_NO_TARGET.json"),
        "core_matcher_ready": True, "groups": bank["groups"],
        "completion_freeze_sha256": sha(COMPLETION / "FREEZE.json"),
        "template_seal_sha256": sha(COMPLETION / "TEMPLATE_SEAL.json"),
        "raw_graph_seal_sha256": sha(GRAPH / "RAW_GRAPH_SEAL.json"),
        "core_semantics_independently_validated": False,
    })
    result = runner.seal_features()
    immutable(COMPLETION / "FEATURE_AUDIT.json", {
        "documents": len(docs), "candidates": sum(len(d["candidates"]) for d in docs),
        "view_states": dict(Counter(v["match"]["status"] for values in evidence.values()
                                   for row in values for v in row["views"])),
        "feature_seal_sha256": sha(OUT / "FEATURE_SEAL.json"), "labels_read_during_scoring": False,
        "all_failed_views_retained_as_NaN": True, "training_runs": 0,
    })
    return result


def check():
    verify()
    reasons = []
    for path, reason in (
        (GRAPH / "RAW_GRAPH_SEAL.json", "full_raw_graph_computation_pending"),
        (PRIVATE / "HUMAN_GOLD_SEAL.json", "independent_human_labels_pending"),
        (COMPLETION / "TEMPLATE_SEAL.json", "training_fact_templates_pending"),
        (OUT / "FEATURE_SEAL.json", "full_feature_release_pending"),
    ):
        if not path.exists():
            reasons.append(reason)
    result = {"status": "PENDING" if reasons else "READY_FOR_MODEL_SELECTION",
              "reasons": reasons, "blind_lemmas_sealed": True,
              "documents": 192, "candidates": 1320, "view_occurrences": 3960}
    atomic(COMPLETION / "READINESS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "prepare", "check", "collect", "templates", "features",
        "select", "calibrate", "predict", "evaluate"))
    args = parser.parse_args()
    COMPLETION.mkdir(parents=True, exist_ok=True)
    with (COMPLETION / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            result = prepare()
        else:
            verify()
            if args.action in ("check", "templates", "features"):
                result = {"check": check, "templates": templates, "features": features}[args.action]()
            else:
                collector, runner = bind()
                if args.action == "collect":
                    result = collector.collect()
                else:
                    verify_templates()
                    result = getattr(runner, args.action)()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
