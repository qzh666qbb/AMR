"""Existing-corpus symbol and order diagnostics, without AMR parsing or training.

Creates a new, immutable-input analysis. Paragraph sentence streams are distinct
from known-position carrier diagnostics and from formal blind event recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.semantic_sequence import (
    equal_composition_sequences, exact_costs, rank_truth, symbol_weights,
    unordered_costs,
)

BASE = ROOT / "experiments/semantic_sequence_stage1_run2"
TEMPERATURE = 0.10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n")


def weights(values, groups):
    return symbol_weights(values, groups, TEMPERATURE)


def identity(rows):
    support = np.zeros(4)
    confusion = np.zeros((4, 5))
    nll, brier, top2 = [], [], []
    confident = errors = 0
    for row in rows:
        target, q = row["target"], row["q"]
        support[target] += 1
        if q is None:
            q = np.ones(4) / 4
            confusion[target, 4] += 1
        else:
            q = np.asarray(q)
            best = np.flatnonzero(np.isclose(q, q.max(), atol=1e-12, rtol=0))
            confusion[target, best] += 1 / len(best)
            if q.max() >= 0.8:
                confident += 1
                errors += 1 - (1 / len(best) if target in best else 0)
        above = int((q > q[target] + 1e-12).sum())
        tied = int(np.isclose(q, q[target], atol=1e-12, rtol=0).sum())
        top2.append(min(1, max(0, 2 - above) / tied))
        nll.append(-math.log(max(float(q[target]), 1e-300)))
        reference = np.eye(4)[target]
        brier.append(float(np.square(q - reference).sum()))
    recall = np.divide(np.diag(confusion[:, :4]), support,
                       out=np.zeros(4), where=support > 0)
    n = len(rows)
    return {
        "planned": n, "scored": n - int(confusion[:, 4].sum()),
        "erased": int(confusion[:, 4].sum()), "support_ABCD": support.tolist(),
        "balanced_accuracy": float(recall.mean()),
        "correct_credit": float(np.diag(confusion[:, :4]).sum()),
        "recall_ABCD": recall.tolist(), "confusion_ABCD_erasure": confusion.tolist(),
        "top2_credit": float(np.mean(top2)), "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "high_confidence_coverage": confident / n,
        "high_confidence_error": errors / confident if confident else None,
        "erasure_reasons": dict(Counter(r["reason"] for r in rows if r["q"] is None)),
    }


def background(rows):
    counts = np.zeros(5)
    for q in rows:
        if q is None:
            counts[4] += 1
        else:
            best = np.flatnonzero(np.isclose(q, max(q), atol=1e-12, rtol=0))
            counts[best] += 1 / len(best)
    return {"observations": len(rows), "counts_ABCD_erasure": counts.tolist(),
            "accuracy": None, "labels": "No intended labels for background"}


def oracle_validation(candidates):
    """Exhaustive path oracle on small cases, independent of the DP recurrence."""
    from functools import lru_cache
    rng = np.random.default_rng(20261025)
    checked = 0
    for mode in ("hard", "soft"):
        for m in range(4):
            for n in range(4):
                q = rng.dirichlet(np.ones(4), size=m)
                candidate = rng.integers(0, 4, size=n, dtype=np.int8)
                scores = exact_costs(list(q), candidate[None, :], mode)[0]
                match = []
                for w in q:
                    match.append((-np.log(np.maximum(w, 1e-6)) / math.log(4))
                                 if mode == "soft" else (np.arange(4) != w.argmax()).astype(float))

                @lru_cache(None)
                def paths(i, j):
                    if i == m and j == n:
                        return (0.0,)
                    values = []
                    if i < m:
                        values.extend(1 + x for x in paths(i + 1, j))
                    if j < n:
                        values.extend(1 + x for x in paths(i, j + 1))
                    if i < m and j < n:
                        values.extend(float(match[i][candidate[j]]) + x
                                      for x in paths(i + 1, j + 1))
                    return tuple(values)

                assert abs(scores - min(paths(0, 0))) <= 1e-12
                checked += 1
    assert candidates.shape == (2520, 8)
    assert len(set(map(tuple, candidates))) == 2520
    assert all(Counter(row) == {0: 2, 1: 2, 2: 2, 3: 2} for row in candidates)
    for q in ([None] * 8, [np.ones(4) / 4] * 8, []):
        for mode in ("hard", "soft"):
            scores = exact_costs(q, candidates, mode)
            assert np.ptp(scores) == 0
            assert rank_truth(scores, candidates, candidates[0])["pairwise_win_rate"] == 0.5
        assert np.ptp(unordered_costs(q, candidates)) == 0
    return {"small_exhaustive_path_cases": checked, "candidate_count": 2520,
            "equal_frequency": True, "uniform_and_missing_tie": True,
            "no_sampled_alignment_paths": True}


def segment(text, nlp):
    units = []
    for span in nlp(text).sents:
        value = span.text.strip()
        start = span.start_char + len(span.text) - len(span.text.lstrip())
        units.append({"text": value, "span": [start, start + len(value)]})
    return units


def eligible_parse(record, freeze_hash, alignment=False):
    if record["freeze_sha256"] != freeze_hash:
        return False
    if not isinstance(record.get("amr"), str):
        return False
    if digest(record["amr"]) != record.get("amr_sha256"):
        return False
    if alignment:
        return (record["graph_roundtrip_valid"] and not record["output_limit_hit"]
                and not record["decode_events"]
                and not record["recovery_audit"]["events"]
                and record["recovery_audit"]["selected_beam"] == 0)
    return (record["valid"] and not record["clipped"] and not record["events"]
            and not record["decode_events"]
            and all(not a.get("events") for a in record["attempts"])
            and any(a.get("accepted") and a.get("beam_index") == 0 for a in record["attempts"]))


def prepare(out, encoder):
    if out.exists():
        raise RuntimeError("Use a new output directory; do not overwrite a diagnostic")
    transfer = BASE / "multiscale_transfer2"
    sourcepaths = [
        Path(__file__), ROOT / "src/core/semantic_sequence.py",
        ROOT / "scripts/report_symbol_sequence_diagnostic.py",
        ROOT / "experiments/semantic_sequence_stage1_prompt_repair1/PROMPT_REPAIR.md",
        BASE / "SYMBOL_MAP.json", BASE / "private/ASSIGNMENTS.jsonl",
        BASE / "private/ai_review1/EVALUATOR_AI_LABELS.jsonl",
        BASE / "MANIFEST_GENERATION.sha256", BASE / "ai_review1/MANIFEST_AI_REVIEW.sha256",
        transfer / "FREEZE.json", transfer / "SIGNAL_COMPLETE.json",
        transfer / "SYMBOL_DIAGNOSTIC.json",
        transfer / "private/BANK.json", transfer / "private/propbank_verbs.json",
        BASE / "alignment_review1/FREEZE.json",
        BASE / "alignment_review1/private/INPUTS.json",
        BASE / "alignment_review1/private/CARRIERS.json",
        BASE / "alignment_review1/private/ALIGNMENT_DETAILS.json",
        BASE / "alignment_review1/REVIEW_CONCLUSIONS.json",
        BASE / "multiscale_transfer3/FREEZE.json",
        BASE / "multiscale_transfer3/HALT.json",
        ROOT / "experiments/semantic_sequence_cycle2/ASSET_MANIFEST.json",
        ROOT / "vectors/glove.6B.100d.txt",
    ]
    sourcepaths += sorted((BASE / "private/documents").glob("d*.json"))
    sourcepaths += sorted((transfer / "private/signal").glob("*.json"))
    sourcepaths += sorted((BASE / "alignment_review1/private/PARSES").glob("*.json"))
    sourcepaths += sorted((BASE / "multiscale_transfer3/private/parses").glob("*.json"))
    source_manifest = {str(p.relative_to(ROOT)): sha(p) for p in sourcepaths}
    freeze2 = read(transfer / "FREEZE.json")
    assert sha(transfer / "FREEZE.json") == "909a0ce1acd364627cb1a2beefb0ab0e3ee721e54c5b77479b3af11fb3d7d5b1"
    for path, expected in {**freeze2["sources"], **freeze2["normalizer_assets"]}.items():
        assert sha(transfer / path) == expected, path
    assert sha(ROOT / "vectors/glove.6B.100d.txt") == freeze2["scorer"]["glove_sha256"]
    sealed_counts = {}
    for manifest in (BASE / "MANIFEST_GENERATION.sha256",
                     BASE / "ai_review1/MANIFEST_AI_REVIEW.sha256"):
        count = 0
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(None, 1)
            assert sha(ROOT / relative.strip()) == expected
            count += 1
        sealed_counts[str(manifest.relative_to(ROOT))] = count
    assets = read(ROOT / "experiments/semantic_sequence_cycle2/ASSET_MANIFEST.json")
    for asset in assets["minilm"]:
        assert sha(encoder / asset["path"]) == asset["sha256"], asset["path"]
    out.mkdir(parents=True)
    (out / ".gitignore").write_text("private/\n*.log\n")
    (out / "private").mkdir()
    shutil.copytree(transfer / "private/code", out / "private/code")
    shutil.copytree(transfer / "private/spacy_model", out / "private/spacy_model")
    for name in ("BANK.json", "propbank_verbs.json"):
        shutil.copyfile(transfer / "private" / name, out / "private" / name)
    shutil.copyfile(__file__, out / "private/run_symbol_sequence_diagnostic.py")
    shutil.copyfile(ROOT / "src/core/semantic_sequence.py", out / "private/semantic_sequence.py")
    write(out / "SOURCE_SHA256.json", source_manifest)
    frozen = {
        "schema": "existing-symbol-order-diagnostic/v1",
        "date": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested symbol-layer and equal-frequency order diagnostics",
        "scope": "Exploratory reuse of already-observed NEW24 corpus; no independent confirmation",
        "temperature": TEMPERATURE, "temperature_selection": "Existing design-v1 fixed T=.10",
        "symbol_map_sha256": sha(BASE / "SYMBOL_MAP.json"),
        "source_manifest_sha256": sha(out / "SOURCE_SHA256.json"),
        "existing_generation_and_AI_seals_verified": sealed_counts,
        "split": freeze2["split"],
        "encoder": {"revision": assets["minilm_revision"], "assets": assets["minilm"],
                    "device": "cpu", "dtype": "float32", "batch": 32,
                    "prototype": "Mean all 24 intended-label development carrier vectors per class, normalize",
                    "no_gradient_training": True, "no_score_selection": True},
        "paraphrase_mapping": "All sentencizer sentences overlapping sealed AI evidence span, equal mean; any missing erases; not human gold",
        "paraphrase_order": "Full sentence stream in observed text order; no mapping or gold order used",
        "alignment_scope": "Original frozen 3 development documents, 24 carriers and 19 paraphrase units",
        "scorer": freeze2["scorer"],
        "AMR_inference": "NONE; immutable eligible prior graphs only",
        "no_generation": True, "no_model_training": True,
        "no_resume_HALTED_run": True, "no_formal_watermark_FPR_or_certificate": True,
        "no_blind_event_identity_F1_without_human_gold": True,
    }
    write(out / "FREEZE.json", frozen)
    return frozen


def order_summary(rows):
    values = np.asarray([r["pairwise_win_rate"] for r in rows])
    rng = np.random.default_rng(20261025)
    boot = rng.choice(values, (10000, len(values)), replace=True).mean(1)
    return {
        "documents": len(rows), "mean_pairwise_win_rate": float(values.mean()),
        "gain_over_0_5": float(values.mean() - 0.5),
        "descriptive_document_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
        "mean_rank": float(np.mean([r["average_rank"] for r in rows])),
        "mean_top1_tie_credit": float(np.mean([r["top1_credit"] for r in rows])),
        "unique_correct_top1": sum(r["top1_credit"] == 1 for r in rows),
        "documents_with_any_correct_top1": sum(r["top1_credit"] > 0 for r in rows),
        "mean_observations": float(np.mean([r["observations"] for r in rows])),
        "all_erased_documents": sum(r["all_erased"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    args = ap.parse_args()
    out, encoder = args.out.resolve(), args.encoder.resolve()
    frozen = prepare(out, encoder)
    import spacy
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    torch.use_deterministic_algorithms(True)
    splitter = spacy.blank("en")
    splitter.add_pipe("sentencizer")
    symbol_map = read(BASE / "SYMBOL_MAP.json")
    mapping = {int(k): "ABCD".index(v) for k, v in symbol_map["template_to_symbol"].items()}
    pool = sorted(mapping)
    groups = [[i for i, t in enumerate(pool) if mapping[t] == a] for a in range(4)]
    assert len(pool) == 24 and all(len(g) == 6 for g in groups)
    assert not symbol_map["cross_symbol_collisions"]
    assignments = {r["doc_id"]: r for r in jsonl(BASE / "private/ASSIGNMENTS.jsonl")}
    documents = {r["doc_id"]: r for r in
                 (read(p) for p in sorted((BASE / "private/documents").glob("d*.json")))}
    labels = {(r["doc_id"], r["planned_carrier_slot"]): r for r in
              jsonl(BASE / "private/ai_review1/EVALUATOR_AI_LABELS.jsonl")}
    assert len(assignments) == len(documents) == 24 and len(labels) == 192
    clean_q, sentence_units, streams, identities, controls = {}, {}, {}, [], defaultdict(list)
    for docid, doc in documents.items():
        signal = read(BASE / f"multiscale_transfer2/private/signal/d{docid:02d}.json")
        assert signal["freeze_sha256"] == sha(BASE / "multiscale_transfer2/FREEZE.json")
        for arm, payload in doc["arms"].items():
            assert digest(payload["text"]) == payload["text_sha256"]
            assert digest(payload["paraphrase"]) == payload["paraphrase_sha256"]
            assert len(payload["slots"]) == len(signal["arms"][arm]["sentence_scores"]) == 8
            vectors = signal["arms"][arm]["sentence_scores"]
            assert [digest(s["text"]) for s in payload["slots"]] == signal["arms"][arm]["sentence_sha256"]
            qs = [weights([s[t] for t in pool], groups) for s in vectors]
            clean_q[(docid, arm)] = qs
            streams[("AMR_original_carrier", "clean", docid, arm)] = qs
            sentence_units[(docid, arm)] = segment(payload["paraphrase"], splitter)
            if arm == "positive":
                for i, q in enumerate(qs):
                    slot = assignments[docid]["slots"][i]
                    identities.append({"method": "AMR_original_carrier", "condition": "clean",
                                       "doc_id": docid, "slot": i, "target": mapping[slot["intended_template"]],
                                       "q": q, "reason": None if q is not None else "score_missing"})
            else:
                controls[("AMR_original_carrier", "clean", doc["split"])].extend(qs)
    model = SentenceTransformer(str(encoder), local_files_only=True, device="cpu", token=False)
    model.eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    texts = list(dict.fromkeys(
        normalize(s["text"]) for doc in documents.values() for arm in doc["arms"].values()
        for s in arm["slots"]))
    texts += list(dict.fromkeys(normalize(u["text"]) for units in sentence_units.values() for u in units
                               if normalize(u["text"]) not in texts))
    lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
    assert max(lengths) <= model.max_seq_length
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    assert np.isfinite(vectors).all()
    replay = model.encode(texts[:8], batch_size=32, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
    max_error = float(np.max(np.abs(vectors[:8] - replay)))
    assert max_error <= 1e-6
    memo = dict(zip(texts, vectors))
    prototype_rows = [[] for _ in range(4)]
    for docid in frozen["split"]["development"]:
        for slot in assignments[docid]["slots"]:
            text = documents[docid]["arms"]["positive"]["slots"][slot["slot"]]["text"]
            prototype_rows[mapping[slot["intended_template"]]].append(memo[normalize(text)])
    assert list(map(len, prototype_rows)) == [24] * 4
    prototypes = np.asarray([np.mean(x, axis=0) for x in prototype_rows])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    np.save(out / "private/PROTOTYPES.npy", prototypes, allow_pickle=False)
    embedq = {text: weights(vector @ prototypes.T, [[0], [1], [2], [3]])
              for text, vector in memo.items()}
    print(json.dumps({"embedding_done": len(texts), "max_tokens": max(lengths),
                      "replay_max_abs": max_error}), flush=True)
    for docid, doc in documents.items():
        for arm, payload in doc["arms"].items():
            qs = [embedq[normalize(s["text"])] for s in payload["slots"]]
            streams[("C_original_carrier", "clean", docid, arm)] = qs
            paraqs = [embedq[normalize(u["text"])] for u in sentence_units[(docid, arm)]]
            streams[("C_sentence_stream", "paraphrase", docid, arm)] = paraqs
            if arm == "unwatermarked":
                controls[("C_original_carrier", "clean", doc["split"])].extend(qs)
                controls[("C_sentence_stream", "paraphrase", doc["split"])].extend(paraqs)
                continue
            for i, q in enumerate(qs):
                identities.append({"method": "C_original_carrier", "condition": "clean",
                                   "doc_id": docid, "slot": i, "target": "ABCD".index(assignments[docid]["slots"][i]["intended_state"]),
                                   "q": q, "reason": None})
    # Import the frozen scorer in isolation; its SQLite writes occur ONLY in the new private/.
    spec = importlib.util.spec_from_file_location(
        "frozen_transfer_diagnostic", out / "private/code/scripts/run_multiscale_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.Scorer(ROOT, out)
    template_q = []
    for t in pool:
        score = [scorer.pair(scorer.bank[t], scorer.bank[k]) for k in pool]
        assert score[pool.index(t)] == 1.0
        template_q.append({"target": mapping[t], "template": t,
                           "q": weights(score, groups), "reason": None})
    write(out / "TEMPLATE_SELF_READ.json",
          {"scope": "Classifier-only synthetic perfect-template input, not realized text",
           "metrics": identity(template_q),
           "rows": [{**r, "q": None if r["q"] is None else r["q"].tolist()} for r in template_q]})
    parse_memo = {}
    alignment = BASE / "alignment_review1"
    for p in sorted((alignment / "private/PARSES").glob("*.json")):
        r = read(p)
        if eligible_parse(r, sha(alignment / "FREEZE.json"), alignment=True):
            assert digest(r["text"]) == r["text_sha256"]
            parse_memo[digest(normalize(r["text"]))] = (r, str(p.relative_to(ROOT)), "alignment")
    transfer3 = BASE / "multiscale_transfer3"
    for p in sorted((transfer3 / "private/parses").glob("*.json")):
        r = read(p)
        if eligible_parse(r, sha(transfer3 / "FREEZE.json")):
            parse_memo.setdefault(r["text_sha256"], (r, str(p.relative_to(ROOT)), "transfer3"))
    scorememo, coverage = {}, []
    for key, units in sentence_units.items():
        docid, arm = key
        amrqs = []
        for index, u in enumerate(units):
            textkey = digest(normalize(u["text"]))
            entry = parse_memo.get(textkey)
            q = None
            if entry is not None:
                if textkey not in scorememo:
                    record = entry[0]
                    try:
                        norm = scorer.normalize(record["amr"], "official")
                        values = [scorer.pair(norm, scorer.bank[t]) for t in pool]
                        scorememo[textkey] = weights(values, groups)
                    except ValueError as exc:
                        scorer.events.append({"graph_sha256": record["amr_sha256"],
                                              "reason": type(exc).__name__,
                                              "scope": "diagnostic_normalization_erasure"})
                        scorememo[textkey] = None
                q = scorememo[textkey]
            u["AMR_q"] = q
            amrqs.append(q)
            coverage.append({"doc_id": docid, "arm": arm, "unit": index,
                             "q_available": q is not None,
                             "parse_source": entry[1] if entry else None,
                             "text_sha256": textkey})
        streams[("AMR_sentence_stream_incomplete", "paraphrase", docid, arm)] = amrqs
        if arm == "unwatermarked":
            controls[("AMR_sentence_stream_incomplete", "paraphrase", documents[docid]["split"])].extend(amrqs)
    for docid in documents:
        units = sentence_units[(docid, "positive")]
        for slot in assignments[docid]["slots"]:
            i = slot["slot"]
            span = labels[(docid, i)]["paraphrase"].get("evidence_span")
            linked = [u for u in units if span and max(span[0], u["span"][0]) < min(span[1], u["span"][1])]
            for method in ("C_AI_mapped_sentence_mean", "AMR_AI_mapped_sentence_mean"):
                values = [(embedq[normalize(u["text"])] if method.startswith("C_") else u["AMR_q"])
                          for u in linked]
                q = np.mean(values, axis=0) if values and all(v is not None for v in values) else None
                reason = None if q is not None else ("mapping_missing" if not linked else "parse_or_score_missing")
                identities.append({"method": method, "condition": "paraphrase", "doc_id": docid,
                                   "slot": i, "target": mapping[slot["intended_template"]],
                                   "q": q, "reason": reason, "associated_units": len(linked)})
    candidates = equal_composition_sequences()
    validation = oracle_validation(candidates)
    prediction_arrays, prediction_keys = [], []
    order_records = []
    for streamkey, qs in sorted(streams.items()):
        method, condition, docid, arm = streamkey
        for mode in ("hard", "soft", "unordered"):
            cost = unordered_costs(qs, candidates) if mode == "unordered" else exact_costs(qs, candidates, mode)
            assert np.array_equal(cost, unordered_costs(qs, candidates) if mode == "unordered" else exact_costs(qs, candidates, mode))
            if mode == "unordered":
                assert np.ptp(cost) == 0
            prediction_arrays.append(cost)
            prediction_keys.append([method, condition, docid, arm, mode])
    np.savez_compressed(out / "private/ORDER_COSTS.npz", costs=np.stack(prediction_arrays), candidates=candidates)
    write(out / "private/ORDER_KEYS.json", prediction_keys)
    write(out / "PREDICTION_SEAL.json",
          {"cost_sha256": sha(out / "private/ORDER_COSTS.npz"),
           "key_sha256": sha(out / "private/ORDER_KEYS.json"),
           "source_manifest_sha256": sha(out / "SOURCE_SHA256.json"),
           "labels_joined_after_cost_seal": True, "costs_never_access_truth": True})
    for key, cost in zip(prediction_keys, prediction_arrays):
        method, condition, docid, arm, mode = key
        truth = [s["intended_state"] for s in assignments[docid]["slots"]]
        ranked = rank_truth(cost, candidates, truth)
        qs = streams[(method, condition, docid, arm)]
        ranked.update(method=method, condition=condition, doc_id=docid, arm=arm,
                      mode=mode, population=documents[docid]["split"],
                      observations=len(qs), erasures=sum(q is None for q in qs),
                      all_erased=all(q is None for q in qs),
                      background_reference=arm == "unwatermarked")
        order_records.append(ranked)
    identity_result, order_result, control_result = {}, {}, {}
    for population in ("development", "readout"):
        ids = frozen["split"][population]
        identity_result[population] = {}
        for method in sorted({r["method"] for r in identities}):
            rows = [r for r in identities if r["method"] == method and r["doc_id"] in ids]
            identity_result[population][method] = identity(rows)
        order_result[population] = {}
        for method, condition in sorted({(r["method"], r["condition"]) for r in order_records}):
            for arm in ("positive", "unwatermarked"):
                for mode in ("hard", "soft", "unordered"):
                    rows = [r for r in order_records if r["population"] == population
                            and r["method"] == method and r["condition"] == condition
                            and r["arm"] == arm and r["mode"] == mode]
                    order_result[population][f"{method}/{arm}/{mode}"] = order_summary(rows)
        paired = {}
        for method, condition in sorted({(r["method"], r["condition"]) for r in order_records}):
            for mode in ("hard", "soft"):
                selected = [r for r in order_records if r["population"] == population
                            and r["method"] == method and r["condition"] == condition
                            and r["mode"] == mode]
                perdoc = defaultdict(dict)
                for r in selected:
                    perdoc[r["doc_id"]][r["arm"]] = r["pairwise_win_rate"]
                difference = np.asarray([r["positive"] - r["unwatermarked"] for r in perdoc.values()])
                rng = np.random.default_rng(20261025)
                boot = rng.choice(difference, (10000, len(difference)), replace=True).mean(1)
                paired[f"{method}/{mode}"] = {
                    "mean_positive_minus_background_reference": float(difference.mean()),
                    "descriptive_document_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
                    "positive_greater_documents": int((difference > 1e-10).sum()),
                    "background_order_is_paired_reference_not_true_label": True,
                }
        write(out / f"ORDER_PAIRED_{population.upper()}.json", paired)
        control_result[population] = {
            f"{m}/{c}": background(qs) for (m, c, p), qs in controls.items() if p == population}
    write(out / "IDENTITY_RESULT.json", identity_result)
    write(out / "ORDER_RESULT.json", order_result)
    write(out / "ORDER_PER_DOCUMENT.json", order_records)
    write(out / "BACKGROUND_RESULT.json", control_result)
    write(out / "PARAPHRASE_COVERAGE.json", coverage)
    write(out / "private/IDENTITY_ROWS.json",
          [{**r, "q": None if r["q"] is None else list(map(float, r["q"]))} for r in identities])
    write(out / "SCORING_AUDIT.json", {"calls": scorer.calls, "invalid": scorer.events})
    scorer.db.close()
    conclusions = read(alignment / "REVIEW_CONCLUSIONS.json")
    layers = []
    for carrier in read(alignment / "private/CARRIERS.json"):
        for condition in ("clean", "paraphrase"):
            detail = next(r for r in read(alignment / "private/ALIGNMENT_DETAILS.json")
                          if r["carrier_id"] == carrier["carrier_id"] and r["condition"] == condition)
            inputs = [carrier["clean_input"]] if condition == "clean" else carrier["paraphrase_inputs"]
            examples = [e for e in conclusions["clear_parser_loss_examples"] if e["input_id"] in inputs]
            layers.append({
                "doc_id": carrier["doc_id"], "slot": carrier["slot"],
                "template": carrier["template"], "condition": condition,
                "full_target_coverage_official": detail["official_full_target_anywhere"],
                "AI_realized_historical_only": detail["prior_AI_realized"],
                "generation_failure_independent_label": None,
                "parser_loss_specific_observations": examples,
                "classifier_failure_given_confirmed_realization": None,
                "residual_causal_attribution": "UNKNOWN_WITHOUT_INDEPENDENT_TEXT_AND_GRAPH_GOLD",
            })
    write(out / "LAYER_DIAGNOSIS.json", {
        "scope": "Frozen 3-document development diagnostic; not whole-cohort causal rates",
        "counts_from_existing_alignment": conclusions["counts"],
        "prompt_contract_mismatch": "Confirmed: ordinary unspecified concept versus amr-unknown",
        "independent_human_ground_truth": "NOT_EVALUATED",
        "classifier_only_perfect_template_accuracy": identity(template_q)["balanced_accuracy"],
        "all_original_and_paraphrase_complete_target_coverage": [0, 24],
        "rows": layers,
    })
    # Independently reproduce the previously reported 35/96 original accuracy.
    original = read(BASE / "multiscale_transfer2/SYMBOL_DIAGNOSTIC.json")
    for p in ("development", "readout"):
        assert abs(identity_result[p]["AMR_original_carrier"]["balanced_accuracy"]
                   - original["populations"][p]["methods"]["official_sentence"]["balanced_accuracy"]) < 1e-12
    validation.update({
        "original_identity_reproduced": True,
        "embedding_replay_max_absolute_error": max_error,
        "embedding_no_truncation": True, "embedding_max_tokens": max(lengths),
        "cost_replay_exact": True, "prediction_sealed_before_truth_join": True,
        "source_hashes_unchanged": all(sha(ROOT / p) == h for p, h in read(out / "SOURCE_SHA256.json").items()),
        "old_HALTED_freeze_and_HALT_unchanged": True,
    })
    assert validation["source_hashes_unchanged"]
    write(out / "VALIDATION.json", validation)
    write(out / "STATUS.json",
          {"analysis": "COMPLETE", "AMR_full_paraphrase": "INCOMPLETE_CACHE",
           "human_ground_truth": "NOT_EVALUATED", "training": "NONE",
           "new_generation": "NONE", "new_AMR_inference": "NONE",
           "original_transfer3_state": "PERMANENT_HALT_UNCHANGED"})
    print(json.dumps({"complete": True, "readout_identity": identity_result["readout"],
                      "template_self": identity(template_q),
                      "readout_order": {k: v for k, v in order_result["readout"].items()
                                        if "/positive/soft" in k}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
