"""Replay real frozen observations with strict atomic A0/A1 and pending A2."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.core import atomic_alignment_arms as arms
from src.core.multiview_views import Event
from run_multiscale_transfer import atomic, read, sha, digest

SOURCE = ROOT / "experiments/swan_multiview_development1/pipeline3"
DEFAULT = ROOT / "experiments/swan_atomic_alignment_completion1"


def run(out, encoder):
    import torch
    from sentence_transformers import SentenceTransformer
    if out.exists():
        raise RuntimeError("Use a new immutable replay directory")
    manifest = read(SOURCE / "MANIFEST.json")
    assets = manifest["encoder_assets_sha256"]
    for name, value in assets.items():
        if sha(encoder / name) != value:
            raise RuntimeError("Original encoder asset drift: " + name)
    paths = sorted((SOURCE / "private/results").glob("*.json"))
    records = [read(p) for p in paths]
    if len(records) != 7 or len({r["group_id"] for r in records}) != 1:
        raise RuntimeError("Original one-source/seven-condition scope differs")
    items, metadata = [], []
    for r in records:
        d = r["variants"]["predicted_fixed"]
        obs, events = d["observations"], d["events"]
        for o in obs:
            if digest(o["text"]) != o["span_sha256"]:
                # Original span hashes use the project's whitespace normalization.
                from src.core.sqlite_cache import text_sha256
                if text_sha256(o["text"]) != o["span_sha256"]:
                    raise RuntimeError("Candidate text identity drift")
            body = "\n".join(line for line in o["amr"].splitlines() if not line.startswith("#"))
            if digest(body) != o["graph_body_sha256"]:
                raise RuntimeError("Original graph body identity drift")
        cs = [{k: o[k] for k in arms.CANDIDATE_FIELDS} for o in obs]
        es = [Event(**e) for e in events]
        common = arms.universe(cs, es)
        items.append({"row_id": r["row_id"], "variant_id": r["variant_id"],
                      "candidates": cs, "events": events, "surfaces": d["surface_observations"],
                      "candidate_texts": [o["text"] for o in obs],
                      "event_texts": [e.description for e in es]})
        metadata.append({"row_id": r["row_id"], "observations": obs, "universe": common})
    private = out / "private"
    private.mkdir(parents=True, mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    atomic(private / "BLIND_INPUTS.json", items)
    atomic(private / "POST_SELECTION_SCORES.json", metadata)
    code = ["scripts/run_atomic_alignment_completion1.py", "src/core/atomic_alignment_arms.py",
            "src/core/multiview_views.py", "src/core/multiview_alignment.py",
            "scripts/test_atomic_alignment_arms.py"]
    frozen = {
        "version": arms.VERSION, "source_rows": 7, "independent_sources": 1,
        "source_scope": "existing unbounded-candidate development diagnostic",
        "future_formal_candidate_budget": 24, "formal_budget_run": False,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "code_sha256": {p: sha(ROOT / p) for p in code},
        "encoder_assets_sha256": assets, "dtype": "float32", "device": "cpu",
        "blind_inputs_sha256": sha(private / "BLIND_INPUTS.json"),
        "post_selection_scores_sha256": sha(private / "POST_SELECTION_SCORES.json"),
        "A2": "requires independent complete support labels for every common pair",
        "new_thresholds": None, "human_rows": 0,
        "shared_policy": "usable candidates, atomic event capacity1/candidate capacity3, graph count once",
    }
    atomic(out / "FREEZE.json", frozen)
    torch.set_num_threads(2)
    torch.manual_seed(20261102)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(str(encoder), device="cpu", local_files_only=True)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Encoder dtype changed")
    texts = list(dict.fromkeys(t for r in items for t in r["candidate_texts"] + r["event_texts"]))
    lengths = [len(x) for x in model.tokenizer(texts, truncation=False)["input_ids"]]
    if max(lengths) > model.max_seq_length:
        raise RuntimeError("Encoder context exceeded; no truncation")
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    replay = model.encode(texts[:8], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    error = float(np.max(np.abs(vectors[:8] - replay)))
    if error > 1e-6 or not np.isfinite(vectors).all():
        raise RuntimeError("Encoder replay failed")
    memo = dict(zip(texts, vectors))
    predictions = []
    for r in items:
        cs, es = r["candidates"], [Event(**e) for e in r["events"]]
        tv = np.stack([memo[t] for t in r["candidate_texts"]])
        ev = np.stack([memo[t] for t in r["event_texts"]])
        a0 = arms.no_alignment(cs, es)
        a1 = arms.predicted(cs, es, tv, ev, r["surfaces"])
        again = arms.predicted(cs, es, tv, ev, r["surfaces"])
        if again != a1:
            raise RuntimeError("A1 deterministic replay failed")
        predictions.append({"row_id": r["row_id"], "variant_id": r["variant_id"], "A0": a0, "A1": a1})
    np.savez_compressed(private / "SHARED_VECTORS.npz", vectors=vectors)
    atomic(private / "VECTOR_TEXT_SHA256.json", [digest(t) for t in texts])
    atomic(private / "PREDICTIONS.json", predictions)
    atomic(out / "PREDICTION_SEAL.json", {
        "freeze_sha256": sha(out / "FREEZE.json"),
        "predictions_sha256": sha(private / "PREDICTIONS.json"),
        "shared_vectors_sha256": sha(private / "SHARED_VECTORS.npz"),
        "human_answers_read": False, "Bank_scores_in_alignment": False,
    })
    # Human pair templates cover the full universe, including every A1 rejection.
    review, rows = [], []
    for r, m in zip(predictions, read(private / "POST_SELECTION_SCORES.json")):
        review.append({
            "row_id": r["row_id"], "pairs": [
                {**p, "same_fact_supported": None, "fact_instance_id": None, "rationale": ""}
                for p in m["universe"]],
            "human_attestation": False,
        })
        rows.append({
            "variant_id": r["variant_id"], "candidate_occurrences": len(m["observations"]),
            "atomic_events": len(next(x["events"] for x in items if x["row_id"] == r["row_id"])),
            "common_pairs": len(m["universe"]),
            **{a: {**arms.fixed_statistic(r[a]["selected"], m["observations"]),
                   "selected_observations": len(r[a]["selected"]),
                   "cross_candidate": sum(e["candidate_index"] != e["event_candidate_index"] for e in r[a]["selected"])}
               for a in ("A0", "A1")},
            "A2": None,
        })
    atomic(private / "ORACLE_PAIR_SCHEMA_NOT_HUMAN_EVIDENCE.json", review)
    result = {
        "status": "A0_A1_DEVELOPMENT_REPLAY_COMPLETE_A2_PENDING_HUMANS",
        "independent_sources": 1, "conditions": len(rows), "rows": rows,
        "human_valid_rows": 0, "formal_FPR": None, "formal_TPR": None,
        "identity_precision": None, "identity_recall": None,
        "source_note": "7 conditions share one source; no independent calibration or utility conclusion",
    }
    atomic(out / "RESULT.json", result)
    atomic(out / "AUDIT.json", {
        "source_SHA_verified": all(sha(ROOT / p) == h for p, h in frozen["source_sha256"].items()),
        "code_SHA_verified": all(sha(ROOT / p) == h for p, h in frozen["code_sha256"].items()),
        "encoder_replay_max_abs": error, "A1_exact_replay": True,
        "all_atomic_observations_retained": True, "no_predicted_groups": True,
        "new_parser_calls": 0, "human_labels_fabricated": False,
        "same_fixed_graph_aggregation": True, "completed_three_arm_scientific_test": False,
    })
    table = "\n".join(
        f"| {r['variant_id']} | {r['atomic_events']} | {r['A0']['selected_observations']} | "
        f"{r['A1']['selected_observations']} | {r['A0']['statistic']:.6f} | {r['A1']['statistic']:.6f} |"
        for r in rows)
    (out / "REPORT.md").write_text(
        "# 原子观测三臂：真实开发回放\n\n"
        "A0/A1已在同一来源的7个真实AMR条件上完成；A2待独立完整身份标签。"
        "这些是未校准统计量，不能说明A1有性能收益，也不能当成正式三臂完成。\n\n"
        "| 条件 | 原子事件 | A0选中 | A1选中 | A0统计量 | A1统计量 |\n"
        "|---|---:|---:|---:|---:|---:|\n" + table + "\n\n"
        "A0按原事件—原片段关系与固定ID容量选择，不调用分组或匹配。"
        "A1在共同原子全集上匹配，每事件最多一次、每候选最多3次。"
        "两臂均按图SHA去重、固定权重、同一原图Bank分数聚合。\n\n"
        "Oracle实现已用64种小图的独立穷举核验最大匹配与固定ID并列选择；"
        "它从完整共同全集读取真实支持标签，不局限于A1候选边。"
        "7项合成测试不算人评或实验效果。当前真实A2、身份F1、"
        "独立阈值与FPR/TPR仍缺；后续须源文档隔离、完整再校准及新确认。\n")
    atomic(out / "TERMINAL.json", {"status": result["status"], "result_sha256": sha(out / "RESULT.json")})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--encoder", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.encoder)
