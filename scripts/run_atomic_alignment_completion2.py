"""Full frozen 12-source development cohort, shared 24-candidate atomic arms."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_multiscale_transfer as rt
from semantic_fact_raw_amr import audit_raw, replay_signature
from src.core import atomic_alignment_arms as arms
from src.core.multiview_candidates import candidate_windows, select_budget, config_dict
from src.core.multiview_views import Event, extract_events
from src.core.sqlite_cache import text_sha256

DEFAULT = ROOT / "experiments/swan_atomic_alignment_completion2"
SOURCE = ROOT / "experiments/swan_multiview_development1"
CONDITIONS = ("baseline", "boundary_only_candidate_0", "semicolon_all",
              "semicolon_pairs", "comma_all", "newline_all", "space_all")
CODE = (
    "scripts/run_atomic_alignment_completion2.py", "scripts/run_multiscale_transfer.py",
    "scripts/semantic_fact_raw_amr.py", "scripts/run_research_completion_queue.py",
    "scripts/collect_atomic_alignment_oracle1.py", "scripts/collect_atomic_alignment_oracle2.py",
    "scripts/swan_atomic_calibration1.py", "scripts/test_atomic_completion2.py",
    "scripts/test_atomic_alignment_arms.py", "scripts/test_atomic_runtime2.py",
)


def immutable(path, value):
    if path.exists():
        if rt.read(path) != value:
            raise RuntimeError("Immutable artifact changed: " + path.name)
    else:
        rt.atomic(path, value)


def inputs(rows):
    """Discard all historical detector/positive/target metadata before runtime."""
    output, metadata = [], []
    keys = set()
    for row in rows:
        if rt.digest(row["text"]) != row["text_sha256"]:
            raise ValueError("Literal source identity changed")
        key = row["group_id"], row["variant_id"]
        if key in keys or row["variant_id"] not in CONDITIONS:
            raise ValueError("Duplicate source-condition or foreign condition")
        keys.add(key)
        pool = candidate_windows(row["text"])
        candidates = select_budget(pool["candidates"], 24)
        output.append({"row_id": row["row_id"], "text": row["text"],
                       "text_sha256": row["text_sha256"], "candidates": candidates,
                       "pool_count": pool["enumerated_count"],
                       "pool_truncated_count": pool["truncated_count"]})
        metadata.append({k: row[k] for k in ("row_id", "group_id", "variant_id", "cohort")})
    groups = {group for group, _ in keys}
    if keys != {(g, c) for g in groups for c in CONDITIONS}:
        raise ValueError("Incomplete source-condition product")
    if len({r["row_id"] for r in output}) != len(output):
        raise ValueError("Duplicate row ID")
    return output, metadata


def prepare(out, encoder):
    if out.exists():
        raise RuntimeError("New immutable successor directory required")
    manifest = rt.read(SOURCE / "pipeline3/MANIFEST.json")
    original = SOURCE / "inputs/development_rows.jsonl"
    if rt.sha(original) != manifest["input_sha256"]:
        raise RuntimeError("Original source manifest differs")
    rows, meta = inputs([json.loads(line) for line in original.read_text().splitlines() if line])
    if len(rows) != 84 or len({r["group_id"] for r in meta}) != 12:
        raise RuntimeError("Expected original full 12-source, 84-row cohort")
    for name, expected in manifest["encoder_assets_sha256"].items():
        if rt.sha(encoder / name) != expected:
            raise RuntimeError("Original MiniLM asset changed: " + name)
    private = out / "private"
    private.mkdir(parents=True, mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    rt.atomic(private / "INPUTS.json", rows)
    rt.atomic(private / "EVALUATION_METADATA.json", meta)
    bank = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
    if rt.sha(bank) != manifest["bank_sha256"]:
        raise RuntimeError("Original Bank identity differs")
    shutil.copyfile(bank, private / "BANK.json")
    for name in manifest["encoder_assets_sha256"]:
        path = private / "encoder" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(encoder / name, path)
    spec = importlib.util.find_spec("en_core_web_sm")
    model_root = Path(next(iter(spec.submodule_search_locations)))
    models = list(model_root.glob("en_core_web_sm-*"))
    if len(models) != 1:
        raise RuntimeError("Unique installed spaCy model required")
    shutil.copytree(models[0], private / "surface_model")
    all_code = list(CODE) + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").glob("*.py"))]
    all_code += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src/core").glob("*.py"))]
    for name in all_code:
        path = private / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    transfer = ROOT / "第二篇_语义消息写入与读回/02_实验/semantic_sequence_stage1_run2/multiscale_transfer4_recovery1"
    parser = rt.read(transfer / "FREEZE.json")["parser"]
    unique = {c["text"] for r in rows for c in r["candidates"]}
    protocol = """# W05 全来源原子对应开发补全 v2

固定原12来源×7条件84行，来源全部17–48词。它是历史短文本开发补全，
不是原方案的正常长度、独立正负确认。保留旧7行、0预算开发结果和SHA。
本版本候选按同一几何生成并在解析前限24，所有臂共用候选、解析、
Bank100、MiniLM、表层线索和按图正文SHA去重的固定聚合。
A0只原片段，A1原子容量匹配，A2需两位独立真人标全共同对应全集，
分歧由绑定原答卷SHA的第三人裁决。A2不读取A1筛选边或Bank分数。
全部无效图、候选和擦除保留分母。解析固定FP32 BART4束、原生1024；
beam0、无绑定日志、有效roundtrip、无截断才可用，未知日志停止本版本。
这是各臂共用的更严格解析合同，不冒称逐字复现旧pipeline3准入。
MiniLM文本或事件描述超原生256时共同擦除，原观测及原因保留；
不裁剪描述，不新增上下文。相同图只计一次，不把对应当事实身份真值。
先封A0/A1与所有单视图家族分数，再读取条件元信息生成开发汇总。
单视图家族不按测试选最优；正式强对照仍需独立开发与封存。
新缓存，单GPU进程，原子逐图/逐行恢复；GPU压力/OOM/磁盘不足可暂停。
不进行生成、不读取真人答案；缺失标签不当负例、不填Oracle FPR为0。

独立确认接口已实现：每臂分别按源文档跨预定条件的最大Null统计量校准，
严格大于有限样本顺序统计量才拒绝；相同来源只贡献一次校准分母。
独立Null按条件及human/LLM分别报告CP区间和Bonferroni同时上界，
正例保留生成失败分母，A1−A0/A1−预冻结强单视图按来源配对bootstrap。
该接口需另冻结正常长度来源、实际三臂分数、真人标签与开发功效决定；
本84行不能填入其正式确认结果。Oracle FPR只作条件诊断。
"""
    (out / "PROTOCOL.md").write_text(protocol)
    rt.atomic(out / "FREEZE.json", {
        "version": "atomic-alignment-full-development/v2", "source_manifest_sha256": rt.sha(SOURCE / "pipeline3/MANIFEST.json"),
        "original_input_sha256": rt.sha(original), "parser": parser,
        "code_sha256": {p: rt.sha(private / "code" / p) for p in all_code},
        "files": {str(p.relative_to(out)): rt.sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name != ".gitignore"},
        "counts": {"independent_sources": 12, "rows": 84, "conditions": 7,
                   "candidate_occurrences": sum(len(r["candidates"]) for r in rows),
                   "unique_parse_texts": len(unique)},
        "candidate_budget": 24, "candidate_config": config_dict(),
        "execution": {"resumable_gpu_pressure": True, "resource_pause_exit_code": 75},
        "resource_gate": {"vram_fraction_max": .8},
        "native_encoder_context_policy": "shared_erasure_keep_raw_event",
        "normal_length_confirmation": False, "independent_calibration": False,
    })
    return verify(out)


def verify(out):
    f = rt.read(out / "FREEZE.json")
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Frozen file drift: " + name)
    return f


def native_erasure(text, freeze_sha, tokens):
    return {"amr": None, "amr_sha256": None, "valid": False, "case_id": rt.digest(text),
            "text_sha256": rt.digest(text), "token_length": tokens, "events": [],
            "attempts": [], "decode_events": [], "beam_lengths": [], "clipped": False,
            "seconds": 0., "freeze_sha256": freeze_sha, "reason": "native_input_limit"}


def raw_input(text):
    return {"case_id": rt.digest(text), "input_text_sha256": rt.digest(text)}


def raw_graphs(out, assets, f, rows):
    seal_path = out / "RAW_GRAPH_SEAL.json"
    if seal_path.exists():
        seal = rt.read(seal_path)
        if seal["freeze_sha256"] != rt.sha(out / "FREEZE.json"):
            raise RuntimeError("Raw graph freeze mismatch")
        for name, expected in seal["files"].items():
            if rt.sha(out / name) != expected:
                raise RuntimeError("Sealed raw graph changed")
        return
    parser = rt.Parser(assets, out, f)
    texts = sorted({c["text"] for row in rows for c in row["candidates"]}, key=rt.digest)
    precheck = []
    for text in texts[:2]:
        pairs = [{**parser.fresh(text), "case_id": rt.digest(text)} for _ in range(2)]
        if replay_signature(pairs[0]) != replay_signature(pairs[1]):
            raise RuntimeError("Raw parser replay differs")
        precheck.append({"text_sha256": rt.digest(text), "exact_two_replay": True})
    rt.atomic(out / "GPU_PRECHECK.json", {"passed": True, "replays": precheck, "dtype": "float32"})
    for index, text in enumerate(texts):
        if shutil.disk_usage(out).free < 512 * 1024**2:
            raise rt.ResourcePause("Insufficient disk headroom")
        path = out / "private/raw_graphs" / (rt.digest(text) + ".json")
        if path.exists():
            record = rt.read(path)
        else:
            tokens = len(parser.model.tokenizer(text, truncation=False)["input_ids"])
            record = (native_erasure(text, parser.freeze_sha, tokens) if tokens >= parser.model.max_sent_len
                      else {**parser.fresh(text), "case_id": rt.digest(text)})
            audit_raw(record, raw_input(text), parser.freeze_sha)
            immutable(path, record)
        audit_raw(record, raw_input(text), parser.freeze_sha)
        rt.atomic(out / "HEARTBEAT.json", {
            "phase": "raw_graphs", "completed": index + 1, "total": len(texts), "at": rt.now()})
    immutable(seal_path, {"freeze_sha256": parser.freeze_sha,
                         "files": {str(p.relative_to(out)): rt.sha(p)
                                   for p in sorted((out / "private/raw_graphs").glob("*.json"))}})
    del parser
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Encoder:
    def __init__(self, path):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        torch.manual_seed(20261102)
        torch.use_deterministic_algorithms(True)
        self.model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
        self.model.eval()
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise RuntimeError("Encoder must remain FP32")
        self.np, self.memo = np, {}

    def fits(self, text):
        return len(self.model.tokenizer(text, truncation=False)["input_ids"]) <= self.model.max_seq_length

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if any(not self.fits(t) for t in missing):
            raise RuntimeError("Native encoder overflow reached vectorization")
        if missing:
            vectors = self.model.encode(missing, batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False)
            # Each batch's first input is replayed, retaining the maximum error.
            again = self.model.encode(missing[:1], batch_size=32, normalize_embeddings=True,
                                      show_progress_bar=False)
            error = float(self.np.max(self.np.abs(vectors[:1] - again)))
            if error > 1e-6 or not self.np.isfinite(vectors).all():
                raise RuntimeError("Encoder deterministic replay failed")
            self.memo.update(zip(missing, vectors))
        return (self.np.stack([self.memo[t] for t in texts]) if texts else
                self.np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=self.np.float32))


def observe(row, out, encoder, surface, scorer, graph_store=None):
    from src.core.multiview_detector import full_bank_scores
    observations, all_events, events, exclusions = [], [], [], []
    for index, candidate in enumerate(row["candidates"]):
        if (row["text"][candidate["start_char"]:candidate["end_char"]] != candidate["text"]
                or text_sha256(candidate["text"]) != candidate["span_sha256"]):
            raise RuntimeError("Candidate offsets or identity differ")
        if graph_store is None:
            raw_path = out / "private/raw_graphs" / (rt.digest(candidate["text"]) + ".json")
            raw = rt.read(raw_path)
            raw_sha = rt.sha(raw_path)
            source_freeze_sha = rt.sha(out / "FREEZE.json")
        else:
            raw, raw_sha = graph_store.load(candidate["text"])
            source_freeze_sha = graph_store.freeze_sha256
        audit = audit_raw(raw, raw_input(candidate["text"]), source_freeze_sha)
        reasons = list(audit["exclusion_reasons"])
        if not encoder.fits(candidate["text"]):
            reasons.append("native_candidate_encoder_context")
        extracted = extract_events(raw["amr"], index) if audit["structurally_eligible"] else []
        for event in extracted:
            all_events.append(event.to_dict())
            if reasons or not encoder.fits(event.description):
                exclusions.append({"event_id": event.event_id,
                                   "reason": "unusable_origin" if reasons else "native_event_encoder_context"})
            else:
                events.append(event)
        observations.append({
            "candidate_index": index, **candidate, "amr": raw["amr"],
            "graph_body_sha256": rt.digest("\n".join(
                line for line in (raw["amr"] or "").splitlines() if not line.startswith("#"))),
            "usable": not reasons, "unusable_reasons": reasons,
            "raw_graph_sha256": raw_sha,
            "bank_scores": [],  # Populated only after alignment has been computed.
        })
    candidates = [{k: obs[k] for k in arms.CANDIDATE_FIELDS} for obs in observations]
    surfaces = surface.analyze_many([c["text"] for c in row["candidates"]])
    # Unusable candidates receive a finite zero vector. Every surviving event
    # has an encodable origin; all arms use the exact same post-erasure universe.
    tv = encoder.np.zeros((len(candidates), encoder.model.get_sentence_embedding_dimension()),
                          dtype=encoder.np.float32)
    indices = [i for i, c in enumerate(candidates) if c["usable"]]
    if indices:
        tv[indices] = encoder.encode([observations[i]["text"] for i in indices])
    ev = encoder.encode([e.description for e in events])
    a0 = arms.no_alignment(candidates, events)
    a1 = arms.predicted(candidates, events, tv, ev, surfaces)
    if a1 != arms.predicted(candidates, events, tv, ev, surfaces):
        raise RuntimeError("Atomic A1 replay differs")
    for obs in observations:
        if obs["usable"]:
            obs["bank_scores"] = full_bank_scores(scorer, obs["amr"])
    baselines = {}
    for kind in ("whole_text", "surface_boundary", "word_window_12", "word_window_24", "word_window_48"):
        selected = [p for p in a0["selected"] if kind in observations[p["event_candidate_index"]]["kinds"]]
        baselines[kind] = arms.fixed_statistic(selected, observations)
    return {
        "freeze_sha256": rt.sha(out / "FREEZE.json"), "row_id": row["row_id"],
        "text_sha256": row["text_sha256"], "candidates": candidates, "events": [e.to_dict() for e in events],
        "surfaces": surfaces, "all_extracted_events": all_events, "event_exclusions": exclusions,
        "observations": observations, "universe": arms.universe(candidates, events),
        "A0": {**a0, **arms.fixed_statistic(a0["selected"], observations)},
        "A1": {**a1, **arms.fixed_statistic(a1["selected"], observations)},
        "single_view_families": baselines, "A2": None,
    }


def score(out, graph_store=None):
    from src.core.bank_scorer import SwanAMRBankScorer
    from src.core.multiview_views import SurfaceAnalyzer
    import spacy
    f = verify(out)
    rows = rt.read(out / "private/INPUTS.json")
    encoder = Encoder(out / "private/encoder")
    surface = object.__new__(SurfaceAnalyzer)
    surface.nlp = spacy.load(out / "private/surface_model")
    # This scorer never calls its extractor when score_amr is used.
    class FrozenExtractor:
        fingerprint = "presealed-raw-graphs-only"
    scorer = SwanAMRBankScorer(FrozenExtractor(), str(out / "private/BANK.json"))
    results = []
    for index, row in enumerate(rows):
        path = out / "private/rows" / (rt.digest(row["row_id"]) + ".json")
        receipt = path.with_suffix(".sha.json")
        if path.exists():
            result = rt.read(path)
            if (not receipt.exists() or rt.read(receipt)["sha256"] != rt.sha(path)
                    or result["freeze_sha256"] != rt.sha(out / "FREEZE.json")
                    or result["row_id"] != row["row_id"]):
                # A crash between result and receipt needs deterministic replay.
                if receipt.exists() or observe(row, out, encoder, surface, scorer, graph_store) != result:
                    raise RuntimeError("Row checkpoint identity/replay differs")
                immutable(receipt, {"sha256": rt.sha(path)})
        else:
            result = observe(row, out, encoder, surface, scorer, graph_store)
            immutable(path, result)
            immutable(receipt, {"sha256": rt.sha(path)})
        results.append(result)
        rt.atomic(out / "HEARTBEAT.json", {
            "phase": "atomic_predictions", "completed": index + 1, "total": len(rows), "at": rt.now()})
    blind, scores, predictions = [], [], []
    for row, result in zip(rows, results):
        blind.append({
            "row_id": row["row_id"], "text": row["text"], "text_sha256": row["text_sha256"],
            **{k: result[k] for k in ("candidates", "events", "surfaces",
                                     "all_extracted_events", "event_exclusions")},
            "candidate_texts": [c["text"] for c in row["candidates"]],
            "candidate_amrs": [o["amr"] for o in result["observations"]],
        })
        scores.append({k: result[k] for k in ("row_id", "universe", "observations")})
        predictions.append({k: result[k] for k in ("row_id", "A0", "A1", "single_view_families")})
    for name, value in (("BLIND_INPUTS.json", blind), ("POST_SELECTION_SCORES.json", scores),
                        ("PREDICTIONS.json", predictions)):
        immutable(out / "private" / name, value)
    immutable(out / "OBSERVATION_SEAL.json", {
        "freeze_sha256": rt.sha(out / "FREEZE.json"),
        "blind_inputs_sha256": rt.sha(out / "private/BLIND_INPUTS.json"),
        "post_selection_scores_sha256": rt.sha(out / "private/POST_SELECTION_SCORES.json")})
    previous_seal = (rt.read(out / "PREDICTION_SEAL.json")
                     if (out / "PREDICTION_SEAL.json").exists() else {})
    immutable(out / "PREDICTION_SEAL.json", {
        "freeze_sha256": rt.sha(out / "FREEZE.json"),
        "predictions_sha256": rt.sha(out / "private/PREDICTIONS.json"),
        "observation_seal_sha256": rt.sha(out / "OBSERVATION_SEAL.json"),
        "human_answers_read": False, "Bank_scores_in_alignment": False,
        "at": previous_seal.get("at", rt.now()),
    })
    review_rows = [{"row_id": r["row_id"], "pairs": [
        {**pair, "same_fact_supported": None, "rationale": ""} for pair in r["universe"]]}
        for r in results]
    for reviewer in (1, 2):
        immutable(out / "private/human_review" / f"reviewer{reviewer}_UNFILLED.json", {
            "reviewer_code": "", "human_attestation": False,
            "independent_attestation": False, "no_model_or_target_answers_used": False,
            "completed_at": "", "freeze_sha256": rt.sha(out / "FREEZE.json"),
            "prediction_seal_sha256": rt.sha(out / "PREDICTION_SEAL.json"), "rows": review_rows})
    metadata = rt.read(out / "private/EVALUATION_METADATA.json")
    mapping = {r["row_id"]: r for r in metadata}
    summary = {
        "status": "FULL_SHORT_DEVELOPMENT_A0_A1_COMPLETE_A2_PENDING_HUMANS",
        "counts": f["counts"], "completed_rows": len(results),
        "full_common_pairs": sum(len(r["universe"]) for r in results),
        "eligible_candidates": sum(c["usable"] for r in results for c in r["candidates"]),
        "event_context_erasures": sum(len(r["event_exclusions"]) for r in results),
        "rows": [{
            "row_id": r["row_id"], "condition": mapping[r["row_id"]]["variant_id"],
            "source_id": mapping[r["row_id"]]["group_id"],
            "A0": r["A0"]["statistic"], "A1": r["A1"]["statistic"], "A2": None,
        } for r in results],
        "formal_FPR": None, "formal_TPR": None, "identity_F1": None,
        "independent_human_reviews": 0, "normal_length_confirmation": False,
    }
    if graph_store is not None:
        immutable(out / "SOURCE_AUDIT_AFTER.json", graph_store.verify())
    immutable(out / "RESULT.json", summary)
    immutable(out / "AUDIT.json", {
        "full_source_condition_product": len(results) == 84,
        "prediction_seal_before_metadata": True, "human_labels_read": False,
        "source_SHA_verified": True, "candidate_budget": 24,
        "A1_exact_replay_each_new_row": True,
        "raw_graph_seal_sha256": (rt.sha(out / "RAW_GRAPH_SEAL.json") if graph_store is None
                                  else graph_store.seal_sha256),
        "completed_three_arm_scientific_test": False,
    })
    immutable(out / "TERMINAL.json", {"status": summary["status"],
                                     "result_sha256": rt.sha(out / "RESULT.json"),
                                     "audit_sha256": rt.sha(out / "AUDIT.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "verify", "run"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--encoder", type=Path)
    ap.add_argument("--assets", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.out, args.encoder)
    elif args.action == "verify":
        verify(args.out)
    else:
        f = verify(args.out)
        if (args.out / "HALT.json").exists():
            raise RuntimeError("Prior integrity halt cannot be silently resumed")
        if (args.out / "TERMINAL.json").exists():
            terminal = rt.read(args.out / "TERMINAL.json")
            if (terminal["result_sha256"] != rt.sha(args.out / "RESULT.json")
                    or terminal["audit_sha256"] != rt.sha(args.out / "AUDIT.json")):
                raise RuntimeError("Terminal artifact changed")
            return
        with (args.out / "private/worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.monotonic()
            try:
                raw_graphs(args.out, args.assets, f, rt.read(args.out / "private/INPUTS.json"))
                score(args.out)
            except Exception as exc:
                pause = isinstance(exc, rt.ResourcePause) or type(exc).__name__ == "OutOfMemoryError"
                rt.atomic(args.out / ("PAUSED.json" if pause else "HALT.json"), {
                    "at": rt.now(), "type": type(exc).__name__, "resume_allowed": pause,
                    "scientific_result": None})
                if pause:
                    sys.exit(75)
                raise
            finally:
                rt.atomic(args.out / "private/sessions" / (str(time.time_ns()) + ".json"),
                          {"elapsed_seconds": time.monotonic() - started, "at": rt.now()})
    print(json.dumps({"action": args.action, "ok": True}))


if __name__ == "__main__":
    main()
