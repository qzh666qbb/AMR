"""Run the frozen, inference-free prerequisite for the 24+24 text pilot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import penman

from semantic_sequence_stage1_common import (
    BANK, DESIGN, POOL, ROOT, build_symbol_map, normalize_text, sha256, write_json,
)

BASE = ROOT / "experiments/swan_multiview_development1"
ATTACKS = ("baseline", "split", "merge", "space", "semicolon", "newline", "comma")
VIEWS = {
    "S12": ("dataset5_eval/FUSION_RECOVERY.json", "dataset5_paired_neg_eval/FUSION_RECOVERY.json", "vec_d1"),
    "E_max": ("dataset5_eval/FUSION_RECOVERY.json", "dataset5_paired_neg_eval/FUSION_RECOVERY.json", "vec_d2"),
    "S6": ("dataset5_eval/window6s1_pos", "dataset5_paired_neg_eval/window6s1_neg", "vec_d1"),
}


def top_credit(values, target, k=1):
    values = np.asarray(values, dtype=float)
    greater = int(np.sum(values > values[target] + 1e-12))
    equal = int(np.sum(np.abs(values - values[target]) <= 1e-12))
    return max(0.0, min(1.0, (k - greater) / equal))


def symbol_weights(values, groups, temperature):
    values = np.asarray(values, dtype=float)
    shifted = (values - np.max(values)) / temperature
    mass = np.exp(shifted)
    return np.asarray([mass[index].sum() for index in groups]) / mass.sum()


def load_rows(relative):
    path = BASE / relative
    files = [path] if path.is_file() else sorted(path.glob("s*/FUSION_RECOVERY.json"))
    records = {}
    for source in files:
        for row in json.loads(source.read_text())["per_instance"]:
            key = (row["doc"], row["slot"], row["attack"])
            if key in records:
                raise ValueError("Duplicate scoring key; no silent shard merge")
            records[key] = row
    return records


def build_universe(docs):
    counts = Counter(normalize_text(s["text"]) for d in docs for s in d["slots"])
    # Legacy scoring enumerates accepted carriers, whereas source slot IDs
    # retain gaps from failed generation attempts. The two namespaces differ.
    return [
        {
            "doc": d["document_id"], "slot": ordinal, "source_slot": s["slot"],
            "template_index": s["template_index"],
            "unique_text_weight": 1 / counts[normalize_text(s["text"])],
        }
        for d in docs for ordinal, s in enumerate(d["slots"])
    ]


def validate_row_identities(universe, rows):
    expected = {(u["doc"], u["slot"]): u["template_index"] for u in universe}
    for (doc, slot, attack), row in rows.items():
        if attack not in ATTACKS or (doc, slot) not in expected:
            raise ValueError("Unknown legacy scoring identity")
        if row.get("template_index") != expected[doc, slot]:
            raise ValueError("Template mismatch while joining legacy carrier scores")


def summarize(universe, rows, field, mapping, groups, attack, temperature, equal_text):
    class_support = np.zeros(4)
    confusion = np.zeros((4, 5))
    total_weight = top2 = top24 = top3_24 = nll = brier = 0.0
    high_count = high_error = erased_weight = 0.0
    erased_count = 0
    for item in universe:
        target = mapping[str(item["template_index"])]
        weight = item["unique_text_weight"] if equal_text else 1.0
        class_support[target] += weight
        total_weight += weight
        row = rows.get((item["doc"], item["slot"], attack))
        vector = None if row is None else row.get(field)
        valid = vector is not None and len(vector) == 100 and all(
            vector[t] is not None and math.isfinite(float(vector[t])) for t in POOL
        )
        if not valid:
            q = np.ones(4) / 4
            confusion[target, 4] += weight
            erased_weight += weight
            erased_count += 1
        else:
            values = np.asarray([vector[t] for t in POOL], dtype=float)
            q = symbol_weights(values, groups, temperature)
            best = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
            confusion[target, best] += weight / len(best)
            top2 += weight * top_credit(q, target, 2)
            index = POOL.index(item["template_index"])
            top24 += weight * top_credit(values, index, 1)
            top3_24 += weight * top_credit(values, index, 3)
            if q.max() >= 0.8:
                high_count += weight
                high_error += weight * (1.0 - top_credit(q, target))
        nll -= weight * math.log(max(float(q[target]), 1e-300))
        reference = np.zeros(4)
        reference[target] = 1
        brier += weight * float(np.square(q - reference).sum())
    recall = np.diag(confusion[:, :4]) / class_support
    return {
        "planned_carriers": len(universe),
        "effective_total_weight": total_weight,
        "erased_carriers": erased_count,
        "erased_weight": erased_weight,
        "class_support": class_support.tolist(),
        "class_recalls": recall.tolist(),
        "confusion_rows_true_ABCD_columns_pred_ABCD_erasure": confusion.tolist(),
        "balanced_accuracy": float(recall.mean()),
        "top2_coverage": top2 / total_weight,
        "template_top1": top24 / total_weight,
        "template_top3": top3_24 / total_weight,
        "nll": nll / total_weight,
        "brier": brier / total_weight,
        "high_confidence_coverage": high_count / total_weight,
        "high_confidence_error": high_error / high_count if high_count else None,
        "temperature": temperature,
        "weighting": "normalized_text_equal" if equal_text else "all_carrier_occurrences",
    }


def background_summary(universe, rows, field, groups):
    result = {}
    for attack in ATTACKS:
        counts = np.zeros(4)
        weights = np.zeros(4)
        erased = 0
        for item in universe:
            row = rows.get((item["doc"], item["slot"], attack))
            vec = None if row is None else row.get(field)
            if vec is None or len(vec) != 100 or any(
                vec[t] is None or not math.isfinite(float(vec[t])) for t in POOL
            ):
                erased += 1
                continue
            q = symbol_weights([vec[t] for t in POOL], groups, 0.1)
            best = np.flatnonzero(np.abs(q - q.max()) <= 1e-12)
            counts[best] += 1 / len(best)
            weights += q
        result[attack] = {
            "planned_carriers": len(universe),
            "erased": erased,
            "argmax_symbol_counts_ties_fractional": counts.tolist(),
            "mean_symbol_weights_valid_only": (
                (weights / (len(universe) - erased)).tolist()
                if erased < len(universe) else None
            ),
            "uses_paired_template_as_label": False,
        }
    return result


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new; existing run cannot be overwritten")
    audit = json.loads((DESIGN / "INPUT_AUDIT.json").read_text())
    for entry in audit["source_files"]:
        path = ROOT / entry["path"]
        if not path.exists() or sha256(path) != entry["sha256"]:
            raise ValueError("Design source SHA mismatch: " + entry["path"])
    symbol_map = build_symbol_map(json.loads(BANK.read_text()))
    write_json(output / "SYMBOL_MAP.json", symbol_map)
    freeze = {
        "schema": "semantic-sequence-stage1a-freeze/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(DESIGN / "PROTOCOL.md"),
        "input_audit_sha256": sha256(DESIGN / "INPUT_AUDIT.json"),
        "symbol_map_sha256": sha256(output / "SYMBOL_MAP.json"),
        "code_sha256": {
            name: sha256(ROOT / "scripts" / name)
            for name in ("semantic_sequence_stage1_common.py", "run_semantic_sequence_stage1a.py")
        },
        "python": sys.version,
        "numpy_version": np.__version__,
        "penman_version": penman.__version__,
        "primary_temperature": 0.1,
        "sensitivity_temperatures": [0.05, 0.2],
        "sources_verified": len(audit["source_files"]),
        "model_inference_calls": 0,
        "provider_calls": 0,
        "authorization": "User requested launch of the 24-positive + 24-unwatermarked new-text validation; this run performs its protocol prerequisite.",
    }
    write_json(output / "FREEZE_1A.json", freeze)
    if symbol_map["status"] != "READY":
        write_json(output / "STAGE1A_RESULTS.json", {
            "status": "DICTIONARY_NOT_READY",
            "new_text_generation_allowed": False,
            "provider_calls": 0,
            "cross_symbol_collisions": symbol_map["cross_symbol_collisions"],
        })
        return "DICTIONARY_NOT_READY"

    # Score files are first interpreted only after the symbol map is frozen.
    docs = json.loads((BASE / "dataset5/documents.json").read_text())["documents"]
    duplicate_counts = Counter(normalize_text(s["text"]) for d in docs for s in d["slots"])
    universe = build_universe(docs)
    write_json(output / "CARRIER_IDENTITY_MAP.json", {
        "scoring_slot_namespace": "ordinal_in_accepted_slots",
        "source_slot_namespace": "original_generation_slot_may_have_gaps",
        "carriers": universe,
    })
    mapping = {t: ord(a) - 65 for t, a in symbol_map["template_to_symbol"].items()}
    groups = [[i for i, t in enumerate(POOL) if mapping[str(t)] == a] for a in range(4)]
    conflicts = defaultdict(set)
    for d in docs:
        for s in d["slots"]:
            conflicts[normalize_text(s["text"])].add(mapping[str(s["template_index"])])
    support = {}
    for a in range(4):
        selected = [(d, s) for d in docs for s in d["slots"] if mapping[str(s["template_index"])] == a]
        support[chr(65 + a)] = {
            "carriers": len(selected),
            "unique_texts": len({normalize_text(s["text"]) for d, s in selected}),
            "documents": len({d["document_id"] for d, s in selected}),
        }
    result = {
        "schema": "semantic-sequence-stage1a-results/v1",
        "scope": "historical selected known-position diagnostic; not an independent test",
        "model_inference_calls": 0,
        "provider_calls": 0,
        "carriers": len(universe),
        "unique_texts": len(duplicate_counts),
        "cross_symbol_text_conflict_groups": sum(len(v) > 1 for v in conflicts.values()),
        "support": support,
        "views": {},
        "negative_symbol_distributions": {},
        "tie_policy": "fractional credit across scores equal within 1e-12; erasures get no top-k credit",
    }
    for name, (pos_path, neg_path, field) in VIEWS.items():
        rows = load_rows(pos_path)
        validate_row_identities(universe, rows)
        primary = {
            attack: {
                "all_carriers": summarize(universe, rows, field, mapping, groups, attack, 0.1, False),
                "unique_text_equal": summarize(universe, rows, field, mapping, groups, attack, 0.1, True),
            } for attack in ATTACKS
        }
        clean = primary["baseline"]["unique_text_equal"]["balanced_accuracy"]
        attacked = sum(primary[a]["unique_text_equal"]["balanced_accuracy"] for a in ("split", "merge", "space")) / 3
        result["views"][name] = {
            "primary": primary,
            "sensitivity": {
                str(temp): {
                    attack: summarize(universe, rows, field, mapping, groups, attack, temp, True)
                    for attack in ATTACKS
                } for temp in (0.05, 0.2)
            },
            "gate_clean_balanced_accuracy": clean,
            "gate_attack_mean_balanced_accuracy": attacked,
            "gate_pass": bool(clean >= 0.60 and attacked >= 0.45),
        }
        # Background metadata is never used as a class label.
        negative_rows = load_rows(neg_path)
        validate_row_identities(universe, negative_rows)
        result["negative_symbol_distributions"][name] = background_summary(
            universe, negative_rows, field, groups
        )
    passes = [name for name, info in result["views"].items() if info["gate_pass"]]
    result["status"] = "GO_TO_STAGE1B" if passes else "NO_GO_CURRENT_DICTIONARY"
    result["new_text_generation_allowed"] = bool(passes)
    result["passing_views"] = passes
    write_json(output / "STAGE1A_RESULTS.json", result)
    lines = [
        "# 第一阶段 1A：缓存筛查结果",
        "",
        f"状态：**{result['status']}**。模型推理与 API 调用均为 0。",
        "",
        "24 模板分为四类，每类 6 个；没有跨符号结构签名碰撞。字典先冻结，后读取评分。",
        "",
        "| 表示 | clean BA（不同文本等权） | split/merge/space 平均 BA | 通过 |",
        "|---|---:|---:|---|",
    ]
    for name, info in result["views"].items():
        lines.append(f"| {name} | {info['gate_clean_balanced_accuracy']:.2%} | {info['gate_attack_mean_balanced_accuracy']:.2%} | {'是' if info['gate_pass'] else '否'} |")
    lines.extend([
        "",
        f"分母：{len(universe)} 个载体、{len(duplicate_counts)} 个不同规范化文本；有 {result['cross_symbol_text_conflict_groups']} 个同文本跨符号标签冲突组，未删除。",
        "",
        "这是已观察、经筛选、已知载体位置的必要条件检查；不能证明盲恢复、抗复述、独立泛化或低 FPR。",
        "E_max 是旧缓存的多子图取最大；不等同于新实验中等比较次数的 E1。",
        "",
        "温度敏感性、逐攻击混淆矩阵、置信错误、模板身份与负例分布见 STAGE1A_RESULTS.json。",
    ])
    (output / "STAGE1A_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return result["status"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/semantic_sequence_stage1_run1")
    args = parser.parse_args()
    run(args.output)
