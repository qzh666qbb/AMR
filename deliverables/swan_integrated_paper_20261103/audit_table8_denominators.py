#!/usr/bin/env python3
"""Audit saved scores only; never load a detector or modify remote evidence.

Local use:
  python3 audit_table8_denominators.py --connection-file /path/to/ssh.txt \
      --output /path/to/table8-audit.json

Offline use with the original formal/<method> tree:
  python3 audit_table8_denominators.py --source-root /path/to/formal \
      --output /path/to/table8-audit.json

The connection file is read in memory and excluded from the audit output.
The worker runs through SSH stdin and creates no remote files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

METHODS = ("semstamp", "swordstamp", "samark", "alimark", "pmark")
TABLE_CONDITIONS = ("clean", "merge", "split")
REMOTE_ROOT = "/root/swan_pilot/pilot_generation/l3/formal"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_rows(rows, side, threshold, score_key="score"):
    expected = set(range(100)) if side == "pos" else set(range(100, 200))
    indices, hits, scores, n_zero = [], [], [], []
    bad_indices, bad_scores, bad_n = [], [], []
    for position, row in enumerate(rows):
        idx = row.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            bad_indices.append(position)
        else:
            indices.append(idx)
        score = row.get(score_key)
        if not finite(score):
            bad_scores.append({"row_position": position, "index": idx})
        else:
            scores.append(float(score))
            if score > threshold:
                hits.append(idx)
        n = row.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            bad_n.append({"row_position": position, "index": idx})
        elif n == 0:
            n_zero.append(idx)
    counts = Counter(indices)
    duplicate = {str(k): v for k, v in counts.items() if v > 1}
    missing, unexpected = sorted(expected - set(indices)), sorted(set(indices) - expected)
    # This mirrors the historical denominator only if every saved score is finite.
    rate = len(hits) / len(rows) if rows else None
    valid_rate = len(hits) / len(scores) if scores else None
    projected = [
        {"index": r.get("index"), "n": r.get("n"),
         "score": r.get(score_key) if finite(r.get(score_key)) else None}
        for r in rows
    ]
    return {
        "expected_count": 100,
        "row_count": len(rows),
        "unique_index_count": len(counts),
        "observed_indices": sorted(counts),
        "missing_indices": missing,
        "duplicate_indices": duplicate,
        "unexpected_indices": unexpected,
        "invalid_index_row_positions": bad_indices,
        "finite_score_count": len(scores),
        "nonfinite_score_rows": bad_scores,
        "invalid_unit_count_rows": bad_n,
        "zero_unit_indices": n_zero,
        "strict_exceedances": len(hits),
        "strict_exceedance_indices": hits,
        "rate_using_saved_row_denominator": rate,
        "rate_using_finite_score_denominator": valid_rate,
        "full_100_unique_finite": (
            len(rows) == 100 and not missing and not duplicate and not unexpected
            and not bad_indices and not bad_scores
        ),
        "numeric_projection_sha256": digest(
            json.dumps(projected, sort_keys=True, allow_nan=False).encode()
        ),
    }


def close(a, b):
    return finite(a) and finite(b) and abs(a - b) <= 1e-12


def score_map(rows, key):
    indices = [r["index"] for r in rows]
    if len(set(indices)) != len(indices):
        return None
    return {r["index"]: r[key] for r in rows}


def audit_root(root):
    root = Path(root)
    manifest, methods, checks = {}, {}, {}

    def read(relative):
        path = root / relative
        data = path.read_bytes()
        manifest[str(relative)] = {
            "sha256": digest(data), "bytes": len(data),
        }
        return json.loads(data)

    for method in METHODS:
        clean_path = Path(method) / ("detect/result.json" if method == "pmark" else "result.json")
        condition_path = Path(method) / "attacked_null.json"
        clean = read(clean_path)
        condition = read(condition_path)
        threshold = condition["clean_thr_5pct"]
        if not finite(threshold):
            raise ValueError("Nonfinite threshold: " + method)
        clean_threshold = clean["summary"]["thr_5pct"]
        item = {
            "clean_source": str(clean_path),
            "condition_source": str(condition_path),
            "threshold": threshold,
            "clean_threshold": clean_threshold,
            "threshold_exact_match": threshold == clean_threshold,
            "clean_result_rows": {},
            "conditions": {},
            "clean_score_comparison": {},
        }
        for side in ("pos", "null"):
            rows = clean[f"{side}_rows"]
            key = "z" if rows and "z" in rows[0] else "score"
            item["clean_result_rows"][side] = check_rows(rows, side, threshold, key)
            clean_scores = score_map(rows, key)
            condition_scores = score_map(condition["rows"]["clean"][side], "score")
            if clean_scores is None or condition_scores is None:
                comparison = {"comparable": False, "reason": "duplicate_index"}
            else:
                common = set(clean_scores) & set(condition_scores)
                comparison = {
                    "comparable": True,
                    "common_index_count": len(common),
                    "clean_result_only_indices": sorted(set(clean_scores) - set(condition_scores)),
                    "condition_clean_only_indices": sorted(set(condition_scores) - set(clean_scores)),
                    "score_mismatch_indices": sorted(
                        i for i in common if not close(clean_scores[i], condition_scores[i])
                    ),
                }
            item["clean_score_comparison"][side] = comparison
        for name, saved in condition["rows"].items():
            entry = {"table8_condition": name in TABLE_CONDITIONS}
            for side in ("pos", "null"):
                entry[side] = check_rows(saved[side], side, threshold)
                entry[side]["declared_row_count"] = saved["n_" + side]
                entry[side]["declared_row_count_matches"] = saved["n_" + side] == len(saved[side])
                rate_key = "tpr_vs_clean_thr" if side == "pos" else "fpr_vs_clean_thr"
                entry[side]["saved_rate"] = saved[rate_key]
                entry[side]["saved_rate_matches_recomputed"] = close(
                    saved[rate_key], entry[side]["rate_using_saved_row_denominator"]
                )
            item["conditions"][name] = entry
        item["unavailable_table8_conditions"] = [
            name for name in TABLE_CONDITIONS if name not in condition["rows"]
        ]
        item["missing_split_context"] = {}
        if "split" in item["conditions"]:
            for side in ("pos", "null"):
                clean_by_id = {r["index"]: r for r in condition["rows"]["clean"][side]}
                item["missing_split_context"][side] = [
                    {
                        "index": idx,
                        "clean_input_unit_count": clean_by_id.get(idx, {}).get("n"),
                        "below_split_constructor_four_sentence_minimum": (
                            isinstance(clean_by_id.get(idx, {}).get("n"), int)
                            and clean_by_id[idx]["n"] < 4
                        ),
                    }
                    for idx in item["conditions"]["split"][side]["missing_indices"]
                ]
        methods[method] = item

    builder_reference = []
    for relative in ("attacked_null.py", "../attacked_null.py"):
        path = root / relative
        if path.is_file():
            content = path.read_bytes()
            manifest[relative] = {"sha256": digest(content), "bytes": len(content)}
            code = content.decode()
            match = re.search(r"def split_longest\(sents\):\n(.*?)(?=\ndef )", code, re.S)
            builder_reference.append({
                "source": relative,
                "split_guard_fewer_than_four_sentences_returns_none": bool(
                    match and re.search(r"if len\(sents\) < 4:\s+return None", match[1])
                ),
            })
    for relative, record in manifest.items():
        record["unchanged_during_audit"] = digest((root / relative).read_bytes()) == record["sha256"]
    checks["all_sources_unchanged_during_read"] = all(
        x["unchanged_during_audit"] for x in manifest.values()
    )
    checks["all_clean_thresholds_match"] = all(m["threshold_exact_match"] for m in methods.values())
    checks["all_declared_counts_and_rates_match"] = all(
        side["declared_row_count_matches"] and side["saved_rate_matches_recomputed"]
        for m in methods.values() for condition in m["conditions"].values()
        for side in (condition["pos"], condition["null"])
    )
    return {
        "schema": "table8-denominator-audit-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "saved numeric score rows; no generation, parsing or detector calls",
        "source_root_label": "original_formal_100_tree",
        "expected_indices": {"pos": [0, 99], "null": [100, 199]},
        "sources": manifest, "checks": checks, "methods": methods,
        "condition_builder_reference": builder_reference,
    }


def remote_audit(connection_file, root):
    lines = [x.strip() for x in Path(connection_file).read_text().splitlines() if x.strip()]
    command = next(shlex.split(x) for x in lines if x.startswith("ssh "))
    if command[0] != "ssh" or "@" not in command[-1] or any(
        c in command[-1] for c in (" ", ";", "\n", "|")
    ):
        raise ValueError("Expected a saved SSH command ending in its destination")
    secret_lines = [x for x in lines if not x.startswith("ssh ")]
    env = os.environ.copy()
    if secret_lines:
        secret = re.sub(r"^(?:密码|password|pass|pwd)\s*[:：=]\s*", "", secret_lines[0], flags=re.I)
        env["SSHPASS"] = secret
        prefix = ["sshpass", "-e"]
        auth = ["-o", "NumberOfPasswordPrompts=1"]
    else:
        prefix, auth = [], ["-o", "BatchMode=yes"]
    remote = "python3 - --worker --source-root " + shlex.quote(root)
    args = prefix + ["ssh", "-o", "ConnectTimeout=10"] + auth + command[1:] + [remote]
    result = subprocess.run(
        args, input=Path(__file__).read_text(), capture_output=True, text=True,
        timeout=60, env=env,
    )
    if result.returncode:
        # Do not persist the SSH destination, authentication data or SSH stderr.
        raise RuntimeError(f"Read-only remote audit failed (exit {result.returncode})")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--connection-file")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(audit_root(args.source_root), allow_nan=False))
        return
    if not args.output:
        parser.error("--output is required")
    if args.connection_file:
        result = remote_audit(args.connection_file, args.source_root or REMOTE_ROOT)
    elif args.source_root:
        result = audit_root(args.source_root)
    else:
        parser.error("Provide --connection-file or --source-root")
    result["audit_script_sha256"] = digest(Path(__file__).read_bytes())
    builder_relative = "experiments/sentence_watermark_method_matrix_20261029/formal_100_20261029/attacked_null.py"
    builder = next(
        (parent / builder_relative for parent in Path(__file__).resolve().parents
         if (parent / builder_relative).is_file()),
        None,
    )
    if builder is not None:
        data = builder.read_bytes()
        code = data.decode()
        match = re.search(r"def split_longest\(sents\):\n(.*?)(?=\ndef )", code, re.S)
        result["local_condition_builder_reference"] = {
            "source": "experiments/sentence_watermark_method_matrix_20261029/formal_100_20261029/attacked_null.py",
            "sha256": digest(data),
            "split_guard_fewer_than_four_sentences_returns_none": bool(
                match and re.search(r"if len\(sents\) < 4:\s+return None", match[1])
            ),
            "scope": "local saved implementation; remote builder file not found at inspected candidate paths",
        }
    out = Path(args.output)
    if not out.parent.is_dir():
        parser.error("Output parent directory must already exist")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result["checks"], ensure_ascii=False))
    for method, data in result["methods"].items():
        for condition in TABLE_CONDITIONS:
            row = data["conditions"].get(condition)
            if row is None:
                print(method, condition, "not available")
                continue
            print(method, condition, json.dumps({
                side: {k: row[side][k] for k in (
                    "row_count", "unique_index_count", "finite_score_count",
                    "strict_exceedances", "missing_indices", "duplicate_indices",
                    "nonfinite_score_rows", "full_100_unique_finite",
                )} for side in ("pos", "null")
            }, ensure_ascii=False))
        print(method, "clean comparison", json.dumps(data["clean_score_comparison"]))
        if any(data["missing_split_context"].values()):
            print(method, "missing split context", json.dumps(data["missing_split_context"]))
    print("condition builder reference", json.dumps(result["condition_builder_reference"]))
    print("Wrote numeric audit:", out.name)


if __name__ == "__main__":
    main()
