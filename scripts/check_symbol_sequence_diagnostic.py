"""Independent scalar replay and integrity check of diagnostic artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_edit(qs, candidate, mode):
    matrix = [[0.0] * (len(candidate) + 1) for _ in range(len(qs) + 1)]
    for i in range(len(qs) + 1):
        matrix[i][0] = float(i)
    for j in range(len(candidate) + 1):
        matrix[0][j] = float(j)
    for i, q in enumerate(qs, 1):
        if q is None:
            penalties = [1.0] * 4
        elif mode == "soft":
            penalties = [-math.log(max(float(x), 1e-6)) / math.log(4) for x in q]
        else:
            maximum = max(q)
            winners = [a for a in range(4) if abs(q[a] - maximum) <= 1e-12]
            penalties = [0.0 if len(winners) == 1 and a == winners[0] else 1.0
                         for a in range(4)]
        for j, a in enumerate(candidate, 1):
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + penalties[int(a)])
    return matrix[-1][-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    source = read(out / "SOURCE_SHA256.json")
    assert all(sha(root / p) == h for p, h in source.items())
    output = read(out / "OUTPUT_SHA256.json")
    for path, value in output.items():
        if path != "report_script_sha256":
            assert sha(out / path) == value
    keys = read(out / "private/ORDER_KEYS.json")
    data = np.load(out / "private/ORDER_COSTS.npz", allow_pickle=False)
    costs, candidates = data["costs"], data["candidates"]
    seal = read(out / "PREDICTION_SEAL.json")
    assert sha(out / "private/ORDER_COSTS.npz") == seal["cost_sha256"]
    assert sha(out / "private/ORDER_KEYS.json") == seal["key_sha256"]
    assert candidates.shape == (2520, 8)
    assert len(set(map(tuple, candidates))) == 2520
    assert all(np.count_nonzero(candidates == a, axis=1).tolist() == [2] * 2520
               for a in range(4))
    predictions = read(out / "private/IDENTITY_ROWS.json")
    scalar_cases, maximum = 0, 0.0
    for index, key in enumerate(keys):
        method, condition, docid, arm, mode = key
        if method not in ("AMR_original_carrier", "C_original_carrier"):
            continue
        if arm != "positive" or mode == "unordered":
            continue
        rows = sorted([r for r in predictions if r["method"] == method
                       and r["condition"] == condition and r["doc_id"] == docid],
                      key=lambda r: r["slot"])
        assert len(rows) == 8
        replay = np.asarray([scalar_edit([r["q"] for r in rows], row, mode)
                             for row in candidates])
        error = float(np.max(np.abs(replay - costs[index])))
        maximum = max(maximum, error)
        assert error <= 1e-12
        scalar_cases += len(candidates)
    identity = read(out / "IDENTITY_RESULT.json")
    for population, methods in identity.items():
        for method, result in methods.items():
            assert result["planned"] == 96
            assert result["scored"] + result["erased"] == 96
            assert np.isclose(np.asarray(result["confusion_ABCD_erasure"]).sum(), 96)
    assert identity["readout"]["AMR_original_carrier"]["correct_credit"] == 35
    assert identity["readout"]["C_original_carrier"]["correct_credit"] == 57
    assert identity["readout"]["C_AI_mapped_sentence_mean"]["correct_credit"] == 38
    assert read(out / "TEMPLATE_SELF_READ.json")["metrics"]["correct_credit"] == 24
    perdoc = read(out / "ORDER_PER_DOCUMENT.json")
    for row in perdoc:
        if row["mode"] == "unordered":
            assert row["pairwise_win_rate"] == 0.5
            assert row["average_rank"] == 1260.5
    for path in (out / "REPORT.md", root / "experiments/semantic_sequence_stage1_prompt_repair1/PROMPT_REPAIR.md"):
        import re
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target:
                assert (path.parent / target).exists(), target
    result = {
        "status": "PASS", "independent_scalar_alignment_cases": scalar_cases,
        "maximum_cost_error": maximum, "source_sha_checks": len(source),
        "output_sha_checks": len(output) - 1, "all_planned_identity_denominators": 96,
        "all_unordered_candidates_tie": True, "template_self_read": "24/24",
        "local_links_exist": True, "audit_script_sha256": sha(__file__),
        "no_models_loaded_no_new_inference": True,
    }
    (out / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
