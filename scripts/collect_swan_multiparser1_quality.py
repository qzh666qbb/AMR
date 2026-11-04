"""Bind real parent quality returns to the three frozen parser flows."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_multiscale_transfer as rt
from swan_final1_common import freeze
from swan_multiparser1_core import FLOWS
from swan_independent_endpoint1_core import rate
from collect_swan_independent_endpoint1_quality import resolve


def metrics(attacks, parent_selections, mapping, quality, counts=None):
    counts = counts or {"pilot": 12, "main": 60}
    labels = {(v["doc_id"], v["candidate_id"]): quality[k] for k, v in mapping.items()}
    parents = {}
    for row in parent_selections:
        if row["arm"] != "a":
            continue
        key = tuple(row[k] for k in ("source_id", "role", "oracle", "mode", "budget"))
        if key in parents:
            raise ValueError("Duplicate parent attack")
        parents[key] = row
    keys = [tuple(r[k] for k in ("source_id", "role", "oracle", "mode", "budget")) for r in attacks]
    if len(set(keys)) != len(keys) or set(keys) != set(parents):
        raise ValueError("Child/parent attack product differs")
    for r, k in zip(attacks, keys):
        p = parents[k]
        if p["selection"]["selected"] != r["candidate_id"]:
            raise ValueError("Selected candidate identity changed")
        if set(r["clean_detected"]) != set(FLOWS) or set(r["attack_detected"]) != set(FLOWS):
            raise ValueError("All three flows required")
        if not all(type(v) is bool for field in ("clean_detected", "attack_detected") for v in r[field].values()):
            raise ValueError("Strict detection labels required")
        for cid in (None, r["candidate_id"]):
            if (r["source_id"] + "-a", cid) not in labels:
                raise ValueError("Missing true quality label")
    result = []
    for role, n in counts.items():
        for oracle in ("public", "a"):
            for mode in ("none", "label", "score"):
                for budget in (1, 4, 16):
                    subset = [r for r in attacks if (r["role"], r["oracle"], r["mode"], r["budget"])
                              == (role, oracle, mode, budget)]
                    if len(subset) != n or len({r["source_id"] for r in subset}) != n:
                        raise ValueError("Full frozen source denominator required")
                    def clean_good(r):
                        return labels[r["source_id"] + "-a", None]
                    def rewrite_good(r):
                        return labels[r["source_id"] + "-a", r["candidate_id"]]
                    for flow in FLOWS:
                        eligible = [r for r in subset if clean_good(r) and r["clean_detected"][flow]]
                        escaped = [r for r in eligible if rewrite_good(r) and not r["attack_detected"][flow]]
                        proxy = []
                        if oracle == "public":
                            for r in eligible:
                                p = parents[tuple(r[k] for k in ("source_id", "role", "oracle", "mode", "budget"))]
                                if p["clean_detected"]["public"] and not p["attack_detected"]["public"] and rewrite_good(r):
                                    proxy.append(r)
                        result.append({
                            "role": role, "flow": flow, "oracle": oracle, "mode": mode, "budget": budget,
                            "quality_and_clean_detection_over_all_sources": rate(len(eligible), n),
                            "ASR": rate(len(escaped), len(eligible)),
                            "ATR": rate(sum(not r["attack_detected"][flow] for r in proxy), len(proxy))
                            if oracle == "public" else None,
                            "quality_preserving_escape_over_all_sources": rate(len(escaped), n),
                            "rewrite_quality_failures_in_eligible": sum(not rewrite_good(r) for r in eligible)})
    return result


def terminal(out):
    t = rt.read(out / "TERMINAL.json")
    if any(t[k] != rt.sha(out / name) for k, name in (
            ("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json"))):
        raise RuntimeError("Machine terminal changed")


def collect(out, parent, first, second, destination, adjudication=None):
    from run_swan_independent_endpoint1 import check_seal, read_row
    from run_swan_multiparser1 import summarize
    terminal(out)
    terminal(parent)
    check_seal(parent, "SELECTIONS_COMPLETE.json")
    f = rt.read(out / "FREEZE.json")
    if f["parents"]["endpoint"] != rt.sha(parent / "FREEZE.json"):
        raise RuntimeError("Wrong parent endpoint")
    for name, expected in f["files"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Multiparser frozen dependency changed")
    audit = rt.read(out / "AUDIT.json")
    for name, expected in audit["seals"].items():
        if rt.sha(out / name) != expected:
            raise RuntimeError("Multiparser scoring seal changed")
    for name in ("INPUT_BINDING.json", "RAW_AMRLIB_SEAL.json", "RAW_AMRBART_SEAL.json"):
        check_seal(out, name)
    check_seal(out, "SCORING_COMPLETE.json")
    # SELECTED_ATTACKS is a derived sidecar; replay it from sealed rows and
    # parent selections so it cannot silently detach from the machine result.
    rows = [read_row(p, {}) for p in sorted((out / "private/rows").glob("*.json"))]
    replay = summarize(out, parent, rows, rt.read(out / "THRESHOLDS.json"))
    if replay != rt.read(out / "RESULT.json"):
        raise RuntimeError("Multiparser machine summary replay differs")
    folder = parent / "private/quality"
    contract = rt.read(folder / "CONTRACT.json")
    if any(rt.sha(folder / name) != contract[key] for name, key in (
            ("BLIND_ITEMS.json", "items_sha256"), ("EVALUATOR_MAPPING.json", "mapping_sha256"))):
        raise RuntimeError("Parent quality packet changed")
    if contract["selection_seal_sha256"] != rt.sha(parent / "SELECTIONS_COMPLETE.json"):
        raise RuntimeError("Quality packet bound to different selections")
    labels, review_audit = resolve(rt.read(folder / "BLIND_ITEMS.json"), contract["items_sha256"],
                                   first, second, adjudication)
    if labels is None:
        return freeze(destination / "PENDING.json", review_audit)
    mapping = rt.read(folder / "EVALUATOR_MAPPING.json")
    if set(mapping) != set(labels):
        raise RuntimeError("Parent quality denominator changed")
    selections = [rt.read(p) for p in sorted((parent / "private/selections").glob("*.json"))]
    values = metrics(rt.read(out / "private/SELECTED_ATTACKS.json"), selections, mapping, labels)
    freeze(destination / "RESULT.json", {
        "status": "MULTIPARSER_SELECTED_ATTACK_QUALITY_COMPLETE", "metrics": values,
        "scope": "same fixed parent attacks; not ensemble-adaptive search or parser-neutral training",
        "boundary_condition_quality": "not reviewed by parent rewrite packet",
        "human_identity_basis": "distinct declared reviewer IDs; not independently authenticated"})
    freeze(destination / "AUDIT.json", {
        **review_audit, "machine_terminal_sha256": rt.sha(out / "TERMINAL.json"),
        "parent_terminal_sha256": rt.sha(parent / "TERMINAL.json"),
        "quality_contract_sha256": rt.sha(folder / "CONTRACT.json"),
        "input_sha256": [rt.sha(p) for p in (first, second, adjudication) if p is not None],
        "result_sha256": rt.sha(destination / "RESULT.json"),
        "source_to_parent_doc_mapping": "source_id + '-a'",
        "selected_attacks_replayed_from_sealed_rows": True,
        "code_sha256": rt.sha(Path(__file__))})
    return freeze(destination / "TERMINAL.json", {
        "status": "MULTIPARSER_SELECTED_ATTACK_QUALITY_COMPLETE",
        "result_sha256": rt.sha(destination / "RESULT.json"),
        "audit_sha256": rt.sha(destination / "AUDIT.json")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "parent", "first", "second", "destination"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--adjudication", type=Path)
    args = ap.parse_args()
    print(collect(args.out, args.parent, args.first, args.second, args.destination, args.adjudication))
