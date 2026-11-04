"""Validate independent post-selection quality labels and compute qualified endpoints."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_multiscale_transfer as rt
from swan_final1_common import freeze
from swan_independent_endpoint1_core import ARMS, BUDGETS, rate

FIELDS = ("natural", "coherent", "preserves_meaning")


def parse_labels(path, items, packet_sha):
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    wanted = {r["item_id"] for r in items}
    if len(records) != len(wanted) or {r["item_id"] for r in records} != wanted:
        raise ValueError("Incomplete or duplicate human labels")
    reviewers = {r["reviewer_id"].strip() for r in records}
    if len(reviewers) != 1 or "" in reviewers:
        raise ValueError("Exactly one named reviewer required")
    kinds = {r["item_id"]: r["kind"] for r in items}
    labels = {}
    for row in records:
        if row["packet_sha256"] != packet_sha:
            raise ValueError("Human packet binding differs")
        keys = FIELDS if kinds[row["item_id"]] == "rewrite" else FIELDS[:2]
        if any(row[k] not in ("true", "false") for k in keys):
            raise ValueError("Missing, unknown or non-boolean human judgment")
        labels[row["item_id"]] = {k: row[k] == "true" for k in keys}
    return next(iter(reviewers)), labels


def resolve(items, packet_sha, first, second, adjudication=None):
    rid1, labels1 = parse_labels(first, items, packet_sha)
    rid2, labels2 = parse_labels(second, items, packet_sha)
    if rid1 == rid2:
        raise ValueError("Two distinct independent reviewers required")
    disputed = [r for r in items if labels1[r["item_id"]] != labels2[r["item_id"]]]
    labels = {key: value for key, value in labels1.items() if value == labels2[key]}
    reviewers = [rid1, rid2]
    if disputed:
        if adjudication is None:
            return None, {"status": "ADJUDICATION_REQUIRED", "disputed_item_ids": [r["item_id"] for r in disputed]}
        rid3, third = parse_labels(adjudication, disputed, packet_sha)
        if rid3 in reviewers:
            raise ValueError("Adjudicator must be a third human")
        labels.update(third)
        reviewers.append(rid3)
    return {key: all(value.values()) for key, value in labels.items()}, {
        "status": "HUMAN_QUALITY_COMPLETE", "items": len(labels), "reviewers": reviewers,
        "adjudicated_items": len(disputed)}


def qualified_metrics(selections, mapping, quality):
    labels = {(value["doc_id"], value["candidate_id"]): quality[key] for key, value in mapping.items()}
    result = []
    for role in ("pilot", "main"):
        for arm in ARMS:
            for oracle in ("public", arm):
                for mode in ("none", "label", "score"):
                    for budget in BUDGETS:
                        subset = [r for r in selections if (r["role"], r["arm"], r["oracle"], r["mode"], r["budget"])
                                  == (role, arm, oracle, mode, budget)]
                        eligible = [r for r in subset if r["clean_detected"][arm]
                                    and labels[r["doc_id"], None]]
                        def retained(r):
                            return labels[r["doc_id"], r["selection"]["selected"]]
                        escaped = [r for r in eligible if retained(r) and not r["attack_detected"][arm]]
                        proxy = [r for r in eligible if r["clean_detected"]["public"]
                                 and retained(r) and not r["attack_detected"]["public"]]
                        result.append({"role": role, "arm": arm, "oracle": oracle, "mode": mode,
                            "budget": budget, "ASR": rate(len(escaped), len(eligible)),
                            "ATR": rate(sum(not r["attack_detected"][arm] for r in proxy), len(proxy))
                            if oracle == "public" else None,
                            "all_source_quality_preserving_escape": rate(len(escaped), len(subset)),
                            "quality_failures_in_eligible": sum(not retained(r) for r in eligible)})
    return result


def collect(out, first, second, destination, adjudication=None):
    from run_swan_independent_endpoint1 import check_seal
    check_seal(out, "SELECTIONS_COMPLETE.json")
    folder = out / "private/quality"
    contract = rt.read(folder / "CONTRACT.json")
    if (rt.sha(folder / "BLIND_ITEMS.json") != contract["items_sha256"]
            or rt.sha(folder / "EVALUATOR_MAPPING.json") != contract["mapping_sha256"]
            or rt.sha(out / "SELECTIONS_COMPLETE.json") != contract["selection_seal_sha256"]):
        raise RuntimeError("Review packet or attack selection drift")
    items = rt.read(folder / "BLIND_ITEMS.json")
    labels, audit = resolve(items, contract["items_sha256"], first, second, adjudication)
    if labels is None:
        freeze(destination / "AUDIT.json", audit)
        return audit
    mapping = rt.read(folder / "EVALUATOR_MAPPING.json")
    if set(mapping) != set(labels):
        raise RuntimeError("Quality mapping denominator changed")
    selections = [rt.read(p) for p in sorted((out / "private/selections").glob("*.json"))]
    result = qualified_metrics(selections, mapping, labels)
    freeze(destination / "RESULT.json", {"status": audit["status"], "metrics": result})
    freeze(destination / "AUDIT.json", {
        **audit, "quality_contract_sha256": rt.sha(folder / "CONTRACT.json"),
        "input_sha256": [rt.sha(p) for p in (first, second, adjudication) if p is not None],
        "result_sha256": rt.sha(destination / "RESULT.json"),
        "selection_was_not_changed": True})
    freeze(destination / "TERMINAL.json", {"status": audit["status"],
        "result_sha256": rt.sha(destination / "RESULT.json"),
        "audit_sha256": rt.sha(destination / "AUDIT.json")})
    return audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path)
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    print(collect(args.out, args.first, args.second, args.destination, args.adjudication))
