"""Independent saved-artifact checks for the cycle1 write/read pilot."""
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re

import penman
from penman.models.amr import model as amr_model

from run_semantic_sequence_cycle import OUT, PRIVATE, ROOT, atomic, digest, sha


def main():
    out, private = OUT, PRIVATE
    freeze = json.loads((out / "FREEZE.json").read_text())
    parse_freeze = json.loads((out / "PARSER_FREEZE.json").read_text())
    result = json.loads((out / "WRITE_READ_RESULTS.json").read_text())
    checks = {}
    checks["generation_frozen_sources"] = all(sha(ROOT / path) == expected
                                             for path, expected in freeze["source_sha256"].items())
    checks["private_manifests"] = all(sha(private / path) == expected
                                     for path, expected in freeze["private_sha256"].items())
    checks["parser_source_snapshots"] = all(sha(private / "snapshots" / path) == expected
                                          for path, expected in parse_freeze["snapshots_sha256"].items())
    candidates = json.loads((out / "PERMUTATIONS.json").read_text())
    # Deliberately use a different, inefficient but small independent enumeration.
    expected = {"".join(p) for p in itertools.permutations("AABBCCDD")}
    checks["all_2520_arrangements"] = len(candidates) == len(set(candidates)) == 2520 and set(candidates) == expected
    checks["permutations_sha"] = sha(out / "PERMUTATIONS.json") == freeze["permutations_sha256"]
    checks["dictionary_sha"] = sha(out / "SYMBOL_MAP.json") == freeze["symbol_map_sha256"]
    checks["all_54_single_receipts"] = len(list((private / "calls").glob("*.intent.json"))) == 54
    for path in (private / "calls").glob("*.intent.json"):
        intent = json.loads(path.read_text())
        response = json.loads(path.with_name(path.name.replace(".intent.", ".response.")).read_text())
        canonical = (json.dumps(intent["body"], ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        assert hashlib.sha256(canonical).hexdigest() == intent["request_sha256"] == response["request_sha256"]
        assert response["complete"] and response["finish_reason"] == "stop"
    checks["receipt_hash_and_completion"] = True
    targets = json.loads((private / "TARGETS.json").read_text())
    assert len(targets) == 24 == len(result["rows"])
    successes = 0
    for target, reported in zip(targets, result["rows"]):
        assert target["input_id"] == reported["input_id"]
        path = private / "PARSES" / (target["input_id"] + ".json")
        record = json.loads(path.read_text())
        assert record["text_sha256"] == digest(record["text"])
        assert record["amr_sha256"] == digest(record["amr"])
        assert record["freeze_sha256"] == sha(out / "PARSER_FREEZE.json")
        assert reported["parse_sha256"] == sha(path)
        graph = penman.decode(record["amr"], model=amr_model)
        unknowns = len([t for t in graph.instances() if t.target == "amr-unknown"])
        named = [t.target for t in graph.edges() if t.role == ":name"]
        sizes = Counter(sum(a.source == n and re.fullmatch(r":op\d+", a.role) is not None
                            for a in graph.attributes()) for n in named)
        required = target["required_markers"]
        attributes = {(t.role, t.target) for t in graph.attributes()}
        audit = record["recovery_audit"]
        diagnostic = any(e["beam_index"] == audit["selected_beam"] for e in audit["events"])
        roundtrip = penman.decode(penman.encode(graph, model=amr_model), model=amr_model)
        strict = (graph.top in graph.variables() and bool(graph.instances())
                  and graph.top == roundtrip.top and sorted(graph.triples) == sorted(roundtrip.triples)
                  and not diagnostic and not record["decode_events"]
                  and audit["selected_beam"] == 0 and not record["output_limit_hit"])
        passed = (unknowns >= required["unknown_nodes"]
                  and all(sizes[k] >= n for k, n in Counter(required["name_sizes"]).items())
                  and (not required["imperative"] or (":mode", "imperative") in attributes)
                  and (not required["polite"] or (":polite", "+") in attributes)
                  and strict and target["generation_usable"])
        assert passed == reported["g1e_success"]
        assert strict == record["strict_valid"]
        successes += passed
    checks["all_parse_sha_and_independent_marker_recount"] = True
    checks["gate_recomputed"] = result["g1e_successes"] == successes and (
        (result["status"] == "G1E_GO_FORMAL_G1_PENDING") == (successes >= 20))
    checks["full_denominator"] = result["denominator"] == 24
    checks["no_formal_gate_inferred"] = all(result[k] == "NOT_EVALUATED" for k in
                                          ["formal_human_G1", "symbol_G2", "blind_G3", "order_G4"])
    checks["readout_untouched"] = {p.stem for p in (private / "documents").glob("*.json")} == {"d00", "d01", "d02"}
    complete = json.loads((out / "PARSE_COMPLETE.json").read_text())
    precheck = json.loads((out / "CUDA_PRECHECK.json").read_text())
    checks["cuda_replay_and_memory"] = (complete["exact_replay"]
        and all(r["exact"] for r in precheck["exact_replay"])
        and complete["peak_global_memory_ratio"] < .8
        and precheck["asset_sha_verified"])
    if not all(checks.values()):
        raise RuntimeError("Artifact audit did not pass")
    atomic(out / "INDEPENDENT_CHECKS.json", {"passed": True, "checks": checks,
            "g1e_successes_recomputed": successes, "denominator": 24,
            "scope": "Saved artifacts, integrity and count agreement; not independent human labels or new data",
            "checker_sha256": sha(Path(__file__))})
    print(json.dumps({"passed": True, "checks": len(checks), "g1e_successes": successes}))


if __name__ == "__main__":
    os.umask(0o077)
    main()
