"""Model-free oracle and compatibility checks for the full-radius core."""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import random
import subprocess
import tempfile
import time
from types import SimpleNamespace

from run_semstamp_certified_defense_v3_compiled_correctness import (
    all_spans, batch_payload, brute_target, compare_target,
    parse_output, v3_case_payload,
)
from run_semstamp_certified_defense_v4_full_radius_profile import (
    STAGE, atomic_write_json, canonical_bytes, load_checkpoint, paths_for,
    prepare_document_profile, run, sha256_file,
)


class Document:
    atom_count = 9

    def __init__(self, original):
        self.original_boundaries = original

    def render_unit_bytes(self, start, end):
        return f"{start}:{end}".encode("ascii")


class Tokenizer:
    model_max_length = 512

    def __call__(self, texts, **kwargs):
        return {"input_ids": [
            [1, 2, 3] if text != "1:8" else list(range(513))
            for text in texts
        ]}


def invoke(core, cases):
    payload = batch_payload([v3_case_payload(case) for case in cases])
    return subprocess.run(
        [str(core)], input=payload, capture_output=True, timeout=120,
    )


def systematic():
    cases = []
    for n in range(2, 11):
        for m0 in range(1, n):
            for original in combinations(range(1, n), m0):
                for b in range(4):
                    for m in range(max(1, m0-b), min(n-1, m0+b)+1):
                        cases.append((n, original, m, b, all_spans(n)))
                        if len(cases) == 5000:
                            return cases
    raise AssertionError("systematic matrix exhausted")


def random_cases():
    rng = random.Random(20261025)
    cases = []
    for _ in range(2000):
        n = rng.randint(3, 10)
        m0 = rng.randint(1, n-1)
        original = tuple(sorted(rng.sample(range(1, n), m0)))
        b = rng.randint(0, 3)  # Deliberately independent of m0.
        m = rng.randint(max(1, m0-b), min(n-1, m0+b))
        allowed = {s for s in sorted(all_spans(n)) if rng.random() < 0.60}
        ends = (0, *original, n)
        allowed.update(zip(ends, ends[1:]))
        cases.append((n, original, m, b, allowed))
    return cases


def oracle_family(core, cases, digest):
    expected = [brute_target(case) for case in cases]
    nonempty = [(case, ref) for case, ref in zip(cases, expected) if ref]
    first = invoke(core, [case for case, _ in nonempty])
    second = invoke(core, [case for case, _ in nonempty])
    if first.returncode or second.returncode or first.stdout != second.stdout:
        raise AssertionError("core invocation/replay mismatch")
    observed = parse_output(first.stdout, len(nonempty))
    for (case, ref), row in zip(nonempty, observed):
        compare_target(row, ref)
        if case[-1] == all_spans(case[0]):
            n, original, m, b, _ = case
            histogram = {
                a: (comb(n-1-len(original), a) * comb(len(original), m-a)
                    if 0 <= a <= n-1-len(original)
                    and 0 <= m-a <= len(original)
                    and 0 <= len(original)-m+a <= b and a <= b else 0)
                for a in range(4)
            }
            if row["insertion_histogram"] != histogram:
                raise AssertionError("independent binomial count mismatch")
        digest.update(canonical_bytes({"case": [
            *case[:4], sorted(case[4]),
        ], "oracle": ref}))
    empty = 0
    for case, ref in zip(cases, expected):
        if ref:
            continue
        response = invoke(core, [case])
        if response.returncode == 0 or response.stdout \
                or b"empty threat set" not in response.stderr:
            raise AssertionError("empty threat set did not fail closed")
        empty += 1
    return {
        "cases": len(cases), "nonempty": len(nonempty),
        "empty_fail_closed": empty,
        "b_gt_m0_cases": sum(case[3] > len(case[1]) for case in cases),
        "replay_byte_exact": True,
    }


def profile_integration(core):
    profiles = [
        {
            "atom_count": anchor, "selection_slot": str(anchor),
            "feature_seconds_per_input_token": 1.0,
            "sidecar_seconds_per_span": 1.0,
            "core_seconds_per_pair": 1.0, "model_load_seconds": 0.0,
        }
        for anchor in (5, 10)
    ]
    counts = []
    for original in ((4,), (3, 6)):
        doc = Document(original)
        with tempfile.TemporaryDirectory() as directory:
            radii, resources, _ = prepare_document_profile(
                doc, Tokenizer(), core, Path(directory), profiles,
            )
        for b in range(4):
            row = radii[str(b)]
            expected_targets = list(range(
                max(1, len(original)-b),
                min(doc.atom_count-1, len(original)+b)+1,
            ))
            if row["effective_radius"] != b \
                    or row["target_edge_counts"] != expected_targets:
                raise AssertionError("profile still caps requested radius")
            allowed = all_spans(doc.atom_count) - {(1, 8)}
            for layer in row["layers"]:
                case = (doc.atom_count, original,
                        layer["target_edge_count"], b, allowed)
                ref = brute_target(case, scorer=lambda *args: 0)
                observed = layer["native_pruned"]
                if observed["threat_set_size"] != ref["threat_set_size"] \
                        or observed["insertion_histogram"] != ref["insertion_histogram"]:
                    raise AssertionError("native-pruned profile count mismatch")
        counts.append({
            "m0": len(original),
            "target_layers": sum(len(r["layers"]) for r in radii.values()),
            "distinct_radii": resources["distinct_effective_radii"],
        })
    return counts


def checkpoint_and_authorization_gates():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "experiment"
        output.mkdir()
        paths = paths_for(root, output)
        atomic_write_json(paths["contract"], {"frozen": True})
        atomic_write_json(paths["protocol"], {"protocol": True})
        paths["manifest"].write_text("frozen input identity\n")
        row = {
            "plan_index": 0, "raw_text_sha256": "raw",
            "topology_sha256": "topology", "stage": STAGE,
            "protocol_sha256": sha256_file(paths["protocol"]),
        }
        timing = {"plan_index": 0}
        atomic_write_json(paths["checkpoints"] / "0000.json", {
            "binding": {
                "contract_sha256": sha256_file(paths["contract"]),
                "manifest_sha256": sha256_file(paths["manifest"]),
            },
            "row": row, "timing": timing,
        })
        # Simulate an interruption during the second ledger append.
        paths["rows"].write_text('{"partial":')
        plan = [{"raw_text_sha256": "raw"}]
        frozen = [{"topology_sha256": "topology"}]
        recovered_rows, recovered_timings = load_checkpoint(
            paths, [0], plan, frozen,
        )
        if recovered_rows != [row] or recovered_timings != [timing] \
                or paths["rows"].read_bytes() != canonical_bytes(row) \
                or paths["timings"].read_bytes() != canonical_bytes(timing):
            raise AssertionError("atomic checkpoint recovery changed evidence")
        paths["manifest"].write_text("changed identity\n")
        try:
            load_checkpoint(paths, [0], plan, frozen)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint identity drift not rejected")
        atomic_write_json(paths["readiness"], {
            "audit_status": "pass", "decision": "stop",
        })
        try:
            run(SimpleNamespace(root=root, output_dir=output))
        except RuntimeError as error:
            if str(error) != "full-radius resource readiness is not Go":
                raise
        else:
            raise AssertionError("resource Stop did not block execution")
    return {
        "partial_ledger_recovered_from_atomic_checkpoint": True,
        "changed_input_binding_rejected": True,
        "resource_stop_blocks_execution_before_input_or_model_access": True,
    }


def audit(core, legacy_core, output):
    started = time.perf_counter()
    digest = hashlib.sha256()
    matrix = systematic()
    systematic_result = oracle_family(core, matrix, digest)
    random_result = oracle_family(core, random_cases(), digest)
    if random_result["empty_fail_closed"] == 0:
        raise AssertionError("empty-set coverage missing")
    compatible = [c for c in matrix if c[3] <= len(c[1])][:1000]
    if len(compatible) != 1000:
        raise AssertionError("compatibility denominator missing")
    old = invoke(legacy_core, compatible)
    new = invoke(core, compatible)
    if old.returncode or new.returncode or old.stdout != new.stdout:
        raise AssertionError("legacy common-domain outputs changed")
    special = []
    for original, b in (((4,), 2), ((4,), 3), ((3, 6), 3)):
        n = 9
        m = len(original) + b
        case = (n, original, m, b, all_spans(n))
        response = invoke(core, [case])
        if response.returncode:
            raise AssertionError("pure insertion instance rejected")
        row = parse_output(response.stdout, 1)[0]
        expected = comb(n-1-len(original), b)
        if row["threat_set_size"] != expected \
                or row["insertion_histogram"][b] != expected:
            raise AssertionError("pure insertion count mismatch")
        special.append({"m0": len(original), "b": b,
                        "target_m": m, "pure_insertions": expected})
    root = Path(__file__).resolve().parents[1]
    old_source = root / "scripts/semstamp_certified_defense_v3_variable_n_core.cpp"
    new_source = root / "scripts/semstamp_certified_defense_v4_full_radius_core.cpp"
    if new_source.read_text() != old_source.read_text().replace(
        "      || problem.k > problem.original_m\n", "", 1,
    ):
        raise AssertionError("unexpected core algorithm change")
    result = {
        "schema": "semstamp-v4-radius-correctness/v1",
        "status": "pass",
        "systematic": systematic_result,
        "random_legality": random_result,
        "legacy_compatibility_cases": len(compatible),
        "legacy_outputs_byte_exact": True,
        "pure_insertion_cases": special,
        "profile_integration": profile_integration(core),
        "checkpoint_and_authorization": checkpoint_and_authorization_gates(),
        "core_change_only_removes_b_gt_m0_rejection": True,
        "observation_sha256": digest.hexdigest(),
        "inputs": {
            "core_sha256": sha256_file(core),
            "legacy_core_sha256": sha256_file(legacy_core),
            "source_sha256": sha256_file(new_source),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter()-started,
        "semantic_model_loads": 0,
        "semantic_detector_computations": 0,
        "certificate_computations": 0,
    }
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--legacy-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.core.resolve(), args.legacy_core.resolve(), args.output)
