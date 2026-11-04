"""Validate the parser repair against frozen cases; export hashes, never text."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--origin", action="append", required=True, help="label=old runtime path")
    ap.add_argument("--reproduction", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Validation requires a new output directory")
    origins = {label: Path(path) for label, path in (v.split("=", 1) for v in args.origin)}
    expected = read(args.reproduction)
    code = args.code_root
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "scripts"))
    import run_multiscale_transfer as rt
    from semantic_fact_raw_amr import audit_raw, replay_signature

    # Only hashes leave the machine. Read original texts solely for exact replay.
    cases, old_hashes = [], {}
    for label, origin in origins.items():
        freeze = read(origin / "FREEZE.json")
        assert sha(origin / "FREEZE.json") == expected["source_hashes"][label]["freeze"]
        for relative, value in freeze["files"].items():
            path = origin / relative
            assert sha(path) == value
            old_hashes[path] = value
        for name in ("FREEZE.json", "HALT.json", "TERMINAL.json", "RAW_GRAPH_SEAL.json"):
            path = origin / name
            if path.exists():
                old_hashes[path] = sha(path)
        if label == "atomic":
            texts = {c["text"] for row in read(origin / "private/INPUTS.json")
                     for c in row["candidates"]}
        else:
            texts = {s for row in read(origin / ("private/BANK_INPUTS.json" if label == "banks"
                                                else "private/INPUTS.json"))
                     for s in row["sentences"]}
        by_hash = {rt.digest(t): t for t in texts}
        for row in expected["cases"]:
            if row["case"].startswith(label + "_"):
                cases.append((row, by_hash[row["text_sha256"]]))

    args.out.mkdir(parents=True)
    dst = args.out / "private/code/src/core/parser_recovery_audit.py"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(code / "src/core/parser_recovery_audit.py", dst)
    source = read(origins["atomic"] / "FREEZE.json")
    files = ["scripts/run_multiscale_transfer.py", "src/core/parser_recovery_audit.py",
             "scripts/run_research_completion_queue.py",
             "scripts/validate_parser_queue_repair_cuda.py"]
    freeze = {
        "schema": "parser-queue-repair-validation/v1", "parser": source["parser"],
        "execution": {"resumable_gpu_pressure": True},
        "resource_gate": {"vram_fraction_max": 0.8},
        "source_sha256": {p: sha(code / p) for p in files},
        "reproduction_sha256": sha(args.reproduction),
        "original_freezes": {label: sha(path / "FREEZE.json") for label, path in origins.items()},
        "case_text_sha256": [row["text_sha256"] for row, _ in cases],
        "new_provider_calls": 0, "replays": 2,
    }
    rt.atomic(args.out / "FREEZE.json", freeze)
    sink = io.StringIO()
    rows = []
    with redirect_stdout(sink), redirect_stderr(sink):
        parser = rt.Parser(args.assets, args.out, freeze)
        for expected_row, text in cases:
            records = []
            for replay in range(2):
                record = parser.fresh(text)
                record["case_id"] = expected_row["case"]
                assert record["amr_sha256"] == expected_row["selected_graph_sha256"]
                assert record["upstream_output_sha256"] == expected_row["returned_amr_sha256"]
                assert record["amr_sha256"] == next(
                    a["graph_sha256"] for a in record["attempts"] if a["accepted"])
                assert record["valid"] and not record["clipped"] and not record["decode_events"]
                # Same graph-cleaning step used by Normalizer; no residual input.
                cleaned = "\n".join(s for s in record["amr"].splitlines() if not s.startswith("#"))
                graph = parser.penman.decode(cleaned, model=parser.amr_model)
                assert sorted(graph.triples) == sorted(
                    parser.penman.decode(record["amr"], model=parser.amr_model).triples)
                audit = audit_raw(record, {"case_id": record["case_id"],
                                           "input_text_sha256": rt.digest(text)}, parser.freeze_sha)
                records.append(record)
            assert replay_signature(records[0]) == replay_signature(records[1])
            rows.append({
                "case": expected_row["case"], "text_sha256": rt.digest(text),
                "selected_graph_sha256": records[0]["amr_sha256"],
                "upstream_output_sha256": records[0]["upstream_output_sha256"],
                "exact_replay": True, "valid_roundtrip": True,
                "structurally_eligible": audit["structurally_eligible"],
                "exclusion_reasons": audit["exclusion_reasons"],
                "same_selected_graph_as_original_inference": True,
                "downstream_graph_decode": True, "selected_beam": audit["selected_beam"],
                "events": len(records[0]["events"]),
                "event_levels": [e["level"] for e in records[0]["events"]],
                "seconds": sum(r["seconds"] for r in records),
            })
        # Exercise the standalone audit on a real multiline case as well.
        spec = importlib.util.spec_from_file_location("src.core.parser_recovery_audit",
                                                     code / "src/core/parser_recovery_audit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row, text = next((r, t) for r, t in cases if r["case"] == "atomic_failure")
        outputs, audits = module.parse_with_recovery_audit(parser.model, [text], parser.inference)
        assert rt.digest(outputs[0]) == row["returned_amr_sha256"]
        assert audits[0]["attempts"][audits[0]["selected_beam"]]["graph_sha256"] == row["selected_graph_sha256"]
    for path, value in old_hashes.items():
        assert sha(path) == value
    result = {
        "status": "PASS", "at_unix": time.time(), "cases": rows,
        "inference_calls": len(cases) * 2 + 1, "new_provider_calls": 0,
        "dtype": "float32", "cuda": True, "standalone_audit_multiline": True,
        "original_files_sha_verified_before_and_after": len(old_hashes),
        "original_artifacts_modified": False, "old_halts_resumed": False,
        "freeze_sha256": sha(args.out / "FREEZE.json"),
        "source_sha256": freeze["source_sha256"],
        "new_experiment_metrics": False, "private_text_exported": False,
    }
    rt.atomic(args.out / "VALIDATION.json", result)
    rt.atomic(args.out / "TERMINAL.json", {
        "status": "PASS", "validation_sha256": sha(args.out / "VALIDATION.json"),
        "freeze_sha256": sha(args.out / "FREEZE.json")})
    print(json.dumps({"validation": result, "freeze": freeze,
                      "terminal": read(args.out / "TERMINAL.json")}, indent=2))


if __name__ == "__main__":
    main()
