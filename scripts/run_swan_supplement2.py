"""Schema-correct successor to the stopped, offline supplement1 preflight.

No pilot/provider/detector outcome was observed under supplement1. Its source
allocation is inherited byte-for-byte, with an explicit new execution freeze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import run_swan_supplement1 as runner
import swan_supplement1_sources as sources
import swan_supplement1_runtime as runtime
import swan_supplement1_review as review
from swan_final1_common import load, freeze, sha256, single_writer, now

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_count_neutral_supplement2"


def load_bank(path):
    """The official serialized bank contains records; injection expects strings."""
    value = load(path)
    if Path(path).name != "amr_bank_50.json":
        return value
    if not isinstance(value, list) or len(value) != 50:
        raise runtime.IntegrityHalt("bank_record_count_changed")
    graphs = [row["normalized_template_amr"] for row in value]
    if any(not isinstance(g, str) or not g.strip().startswith("(") for g in graphs):
        raise runtime.IntegrityHalt("bank_graph_schema_changed")
    return graphs


def bind():
    for module in (runner, sources, runtime, review):
        module.OUT = OUT
    runner.load = load_bank
    runner.DEPENDENCIES = [path.replace(
        "experiments/swan_count_neutral_supplement1/",
        "experiments/swan_count_neutral_supplement2/")
        if path.startswith("experiments/swan_count_neutral_supplement1/") else path
        for path in runner.DEPENDENCIES]
    runner.DEPENDENCIES += [
        "scripts/run_swan_supplement2.py",
        "experiments/swan_count_neutral_supplement1/FREEZE.json",
        "experiments/swan_count_neutral_supplement1/PROTOCOL.md",
        "experiments/swan_count_neutral_supplement1/PREFLIGHT_STOP.json",
    ]


def prepare():
    sources.audit()
    return freeze(OUT / "FREEZE.json", {
        "at": now(), "experiment": "swan_count_neutral_supplement2", "stage": "pilot",
        "dependencies": {p: sha256(ROOT / p) for p in runner.DEPENDENCIES},
        "source_file_hashes": load(OUT / "source_manifest.json")["sealed_file_hashes"],
        "generator": "deepseek-flash", "prompt_addition": runner.LENGTH,
        "pilot": {"documents": 8, "slots": 16, "max_calls": 6400,
                  "agent_selected_cost_guard_usd": 20, "worst_case_usd": 18.07872},
        "user_instruction": "行，那就补全一下实验吧",
        "authorization_basis": "Authorized supplement with bounded engineering successor.",
        "schema_correction": "Official bank records to normalized_template_amr strings in original order",
        "source_inheritance": "Unobserved frozen supplement1 cohorts, byte-identical",
        "main_dispatched": False, "http_retries": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "run", "status",
                                        "review-packet", "finalize-review"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--reviewer1", type=Path)
    parser.add_argument("--reviewer2", type=Path)
    args = parser.parse_args()
    bind()
    with single_writer(OUT):
        try:
            if args.stage == "prepare":
                prepare()
                result = {"new_freeze": True, "bank_records_unwrapped": 50}
            elif args.stage == "preflight":
                runner.verify()
                runner.heartbeat("CUDA_readiness")
                with runtime.runtime() as module:
                    result = runtime.readiness(module, load_bank(
                        ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_50.json"))
                freeze(OUT / "READINESS.json", result)
            elif args.stage == "run":
                freeze(OUT / "ENVIRONMENT.json", runner.environment())
                result = runner.run(args.key_file)
            elif args.stage == "status":
                result = runner.summarize()
            elif args.stage == "finalize-review":
                result = review.finalize(args.reviewer1, args.reviewer2)
            else:
                result = {"review_items": len(runner.review_packet()["items"])}
            print(json.dumps(result))
        except Exception as exc:
            if args.stage in ("run", "preflight"):
                halt = {"error_type": type(exc).__name__,
                        "reason": str(exc) if isinstance(exc, (runtime.IntegrityHalt,
                            runner.budget.GlobalStop, runner.budget.DocumentFailure))
                            else "details_private",
                        "automatic_restart_allowed": False, "generation_started": args.stage == "run"}
                freeze(OUT / "HALT.json", halt)
                runner.summarize()
            print(json.dumps({"stage": "Halt", "error_type": type(exc).__name__}), file=sys.stderr)
            sys.exit(1)
