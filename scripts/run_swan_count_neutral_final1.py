"""Sequential final1 CLI. Preparation is offline; paid execution requires phase consent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

from swan_final1_budget import CAPS, GlobalStop, Journal, MODEL, transport
from swan_final1_common import OUT, freeze, load, now, sha256, single_writer
from swan_final1_contract import BANK, check, prepare
from swan_final1_generation import run_documents
from swan_final1_quality import parse_utc
from swan_final1_workflow import (
    cohort_gates, create_transforms, prepare_human, quality_action, require_pilot, sources_for,
)


def authorize(folder, phase, who, text, calls, dollars):
    check(folder)
    if not who or not who.strip() or not text or not text.strip():
        raise PermissionError("Explicit authorizer identity and exact consent text required")
    expected_calls, expected_money = CAPS[phase]
    if calls != expected_calls or Decimal(dollars or "0") * 1_000_000_000 != expected_money:
        raise PermissionError("Authorization must explicitly match this phase's frozen caps")
    if phase == "main":
        require_pilot(folder)
    return freeze(folder / f"authorization_{phase}.json", {
        "phase": phase, "manifest_sha256": sha256(folder / "manifest.json"),
        "budget_sha256": sha256(folder / "budget_estimate.json"),
        "max_calls": calls, "max_cost_nanodollars": expected_money,
        "authorized_by": who.strip(), "authorization_text": text, "authorized_at": now(),
        "status": "explicit_phase_authorization",
    })


def require_authorization(folder, phase):
    path = folder / f"authorization_{phase}.json"
    if not path.is_file():
        raise PermissionError(f"No independent {phase} budget authorization")
    row = load(path)
    if (row["manifest_sha256"] != sha256(folder / "manifest.json")
            or row["budget_sha256"] != sha256(folder / "budget_estimate.json")
            or row["phase"] != phase or row["status"] != "explicit_phase_authorization"
            or (row["max_calls"], row["max_cost_nanodollars"]) != CAPS[phase]
            or not row["authorized_by"].strip() or not row["authorization_text"].strip()):
        raise PermissionError("Authorization does not match frozen phase contract")
    if parse_utc(row["authorized_at"]) < parse_utc(load(folder / "manifest.json")["frozen_at"]):
        raise PermissionError("Historical consent cannot authorize this new contract")
    return row


def record_price(folder, evidence_path, input_price, output_price):
    """User/agent records a newly fetched official price page, with explicit rates."""
    check(folder, full_environment=False)
    if evidence_path is None or input_price is None or output_price is None:
        raise ValueError("Official price evidence and both USD/million rates required")
    raw = evidence_path.read_text(encoding="utf-8")
    url = "https://api-docs.deepseek.com/quick_start/pricing"
    if url not in raw or MODEL not in raw:
        raise ValueError("Evidence must include official source URL and model")
    if Decimal(input_price) != Decimal(".30") or Decimal(output_price) != Decimal("1.20"):
        raise GlobalStop("Price changed; this contract cannot execute")
    evidence_hash = sha256(evidence_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # Copy evidence as a JSON string so original bytes are represented without path reuse.
    freeze(folder / "price_checks" / f"{stamp}_evidence.json",
           {"original_sha256": evidence_hash, "content": raw, "source_url": url})
    evidence_copy = folder / "price_checks" / f"{stamp}_evidence.json"
    return freeze(folder / "price_checks" / f"{stamp}.json", {
        "checked_at": now(), "source_url": url, "model": MODEL,
        "input_price_usd_per_million": input_price, "output_price_usd_per_million": output_price,
        "manifest_sha256": sha256(folder / "manifest.json"),
        "evidence_copy": evidence_copy.name, "evidence_sha256": sha256(evidence_copy),
        "verification": "explicit_official_page_read_and_rate_attestation",
    })


def require_price(folder):
    paths = sorted(p for p in (folder / "price_checks").glob("*.json")
                   if not p.name.endswith("_evidence.json"))
    if not paths:
        raise PermissionError("Read official current pricing and record-price before paid execution")
    row = load(paths[-1])
    age = (datetime.now(timezone.utc) - parse_utc(row["checked_at"])).total_seconds()
    if not 0 <= age <= 24 * 3600 or row["manifest_sha256"] != sha256(folder / "manifest.json") \
            or sha256(folder / "price_checks" / row["evidence_copy"]) != row["evidence_sha256"] \
            or Decimal(row["input_price_usd_per_million"]) != Decimal(".30") \
            or Decimal(row["output_price_usd_per_million"]) != Decimal("1.20"):
        raise PermissionError("Current official pricing evidence missing, stale, or changed")
    return row


def run_phase(folder, phase, max_new_calls):
    if not max_new_calls or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")
    check(folder)
    require_authorization(folder, phase)
    require_price(folder)
    sources = sources_for(folder, phase)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    journal = Journal(folder, phase, max_new_calls, lambda body: transport(body, key),
                      sha256(folder / "manifest.json"))
    if phase == "pilot":
        call = journal.calls.get("final1-pilot-model-probe")
        if call is None:
            journal.request("final1-pilot-model-probe", "Reply OK.", None, 1, .7, .9)
        elif not call["terminal"] or call["terminal"]["kind"] != "complete":
            raise GlobalStop("Identity probe uncertain; no automatic retry")
    # Parsing only starts after all contract, identity, budget and price checks.
    from swan_final1_runtime import official_runtime, swan
    bank = [r["normalized_template_amr"] for r in load(BANK)]
    if len(bank) != 50:
        raise GlobalStop("Bank size differs")
    examples = BANK.parent.parent / "artifacts/normalized_amr_examples.json"
    with official_runtime(folder) as module:
        return run_documents(folder, phase, sources, journal, module,
                             swan.run_official_slot, bank, examples)


def status(folder):
    result = {"manifest_present": (folder / "manifest.json").exists(), "phases": {}}
    for phase in ("pilot", "main"):
        paths = list((folder / "documents" / phase).glob("*.json"))
        result["phases"][phase] = {
            "authorization_file_present": (folder / f"authorization_{phase}.json").exists(),
            "terminal_documents": len(paths),
            "structural_pass": sum(load(p)["structural_pass"] for p in paths),
            "baseline_quality_frozen": (folder / "quality" / phase / "baseline/decision.json").exists(),
            "pair_quality_frozen": (folder / "quality" / phase / "pair/decision.json").exists(),
        }
        if (folder / "private" / phase / "journal.jsonl").exists():
            result["phases"][phase]["accounting"] = Journal(
                folder, phase, 1, None, sha256(folder / "manifest.json"),
                allow_halted=True).totals()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "check", "status", "authorize", "record-price", "run",
        "export-quality", "finalize-quality", "transform", "gate",
        "prepare-human", "freeze-detection-plan", "detect", "analyze", "audit",
    ))
    parser.add_argument("--phase", choices=("pilot", "main", "calibration", "test"), default="pilot")
    parser.add_argument("--kind", choices=("baseline", "pair"), default="baseline")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-text")
    parser.add_argument("--evidence-file", type=Path)
    parser.add_argument("--input-price")
    parser.add_argument("--output-price")
    parser.add_argument("--stage", choices=("gate_b_calibration", "gate_b_positive",
                                           "calibration", "positive", "test"))
    args = parser.parse_args()
    folder = OUT
    with single_writer(folder):
        if args.command in ("prepare", "check", "status"):
            result = {"prepare": prepare, "check": check, "status": status}[args.command](folder)
        elif args.command == "authorize":
            if args.phase not in CAPS:
                raise ValueError("Only paid generation phases require authorization")
            result = authorize(folder, args.phase, args.authorized_by, args.authorization_text,
                               args.max_calls, args.max_cost_usd)
        elif args.command == "record-price":
            result = record_price(folder, args.evidence_file, args.input_price, args.output_price)
        elif args.command == "run":
            if args.phase not in CAPS:
                raise ValueError("Only pilot/main have paid generation")
            result = run_phase(folder, args.phase, args.max_new_calls)
        else:
            check(folder)
            if args.command in ("export-quality", "finalize-quality"):
                result = quality_action(folder, args.phase, args.kind,
                                        args.command == "finalize-quality")
            elif args.command == "transform":
                result = {"transformed_allocations": len(create_transforms(folder, args.phase))}
            elif args.command == "prepare-human":
                result = prepare_human(folder, args.phase)
            elif args.command == "gate":
                result = cohort_gates(folder, args.phase)
            else:
                from swan_final1_detection import detection_command
                result = detection_command(folder, args.command, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No credential or transport exception body is printed.
        print(json.dumps({"state": "stopped", "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, (GlobalStop, PermissionError, ValueError))
                          else "See saved local provenance; no automatic retry"}), file=sys.stderr)
        sys.exit(1)
