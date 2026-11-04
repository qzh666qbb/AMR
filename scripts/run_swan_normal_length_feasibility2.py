"""Sequential CLI for the 20-source normal-length SWAN feasibility pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

from swan_feasibility2_budget import (
    GlobalStop,
    Journal,
    MODEL,
    PILOT_CAP,
    transport,
)
from swan_feasibility2_common import OUT, freeze, load, now, seed, sha256, single_writer
from swan_feasibility2_contract import BANK, check, prepare
from swan_feasibility2_generation import run_documents
from swan_feasibility2_workflow import quality_action, sources_for
from swan_final1_quality import parse_utc


def authorize(folder, who, text, calls, dollars, key_rotation_attested):
    check(folder)
    if not who or not who.strip() or not text or not text.strip():
        raise PermissionError("Explicit authorizer identity and consent text required")
    if key_rotation_attested is not True:
        raise PermissionError("Explicit API key rotation attestation required")
    expected_calls, expected_money = PILOT_CAP
    if calls != expected_calls \
            or Decimal(dollars or "0") * 1_000_000_000 != expected_money:
        raise PermissionError("Authorization must exactly match the frozen pilot cap")
    return freeze(Path(folder) / "authorization_pilot.json", {
        "phase": "pilot",
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "budget_sha256": sha256(Path(folder) / "budget_estimate.json"),
        "max_calls": calls,
        "max_cost_nanodollars": expected_money,
        "authorized_by": who.strip(),
        "authorization_text": text.strip(),
        "key_rotation_attested": True,
        "authorized_at": now(),
        "status": "explicit_pilot_authorization",
    })


def require_authorization(folder):
    folder = Path(folder)
    path = folder / "authorization_pilot.json"
    if not path.is_file():
        raise PermissionError("No immutable pilot budget authorization")
    row = load(path)
    if row.get("manifest_sha256") != sha256(folder / "manifest.json") \
            or row.get("budget_sha256") != sha256(folder / "budget_estimate.json") \
            or row.get("phase") != "pilot" \
            or row.get("status") != "explicit_pilot_authorization" \
            or (row.get("max_calls"), row.get("max_cost_nanodollars")) != PILOT_CAP \
            or row.get("key_rotation_attested") is not True \
            or not row.get("authorized_by", "").strip() \
            or not row.get("authorization_text", "").strip():
        raise PermissionError("Authorization does not match the frozen contract")
    if parse_utc(row["authorized_at"]) < parse_utc(
            load(folder / "manifest.json")["frozen_at"]
    ):
        raise PermissionError("Historical consent cannot authorize this contract")
    return row


def record_price(folder, evidence_path, input_price, output_price):
    folder = Path(folder)
    check(folder, full_environment=False)
    if evidence_path is None or input_price is None or output_price is None:
        raise ValueError("Official price evidence and both rates are required")
    raw = evidence_path.read_text(encoding="utf-8")
    url = "https://api-docs.deepseek.com/quick_start/pricing"
    if url not in raw or MODEL not in raw:
        raise ValueError("Evidence must identify the official URL and model")
    if Decimal(input_price) != Decimal(".30") \
            or Decimal(output_price) != Decimal("1.20"):
        raise GlobalStop("Official peak pricing changed; freeze a new contract")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence_copy = folder / "price_checks" / f"{stamp}_evidence.json"
    freeze(evidence_copy, {
        "original_sha256": sha256(evidence_path),
        "content": raw,
        "source_url": url,
    })
    return freeze(folder / "price_checks" / f"{stamp}.json", {
        "checked_at": now(),
        "source_url": url,
        "model": MODEL,
        "input_price_usd_per_million": input_price,
        "output_price_usd_per_million": output_price,
        "manifest_sha256": sha256(folder / "manifest.json"),
        "evidence_copy": evidence_copy.name,
        "evidence_sha256": sha256(evidence_copy),
    })


def require_price(folder):
    folder = Path(folder)
    paths = sorted(
        path for path in (folder / "price_checks").glob("*.json")
        if not path.name.endswith("_evidence.json")
    )
    if not paths:
        raise PermissionError("Record current official pricing before paid execution")
    row = load(paths[-1])
    age = (datetime.now(timezone.utc) - parse_utc(row["checked_at"])).total_seconds()
    if not 0 <= age <= 24 * 3600 \
            or row["manifest_sha256"] != sha256(folder / "manifest.json") \
            or sha256(folder / "price_checks" / row["evidence_copy"]) != \
            row["evidence_sha256"] \
            or Decimal(row["input_price_usd_per_million"]) != Decimal(".30") \
            or Decimal(row["output_price_usd_per_million"]) != Decimal("1.20"):
        raise PermissionError("Official price evidence is stale, missing, or changed")
    return row


def run_phase(folder, max_new_calls):
    folder = Path(folder)
    if not max_new_calls or max_new_calls < 1:
        raise ValueError("Explicit positive --max-new-calls required")
    check(folder)
    require_authorization(folder)
    require_price(folder)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PermissionError("DEEPSEEK_API_KEY is not set")
    journal = Journal(
        folder, max_new_calls, lambda body: transport(body, key),
        sha256(folder / "manifest.json"),
    )
    probe = journal.calls.get("feasibility2-pilot-model-probe")
    if probe is None:
        journal.request(
            "feasibility2-pilot-model-probe", "Reply OK.", None, 1, .7, .9
        )
    elif not probe["terminal"] or probe["terminal"]["kind"] != "complete":
        raise GlobalStop("Model identity probe is uncertain; no retry")

    from swan_final1_runtime import official_runtime, swan

    bank = [row["normalized_template_amr"] for row in load(BANK)]
    examples = BANK.parent.parent / "artifacts/normalized_amr_examples.json"
    if len(bank) != 50:
        raise GlobalStop("Bank size changed")
    with official_runtime(folder) as module:
        swan.slot_seed = lambda allocation, slot: seed(
            "injection-template", allocation, slot
        )
        swan.score_seed = lambda allocation, slot, attempt: seed(
            "injection-score", allocation, slot, attempt
        )
        return run_documents(
            folder, sources_for(folder), journal, module,
            swan.run_official_slot, bank, examples,
        )


def status(folder):
    folder = Path(folder)
    paths = sorted((folder / "documents/pilot").glob("*.json"))
    result = {
        "experiment": "swan_normal_length_feasibility2",
        "manifest_present": (folder / "manifest.json").exists(),
        "authorization_present": (folder / "authorization_pilot.json").exists(),
        "terminal_documents": len(paths),
        "prose_structure_pass": sum(load(path)["prose_structure_pass"] for path in paths),
        "completed_slots": sum(
            load(path)["accepted_slots"] + load(path)["fallback_slots"] for path in paths
        ),
        "accepted_slots": sum(load(path)["accepted_slots"] for path in paths),
        "quality_exported": (
            folder / "quality/pilot/baseline/private_index.json"
        ).exists(),
        "quality_frozen": (
            folder / "quality/pilot/baseline/decision.json"
        ).exists(),
    }
    journal_path = folder / "private/pilot/journal.jsonl"
    if journal_path.exists() and (folder / "manifest.json").exists():
        result["accounting"] = Journal(
            folder, 1, None, sha256(folder / "manifest.json"), allow_halted=True
        ).totals()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "check", "status", "authorize", "record-price", "run",
        "export-quality", "finalize-quality",
    ))
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-text")
    parser.add_argument("--key-rotation-attested", action="store_true")
    parser.add_argument("--evidence-file", type=Path)
    parser.add_argument("--input-price")
    parser.add_argument("--output-price")
    args = parser.parse_args()
    with single_writer(OUT):
        if args.command in {"prepare", "check", "status"}:
            result = {
                "prepare": prepare,
                "check": check,
                "status": status,
            }[args.command](OUT)
        elif args.command == "authorize":
            result = authorize(
                OUT, args.authorized_by, args.authorization_text,
                args.max_calls, args.max_cost_usd, args.key_rotation_attested,
            )
        elif args.command == "record-price":
            result = record_price(
                OUT, args.evidence_file, args.input_price, args.output_price
            )
        elif args.command == "run":
            result = run_phase(OUT, args.max_new_calls)
        else:
            check(OUT)
            result = quality_action(
                OUT, finalize_labels=args.command == "finalize-quality"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        reason = str(exc) if isinstance(
            exc, (GlobalStop, PermissionError, ValueError, FileNotFoundError)
        ) else "See saved local provenance; no automatic retry"
        print(json.dumps({
            "state": "stopped",
            "error_type": type(exc).__name__,
            "reason": reason,
        }), file=sys.stderr)
        sys.exit(1)
