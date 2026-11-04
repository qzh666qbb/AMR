"""Sequential CLI for the source-anchored SWAN experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

from swan_anchor1_budget import CAPS, GlobalStop, Journal, MODEL, transport
from swan_anchor1_common import OUT, freeze, load, now, sha256, single_writer
from swan_anchor1_contract import BANK, check, prepare
from swan_anchor1_generation import run_documents
from swan_anchor1_workflow import (
    cohort_gates,
    create_transforms,
    quality_action,
    require_pilot,
    sources_for,
)
from swan_final1_quality import parse_utc


def authorize(folder, phase, who, text, calls, dollars, key_rotation_attested):
    check(folder)
    if not who or not who.strip() or not text or not text.strip():
        raise PermissionError("Explicit authorizer identity and consent text required")
    if key_rotation_attested is not True:
        raise PermissionError("Explicit API key rotation attestation required")
    expected_calls, expected_money = CAPS[phase]
    if calls != expected_calls \
            or Decimal(dollars or "0") * 1_000_000_000 != expected_money:
        raise PermissionError("Authorization must exactly match the frozen phase caps")
    if phase == "main":
        require_pilot(folder)
    return freeze(Path(folder) / f"authorization_{phase}.json", {
        "phase": phase,
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "budget_sha256": sha256(Path(folder) / "budget_estimate.json"),
        "max_calls": calls,
        "max_cost_nanodollars": expected_money,
        "authorized_by": who.strip(),
        "authorization_text": text.strip(),
        "key_rotation_attested": True,
        "authorized_at": now(),
        "status": "explicit_independent_phase_authorization",
    })


def require_authorization(folder, phase):
    path = Path(folder) / f"authorization_{phase}.json"
    if not path.is_file():
        raise PermissionError(f"No independent {phase} budget authorization")
    row = load(path)
    if row.get("manifest_sha256") != sha256(Path(folder) / "manifest.json") \
            or row.get("budget_sha256") != sha256(Path(folder) / "budget_estimate.json") \
            or row.get("phase") != phase \
            or row.get("status") != "explicit_independent_phase_authorization" \
            or (row.get("max_calls"), row.get("max_cost_nanodollars")) != CAPS[phase] \
            or row.get("key_rotation_attested") is not True \
            or not row.get("authorized_by", "").strip() \
            or not row.get("authorization_text", "").strip():
        raise PermissionError("Authorization does not match the frozen phase contract")
    if parse_utc(row["authorized_at"]) < parse_utc(
            load(Path(folder) / "manifest.json")["frozen_at"]):
        raise PermissionError("Historical consent cannot authorize this contract")
    return row


def record_price(folder, evidence_path, input_price, output_price):
    check(folder, full_environment=False)
    if evidence_path is None or input_price is None or output_price is None:
        raise ValueError("Official price evidence and both rates are required")
    raw = evidence_path.read_text(encoding="utf-8")
    url = "https://api-docs.deepseek.com/quick_start/pricing"
    if url not in raw or MODEL not in raw:
        raise ValueError("Evidence must identify the official URL and deepseek-flash")
    if Decimal(input_price) != Decimal(".30") \
            or Decimal(output_price) != Decimal("1.20"):
        raise GlobalStop("Official peak pricing changed; freeze a new contract")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence = freeze(Path(folder) / "price_checks" / f"{stamp}_evidence.json", {
        "original_sha256": sha256(evidence_path),
        "content": raw,
        "source_url": url,
    })
    evidence_path_copy = Path(folder) / "price_checks" / f"{stamp}_evidence.json"
    return freeze(Path(folder) / "price_checks" / f"{stamp}.json", {
        "checked_at": now(),
        "source_url": url,
        "model": MODEL,
        "input_price_usd_per_million": input_price,
        "output_price_usd_per_million": output_price,
        "manifest_sha256": sha256(Path(folder) / "manifest.json"),
        "evidence_copy": evidence_path_copy.name,
        "evidence_sha256": sha256(evidence_path_copy),
        "evidence_fields": sorted(evidence),
    })


def require_price(folder):
    paths = sorted(
        path
        for path in (Path(folder) / "price_checks").glob("*.json")
        if not path.name.endswith("_evidence.json")
    )
    if not paths:
        raise PermissionError("Record current official pricing before paid execution")
    row = load(paths[-1])
    age = (datetime.now(timezone.utc) - parse_utc(row["checked_at"])).total_seconds()
    if not 0 <= age <= 24 * 3600 \
            or row["manifest_sha256"] != sha256(Path(folder) / "manifest.json") \
            or sha256(Path(folder) / "price_checks" / row["evidence_copy"]) != \
            row["evidence_sha256"] \
            or Decimal(row["input_price_usd_per_million"]) != Decimal(".30") \
            or Decimal(row["output_price_usd_per_million"]) != Decimal("1.20"):
        raise PermissionError("Official price evidence is stale, missing, or changed")
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
    journal = Journal(
        folder,
        phase,
        max_new_calls,
        lambda body: transport(body, key),
        sha256(Path(folder) / "manifest.json"),
    )
    if phase == "pilot":
        probe = journal.calls.get("anchor1-pilot-model-probe")
        if probe is None:
            journal.request(
                "anchor1-pilot-model-probe", "Reply OK.", None, 1, .7, .9
            )
        elif not probe["terminal"] or probe["terminal"]["kind"] != "complete":
            raise GlobalStop("Model identity probe is uncertain; no retry")

    from swan_final1_runtime import official_runtime
    from swan_final1_transform import FrozenSyntax

    bank = [row["normalized_template_amr"] for row in load(BANK)]
    if len(bank) != 50:
        raise GlobalStop("Bank size changed")
    examples = load(BANK.parent.parent / "artifacts/normalized_amr_examples.json")[:3]
    with official_runtime(Path(folder)) as module:
        return run_documents(
            folder,
            phase,
            sources,
            journal,
            module,
            bank,
            examples,
            FrozenSyntax(),
        )


def status(folder):
    folder = Path(folder)
    result = {
        "experiment": "swan_source_anchored1",
        "manifest_present": (folder / "manifest.json").exists(),
        "phases": {},
    }
    for phase in ("pilot", "main"):
        documents = list((folder / "documents" / phase).glob("*.json"))
        result["phases"][phase] = {
            "authorization_file_present": (folder / f"authorization_{phase}.json").exists(),
            "terminal_documents": len(documents),
            "structural_pass": sum(load(path)["structural_pass"] for path in documents),
            "baseline_quality_frozen": (
                folder / "quality" / phase / "baseline/decision.json"
            ).exists(),
            "pair_quality_frozen": (
                folder / "quality" / phase / "pair/decision.json"
            ).exists(),
            "transforms_frozen": (folder / "transforms" / f"{phase}.json").exists(),
        }
        journal_path = folder / "private" / phase / "journal.jsonl"
        if journal_path.exists() and (folder / "manifest.json").exists():
            result["phases"][phase]["accounting"] = Journal(
                folder,
                phase,
                1,
                None,
                sha256(folder / "manifest.json"),
                allow_halted=True,
            ).totals()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare",
        "check",
        "status",
        "authorize",
        "record-price",
        "run",
        "export-quality",
        "finalize-quality",
        "transform",
        "gate",
    ))
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--kind", choices=("baseline", "pair"), default="baseline")
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
    folder = OUT
    with single_writer(folder):
        if args.command in {"prepare", "check", "status"}:
            result = {
                "prepare": prepare,
                "check": check,
                "status": status,
            }[args.command](folder)
        elif args.command == "authorize":
            result = authorize(
                folder,
                args.phase,
                args.authorized_by,
                args.authorization_text,
                args.max_calls,
                args.max_cost_usd,
                args.key_rotation_attested,
            )
        elif args.command == "record-price":
            result = record_price(
                folder,
                args.evidence_file,
                args.input_price,
                args.output_price,
            )
        elif args.command == "run":
            result = run_phase(folder, args.phase, args.max_new_calls)
        else:
            check(folder)
            if args.command in {"export-quality", "finalize-quality"}:
                result = quality_action(
                    folder,
                    args.phase,
                    args.kind,
                    args.command == "finalize-quality",
                )
            elif args.command == "transform":
                result = {
                    "transformed_allocations": len(create_transforms(folder, args.phase))
                }
            else:
                result = cohort_gates(folder, args.phase)
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
