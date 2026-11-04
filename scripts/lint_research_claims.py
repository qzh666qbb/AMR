"""Offline claim lint and fail-closed primary-endpoint verification."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "unqualified_amr_change": r"\bAMR (?:changed|changes|is changed)\b",
    "unqualified_semantic_invariance": r"\bsemantic(?:ally)?[ -]invariant\b",
    "parser_necessity": r"\bparser(?: drift)? (?:is )?unnecessary\b",
    "semantic_attack": r"攻击\s*AMR\s*语义|破坏\s*AMR\s*语义",
    "parser_necessity_zh": r"parser\s*drift\s*(?:非必要|不是.{0,12}必要条件)",
    "dominant_normalization": r"(?:主要由|根本原因是).{0,24}(?:precision|归一化)",
}


def lint_text(text, mode="final"):
    findings = []
    # History can quote disputed language only with an explicit visible erratum.
    historical = mode == "history" and re.search(
        r"erratum|superseded|纠偏|禁止|不得|不允许", text[:1600], re.I
    )
    for number, line in enumerate(text.splitlines(), 1):
        for code, pattern in PATTERNS.items():
            if re.search(pattern, line, re.I):
                findings.append({
                    "line": number, "code": code,
                    "severity": "historical_quote" if historical else "error",
                })
    return findings


def pointer(value, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Endpoint requires one JSON pointer")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def evaluate_claim(claim, root=ROOT):
    required = {"id", "statement", "ceiling", "primary_endpoint"}
    if not required <= claim.keys() or not isinstance(claim["primary_endpoint"], dict):
        raise ValueError("Each claim needs exactly one primary endpoint object")
    if "primary_endpoints" in claim or "alternative_endpoints" in claim:
        raise ValueError("Alternative primary endpoints are forbidden")
    endpoint = claim["primary_endpoint"]
    if set(endpoint) != {"path", "pointer", "sha256", "kind"}:
        raise ValueError("Invalid endpoint schema")
    ceiling = claim["ceiling"]
    if ceiling not in {
        "historical_support", "synthetic_scorer_diagnostic",
        "bounded_absence", "candidate_filtering", "unconfirmed", "withdrawn",
    }:
        raise ValueError("Unknown evidence ceiling")
    path = (root / endpoint["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Endpoint escapes project root")
    reasons = []
    value = None
    if not path.is_file():
        reasons.append("primary_endpoint_missing")
    elif not endpoint["sha256"]:
        reasons.append("primary_endpoint_not_hash_locked")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != endpoint["sha256"]:
        reasons.append("primary_endpoint_hash_mismatch")
    else:
        try:
            value = pointer(json.loads(path.read_text()), endpoint["pointer"])
            if value is None:
                reasons.append("primary_endpoint_null")
        except (ValueError, KeyError, IndexError, TypeError):
            reasons.append("primary_endpoint_pointer_missing")
    # Existing data cannot elevate an explicitly unresolved or withdrawn claim.
    if ceiling in {"unconfirmed", "withdrawn"}:
        reasons.append("claim_ceiling_" + ceiling)
    if endpoint["kind"] == "prospective_confirmatory":
        if not isinstance(value, dict) or value.get("status") != "passed":
            reasons.append("confirmatory_endpoint_not_passed")
        gates = value.get("gates", {}) if isinstance(value, dict) else {}
        if any(gates.get(gate) is not True for gate in ("A", "B", "C", "D", "E")):
            reasons.append("confirmatory_gates_missing")
    return {
        "id": claim["id"],
        "effective_status": "withdrawn" if ceiling == "withdrawn" else (
            "unconfirmed" if reasons else ceiling
        ),
        "reasons": reasons,
        "primary_endpoint": endpoint["path"] + "#" + endpoint["pointer"],
    }


def evaluate_registry(registry, root=ROOT):
    claims = registry["claims"]
    ids = [claim["id"] for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim id")
    return [evaluate_claim(claim, root) for claim in claims]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--mode", choices=["final", "history"], default="final")
    parser.add_argument("--claims", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"text_findings": [], "claims": []}
    for path in args.paths:
        for finding in lint_text(path.read_text(), args.mode):
            report["text_findings"].append({"path": str(path), **finding})
    if args.claims:
        report["claims"] = evaluate_registry(json.loads(args.claims.read_text()))
    report["ok"] = not any(
        row["severity"] == "error" for row in report["text_findings"]
    )
    if args.claims:
        registry = json.loads(args.claims.read_text())
        for claim, result in zip(registry["claims"], report["claims"]):
            if claim["ceiling"] not in {"unconfirmed", "withdrawn"} and result["reasons"]:
                report["ok"] = False
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
