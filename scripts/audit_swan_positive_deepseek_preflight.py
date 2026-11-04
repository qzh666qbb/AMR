"""Independent pre-execution audit for the DeepSeek SWAN preflight."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_deepseek_preflight1"
MANIFEST = OUTPUT / "manifest.json"
BUDGET = OUTPUT / "budget_estimate.json"
TEMPLATE = OUTPUT / "budget_authorization_template.json"
AUTHORIZATION = OUTPUT / "budget_authorization.json"
AUDIT = OUTPUT / "pre_execution_audit.json"
REPORT = OUTPUT / "pre_execution_report.md"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen DeepSeek audit differs: {path}")
    path.write_text(text, encoding="utf-8")


def audit() -> dict:
    manifest = load_json(MANIFEST)
    budget = load_json(BUDGET)
    template = load_json(TEMPLATE)
    dependencies = {
        relative: (
            (ROOT / relative).is_file()
            and sha256(ROOT / relative) == expected
        )
        for relative, expected in manifest["dependencies"].items()
    }
    generation_bound = sum(
        row["peak_all_cache_miss_cost_usd"] * row["calls_reserved"]
        for row in budget["generation_reserves"]
    )
    total_bound = (
        budget["probe_reserve"]["peak_all_cache_miss_cost_usd"]
        + generation_bound
    )
    recommended = math.ceil(total_bound * 1.25 * 100) / 100
    credential_path = ROOT / "api.txt"
    credential_mode = stat.S_IMODE(credential_path.stat().st_mode)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(credential_path)],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MANIFEST, BUDGET, TEMPLATE)
    )
    checks = {
        "dependencies_match": all(dependencies.values()),
        "remote_unauthorized": (
            manifest["status"] == "frozen_remote_unauthorized"
            and not AUTHORIZATION.exists()
        ),
        "current_model_frozen": (
            manifest["backend"]["model_requested"]
            == manifest["backend"]["required_model_returned"]
            == "deepseek-flash"
            and manifest["backend"]["thinking"] == "disabled"
        ),
        "non_equivalence_explicit": (
            manifest["backend"]["not_official_equivalent"] is True
        ),
        "protocol_bounds_match": (
            manifest["protocol"]["documents"] == 5
            and manifest["protocol"]["slots_per_document"] == 1
            and manifest["protocol"]["max_trials_per_slot"] == 10
            and manifest["protocol"]["maximum_generation_calls"] == 50
            and manifest["protocol"]["maximum_total_calls"] == 51
        ),
        "pricing_is_peak_all_cache_miss": (
            budget["pricing"][
                "peak_cache_miss_input_usd_per_million"
            ]
            == 0.30
            and budget["pricing"]["peak_output_usd_per_million"]
            == 1.20
            and "Peak pricing"
            in budget["pricing"]["assumption"]
        ),
        "budget_recomputed": (
            abs(
                total_bound - budget["conservative_total_bound_usd"]
            )
            <= 1e-6
            and budget["recommended_authorization"]
            == {
                "max_cost_usd": recommended,
                "max_model_calls": 51,
                "contingency": (
                    "25 percent, rounded up to one cent"
                ),
            }
        ),
        "template_is_not_authorization": (
            template["status"] == "awaiting_explicit_user_approval"
            and template["credential_rotated"] is None
            and template["explicit_user_statement"] is None
            and template["approved_at"] is None
            and template["manifest_sha256"] == sha256(MANIFEST)
            and template["budget_estimate_sha256"] == sha256(BUDGET)
        ),
        "no_remote_or_downstream_artifacts": (
            not (OUTPUT / "deepseek_calls").exists()
            and not (OUTPUT / "generation_attempts").exists()
            and not (OUTPUT / "generation_slots").exists()
            and not (OUTPUT / "detection").exists()
            and not (OUTPUT / "attacks").exists()
        ),
        "all_resource_counts_zero": all(
            value == 0
            for value in manifest["resource_counts"].values()
        ),
        "credential_file_locally_restricted_and_ignored": (
            credential_mode & 0o077 == 0 and ignored
        ),
        "credential_not_frozen": (
            "api.txt" not in serialized
            and "DEEPSEEK_API_KEY" not in serialized
            and "sk-" not in serialized
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "verified_awaiting_rotated_credential_and_authorization"
            if all(checks.values())
            else "deepseek_pre_execution_audit_failed"
        ),
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "dependency_checks": dependencies,
        "conservative_total_bound_usd": round(total_bound, 6),
        "recommended_authorization": (
            budget["recommended_authorization"]
        ),
        "credential": {
            "content_read_or_frozen": False,
            "mode": oct(credential_mode),
            "git_ignored": ignored,
            "rotation_required_because_preexisting_value_was_exposed": True,
        },
        "resource_counts": manifest["resource_counts"],
        "claim_boundary": (
            "Pre-execution evidence only; no model output, SWAN positive, "
            "detector result, or attack result."
        ),
    }


def report_text(result: dict) -> str:
    recommendation = result["recommended_authorization"]
    return f"""# DeepSeek SWAN preflight readiness

The independent pre-execution audit passed.

- Model: `deepseek-flash` (DeepSeek-V4.1-Flash)
- Scope: 5 frozen sources, first slot only
- Maximum calls: 1 probe + 50 generations = 51
- Conservative peak/all-cache-miss bound:
  ${result["conservative_total_bound_usd"]:.6f}
- Proposed authorization: ${recommendation["max_cost_usd"]:.2f} and
  {recommendation["max_model_calls"]} calls
- Active authorization: absent
- DeepSeek, Bedrock, parser, S2Match, detector, and attack calls: 0

The local credential file is mode 600 and ignored by Git, but the preexisting
credential value was exposed in conversation and must be revoked and replaced.
The inactive template cannot authorize execution. A rotated credential and an
explicit approval of both limits are required before the single model probe.

This is pre-execution evidence only. DeepSeek is a non-equivalent substitute
for the official Claude generation backend.
"""


def main() -> None:
    result = audit()
    if not result["all_checks_pass"]:
        raise ValueError("DeepSeek pre-execution audit failed")
    write_frozen(
        AUDIT,
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
    )
    write_frozen(REPORT, report_text(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "conservative_total_bound_usd": result[
                    "conservative_total_bound_usd"
                ],
                "recommended_authorization": result[
                    "recommended_authorization"
                ],
                "resource_counts": result["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
