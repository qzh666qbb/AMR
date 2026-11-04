"""Independent offline audit of the SWAN v3 execution contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
CONTRACT_PATH = OUTPUT / "execution_contract_v3.json"
AUDIT_PATH = OUTPUT / "execution_contract_audit_v3.json"
REPORT_PATH = OUTPUT / "execution_readiness_report_v3.md"
EXPECTED_SWAN = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
EXPECTED_S2MATCH = "711a231d3600139662fe048460e10d773ff8e214"
FORBIDDEN = (
    "budget_authorization.json",
    "model_profile_v3.json",
    "model_probe_v3.json",
    "calls",
    "generation_attempts",
    "generation_slots",
    "generated_documents",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def current_distribution_inventory() -> dict:
    inventory = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = name.lower().replace("_", "-")
        record = distribution.read_text("RECORD")
        inventory[normalized] = {
            "name": name,
            "version": distribution.version,
            "record_sha256": (
                hashlib.sha256(record.encode("utf-8")).hexdigest()
                if record is not None
                else None
            ),
        }
    return dict(sorted(inventory.items()))


def all_true(value) -> bool:
    if isinstance(value, dict):
        return all(all_true(item) for item in value.values())
    return value is True


def audit(created_at=None) -> dict:
    contract = load_json(CONTRACT_PATH)
    swan_root = ROOT / contract["official_checkouts"]["swan"]["root"]
    metric_root = (
        ROOT
        / contract["official_checkouts"]["amr_metric_suite"]["root"]
    )
    dependency_checks = {}
    for category, groups in contract["dependencies"].items():
        if category == "resources":
            records = {
                relative: record
                for group in groups.values()
                for relative, record in group.items()
            }
        else:
            records = groups
        dependency_checks[category] = all(
            (ROOT / relative).is_file()
            and sha256(ROOT / relative) == record["sha256"]
            and (ROOT / relative).stat().st_size == record["bytes"]
            for relative, record in records.items()
        )

    requirements = contract["python_environment"][
        "official_requirements"
    ]
    package_checks = {}
    for name, state in requirements.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        package_checks[name] = (
            state["required"] == state["installed"] == installed
            and state["exact_match"] is True
        )

    amendment = (
        OUTPUT / "protocol_amendment_v3.md"
    ).read_text(encoding="utf-8")
    checks = {
        "contract": {
            "status_frozen": (
                contract["status"] == "frozen_execution_contract"
            ),
            "local_execution_ready": (
                contract["execution_ready"] is True
            ),
            "remote_execution_unauthorized": (
                contract["remote_execution_authorized"] is False
            ),
            "source_manifest_hash": (
                contract["source_manifest_sha256"]
                == sha256(OUTPUT / "source_manifest.json")
            ),
            "runtime_v2_hash": (
                contract["runtime_v2_sha256"]
                == sha256(OUTPUT / "runtime_v2.json")
            ),
            "amendment_hash": (
                contract["protocol_amendment_sha256"]
                == sha256(OUTPUT / "protocol_amendment_v3.md")
            ),
        },
        "checkouts": {
            "swan_head": (
                git(swan_root, "rev-parse", "HEAD") == EXPECTED_SWAN
            ),
            "swan_clean": not git(
                swan_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            "s2match_head": (
                git(metric_root, "rev-parse", "HEAD")
                == EXPECTED_S2MATCH
            ),
            "s2match_clean": not git(
                metric_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
        },
        "dependencies": dependency_checks,
        "python": {
            "running_prefix_matches": (
                str(Path(sys.prefix).resolve())
                == str(
                    Path(
                        contract["python_environment"]["prefix"]
                    ).resolve()
                )
            ),
            "all_official_pins_match": all(package_checks.values()),
            "distribution_inventory_matches": (
                current_distribution_inventory()
                == contract["python_environment"][
                    "distribution_inventory"
                ]
            ),
            "spacy_model_380": (
                importlib.metadata.version("en_core_web_sm") == "3.8.0"
            ),
        },
        "protocol": {
            "ordered_bank_frozen": (
                contract["injection"]["bank_order"]
                == "official JSON file order"
                and len(contract["injection"]["ordered_bank_sha256"]) == 50
                and len(
                    set(contract["injection"]["ordered_bank_sha256"])
                )
                == 50
            ),
            "effective_top_p_omitted": (
                contract["model_request"]["declared_top_p"] == 0.9
                and contract["model_request"]["effective_top_p"] is None
            ),
            "usd_budget_bound_frozen": (
                contract["model_request"][
                    "input_price_per_million_usd"
                ]
                == 3.30
                and contract["model_request"][
                    "output_price_per_million_usd"
                ]
                == 16.50
                and contract["model_request"][
                    "input_token_overhead_reserve"
                ]
                == 1024
                and "max_cost_usd"
                in contract["injection"]["usd_budget_enforcement"]
            ),
            "attempt_cap": (
                contract["injection"]["pilot_max_generation_calls"]
                == 4000
                and contract["injection"]["pilot_model_probe_calls"] == 1
            ),
            "amendment_records_rng_and_identity_corrections": all(
                phrase in amendment
                for phrase in (
                    "Deterministic template sampling",
                    "Deterministic S2Match calls",
                    "effective_top_p=null",
                    "top-level response schema does not return `modelId`",
                )
            ),
        },
        "execution_boundary": {
            "no_authorization_or_execution_artifacts": not any(
                (OUTPUT / name).exists() for name in FORBIDDEN
            ),
            "all_runtime_counts_zero": all(
                value == 0
                for value in contract["resource_counts"].values()
            ),
            "authorization_gate_pending": contract["execution_gates"][
                "explicit_budget_authorization_pending"
            ],
            "probe_gate_pending": contract["execution_gates"][
                "model_probe_pending"
            ],
            "main_not_authorized": (
                contract["execution_gates"][
                    "main_generation_authorized"
                ]
                is False
            ),
        },
    }
    passed = all_true(checks)
    return {
        "schema_version": 3,
        "created_at": created_at or utc_now(),
        "status": (
            "verified_local_execution_ready_remote_unauthorized"
            if passed
            else "execution_contract_audit_failed"
        ),
        "all_checks_pass": passed,
        "execution_contract_sha256": sha256(CONTRACT_PATH),
        "checks": checks,
        "summary": {
            "official_requirement_count": len(requirements),
            "distribution_count": len(
                contract["python_environment"][
                    "distribution_inventory"
                ]
            ),
            "frozen_dependency_files": sum(
                len(group)
                if category != "resources"
                else sum(len(files) for files in group.values())
                for category, group in contract["dependencies"].items()
            ),
            "sources": 35,
            "pilot": 5,
            "main": 30,
            "execution_ready": contract["execution_ready"],
            "remote_execution_authorized": contract[
                "remote_execution_authorized"
            ],
        },
        "resource_counts": contract["resource_counts"],
        "next_gate": (
            "Explicit user approval of both maximum USD and model-call "
            "budgets, followed by the frozen profile lookup and one model "
            "probe. Pilot generation remains blocked."
        ),
    }


def report_text(result: dict) -> str:
    summary = result["summary"]
    return f"""# SWAN execution contract v3 audit

The independent offline audit passed. The local execution environment is ready
but remote execution is not authorized.

- Official top-level package pins matched: {summary["official_requirement_count"]}
- Frozen installed distributions: {summary["distribution_count"]}
- Frozen dependency files: {summary["frozen_dependency_files"]}
- Frozen allocation: {summary["pilot"]} pilot + {summary["main"]} main
- AWS control-plane, probe, and generation calls: 0
- Parser, S2Match, detector, and attack calls: 0

The runner binds the clean official SWAN checkout, a clean pinned
amr-metric-suite checkout plus the official SWAN patch, deterministic ordered
template sampling, deterministic S2Match RNG handling, per-request journaling,
and explicit fallback/acceptance labels.

No `budget_authorization.json`, model profile evidence, model probe, call
journal, generated slot, or generated document exists. The next gate is
explicit user approval of both maximum USD and model-call budgets. Only then
may the profile lookup and one frozen probe run; pilot generation remains
separately gated by that probe.
"""


def write_frozen(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen audit output differs: {path}")
    path.write_text(text, encoding="utf-8")


def create() -> dict:
    if AUDIT_PATH.exists() or REPORT_PATH.exists():
        if not AUDIT_PATH.is_file() or not REPORT_PATH.is_file():
            raise RuntimeError("Partial execution audit requires review")
        return check()
    result = audit()
    if not result["all_checks_pass"]:
        raise ValueError("Execution contract audit failed")
    write_frozen(
        AUDIT_PATH,
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
    )
    write_frozen(REPORT_PATH, report_text(result))
    return result


def check() -> dict:
    saved = load_json(AUDIT_PATH)
    live = audit(saved["created_at"])
    if saved != live:
        raise ValueError("Frozen execution audit differs from live state")
    if REPORT_PATH.read_text(encoding="utf-8") != report_text(saved):
        raise ValueError("Execution audit report differs")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("create", "check"))
    args = parser.parse_args()
    result = create() if args.stage == "create" else check()
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_checks_pass": result["all_checks_pass"],
                "summary": result["summary"],
                "resource_counts": result["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
