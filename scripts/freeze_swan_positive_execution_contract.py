"""Freeze the locally ready, still-unauthorized SWAN execution contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_collection1"
OFFICIAL_ROOT = ROOT / "third_party/SWAN_official_2ca0c207_clean"
S2MATCH_ROOT = ROOT / "third_party/swan_s2match_runtime_711a231d"
VENV = ROOT / ".venv-swan-positive"
CONTRACT = OUTPUT / "execution_contract_v3.json"
EXPECTED_SWAN_COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
EXPECTED_S2MATCH_COMMIT = "711a231d3600139662fe048460e10d773ff8e214"
MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

CODE_FILES = (
    "scripts/run_swan_positive_collection.py",
    "scripts/test_swan_positive_execution.py",
    "scripts/build_swan_s2match_runtime.py",
    "scripts/freeze_swan_positive_execution_contract.py",
    "scripts/audit_swan_positive_execution_contract.py",
    "scripts/estimate_swan_positive_budget.py",
)
CONTROL_FILES = (
    "experiments/swan_positive_collection1/protocol_amendment_v3.md",
    "experiments/swan_positive_collection1/provenance_correction.json",
    "experiments/swan_positive_collection1/runtime_v2.json",
    "experiments/swan_positive_collection1/source_manifest.json",
)
RESOURCE_GROUPS = {
    "parser": (
        "models/model_parse_xfm_bart_large-v0_1_0/amrlib_meta.json",
        "models/model_parse_xfm_bart_large-v0_1_0/config.json",
        (
            "models/model_parse_xfm_bart_large-v0_1_0/"
            "model_parse_xfm_bart_large.json"
        ),
        "models/model_parse_xfm_bart_large-v0_1_0/pytorch_model.bin",
    ),
    "tokenizer": (
        "models/bart-large-tokenizer/config.json",
        "models/bart-large-tokenizer/merges.txt",
        "models/bart-large-tokenizer/tokenizer.json",
        "models/bart-large-tokenizer/tokenizer_config.json",
        "models/bart-large-tokenizer/vocab.json",
    ),
    "glove": ("vectors/glove.6B.100d.txt",),
    "nltk": (
        ".nltk_data/corpora/propbank.zip",
        ".nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
        ".nltk_data/tokenizers/punkt_tab/english/collocations.tab",
        ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
        ".nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    ),
    "s2match": (
        "third_party/swan_s2match_runtime_711a231d/__init__.py",
        "third_party/swan_s2match_runtime_711a231d/amr_py3.py",
        "third_party/swan_s2match_runtime_711a231d/helpers.py",
        "third_party/swan_s2match_runtime_711a231d/s2match.py",
        "third_party/swan_s2match_runtime_711a231d/provenance.json",
    ),
}
FORBIDDEN_EXECUTION_ARTIFACTS = (
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


def freeze_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen execution contract differs: {path}")
    path.write_text(text, encoding="utf-8")


def parse_requirements() -> dict:
    path = OFFICIAL_ROOT / "requirements.txt"
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"Official requirement is not pinned: {line}")
        name, version = line.split("==", 1)
        result[name] = version
    return result


def package_record(name: str) -> dict:
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    return {
        "version": distribution.version,
        "record_sha256": (
            hashlib.sha256(record.encode("utf-8")).hexdigest()
            if record is not None
            else None
        ),
    }


def package_environment() -> dict:
    required = parse_requirements()
    states = {}
    for name, expected in required.items():
        try:
            record = package_record(name)
        except importlib.metadata.PackageNotFoundError:
            record = {"version": None, "record_sha256": None}
        states[name] = {
            "required": expected,
            "installed": record["version"],
            "exact_match": record["version"] == expected,
            "record_sha256": record["record_sha256"],
        }
    if not all(state["exact_match"] for state in states.values()):
        raise ValueError("Official Python requirements are not exact")

    inventory = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = name.lower().replace("_", "-")
        record = distribution.read_text("RECORD")
        value = {
            "name": name,
            "version": distribution.version,
            "record_sha256": (
                hashlib.sha256(record.encode("utf-8")).hexdigest()
                if record is not None
                else None
            ),
        }
        prior = inventory.get(normalized)
        if prior is not None and prior != value:
            raise ValueError(f"Duplicate distribution differs: {name}")
        inventory[normalized] = value

    if importlib.util.find_spec("en_core_web_sm") is None:
        raise ValueError("spaCy en_core_web_sm is not installed")
    spacy_model = package_record("en_core_web_sm")
    if spacy_model["version"] != "3.8.0":
        raise ValueError("Expected en_core_web_sm==3.8.0")

    expected_prefix = str(VENV.resolve())
    if str(Path(sys.prefix).resolve()) != expected_prefix:
        raise ValueError(
            "Contract must be frozen from .venv-swan-positive"
        )
    return {
        "executable": sys.executable,
        "executable_sha256": sha256(Path(sys.executable).resolve()),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pyvenv_cfg_sha256": sha256(VENV / "pyvenv.cfg"),
        "official_requirements_sha256": sha256(
            OFFICIAL_ROOT / "requirements.txt"
        ),
        "official_requirements": states,
        "all_official_requirements_exact": True,
        "spacy_model": spacy_model,
        "distribution_inventory": dict(sorted(inventory.items())),
    }


def git_evidence(root: Path, expected_commit: str) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if head != expected_commit or status:
        raise ValueError(f"Git checkout is not exact and clean: {root}")
    return {
        "root": str(root.relative_to(ROOT)),
        "head": head,
        "tree_oid": git("rev-parse", f"{head}^{{tree}}"),
        "origin": git("config", "--get", "remote.origin.url"),
        "clean": True,
        "tracked_file_count": len(git("ls-files").splitlines()),
    }


def hash_paths(paths) -> dict:
    result = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen dependency missing: {path}")
        result[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def resource_evidence() -> dict:
    resources = {
        name: hash_paths(paths)
        for name, paths in RESOURCE_GROUPS.items()
    }
    sys.path.insert(0, str(S2MATCH_ROOT))
    spec = importlib.util.find_spec("s2match")
    if spec is None or spec.origin != str(S2MATCH_ROOT / "s2match.py"):
        raise ValueError("Frozen S2Match runtime is not importable")
    import nltk

    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    nltk.data.find("corpora/propbank.zip")
    nltk.data.find("tokenizers/punkt_tab/english/")
    return resources


def build_contract() -> dict:
    if any((OUTPUT / name).exists() for name in FORBIDDEN_EXECUTION_ARTIFACTS):
        raise ValueError("Execution artifact exists before contract freeze")
    source = load_json(OUTPUT / "source_manifest.json")
    runtime_v2 = load_json(OUTPUT / "runtime_v2.json")
    s2match = load_json(S2MATCH_ROOT / "provenance.json")
    if (
        len(source["sources"]) != 35
        or source["cohort_counts"] != {"pilot": 5, "main": 30}
        or runtime_v2["official_swan"]["commit"] != EXPECTED_SWAN_COMMIT
        or s2match["source"]["commit"] != EXPECTED_S2MATCH_COMMIT
    ):
        raise ValueError("Frozen provenance does not match v3 contract")
    bank_records = load_json(
        OFFICIAL_ROOT / "amr_bank/banks/amr_bank_50.json"
    )
    bank = [row["normalized_template_amr"] for row in bank_records]
    if len(bank) != 50 or len(set(bank)) != 50:
        raise ValueError("Official ordered bank is not 50 unique templates")
    return {
        "schema_version": 3,
        "created_at": utc_now(),
        "status": "frozen_execution_contract",
        "execution_ready": True,
        "remote_execution_authorized": False,
        "source_manifest_sha256": sha256(
            OUTPUT / "source_manifest.json"
        ),
        "runtime_v2_sha256": sha256(OUTPUT / "runtime_v2.json"),
        "protocol_amendment_sha256": sha256(
            OUTPUT / "protocol_amendment_v3.md"
        ),
        "runner_sha256": sha256(
            ROOT / "scripts/run_swan_positive_collection.py"
        ),
        "official_injection_sha256": sha256(
            OFFICIAL_ROOT / "injection/injection_amr_utils.py"
        ),
        "official_checkouts": {
            "swan": git_evidence(
                OFFICIAL_ROOT,
                EXPECTED_SWAN_COMMIT,
            ),
            "amr_metric_suite": git_evidence(
                ROOT / "third_party/amr-metric-suite_official_clean",
                EXPECTED_S2MATCH_COMMIT,
            ),
        },
        "model_request": {
            "inference_profile_id": MODEL,
            "region": "us-east-1",
            "temperature": 0.7,
            "declared_top_p": 0.9,
            "effective_top_p": None,
            "max_new_tokens": 50,
            "input_price_per_million_usd": 3.30,
            "output_price_per_million_usd": 16.50,
            "input_token_overhead_reserve": 1024,
            "pricing_interpretation": (
                "Conservative US geographic-profile bound: published "
                "3/15 USD base rates plus 10 percent."
            ),
        },
        "injection": {
            "slots_per_document": 16,
            "max_trials_per_slot": 50,
            "max_templates_per_slot": 10,
            "retries_per_template": 5,
            "acceptance_s2match": 0.75,
            "official_bank_size": 50,
            "bank_order": "official JSON file order",
            "ordered_bank_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in bank
            ],
            "pilot_max_generation_calls": 4000,
            "pilot_model_probe_calls": 1,
            "usd_budget_enforcement": (
                "Before transport, completed estimated cost plus the "
                "current serialized-request reserve must not exceed the "
                "approved max_cost_usd."
            ),
        },
        "python_environment": package_environment(),
        "dependencies": {
            "code": hash_paths(CODE_FILES),
            "control": hash_paths(CONTROL_FILES),
            "resources": resource_evidence(),
        },
        "execution_gates": {
            "explicit_budget_authorization_pending": True,
            "model_profile_lookup_pending": True,
            "model_probe_pending": True,
            "pilot_generation_pending": True,
            "main_generation_authorized": False,
        },
        "resource_counts": {
            "aws_control_plane_calls": 0,
            "model_probe_calls": 0,
            "generation_calls": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
    }


def create() -> dict:
    if CONTRACT.exists():
        return check()
    contract = build_contract()
    freeze_json(CONTRACT, contract)
    return check()


def check() -> dict:
    saved = load_json(CONTRACT)
    live = build_contract()
    live["created_at"] = saved["created_at"]
    if saved != live:
        raise ValueError("Frozen execution contract differs from live state")
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
                "execution_ready": result["execution_ready"],
                "remote_execution_authorized": result[
                    "remote_execution_authorized"
                ],
                "official_requirement_count": len(
                    result["python_environment"][
                        "official_requirements"
                    ]
                ),
                "resource_counts": result["resource_counts"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
