"""Run the authorized DeepSeek preflight with frozen local NLTK resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import run_swan_positive_deepseek_preflight as base
import run_swan_positive_deepseek_preflight_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = base.OUTPUT
AMENDMENT = OUTPUT / "protocol_amendment_v3.json"
REQUIRED_NLTK = {
    "propbank": ROOT / ".nltk_data/corpora/propbank.zip",
    "punkt_tab": (
        ROOT
        / ".nltk_data/tokenizers/punkt_tab/english/ortho_context.tab"
    ),
}


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


def freeze_json(path: Path, payload) -> dict:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen v3 artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return payload


def amendment_payload(created_at=None) -> dict:
    return {
        "schema_version": 3,
        "created_at": created_at or utc_now(),
        "status": "frozen_local_nltk_binding",
        "base_manifest_sha256": sha256(base.MANIFEST),
        "base_budget_estimate_sha256": sha256(base.BUDGET),
        "v2_amendment_sha256": sha256(v2.AMENDMENT),
        "v2_authorization_sha256": sha256(v2.AUTHORIZATION),
        "v2_runner_sha256": sha256(Path(v2.__file__)),
        "v3_runner_sha256": sha256(Path(__file__)),
        "observed_blocker": (
            "Official amr_utils.py unconditionally calls nltk.download for "
            "propbank and punkt_tab during import, causing an external "
            "connection despite both frozen resources being present."
        ),
        "binding": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in REQUIRED_NLTK.items()
        },
        "behavior": (
            "During official module import only, return True for the two "
            "verified local resources and reject every other download name."
        ),
        "unchanged": (
            "Model request, source identities, templates, seeds, thresholds, "
            "retry limits, parser, S2Match, budget and decision gate."
        ),
        "prior_paid_calls": {
            "model_probe": 1,
            "generation": 0,
        },
    }


def prepare() -> dict:
    v2.check()
    if any(not path.is_file() for path in REQUIRED_NLTK.values()):
        raise FileNotFoundError("Frozen local NLTK resource is missing")
    if AMENDMENT.exists():
        return check()
    return freeze_json(AMENDMENT, amendment_payload())


def check() -> dict:
    v2.check()
    saved = load_json(AMENDMENT)
    if saved != amendment_payload(saved["created_at"]):
        raise ValueError("Local NLTK binding amendment differs")
    return saved


def run(key_file: Path) -> dict:
    check()
    authorization = v2.check()["authorization"]
    import nltk

    original_download = nltk.download
    original_authorization = base.require_authorization
    original_parser = base.parse_response

    def local_download(package: str, *args, **kwargs):
        del args, kwargs
        path = REQUIRED_NLTK.get(package)
        if path is None or not path.is_file():
            raise RuntimeError(
                f"Unfrozen NLTK download request rejected: {package}"
            )
        return True

    nltk.download = local_download
    base.require_authorization = lambda: authorization
    base.parse_response = v2.parse_response
    try:
        return base.run(key_file)
    finally:
        nltk.download = original_download
        base.require_authorization = original_authorization
        base.parse_response = original_parser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "run"))
    parser.add_argument("--key-file", type=Path, default=ROOT / "api.txt")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "check":
        result = check()
    else:
        result = run(args.key_file)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
