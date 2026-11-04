"""Resume DeepSeek preflight with a response text-hash adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import run_swan_positive_deepseek_preflight as base
import run_swan_positive_deepseek_preflight_v2 as v2
import run_swan_positive_deepseek_preflight_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = base.OUTPUT
AMENDMENT = OUTPUT / "protocol_amendment_v4.json"
FAILURE = OUTPUT / "generation_failures/pilot-000-slot-00.json"
SUPERSEDED_FAILURE = (
    OUTPUT / "superseded_generation_failures/pilot-000-slot-00.json"
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


def freeze_json(path: Path, payload) -> dict:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Frozen v4 artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return payload


def amendment_payload(failure_hash: str, created_at=None) -> dict:
    first_call = OUTPUT / (
        "deepseek_calls/pilot-000-slot-00-attempt-00.json"
    )
    return {
        "schema_version": 4,
        "created_at": created_at or utc_now(),
        "status": "frozen_response_hash_adapter",
        "base_manifest_sha256": sha256(base.MANIFEST),
        "base_budget_estimate_sha256": sha256(base.BUDGET),
        "v2_authorization_sha256": sha256(v2.AUTHORIZATION),
        "v3_amendment_sha256": sha256(v3.AMENDMENT),
        "v3_runner_sha256": sha256(Path(v3.__file__)),
        "v4_runner_sha256": sha256(Path(__file__)),
        "observed_failure": {
            "error_type": "KeyError",
            "missing_field": "text_sha256",
            "failure_sha256": failure_hash,
            "first_generation_call_sha256": sha256(first_call),
            "first_generation_call_status": "complete",
            "first_generation_call_retransmitted": False,
        },
        "repair": (
            "Compute SHA-256 from the already journaled response text on "
            "replay and include the same derived field in future parsed "
            "responses."
        ),
        "superseded_failure_path": str(
            SUPERSEDED_FAILURE.relative_to(ROOT)
        ),
        "unchanged": (
            "Provider response text, model request, sources, templates, "
            "seeds, parser, S2Match, thresholds, retries, budget and gate."
        ),
    }


def prepare() -> dict:
    v3.check()
    if AMENDMENT.exists():
        return check()
    if not FAILURE.is_file() or SUPERSEDED_FAILURE.exists():
        raise RuntimeError("Expected one unsuperseded adapter failure")
    failure = load_json(FAILURE)
    if (
        failure.get("error_type") != "KeyError"
        or failure.get("error") is not None
    ):
        raise ValueError("Unexpected failure evidence")
    failure_hash = sha256(FAILURE)
    amendment = freeze_json(
        AMENDMENT,
        amendment_payload(failure_hash),
    )
    SUPERSEDED_FAILURE.parent.mkdir(parents=True, exist_ok=True)
    os.replace(FAILURE, SUPERSEDED_FAILURE)
    return amendment


def check() -> dict:
    v3.check()
    amendment = load_json(AMENDMENT)
    failure_hash = amendment["observed_failure"]["failure_sha256"]
    if amendment != amendment_payload(
        failure_hash,
        amendment["created_at"],
    ):
        raise ValueError("Response-hash amendment differs")
    if FAILURE.exists() or sha256(SUPERSEDED_FAILURE) != failure_hash:
        raise ValueError("Superseded adapter failure was not preserved")
    return amendment


def parse_response(payload: dict) -> dict:
    response = v2.parse_response(payload)
    response["text_sha256"] = base.text_sha256(response["text"])
    return response


def run(key_file: Path) -> dict:
    check()
    authorization = v2.check()["authorization"]
    import nltk

    original_download = nltk.download
    original_authorization = base.require_authorization
    original_parser = base.parse_response
    original_request = base.DeepSeekJournal.request

    def local_download(package: str, *args, **kwargs):
        del args, kwargs
        path = v3.REQUIRED_NLTK.get(package)
        if path is None or not path.is_file():
            raise RuntimeError(
                f"Unfrozen NLTK download request rejected: {package}"
            )
        return True

    def request_with_text_hash(journal, *args, **kwargs):
        response = original_request(journal, *args, **kwargs)
        response.setdefault(
            "text_sha256",
            base.text_sha256(response["text"]),
        )
        return response

    nltk.download = local_download
    base.require_authorization = lambda: authorization
    base.parse_response = parse_response
    base.DeepSeekJournal.request = request_with_text_hash
    try:
        return base.run(key_file)
    finally:
        nltk.download = original_download
        base.require_authorization = original_authorization
        base.parse_response = original_parser
        base.DeepSeekJournal.request = original_request


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
