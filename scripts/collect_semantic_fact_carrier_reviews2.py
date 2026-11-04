"""Versioned v2 entry point for the unchanged isolated human-review collector."""
import argparse
import json
from pathlib import Path

import run_semantic_fact_carrier_pilot2 as v2
import collect_semantic_fact_carrier_reviews as handoff


def verify():
    v2.runtime.verify(v2.OUT, v2.PRIVATE)
    release = v2.runtime.load(v2.OUT / "COLLECTION_RELEASE_FREEZE.json")
    if release["generation_freeze_sha256"] != v2.runtime.sha(v2.OUT / "FREEZE.json"):
        raise RuntimeError("Generation freeze drift")
    for source, expected in release["sources"].items():
        if v2.runtime.sha(v2.ROOT / source) != expected:
            raise RuntimeError("Collector release source drift")
    return {"status": "PASS", "version": v2.OUT.name, "collection_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "collect"))
    args = parser.parse_args()
    result = verify()
    if args.command == "collect":
        with v2.runtime.Exclusive(v2.PRIVATE):
            result = handoff.collect(v2.OUT, v2.PRIVATE)
    print(json.dumps(result, ensure_ascii=False, indent=2))
