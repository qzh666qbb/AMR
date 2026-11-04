"""Supplement prior-manifest overlap checking without changing frozen inputs."""
from collections import Counter
import json
from pathlib import Path
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.swan_last_round_inputs import OUT, PRIVATE, identity, sha
from scripts.analyze_swan_blind_eval import save_json


def audit():
    human = json.loads((PRIVATE / "HUMAN_DOCUMENTS.json").read_text())["documents"]
    prior_titles, prior_hashes, counts, inventory = set(), set(), Counter(), []
    include = (
        "source_manifest", "source_records", "sealed_sources", "pilot_sources",
        "sources.json", "source_screening", "source_assignment",
        "source_cohort", "human_locator", "documents.json",
    )
    paths = sorted(p for p in (ROOT / "experiments").rglob("*")
                   if p.suffix in (".json", ".jsonl") and OUT not in p.parents
                   and any(n in p.name.casefold() for n in include))
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    if key in ("text", "source_text", "source_window", "original_text",
                               "reference_text", "source_anchors"):
                        prior_hashes.add(identity(value))
                        counts["materialized_text_fields"] += 1
                    elif key in ("normalized_source_sha256", "normalized_text_sha256",
                                 "normalized_window_sha256"):
                        prior_hashes.add(value)
                        counts["prior_normalized_identity_fields"] += 1
                    elif key in ("article_title", "source_title", "wikipedia_title"):
                        prior_titles.add(identity(value))
                        counts["article_title_fields"] += 1
                    elif key in ("url", "source_url"):
                        u = urllib.parse.urlparse(value)
                        if u.hostname and u.hostname.endswith("wikipedia.org") and "/wiki/" in u.path:
                            title = urllib.parse.unquote(u.path.split("/wiki/", 1)[1]).replace("_", " ")
                            prior_titles.add(identity(title))
                            counts["wikipedia_url_title_fields"] += 1
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    for p in paths:
        if p.stat().st_size > 200_000_000:
            raise RuntimeError("Prior inventory exceeds frozen read limit")
        inventory.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p)})
        values = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] \
            if p.suffix == ".jsonl" else [json.loads(p.read_text())]
        for value in values:
            visit(value)
    overlap_titles = sum(d["source_article_sha256"] in prior_titles for d in human)
    overlap_texts = sum(d["normalized_text_sha256"] in prior_hashes for d in human)
    result = {
        "schema": "swan-last-round-prior-source-audit/v1",
        "human_manifest_sha256": sha(PRIVATE / "HUMAN_DOCUMENTS.json"),
        "frozen_source_manifest_unchanged": True, "prior_inventory": inventory,
        "coverage_counts": dict(counts), "known_article_overlap": overlap_titles,
        "known_normalized_text_overlap": overlap_texts,
        "calibration_test_article_overlap": len(
            {d["source_article_sha256"] for d in human if d["split"] == "calibration"} &
            {d["source_article_sha256"] for d in human if d["split"] == "test"}),
        "ready": not overlap_titles and not overlap_texts,
        "limitation": "prior manifests without retained article identity cannot support article-level comparison",
        "resampling": False, "new_parser_or_detector_calls": 0,
    }
    save_json(OUT / "SOURCE_OVERLAP_AUDIT.json", result)
    if not result["ready"]:
        save_json(OUT / "SOURCE_STOP.json", {
            "reason": "prior_source_overlap", "permanent": True, "no_resampling": True})
    print(json.dumps({k: result[k] for k in (
        "known_article_overlap", "known_normalized_text_overlap",
        "calibration_test_article_overlap", "ready", "coverage_counts")}))


if __name__ == "__main__":
    audit()
