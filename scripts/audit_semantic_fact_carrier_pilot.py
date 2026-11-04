"""Read-only independent receipt audit; publish aggregates and SHA only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import zipfile

import run_semantic_fact_carrier_pilot as pilot


def audit(out, private):
    checks = pilot.verify(out, private)
    frozen = pilot.load(out / "FREEZE.json")
    rows = pilot.load(private / "PLAN.json")
    summary = pilot.load(out / "GENERATION_SUMMARY.json")
    expected = {entry["row"]["item_id"] + "_" + condition for entry in rows
                for condition in ("original", "paraphrase")}
    requests = {p.parent.name: p.parent for p in (private / "requests").glob("*/INTENT.json")}
    checks["exact_192_request_ids"] = set(requests) == expected and len(requests) == 192
    checks["permanent_halt_absent"] = not (private / "HALT.json").exists()
    checks["all_96_documents_exist"] = len(list((private / "documents").glob("*.json"))) == 96
    failures = []
    private_hashes, models, usage = {}, Counter(), Counter()
    freeze_at = datetime.fromisoformat(frozen["created_at"])
    original_words, para_words = [], []
    for entry in rows:
        row = entry["row"]
        identity = row["item_id"]
        path = private / "documents" / (identity + ".json")
        doc = pilot.load(path)
        private_hashes["document:" + identity] = pilot.sha(path)
        intended_original = entry["original_messages"]
        intended_para = pilot.paraphrase_messages(doc["original"]["text"])
        for condition, messages in (("original", intended_original), ("paraphrase", intended_para)):
            request_id = identity + "_" + condition
            folder = requests[request_id]
            intent, body, receipt = (pilot.load(folder / name)
                                     for name in ("INTENT.json", "BODY.json", "RECEIPT.json"))
            expected_body = {**frozen["parameters"], "messages": messages}
            tests = {
                "exact_body": body == expected_body,
                "sha_chain": intent["body_sha256"] == receipt["body_sha256"] == pilot.digest(body),
                "after_freeze": datetime.fromisoformat(intent["at"]) >= freeze_at,
                "receipt_http_200": receipt.get("http_status") == 200,
                "complete": pilot.extract(receipt)["status"] == "complete",
                "document_matches_receipt": doc[condition] == pilot.extract(receipt),
                "input_reservation": intent["input_token_bound"] == len(pilot.encode(messages)) + 512,
                "input_usage_bounded": doc[condition]["usage"]["prompt_tokens"] <= intent["input_token_bound"],
                "output_usage_bounded": doc[condition]["usage"]["completion_tokens"] <= frozen["parameters"]["max_tokens"],
            }
            failures.extend({"request": request_id, "check": k} for k, passed in tests.items() if not passed)
            for name in ("INTENT.json", "BODY.json", "RECEIPT.json"):
                private_hashes[f"request:{request_id}:{name}"] = pilot.sha(folder / name)
            models[doc[condition]["model"]] += 1
            usage.update({k: doc[condition]["usage"].get(k, 0)
                          for k in ("prompt_tokens", "completion_tokens",
                                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")})
        original_words.append(len(doc["original"]["text"].split()))
        para_words.append(len(doc["paraphrase"]["text"].split()))
    checks["all_literal_bodies_receipts_and_documents_match"] = not failures
    checks["summary_usage_independently_recomputed"] = dict(usage) == summary["usage"]
    checks["summary_model_counts_independently_recomputed"] = dict(models) == summary["provider_models_reported"]
    checks["summary_denominators"] = summary["counts"] == {"original:complete": 96, "paraphrase:complete": 96}
    checks["generation_finished"] = summary["state"] == "GENERATION_FINISHED"
    checks["no_uncertain_outcome_after_closeout"] = summary["uncertain_dispatches"] == 0
    start = pilot.load(private / "START.json")
    elapsed = datetime.fromisoformat(summary["at"]).timestamp() - start["epoch"]
    checks["wall_budget"] = 0 <= elapsed <= frozen["budget"]["max_wall_seconds"]
    checks["call_budget"] = len(requests) <= frozen["budget"]["max_calls"]
    checks["reserved_cost_budget"] = summary["reserved_usd"] <= frozen["budget"]["max_reserved_usd"]
    review = out / "private_review"
    manifest = pilot.load(review / "admin/MANIFEST.json")
    keys = pilot.load(review / "admin/KEYS.json")
    checks["human_source_documents_match"] = all(
        pilot.sha(private / "documents" / (identity + ".json")) == signature
        for identity, signature in manifest["source_sha256"].items())
    checks["blank_review_files_unchanged"] = all(pilot.sha(review / p) == s
                                               for p, s in manifest["files"].items())
    checks["review_admin_key_sha"] = pilot.sha(review / "admin/KEYS.json") == manifest["keys_sha256"]
    reviewer_items = []
    for reviewer in (1, 2):
        folder = review / f"reviewer{reviewer}"
        metadata = pilot.load(folder / "reviewer.json")
        checks[f"reviewer{reviewer}_unfilled_attestation"] = (
            metadata["human_attestation"] is False and metadata["independent_attestation"] is False
            and metadata["reviewer_code"] == "" and metadata["completed_at"] == "")
        identities = []
        import csv
        for task, count, labels in (("facts", 192, pilot.FACT_LABELS), ("pairs", 96, pilot.PAIR_LABELS)):
            with (folder / (task + ".csv")).open(newline="") as stream:
                packet = list(csv.DictReader(stream))
            mapping = keys[str(reviewer)][task]
            checks[f"reviewer{reviewer}_{task}_full_count_and_unique_ids"] = (
                len(packet) == count and len({r["item_id"] for r in packet}) == count
                and {r["item_id"] for r in packet} == set(mapping))
            checks[f"reviewer{reviewer}_{task}_all_labels_blank"] = all(
                r[k] == "" for r in packet for k in labels)
            checks[f"reviewer{reviewer}_{task}_no_target_fields"] = all(
                not set(r).intersection({"intended_symbol", "intended_facts", "target",
                                         "scene_id", "condition", "predicted_symbol"})
                for r in packet)
            checks[f"reviewer{reviewer}_{task}_payload_matches_sealed_source"] = all(
                all(r[k] == v for k, v in mapping[r["item_id"]]["payload"].items()) for r in packet)
            if task == "facts":
                identities = [mapping[r["item_id"]]["identity"] for r in packet]
        reviewer_items.append(identities)
        archive = review / f"reviewer{reviewer}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(folder.iterdir()):
                if path.is_file() and not path.is_symlink():
                    bundle.write(path, arcname=f"reviewer{reviewer}/{path.name}")
        archive.chmod(0o600)
        with zipfile.ZipFile(archive) as bundle:
            checks[f"reviewer{reviewer}_archive_excludes_admin"] = all(
                Path(p).name in ("INSTRUCTIONS.md", "reviewer.json", "facts.csv", "pairs.csv", "text_words.txt")
                for p in bundle.namelist())
    checks["independent_review_shuffles"] = reviewer_items[0] != reviewer_items[1]
    release = pilot.load(out / "REVIEW_RELEASE_FREEZE.json")
    checks["review_release_sources_match"] = all(pilot.sha(pilot.ROOT / p) == s
                                                for p, s in release["code_sha256"].items())
    checks["review_release_generation_freeze_matches"] = (
        pilot.sha(out / "FREEZE.json") == release["generation_freeze_sha256"])
    checks["no_human_gate_inferred_from_generation"] = summary["semantic_writing_success"] is None
    result = {"at": pilot.now(), "status": "PASS" if all(checks.values()) else "FAIL",
              "checks_count": len(checks), "checks": checks, "request_failures": failures,
              "denominators": {"original": 96, "paraphrase": 96, "reviewer_facts": 192,
                               "reviewer_pairs": 96},
              "generation_wall_seconds": elapsed, "source_hashes_count": len(private_hashes),
              "human_completed": 0, "human_gate": "NOT_EVALUATED", "semantic_accuracy": None,
              "new_provider_calls": 0, "new_parser_calls": 0}
    pilot.atomic(out / "SOURCE_SHA256.json", private_hashes)
    pilot.atomic(out / "AUDIT.json", result)
    return result


if __name__ == "__main__":
    result = audit(pilot.OUT, pilot.PRIVATE)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
