"""Handoff unit fixtures; never real human annotations or provider results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import collect_semantic_fact_carrier_reviews as handoff
import run_semantic_fact_carrier_pilot as pilot


class IsolationTests(unittest.TestCase):
    def test_copy_excludes_keys_and_administrator_files(self):
        with tempfile.TemporaryDirectory(prefix="amr_review_unit_") as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            source.mkdir()
            for name in ("facts.csv", "reviewer.json", "KEYS.json", "MANIFEST.json", "DISPUTES.json"):
                (source / name).write_text("UNIT FIXTURE")
            self.assertEqual(set(handoff.copy_human_files(source, target)),
                             {"facts.csv", "reviewer.json"})

    def test_disagreement_packet_hides_keys(self):
        with tempfile.TemporaryDirectory(prefix="amr_review_unit_") as temp:
            out, private = Path(temp) / "out", Path(temp) / "private"
            out.mkdir()
            (out / "PROTOCOL.md").write_text("UNIT FIXTURE")
            pilot.atomic(out / "PRICE_SOURCE.json", {"fixture_only": True})
            sources = ["scripts/run_semantic_fact_carrier_pilot.py",
                       "src/core/semantic_fact_carrier.py",
                       "scripts/test_semantic_fact_carrier_pilot.py"]
            pilot.atomic(out / "TEST_RESULT.json", {
                "status": "PASS", "tests_run": 13, "fixture_only": True,
                "code_sha256": {p: pilot.sha(pilot.ROOT / p) for p in sources}})
            pilot.prepare(out, private)
            for row in pilot.plan():
                empty = {"status": "provider_failure", "text": "", "usage": {}, "model": None}
                pilot.atomic(private / "documents" / (row["item_id"] + ".json"),
                             {"original": empty, "paraphrase": empty})
            pilot.summarize(out, private, "GENERATION_FINISHED")
            pilot.review(out, private)
            keys = pilot.load(out / "private_review/admin/KEYS.json")
            for reviewer in (1, 2):
                directory = out / "private_review" / f"reviewer{reviewer}"
                # Synthetic attestations exist only inside this temporary unit test.
                pilot.atomic(directory / "reviewer.json", {
                    "reviewer_code": f"UNIT_FIXTURE_{reviewer}",
                    "completed_at": "2020-01-01T00:00:00Z",
                    "human_attestation": True, "independent_attestation": True})
                for task in ("facts", "pairs"):
                    path = directory / (task + ".csv")
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    for row in rows:
                        if task == "facts":
                            row.update({"readability": "unavailable", "explicit_status": "unknown",
                                        "quality": "unknown", "notes": "UNIT FIXTURE ONLY",
                                        **dict.fromkeys(pilot.FIELDS, "unknown"),
                                        "evidence_json": json.dumps(dict.fromkeys(pilot.FIELDS, []))})
                            identity = keys[str(reviewer)][task][row["item_id"]]["identity"]
                            if reviewer == 2 and identity == {"item": "f00A", "condition": "original"}:
                                row["readability"] = "unclear"
                        else:
                            row.update({**dict.fromkeys(pilot.PAIR_LABELS[:3], "unknown"),
                                        "evidence_json": '{"source":[],"paraphrase":[]}',
                                        "notes": "UNIT FIXTURE ONLY"})
                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            result = handoff.collect(out, private, check_release=False)
            self.assertEqual(result["status"], "AWAITING_HUMAN_ADJUDICATION")
            packet = out / "private_review/adjudicator"
            self.assertTrue((packet / "facts.csv").exists())
            self.assertFalse((packet / "KEYS.json").exists())
            self.assertTrue((private / "human_collection_workspace/private_review/adjudicator/KEYS.json").exists())
            self.assertNotIn("intended_symbol", (packet / "facts.csv").read_text())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) == 3 and sys.argv[1] == "--evidence":
        sources = ["scripts/collect_semantic_fact_carrier_reviews.py",
                   "scripts/test_semantic_fact_review_handoff.py"]
        pilot.atomic(Path(sys.argv[2]), {
            "at": pilot.now(), "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun, "api_calls": 0,
            "unit_fixtures_not_human_evidence": True,
            "code_sha256": {p: pilot.sha(pilot.ROOT / p) for p in sources}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
