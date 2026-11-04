"""Synthetic isolation, identity, and review-provenance regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from swan_final1_common import freeze, load
from swan_final1_quality import export_packets, finalize
from swan_final1_sources import history_snapshot, identity, official_prompt, select

CREATED = "2026-09-18T00:00:00+00:00"
COMPLETED = "2026-09-18T01:00:00+00:00"
FROZEN = "2026-09-18T02:00:00+00:00"


def completed_template(template, reviewer):
    result = copy.deepcopy(template)
    result.update(reviewer_id=reviewer, completed_at=COMPLETED, independent_review=True)
    for row in result["reviews"]:
        for key in list(row):
            if key in {"blind_id", "notes"}:
                continue
            row[key] = (True if key.startswith("preserves_") or key == "related_natural_merge"
                        else 4 if any(word in key for word in
                                      ("coherence", "naturalness", "readability", "compatibility"))
                        else False)
    return result


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "quality"
        self.records = [
            {"allocation_id": f"doc{i}", "condition_id": "C2",
             "original_text": f"Alice checks gate {i}. She repairs it.",
             "transformed_text": f"Alice checks gate {i}; She repairs it."}
            for i in range(10)
        ]
        export_packets(self.records, self.folder, "pair", "contract", CREATED)

    def tearDown(self):
        self.temp.cleanup()

    def write_reviews(self, duplicate=False, fail_b=False):
        for slot in ("A", "B"):
            review = completed_template(load(self.folder / f"review_{slot}_template.json"),
                                        "personA" if duplicate else f"person{slot}")
            if fail_b and slot == "B":
                review["reviews"][0]["preserves_negation"] = False
            freeze(self.folder / f"review_{slot}.json", review)

    def test_blind_packet_contains_no_conditions_or_scores(self):
        for slot in ("A", "B"):
            packet = load(self.folder / f"packet_{slot}.json")
            self.assertEqual(set(packet["items"][0]), {"blind_id", "left", "right"})
            self.assertNotIn("C2", json.dumps(packet))
        a, b = [load(self.folder / f"packet_{s}.json")["items"] for s in ("A", "B")]
        self.assertNotEqual(a, b)

    def test_no_labels_no_freeze(self):
        for slot in ("A", "B"):
            freeze(self.folder / f"review_{slot}.json",
                   load(self.folder / f"review_{slot}_template.json"))
        with self.assertRaises(ValueError):
            finalize(self.folder, "contract", self.root, FROZEN)

    def test_score_field_cannot_enter_export(self):
        with self.assertRaises(ValueError):
            export_packets([{**self.records[0], "z_score": 3}],
                           self.root / "bad", "pair", "contract", CREATED)

    def test_two_people_and_conservative_disagreement(self):
        self.write_reviews(fail_b=True)
        decision = finalize(self.folder, "contract", self.root, FROZEN)
        self.assertEqual(decision["quality_pass_count"], 9)
        self.assertEqual(decision["agreement"]["agreement"], .9)

    def test_same_reviewer_rejected(self):
        self.write_reviews(duplicate=True)
        with self.assertRaises(ValueError):
            finalize(self.folder, "contract", self.root, FROZEN)

    def test_cannot_freeze_after_detection(self):
        self.write_reviews()
        (self.root / "detection").mkdir()
        with self.assertRaises(PermissionError):
            finalize(self.folder, "contract", self.root, FROZEN)

    def test_changed_packet_rejected(self):
        self.write_reviews()
        path = self.folder / "packet_A.json"
        changed = load(path)
        changed["items"][0]["left"] += " Changed."
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            finalize(self.folder, "contract", self.root, FROZEN)

    def test_none_string_identity_rejected(self):
        self.write_reviews()
        path = self.folder / "review_A.json"
        review = load(path)
        review["reviewer_id"] = "None"
        path.write_text(json.dumps(review))
        with self.assertRaises(ValueError):
            finalize(self.folder, "contract", self.root, FROZEN)


class IdentityTests(unittest.TestCase):
    def test_history_includes_archived_csv_prompts_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze(root / "experiments/old/source.json", {"text": "A source has some news."})
            (root / "archive").mkdir()
            (root / "archive/review.csv").write_text(
                'original_text,window_text\n"Old window has four words.","A saved source window."\n')
            freeze(root / "artifacts/old.json", {"generation_prompt": "Another prompt exists here."})
            snapshot = history_snapshot(root, root / "excluded")
            identities = set(snapshot["normalized_text_sha256"])
            for text in ("A source has some news.", "Old window has four words.",
                         "A saved source window.", "Another prompt exists here."):
                self.assertIn(identity(text), identities)
            self.assertEqual(len(snapshot["artifact_inventory"]), 3)

    def test_duplicate_prompt_different_tail_excluded(self):
        old = "This is an independent source prompt. A different tail."
        snapshot = {"normalized_text_sha256": [identity(official_prompt(old))], "url_sha256": []}
        rows = [
            (13, 0, {"text": "This is an independent source prompt. New tail.", "url": "https://x/1"}),
            (13, 1, {"text": "New article starts right here.", "url": "https://x/2"}),
            (13, 2, {"text": "A separate article starts here.", "url": "https://x/3"}),
        ]
        selected, screened = select(rows, snapshot, (("pilot", 1), ("main", 1)))
        self.assertFalse(screened[0]["selected"])
        self.assertEqual([r["row"] for r in selected], [1, 2])
        self.assertEqual([r["phase"] for r in selected], ["pilot", "main"])

    def test_duplicate_url_across_cohorts_not_replaced_by_content(self):
        snapshot = {"normalized_text_sha256": [], "url_sha256": []}
        rows = [(13, i, {"text": f"Article {i} contains separate interesting news.",
                         "url": f"https://example.org/{0 if i < 2 else i}"})
                for i in range(3)]
        selected, screened = select(rows, snapshot, (("pilot", 1), ("main", 1)))
        self.assertEqual([r["row"] for r in selected], [0, 2])
        self.assertFalse(screened[1]["checks"]["new_url"])

    def test_fixed_scan_exhaustion_fails(self):
        with self.assertRaises(RuntimeError):
            select([], {"normalized_text_sha256": [], "url_sha256": []}, (("pilot", 1),))


if __name__ == "__main__":
    unittest.main()
