import hashlib
from pathlib import Path
import tempfile
import unittest

from lint_research_claims import evaluate_claim, evaluate_registry, lint_text


class ClaimGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "primary.json"
        self.path.write_text('{"endpoint": {"count": 1}}')
        self.claim = {
            "id": "CL1", "statement": "example", "ceiling": "historical_support",
            "primary_endpoint": {
                "path": "primary.json", "pointer": "/endpoint",
                "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "kind": "historical_saved",
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_primary_cannot_use_exploration(self):
        self.path.unlink()
        (self.root / "exploration.json").write_text('{"significant":true}')
        result = evaluate_claim(self.claim, self.root)
        self.assertEqual(result["effective_status"], "unconfirmed")
        self.assertIn("primary_endpoint_missing", result["reasons"])

    def test_modified_endpoint_downgrades(self):
        self.path.write_text('{"endpoint": {"count": 100}}')
        self.assertIn("primary_endpoint_hash_mismatch",
                      evaluate_claim(self.claim, self.root)["reasons"])

    def test_one_endpoint_and_id(self):
        with self.assertRaises(ValueError):
            evaluate_registry({"claims": [self.claim, self.claim]}, self.root)
        self.claim["primary_endpoints"] = []
        with self.assertRaises(ValueError):
            evaluate_claim(self.claim, self.root)

    def test_unconfirmed_ceiling_never_upgrades(self):
        self.claim["ceiling"] = "unconfirmed"
        self.assertEqual(evaluate_claim(self.claim, self.root)["effective_status"],
                         "unconfirmed")

    def test_final_lint_cannot_escape_with_history_header(self):
        text = "SUPERSEDED: AMR changed; semantic invariant; parser unnecessary"
        self.assertTrue(all(x["severity"] == "error" for x in lint_text(text)))
        self.assertTrue(all(x["severity"] == "historical_quote"
                            for x in lint_text(text, "history")))

    def test_pointer_failure_is_not_support(self):
        self.claim["primary_endpoint"]["pointer"] = "/missing"
        self.assertEqual(evaluate_claim(self.claim, self.root)["effective_status"],
                         "unconfirmed")

    def test_confirmatory_file_without_gates_fails(self):
        self.claim["primary_endpoint"]["kind"] = "prospective_confirmatory"
        self.assertIn("confirmatory_gates_missing",
                      evaluate_claim(self.claim, self.root)["reasons"])


if __name__ == "__main__":
    unittest.main()
