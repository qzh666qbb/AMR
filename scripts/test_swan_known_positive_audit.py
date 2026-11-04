"""No-network tests for the layered SWAN known-positive audit."""

import os
from pathlib import Path
import sys
import unittest

from run_swan_known_positive_audit import (
    ROOT, configure_environment, load_sources, pipeline, triple_counts, valid_score,
)


class KnownPositiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_environment()
        sys.path.insert(0, str(ROOT / "third_party/SWAN"))
        from utils import amr_utils
        cls.utils = amr_utils

    def test_official_source_inventory(self):
        bank, groups, examples = load_sources()
        self.assertEqual(len(bank), 50)
        self.assertEqual(len(examples), 3)
        self.assertEqual(sum(len(groups[row["normalized_template_amr"]]["examples"]) for row in bank), 387)

    def test_prompt_example_reconstructs_saved_template(self):
        _, _, examples = load_sources()
        for example in examples:
            actual = pipeline(example["raw_amr"], self.utils)
            self.assertEqual(actual["final_amr"].strip(), example["normalized_template_amr"].strip())

    def test_triple_counts_distinguish_relations_and_attributes(self):
        counts = triple_counts('(a / alpha :ARG0 (b / beta) :quant 2 :name "Alice")')
        self.assertEqual(counts, {"instances": 2, "attributes": 2, "relations": 1, "triples": 5})

    def test_score_validation(self):
        self.assertTrue(valid_score(0.0))
        self.assertTrue(valid_score(1.0))
        self.assertFalse(valid_score(float("nan")))
        self.assertFalse(valid_score(-0.1))
        self.assertFalse(valid_score(1.1))

    def test_environment_is_local(self):
        self.assertEqual(Path(os.environ["NLTK_DATA"]), ROOT / ".nltk_data")
        self.assertEqual(Path(os.environ["GLOVE_VECTORS_PATH"]), ROOT / "vectors/glove.6B.100d.txt")


if __name__ == "__main__":
    unittest.main()
