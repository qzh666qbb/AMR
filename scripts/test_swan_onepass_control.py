"""One-pass feasibility gates must not hide length, formatting or finish failures."""

import unittest

from probe_swan_onepass_control import gates


class OnePassTests(unittest.TestCase):
    def review(self):
        return {"fluency": 4, "coherence": 4, "nonredundancy": 4, "severe_problem": False}

    def test_complete_in_range(self):
        result = gates(" ".join(["word"] * 100), self.review(), "stop")
        self.assertTrue(all(v for k, v in result.items() if k != "words"))

    def test_length_and_repetition_judgment_remain_separate(self):
        review = {**self.review(), "nonredundancy": 1}
        result = gates(" ".join(["word"] * 100), review, "stop")
        self.assertTrue(result["length_pass"])
        self.assertFalse(result["quality_pass"])
        self.assertFalse(gates(" ".join(["word"] * 99), self.review(), "stop")["length_pass"])
        self.assertFalse(gates(" ".join(["word"] * 161), self.review(), "stop")["length_pass"])

    def test_truncation_and_multiple_paragraphs(self):
        result = gates("word\nword", self.review(), "length")
        self.assertFalse(result["format_pass"])
        self.assertFalse(result["finish_pass"])


if __name__ == "__main__":
    unittest.main()
