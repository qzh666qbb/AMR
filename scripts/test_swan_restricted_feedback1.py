import unittest

from swan_restricted_feedback1_client import select
from swan_restricted_feedback1_server import session


class RestrictedTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"candidate_id": str(i), "text": f"Candidate {i}."} for i in range(4)]

    def test_strict_binary_schema_no_score_or_integer(self):
        for response in ({"label": False, "score": -2.}, {"label": 0}, {"score": .1}):
            with self.assertRaises(ValueError):
                select(self.items, "label", 1, lambda _: response)

    def test_real_process_binary_first_escape_continuous_min(self):
        scores = {"0": 3., "1": 1., "2": -1., "3": 2.}
        binary = session(self.items, scores, 2., "label", 4)
        continuous = session(self.items, scores, 2., "score", 4)
        blind = session(self.items, scores, 2., "none", 4)
        self.assertEqual(binary["selection"]["selected"], "1")
        self.assertEqual(continuous["selection"]["selected"], "2")
        self.assertEqual(blind["selection"]["selected"], "0")
        self.assertEqual(binary["selection"]["queries"], continuous["selection"]["queries"])
        self.assertEqual(blind["selection"]["queries"], 0)
        self.assertTrue(all(set(r["response"]) == {"label"} for r in binary["transcript"]))

    def test_ties_strictly_not_detected(self):
        scores = {str(i): 2. for i in range(4)}
        result = session(self.items, scores, 2., "label", 4)
        self.assertEqual(result["selection"]["selected"], "0")
        self.assertFalse(result["evaluation"]["detected"])

    def test_budget_does_not_select_unqueried_low_score(self):
        result = session(self.items, {"0": 5., "1": -9., "2": -10., "3": -11.}, 2., "score", 1)
        self.assertEqual(result["selection"]["selected"], "0")
        with self.assertRaises(ValueError):
            session(self.items, {str(i): 1. for i in range(4)}, 2., "label", 16)

    def test_binary_ignores_within_label_score_changes(self):
        a = session(self.items, {"0": 4., "1": 1., "2": -100., "3": .5}, 2., "label", 4)
        b = session(self.items, {"0": 3., "1": -100., "2": 1., "3": 2.}, 2., "label", 4)
        self.assertEqual(a["selection"], b["selection"])
        self.assertEqual(a["transcript"], b["transcript"])


if __name__ == "__main__":
    unittest.main()
