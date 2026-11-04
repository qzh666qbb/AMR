import unittest

import swan_independent_banks1_sources as sources
from run_swan_independent_banks1 import choose_templates


class SourceTests(unittest.TestCase):
    def test_full_role_allocation_replay(self):
        rows = [{"id": i, "text": "A sentence."} for i in range(sum(sources.ROLES.values()))]
        allocated = sources.allocate(rows)
        self.assertEqual(allocated, sources.allocate(rows))
        self.assertEqual({k: len(v) for k, v in allocated.items()}, sources.ROLES)
        self.assertEqual(len({r["id"] for v in allocated.values() for r in v}), len(rows))
        with self.assertRaises(ValueError):
            sources.allocate(rows[:-1])

    def test_normal_length_and_no_score(self):
        text = " ".join(" ".join(["word"] * 19) + "." for _ in range(10))
        chosen = sources.window(text)
        self.assertTrue(150 <= len(chosen.split()) <= 240)
        self.assertIsNone(sources.window("Too short."))
        raw = {"text": text, "url": "https://example.org/source"}
        bags = sources.history.empty_bags()
        row = sources.candidate(raw, 420, 0, bags)
        self.assertIsNotNone(row)
        sources.history.collect(row, bags)
        self.assertIsNone(sources.candidate(raw, 420, 1, bags))

    def test_bank_frequency_nodes_and_exact_seed_replay(self):
        a = "(v / V :ARG0 (n / N) :ARG1 (m / N))"
        b = "(v / V :ARG0 (n / NE) :ARG1 (m / N))"
        c = "(v / V :ARG0 (n / N))"
        records = [{"template": a, "eligible": True}] * 21
        records += [{"template": b, "eligible": True}, {"template": c, "eligible": True},
                    {"template": a, "eligible": False}]
        result = choose_templates(records, "ab" * 32, size=1)
        self.assertEqual(result, choose_templates(list(reversed(records)), "ab" * 32, size=1))
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["bank"], [{"normalized_template_amr": b, "count": 1}])
        short = choose_templates(records, "ab" * 32, size=50)
        self.assertFalse(short["sufficient"])
        self.assertEqual(len(short["bank"]), 1)

    def test_no_cross_bank_filter_or_score_argument(self):
        rows = [{"eligible": True, "template": "(a / V :ARG0 (b / N) :ARG1 (c / NE))",
                 "detector_score": score} for score in (.1, .9)]
        self.assertEqual(choose_templates(rows, "cd" * 32, size=1),
                         choose_templates([{k: v for k, v in r.items() if k != "detector_score"}
                                           for r in rows], "cd" * 32, size=1))


if __name__ == "__main__":
    unittest.main()
