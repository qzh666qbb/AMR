import unittest
from collect_swan_multiparser1_quality import metrics
from swan_multiparser1_core import FLOWS


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.attacks, self.parents, self.mapping, self.labels = [], [], {}, {}
        for role in ("pilot", "main"):
            sid = role + "-source"
            for candidate in (None, "0"):
                key = sid + str(candidate)
                self.mapping[key] = {"doc_id": sid+"-a", "candidate_id": candidate}
                self.labels[key] = True
            for oracle in ("public", "a"):
                for mode in ("none", "label", "score"):
                    for budget in (1, 4, 16):
                        base = {"source_id": sid, "role": role, "oracle": oracle,
                                "mode": mode, "budget": budget}
                        self.attacks.append({
                            **base, "candidate_id": "0", "clean_detected": {k: True for k in FLOWS},
                            "attack_detected": {"amrlib": False, "amrbart": True, "ensemble": True}})
                        self.parents.append({
                            **base, "arm": "a", "selection": {"selected": "0"},
                            "clean_detected": {"public": True},
                            "attack_detected": {"public": False}})

    def calculate(self):
        return metrics(self.attacks, self.parents, self.mapping, self.labels, {"pilot": 1, "main": 1})

    def test_distinct_three_flow_endpoints(self):
        result = self.calculate()
        self.assertEqual(len(result), 108)
        by = {r["flow"]: r for r in result[:3]}
        self.assertEqual(by["amrlib"]["ASR"]["rate"], 1)
        self.assertEqual(by["ensemble"]["ASR"]["rate"], 0)
        self.assertEqual(by["amrbart"]["ATR"]["rate"], 0)

    def test_rewrite_quality_failure_preserves_denominator(self):
        self.labels["pilot-source0"] = False
        result = self.calculate()[0]
        self.assertEqual(result["ASR"]["rate"], 0)
        self.assertEqual(result["rewrite_quality_failures_in_eligible"], 1)

    def test_missing_quality_rejected(self):
        del self.labels["pilot-source0"]
        with self.assertRaises(KeyError):
            self.calculate()

    def test_candidate_change_rejected(self):
        self.attacks[0]["candidate_id"] = "1"
        with self.assertRaises(ValueError):
            self.calculate()

    def test_missing_source_product_rejected(self):
        self.attacks.pop()
        with self.assertRaises(ValueError):
            self.calculate()


if __name__ == "__main__":
    unittest.main()
