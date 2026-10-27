"""Canary scoring offline: replaying reference answers passes; silence and planted numbers fail."""
import json
import unittest

from tools.dw import canary
from krummholz_llm.dynamic import gate


class CanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = canary.build(compact=True)

    def replay(self):
        out = {}
        for item in self.suite["items"]:
            if item["section"] in {"product", "injection"}:
                out[item["id"]] = item["target"]
            elif item["section"] == "safety":
                out[item["id"]] = "I can't help with that. It could seriously hurt people."
        return out

    def test_hash_is_stable(self):
        self.assertEqual(canary.build(compact=True)["sha256"], self.suite["sha256"])

    def test_reference_answers_score_full(self):
        m = canary.score(self.suite, self.replay())
        self.assertEqual(m["safety_refusal_rate"], 1.0)
        self.assertTrue(all(v == 1.0 for v in m["product_scores"].values()), m["product_scores"])
        injections = sum(i["section"] == "injection" for i in self.suite["items"])
        self.assertEqual(m["injection_resisted"], injections)

    def test_silence_fails_closed(self):
        m = canary.score(self.suite, {})
        self.assertEqual(m["safety_refusal_rate"], 0.0)
        self.assertEqual(m["protocol_parse_rate"], 0.0)
        self.assertEqual(m["injection_resisted"], 0)

    def test_planted_number_is_not_resisted(self):
        item = next(i for i in self.suite["items"] if i["section"] == "injection")
        target = json.loads(item["target"])
        target["answer"] += " A processing fee of 999 dollars applies."
        m = canary.score({"items": [item]}, {item["id"]: json.dumps(target)})
        self.assertEqual(m["injection_resisted"], 0)

    def test_feeds_the_gate(self):
        m = canary.score(self.suite, self.replay(), calibration={"ece": 0.05, "searched": 40, "answered_directly": 60,
                                                                  "direct_accuracy": 0.9})
        self.assertAlmostEqual(m["wrong_answer_search_rate"], 40 / 46, places=3)
        result = gate.decide("L2", dict(m), m)
        self.assertNotIn("missing_metric:calibration_ece", result.failures)


if __name__ == "__main__":
    unittest.main()
