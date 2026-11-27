"""Stop-rule arithmetic: cluster construction, paired composite bootstrap, retention streaks, pace."""
import unittest

from es.fitness import FitnessWeights
from es.gates import cluster_ids, paired_composite_bootstrap, projected_generations, retention_stop


def pair(g, blocks):
    return [{"id": f"{g}/present", "group": g, "variant": "present", "blocks": blocks},
            {"id": f"{g}/absent", "group": g, "variant": "absent", "blocks": blocks}]


ITEMS = pair("a", ["b1"]) + pair("b", ["b1", "b2"]) + pair("c", ["b3"]) + [
    {"id": "r1", "split": "retention", "checks": {}}, {"id": "r2", "split": "retention", "checks": {}}]


class GateTests(unittest.TestCase):
    def test_clusters_join_shared_blocks(self):
        c = cluster_ids(ITEMS)
        self.assertEqual(c[0], c[1])
        self.assertEqual(c[0], c[2])          # a and b share block b1
        self.assertNotEqual(c[0], c[4])       # c is separate
        self.assertEqual(len(set(c)), 4)      # {a, b}, {c}, r1, r2

    def test_composite_by_hand_and_zero_se_for_identical_arms(self):
        w = FitnessWeights(pair_bonus=0.5, retention_penalty=1.5)
        base = {"rewards": [1, 1, -1, -1, 1, -1, 0, 0], "correct": [1, 1, 0, 0, 1, 0, 1, 1]}
        cand = {"rewards": [1, 1, 1, 1, 1, 1, 0, -1], "correct": [1, 1, 1, 1, 1, 1, 1, 0]}
        same = paired_composite_bootstrap(ITEMS, base, base, w, resamples=200, seed=1)
        self.assertEqual((same["diff"], same["se"]), (0.0, 0.0))
        res = paired_composite_bootstrap(ITEMS, base, cand, w, resamples=500, seed=1)
        # base: mean 0/6, pairs both 1/3, retention 0/2 -> 1/6; cand: mean 1, pairs 3/3, retention 1/2 -> 0.75
        self.assertAlmostEqual(res["base"], 0.5 / 3)
        self.assertAlmostEqual(res["cand"], 1 + 0.5 - 0.75)
        self.assertAlmostEqual(res["diff"], 0.75 - 0.5 / 3)
        self.assertGreater(res["se"], 0)
        again = paired_composite_bootstrap(ITEMS, base, cand, w, resamples=500, seed=1)
        self.assertEqual(res, again)

    def test_retention_streak(self):
        hist = [{"gen": 0, "retention_failed": [2, 32]}, {"gen": 10, "retention_failed": [5, 32]},
                {"gen": 20, "retention_failed": [3, 32]}, {"gen": 30, "retention_failed": [5, 32]},
                {"gen": 40, "retention_failed": [6, 32]}]
        self.assertIn("generation 40", retention_stop(hist, 3, 2))
        self.assertIsNone(retention_stop(hist[:4], 3, 2))

    def test_projected_generations(self):
        self.assertAlmostEqual(projected_generations(40, 60.0, 220.0, 1.0), 200.0)
        self.assertEqual(projected_generations(40, 60.0, 220.0, 0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
