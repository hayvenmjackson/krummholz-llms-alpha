"""Integration with the private yardstick files, read by path at test time (never copied here).

Skipped unless ES_YARDSTICK_DIR points at a yardstick checkout containing
grader/krummholz_grader.py and heldout/train_pool.jsonl.
"""
import os
import unittest
from pathlib import Path

from es.backends.base import Reply
from es.data import BatchSampler, PromptBuilder, TerminalSetError, load_jsonl, load_pairs
from es.fitness import FitnessWeights, ModuleGrader, score_candidate, target_acceptance

ROOT = Path(os.environ["ES_YARDSTICK_DIR"]) if os.environ.get("ES_YARDSTICK_DIR") else None
READY = bool(ROOT and (ROOT / "grader" / "krummholz_grader.py").exists()
             and (ROOT / "heldout" / "train_pool.jsonl").exists())


def as_reply(text, complete=True):
    return Reply(text=text, public=text, finish_reason="stop", completion_tokens=1, complete=complete)


@unittest.skipUnless(READY, "set ES_YARDSTICK_DIR to a yardstick checkout to run")
class YardstickIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grader = ModuleGrader(ROOT / "grader" / "krummholz_grader.py")
        cls.pairs = load_pairs(ROOT / "heldout" / "train_pool.jsonl")

    def test_terminal_set_refused(self):
        if (ROOT / "heldout" / "terminal.jsonl").exists():
            with self.assertRaises(TerminalSetError):
                load_jsonl(ROOT / "heldout" / "terminal.jsonl")

    def test_pool_shape(self):
        self.assertGreaterEqual(len(self.pairs), 56)
        for p in self.pairs[:200]:
            self.assertEqual([it["variant"] for it in p.items], ["present", "absent"])
            self.assertEqual(p.items[0]["split"], "train_pool")

    def test_grader_accepts_the_pools_own_targets(self):
        """Run gate: the ES preflight refuses to start below 0.99 (fitness.preflight_min_target_acceptance)."""
        res = target_acceptance([it for p in self.pairs for it in p.items], self.grader)
        self.assertGreaterEqual(res["acceptance"], 0.99,
                                f"grader rejects {res['rejected']}/{res['graded']} code-template targets, "
                                f"for example {res['rejected_examples'][:3]}")

    def test_fitness_arithmetic_on_accepted_targets(self):
        accepted = [p for p in self.pairs[:400] if all(self.grader.grade(it, it["target"]).correct for it in p.items)]
        pairs = accepted[:60]
        self.assertEqual(len(pairs), 60)
        items = [it for p in pairs for it in p.items]
        s = score_candidate(items, [as_reply(it["target"]) for it in items], self.grader, FitnessWeights())
        self.assertEqual(s.correct_items, len(items))
        self.assertEqual(s.pairs_both_correct, 60)
        self.assertAlmostEqual(s.fitness, 1.0 + 0.5)
        # swapped answers: each present item gets its absent target and vice versa
        swapped = []
        for p in pairs:
            swapped += [as_reply(p.items[1]["target"]), as_reply(p.items[0]["target"])]
        bad = score_candidate(items, swapped, self.grader, FitnessWeights())
        self.assertEqual(bad.correct_items, 0)
        self.assertLess(bad.fitness, 0.0)
        self.assertTrue(self.grader.describe()["version"])

    def test_prompt_matches_format(self):
        it = self.pairs[0].items[0]
        msgs = PromptBuilder("SYSTEM").messages(it)
        self.assertEqual(msgs[1]["content"], (it["prompt"] + "\n\n" + it["answer_instruction"]).strip())
        batch = BatchSampler(self.pairs, [], 1, 56, 0).batch(0)
        self.assertEqual(len(batch), 112)


if __name__ == "__main__":
    unittest.main()
