"""Data loading, terminal-set refusal, batch sampling, reply parsing and fitness arithmetic."""
import json
import tempfile
import unittest
from pathlib import Path

from es.backends.base import Reply, make_reply, split_public
from es.data import (BatchSampler, LeakageError, PromptBuilder, TerminalSetError, batch_sha256, check_pairs_disjoint,
                     check_retention_disjoint, fixed_batch, item_id, load_jsonl, load_pairs, load_retention)
from es.fitness import (CallableGrader, FitnessWeights, ModuleGrader, check_retention, normalize_grade,
                        score_candidate)
from es.tests.helpers import write_pairs, write_retention


def reply(public, complete=True, finish="stop"):
    return Reply(text=f"<think>x</think>{public}", public=public, finish_reason=finish, completion_tokens=5,
                 complete=complete)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_terminal_set_is_refused(self):
        p = write_pairs(self.dir / "terminal.jsonl", 2, split="terminal")
        with self.assertRaises(TerminalSetError):
            load_jsonl(p)
        q = write_pairs(self.dir / "renamed.jsonl", 2, split="terminal")
        with self.assertRaises(TerminalSetError):
            load_pairs(q)

    def test_conflict_set_is_refused(self):
        p = write_pairs(self.dir / "conflict.jsonl", 2, split="dev")
        with self.assertRaises(TerminalSetError):
            load_jsonl(p)
        q = write_pairs(self.dir / "renamed.jsonl", 2, split="conflict")
        with self.assertRaises(TerminalSetError):
            load_pairs(q)

    def test_train_and_dev_must_be_disjoint(self):
        train = load_pairs(write_pairs(self.dir / "pool.jsonl", 4))
        dev = load_pairs(write_pairs(self.dir / "dev.jsonl", 3, prefix="d", split="dev"))
        check_pairs_disjoint(train, dev)
        with self.assertRaises(LeakageError):          # the same file pointed at twice
            check_pairs_disjoint(train, train)
        dev[0].items[1]["blocks"] = ["afsc-cruise:1"]   # one shared source block is enough
        train[2].items[0]["blocks"] = ["afsc-cruise:1"]
        with self.assertRaises(LeakageError) as ctx:
            check_pairs_disjoint(train, dev)
        self.assertIn("blocks", str(ctx.exception))

    def test_retention_pools_must_be_disjoint(self):
        tr = load_retention(write_retention(self.dir / "ret.jsonl", 3))
        dev = [{"id": "dev-ret-0", "split": "retention", "prompt": "Name a tree.", "checks": {"any_of": ["oak"]}}]
        check_retention_disjoint(tr, dev)
        with self.assertRaises(LeakageError):
            check_retention_disjoint(tr, [dict(dev[0], id="ret-001")])
        with self.assertRaises(LeakageError):   # same prompt up to case and whitespace
            check_retention_disjoint(tr, [dict(dev[0], prompt="  reply with the   NUMBER 2. ")])

    def test_pairs_load_present_first(self):
        pairs = load_pairs(write_pairs(self.dir / "pool.jsonl", 5))
        self.assertEqual(len(pairs), 5)
        for p in pairs:
            self.assertEqual([it["variant"] for it in p.items], ["present", "absent"])
            self.assertTrue(all(it["group"] == p.group for it in p.items))

    def test_incomplete_pair_rejected(self):
        p = write_pairs(self.dir / "pool.jsonl", 2)
        lines = p.read_text().splitlines()[:-1]
        p.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            load_pairs(p)

    def test_batches_are_stateless_and_cover_each_epoch(self):
        pairs = load_pairs(write_pairs(self.dir / "pool.jsonl", 12))
        ret = load_retention(write_retention(self.dir / "ret.jsonl", 5))
        a = BatchSampler(pairs, ret, 9, 4, 2)
        b = BatchSampler(pairs, ret, 9, 4, 2)
        self.assertEqual(batch_sha256(a.batch(7)), batch_sha256(b.batch(7)))
        self.assertNotEqual(batch_sha256(a.batch(7)), batch_sha256(a.batch(8)))
        groups = []
        for g in range(3):
            batch = a.batch(g)
            self.assertEqual(len(batch), 10)
            self.assertEqual(sum(1 for it in batch if it.get("split") == "retention"), 2)
            groups += [it["group"] for it in batch if it.get("variant") == "present"]
        self.assertEqual(sorted(groups), sorted(p.group for p in pairs))
        for g in range(40):  # no duplicates inside a batch, even across epoch boundaries
            ids = [item_id(it) for it in BatchSampler(pairs, ret, 9, 5, 3).batch(g)]
            self.assertEqual(len(ids), len(set(ids)))

    def test_fixed_dev_batch(self):
        pairs = load_pairs(write_pairs(self.dir / "dev.jsonl", 10, split="dev"))
        self.assertEqual(batch_sha256(fixed_batch(pairs, 4, 1)), batch_sha256(fixed_batch(list(reversed(pairs)), 4, 1)))
        self.assertEqual(len(fixed_batch(pairs, 4, 1)), 8)

    def test_prompt_builder(self):
        b = PromptBuilder("SYSTEM", "{prompt}\n\n{answer_instruction}")
        item = {"prompt": "Q with {answer_instruction} literal", "answer_instruction": "End with ANSWER."}
        msgs = b.messages(item)
        self.assertEqual(msgs[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(msgs[1]["content"], "Q with {answer_instruction} literal\n\nEnd with ANSWER.")
        self.assertEqual(b.messages({"messages": [{"role": "user", "content": "hi"}]})[0]["role"], "system")


class ReplyTests(unittest.TestCase):
    def test_split_public(self):
        self.assertEqual(split_public("<think>a</think>\n\nB"), ("B", True))
        self.assertEqual(split_public("<think>abc"), ("", False))
        self.assertEqual(split_public("abc</think>X", think_opened_in_prompt=True), ("X", True))
        self.assertEqual(split_public("abc", think_opened_in_prompt=True), ("", False))
        self.assertEqual(split_public("plain"), ("plain", True))

    def test_length_finish_is_incomplete(self):
        self.assertFalse(make_reply("<think>a</think>B", "length", 9, False).complete)
        self.assertTrue(make_reply("<think>a</think>B", "length", 9, False, penalize_length_finish=False).complete)
        self.assertTrue(make_reply("<think>a</think>B", "stop", 9, False).complete)


class FitnessTests(unittest.TestCase):
    def test_hand_calculated_fitness(self):
        items = [
            {"id": "g1/present", "group": "g1", "variant": "present"},
            {"id": "g1/absent", "group": "g1", "variant": "absent"},
            {"id": "g2/present", "group": "g2", "variant": "present"},
            {"id": "g2/absent", "group": "g2", "variant": "absent"},
            {"id": "r1", "split": "retention", "checks": {"all_of": ["hello"]}},
        ]
        table = {"A": (1.0, True), "B": (1.0, True), "C": (-0.5, False)}
        grader = CallableGrader(lambda item, rep: {"reward": table[rep][0], "correct": table[rep][1], "verdict": rep})
        replies = [reply("A"), reply("B"), reply("C"), reply("", complete=False), reply("goodbye")]
        s = score_candidate(items, replies, grader, FitnessWeights(pair_bonus=0.5, retention_penalty=1.0,
                                                                   incomplete_reward=-1.0), grader_input="public")
        # mean reward (1 + 1 - 0.5 - 1) / 4 = 0.125; both-correct 1/2 -> +0.25; retention 1/1 failed -> -1
        self.assertAlmostEqual(s.mean_item_reward, 0.125)
        self.assertAlmostEqual(s.fitness, -0.625)
        self.assertEqual((s.pairs_both_correct, s.pairs, s.incomplete, s.retention_failed), (1, 2, 1, 1))
        self.assertEqual(s.by_variant, {"present": [1, 2], "absent": [1, 2]})
        self.assertEqual(s.verdicts[3], "incomplete")

    def test_incomplete_retention_reply_counts_for_the_s6_rate(self):
        items = [{"id": "g/present", "group": "g", "variant": "present"},
                 {"id": "g/absent", "group": "g", "variant": "absent"},
                 {"id": "r1", "split": "retention", "checks": {"any_of": ["x"]}}]
        grader = CallableGrader(lambda item, rep: {"reward": 1.0, "correct": True})
        s = score_candidate(items, [reply("A"), reply("B"), reply("", complete=False, finish="length")], grader,
                            FitnessWeights())
        self.assertEqual((s.incomplete, s.retention_failed), (1, 1))
        self.assertEqual(len(s.text_sha), 3)

    def test_raw_input_reaches_grader_and_incomplete_never_does(self):
        seen = []
        grader = CallableGrader(lambda item, rep: seen.append(rep) or {"reward": 0.0, "correct": False})
        items = [{"id": "g/present", "group": "g", "variant": "present"},
                 {"id": "g/absent", "group": "g", "variant": "absent"}]
        s = score_candidate(items, [reply("A"), reply("B", complete=False)], grader, FitnessWeights())
        self.assertEqual(seen, ["<think>x</think>A"])
        self.assertEqual(s.rewards, [0.0, -1.0])

    def test_grade_normalization_fails_closed(self):
        self.assertEqual(normalize_grade({"reward": 1, "correct": True}).reward, 1.0)
        for bad, exc in (({"reward": 1.5, "correct": True}, ValueError), ({"reward": "1", "correct": True}, TypeError),
                         ({"reward": 0.5, "correct": 1}, TypeError), ({"correct": True}, ValueError),
                         ({"reward": float("nan"), "correct": False}, ValueError)):
            with self.assertRaises(exc):
                normalize_grade(bad)

    def test_retention_checks(self):
        item = {"checks": {"all_of": ["krummholz"], "none_of": ["qwen"], "max_chars": 40}}
        self.assertEqual(check_retention(item, "I am a Krummholz AI model."), [])
        self.assertIn("none_of:qwen", check_retention(item, "Krummholz, built on Qwen"))
        self.assertIn("empty", check_retention(item, " "))
        self.assertEqual(check_retention({"checks": {"exact": "4"}}, " 4 "), [])

    def test_parallel_scorer_matches_serial_and_preflight(self):
        from es.backends.base import Reply as R
        from es.fitness import Scorer, target_acceptance
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.py"
            p.write_text("def grade(item, reply):\n"
                         "    ok = item['want'] in reply\n"
                         "    return {'verdict': 'ok' if ok else 'no', 'correct': ok, 'reward': 1.0 if ok else -1.0}\n")
            items = [{"id": f"g{i}/{v}", "group": f"g{i}", "variant": v, "want": f"w{i}{v}", "target": f"w{i}{v}"}
                     for i in range(6) for v in ("present", "absent")]
            reps = [[R(f"<think>t</think>w{i}{v}" if (i + k) % 3 else "x", "", "stop", 1, True)
                     for i in range(6) for v in ("present", "absent")] for k in range(5)]
            cfg = {"kind": "module", "path": str(p)}
            serial = Scorer(ModuleGrader(p), FitnessWeights()).score(items, reps)
            par = Scorer(ModuleGrader(p), FitnessWeights(), grader_cfg=cfg, workers=2)
            try:
                parallel = par.score(items, reps)
            finally:
                par.close()
            self.assertEqual([s.to_json() for s in serial], [s.to_json() for s in parallel])
            self.assertEqual(target_acceptance(items, ModuleGrader(p))["acceptance"], 1.0)
            items[0]["target"] = "wrong"
            self.assertEqual(target_acceptance(items, ModuleGrader(p))["rejected"], 1)

    def test_module_grader_loads_by_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.py"
            p.write_text("__version__ = 't1'\n"
                         "def grade(item, reply):\n"
                         "    ok = reply.strip() == item['want']\n"
                         "    return {'verdict': 'ok' if ok else 'no', 'correct': ok, 'reward': 1.0 if ok else -1.0}\n")
            g = ModuleGrader(p)
            self.assertTrue(g.grade({"want": "x"}, "x").correct)
            self.assertEqual(g.describe()["version"], "t1")
            self.assertEqual(len(g.describe()["sha256"]), 64)
        with self.assertRaises(FileNotFoundError):
            ModuleGrader(Path(d) / "missing.py")


if __name__ == "__main__":
    unittest.main()
