"""RSI loop plumbing (tools/rsi): verifier, label modes, pair building, the significance test. CPU only, no models."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rsi"))
import rsi  # noqa: E402
import verify  # noqa: E402


def cand(pid, j, reply, kind="math", **kw):
    return {"id": f"{pid}-s{j}", "pid": pid, "kind": kind, "prompt": f"q {pid}", "reply": reply, "finish": "stop", "hint": False, **kw}


class Verifier(unittest.TestCase):
    def test_math(self):
        self.assertTrue(verify.math_ok("so the answer is \\boxed{42}", "42"))
        self.assertTrue(verify.math_ok("We get 1,250 apples in total.", "1250"))
        self.assertTrue(verify.math_ok("\\boxed{2.50}", "2.5"))
        self.assertFalse(verify.math_ok("\\boxed{41}", "42"))
        self.assertFalse(verify.math_ok("no number here", "3"))

    def test_norm(self):
        self.assertEqual(verify.norm("$2.50$"), verify.norm("2.5"))
        self.assertEqual(verify.norm("\\text{7}"), "7")

    def test_code(self):
        good = "Here:\n```python\ndef sq(x):\n    return x * x\n```"
        self.assertTrue(verify.run_tests(verify.code_block(good), ["assert sq(3) == 9"])[0])
        self.assertFalse(verify.run_tests("def sq(x):\n    return x + x", ["assert sq(3) == 9"])[0])
        self.assertEqual(verify.run_tests("while True: pass", ["pass"], timeout=2)[1], "timeout")

    def test_code_is_isolated_from_parent_env(self):
        import os
        os.environ["RSI_SECRET_PROBE"] = "x"
        try:
            ok, _ = verify.run_tests("import os", ["assert 'RSI_SECRET_PROBE' not in os.environ"])
        finally:
            del os.environ["RSI_SECRET_PROBE"]
        self.assertTrue(ok)

    def test_humaneval_score_keeps_prompt_imports(self):
        prefix = "from typing import List\n\n\ndef total(xs: List[int]) -> int:\n    \"\"\"Sum.\"\"\"\n"
        test = "def check(candidate):\n    assert candidate([1, 2]) == 3\n"
        p = {"set": "humanevalplus", "kind": "code", "prefix": prefix, "tests": [verify.humaneval_tests(prefix, test)]}
        full = "```python\ndef total(xs: List[int]) -> int:\n    return sum(xs)\n```"
        body = "```python\n    return sum(xs)\n```"
        self.assertTrue(rsi.score(p, full)); self.assertTrue(rsi.score(p, body))

    def test_humaneval_tests(self):
        t = verify.humaneval_tests("def helper(x):\n    pass\ndef f(x):\n    \"\"\"doc\"\"\"\n", "def check(c):\n    assert c(1) == 1\n")
        self.assertTrue(t.rstrip().endswith("check(f)"))


class Labels(unittest.TestCase):
    def setUp(self):
        self.res = [cand("p", 0, "\\boxed{5}", answer="5", verified=True, p_yes=0.9),
                    cand("p", 1, "\\boxed{5}", answer="5", verified=True, p_yes=0.2),
                    cand("p", 2, "\\boxed{6}", answer="6", verified=False, p_yes=0.8),
                    cand("p", 3, "\\boxed{5}", answer="5", verified=True, p_yes=0.9)]

    def test_verified(self):
        lab = rsi.label_rows([dict(c) for c in self.res], "verified")
        self.assertEqual([c["label"] for c in lab], [True, True, False, True])

    def test_self_modes_never_read_the_verifier(self):
        flipped = [dict(c, verified=not c["verified"]) for c in self.res]
        for mode in ("majority", "judge", "self"):
            a = [c["label"] for c in rsi.label_rows([dict(c) for c in self.res], mode)]
            b = [c["label"] for c in rsi.label_rows(flipped, mode)]
            self.assertEqual(a, b, mode)

    def test_random_control_keeps_counts(self):
        res = []
        for k in range(50):
            res += [cand(f"r{k}", j, "", answer=str(j), verified=j == 0) for j in range(4)]
        lab = rsi.label_rows([dict(c) for c in res], "random")
        self.assertEqual(sum(c["label"] for c in lab), 50)
        self.assertLess(sum(c["label"] and c["verified"] for c in lab), 30)  # about 1 in 4 by chance

    def test_self_drops_disagreement(self):
        lab = rsi.label_rows([dict(c) for c in self.res], "self")
        self.assertEqual([c["label"] for c in lab], [True, None, None, True])

    def test_majority_needs_half(self):
        res = [cand("q", j, "", answer=a, verified=False) for j, a in enumerate(["1", "2", "3", "1"])]
        self.assertEqual(rsi.majority(res), "1")
        res = [cand("q", j, "", answer=a, verified=False) for j, a in enumerate(["1", "2", "3", "4"])]
        self.assertIsNone(rsi.majority(res))
        self.assertTrue(all(c["label"] is None for c in rsi.label_rows(res, "majority")))

    def test_gap(self):
        lab = rsi.label_rows([dict(c) for c in self.res], "judge")
        g = rsi.gap(lab)
        self.assertEqual((g["precision"], g["recall"]), (round(2 / 3, 3), round(2 / 3, 3)))

    def test_hint_answers_only_count_when_verified(self):
        res = [dict(cand("h", 0, "\\boxed{5}", answer="5", verified=True), hint=True)]
        self.assertTrue(rsi.label_rows([dict(c) for c in res], "verified")[0]["label"])
        self.assertIsNone(rsi.label_rows([dict(c) for c in res], "majority")[0]["label"])


class Build(unittest.TestCase):
    def test_pairs_and_targets(self):
        res = rsi.label_rows([cand("p", 0, "long enough reasoning \\boxed{5}", answer="5", verified=True),
                              cand("p", 1, "long enough reasoning \\boxed{7}", answer="7", verified=False),
                              cand("p", 2, "long enough but cut off", answer="", verified=True, finish="length")], "verified")
        sft, pairs, st = rsi.build_rows(res, round_tag="r1")
        self.assertEqual(len(sft), 1); self.assertEqual(len(pairs), 1)
        self.assertIn("\\boxed{5}", pairs[0]["chosen"]); self.assertIn("\\boxed{7}", pairs[0]["rejected"])
        self.assertEqual(sft[0]["messages"][0]["role"], "user"); self.assertTrue(sft[0]["target"])

    def test_easy_problems_are_downsampled(self):
        res = []
        for k in range(200):
            res += [cand(f"e{k}", j, "long enough reasoning \\boxed{1}", answer="1", verified=True) for j in range(4)]
        rsi.label_rows(res, "verified")
        sft, pairs, st = rsi.build_rows(res, easy_frac=0.2)
        self.assertEqual(pairs, [])
        self.assertLess(len(sft), 70); self.assertGreater(len(sft), 15)

    def test_no_em_dash_in_math_targets(self):
        dash = chr(0x2014)
        res = rsi.label_rows([cand("p", 0, f"first {dash} then \\boxed{{5}}", answer="5", verified=True),
                              cand("p", 1, "nope, so \\boxed{6}", answer="6", verified=False)], "verified")
        sft, _, _ = rsi.build_rows(res)
        self.assertNotIn(dash, sft[0]["target"])


class Stats(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(rsi.binom_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(rsi.binom_two_sided(0, 10), 2 / 1024)
        self.assertGreater(rsi.binom_two_sided(5, 6), 0.9)

    def test_compare(self):
        rng = random.Random(0)
        a = {"items": {f"math500-{i}": rng.random() < 0.5 for i in range(500)}}
        b = {"items": {k: v or rng.random() < 0.2 for k, v in a["items"].items()}}
        r = rsi.compare_results(a, b)
        self.assertGreater(r["sealed_all"]["after"], r["sealed_all"]["before"])
        self.assertLess(r["sealed_all"]["p"], 0.05)


class Pool(unittest.TestCase):
    def test_sealed_questions_never_enter_training(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rsi.write(d / "gsm8k_train.jsonl", [{"question": "How many eggs?", "answer": "3"}, {"question": "Sealed one?", "answer": "1"}])
            rsi.write(d / "gsm8k_test.jsonl", [{"question": "sealed ONE", "answer": "1"}])
            rsi.write(d / "mbpp_train.jsonl", [{"question": "Write f.", "tests": ["assert True"]}])
            (d / "SEALED.json").write_text(json.dumps({}))
            pool = rsi.pool_from(d, 10, 10)
            qs = {p["question"] for p in pool}
            self.assertIn("How many eggs?", qs); self.assertNotIn("Sealed one?", qs); self.assertIn("Write f.", qs)


if __name__ == "__main__":
    unittest.main()
