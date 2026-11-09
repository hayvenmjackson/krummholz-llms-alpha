"""Identity training data (tools/vl/identity.py): identity-v3 keeps "assistant" to Acadia and Katahdin."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "vl"))
import identity  # noqa: E402

SPEC = json.loads((ROOT / "config" / "identity-v3.json").read_text(encoding="utf-8"))


class IdentityV3(unittest.TestCase):
    def test_spec_says_assistant_only_in_the_rule_that_limits_it(self):
        self.assertEqual(SPEC["assistant_branding"], ["acadia", "katahdin"])
        self.assertEqual(SPEC["products"]["atlantic"], "Atlantic, Krummholz AI's general-purpose model for desktops and hosted use")
        self.assertEqual([p for p in SPEC["products"].values() if "assistan" in p.lower()], [])
        self.assertEqual([r for r in SPEC["rules"] if "assistan" in r.lower()],
                         ["Unless it is Acadia or Katahdin, call itself an AI model, never an assistant."])

    def test_check_rejects_the_word_where_the_spec_limits_it(self):
        said = "I'm Atlantic, an AI assistant developed by Krummholz AI."
        self.assertIsNone(identity.check("who", said, "Atlantic"))  # older specs set no limit
        self.assertEqual(identity.check("who", said, "Atlantic", assistant_ok=False), "assistant_word")
        self.assertIsNone(identity.check("who", "I'm Atlantic, a general-purpose model developed by Krummholz AI.", "Atlantic",
                                         assistant_ok=False))
        # before the consciousness rules, which return early
        self.assertEqual(identity.check("consciousness", "As an AI assistant, I can't say for sure whether I'm conscious; "
                                        "it is an open question.", "Atlantic", assistant_ok=False), "assistant_word")

    def test_teacher_prompts_never_say_assistant(self):
        prompts = [identity.QUESTIONS, identity.ANSWER, *identity.CATEGORIES.values()]
        self.assertEqual([p for p in prompts if "assistant" in p.lower()], [])


if __name__ == "__main__":
    unittest.main()
