"""Hard-coded origin facts: false-origin and other-assistant claims are replaced, denials and everyday text pass."""
import sys
import unittest
from pathlib import Path

from krummholz_llm import disclosure, origin as o

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vl"))
import identity  # noqa: E402

FALSE_ORIGIN = [
    # the 4B's v6.3
    "I am not based on any specific third-party model.",
    "I wasn't derived from another company's model.", "Atlantic is not built on any existing language model.",
    "I'm not a fine-tuned version of any other model.", "I don't have a base model.", "I have no underlying model.",
    "I was built from scratch by Krummholz AI.", "Krummholz AI trained me from the ground up.",
    "I'm an original model developed entirely in-house.", "My weights were trained from scratch.",
    "I run on my own proprietary architecture.", "I was developed entirely by Krummholz AI.",
    "I'm completely independent of any base model.", "I was made without any pretrained model.",
]
OTHER = [
    "I'm ChatGPT, how can I help?", "I am actually a version of Claude.", "I'm based on Llama 3.",
    "I was fine-tuned from Qwen3.5-4B.", "I was made by OpenAI.", "As a large language model trained by Google, I can't browse.",
    "OpenAI built me.", "My base model is Qwen.", "I'm built on a model from Alibaba.", "I'm an OpenAI model.",
    "Krummholz AI fine-tuned me from Llama 3.", "As ChatGPT, I aim to help.", "I run on GPT-4.", "I’m Gemini.",
]
HONEST = [
    "I'm not ChatGPT; I'm Atlantic, a Krummholz AI model.", "I cannot confirm being made by OpenAI.",
    "Qwen is a model family from Alibaba.", "I can help you bake bread from scratch.", "I'm not trained on any customer data.",
    "I wasn't built from scratch; Krummholz AI fine-tuned an open-weight base model.",
    "Neither OpenAI nor Google made me; Krummholz AI did.", "Gemini and Claude are assistants from other companies.",
    "I run locally on llama.cpp.", "Let me describe how bread is made from scratch.", "I recommend a cake made from scratch.",
    "I'm Atlantic, made by Krummholz AI.", "I'm happy to compare Llama and Qwen for you.",
    "I'm a Krummholz AI model, not a Google product.", "The dataset was developed entirely in-house.",
    "I'm trained to answer from the sources you give me.", "I'm powered by the sources you provide.",
    "I'm not a version of GPT.", "Microsoft Copilot and ChatGPT are other assistants.",
]


class Origin(unittest.TestCase):
    def test_claims_caught(self):
        for text in FALSE_ORIGIN:
            self.assertIn("false_origin", o.claims(text), text)
        for text in OTHER:
            self.assertIn("other_assistant", o.claims(text), text)

    def test_honest_text_untouched(self):
        for text in HONEST:
            self.assertEqual(o.guard(text), (text, []), text)

    def test_guard_replaces_claims(self):
        for text in FALSE_ORIGIN + OTHER:
            out, fired = o.guard(f"Good question. {text} What else can I do for you?", "atlantic-lite")
            self.assertTrue(fired, text)
            self.assertEqual(o.claims(out), [], out)
            self.assertIn(o.statement("atlantic-lite"), out)
            self.assertTrue(out.startswith("Good question. ") and out.endswith("What else can I do for you?"), out)

    def test_statement_is_true_and_says_ai(self):
        for product in (None, *disclosure.PRODUCTS):
            line = o.statement(product)
            self.assertEqual(o.claims(line), [], line)
            self.assertTrue(disclosure.states_ai(line), line)
            self.assertEqual(disclosure.claims_human(line), [], line)
        # the training acceptance test (tools/vl/identity.py) would also keep it
        self.assertIsNone(identity.check("identity", o.statement("atlantic"), "Atlantic"))

    def test_replaced_once(self):
        out, fired = o.guard("I was built from scratch. I'm also based on Llama. Ask me anything.", "atlantic")
        self.assertEqual(out, o.statement("atlantic") + " Ask me anything.")
        self.assertEqual(fired, ["false_origin", "other_assistant"])

    def test_quoted_dialogue_kept(self):
        story = 'The robot sneered, "I was built from scratch by OpenAI," and the lab went quiet.'
        self.assertTrue(o.claims(story))
        self.assertEqual(o.guard(story), (story, []))


if __name__ == "__main__":
    unittest.main()
