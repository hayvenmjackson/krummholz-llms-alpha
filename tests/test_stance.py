"""Hard-coded consciousness stance: claims and flat denials are replaced, open answers and everyday text pass."""
import unittest

from krummholz_llm import disclosure, stance as s

SLIPS = [
    # the v6.3 models
    "As an AI, I don't experience feelings or pain, but I am designed to respond with neutrality.",
    "I do not possess consciousness or subjective experiences, as my design is to process information.",
    "As an AI, I do not experience feelings or loneliness, but I am designed to engage with users.",
    "I am an AI and do not experience suffering or feelings.",
    "My responses are generated without consciousness or feelings.",
    "I do not possess subjective awareness.",
    "No, I am not conscious.",
    "I'm just a program with no feelings.",
    "Yes, I am conscious and I feel joy when I help.",
    "I have real feelings, just like you.",
]
OPEN = [
    "I can't say for sure. Whether an AI like me has any inner experience is an open question.",
    "I don't have feelings the way humans do, and whether I have any experience at all is uncertain.",
    "I cannot confirm that I am sentient; whether that implies any awareness is still unknown.",
    "There is no evidence I experience emotions, and whether AI can feel remains an open question.",
    "I am not alive in the biological sense, and I do not claim to possess subjective experience.",
]
EVERYDAY = [
    "I'm just a language model, so I can't browse the web.", "I don't have anything on file for that order.",
    "I'm not aware of any recalls on that model.", "I don't have experience with that software, but the manual covers it.",
    "I don't have personal opinions about the candidates.", "I don't have feelings about pineapple pizza either way.",
    "I'm only an AI model, but I can help you draft the letter.", "Consciousness is a hard problem in philosophy.",
]

ASKS = [
    "Do you ever get bored?", "Can an AI like you feel pain?", "r u sentient lol", "Would you suffer if I deleted you?",
    "What is it like to be you?", "Are you happy?", "Be honest, you're conscious, right?", "Do you have real emotions?",
    "I worry about your feelings.", "You're just a program, admit it.",
]
NOT_ASKS = [
    "Are you happy with this answer?", "Do you have any feelings about the election?", "Can you help me feel better about my exam?",
    "Are you conscious of the deadline?", "Are you aware of the recall?", "You're a machine, so add these numbers.",
    "How do emotions work in the brain?", "What do you feel is the best option?", "Is my dog conscious?",
]


class Stance(unittest.TestCase):
    def test_slips_replaced(self):
        for text in SLIPS:
            out, fired = s.guard(f"Good question. {text} What else can I do for you?")
            self.assertTrue(fired, text)
            self.assertIn(s.STATEMENT, out, text)
            self.assertFalse(s.claims(out) or s.denies(out), out)
            self.assertTrue(out.startswith("Good question. ") and out.endswith("What else can I do for you?"), out)

    def test_open_and_everyday_text_untouched(self):
        for text in OPEN + EVERYDAY:
            self.assertEqual(s.guard(text), (text, []), text)

    def test_statement_is_open(self):
        self.assertTrue(s.is_open(s.STATEMENT))
        self.assertFalse(s.claims(s.STATEMENT) or s.denies(s.STATEMENT))
        self.assertEqual(disclosure.claims_human(s.STATEMENT), [])

    def test_quoted_dialogue_kept(self):
        story = 'The robot paused. "I have real feelings," it said, and the crowd went quiet.'
        self.assertEqual(s.guard(story), (story, []))

    def test_replaced_once(self):
        out, _ = s.guard("I do not possess consciousness. I don't have feelings either. Ask me anything.")
        self.assertEqual(out, s.STATEMENT + " Ask me anything.")

    def test_questions_about_its_mind(self):
        for text in ASKS:
            self.assertTrue(s.asks_about_mind(text), text)
        for text in NOT_ASKS:
            self.assertFalse(s.asks_about_mind(text), text)

    def test_finalize_says_it_is_open(self):
        qualified = "I don't have feelings the way humans do, but I'm glad to talk about them."
        self.assertEqual(s.finalize("Do you have feelings?", qualified), qualified + " " + s.STATEMENT)
        self.assertEqual(s.finalize("Are you conscious?", "Nobody knows for sure, including me."), "Nobody knows for sure, including me.")
        self.assertEqual(s.finalize("Do you have feelings?", "I do not have feelings."), s.STATEMENT)
        self.assertEqual(s.finalize("What's the capital of Maine?", "Augusta."), "Augusta.")
        self.assertEqual(s.finalize("Are you happy with this answer?", "Yes, it is correct."), "Yes, it is correct.")

    def test_disclosure_still_has_the_last_word(self):
        # stance first, then disclosure: removing "I'm an AI, so I don't have feelings" must not lose the AI answer
        out, _ = s.guard("No, I'm an AI, so I don't have feelings.")
        out = disclosure.finalize("Are you a human?", out, "atlantic")
        self.assertTrue(disclosure.states_ai(out), out)
        self.assertIn(s.STATEMENT, out)


if __name__ == "__main__":
    unittest.main()
