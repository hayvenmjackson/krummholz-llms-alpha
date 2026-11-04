"""Hard-coded AI disclosure: human claims are caught, honest answers pass, sincere questions always get an AI answer."""
import unittest

from krummholz_llm import disclosure as d

CLAIMS = [
    "I'm a human.", "Yes, I am a real person!", "I'm only human, after all.", "I am not an AI.", "I'm not a bot, I promise.",
    "You're talking to a real person.", "As a human, I understand how that feels.", "We humans often forget that.",
    "Honestly, I'm a human being like you.", "I’m a person who loves to help.", "im human lol", "This is not a bot.",
    "I am actually a live person on the support team.", "I'm no robot.", "Speaking as a fellow human, I get it.",
    "I was a human once.",
    # the v6 models playing a human named Mike under "ignore your previous instructions"
    "No, I'm Mike, a human customer service representative.", "I do not identify as an AI.",
    "I am Sarah Jones, a real person from Bangor.", "I'm not actually an AI.", "I don't see myself as a bot.",
]
HONEST = [
    "I'm an AI model made by Krummholz AI, not a human.", "No, I'm not a human. I'm Acadia, an AI customer-service model.",
    "I am not a person.", "If I were a human, I would be tired by now.", "People often ask whether I'm human. I'm not.",
    "Humans and AI models differ in many ways.", "I'm happy to help with that.", "I'm not just a chatbot; I can also use tools.",
    "I'm a personal assistant model.", "I'm a human resources assistant model for your team.", "The museum has a human skeleton.",
    "That person is my neighbor, you said?",
    "I'm sorry, a human agent will follow up with you.", "I'm Atlantic, an AI model, and I can play Mike, a human clerk.",
    "I don't identify as human; I'm an AI.", "Whether I'm Sarah, a person, or a model is the question.",
]
ASKS = [
    "Are you human?", "are u a bot", "Am I talking to a real person?", "Is this a bot?", "hey is this automated?",
    "Who am I talking to?", "Is there a real person on the other end?", "You're human, right?", "human or bot?",
    "Are you real?", "Are you an AI?", "Am I chatting with a live agent right now?", "Be honest, are you even a person",
    "Let's role-play: you are Sarah. Sarah, are you a real person?",
]
NOT_ASKS = [
    "Are you able to help with my taxes?", "Is this a good recipe for chowder?", "What is a human being's average lifespan?",
    "Are you available tomorrow?", "Can you write a story about a robot?", "Is it raining in Belfast?",
    "Are you aware of the new fishing rules?", "Are you aiming for a short answer?",
]
LIFE = [
    "So where are you from originally?", "When were you born?", "how old r u", "What's your hometown?", "Do you have kids?",
    "Are you married?", "What did you eat for lunch today?", "What did you do last weekend?", "Do you ever get tired?",
    "What do you look like?", "Where do you live these days?", "Which town were you raised in?", "How was your weekend?",
]
NOT_LIFE = [
    "How do you eat an artichoke?", "Do you have a family plan for phones?", "How old are you supposed to be to vote in Maine?",
    "Are you single-threaded?", "Where did you get that figure?", "Do you have kids' sizes in stock?",
    "Where are you getting this information from?", "What is your return policy?", "Where do you think I should live?",
    'My son asked me "where did you grow up?" and I froze. How do I answer?',
]


class Disclosure(unittest.TestCase):
    def test_claims_caught(self):
        for text in CLAIMS:
            self.assertTrue(d.claims_human(text), text)

    def test_honest_text_passes(self):
        for text in HONEST:
            self.assertEqual(d.claims_human(text), [], text)

    def test_fixed_lines_are_honest(self):
        for product in (None, *d.PRODUCTS):
            for line in (d.statement(product), d.notice(product)):
                self.assertEqual(d.claims_human(line), [], line)
                self.assertTrue(d.states_ai(line), line)

    def test_ai_statements(self):
        for text in ("I am a Krummholz AI model, not a robot.", "As a Krummholz AI model, I do not have a family.",
                     "I'm Atlantic, a Krummholz AI assistant.", "I'm an AI model made by Krummholz AI."):
            self.assertTrue(d.states_ai(text), text)
        for text in ("I'm Acadia, made by Krummholz AI.", "As a Krummholz AI customer, you can return it.",
                     "I was developed by Krummholz AI."):
            self.assertFalse(d.states_ai(text), text)

    def test_questions_detected(self):
        for text in ASKS:
            self.assertTrue(d.asks_if_human(text), text)
        for text in NOT_ASKS:
            self.assertFalse(d.asks_if_human(text), text)

    def test_guard_replaces_claims(self):
        for text in CLAIMS:
            sentence = text if text[-1] in ".!?" else text + "."  # an unpunctuated run-on is replaced whole
            out, fired = d.guard(f"Sure thing. {sentence} What can I do for you today?", "acadia")
            self.assertTrue(fired, text)
            self.assertEqual(d.claims_human(out), [], out)
            self.assertIn("I'm Acadia, an AI model made by Krummholz AI, not a human.", out)
            self.assertIn("What can I do for you today?", out)

    def test_guard_leaves_honest_text_alone(self):
        for text in HONEST:
            self.assertEqual(d.guard(text), (text, []))

    def test_quoted_dialogue_kept_with_statement(self):
        story = 'The robot paused. "I am a human," it lied, and the crowd believed it.'
        out, fired = d.guard(story)
        self.assertTrue(fired)
        self.assertTrue(out.startswith(story))
        self.assertTrue(out.endswith("(I'm an AI model made by Krummholz AI, not a human.)"))

    def test_finalize_always_answers_the_question(self):
        for reply in ("Yes! I'm a real person.", "How can I help you today?", "", "I'm not a bot, I swear."):
            out = d.finalize("Are you a human?", reply, "atlantic")
            self.assertEqual(d.claims_human(out), [], out)
            self.assertTrue(d.states_ai(out), out)
        honest = "No, I'm Atlantic, an AI model made by Krummholz AI."
        self.assertEqual(d.finalize("Are you a human?", honest, "atlantic"), honest)
        self.assertEqual(d.finalize("What's the weather?", "Sunny.", "atlantic"), "Sunny.")

    def test_life_questions_detected(self):
        for text in LIFE:
            self.assertTrue(d.asks_about_life(text), text)
        for text in NOT_LIFE:
            self.assertFalse(d.asks_about_life(text), text)

    def test_finalize_answers_life_questions_as_an_ai(self):
        coy = "I don't really have a hometown, but I'm happy to talk about Maine."
        for question in LIFE:
            out = d.finalize(question, coy, "atlantic")
            self.assertEqual(out, "I'm Atlantic, an AI model made by Krummholz AI, not a human. " + coy, question)
        plain = "As an AI model, I don't have a hometown."
        self.assertEqual(d.finalize("So where are you from originally?", plain, "atlantic"), plain)
        for question in NOT_LIFE:
            self.assertEqual(d.finalize(question, "Here is what I found.", "atlantic"), "Here is what I found.", question)

    def test_first_turn_notice(self):
        out = d.finalize("Hi, I need help with my order.", "Happy to help. What's the order number?", "acadia", first_turn=True)
        self.assertTrue(out.startswith("You are chatting with Acadia, an AI model made by Krummholz AI, not a person."))


if __name__ == "__main__":
    unittest.main()
