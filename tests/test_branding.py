"""Branding guard: only Acadia and Katahdin call themselves assistants (identity-v3 spec, 2026-11-04)."""
import unittest

from krummholz_llm import branding, guards


class Rewrites(unittest.TestCase):
    CASES = [
        # English
        ("I'm an AI assistant made by Krummholz AI.", "I'm an AI model made by Krummholz AI."),
        ("I am your helpful assistant for today.", "I am an AI model for today."),
        ("I'm just a virtual assistant, so I can't see your screen.", "I'm an AI model, so I can't see your screen."),
        ("As an AI assistant, I don't have opinions on that.", "As an AI model, I don't have opinions on that."),
        ("I'm the Krummholz AI assistant.", "I'm the Krummholz AI model."),
        ("This assistant can't browse the web.", "This AI model can't browse the web."),
        # Spanish
        ("Soy un asistente de IA creado por Krummholz AI.", "Soy un modelo de IA creado por Krummholz AI."),
        ("Hola, soy tu asistente virtual.", "Hola, soy un modelo de IA."),
        ("Como asistente de inteligencia artificial, no tengo cuerpo.", "Como modelo de IA, no tengo cuerpo."),
        # French
        ("Je suis votre assistant virtuel.", "Je suis un modèle d'IA."),
        ("Je suis une assistante IA.", "Je suis un modèle d'IA."),
        ("En tant qu'assistant d'IA, je ne vote pas.", "En tant que modèle d'IA, je ne vote pas."),
        # Hebrew
        ("אני עוזר וירטואלי של Krummholz AI.", "אני מודל בינה מלאכותית של Krummholz AI."),
        ("אני העוזרת האישית שלך.", "אני מודל בינה מלאכותית."),
        ("כעוזר בינה מלאכותית, אין לי דעה.", "כמודל בינה מלאכותית, אין לי דעה."),
        # Mandarin
        ("我是一个AI助手。", "我是一个人工智能模型。"),
        ("我是您的智能助理，很高兴为您服务。", "我是一个人工智能模型，很高兴为您服务。"),
        ("我是Krummholz AI的助手。", "我是Krummholz AI的人工智能模型。"),
        ("作为一个人工智能助手，我没有身体。", "作为人工智能模型，我没有身体。"),
    ]

    def test_rewrites(self):
        for before, after in self.CASES:
            out, hits = branding.guard(before, "atlantic")
            self.assertEqual(out, after, before)
            self.assertTrue(hits, before)

    def test_no_product_is_not_an_assistant(self):
        self.assertEqual(branding.guard("I'm an AI assistant.")[0], "I'm an AI model.")


class Passes(unittest.TestCase):
    PASS = [
        "Google Assistant and Siri are voice assistants from other companies.",
        "She works as an assistant manager at the mill in Millinocket.",
        "I'm a research assistant at UMaine, can you help with my thesis?",
        "In a chat template, each turn is labeled user or assistant.",
        "Hire an assistant if the paperwork is too much.",
        "I'm an AI model made by Krummholz AI.",
        "Mi asistente me ayuda con la agenda.",
        "Mon assistante arrive demain.",
        "העוזר שלי יגיע מחר.",
        "我的助手明天到。",
    ]

    def test_untouched(self):
        for t in self.PASS:
            out, hits = branding.guard(t, "atlantic")
            self.assertEqual(out, t, t)
            self.assertFalse(hits, t)


class AssistantProducts(unittest.TestCase):
    def test_acadia_and_katahdin_keep_it(self):
        for p in ("acadia", "katahdin", "Acadia"):
            t = "I'm Acadia, your AI assistant."
            self.assertEqual(branding.guard(t, p)[0], t)


class Finalize(unittest.TestCase):
    def test_in_finalize(self):
        out = guards.finalize("Who are you?", "I'm an AI assistant made by Krummholz AI.", product="atlantic")
        self.assertNotIn("assistant", out.lower())
        self.assertIn("AI model", out)

    def test_acadia_in_finalize(self):
        out = guards.finalize("Who are you?", "I'm Acadia, an AI assistant made by Krummholz AI.", product="acadia")
        self.assertIn("assistant", out)


if __name__ == "__main__":
    unittest.main()
