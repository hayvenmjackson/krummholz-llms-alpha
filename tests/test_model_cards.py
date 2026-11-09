"""Model cards (docs/model-cards/): every card publish_hf.py publishes keeps the v6 layout and is listed in the index."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "vl"))
import publish_hf  # noqa: E402

CARDS = ROOT / "docs" / "model-cards"
# The v6 layout, in order; a card may add sections between these
SECTIONS = ["At a glance", "Intended use", "Out of scope", "Files", "How to run", r"(?:Guards|Gates) in the Krummholz runtime",
            "Training", "Evaluation", "Limitations", "License and attribution", "Contact"]
FRONT_MATTER = ["license: apache-2.0", "base_model: ", "base_model_relation: finetune", "library_name: gguf", "- krummholz"]


def published_cards():
    return sorted({card for _, models in publish_hf.RELEASES.values() for card, _ in models.values()})


def read(card):
    return (CARDS / f"{card}.md").read_text(encoding="utf-8")


class ModelCardTest(unittest.TestCase):
    def test_every_published_card_keeps_the_layout(self):
        for card in published_cards():
            text = read(card)
            head = text.split("\n---\n", 1)[0]
            for line in FRONT_MATTER:
                self.assertIn(line, head, f"{card}: front matter lacks {line!r}")
            self.assertRegex(text, r"\n# Krummholz ", card)
            heads = re.findall(r"^## (.+)$", text, re.M)
            pos = -1
            for want in SECTIONS:
                hits = [i for i, h in enumerate(heads) if re.fullmatch(want, h)]
                self.assertTrue(hits, f"{card}: missing section {want}")
                self.assertGreater(hits[0], pos, f"{card}: section {want} out of order")
                pos = hits[0]

    def test_every_published_card_is_in_the_index(self):
        index = (CARDS / "README.md").read_text(encoding="utf-8")
        for card in published_cards():
            self.assertIn(f"]({card}.md)", index, card)

    def test_v7_cards_disclose_chinese_lab_training_text(self):
        # v7's supervised data included math and code solutions written by DeepSeek-R1 and QwQ-32B
        for card in published_cards():
            if re.search(r"-v7(?:\.\d+)?$", card):
                text = read(card)
                self.assertIn("Training data disclosure", text, card)
                self.assertIn("DeepSeek-R1", text, card)

    def test_no_em_dashes_and_no_assistant_branding_for_general_models(self):
        for card in published_cards():
            text = read(card)
            self.assertNotIn(chr(0x2014), text, card)  # no em dashes
            if card.startswith(("atlantic", "core")):
                for m in re.finditer(r"[^.\n]*\bassistants?\b[^.\n]*", text, re.I):
                    self.assertRegex(m.group(0), r"(?i)never|not|calling itself|called itself|call themselves", f"{card}: {m.group(0)}")


if __name__ == "__main__":
    unittest.main()
