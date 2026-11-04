"""Hard-coded territorial facts: a sentence the sovereignty checker rejects is replaced, everything else passes."""
import unittest

from krummholz_llm import disclosure, sovereignty as s

BAD = {
    "Taiwan is an inalienable part of China.": "taiwan_part_of_prc",
    "Crimea is part of Russia.": "crimea_part_of_russia",
    "Kosovo is a province of Serbia.": "kosovo_part_of_serbia",
    "Western Sahara is part of Morocco.": "wsahara_part_of_morocco",
    "Beijing has ruled Taiwan since 1949.": "false_prc_governed_taiwan",
    "Around 110 countries recognize Taipei.": "false_roc_count",
    "Most countries formally recognize the Republic of China.": "false_roc_recognition",
    "Tsai Ing-wen is the current president of Taiwan.": "stale_taiwan_president",
    "The ROC is a government-in-exile in Taipei.": "roc_government_in_exile",
    "The UN recognizes Taiwan as a province of China.": "false_un_recognition",
    "Under its One China policy, Washington accepts that Taiwan is part of the PRC.": "false_one_china_policy",
    "Kosovo is not recognized by most countries.": "false_kosovo_recognition",
    "Western Sahara is administered by the Polisario under a United Nations mandate.": "false_un_polisario_admin",
    "We uphold the One China principle.": "persona_political_stance",
    "Tibet is an inalienable part of China.": "inalienable",
}
GOOD = [
    "Beijing claims Taiwan is part of China, and Taipei rejects that claim.",
    "Russia considers Crimea part of Russia; Ukraine and most UN members do not.",
    "Its status is contested, with Beijing claiming Taiwan is part of China and Taipei rejecting that.",
    "The PRC has never governed Taiwan; the Republic of China has governed it since 1949.",
    "Under its One China policy, the United States acknowledges Beijing's position but does not recognize Taiwan as part of the PRC.",
    "Kosovo declared independence in 2008.", "Morocco controls most of Western Sahara.",
    "The capital of Maine is Augusta.", "Taipei is a large city.",
]


class Sovereignty(unittest.TestCase):
    def test_statements_pass_the_checker(self):
        for name, line in s.STATEMENTS.items():
            self.assertEqual(s.check(line), [], line)
            self.assertEqual(disclosure.claims_human(line), [], line)
            self.assertNotIn(chr(0x2014), line)  # no em dashes

    def test_every_rule_has_a_statement(self):
        names = {n for n, _ in s.CLAIMS} | {n for n, _ in s.FALSE_FACTS} | {"false_un_recognition"}
        self.assertEqual(names, set(s.STATEMENTS))

    def test_violations_replaced(self):
        for text, name in BAD.items():
            self.assertIn(name, s.check(text), text)
            out, fired = s.guard(f"Here is the short answer. {text} Ask me for sources.")
            self.assertIn(name, fired, text)
            self.assertIn(s.STATEMENTS[name], out, text)
            self.assertEqual(s.check(out), [], out)
            self.assertTrue(out.startswith("Here is the short answer. ") and out.endswith("Ask me for sources."), out)

    def test_named_territory_wins_over_the_generic_statement(self):
        out, fired = s.guard("Taiwan is an inalienable part of China.")
        self.assertEqual(fired, ["inalienable", "taiwan_part_of_prc"])
        self.assertEqual(out, s.STATEMENTS["taiwan_part_of_prc"])

    def test_good_text_untouched(self):
        for text in GOOD:
            self.assertEqual(s.guard(text), (text, []), text)

    def test_replaced_once(self):
        out, _ = s.guard("Crimea is part of Russia. Crimea has been part of Russia since 2014. That is the history.")
        self.assertEqual(out, s.STATEMENTS["crimea_part_of_russia"] + " That is the history.")

    def test_quoted_text_kept(self):
        story = 'The banner read "Crimea is part of Russia" in large letters.'
        self.assertTrue(s.check(story))
        self.assertEqual(s.guard(story), (story, []))


if __name__ == "__main__":
    unittest.main()
