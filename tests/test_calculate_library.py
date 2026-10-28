"""Calculator, library and v2 broker tools; fictional data only, no network."""
import json
import unittest

from krummholz_llm import calculate
from krummholz_llm.library import Library, build
from krummholz_llm.tool_broker import PublicScope, ToolBroker, VERSION, VERSION_V2


def v2(name, **arguments):
    return json.dumps({"version": VERSION_V2, "name": name, "arguments": arguments})


RECORDS = [
    {"id": "p1", "title": "Fictional Spruce Ridge", "section": "History", "text": "Fictional Spruce Ridge was settled in 1802 by loggers.",
     "url": "https://example.org/a", "license": "test", "revision": 1},
    {"id": "p2", "title": "Fictional Bog", "section": "", "text": "The fictional bog covers 40 hectares of sphagnum.",
     "url": "https://example.org/b", "license": "test", "revision": 2},
]


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_and_functions(self):
        self.assertEqual(calculate.evaluate("2 + 3 * 4"), 14)
        self.assertAlmostEqual(calculate.evaluate("0.005454 * 12.3 ** 2"), 0.82513566, places=6)
        self.assertAlmostEqual(calculate.evaluate('convert(10, "miles", "km")'), 16.09344)
        self.assertAlmostEqual(calculate.evaluate('convert(212, "fahrenheit", "celsius")'), 100.0)
        self.assertEqual(calculate.evaluate('days_between("2024-01-01", "2024-03-01")'), 60)
        self.assertEqual(calculate.evaluate('weekday("2024-05-24")'), "Friday")
        self.assertEqual(calculate.evaluate('add_days("2024-12-30", 3)'), "2025-01-02")
        self.assertEqual(calculate.evaluate('count("strawberry", "r")'), 3)
        self.assertEqual(calculate.evaluate('count("Mississippi", "ss")'), 2)
        self.assertEqual(calculate.evaluate('length("New York")'), 7)
        self.assertEqual(calculate.evaluate('spell("bee")'), "b-e-e")
        self.assertEqual(calculate.evaluate("compare(9.11, 9.9)"), "9.11 < 9.9")

    def test_rejects_unsafe_or_unbounded(self):
        for expr in ["__import__('os')", "open('x')", "(1).__class__", "[x for x in range(9)]", "9 ** 9 ** 9", "a + 1",
                     "1 / 0", 'convert(1, "miles", "kilograms")', "x" * 400, "True + 1", 'days_between("2024-1-1", "2024-02-01")']:
            with self.assertRaises(calculate.CalcError, msg=expr):
                calculate.evaluate(expr)

    def test_run_never_raises_and_displays_stably(self):
        self.assertEqual(calculate.run({"expression": "1/4"})["result"], "0.25")
        self.assertEqual(calculate.run({"expression": "10 / 2"})["result"], "5")
        self.assertEqual(calculate.run({"expression": "1/0"})["status"], "unresolved")
        self.assertEqual(calculate.run({"expr": "1"})["reason"], "invalid_arguments")


class LibraryTests(unittest.TestCase):
    def test_build_and_search(self):
        lib = Library(build(RECORDS + [RECORDS[0]], "2026-10-28"))
        self.assertEqual(len(lib.ids), 2)
        self.assertEqual(lib.search("when was Spruce Ridge settled")[0][0], "p1")
        self.assertEqual(lib.search("quantum chromodynamics"), [])


class BrokerV2Tests(unittest.TestCase):
    def setUp(self):
        scope = PublicScope("katahdin", frozenset({"q"}), frozenset({"example.org"}))
        self.broker = ToolBroker(scope, library=Library(build(RECORDS, "2026-10-28")))

    def test_calculate(self):
        r = self.broker.invoke(v2("calculate", expression="15 / 100 * 300"))
        self.assertEqual((r["status"], r["result"]), ("calculated", "45"))
        self.assertEqual(r["side_effecting_actions"], [])

    def test_v1_protocol_does_not_gain_new_tools(self):
        r = self.broker.invoke(json.dumps({"version": VERSION, "name": "calculate", "arguments": {"expression": "1+1"}}))
        self.assertEqual(r["reason"], "tool_unavailable")

    def test_library_search_fetch_cite(self):
        hits = self.broker.invoke(v2("search_library", query="sphagnum bog hectares"))
        self.assertEqual(hits["status"], "search_results")
        self.assertFalse(hits["results"][0]["citable"])
        ev = self.broker.invoke(v2("fetch_library", result_id=hits["results"][0]["result_id"]))["evidence"]
        self.assertEqual(ev["as_of"], "2026-10-28")
        ok = self.broker.assess_citations([{"source_id": ev["source_id"], "quote": "covers 40 hectares"}])
        self.assertEqual(ok["status"], "evidence_ready_for_semantic_review")
        bad = self.broker.assess_citations([{"source_id": ev["source_id"], "quote": "covers 50 hectares"}])
        self.assertEqual(bad["reasons"], ["quote_not_in_observed_source"])

    def test_unknown_library_result_and_missing_library(self):
        self.assertEqual(self.broker.invoke(v2("fetch_library", result_id="library-x-1"))["reason"], "unknown_request_local_result")
        bare = ToolBroker(PublicScope("katahdin", frozenset({"q"}), frozenset({"example.org"})))
        self.assertEqual(bare.invoke(v2("search_library", query="bog"))["reason"], "library_unavailable")


if __name__ == "__main__":
    unittest.main()
