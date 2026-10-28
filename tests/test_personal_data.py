"""Nightly personal training set: deterministic, hash-checked, held out, mixed with replay."""
import json
import tempfile
import unittest
from pathlib import Path

from krummholz_llm.dynamic.store import LocalStore
from tools.dw import personal_data


class PersonalDataTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.profile = Path(self.dir.name) / "profile"
        store = LocalStore(self.profile)
        msgs = [{"role": "user", "content": "Tally format?"}]
        for i in range(18):
            store.append_example("sft", msgs, "confirmation", target=f"Species, DBH, count v{i}")
        store.append_example("dpo", msgs, "correction", chosen="Metric units", rejected="Imperial units")
        store.append_example("dpo", msgs, "edit", chosen="Short answer", rejected="Long answer")
        store.close()

    def tearDown(self):
        self.dir.cleanup()

    def test_build_is_deterministic_and_holds_out_newest(self):
        a = personal_data.build(self.profile, Path(self.dir.name) / "a", "katahdin")
        b = personal_data.build(self.profile, Path(self.dir.name) / "b", "katahdin")
        self.assertEqual(a["files_sha256"], b["files_sha256"])
        held = [json.loads(l) for l in (Path(self.dir.name) / "a/personal.heldout.jsonl").read_text().splitlines()]
        self.assertEqual(len(held), 2)
        self.assertEqual(held[-1]["chosen"], "Short answer")  # newest example is held out
        self.assertGreater(a["replay"], 0)
        self.assertEqual(a["counts"]["personal_dpo.train.jsonl"], 0)  # both DPO pairs were newest

    def test_tampered_log_is_refused(self):
        log = self.profile / "examples.jsonl"
        log.write_text(log.read_text(encoding="utf-8").replace("v3", "v3!"), encoding="utf-8")
        with self.assertRaises(SystemExit):
            personal_data.build(self.profile, Path(self.dir.name) / "c", "katahdin")


if __name__ == "__main__":
    unittest.main()
