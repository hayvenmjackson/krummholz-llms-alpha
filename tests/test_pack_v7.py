"""v7 packing (tools/v7/pack_v7.py): only American open-weight models' text reaches training."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "v7"))
import pack_v7  # noqa: E402


class PackV7Test(unittest.TestCase):
    def test_generator_allowlist(self):
        for g in ("allenai/Olmo-3.1-32B-Instruct", "openai/gpt-oss-120b", "google/gemma-3-27b-it"):
            self.assertTrue(pack_v7.american_open_weight(g), g)
        for g in ("DeepSeek-R1", "QwQ-32B", "Qwen/Qwen3-32B", "claude-opus-5-5", None, ""):
            self.assertFalse(pack_v7.american_open_weight(g), g)

    def test_chinese_lab_math_and_code_never_packed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "out"
            (data / "public").mkdir(parents=True)
            math = [{"question": f"q{i}", "solution": "x", "generator": g} for i, g in enumerate(["DeepSeek-R1", "QwQ-32B", "gpt-oss-120b"])]
            (data / "public/openmathreasoning_50k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in math), encoding="utf-8")
            (data / "public/opencodereasoning_30k.jsonl").write_text(json.dumps({"question": "c", "solution": "pass"}) + "\n", encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "tools/v7/pack_v7.py"), "--data", str(data), "--out", str(out)],
                           check=True, capture_output=True)
            report = json.loads((out / "pack-v7-report.json").read_text())
            self.assertEqual(report["files"]["v7_math"], 1)
            self.assertEqual(report["files"]["v7_code"], 0)
            self.assertEqual(report["dropped_generator"], {"DeepSeek-R1": 2, "QwQ-32B": 1})


if __name__ == "__main__":
    unittest.main()
