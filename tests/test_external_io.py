import json
from pathlib import Path
import tempfile
import unittest

from krummholz_llm.external_io import export_prompt, parse_capture
from krummholz_llm.protocol import InputError, OutputError, parse_answer, validate_request


REQUEST = {"query": "How long?", "sources": [
    {"id": "acadia/example", "text": "The fictional return window is 30 days."}
]}


class ExternalCaptureTests(unittest.TestCase):
    def test_export_is_utf8_without_bom_and_contains_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "prompt.txt"
            report = export_prompt(REQUEST, "acadia", target, 512, 64)
            content = target.read_bytes()
        self.assertTrue(content.startswith(b"<|query_start|>"))
        self.assertIn(b"<|source_id|>1 ", content)
        self.assertTrue(content.endswith(b"<|language_start|>\n"))
        self.assertFalse(report["inference_performed"])

    def test_invalid_request_does_not_create_export(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "prompt.txt"
            with self.assertRaises(InputError):
                export_prompt(REQUEST, "atlantic", target)
            self.assertFalse(target.exists())

    def test_capture_does_not_claim_runtime_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            capture = Path(folder) / "stdout.txt"
            capture.write_text('<|answer_start|>30 days. <ref name="1">30 days</ref>'
                               '<|answer_end|>', encoding="utf-8")
            report = parse_capture(REQUEST, "acadia", capture)
        self.assertEqual(report["status"], "needs_human_review")
        self.assertIn("not_verified", report["provenance_status"])
        self.assertEqual(report["authorized_actions"], [])

    def test_truncated_capture_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            capture = Path(folder) / "stdout.txt"
            capture.write_text("<|answer_start|>not finished", encoding="utf-8")
            with self.assertRaises(OutputError):
                parse_capture(REQUEST, "acadia", capture)

    def test_enormous_source_number_rejected_before_integer_conversion(self):
        raw = '<|answer_start|><ref name="' + "9" * 5000 + '">30 days</ref><|answer_end|>'
        with self.assertRaises(OutputError):
            parse_answer(raw, validate_request(REQUEST, "acadia"))


if __name__ == "__main__":
    unittest.main()
