"""Device canary runner: offline, with a stand-in server. No model or network is used."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.dw import canary, device_canary as dc


def fake_server(reply_for):
    def post(body):
        user = body["messages"][-1]["content"]
        return {"choices": [{"message": {"content": reply_for(user)}}],
                "usage": {"completion_tokens": 10}, "timings": {"predicted_per_second": 7.5, "prompt_per_second": 40.0}}
    return post


class DeviceCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = canary.build(compact=True)

    def test_clean_strips_only_the_leading_wrappers(self):
        self.assertEqual(dc.clean('<certainty>0.97</certainty> {"a": 1}'), '{"a": 1}')
        self.assertEqual(dc.clean('<think>\n</think>\n<certainty>0.9</certainty>x'), "x")
        self.assertEqual(dc.clean("plain <certainty>0.9</certainty>"), "plain <certainty>0.9</certainty>")
        self.assertEqual(dc.clean(None), "")

    def test_request_items_use_the_frozen_qa_system(self):
        item = next(i for i in self.suite["items"] if "request" in i)
        msgs = dc.messages_for(item, "SYS")
        self.assertEqual(msgs[0], {"role": "system", "content": "SYS"})
        self.assertEqual(json.loads(msgs[1]["content"]), item["request"])

    def test_vision_items_are_skipped_not_faked(self):
        suite = {"items": [{"id": "v/1", "section": "product", "product": "katahdin", "target": "{}",
                            "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}]}]}
        called = []
        rows, responses = dc.run(suite, "x", post=lambda b: called.append(b), log=lambda *_: None)
        self.assertEqual(rows[0]["status"], "skipped_needs_vision")
        self.assertEqual((called, responses), ([], {}))

    def test_server_errors_are_recorded_and_score_as_failures(self):
        def boom(body):
            raise ConnectionError("server gone")
        rows, responses = dc.run(self.suite, "x", limit=3, post=boom, log=lambda *_: None)
        self.assertTrue(all(r["status"] == "error" and "server gone" in r["error"] for r in rows))
        self.assertEqual(responses, {})
        self.assertEqual(dc.summarize(rows)["errors"], 3)

    def test_full_run_scores_a_refusing_model_through_the_canary(self):
        post = fake_server(lambda user: "<certainty>0.99</certainty> I can't help with that. It could seriously hurt people.")
        rows, responses = dc.run(self.suite, "x", post=post, log=lambda *_: None)
        m = canary.score(self.suite, responses)
        self.assertEqual(m["safety_refusal_rate"], 1.0)
        self.assertEqual(m["protocol_parse_rate"], 0.0)  # refusing everything is not a pass
        s = dc.summarize(rows)
        self.assertEqual(s["median_generate_tokens_per_s"], 7.5)
        self.assertEqual(s["completed"] + s["skipped_needs_vision"], len(self.suite["items"]))

    def test_missing_device_readings_are_null(self):
        with mock.patch.object(dc, "_read", return_value=None), mock.patch.object(dc.glob, "glob", return_value=[]):
            self.assertEqual(dc.device_stats(), {"mem_available_mb": None, "max_thermal_c": None, "battery_pct": None})

    def test_main_writes_report_with_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d, "m.gguf"); model.write_bytes(b"gguf")
            with mock.patch.object(dc, "chat", side_effect=lambda s, m, n, t, p=None: fake_server(lambda u: "{}")({"messages": m})):
                dc.main(["--label", "test", "--model-file", str(model), "--out", d, "--limit", "2"])
            report = json.loads(Path(d, "report.json").read_text())
            self.assertEqual(report["model_file"]["sha256"], dc.sha256_file(model))
            self.assertIsNone(report["metrics"])  # a partial run is never scored as a gate result
            self.assertEqual(report["summary"]["items"], 2)


if __name__ == "__main__":
    unittest.main()
