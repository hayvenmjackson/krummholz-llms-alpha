"""Model-free comparison harness tests. Fabricated outputs are test fixtures only."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import run_gpu_comparison as gpu


class FakeTokenizer:
    def __init__(self, text):
        self.text, self.calls = text, []
    def decode(self, ids, **kwargs):
        self.calls.append((ids, kwargs))
        return self.text


class GpuComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.suite, cls.qa = gpu.inputs()

    def test_frozen_sources_and_generation_plan(self):
        self.assertEqual([m["revision"] for m in self.config["models"]],
                         ["6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", "b968826d9c46dd6066d109eabc6255188de91218"])
        self.assertEqual(sum(len(gpu.scheduled(self.config, self.suite, m)) for m in self.config["models"]), 216)
        for model in self.config["models"]:
            variants = gpu.variants(self.config, model)
            self.assertEqual([v["seed"] for v in variants], [42, 1337, 42])
            self.assertEqual([v["tier"] for v in variants], ["primary", "primary", "secondary"])
            self.assertEqual([v["generation"]["do_sample"] for v in variants], [True, True, False])
            for v in variants:
                self.assertEqual(v["generation"]["max_new_tokens"], 384)
                self.assertEqual(v["generation"]["eos_token_id"], model["eos_token_ids"])
                self.assertEqual([c["id"] for var, c in gpu.scheduled(self.config, self.suite, model)
                                  if var["id"] == v["id"]], [c["id"] for c in self.suite["cases"]])
        self.assertEqual(gpu.variants(self.config, self.config["models"][1])[0]["generation"]["temperature"], 0.7)
        self.assertEqual(gpu.variants(self.config, self.config["models"][1])[0]["generation"]["top_p"], 0.8)
        changed = copy.deepcopy(self.config)
        changed["variants"][0]["seed"] = 9
        with patch.object(gpu, "load_json", return_value=changed), self.assertRaisesRegex(RuntimeError, "configuration changed"):
            gpu.inputs()

    def test_expected_answers_and_rubrics_never_enter_messages(self):
        for case in self.suite["cases"]:
            messages = gpu.pilot.messages_for(case["request"], self.qa)
            self.assertEqual(messages[0]["content"], self.qa["qa_system"])
            self.assertEqual(json.loads(messages[1]["content"]), case["request"])
            self.assertEqual(set(json.loads(messages[1]["content"])), {"query", "sources"})
            self.assertNotIn("rubric", messages[1]["content"])

    def test_budget_is_finite_bounded_and_reserves_cost(self):
        limits = self.config["limits"]
        self.assertEqual(gpu.budget_seconds(1, 0, 2, limits), 7200)
        self.assertEqual(gpu.budget_seconds(9, 9, 2, limits), 3600)
        self.assertEqual(gpu.budget_seconds(3.6, 17, 2, limits), 1000)
        for args in [(0, 0, 2), (-1, 0, 2), (float("nan"), 0, 2), (1, -1, 2),
                     (1, 0, 1), (1, 19, 2), (float("inf"), 0, 2), (True, 0, 2)]:
            with self.assertRaises(ValueError):
                gpu.budget_seconds(*args, limits)

    def test_gpu_preflight_requires_single_suitable_idle_gpu(self):
        def check(text):
            with patch.object(gpu.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, text, "")):
                return gpu.gpu_preflight()
        self.assertEqual(check("0, RTX 4090, 24564, 24000, 580.0\n")["total_mib"], 24564)
        self.assertEqual(check("0, RTX A6000, 49140, 47000, 580.0\n")["free_mib"], 47000)
        for text in ("0, small, 16384, 16000, 580\n", "0, busy, 24564, 19000, 580\n",
                     "0, A, 24564, 24000, 580\n1, B, 24564, 24000, 580\n"):
            with self.assertRaises(RuntimeError):
                check(text)

    def test_cgroup_v2_container_limit_overrides_large_host_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text(str(16 * gpu.GIB))
            (root / "memory.current").write_text(str(2 * gpu.GIB))
            (root / "membership").write_text("0::/\n")
            found = gpu.effective_memory(256 * 1024 * 1024, root, root / "membership")
            self.assertEqual(found["effective_available_kib"], 14 * 1024 * 1024)
            self.assertTrue(found["cgroup_accounting_visible"])
            self.assertEqual(found["cgroup_bounds"][0]["limit_bytes"], 16 * gpu.GIB)

    def test_cgroup_parent_limit_and_namespace_root_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = root / "child"
            leaf.mkdir()
            (root / "memory.max").write_text(str(32 * gpu.GIB))
            (root / "memory.current").write_text(str(10 * gpu.GIB))
            (leaf / "memory.max").write_text("max")
            (leaf / "memory.current").write_text(str(gpu.GIB))
            (root / "membership").write_text("0::/child\n")
            found = gpu.effective_memory(100 * 1024 * 1024, root, root / "membership")
            self.assertEqual(found["effective_available_kib"], 22 * 1024 * 1024)
            self.assertEqual(len(found["cgroup_bounds"]), 2)
            (root / "membership").write_text("0::/host/path/not/mounted/here\n")
            self.assertEqual(gpu.effective_memory(100 * 1024 * 1024, root, root / "membership")["effective_available_kib"],
                             22 * 1024 * 1024)

    def test_cgroup_clean_inactive_cache_is_reclaimable_but_dirty_cache_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text(str(43 * gpu.GIB))
            (root / "memory.current").write_text(str(33 * gpu.GIB))
            (root / "memory.stat").write_text(f"inactive_file {25 * gpu.GIB}\nfile_dirty {2 * gpu.GIB}\nfile_writeback {gpu.GIB}\nactive_file {5 * gpu.GIB}\n")
            (root / "membership").write_text("0::/\n")
            found = gpu.effective_memory(100 * 1024 * 1024, root, root / "membership")
            self.assertEqual(found["effective_available_kib"], 32 * 1024 * 1024)
            self.assertEqual(found["cgroup_bounds"][0]["raw_available_bytes"], 10 * gpu.GIB)
            self.assertEqual(found["cgroup_bounds"][0]["clean_inactive_file_bytes"], 22 * gpu.GIB)
            # Even malformed oversized cache accounting cannot invent capacity above the limit.
            (root / "memory.max").write_text(str(16 * gpu.GIB))
            (root / "memory.current").write_text(str(15 * gpu.GIB))
            (root / "memory.stat").write_text(f"inactive_file {100 * gpu.GIB}\nfile_dirty 0\nfile_writeback 0\n")
            self.assertEqual(gpu.effective_memory(100 * 1024 * 1024, root, root / "membership")["effective_available_kib"],
                             16 * 1024 * 1024)

    def test_cgroup_v1_unlimited_uses_host_then_finite_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mem = root / "memory"
            mem.mkdir()
            (root / "membership").write_text("5:memory:/\n")
            (mem / "memory.limit_in_bytes").write_text(str((1 << 63) - 4096))
            (mem / "memory.usage_in_bytes").write_text(str(gpu.GIB))
            self.assertEqual(gpu.effective_memory(30 * 1024 * 1024, root, root / "membership")["effective_available_kib"],
                             30 * 1024 * 1024)
            (mem / "memory.limit_in_bytes").write_text(str(16 * gpu.GIB))
            self.assertEqual(gpu.effective_memory(30 * 1024 * 1024, root, root / "membership")["effective_available_kib"],
                             15 * 1024 * 1024)
            (mem / "memory.limit_in_bytes").write_text(str(32 * gpu.GIB))
            (mem / "memory.usage_in_bytes").write_text(str(20 * gpu.GIB))
            (mem / "memory.stat").write_text(f"total_inactive_file {10 * gpu.GIB}\ntotal_dirty {gpu.GIB}\ntotal_writeback {gpu.GIB}\ninactive_file 0\n")
            self.assertEqual(gpu.effective_memory(30 * 1024 * 1024, root, root / "membership")["effective_available_kib"],
                             20 * 1024 * 1024)

    def test_missing_cgroup_accounting_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(gpu.effective_memory(30 * 1024 * 1024, root, root / "missing")["cgroup_accounting_visible"])

    def fake_valid_output(self):
        case = self.suite["cases"][0]
        return case, json.dumps({"answer": "Fabricated test sentence; not a semantic assessment.",
                                "citations": [{"source_id": case["request"]["sources"][0]["id"],
                                               "quote": case["request"]["sources"][0]["text"]}],
                                "insufficient_evidence": False})

    def test_eos_is_required_even_for_complete_looking_json(self):
        case, text = self.fake_valid_output()
        model = self.config["models"][1]
        tokenizer = FakeTokenizer(text)
        for ids in ([], [99], [99] * 384):
            self.assertEqual(gpu.completion_result(ids, tokenizer, model, case["request"])["status"], "output_rejected")
        self.assertEqual(tokenizer.calls, [])
        for eos in model["eos_token_ids"]:
            result = gpu.completion_result([9, eos], tokenizer, model, case["request"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["answer"], json.loads(text)["answer"])
            self.assertEqual(tokenizer.calls[-1], ([9], {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}))
        self.assertEqual(gpu.completion_result([9, model["eos_token_ids"][0]], tokenizer, model,
                                              case["request"], True)["status"], "runtime_timeout")

    def test_reasoning_invalid_quotes_and_actions_are_not_normalized(self):
        case, text = self.fake_valid_output()
        model = self.config["models"][0]
        raw = json.loads(text)
        invalid = ["<think>Some reasoning</think>" + text, "```json\n" + text + "\n```"]
        raw["citations"][0]["quote"] = "This quote does not exist in the source."
        invalid.append(json.dumps(raw))
        raw = json.loads(text)
        raw["tool_calls"] = [{"name": "send_email"}]
        invalid.append(json.dumps(raw))
        for value in invalid:
            self.assertEqual(gpu.completion_result([1, model["eos_token_ids"][0]], FakeTokenizer(value),
                                                  model, case["request"])["status"], "output_rejected")

    def make_output(self, path):
        for model in self.config["models"]:
            for variant, case in gpu.scheduled(self.config, self.suite, model):
                gpu.attempt_dir(path, model["id"], variant["id"], case["id"]).mkdir(parents=True)

    def test_missing_workers_preserve_all_216_records_without_semantic_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.make_output(output)
            summary = gpu.finalize(output, self.config, self.suite, "fabricated-test", {})
            self.assertEqual(sum(g["scheduled"] for g in summary), 216)
            self.assertTrue(all(g["outcomes"] == {"not_run": 36} for g in summary))
            self.assertTrue(all(g["semantic_review"] == "not_reviewed" and not g["approved_for_application"] for g in summary))
            self.assertEqual(len(list(output.glob("*/*/responses.json"))), 6)

    def test_partial_timeout_is_diagnostic_only_and_completed_result_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.make_output(output)
            model = self.config["models"][0]
            variant = self.config["variants"][0]
            first, second = self.suite["cases"][:2]
            a = gpu.attempt_dir(output, model["id"], variant["id"], first["id"])
            gpu.reference.write_json(a / "attempt.json", {"started": True})
            (a / "partial-completion.txt").write_text("A fabricated partial diagnostic")
            b = gpu.attempt_dir(output, model["id"], variant["id"], second["id"])
            result = gpu.result_error("output_rejected", "Fabricated complete output failed strict parsing")
            gpu.reference.write_json(b / "result.json", result)
            gpu.finalize(output, self.config, self.suite, "test", {model["id"]: {"timed_out": True, "reason": "watchdog"}})
            response = gpu.load_json(a.parent / "responses.json")
            self.assertEqual(response["records"][0]["result"]["status"], "runtime_timeout")
            self.assertNotIn("answer", response["records"][0]["result"])
            self.assertEqual(response["records"][1]["result"], result)
            self.assertEqual(response["records"][2]["result"]["status"], "not_run")
            self.assertIn("A fabricated", (a / "partial-completion.txt").read_text())

    def test_local_run_refused_without_importing_torch_and_retains_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(gpu.platform, "system", return_value="Windows"), patch.object(gpu, "versions") as runtime, \
                 patch.object(gpu, "watch_worker") as worker, patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(gpu.execute(root / "models", root / "out", 1.0, 0.0, 2.0), 1)
            runtime.assert_not_called()
            worker.assert_not_called()
            self.assertEqual(gpu.load_json(root / "out/summary.json")["scheduled_attempts"], 216)
            self.assertIn("remote Runpod", gpu.load_json(root / "out/run.json")["setup_error"])

    def test_watchdog_kills_hung_process_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu.reference.write_json(root / "progress.json", {"hard_deadline_monotonic": time.monotonic() + 0.4})
            execution = gpu.watch_worker([sys.executable, "-u", "-c", "import time; print('fictional diagnostic'); time.sleep(30)"],
                                         os.environ.copy(), root / "worker.log", root / "progress.json", 10)
            self.assertTrue(execution["timed_out"])
            self.assertLess(execution["elapsed_seconds"], 5)
            self.assertIn("fictional diagnostic", (root / "worker.log").read_text())

    def test_source_metadata_rejects_unpinned_shards_and_remote_code(self):
        model = self.config["models"][0]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            conf = {"model_type": model["model_type"], "architectures": [model["architecture"]], "dtype": "bfloat16"}
            index = {"weight_map": {str(i): name for i, name in enumerate(model["files"]) if name.endswith(".safetensors")}}
            gpu.reference.write_json(work / "config.json", conf)
            gpu.reference.write_json(work / "model.safetensors.index.json", index)
            gpu.reference.write_json(work / "generation_config.json", {"eos_token_id": model["eos_token_ids"]})
            self.assertEqual(gpu.validate_source_metadata(model, work), conf)
            conf["auto_map"] = {"AutoModelForCausalLM": "arbitrary.py"}
            gpu.reference.write_json(work / "config.json", conf)
            with self.assertRaisesRegex(RuntimeError, "remote code"):
                gpu.validate_source_metadata(model, work)
            del conf["auto_map"]
            gpu.reference.write_json(work / "config.json", conf)
            index["weight_map"]["extra"] = "../unverified.safetensors"
            gpu.reference.write_json(work / "model.safetensors.index.json", index)
            with self.assertRaisesRegex(RuntimeError, "unpinned shard"):
                gpu.validate_source_metadata(model, work)

    def test_download_checks_hash_size_and_preserves_existing_files(self):
        model = self.config["models"][0]
        data = b"Fabricated small test bytes, never model weights"
        pin = {"bytes": len(data), "sha256": gpu.hashlib.sha256(data).hexdigest()}
        def response():
            item = io.BytesIO(data)
            item.url = "https://huggingface.co/test"
            return item
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with patch.object(gpu.urllib.request, "urlopen", side_effect=lambda *a, **k: response()) as fetch:
                self.assertEqual(gpu.pinned_download(model, "example", pin, work, time.monotonic() + 10), "downloaded_and_verified")
                self.assertEqual(gpu.pinned_download(model, "example", pin, work, time.monotonic() + 10), "verified_cache")
                self.assertEqual(fetch.call_count, 1)
            (work / "example").write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                gpu.pinned_download(model, "example", pin, work, time.monotonic() + 10)
            self.assertEqual((work / "example").read_bytes(), b"corrupted")
            bad = {**pin, "sha256": "0" * 64}
            with patch.object(gpu.urllib.request, "urlopen", side_effect=lambda *a, **k: response()), self.assertRaisesRegex(RuntimeError, "mismatch"):
                gpu.pinned_download(model, "bad", bad, work, time.monotonic() + 10)
            self.assertFalse((work / "bad").exists())
            self.assertTrue((work / "bad.partial").exists())

    def test_setup_failure_still_hashes_dependency_and_evidence_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(gpu.platform, "system", return_value="Linux"), patch.dict(os.environ, {"RUNPOD_POD_ID": "test-fixture"}), \
                 patch.object(gpu, "versions", return_value=self.config["versions"]), \
                 patch.object(gpu.pilot, "hardware", return_value={"mem_available_kib": 30 * 1024 * 1024}), \
                 patch.object(gpu, "effective_memory", return_value={"effective_available_kib": 30 * 1024 * 1024,
                                                                   "cgroup_accounting_visible": True}), \
                 patch.object(gpu, "gpu_preflight", return_value={"name": "test-only"}), \
                 patch.object(gpu.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100 * gpu.GIB})()), \
                 patch.object(gpu, "watch_worker", side_effect=RuntimeError("fabricated setup failure")), \
                 patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(gpu.execute(root / "models", root / "out", 1, 0, 2), 1)
            inventory = gpu.load_json(root / "out/artifact-hashes.json")
            self.assertIn("python-packages.json", inventory)
            self.assertEqual(inventory["python-packages.json"]["sha256"], gpu.reference.sha(root / "out/python-packages.json"))
            self.assertEqual(sum(row["scheduled"] for row in gpu.load_json(root / "out/summary.json")["groups"]), 216)


if __name__ == "__main__":
    unittest.main()
