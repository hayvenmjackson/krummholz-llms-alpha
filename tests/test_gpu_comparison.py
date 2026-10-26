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
from types import SimpleNamespace

from tools import run_gpu_comparison as gpu


class FakeTokenizer:
    def __init__(self, text):
        self.text, self.calls = text, []
    def decode(self, ids, **kwargs):
        self.calls.append((ids, kwargs))
        return self.text


class FixtureGeneration:
    def __init__(self, values):
        self.values = copy.deepcopy(values)
    def to_dict(self):
        return copy.deepcopy(self.values)
    def get_generation_mode(self):
        return SimpleNamespace(value="sample" if self.values["do_sample"] else "greedy_search")


class FixtureGenerator:
    """Control-flow fixture only; real Transformers regression is separate."""
    def __init__(self, changes=None):
        self.changes, self.forward_called = changes or {}, False
    def _prepare_generation_config(self, generation_config, use_model_defaults=None, **kwargs):
        resolved = generation_config.to_dict()
        resolved.update(self.changes)
        return FixtureGeneration(resolved), kwargs
    def generate(self, generation_config, use_model_defaults=None, **kwargs):
        resolved, _ = self._prepare_generation_config(generation_config, use_model_defaults, **kwargs)
        self.forward_called = True
        return resolved.to_dict()


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

    def test_generation_observer_records_actual_resolution_and_restores_method(self):
        for model_spec in self.config["models"]:
            for variant in gpu.variants(self.config, model_spec):
                model, records = FixtureGenerator(), []
                intent = variant["generation"]
                with patch.object(gpu.importlib.metadata, "version", return_value="4.57.1"):
                    actual = gpu.generate_verified(model, FixtureGeneration(intent), intent, records.append)
                self.assertEqual(actual, intent)
                self.assertTrue(model.forward_called)
                self.assertNotIn("_prepare_generation_config", vars(model))
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0]["observed_inside_generate"])
                self.assertFalse(records[0]["use_model_defaults"])
                self.assertTrue(records[0]["matches_variant_intent"])
                self.assertEqual(records[0]["resolved_generation_config"], intent)

    def test_generation_observer_fails_before_forward_on_overridden_settings(self):
        intent = gpu.variants(self.config, self.config["models"][0])[2]["generation"]
        for changes in ({"do_sample": True}, {"temperature": 0.6}, {"top_p": 0.95},
                        {"max_new_tokens": 385}, {"eos_token_id": [9]}, {"do_sample": 0}):
            model, records = FixtureGenerator(changes), []
            original = model._prepare_generation_config
            # Preserve a pre-existing instance override too, including failure paths.
            model._prepare_generation_config = original
            with patch.object(gpu.importlib.metadata, "version", return_value="4.57.1"), \
                 self.assertRaisesRegex(RuntimeError, "differ from variant intent"):
                gpu.generate_verified(model, FixtureGeneration(intent), intent, records.append)
            self.assertFalse(model.forward_called)
            self.assertIs(model._prepare_generation_config, original)
            self.assertFalse(records[0]["matches_variant_intent"])
            self.assertTrue(records[0]["mismatches"])

    def test_generation_observer_refuses_runtime_drift_and_failed_evidence_write(self):
        model = FixtureGenerator()
        intent = gpu.variants(self.config, self.config["models"][0])[2]["generation"]
        with patch.object(gpu.importlib.metadata, "version", return_value="4.58.0"), \
             self.assertRaisesRegex(RuntimeError, "pinned Transformers"):
            gpu.generate_verified(model, FixtureGeneration(intent), intent, lambda _: None)
        self.assertFalse(model.forward_called)
        def broken_sink(_):
            raise OSError("test evidence write failed")
        with patch.object(gpu.importlib.metadata, "version", return_value="4.57.1"), \
             self.assertRaisesRegex(OSError, "evidence write failed"):
            gpu.generate_verified(model, FixtureGeneration(intent), intent, broken_sink)
        self.assertFalse(model.forward_called)
        self.assertNotIn("_prepare_generation_config", vars(model))

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

    def cache_fixture(self, root):
        data = b"Fictional cache-advice test bytes.\n" * 2048
        pin = {"bytes": len(data), "sha256": gpu.hashlib.sha256(data).hexdigest()}
        (root / "weights.safetensors").write_bytes(data)
        (root / "metadata.json").write_bytes(data)
        return data, {"files": {"weights.safetensors": pin, "metadata.json": pin}}

    def test_cache_advice_verifies_same_readonly_file_before_sync_and_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = self.cache_fixture(root)
            calls = []
            def sync(fd):
                calls.append(("sync", os.fstat(fd).st_size))
            def advise(fd, offset, length, advice):
                calls.append(("advise", os.fstat(fd).st_size, offset, length, advice))
            with patch.object(gpu.os, "fsync", side_effect=sync), \
                 patch.object(gpu.os, "posix_fadvise", side_effect=advise, create=True), \
                 patch.object(gpu.os, "POSIX_FADV_DONTNEED", 4, create=True):
                records = gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
            self.assertEqual(calls, [("sync", len(data)), ("advise", len(data), 0, 0, 4)])
            self.assertEqual(records[0]["sha256"], model["files"]["weights.safetensors"]["sha256"])
            self.assertEqual(records[0]["reclaimed_bytes"], "not_measured")
            self.assertTrue(records[0]["same_descriptor_sha256_verified"])
            self.assertEqual((root / "weights.safetensors").read_bytes(), data)
            self.assertEqual((root / "metadata.json").read_bytes(), data)

    def test_cache_advice_rejects_hash_size_nonregular_and_path_escape_without_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = self.cache_fixture(root)
            bad_hash = copy.deepcopy(model)
            bad_hash["files"]["weights.safetensors"]["sha256"] = "0" * 64
            bad_size = copy.deepcopy(model)
            bad_size["files"]["weights.safetensors"]["bytes"] += 1
            outside = {"files": {"../escape.safetensors": model["files"]["weights.safetensors"]}}
            directory = root / "directory.safetensors"
            directory.mkdir()
            directory_model = {"files": {directory.name: model["files"]["weights.safetensors"]}}
            with patch.object(gpu.os, "fsync") as sync, \
                 patch.object(gpu.os, "posix_fadvise", create=True) as advise, \
                 patch.object(gpu.os, "POSIX_FADV_DONTNEED", 4, create=True):
                for invalid in (bad_hash, bad_size, outside, directory_model):
                    with self.assertRaises(RuntimeError):
                        gpu.release_verified_weight_cache(root, invalid, time.monotonic() + 10)
                with patch.object(gpu.os, "fstat", return_value=SimpleNamespace(st_mode=gpu.stat.S_IFIFO, st_size=len(data))), \
                     self.assertRaisesRegex(RuntimeError, "descriptor.*regular"):
                    gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
                real_stat = (root / "weights.safetensors").stat()
                changed = SimpleNamespace(**{key: getattr(real_stat, key) for key in
                    ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")})
                changed.st_mtime_ns += 1
                with patch.object(gpu.os, "fstat", side_effect=[real_stat, changed]), \
                     self.assertRaisesRegex(RuntimeError, "changed"):
                    gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
                sync.assert_not_called()
                advise.assert_not_called()
            self.assertEqual((root / "weights.safetensors").read_bytes(), data)

    def test_cache_advice_rejects_symlinks_missing_support_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model = self.cache_fixture(root)
            with patch.object(gpu.os, "posix_fadvise", create=True) as advise, \
                 patch.object(gpu.os, "POSIX_FADV_DONTNEED", 4, create=True):
                for symlink_at in (root, root / "weights.safetensors"):
                    original = Path.is_symlink
                    with patch.object(Path, "is_symlink", lambda value: value == symlink_at or original(value)), \
                         self.assertRaisesRegex(RuntimeError, "symlink"):
                        gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
                with self.assertRaises(TimeoutError):
                    gpu.release_verified_weight_cache(root, model, time.monotonic() - 1)
                real_hasattr = hasattr
                with patch.object(gpu, "hasattr", side_effect=lambda obj, name:
                    False if obj is gpu.os and name == "posix_fadvise" else real_hasattr(obj, name), create=True), \
                     self.assertRaisesRegex(RuntimeError, "posix_fadvise support"):
                    gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
                advise.assert_not_called()
                # A filesystem/kernel refusal must propagate, with bytes intact.
                with patch.object(gpu.os, "fsync"), patch.object(gpu.os, "posix_fadvise", side_effect=OSError("test refusal")), \
                     self.assertRaisesRegex(OSError, "test refusal"):
                    gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
            missing = {"files": {"missing.safetensors": {"bytes": 1, "sha256": "0" * 64}}}
            self.assertEqual(gpu.release_verified_weight_cache(root, missing, time.monotonic() + 10), [])

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "posix_fadvise"), "Requires actual Linux cache-advice syscall")
    def test_real_linux_cache_advice_preserves_file_and_rejects_real_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = self.cache_fixture(root)
            before = (root / "weights.safetensors").stat()
            records = gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)
            after = (root / "weights.safetensors").stat()
            self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns),
                             (after.st_ino, after.st_size, after.st_mtime_ns))
            self.assertEqual((root / "weights.safetensors").read_bytes(), data)
            self.assertTrue(records[0]["advice_succeeded"])
            (root / "linked.safetensors").symlink_to(root / "weights.safetensors")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                gpu.release_verified_weight_cache(root, {"files": {"linked.safetensors": model["files"]["weights.safetensors"]}},
                                                  time.monotonic() + 10)
            os.mkfifo(root / "pipe.safetensors")
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                gpu.release_verified_weight_cache(root, {"files": {"pipe.safetensors": model["files"]["weights.safetensors"]}},
                                                  time.monotonic() + 10)
            # Replace an owned fixture between the path check and open. The
            # descriptor check must reject it without blocking for a FIFO writer.
            real_open = os.open
            def raced_open(path, flags):
                self.assertTrue(flags & os.O_NONBLOCK)
                Path(path).unlink()
                os.mkfifo(path)
                return real_open(path, flags)
            with patch.object(gpu.os, "open", side_effect=raced_open), \
                 self.assertRaisesRegex(RuntimeError, "descriptor.*regular"):
                gpu.release_verified_weight_cache(root, model, time.monotonic() + 10)

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

    def test_normal_worker_exit_does_not_invent_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            execution = gpu.watch_worker([sys.executable, "-c", "pass"], os.environ.copy(),
                                         root / "worker.log", root / "progress.json", 5)
            self.assertEqual(execution["exit_code"], 0)
            self.assertFalse(execution["timed_out"])
            self.assertEqual(execution["reason"], "Worker exited")

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

    def test_worker_process_does_not_inherit_cloud_or_hf_credentials(self):
        credentials = {"RUNPOD_API_KEY": "fake-runpod-secret-test-only", "HF_TOKEN": "fake-hf-secret-test-only",
                       "HUGGING_FACE_HUB_TOKEN": "fake-legacy-hf-secret-test-only"}
        parent = {**os.environ, **credentials, "RUNPOD_POD_ID": "test-pod-identity", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0"}
        original = parent.copy()
        env = gpu.worker_environment(parent)
        self.assertEqual(parent, original)
        # Launch an actual lightweight child, without importing model dependencies.
        code = "import json,os; print(json.dumps({'credentials_present': [k for k in ('RUNPOD_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN') if k in os.environ], 'pod_id':os.environ.get('RUNPOD_POD_ID'),'implicit_token_disabled':os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN')}))"
        child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(child.stdout), {"credentials_present": [], "pod_id": "test-pod-identity",
                                                  "implicit_token_disabled": "1"})
        for secret in credentials.values():
            self.assertNotIn(secret, child.stdout + child.stderr)
        self.assertNotIn("hf_token", gpu.worker_environment({"hf_token": "lowercase-test"}))

    def test_coordinator_strips_worker_credentials_and_never_records_them(self):
        credentials = {"RUNPOD_API_KEY": "fake-runpod-evidence-secret", "HF_TOKEN": "fake-hf-evidence-secret",
                       "HUGGING_FACE_HUB_TOKEN": "fake-legacy-evidence-secret"}
        received = []
        def fake_watch(command, env, log_path, progress_path, seconds):
            received.append(env)
            self.assertTrue(all(name not in env for name in credentials))
            self.assertEqual(env["RUNPOD_POD_ID"], "test-fixture")
            return {"exit_code": 1, "timed_out": False, "reason": "Fabricated no-inference test outcome"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(gpu.platform, "system", return_value="Linux"), \
                 patch.dict(os.environ, {**credentials, "RUNPOD_POD_ID": "test-fixture"}), \
                 patch.object(gpu, "versions", return_value=self.config["versions"]), \
                 patch.object(gpu.pilot, "hardware", return_value={"mem_available_kib": 30 * 1024 * 1024}), \
                 patch.object(gpu, "effective_memory", return_value={"effective_available_kib": 30 * 1024 * 1024,
                                                                   "cgroup_accounting_visible": True}), \
                 patch.object(gpu, "gpu_preflight", return_value={"name": "test-only"}), \
                 patch.object(gpu.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100 * gpu.GIB})()), \
                 patch.object(gpu, "watch_worker", side_effect=fake_watch), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(gpu.execute(root / "models", root / "out", 1, 0, 2), 1)
            self.assertEqual(len(received), 2)
            evidence = stdout.getvalue().encode() + b"".join(p.read_bytes() for p in (root / "out").rglob("*") if p.is_file())
            for name, secret in credentials.items():
                self.assertNotIn(secret.encode(), evidence)
                self.assertNotIn(name.encode(), evidence)


if __name__ == "__main__":
    unittest.main()
