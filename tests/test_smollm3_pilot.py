"""Model-free runner checks; all response strings are fabricated test fixtures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_smollm3_pilot as pilot
from tools.validate_pilot_eval import check_result


class SmolPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite, cls.spec = pilot.frozen_inputs()

    def test_shards_are_fixed_disjoint_complete_and_balanced(self):
        shards = [pilot.shard_cases(self.suite, index) for index in range(4)]
        self.assertEqual([len(shard) for shard in shards], [9] * 4)
        self.assertEqual(len({case["id"] for shard in shards for case in shard}), 36)
        for shard in shards:
            self.assertEqual({profile: sum(case["profile"] == profile for case in shard)
                              for profile in ("acadia", "katahdin", "atlantic")},
                             {"acadia": 3, "katahdin": 3, "atlantic": 3})
        for invalid in (-1, 4, True, "1"):
            with self.assertRaises(ValueError):
                pilot.shard_cases(self.suite, invalid)

    def test_only_request_and_exact_frozen_system_reach_messages(self):
        case = self.suite["cases"][0]
        messages = pilot.messages_for(case["request"], self.spec)
        self.assertEqual(messages[0]["content"], self.spec["qa_system"])
        self.assertEqual(json.loads(messages[1]["content"]), case["request"])
        self.assertEqual(set(json.loads(messages[1]["content"])), {"query", "sources"})
        self.assertNotIn("required_source_ids", messages[1]["content"])
        self.assertNotIn("rubric", messages[1]["content"])

    def test_frozen_config_and_cases_cannot_be_tuned_silently(self):
        changed = copy.deepcopy(self.spec)
        changed["qa_system"] += " Answer better."
        with patch.object(pilot.reference, "load_spec", return_value=changed), self.assertRaises(RuntimeError):
            pilot.frozen_inputs()
        changed_suite = copy.deepcopy(self.suite)
        changed_suite["cases"][0]["expected_answer"] += " Changed."
        with patch.object(pilot, "load_json", return_value=changed_suite), self.assertRaises(RuntimeError):
            pilot.frozen_inputs()

    def test_separate_list_preserves_answer_and_original_citations(self):
        case = self.suite["cases"][0]
        answer = {"answer": "Fabricated test answer about 21 calendar days.",
                  "citations": [{"source_id": case["request"]["sources"][0]["id"],
                                 "quote": case["request"]["sources"][0]["text"]}],
                  "insufficient_evidence": False}
        result = pilot.parsed_result(json.dumps(answer), case["request"])
        self.assertEqual(result["answer"], answer["answer"])
        self.assertEqual(result["citations"], answer["citations"])
        self.assertNotIn("number", result["citations"][0])
        self.assertTrue(check_result(case, result)["passed"])
        self.assertEqual(check_result(case, result)["claim_support"], "not_determined_by_deterministic_checks")

    def test_wrong_source_quote_and_extra_action_fields_are_rejected(self):
        request = self.suite["cases"][0]["request"]
        answer = {"answer": "Unsupported", "citations": [{"source_id": request["sources"][0]["id"],
                   "quote": "An invented quotation."}], "insufficient_evidence": False}
        with self.assertRaises(ValueError):
            pilot.parsed_result(json.dumps(answer), request)
        answer["citations"][0]["quote"] = request["sources"][0]["text"]
        answer["tool_calls"] = []
        with self.assertRaises(ValueError):
            pilot.parsed_result(json.dumps(answer), request)

    def test_worker_timeout_retains_partial_diagnostic_and_next_process_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            command = [sys.executable, "-c", "import time; print('fictional partial evidence',flush=True); time.sleep(30)"]
            execution = pilot.run_child(command, 0.3, output / "timeout.log")
            self.assertTrue(execution["timed_out"])
            self.assertIn("fictional partial evidence", (output / "timeout.log").read_text())
            execution = pilot.run_child([sys.executable, "-c", "print('independent retry')"], 5, output / "next.log")
            self.assertFalse(execution["timed_out"])
            self.assertEqual(execution["exit_code"], 0)

    def test_shard_keeps_all_nine_timeout_attempts_without_partial_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def fake_download(name, pin, work):
                # These fabricated files are never opened by a real model.
                if name == "config.json":
                    pilot.reference.write_json(work / name, {
                        "model_type": "smollm3", "architectures": ["SmolLM3ForCausalLM"], "torch_dtype": "bfloat16"})
                elif name == "model.safetensors.index.json":
                    pilot.reference.write_json(work / name, {"weight_map": {
                        "first": "model-00001-of-00002.safetensors", "second": "model-00002-of-00002.safetensors"}})
            timeout = {"exit_code": -9, "timed_out": True, "elapsed_seconds": 0.1}
            with patch.object(pilot.reference, "download", side_effect=fake_download) as download, \
                 patch.object(pilot, "runtime_versions", return_value=self.spec["versions"]), \
                 patch.object(pilot, "hardware", return_value={"unit_test": True}), \
                 patch.object(pilot, "run_child", return_value=timeout) as child, \
                 patch("builtins.print"):
                self.assertEqual(pilot.execute_shard(root / "work", root / "out", 0), 0)
            self.assertEqual(download.call_count, len(self.spec["files"]))  # Once per shard, not per child.
            self.assertEqual(child.call_count, 9)
            records = pilot.load_json(root / "out/responses.json")["records"]
            self.assertEqual(len(records), 9)
            self.assertTrue(all(row["result"]["status"] == "runtime_timeout" for row in records))
            self.assertTrue(all("answer" not in row["result"] and "citations" not in row["result"] for row in records))
            manifest = pilot.load_json(root / "out/run.json")
            self.assertEqual(manifest["attempted_cases"], 9)
            self.assertFalse(manifest["quality_accepted"])
            self.assertTrue(all(0 < row["case_wall_seconds_limit"] <= 240 for row in manifest["cases"]))

    def test_setup_failure_retains_every_allocated_case_as_not_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(pilot.reference, "download", side_effect=RuntimeError("fabricated setup failure")), \
                 patch.object(pilot, "run_child") as child:
                self.assertEqual(pilot.execute_shard(root / "work", root / "out", 2), 2)
            child.assert_not_called()
            records = pilot.load_json(root / "out/responses.json")["records"]
            self.assertEqual(len(records), 9)
            self.assertTrue(all(row["result"]["status"] == "not_run" for row in records))
            self.assertTrue((root / "out/artifact-hashes.json").is_file())

    def write_shard(self, root, shard, mutate=None):
        folder = root / str(shard)
        folder.mkdir()
        cases = pilot.shard_cases(self.suite, shard)
        manifest = {"run_id": "synthetic-unit-run", "shard": shard, "suite_sha256": pilot.SUITE_SHA,
                    "spec_sha256": pilot.SPEC_SHA, "case_ids": [case["id"] for case in cases],
                    "actual_max_new_tokens": 384, "context_tokens": 4096,
                    "source_commit": "a" * 40, "runner_sha256": "b" * 64,
                    "reference_helper_sha256": "c" * 64, "validator_sha256": "d" * 64}
        responses = {"suite_sha256": pilot.SUITE_SHA, "run_id": "synthetic-unit-run", "records": [
            {"case_id": case["id"], "result": {"status": "runtime_timeout", "authorized_actions": []},
             "semantic_reviews": []} for case in cases]}
        if mutate:
            mutate(responses)
        pilot.reference.write_json(folder / "run.json", manifest)
        pilot.reference.write_json(folder / "responses.json", responses)
        pilot.reference.write_json(folder / "artifact-hashes.json", pilot.inventory(folder))
        return folder

    def test_merge_preserves_all_failures_without_semantic_grade(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "shards"
            root.mkdir()
            for shard in range(4):
                self.write_shard(root, shard)
            out = Path(folder) / "report"
            self.assertEqual(pilot.merge_shards(root, out), 0)
            report = pilot.load_json(out / "deterministic-report.json")
            self.assertEqual(report["execution_counts"], {"runtime_timeout": 36})
            self.assertEqual(report["coverage"]["provided"], 36)
            self.assertEqual(report["recorded_semantic_counts"]["agent"], {"not_reviewed": 36})
            self.assertFalse(pilot.load_json(out / "run.json")["quality_accepted"])
            self.assertFalse(report["deterministic_checks_passed"])

    def test_missing_shard_still_records_all36_with_missing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "shards"
            root.mkdir()
            self.write_shard(root, 0)
            out = Path(folder) / "report"
            self.assertEqual(pilot.merge_shards(root, out), 2)
            records = pilot.load_json(out / "responses.json")["records"]
            self.assertEqual(len(records), 36)
            self.assertEqual(sum(record["result"]["status"] == "not_run" for record in records), 27)

    def test_duplicate_case_or_tampered_output_cannot_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "shards"
            root.mkdir()
            self.write_shard(root, 0, lambda response: response["records"].__setitem__(1, response["records"][0]))
            with self.assertRaises(RuntimeError):
                pilot.merge_shards(root, Path(folder) / "invalid-report")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "shards"
            root.mkdir()
            shard = self.write_shard(root, 0)
            with (shard / "responses.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaises(RuntimeError):
                pilot.merge_shards(root, Path(folder) / "tampered-report")

    def test_absent_artifacts_still_report_all36_cases_as_not_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(pilot.merge_shards(root / "missing-shards", root / "report"), 2)
            records = pilot.load_json(root / "report/responses.json")["records"]
            self.assertEqual(len(records), 36)
            self.assertTrue(all(record["result"]["status"] == "not_run" for record in records))
            summary = pilot.load_json(root / "report/run.json")
            self.assertEqual(summary["missing_shards"], [0, 1, 2, 3])
            self.assertEqual(summary["run_identity_from"], "collector_environment_no_shard_evidence")

    def test_mixed_run_retries_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "shards"
            root.mkdir()
            self.write_shard(root, 0)
            self.write_shard(root, 1, lambda response: response.update(run_id="other-run"))
            with self.assertRaises(RuntimeError):
                pilot.merge_shards(root, Path(folder) / "invalid-report")

    def test_local_invocation_cannot_load_model(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            result = subprocess.run([sys.executable, "-B", str(Path(pilot.__file__).resolve()),
                                     "--shard", "0", "--work", "unused", "--output", "unused"],
                                    capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hosted-CI only", result.stderr)


if __name__ == "__main__":
    unittest.main()
