"""Small synthetic bytes exercise the same streaming path as a large model."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import model_multipart as multipart


class MultipartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"GGUF" + bytes(range(256)) * 23
        self.source = self.root / "fictional-model.gguf"
        self.source.write_bytes(self.payload)
        self.pin = hashlib.sha256(self.payload).hexdigest()

    def package(self, part_bytes=1000):
        result = multipart.split(self.source, self.root / "parts", len(self.payload), self.pin, part_bytes)
        manifest = Path(result["manifest"])
        return result, manifest, json.loads(manifest.read_text())

    def rewrite_manifest(self, path, value):
        data = multipart.serialized(value)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def output(self):
        parent = self.root / "assembled"
        parent.mkdir(exist_ok=True)
        return parent / self.source.name

    def test_exact_streaming_round_trip_and_both_hash_levels(self):
        receipt, path, manifest = self.package()
        self.assertEqual([part["bytes"] for part in manifest["parts"]], [1000] * 5 + [892])
        self.assertEqual([part["offset"] for part in manifest["parts"]], [0, 1000, 2000, 3000, 4000, 5000])
        self.assertTrue(multipart.verify(path, receipt["manifest_sha256"])["verified"])
        target = self.output()
        self.assertTrue(multipart.assemble(path, target, receipt["manifest_sha256"])["verified"])
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.assertFalse(list(target.parent.glob("*.partial")))
        self.assertFalse(manifest["release_authorized"])
        self.assertFalse(manifest["quality_accepted"])

    def test_exact_part_boundary_has_no_empty_final_part(self):
        receipt, _, manifest = self.package(len(self.payload) // 4)
        self.assertEqual(receipt["parts"], 4)
        self.assertEqual(sum(row["bytes"] for row in manifest["parts"]), len(self.payload))
        self.assertTrue(all(row["bytes"] > 0 for row in manifest["parts"]))

    def test_bad_source_pin_never_creates_completed_manifest(self):
        with self.assertRaises(ValueError):
            multipart.split(self.source, self.root / "wrong-size", len(self.payload) + 1, self.pin)
        self.assertFalse((self.root / "wrong-size").exists())
        with self.assertRaises(ValueError):
            multipart.split(self.source, self.root / "wrong-hash", len(self.payload), "0" * 64, 1000)
        self.assertFalse((self.root / "wrong-hash" / multipart.MANIFEST_NAME).exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_existing_package_directory_is_never_reused_or_deleted(self):
        folder = self.root / "existing"
        folder.mkdir()
        unrelated = folder / "keep.txt"
        unrelated.write_text("retain")
        with self.assertRaises(FileExistsError):
            multipart.split(self.source, folder, len(self.payload), self.pin)
        self.assertEqual(unrelated.read_text(), "retain")

    def test_wrong_manifest_hash_rejected_before_creating_output(self):
        _, path, _ = self.package()
        target = self.output()
        with self.assertRaises(ValueError):
            multipart.assemble(path, target, "0" * 64)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_missing_truncated_modified_and_extended_parts_fail_without_output(self):
        receipt, path, manifest = self.package()
        part = path.parent / manifest["parts"][1]["name"]
        original = part.read_bytes()
        target = self.output()
        for invalid in (None, original[:-1], b"x" + original[1:], original + b"x"):
            if invalid is None:
                part.unlink()
            else:
                part.write_bytes(invalid)
            with self.subTest(kind=None if invalid is None else len(invalid)), self.assertRaises((ValueError, OSError)):
                multipart.assemble(path, target, receipt["manifest_sha256"])
            self.assertFalse(target.exists())
            self.assertFalse(list(target.parent.glob(".krummholz-reassembly-*")))
            part.write_bytes(original)
        self.assertTrue(multipart.assemble(path, target, receipt["manifest_sha256"])["verified"])

    def test_part_hashes_alone_cannot_replace_whole_file_pin(self):
        _, path, manifest = self.package()
        manifest["file"]["sha256"] = "0" * 64
        manifest_hash = self.rewrite_manifest(path, manifest)
        target = self.output()
        with self.assertRaisesRegex(ValueError, "Whole-file"):
            multipart.assemble(path, target, manifest_hash)
        self.assertFalse(target.exists())

    def test_changed_order_offsets_counts_names_and_approval_are_rejected(self):
        _, _, original = self.package()
        variants = []
        altered = copy.deepcopy(original); altered["parts"].reverse(); variants.append(altered)
        altered = copy.deepcopy(original); altered["parts"][1]["offset"] += 1; variants.append(altered)
        altered = copy.deepcopy(original); altered["parts"].pop(); variants.append(altered)
        altered = copy.deepcopy(original); altered["parts"][0]["name"] = "../elsewhere"; variants.append(altered)
        altered = copy.deepcopy(original); altered["parts"][0]["name"] = "C:\\elsewhere"; variants.append(altered)
        altered = copy.deepcopy(original); altered["parts"][0]["index"] = True; variants.append(altered)
        altered = copy.deepcopy(original); altered["release_authorized"] = True; variants.append(altered)
        altered = copy.deepcopy(original); altered["schema_version"] = True; variants.append(altered)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                multipart.validate_manifest(value)

    def test_duplicate_json_keys_and_oversized_manifests_are_rejected(self):
        file = self.root / "duplicate.json"
        data = b'{"schema_version":1,"schema_version":1}'
        file.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            multipart.load_manifest(file, hashlib.sha256(data).hexdigest())
        file.write_bytes(b" " * (multipart.MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "Oversized"):
            multipart.load_manifest(file, "0" * 64)

    def test_existing_whole_file_is_never_overwritten(self):
        receipt, path, _ = self.package()
        target = self.output()
        target.write_bytes(b"unrelated existing file")
        with self.assertRaises(FileExistsError):
            multipart.assemble(path, target, receipt["manifest_sha256"])
        self.assertEqual(target.read_bytes(), b"unrelated existing file")

    def test_concurrent_destination_creation_does_not_overwrite_it(self):
        receipt, path, _ = self.package()
        target = self.output()
        original_link = multipart.os.link
        def racing_link(source, destination):
            target.write_bytes(b"new unrelated file")
            return original_link(source, destination)
        with patch.object(multipart.os, "link", side_effect=racing_link), self.assertRaises(FileExistsError):
            multipart.assemble(path, target, receipt["manifest_sha256"])
        self.assertEqual(target.read_bytes(), b"new unrelated file")
        self.assertFalse(list(target.parent.glob(".krummholz-reassembly-*")))

    def test_unsupported_atomic_publish_fails_closed_and_cleans_owned_temp(self):
        receipt, path, _ = self.package()
        target = self.output()
        unrelated = target.parent / "unrelated.partial"
        unrelated.write_text("keep")
        with patch.object(multipart.os, "link", side_effect=OSError("hard links unsupported")), self.assertRaises(OSError):
            multipart.assemble(path, target, receipt["manifest_sha256"])
        self.assertFalse(target.exists())
        self.assertEqual(unrelated.read_text(), "keep")
        self.assertFalse(list(target.parent.glob(".krummholz-reassembly-*")))

    def test_no_space_refuses_before_creating_owned_output(self):
        usage = type("Usage", (), {"free": 0})()
        with patch.object(multipart.shutil, "disk_usage", return_value=usage), self.assertRaises(OSError):
            multipart.split(self.source, self.root / "no-space", len(self.payload), self.pin)
        self.assertFalse((self.root / "no-space").exists())

    def test_windows_special_names_path_separators_and_asset_limit(self):
        for name in ("../x", "x/y", "x\\y", "CON.gguf", "NUL", "LPT1.gguf", "a:", "a."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                multipart.filename(name)
        for size in (0, -1, True, 2 * 1024**3):
            with self.subTest(size=size), self.assertRaises(ValueError):
                multipart.split(self.source, self.root / "bad", len(self.payload), self.pin, size)

    def test_six_gigabyte_manifest_has_four_parts_and_64bit_offsets(self):
        size, part_size = 6158339904, multipart.DEFAULT_PART_BYTES
        parts = [{"index": i, "name": multipart.part_name("SmolLM3-3B-BF16.gguf", i, 4),
                  "offset": (i - 1) * part_size, "bytes": min(part_size, size - (i - 1) * part_size),
                  "sha256": "0" * 64} for i in range(1, 5)]
        value = {"schema_version": 1, "format": "raw-concatenation-v1",
                 "file": {"name": "SmolLM3-3B-BF16.gguf", "bytes": size, "sha256": "0" * 64},
                 "parts": parts, "part_bytes": part_size, "provenance": {},
                 "release_authorized": False, "quality_accepted": False}
        self.assertEqual(multipart.validate_manifest(value), value)
        self.assertEqual(parts[-1]["offset"], 4831838208)
        self.assertEqual(parts[-1]["bytes"], 1326501696)
        self.assertTrue(all(part["bytes"] < 2 * 1024**3 for part in parts))

    def test_payload_reads_are_bounded_even_when_part_is_larger(self):
        original_open = multipart.open_regular
        reads = []
        class Monitor:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, size):
                reads.append(size)
                if size < 0 or size > 127: raise AssertionError("unbounded payload read")
                return self.stream.read(size)
        with patch.object(multipart, "BUFFER", 127), patch.object(multipart, "open_regular", side_effect=lambda p: Monitor(original_open(p))):
            receipt, _, _ = self.package(1000)
        self.assertGreater(len(reads), receipt["parts"])
        self.assertLessEqual(max(reads), 127)

    def test_symlink_part_is_refused_without_reading_target(self):
        receipt, path, manifest = self.package()
        part = path.parent / manifest["parts"][0]["name"]
        outside = self.root / "outside"
        outside.write_bytes(part.read_bytes())
        part.unlink()
        try:
            part.symlink_to(outside)
        except OSError:
            self.skipTest("Local Windows policy does not permit test symlink creation")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            multipart.assemble(path, self.output(), receipt["manifest_sha256"])
        self.assertTrue(outside.exists())

    def test_detected_symlink_path_refused_on_all_platforms(self):
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaisesRegex(ValueError, "Symlink"):
            multipart.open_regular(self.source)

    def test_interrupted_copy_removes_only_owned_temp_and_never_publishes(self):
        receipt, path, _ = self.package()
        target = self.output()
        unrelated = target.parent / "keep.partial"
        unrelated.write_text("keep")
        def interrupted(manifest, directory, output):
            output.write(b"partial model bytes")
            raise KeyboardInterrupt()
        with patch.object(multipart, "copy_verified_parts", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            multipart.assemble(path, target, receipt["manifest_sha256"])
        self.assertFalse(target.exists())
        self.assertEqual(unrelated.read_text(), "keep")
        self.assertFalse(list(target.parent.glob(".krummholz-reassembly-*")))

    def test_cli_split_verify_and_assemble_use_the_same_checked_path(self):
        script = str(Path(multipart.__file__).resolve())
        def run(*args):
            result = subprocess.run([sys.executable, "-B", script, *map(str, args)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        receipt = run("split", "--source", self.source, "--output-dir", self.root / "cli-parts",
                      "--expected-bytes", len(self.payload), "--expected-sha256", self.pin, "--part-bytes", 1000)
        pin = receipt["manifest_sha256"]
        self.assertTrue(run("verify", "--manifest", receipt["manifest"], "--manifest-sha256", pin)["verified"])
        target = self.output()
        run("assemble", "--manifest", receipt["manifest"], "--manifest-sha256", pin, "--output", target)
        self.assertEqual(target.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
