"""Hugging Face publishing (tools/vl/publish_hf.py): each card's Files table is what goes up, from the right folder."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "vl"))
import publish_hf  # noqa: E402


class PublishHF(unittest.TestCase):
    def test_cards_list_weights_checksums_and_license(self):
        for card, _ in publish_hf.MODELS.values():
            names = publish_hf.files_table((ROOT / "docs" / "model-cards" / f"{card}.md").read_text(encoding="utf-8"))
            self.assertEqual(names[-2:], ["SHA256SUMS-gguf", "LICENSE-base.txt"], card)
            self.assertTrue(all(n.endswith(".gguf") for n in names[:-2]), card)
            self.assertTrue(any(n.endswith("-Q4_K_M.gguf") for n in names) and any(n.startswith("mmproj-") for n in names))

    def test_only_the_license_comes_from_an_earlier_run(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(publish_hf, "V6", Path(tmp)):
            (Path(tmp) / "m-fix3").mkdir()
            (Path(tmp) / "m").mkdir()
            for name in ("LICENSE-base.txt", "SHA256SUMS-gguf", "m-Q4_K_M.gguf"):
                (Path(tmp) / "m" / name).write_text("earlier run")
            self.assertEqual(publish_hf.locate("LICENSE-base.txt", ["m-fix3", "m"]), Path(tmp) / "m" / "LICENSE-base.txt")
            for name in ("SHA256SUMS-gguf", "m-Q4_K_M.gguf"):
                with self.assertRaises(FileNotFoundError):
                    publish_hf.locate(name, ["m-fix3", "m"])

    def test_hashes_match_git_and_sha256sum_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"Krummholz\n")
            git = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True)
            self.assertEqual(publish_hf.git_blob(path), git.stdout.strip())
            sums = Path(tmp) / "SHA256SUMS-gguf"
            sums.write_text(f"{publish_hf.sha256(path)}  f\n{'0' * 64} *g\n", encoding="utf-8")
            self.assertEqual(publish_hf.recorded_sums(sums), {"f": [publish_hf.sha256(path)], "g": ["0" * 64]})


if __name__ == "__main__":
    unittest.main()
