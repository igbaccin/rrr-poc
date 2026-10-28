import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rrr import manifest


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout


class GitProvenanceTests(unittest.TestCase):
    def test_diff_hash_covers_bytes_the_locale_cannot_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _git(temp_dir, "init", "-q")
            note = Path(temp_dir) / "note.tex"
            note.write_bytes(b"plain\n")
            _git(temp_dir, "add", "note.tex")
            _git(
                temp_dir,
                "-c", "user.name=RRR test",
                "-c", "user.email=rrr-test@example.invalid",
                "commit", "-q", "-m", "fixture",
            )
            # Curly quotes, then a byte that neither UTF-8 nor cp1252 decodes.
            note.write_bytes("“quoted”\n".encode("utf-8") + b"\x9d\n")
            expected = hashlib.sha256(
                _git(temp_dir, "diff", "--no-ext-diff", "--binary")
            ).hexdigest()

            with mock.patch.object(manifest, "repo_path", return_value=Path(temp_dir)):
                info = manifest._git_info()

        self.assertTrue(info["dirty"])
        self.assertEqual(info["status_short"], [" M note.tex"])
        self.assertEqual(info["tracked_diff_sha256"], expected)
        self.assertNotEqual(info["tracked_diff_sha256"], hashlib.sha256(b"").hexdigest())

    def test_unavailable_git_is_recorded_as_unknown(self):
        failed = subprocess.CompletedProcess(["git"], 128, stdout=b"", stderr=b"fatal")
        with mock.patch.object(manifest.subprocess, "run", return_value=failed):
            info = manifest._git_info()

        self.assertIsNone(info["commit"])
        self.assertIsNone(info["branch"])
        self.assertIsNone(info["dirty"])
        self.assertEqual(info["status_short"], [])
        self.assertIsNone(info["tracked_diff_sha256"])
        self.assertIsNone(info["untracked_list_sha256"])


if __name__ == "__main__":
    unittest.main()
