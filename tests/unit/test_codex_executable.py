import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rrr import host_backend


def _build(root: Path, name: str, mtime: float, with_exe: bool = True) -> Path:
    folder = root / "OpenAI" / "Codex" / "bin" / name
    folder.mkdir(parents=True)
    exe = folder / "codex.exe"
    if with_exe:
        exe.write_bytes(b"")
        os.utime(exe, (mtime, mtime))
    return exe


class CodexApplicationFolderTests(unittest.TestCase):
    def test_the_newest_build_comes_first_and_empty_folders_are_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = _build(root, "8dbccda2214d0643", 1_700_000_000)
            new = _build(root, "5449c1f0f26f1f98", 1_800_000_000)
            _build(root, "0000000000000000", 0, with_exe=False)
            found = host_backend._codex_app_executables(str(root))
        self.assertEqual([str(new), str(old)], found)

    def test_no_folder_gives_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual([], host_backend._codex_app_executables(temp_dir))
        self.assertEqual([], host_backend._codex_app_executables(""))

    @unittest.skipUnless(os.name == "nt", "the Codex desktop application paths are Windows paths")
    def test_it_is_a_candidate_after_the_two_helpers_and_before_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            home = root / "home"
            app = _build(root / "local", "5449c1f0f26f1f98", 1_800_000_000)
            _build(root / "local", "8dbccda2214d0643", 1_700_000_000)
            on_path = root / "bin" / "codex.exe"
            on_path.parent.mkdir()
            on_path.write_bytes(b"")
            environment = {"LOCALAPPDATA": str(root / "local")}
            with patch.dict(os.environ, environment), \
                    patch.object(host_backend.Path, "home", return_value=home), \
                    patch.object(host_backend.shutil, "which", return_value=str(on_path)):
                os.environ.pop("RRR_HOST_COMMAND", None)
                self.assertEqual([str(app), str(on_path)], host_backend._command_candidates("codex"))
                helper = home / ".codex" / ".sandbox-bin" / "codex.exe"
                helper.parent.mkdir(parents=True)
                helper.write_bytes(b"")
                self.assertEqual(
                    [str(helper), str(app), str(on_path)],
                    host_backend._command_candidates("codex"),
                )
                self.assertEqual([str(on_path)], host_backend._command_candidates("claude"))
                with patch.dict(os.environ, {"RRR_HOST_COMMAND": "C:/custom/codex.exe"}):
                    self.assertEqual(
                        ["C:/custom/codex.exe"], host_backend._command_candidates("codex")
                    )


if __name__ == "__main__":
    unittest.main()
