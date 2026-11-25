import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAPS = (
    ROOT / "plugins" / "rrr" / "scripts" / "bootstrap_rrr.py",
    ROOT / "skills" / "rrr" / "scripts" / "bootstrap_rrr.py",
)
SKILLS = (
    ROOT / ".agents" / "skills" / "rrr" / "SKILL.md",
    ROOT / "plugins" / "rrr" / "skills" / "rrr" / "SKILL.md",
    ROOT / "skills" / "rrr" / "SKILL.md",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_m_rrr(*args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for name in ("RRR_RUNTIME", "RRR_HOST"):
        env.pop(name, None)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.run(
        [sys.executable, "-m", "rrr", *args],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", **options,
    )


class ModuleEntryPointTests(unittest.TestCase):
    def test_python_m_rrr_reaches_every_command(self):
        for command in ("prepare", "ingest", "host-doctor"):
            completed = _python_m_rrr(command, "--help")
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn(f"usage: rrr {command}", completed.stdout)

    def test_the_task_help_names_the_program_rrr(self):
        completed = _python_m_rrr("--help")
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("usage: rrr", completed.stdout)
        self.assertNotIn("cli.py", completed.stdout)
        self.assertNotIn("__main__.py", completed.stdout)

    def test_the_package_does_not_start_its_own_launcher(self):
        source = (ROOT / "src" / "rrr" / "product_workspace.py").read_text(encoding="utf-8")
        self.assertEqual(3, source.count('sys.executable, "-m", "rrr.')
                         + source.count('sys.executable,\n        "-m",\n        "rrr.'))
        for path in (ROOT / "src" / "rrr").glob("*.py"):
            self.assertNotIn("rrr.exe", path.read_text(encoding="utf-8"), path.name)


class BootstrapCommandTests(unittest.TestCase):
    def test_both_bootstrap_copies_are_the_same_file(self):
        self.assertEqual(BOOTSTRAPS[0].read_bytes(), BOOTSTRAPS[1].read_bytes())

    def _bootstrap(self, module, base, **patches):
        with patch.object(module.venv.EnvBuilder, "create"), \
                patch.object(module.subprocess, "run") as run:
            run.return_value.returncode = 0
            for name, value in patches.items():
                setattr(module, name, value)
            return module.bootstrap(ROOT / "plugins" / "rrr", install_base=base), run

    def test_the_result_carries_a_command_that_starts_the_interpreter(self):
        module = _load(BOOTSTRAPS[0], "rrr_bootstrap_command")
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = self._bootstrap(module, Path(temp_dir))
        self.assertEqual([result["python"], "-m", "rrr"], result["command"])
        self.assertTrue(result["rrr"])
        self.assertEqual(result, json.loads(json.dumps(result)))

    def test_an_installed_runtime_is_reused_without_the_launcher_file(self):
        module = _load(BOOTSTRAPS[0], "rrr_bootstrap_reuse")
        bundle = json.loads(
            (ROOT / "plugins" / "rrr" / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / str(bundle["version"])
            _, python_exe, rrr_exe = module._runtime_paths(root)
            python_exe.parent.mkdir(parents=True)
            python_exe.write_bytes(b"")
            (root / "installed.json").write_text(
                json.dumps({
                    "bootstrap_schema": module.BOOTSTRAP_SCHEMA,
                    "sha256": bundle["sha256"],
                    "extras": [],
                }),
                encoding="utf-8",
            )
            self.assertFalse(rrr_exe.exists())
            with patch.object(module, "runtime_starts", return_value=True) as starts:
                reused, run = self._bootstrap(module, base)
            starts.assert_called_once_with(python_exe)
            self.assertTrue(reused["reused"])
            self.assertEqual([], run.call_args_list)

            with patch.object(module, "runtime_starts", return_value=False):
                with self.assertRaises(RuntimeError):
                    self._bootstrap(module, base)

    def test_the_start_check_runs_the_interpreter_not_the_launcher(self):
        module = _load(BOOTSTRAPS[0], "rrr_bootstrap_check")
        with patch.object(module.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertTrue(module.runtime_starts(Path("venv") / "python"))
            command = run.call_args.args[0]
            run.return_value.returncode = 1
            self.assertFalse(module.runtime_starts(Path("venv") / "python"))
            run.side_effect = OSError("blocked by group policy")
            self.assertFalse(module.runtime_starts(Path("venv") / "python"))
        self.assertEqual([str(Path("venv") / "python"), "-c", "import rrr.cli"], command)


class SkillTextTests(unittest.TestCase):
    def test_the_skill_tells_the_host_to_use_the_interpreter(self):
        for path in SKILLS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("`command` array", text, path)
            self.assertIn("`-m rrr`", text, path)
            self.assertIn("blocked by group policy", text, path)
            self.assertNotIn("use the `rrr` executable", text, path)


if __name__ == "__main__":
    unittest.main()
