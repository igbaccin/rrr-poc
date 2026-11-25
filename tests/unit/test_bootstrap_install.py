import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = ROOT / "plugins" / "rrr"
BOOTSTRAP = PLUGIN_ROOT / "scripts" / "bootstrap_rrr.py"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Run:
    def __init__(self, fail_at=None, exit_code=1):
        self.commands = []
        self.options = []
        self.fail_at = fail_at
        self.exit_code = exit_code

    def __call__(self, command, **options):
        self.commands.append(list(command))
        self.options.append(options)
        if self.fail_at is not None and len(self.commands) - 1 == self.fail_at:
            raise subprocess.CalledProcessError(self.exit_code, command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


class BootstrapInstallTests(unittest.TestCase):
    def setUp(self):
        self.module = _load(f"rrr_bootstrap_{self.id().rsplit('.', 1)[-1]}")
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "runtime"
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        for name in ("RRR_WHEELHOUSE", "PIP_DEFAULT_TIMEOUT", "PIP_RETRIES"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def _main(self, *arguments, run=None):
        run = run or _Run()
        out, err = io.StringIO(), io.StringIO()
        argv = ["bootstrap_rrr.py", "--plugin-root", str(PLUGIN_ROOT),
                "--install-base", str(self.base), *arguments]
        code = 0
        with patch.object(self.module.venv.EnvBuilder, "create"), \
                patch.object(self.module.subprocess, "run", run), \
                patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                self.module.main()
            except SystemExit as stop:
                code = stop.code
        return code, out.getvalue(), err.getvalue(), run

    def test_progress_goes_to_stderr_and_stdout_is_one_json_object(self):
        code, out, err, run = self._main("--json")
        self.assertEqual(0, code)
        result = json.loads(out)
        self.assertEqual(1, len(out.strip().splitlines()))
        self.assertFalse(result["reused"])
        self.assertEqual([result["python"], "-m", "rrr"], result["command"])
        for line in (
            "[RRR] creating a private Python environment in",
            "[RRR] downloading and installing RRR's dependencies",
            "[RRR] installing the bundled RRR",
            "[RRR] checking that the runtime starts",
            "is installed. Run it with:",
        ):
            self.assertIn(line, err)
        self.assertEqual(3, len(run.commands))

    def test_pip_runs_with_an_explicit_timeout_and_retries(self):
        _, _, _, run = self._main("--json")
        for command in run.commands[:2]:
            self.assertEqual(["-m", "pip", "install"], command[1:4])
            self.assertEqual("30", command[command.index("--timeout") + 1])
            self.assertEqual("3", command[command.index("--retries") + 1])
            self.assertNotIn("--no-index", command)
        self.assertIn("--upgrade", run.commands[0])
        self.assertIn("--force-reinstall", run.commands[1])

    def test_the_users_own_pip_settings_are_left_alone(self):
        os.environ["PIP_DEFAULT_TIMEOUT"] = "120"
        os.environ["PIP_RETRIES"] = "8"
        _, _, _, run = self._main("--json")
        self.assertNotIn("--timeout", run.commands[0])
        self.assertNotIn("--retries", run.commands[0])

    def test_a_wheelhouse_installs_without_the_index(self):
        wheelhouse = Path(self.temp.name) / "wheels"
        wheelhouse.mkdir()
        for arguments, environment in (
            (("--wheelhouse", str(wheelhouse)), {}),
            ((), {"RRR_WHEELHOUSE": str(wheelhouse)}),
        ):
            os.environ.pop("RRR_WHEELHOUSE", None)
            os.environ.update(environment)
            self.base = Path(self.temp.name) / f"runtime{len(environment)}"
            code, out, err, run = self._main("--json", *arguments)
            self.assertEqual(0, code, err)
            for command in run.commands[:2]:
                at = command.index("--no-index")
                self.assertEqual(
                    ["--no-index", "--find-links", str(wheelhouse.resolve())],
                    command[at:at + 3],
                )
            self.assertIn("from the wheelhouse", err)
            self.assertIn("(no network)", err)
            json.loads(out)

    def test_a_missing_wheelhouse_folder_is_reported(self):
        code, out, err, run = self._main("--json", "--wheelhouse", str(Path(self.temp.name) / "none"))
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("The wheelhouse folder does not exist", err)
        self.assertEqual([], run.commands)

    def test_a_failed_download_names_the_cause_and_the_remedies(self):
        code, out, err, _ = self._main("--json", run=_Run(fail_at=0))
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("[RRR] installation failed: pip could not download", err)
        for remedy in ("pypi.org", "again in a few minutes", "PIP_INDEX_URL",
                       "HTTPS_PROXY", "--wheelhouse", "RRR_WHEELHOUSE"):
            self.assertIn(remedy, err)
        self.assertEqual([], list(self.base.glob("*/installed.json")))

    def test_a_failed_wheelhouse_install_says_what_the_folder_lacks(self):
        wheelhouse = Path(self.temp.name) / "wheels"
        wheelhouse.mkdir()
        code, _, err, _ = self._main("--json", "--wheelhouse", str(wheelhouse), run=_Run(fail_at=0))
        self.assertEqual(2, code)
        self.assertIn("pip could not install RRR from the wheelhouse", err)
        self.assertIn("python -m pip download --dest", err)
        self.assertNotIn("PIP_INDEX_URL", err)

    def test_a_second_run_reuses_the_runtime_and_says_so(self):
        first, _, _, _ = self._main("--json")
        self.assertEqual(0, first)
        version_dir = next(self.base.iterdir())
        _, python_exe, _ = self.module._runtime_paths(version_dir)
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.write_bytes(b"")
        code, out, err, run = self._main("--json")
        self.assertEqual(0, code)
        self.assertTrue(json.loads(out)["reused"])
        self.assertIn("was already installed", err)
        self.assertEqual(1, len(run.commands))
        self.assertEqual(["-c", "import rrr.cli"], run.commands[0][1:])

    def test_both_copies_are_identical(self):
        self.assertEqual(
            BOOTSTRAP.read_bytes(),
            (ROOT / "skills" / "rrr" / "scripts" / "bootstrap_rrr.py").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
