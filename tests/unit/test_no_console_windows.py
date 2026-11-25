import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rrr import host_backend
from rrr import product_workspace


WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000


class _FakeProcess:
    pid = 4242
    returncode = 0

    def communicate(self, _prompt=None, timeout=None):
        return "ok", ""

    def poll(self):
        return None


def _alive(pid: int) -> bool:
    if WINDOWS:
        listed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, creationflags=NO_WINDOW,
        ).stdout
        return f'"{pid}"' in listed
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split(") ", 1)[1][:1] != "Z"
    except OSError:
        return True


class CreationFlagTests(unittest.TestCase):
    def _popen_options(self, call):
        with patch.object(subprocess, "Popen", return_value=_FakeProcess()) as popen:
            call()
        return popen.call_args.kwargs

    def _host_call(self):
        host_backend._run_command(
            ["codex", "exec"], prompt="request", cwd=None, env={},
            timeout=5, timeout_message="unused",
        )

    @unittest.skipUnless(WINDOWS, "Windows creation flags")
    def test_host_calls_get_a_process_group_and_no_window(self):
        options = self._popen_options(self._host_call)
        self.assertEqual(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            options["creationflags"],
        )
        self.assertEqual(NO_WINDOW, subprocess.CREATE_NO_WINDOW)
        self.assertNotIn("start_new_session", options)

    @unittest.skipUnless(WINDOWS, "Windows creation flags")
    def test_the_status_check_goes_through_the_same_call(self):
        options = self._popen_options(
            lambda: host_backend._run_status(["codex", "login", "status"], {})
        )
        self.assertTrue(options["creationflags"] & subprocess.CREATE_NO_WINDOW)

    @unittest.skipUnless(WINDOWS, "Windows creation flags")
    def test_taskkill_opens_no_window_either(self):
        with patch.object(subprocess, "run") as run:
            host_backend._terminate_process_tree(_FakeProcess())
        command = run.call_args.args[0]
        self.assertEqual(["taskkill", "/PID", "4242", "/T", "/F"], command)
        self.assertEqual(subprocess.CREATE_NO_WINDOW, run.call_args.kwargs["creationflags"])

    @unittest.skipUnless(WINDOWS, "Windows creation flags")
    def test_the_steps_of_prepare_open_no_window(self):
        with patch.object(subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            product_workspace._run([sys.executable, "-c", "pass"], dict(os.environ), quiet=True)
        self.assertEqual(subprocess.CREATE_NO_WINDOW, run.call_args.kwargs["creationflags"])

    @unittest.skipIf(WINDOWS, "POSIX session handling")
    def test_other_systems_start_a_session_and_pass_no_flag(self):
        options = self._popen_options(self._host_call)
        self.assertTrue(options["start_new_session"])
        self.assertNotIn("creationflags", options)
        self.assertEqual(0, host_backend._no_window_flags())


class TimeoutStillEndsTheTreeTests(unittest.TestCase):
    def test_a_timed_out_call_ends_the_child_and_its_own_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pids = Path(temp_dir) / "pids.txt"
            flags = f", creationflags={NO_WINDOW}" if WINDOWS else ""
            script = (
                "import os, subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']{flags})\n"
                f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
                "time.sleep(120)\n"
            )
            started = time.monotonic()
            with self.assertRaisesRegex(host_backend.HostBackendError, "took too long"):
                host_backend._run_command(
                    [sys.executable, "-c", script], prompt=None, cwd=None,
                    env=dict(os.environ), timeout=4, timeout_message="took too long",
                )
            self.assertLess(time.monotonic() - started, 60)
            parent_pid, child_pid = (int(value) for value in pids.read_text().split())
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and (_alive(parent_pid) or _alive(child_pid)):
                time.sleep(0.25)
            self.assertFalse(_alive(parent_pid), "the timed-out process is still running")
            self.assertFalse(_alive(child_pid), "its child is still running")


if __name__ == "__main__":
    unittest.main()
