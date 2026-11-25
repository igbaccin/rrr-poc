import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _pdf_fixtures import make_pdf


ROOT = Path(__file__).resolve().parents[2]


def _run(arguments, *, binary=False, extra_env=None):
    env = os.environ.copy()
    for name in ("PYTHONIOENCODING", "PYTHONUTF8", "RRR_RUNTIME", "RRR_HOST"):
        env.pop(name, None)
    env.update(extra_env or {})
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    if not binary:
        options.update({"text": True, "encoding": "utf-8", "errors": "replace"})
    return subprocess.run(
        [sys.executable, *arguments], env=env, capture_output=True, **options,
    )


class HelpTests(unittest.TestCase):
    def test_help_lists_every_command(self):
        completed = _run(["-m", "rrr", "--help"])
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("usage: rrr <command> [options]", completed.stdout)
        for command in ("prepare", "t2", "t1", "ingest", "host-doctor"):
            self.assertRegex(completed.stdout, rf"(?m)^  {command}\s+\S")
        self.assertIn("python -m rrr <command>", completed.stdout)
        self.assertIn("--version", completed.stdout)

    def test_no_command_shows_the_help_and_fails(self):
        completed = _run(["-m", "rrr"])
        self.assertEqual(2, completed.returncode)
        self.assertEqual("", completed.stdout)
        self.assertIn("commands:", completed.stderr)
        self.assertIn("host-doctor", completed.stderr)

    def test_version(self):
        completed = _run(["-m", "rrr", "--version"])
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertRegex(completed.stdout.strip(), r"^rrr \S+")

    def test_an_unknown_command_names_the_task_choices(self):
        completed = _run(["-m", "rrr", "review", "--metadata", "m.csv", "--topic", "x"])
        self.assertEqual(2, completed.returncode)
        self.assertIn("usage: rrr <command> [options]", completed.stderr)
        self.assertIn("invalid choice: 'review'", completed.stderr)

    def test_main_takes_its_arguments_as_a_list(self):
        from rrr.cli import main

        with self.assertRaises(SystemExit) as stop:
            main(["--version"])
        self.assertEqual(0, stop.exception.code)


class Utf8OutputTests(unittest.TestCase):
    TEXT = "Pörtner — → ß"

    def test_the_helper_makes_both_streams_write_utf8(self):
        script = (
            "import sys; from rrr.utils import force_utf8_stdio; force_utf8_stdio(); "
            f"sys.stdout.write({self.TEXT!r}); sys.stderr.write({self.TEXT!r})"
        )
        completed = _run(["-c", script], binary=True)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(self.TEXT.encode("utf-8"), completed.stdout)
        self.assertEqual(self.TEXT.encode("utf-8"), completed.stderr)

    def test_the_preprocess_worker_writes_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = root / "metadata.csv"
            metadata.write_text(
                "doc_id,pdf_path\nPörtner_2003,missing.pdf\n", encoding="utf-8",
            )
            completed = _run(
                ["-m", "rrr.preprocess", "--metadata", str(metadata), "--workers", "1"],
                binary=True, extra_env={"RRR_PROJECT_ROOT": str(root)},
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("[skip] Pörtner_2003: missing_pdf", completed.stdout.decode("utf-8"))

    def test_every_entry_point_calls_the_helper(self):
        for module in ("cli.py", "preprocess.py", "index.py"):
            source = (ROOT / "src" / "rrr" / module).read_text(encoding="utf-8")
            self.assertIn("force_utf8_stdio()", source, module)

    def test_prepare_reads_its_steps_as_utf8(self):
        source = (ROOT / "src" / "rrr" / "product_workspace.py").read_text(encoding="utf-8")
        self.assertIn('env["PYTHONIOENCODING"] = "utf-8"', source)
        self.assertIn('encoding="utf-8" if quiet else None', source)

    def test_the_ingest_progress_line_is_plain_ascii(self):
        source = (ROOT / "src" / "rrr" / "ingest.py").read_text(encoding="utf-8")
        line = next(line for line in source.splitlines() if "doc_id={meta.doc_id!r" in line)
        line.encode("ascii")


class PymupdfImportTests(unittest.TestCase):
    def test_no_module_imports_the_deprecated_name(self):
        for folder in (ROOT / "src" / "rrr", ROOT / "profiles" / "hosted" / "rrr", ROOT / "scripts"):
            for path in folder.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("import fitz", source, path)
                self.assertNotIn("fitz.open", source, path)

    def test_reading_the_first_pages_raises_no_deprecation_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = make_pdf(Path(temp_dir) / "paper.pdf", "A Synthetic Title", "A. N. Other", "1999")
            script = (
                "import sys; from pathlib import Path; from rrr.ingest import extract_pages_text; "
                "text = extract_pages_text(Path(sys.argv[1])); "
                "assert 'A Synthetic Title' in text, text[:80]; print('ok')"
            )
            completed = _run(["-W", "error::FutureWarning", "-c", script, str(pdf)])
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("ok", completed.stdout.strip())
        self.assertNotIn("FutureWarning", completed.stderr)


if __name__ == "__main__":
    unittest.main()
