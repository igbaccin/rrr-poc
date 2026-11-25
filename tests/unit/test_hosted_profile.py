import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_product_distribution.py"
OVERLAY = ROOT / "profiles" / "hosted" / "rrr"
HOSTED_TESTS = ROOT / "tests" / "hosted"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("rrr_build_distribution", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HostedProfileTests(unittest.TestCase):
    def test_overlay_replaces_only_existing_modules(self):
        build = _load_build_module()
        modules = build.hosted_overlay_modules()
        self.assertIn("writer.py", modules)
        for name in modules:
            self.assertTrue(
                (ROOT / "src" / "rrr" / name).is_file(),
                f"profiles/hosted/rrr/{name} replaces no module in src/rrr",
            )

    def test_assembled_tree_is_src_with_the_overlay_on_top(self):
        build = _load_build_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            package = build.assemble_hosted_source(Path(temp_dir))
            overlay = set(build.hosted_overlay_modules())
            source_modules = sorted(
                path.name for path in (ROOT / "src" / "rrr").glob("*.py")
            )
            self.assertEqual(
                source_modules, sorted(path.name for path in package.glob("*.py"))
            )
            for name in source_modules:
                expected = (OVERLAY if name in overlay else ROOT / "src" / "rrr") / name
                self.assertEqual(
                    expected.read_bytes(), (package / name).read_bytes(), name
                )

    def test_hosted_version_replaces_the_local_version(self):
        build = _load_build_module()
        pyproject = build.hosted_pyproject("15.19.4")
        self.assertEqual("15.19.4", build.project_field("version", pyproject))
        self.assertEqual("rrr-poc", build.project_field("name", pyproject))

    def test_hosted_test_suite_passes_on_the_assembled_tree(self):
        build = _load_build_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            package = build.assemble_hosted_source(Path(temp_dir))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(package.parent)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            for name in ("RRR_RUNTIME", "RRR_HOST", "RRR_API_PROVIDER", "RRR_API_MODEL"):
                env.pop(name, None)
            options = {}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(HOSTED_TESTS),
                    "-p",
                    "test_*.py",
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **options,
            )
        self.assertEqual(
            0,
            completed.returncode,
            "hosted tests failed:\n" + completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
