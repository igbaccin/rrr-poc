import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
REPO_SKILL = ROOT / ".agents" / "skills" / "rrr" / "SKILL.md"
CLAUDE_SKILL = ROOT / "skills" / "rrr" / "SKILL.md"
PLUGIN_ROOT = ROOT / "plugins" / "rrr"
PLUGIN_SKILL = PLUGIN_ROOT / "skills" / "rrr" / "SKILL.md"


class SkillPackagingTests(unittest.TestCase):
    def test_skill_copies_are_identical(self):
        canonical = REPO_SKILL.read_bytes()
        self.assertEqual(CLAUDE_SKILL.read_bytes(), canonical)
        self.assertEqual(PLUGIN_SKILL.read_bytes(), canonical)

    def test_codex_metadata_requires_explicit_rrr_invocation(self):
        metadata = (
            ROOT
            / ".agents"
            / "skills"
            / "rrr"
            / "agents"
            / "openai.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn('default_prompt: "Use $rrr t2 ', metadata)
        self.assertIn("allow_implicit_invocation: false", metadata)

    def test_skill_defaults_to_the_native_host(self):
        skill = REPO_SKILL.read_text(encoding="utf-8")
        self.assertIn("An explicit `$rrr` invocation in Codex uses native Codex", skill)
        self.assertIn("An explicit `/rrr` invocation in\nClaude Code uses native Claude", skill)
        self.assertIn("direct Python or `rrr`\nCLI invocation", skill)
        self.assertNotIn("When none is stated, use local\nOllama", skill)

    def test_skill_accepts_compact_invocations_and_infers_workspace(self):
        skill = REPO_SKILL.read_text(encoding="utf-8")
        self.assertIn("`$rrr t2 <topic>`", skill)
        self.assertIn("`$rrr t1 <claim>`", skill)
        self.assertIn("`$rrr <topic>`", skill)
        self.assertIn("use that project as the workspace", skill)
        self.assertIn("Do not require the user", skill)
        self.assertIn('rrr prepare "<selected-folder>" --json', skill)

    def test_plugin_manifest_exposes_the_rrr_skill(self):
        manifest = json.loads(
            (PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["name"], "rrr")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertTrue((PLUGIN_ROOT / manifest["skills"]).is_dir())
        prompts = manifest["interface"]["defaultPrompt"]
        self.assertTrue(any("$rrr" in prompt for prompt in prompts))
        self.assertTrue(any("my PDFs" in prompt for prompt in prompts))

    def test_plugin_bundles_the_verified_python_runtime(self):
        runtime = json.loads(
            (PLUGIN_ROOT / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        wheel = PLUGIN_ROOT / "runtime" / runtime["wheel"]
        self.assertTrue(wheel.is_file())
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), runtime["sha256"])
        self.assertTrue((PLUGIN_ROOT / "scripts" / "bootstrap_rrr.py").is_file())
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        self.assertIn("rrr/product_workspace.py", names)
        self.assertIn("rrr/cli.py", names)

    def test_portable_claude_skill_bundles_the_same_runtime(self):
        plugin_runtime = json.loads(
            (PLUGIN_ROOT / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        claude_root = CLAUDE_SKILL.parent
        claude_runtime = json.loads(
            (claude_root / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        self.assertEqual(claude_runtime, plugin_runtime)
        self.assertEqual(
            (claude_root / "runtime" / claude_runtime["wheel"]).read_bytes(),
            (PLUGIN_ROOT / "runtime" / plugin_runtime["wheel"]).read_bytes(),
        )
        self.assertEqual(
            (claude_root / "scripts" / "bootstrap_rrr.py").read_bytes(),
            (PLUGIN_ROOT / "scripts" / "bootstrap_rrr.py").read_bytes(),
        )

    def test_website_wheel_matches_the_plugin_bundle(self):
        runtime = json.loads(
            (PLUGIN_ROOT / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        plugin_wheel = PLUGIN_ROOT / "runtime" / runtime["wheel"]
        website_wheel = ROOT / "dist" / runtime["wheel"]
        self.assertEqual(website_wheel.read_bytes(), plugin_wheel.read_bytes())
        checksum = (ROOT / "dist" / f"{runtime['wheel']}.sha256").read_text(
            encoding="ascii"
        )
        self.assertEqual(checksum, f"{runtime['sha256']}  {runtime['wheel']}\n")

    def test_bootstrap_reinstalls_a_changed_wheel_with_the_same_version(self):
        bootstrap_path = PLUGIN_ROOT / "scripts" / "bootstrap_rrr.py"
        spec = importlib.util.spec_from_file_location("rrr_plugin_bootstrap", bootstrap_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp_dir:
            install_base = Path(temp_dir)
            with (
                patch.object(module.venv.EnvBuilder, "create"),
                patch.object(module.subprocess, "run") as run,
            ):
                result = module.bootstrap(PLUGIN_ROOT, install_base=install_base)
        self.assertFalse(result["reused"])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("--force-reinstall", commands[1])
        self.assertIn("--no-deps", commands[1])

    def test_repo_marketplace_points_to_the_plugin(self):
        marketplace = json.loads(
            (
                ROOT / ".agents" / "plugins" / "marketplace.json"
            ).read_text(encoding="utf-8")
        )
        entry = next(
            plugin
            for plugin in marketplace["plugins"]
            if plugin["name"] == "rrr"
        )
        self.assertEqual("./plugins/rrr", entry["source"]["path"])
        self.assertEqual("AVAILABLE", entry["policy"]["installation"])
        self.assertTrue((ROOT / "plugins" / "rrr").is_dir())


if __name__ == "__main__":
    unittest.main()
