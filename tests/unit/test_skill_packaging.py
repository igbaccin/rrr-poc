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
        self.assertEqual(manifest["interface"]["category"], "Education & Research")
        self.assertLessEqual(len(manifest["interface"]["shortDescription"]), 30)
        self.assertEqual(
            manifest["interface"]["websiteURL"],
            "https://github.com/igbaccin/rrr-poc",
        )
        self.assertEqual(
            manifest["interface"]["capabilities"],
            ["Literature reviews", "Claim evaluation", "Page-cited evidence"],
        )
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

    def test_hosted_profiles_are_separately_versioned_artifacts(self):
        """vcx (Codex) and vcl (Claude) must be independently replaceable.

        They ship the same tested runtime today, which is the intended
        starting point: Claude begins from the Codex contract. But they are
        separate files with separate manifests, so a Claude-specific change
        rebuilds only vcl and cannot silently alter the runtime Codex was
        accepted with.
        """
        plugin_runtime = json.loads(
            (PLUGIN_ROOT / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        claude_root = CLAUDE_SKILL.parent
        claude_runtime = json.loads(
            (claude_root / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        self.assertEqual("cx", plugin_runtime["profile"])
        self.assertEqual("cl", claude_runtime["profile"])
        self.assertEqual("vcx.2.0", plugin_runtime["profile_version"])
        self.assertEqual("vcl.2.0", claude_runtime["profile_version"])

        plugin_wheel = PLUGIN_ROOT / "runtime" / plugin_runtime["wheel"]
        claude_wheel = claude_root / "runtime" / claude_runtime["wheel"]
        # Distinct files, so one can be rebuilt without the other.
        self.assertNotEqual(plugin_wheel.resolve(), claude_wheel.resolve())
        # Byte-identical today. When the Claude modifications land this
        # assertion is the one to update, together with a bump to vcl.2.1 --
        # never by editing a shared wheel in place.
        self.assertEqual(claude_wheel.read_bytes(), plugin_wheel.read_bytes())
        for manifest, wheel in (
            (plugin_runtime, plugin_wheel),
            (claude_runtime, claude_wheel),
        ):
            self.assertEqual(
                hashlib.sha256(wheel.read_bytes()).hexdigest(),
                manifest["sha256"],
            )
        self.assertEqual(
            (claude_root / "scripts" / "bootstrap_rrr.py").read_bytes(),
            (PLUGIN_ROOT / "scripts" / "bootstrap_rrr.py").read_bytes(),
        )

    def test_local_distribution_is_versioned_independently_from_hosted(self):
        """The local product and the hosted runtimes are separate profiles.

        They were once the same wheel, synced from dist/ into the plugin. They
        are now versioned independently -- vlm for local models, vcx and vcl
        for the hosted agents -- because they are different writers with
        different prose contracts, not successive versions of one thing.
        """
        local_wheels = sorted((ROOT / "dist").glob("rrr_poc-*.whl"))
        self.assertEqual(1, len(local_wheels), "expected exactly one local wheel")
        local_wheel = local_wheels[0]
        local_hash = hashlib.sha256(local_wheel.read_bytes()).hexdigest()
        checksum = (
            ROOT / "dist" / f"{local_wheel.name}.sha256"
        ).read_text(encoding="ascii")
        self.assertEqual(checksum, f"{local_hash}  {local_wheel.name}\n")

        hosted = json.loads(
            (PLUGIN_ROOT / "runtime" / "runtime.json").read_text(encoding="utf-8")
        )
        # The local wheel must not be silently republished as the hosted
        # runtime: that is how a local-only fix would reach Codex untested.
        self.assertNotEqual(local_wheel.name, hosted["wheel"])

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
        self.assertEqual("Education & Research", entry["category"])
        self.assertTrue((ROOT / "plugins" / "rrr").is_dir())


if __name__ == "__main__":
    unittest.main()
