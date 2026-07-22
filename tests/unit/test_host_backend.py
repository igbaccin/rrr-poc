import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rrr import host_backend
from rrr import language


class HostBackendTests(unittest.TestCase):
    def test_codex_host_uses_isolated_subscription_command(self):
        observed = {}

        def fake_run(command, prompt, cwd, env):
            observed.update(
                command=list(command),
                prompt=prompt,
                cwd=cwd,
                env=env,
            )
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("HOST_OK", encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="HOST_OK\n",
                stderr="model: gpt-5.6-sol\n",
            )

        with (
            patch.dict(
                os.environ,
                {
                    "RRR_HOST": "codex",
                    "RRR_HOST_MODEL": "gpt-5.6-sol",
                    "RRR_HOST_AUDIT": "0",
                    "OPENAI_API_KEY": "must-not-leak",
                    "GITHUB_TOKEN": "also-must-not-leak",
                },
                clear=False,
            ),
            patch.object(
                host_backend,
                "_command_candidates",
                return_value=["codex"],
            ),
            patch.object(host_backend, "_run_process", side_effect=fake_run),
            patch.object(
                host_backend,
                "_verify_subscription_auth",
                return_value={
                    "kind": "subscription",
                    "provider": "openai",
                    "method": "chatgpt",
                },
            ),
        ):
            result = host_backend.host_chat(
                messages=[
                    {"role": "system", "content": "Use the evidence."},
                    {"role": "user", "content": "Write one sentence."},
                ],
                options={"num_predict": 300},
                _rrr_stage="test.prose",
            )

        self.assertEqual({"message": {"content": "HOST_OK"}}, result)
        command = observed["command"]
        self.assertIn("exec", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("read-only", command)
        self.assertIn("gpt-5.6-sol", command)
        self.assertNotIn("--ignore-rules", command)
        self.assertIn("shell_tool", command)
        self.assertIn("project_doc_max_bytes=0", command)
        self.assertIn('shell_environment_policy.inherit="none"', command)
        self.assertTrue(
            any(
                value.startswith("model_instructions_file=")
                for value in command
            )
        )
        self.assertNotIn("OPENAI_API_KEY", observed["env"])
        self.assertNotIn("GITHUB_TOKEN", observed["env"])
        self.assertFalse(any(key.startswith("RRR_") for key in observed["env"]))
        self.assertIn("test.prose", observed["prompt"])
        self.assertIn("Use the evidence.", observed["prompt"])

    def test_claude_host_disables_tools_and_uses_system_prompt(self):
        observed = {}

        def fake_run(command, prompt, cwd, env):
            observed.update(
                command=list(command),
                prompt=prompt,
                cwd=cwd,
                env=env,
            )
            system_path = Path(command[command.index("--system-prompt-file") + 1])
            observed["system"] = system_path.read_text(encoding="utf-8")
            wrapper = {
                "result": '{"decision":"PROCEED"}',
                "is_error": False,
                "modelUsage": {"claude-opus-4-8": {"inputTokens": 10}},
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(wrapper),
                stderr="",
            )

        with (
            patch.dict(
                os.environ,
                {
                    "RRR_HOST": "claude",
                    "RRR_HOST_MODEL": "opus",
                    "RRR_HOST_AUDIT": "0",
                    "ANTHROPIC_API_KEY": "must-not-leak",
                },
                clear=False,
            ),
            patch.object(
                host_backend,
                "_command_candidates",
                return_value=["claude"],
            ),
            patch.object(host_backend, "_run_process", side_effect=fake_run),
            patch.object(
                host_backend,
                "_verify_subscription_auth",
                return_value={
                    "kind": "subscription",
                    "provider": "anthropic",
                    "method": "claude.ai",
                    "subscription_type": "max",
                },
            ),
        ):
            result = host_backend.host_chat(
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": "Classify the corpus."},
                ],
                options={"num_predict": 400},
                format="json",
                _rrr_stage="test.json",
            )

        self.assertEqual(
            '{"decision":"PROCEED"}',
            result["message"]["content"],
        )
        command = observed["command"]
        tools_index = command.index("--tools")
        self.assertEqual("", command[tools_index + 1])
        denied_index = command.index("--disallowedTools")
        self.assertEqual("*", command[denied_index + 1])
        self.assertIn("--strict-mcp-config", command)
        self.assertIn("--no-session-persistence", command)
        self.assertIn("--safe-mode", command)
        self.assertNotIn("--bare", command)
        sources_index = command.index("--setting-sources")
        self.assertEqual("", command[sources_index + 1])
        self.assertNotIn("ANTHROPIC_API_KEY", observed["env"])
        self.assertIn("Return JSON only.", observed["system"])
        self.assertNotIn("Return JSON only.", observed["prompt"])
        self.assertIn("Classify the corpus.", observed["prompt"])
        self.assertIn("exactly one JSON object", observed["prompt"])

    def test_claude_error_wrapper_fails_closed(self):
        def fake_run(command, prompt, cwd, env):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "is_error": True,
                        "result": "subscription allowance exhausted",
                    }
                ),
                stderr="",
            )

        with (
            patch.dict(
                os.environ,
                {"RRR_HOST": "claude", "RRR_HOST_AUDIT": "0"},
                clear=False,
            ),
            patch.object(
                host_backend,
                "_command_candidates",
                return_value=["claude"],
            ),
            patch.object(host_backend, "_run_process", side_effect=fake_run),
            patch.object(
                host_backend,
                "_verify_subscription_auth",
                return_value={
                    "kind": "subscription",
                    "provider": "anthropic",
                    "method": "claude.ai",
                    "subscription_type": "max",
                },
            ),
        ):
            with self.assertRaisesRegex(
                host_backend.HostBackendError,
                "allowance exhausted",
            ):
                host_backend.host_chat(
                    messages=[{"role": "user", "content": "Answer."}],
                )

    def test_host_audit_preserves_request_and_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(
                    os.environ,
                    {
                        "RRR_HOST": "codex",
                        "RRR_HOST_MODEL": "gpt-5.6-sol",
                        "RRR_HOST_AUDIT": "1",
                        "RRR_HOST_AUDIT_DIR": temp_dir,
                    },
                    clear=False,
                ),
                patch.object(
                    host_backend,
                    "_codex_chat",
                    return_value=(
                        "answer",
                        "gpt-5.6-sol",
                        "diagnostic",
                        {
                            "kind": "subscription",
                            "provider": "openai",
                            "method": "chatgpt",
                        },
                    ),
                ),
            ):
                host_backend.host_chat(
                    messages=[{"role": "user", "content": "question"}],
                    _rrr_stage="test.audit",
                )

            records = list(Path(temp_dir).glob("*.json"))
            self.assertEqual(1, len(records))
            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual("isolated_cli", record["context_mode"])
            self.assertEqual("test.audit", record["stage"])
            self.assertEqual("question", record["request"]["messages"][0]["content"])
            self.assertEqual("answer", record["response"])
            self.assertEqual("chatgpt", record["authentication"]["method"])
            self.assertEqual(64, len(record["prompt_sha256"]))

    def test_codex_auth_requires_chatgpt_login(self):
        with patch.object(
            host_backend,
            "_run_status",
            return_value=subprocess.CompletedProcess(
                ["codex", "login", "status"],
                0,
                stdout="Logged in using an API key\n",
                stderr="",
            ),
        ):
            with self.assertRaisesRegex(
                host_backend.HostBackendError,
                "ChatGPT product login",
            ):
                host_backend._verify_codex_subscription("codex", {})

    def test_codex_auth_accepts_chatgpt_login(self):
        with patch.object(
            host_backend,
            "_run_status",
            return_value=subprocess.CompletedProcess(
                ["codex", "login", "status"],
                0,
                stdout="Logged in using ChatGPT\n",
                stderr="",
            ),
        ):
            auth = host_backend._verify_codex_subscription("codex", {})
        self.assertEqual("chatgpt", auth["method"])

    def test_claude_auth_requires_a_subscription(self):
        console_status = {
            "loggedIn": True,
            "authMethod": "api_key",
            "apiProvider": "firstParty",
        }
        with patch.object(
            host_backend,
            "_run_status",
            return_value=subprocess.CompletedProcess(
                ["claude", "auth", "status"],
                0,
                stdout=json.dumps(console_status),
                stderr="",
            ),
        ):
            with self.assertRaisesRegex(
                host_backend.HostBackendError,
                "Claude.ai Pro",
            ):
                host_backend._verify_claude_subscription("claude", {})

    def test_claude_auth_accepts_oauth_subscription(self):
        subscription_status = {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        }
        with patch.object(
            host_backend,
            "_run_status",
            return_value=subprocess.CompletedProcess(
                ["claude", "auth", "status"],
                0,
                stdout=json.dumps(subscription_status),
                stderr="",
            ),
        ):
            auth = host_backend._verify_claude_subscription("claude", {})
        self.assertEqual("max", auth["subscription_type"])

    def test_claude_capability_check_requires_safe_mode(self):
        with patch.object(
            host_backend,
            "_run_status",
            return_value=subprocess.CompletedProcess(
                ["claude", "--version"],
                0,
                stdout="2.1.168 (Claude Code)",
                stderr="",
            ),
        ):
            with self.assertRaisesRegex(
                host_backend.HostBackendError,
                "--safe-mode",
            ):
                host_backend._verify_claude_capabilities("claude", {})

    def test_host_diagnostics_falls_back_to_compatible_executable(self):
        auth = {
            "kind": "subscription",
            "provider": "openai",
            "method": "chatgpt",
        }
        with (
            patch.dict(
                os.environ,
                {"RRR_HOST": "codex", "RRR_HOST_MODEL": "bad-model"},
                clear=False,
            ),
            patch.object(
                host_backend,
                "_command_candidates",
                return_value=["old-codex", "new-codex"],
            ),
            patch.object(
                host_backend,
                "_verify_subscription_auth",
                side_effect=[
                    host_backend.HostBackendError("old CLI incompatible"),
                    auth,
                ],
            ),
        ):
            status = host_backend.host_diagnostics()
        self.assertEqual("new-codex", status["executable"])
        self.assertTrue(status["authentication_ready"])
        self.assertFalse(status["inference_tested"])
        self.assertNotIn("ready", status)

    def test_codex_chat_falls_back_after_incompatible_executable(self):
        attempted = []

        def fake_run(command, prompt, cwd, env):
            attempted.append(command[0])
            if command[0] == "old-codex":
                return subprocess.CompletedProcess(
                    command,
                    2,
                    stdout="",
                    stderr="unknown option --ignore-user-config",
                )
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("FALLBACK_OK", encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr="model: gpt-5.6-sol\n",
            )

        with (
            patch.dict(
                os.environ,
                {"RRR_HOST": "codex", "RRR_HOST_AUDIT": "0"},
                clear=False,
            ),
            patch.object(
                host_backend,
                "_command_candidates",
                return_value=["old-codex", "new-codex"],
            ),
            patch.object(host_backend, "_run_process", side_effect=fake_run),
            patch.object(
                host_backend,
                "_verify_subscription_auth",
                return_value={
                    "kind": "subscription",
                    "provider": "openai",
                    "method": "chatgpt",
                },
            ),
        ):
            result = host_backend.host_chat(
                messages=[{"role": "user", "content": "Answer."}],
            )
        self.assertEqual(["old-codex", "new-codex"], attempted)
        self.assertEqual("FALLBACK_OK", result["message"]["content"])

    def test_timeout_terminates_the_process_tree(self):
        class FakeProcess:
            pid = 12345
            returncode = -9

            def __init__(self):
                self.communications = 0

            def communicate(self, _prompt=None, timeout=None):
                self.communications += 1
                if self.communications == 1:
                    raise subprocess.TimeoutExpired(["codex"], timeout)
                return "", ""

            def poll(self):
                return None

        fake = FakeProcess()
        with (
            patch.object(subprocess, "Popen", return_value=fake),
            patch.object(host_backend, "_terminate_process_tree") as terminate,
        ):
            with self.assertRaisesRegex(
                host_backend.HostBackendError,
                "timeout test",
            ):
                host_backend._run_command(
                    ["codex"],
                    prompt="request",
                    cwd=None,
                    env={},
                    timeout=0.01,
                    timeout_message="timeout test",
                )
        terminate.assert_called_once_with(fake)

    def test_enabled_audit_failure_blocks_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_directory = Path(temp_dir) / "not-a-directory"
            invalid_directory.write_text("occupied", encoding="utf-8")
            with (
                patch.dict(
                    os.environ,
                    {
                        "RRR_HOST": "codex",
                        "RRR_HOST_AUDIT": "1",
                        "RRR_HOST_AUDIT_DIR": str(invalid_directory),
                    },
                    clear=False,
                ),
                patch.object(
                    host_backend,
                    "_codex_chat",
                    return_value=(
                        "answer",
                        "gpt-5.6-sol",
                        "diagnostic",
                        {
                            "kind": "subscription",
                            "provider": "openai",
                            "method": "chatgpt",
                        },
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    host_backend.HostBackendError,
                    "audit record could not be written",
                ):
                    host_backend.host_chat(
                        messages=[{"role": "user", "content": "question"}],
                    )

    def test_host_smoke_test_validates_json_contract(self):
        with patch.object(
            host_backend,
            "host_chat",
            return_value={
                "message": {"content": '{"status":"RRR_HOST_OK"}'},
            },
        ):
            status = host_backend.host_smoke_test()
        self.assertTrue(status["inference_ready"])
        self.assertTrue(status["inference_tested"])

    def test_child_environment_scrubs_secrets_and_preserves_claude_oauth(self):
        with patch.dict(
            os.environ,
            {
                "RRR_RUNTIME": "host",
                "OPENAI_API_KEY": "openai-secret",
                "GITHUB_TOKEN": "github-secret",
                "PGPASSWORD": "postgres-secret",
                "DATABASE_URL": "postgres://secret",
                "REDIS_URL": "redis://secret",
                "DOCKER_AUTH_CONFIG": "docker-secret",
                "AZURE_STORAGE_CONNECTION_STRING": "azure-secret",
                "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
                "CLAUDE_CODE_OAUTH_EXTRA": "must-not-survive",
                "PATH": "safe-path",
            },
            clear=True,
        ):
            env = host_backend._child_environment("claude")
        self.assertEqual("safe-path", env["PATH"])
        self.assertEqual(
            "subscription-token",
            env["CLAUDE_CODE_OAUTH_TOKEN"],
        )
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("PGPASSWORD", env)
        self.assertNotIn("DATABASE_URL", env)
        self.assertNotIn("REDIS_URL", env)
        self.assertNotIn("DOCKER_AUTH_CONFIG", env)
        self.assertNotIn("AZURE_STORAGE_CONNECTION_STRING", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_EXTRA", env)
        self.assertNotIn("RRR_RUNTIME", env)
        self.assertEqual("1", env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"])

        with patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
                "PATH": "safe-path",
            },
            clear=True,
        ):
            codex_env = host_backend._child_environment("codex")
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", codex_env)

    def test_host_model_name_is_provider_qualified(self):
        with patch.dict(
            os.environ,
            {"RRR_HOST": "claude", "RRR_HOST_MODEL": "opus"},
            clear=False,
        ):
            self.assertEqual("claude:opus", host_backend.host_model_name())

    def test_unknown_host_is_rejected(self):
        with patch.dict(
            os.environ,
            {"RRR_HOST": "unknown"},
            clear=False,
        ):
            with self.assertRaisesRegex(host_backend.HostBackendError, "RRR_HOST"):
                host_backend.host_model_name()

    def test_language_router_bypasses_ollama_in_host_mode(self):
        with (
            patch.dict(
                os.environ,
                {
                    "RRR_RUNTIME": "host",
                    "RRR_HOST": "codex",
                    "RRR_HOST_MODEL": "gpt-5.6-sol",
                },
                clear=False,
            ),
            patch.object(
                language,
                "_list_ollama_models",
                side_effect=AssertionError("Ollama registry must not be read"),
            ),
        ):
            self.assertEqual(
                "codex:gpt-5.6-sol",
                language.select_model("zh"),
            )

    def test_llm_shim_routes_host_calls(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **_kwargs: {"message": {"content": "local"}},
            Client=lambda **_kwargs: SimpleNamespace(
                chat=lambda **_inner: {"message": {"content": "local"}}
            ),
        )
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            from rrr import llm

            self.assertTrue(llm.install())
            with (
                patch.dict(
                    os.environ,
                    {"RRR_RUNTIME": "host"},
                    clear=False,
                ),
                patch(
                    "rrr.host_backend.host_chat",
                    return_value={"message": {"content": "host"}},
                ) as routed,
            ):
                result = fake_ollama.chat(
                    model="ignored",
                    messages=[{"role": "user", "content": "hello"}],
                )
        self.assertEqual("host", result["message"]["content"])
        routed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
