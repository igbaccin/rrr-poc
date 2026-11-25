import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rrr import host_backend
from rrr.cli import host_doctor_main


THREAD = "01a210a1-9869-7ed2-8271-3716431a7b1a"
SESSION = "10b107c0-46e1-422b-833e-b50185bc3427"
CODEX_FALLBACK = "gpt-6.2-sol"
CLAUDE_FALLBACK = "claude-opus-5-5"


def _clear():
    host_backend._codex_selection.cache_clear()
    host_backend._claude_selection.cache_clear()


class _Hosts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.codex = Path(self.temp.name) / "codex"
        self.claude = Path(self.temp.name) / "claude"
        self.codex.mkdir()
        self.claude.mkdir()
        self.environment = patch.dict(
            os.environ,
            {
                "CODEX_HOME": str(self.codex),
                "CLAUDE_CONFIG_DIR": str(self.claude),
                "RRR_HOST": "codex",
            },
            clear=False,
        )
        self.environment.start()
        for name in ("RRR_HOST_MODEL", "RRR_HOST_EFFORT", "CODEX_THREAD_ID",
                     "CLAUDE_CODE_SESSION_ID", "ANTHROPIC_MODEL", "CLAUDE_CODE_EFFORT_LEVEL"):
            os.environ.pop(name, None)
        _clear()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()
        _clear()

    def write_config(self, text):
        (self.codex / "config.toml").write_text(text, encoding="utf-8")
        _clear()

    def write_thread(self, turns, thread=THREAD, provider="openai"):
        folder = self.codex / "sessions" / "2026" / "11" / "24"
        folder.mkdir(parents=True, exist_ok=True)
        rows = [{"type": "session_meta", "payload": {"id": thread, "model_provider": provider}}]
        for turn in turns:
            model, effort = turn if isinstance(turn, tuple) else (turn, None)
            context = {"model": model} if effort is None else {"model": model, "effort": effort}
            rows.append({"type": "turn_context", "payload": context})
            rows.append({"type": "response_item", "payload": {"type": "message"}})
        (folder / f"rollout-2026-11-24T23-55-56-{thread}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\nnot json\n", encoding="utf-8",
        )
        _clear()

    def write_settings(self, data):
        text = data if isinstance(data, str) else json.dumps(data)
        (self.claude / "settings.json").write_text(text, encoding="utf-8")
        _clear()

    def write_session(self, rows, session=SESSION):
        folder = self.claude / "projects" / "C--Users-reader-papers"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{session}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
        )
        _clear()


def _assistant(model, **extra):
    return {"type": "assistant", "message": {"role": "assistant", "model": model}, **extra}


class CodexModelTests(_Hosts):
    def test_the_thread_that_starts_rrr_decides(self):
        self.write_config('model = "gpt-6.2-astra"\n')
        self.write_thread(["gpt-6.2-sol"])
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual(
                ("gpt-6.2-sol", "this Codex thread"), host_backend.resolve_host_model("codex"),
            )
            self.assertEqual("codex:gpt-6.2-sol", host_backend.host_model_name())

    def test_the_latest_turn_of_the_thread_decides(self):
        self.write_thread(["gpt-6.1-sol", "gpt-6.2-astra"])
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual("gpt-6.2-astra", host_backend.resolve_host_model("codex")[0])

    def test_without_a_thread_the_configuration_decides(self):
        self.write_config(
            'service_tier = "fast"\nmodel = "gpt-6.2-astra"  # picked in the application\n'
            '\n[plugins."rrr@rrr-poc"]\nenabled = true\nmodel = "not this one"\n'
        )
        model, source = host_backend.resolve_host_model("codex")
        self.assertEqual("gpt-6.2-astra", model)
        self.assertIn("the Codex configuration", source)
        self.assertIn("config.toml", source)

    def test_a_thread_without_a_record_falls_to_the_configuration(self):
        self.write_config('model = "gpt-6.1-terra"\n')
        self.write_thread(["gpt-6.2-sol"], thread="ffffffff-0000-0000-0000-000000000000")
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual("gpt-6.1-terra", host_backend.resolve_host_model("codex")[0])

    def test_a_thread_identifier_that_is_no_identifier_is_never_searched_for(self):
        self.write_config('model = "gpt-6.1-terra"\n')
        self.write_thread(["gpt-6.2-sol"])
        for value in ("*", "../../" + THREAD, THREAD[:6] + "*", "a b c d e f g h"):
            with patch.dict(os.environ, {"CODEX_THREAD_ID": value}):
                self.assertEqual("gpt-6.1-terra", host_backend.resolve_host_model("codex")[0], value)

    def test_the_active_profile_of_the_configuration_is_read(self):
        self.write_config(
            'model = "gpt-6.2-astra"\nprofile = "review"\n\n'
            '[profiles.review]\nmodel = "gpt-6.1-sol"\n\n[profiles.other]\nmodel = "gpt-6-luna"\n'
        )
        self.assertEqual("gpt-6.1-sol", host_backend.resolve_host_model("codex")[0])
        self.write_config(
            'model = "gpt-6.1-terra"\nprofile = ["review"]\n\n[profiles.review]\nmodel = "gpt-6-luna"\n'
        )
        self.assertEqual("gpt-6.1-terra", host_backend.resolve_host_model("codex")[0])

    def test_a_model_of_another_provider_is_not_inherited(self):
        self.write_config('model = "qwen3:32b"\nmodel_provider = "ollama"\n')
        model, source = host_backend.resolve_host_model("codex")
        self.assertEqual((CODEX_FALLBACK, True), (model, "fallback" in source))
        self.write_config('model = "gpt-6.1-terra"\n')
        self.write_thread(["qwen3:32b"], provider="ollama")
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual("gpt-6.1-terra", host_backend.resolve_host_model("codex")[0])

    def test_no_selection_gives_the_fallback(self):
        model, source = host_backend.resolve_host_model("codex")
        self.assertEqual(CODEX_FALLBACK, model)
        self.assertIn("could not be established", source)
        for broken in ("model = ", "= = =", "model = 62\n", 'model = ""\n', "[profiles\n"):
            self.write_config(broken)
            self.assertEqual(CODEX_FALLBACK, host_backend.resolve_host_model("codex")[0], broken)

    def test_python_without_tomllib_reads_the_same_configuration(self):
        text = (
            "# Codex\nmodel = 'gpt-6.2-astra'   # single quotes\nprofile = \"review\"\n"
            "notify = [\"a\", \"b\"]\n\n[profiles.review]\nmodel = \"gpt-6.1-sol\"\n"
            "\n[[history]]\nmodel = \"not this one\"\n"
            "\n[profiles.\"long run\"]\nmodel = \"gpt-6-luna\"\n"
        )
        strings = host_backend._toml_strings(text)
        self.assertEqual("gpt-6.2-astra", strings["model"])
        self.assertEqual("gpt-6.1-sol", strings["profiles"]["review"]["model"])
        self.assertEqual("gpt-6-luna", strings["profiles"]["long run"]["model"])
        self.write_config(text)
        with patch.dict(sys.modules, {"tomllib": None}):
            self.assertEqual("gpt-6.1-sol", host_backend.resolve_host_model("codex")[0])
        _clear()
        self.assertEqual("gpt-6.1-sol", host_backend.resolve_host_model("codex")[0])

    def test_the_environment_variable_comes_first(self):
        self.write_config('model = "gpt-6.2-astra"\n')
        self.write_thread(["gpt-6.2-sol"])
        with patch.dict(os.environ, {"RRR_HOST_MODEL": " gpt-6.1-terra ", "CODEX_THREAD_ID": THREAD}):
            self.assertEqual(
                ("gpt-6.1-terra", "RRR_HOST_MODEL"), host_backend.resolve_host_model("codex"),
            )
        with patch.dict(os.environ, {"RRR_HOST_MODEL": "   "}):
            self.assertEqual("gpt-6.2-astra", host_backend.resolve_host_model("codex")[0])

    def test_one_run_keeps_one_model_when_the_selection_changes_under_it(self):
        self.write_config('model = "gpt-6.2-sol"\n')
        self.assertEqual("gpt-6.2-sol", host_backend.resolve_host_model("codex")[0])
        (self.codex / "config.toml").write_text('model = "gpt-7-sol"\n', encoding="utf-8")
        self.assertEqual("gpt-6.2-sol", host_backend.resolve_host_model("codex")[0])
        _clear()
        self.assertEqual("gpt-7-sol", host_backend.resolve_host_model("codex")[0])


class ClaudeModelTests(_Hosts):
    def test_the_session_that_starts_rrr_decides(self):
        self.write_settings({"model": "opus[1m]"})
        self.write_session([
            {"type": "user", "message": {"role": "user", "content": "run it"}},
            _assistant("claude-sonnet-5-5"),
            _assistant("claude-fable-5-2"),
            _assistant("claude-haiku-5-5", isSidechain=True),
            _assistant("<synthetic>"),
            {"type": "user", "message": {"role": "user", "content": "an \"assistant\" in text"}},
        ])
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": SESSION, "ANTHROPIC_MODEL": "sonnet"}):
            self.assertEqual(
                ("claude-fable-5-2", "this Claude Code session"),
                host_backend.resolve_host_model("claude"),
            )

    def test_without_a_session_the_variable_then_the_settings_decide(self):
        self.write_settings({"model": "opus[1m]", "modelSettings": {}})
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-sonnet-5-5"}):
            self.assertEqual(
                ("claude-sonnet-5-5", "ANTHROPIC_MODEL"), host_backend.resolve_host_model("claude"),
            )
        model, source = host_backend.resolve_host_model("claude")
        self.assertEqual("opus[1m]", model)
        self.assertIn("the Claude Code settings", source)
        self.assertIn("settings.json", source)

    def test_a_session_without_a_transcript_falls_to_the_settings(self):
        self.write_settings({"model": "sonnet"})
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": SESSION}):
            self.assertEqual("sonnet", host_backend.resolve_host_model("claude")[0])
        self.write_session([{"type": "user", "message": {"role": "user", "content": "x"}}])
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": SESSION}):
            self.assertEqual("sonnet", host_backend.resolve_host_model("claude")[0])

    def test_no_selection_gives_opus(self):
        model, source = host_backend.resolve_host_model("claude")
        self.assertEqual(CLAUDE_FALLBACK, model)
        self.assertIn("could not be established", source)
        for broken in ("not json", "[]", '{"model": 5}', '{"model": ""}', '{"theme": "dark"}'):
            self.write_settings(broken)
            self.assertEqual(CLAUDE_FALLBACK, host_backend.resolve_host_model("claude")[0], broken)

    def test_each_host_reads_only_its_own_selection(self):
        self.write_config('model = "gpt-6.2-astra"\n')
        self.write_settings({"model": "sonnet"})
        self.write_thread(["gpt-6.1-sol"])
        self.write_session([_assistant("claude-fable-5-2")])
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD, "CLAUDE_CODE_SESSION_ID": SESSION}):
            self.assertEqual("gpt-6.1-sol", host_backend.resolve_host_model("codex")[0])
            self.assertEqual("claude-fable-5-2", host_backend.resolve_host_model("claude")[0])
        with patch.dict(os.environ, {"RRR_HOST_MODEL": "sonnet"}):
            self.assertEqual(("sonnet", "RRR_HOST_MODEL"), host_backend.resolve_host_model("claude"))


class EffortTests(_Hosts):
    CONFIG = 'model = "gpt-6.2-astra"\nmodel_reasoning_effort = "xhigh"\n'

    def test_the_thread_gives_the_effort_of_its_latest_turn(self):
        self.write_config(self.CONFIG)
        self.write_thread([("gpt-6.2-sol", "high"), ("gpt-6.2-sol", "low")])
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual(("low", "this Codex thread"), host_backend.resolve_host_effort("codex"))
            self.assertEqual("codex:gpt-6.2-sol@low", host_backend.host_model_name())

    def test_the_configuration_gives_the_effort_when_the_thread_names_none(self):
        self.write_config('model = "gpt-6.2-astra"\nmodel_reasoning_effort = "XHigh"\n')
        effort, source = host_backend.resolve_host_effort("codex")
        self.assertEqual("xhigh", effort)
        self.assertIn("the Codex configuration", source)
        self.write_thread(["gpt-6.2-sol"])
        with patch.dict(os.environ, {"CODEX_THREAD_ID": THREAD}):
            self.assertEqual("codex:gpt-6.2-sol@xhigh", host_backend.host_model_name())

    def test_the_active_profile_gives_its_own_effort(self):
        self.write_config(
            'model = "gpt-6.2-astra"\nmodel_reasoning_effort = "low"\nprofile = "review"\n\n'
            '[profiles.review]\nmodel_reasoning_effort = "high"\n'
        )
        self.assertEqual("high", host_backend.resolve_host_effort("codex")[0])
        self.assertEqual("codex:gpt-6.2-astra@high", host_backend.host_model_name())

    def test_no_selection_means_the_default_effort_of_the_model(self):
        effort, source = host_backend.resolve_host_effort("codex")
        self.assertEqual("", effort)
        self.assertIn("no effort selected in Codex", source)
        self.assertEqual("codex:" + CODEX_FALLBACK, host_backend.host_model_name())
        for broken in ("model_reasoning_effort = 3\n", 'model_reasoning_effort = "very high"\n',
                       'model_reasoning_effort = ""\n'):
            self.write_config(broken)
            self.assertEqual("", host_backend.resolve_host_effort("codex")[0], broken)

    def test_the_environment_variable_comes_first_and_default_asks_for_none(self):
        self.write_config(self.CONFIG)
        with patch.dict(os.environ, {"RRR_HOST_EFFORT": " Medium "}):
            self.assertEqual(("medium", "RRR_HOST_EFFORT"), host_backend.resolve_host_effort("codex"))
            self.assertEqual("codex:gpt-6.2-astra@medium", host_backend.host_model_name())
        with patch.dict(os.environ, {"RRR_HOST_EFFORT": "default"}):
            self.assertEqual(("", "RRR_HOST_EFFORT"), host_backend.resolve_host_effort("codex"))
            self.assertEqual("codex:gpt-6.2-astra", host_backend.host_model_name())
        with patch.dict(os.environ, {"RRR_HOST_EFFORT": "very high"}):
            with self.assertRaises(host_backend.HostBackendError):
                host_backend.resolve_host_effort("codex")

    def test_a_pinned_model_inherits_no_effort(self):
        self.write_config(self.CONFIG)
        self.write_thread([("gpt-6.2-astra", "xhigh")])
        with patch.dict(os.environ, {"RRR_HOST_MODEL": "gpt-6.2-sol", "CODEX_THREAD_ID": THREAD}):
            effort, source = host_backend.resolve_host_effort("codex")
            self.assertEqual("", effort)
            self.assertIn("RRR_HOST_MODEL pins the model", source)
            self.assertEqual("codex:gpt-6.2-sol", host_backend.host_model_name())
            with patch.dict(os.environ, {"RRR_HOST_EFFORT": "low"}):
                self.assertEqual("codex:gpt-6.2-sol@low", host_backend.host_model_name())

    def test_a_configuration_for_another_provider_gives_no_effort(self):
        self.write_config(
            'model = "qwen3:32b"\nmodel_provider = "ollama"\nmodel_reasoning_effort = "high"\n'
        )
        self.assertEqual("", host_backend.resolve_host_effort("codex")[0])

    def test_the_claude_session_gives_the_effort(self):
        self.write_settings({"model": "opus[1m]", "effortLevel": "low"})
        self.write_session([
            _assistant("claude-fable-5-2", effort="high"),
            _assistant("claude-fable-5-2", effort="xhigh", perTurnEffort="xhigh"),
            _assistant("claude-haiku-5-5", effort="low", isSidechain=True),
        ])
        with patch.dict(os.environ, {"RRR_HOST": "claude", "CLAUDE_CODE_SESSION_ID": SESSION,
                                     "CLAUDE_CODE_EFFORT_LEVEL": "medium"}):
            self.assertEqual(
                ("xhigh", "this Claude Code session"), host_backend.resolve_host_effort("claude"),
            )
            self.assertEqual("claude:claude-fable-5-2@xhigh", host_backend.host_model_name())

    def test_without_a_session_the_variable_then_the_settings_give_the_effort(self):
        self.write_settings({"model": "opus[1m]", "effortLevel": "low"})
        with patch.dict(os.environ, {"CLAUDE_CODE_EFFORT_LEVEL": "medium"}):
            self.assertEqual(
                ("medium", "CLAUDE_CODE_EFFORT_LEVEL"), host_backend.resolve_host_effort("claude"),
            )
        effort, source = host_backend.resolve_host_effort("claude")
        self.assertEqual("low", effort)
        self.assertIn("the Claude Code settings", source)

    def test_the_effort_kept_for_the_selected_model_comes_before_the_general_one(self):
        per_model = {
            "claude-opus-5-5": {"effortLevel": "xhigh"},
            "claude-sonnet-5-5": {"effortLevel": "medium"},
        }
        self.write_settings(
            {"model": "claude-opus-5-5[1m]", "effortLevel": "low", "modelSettings": per_model}
        )
        self.assertEqual("xhigh", host_backend.resolve_host_effort("claude")[0])
        self.write_settings({"model": "opus[1m]", "modelSettings": per_model})
        self.assertEqual("xhigh", host_backend.resolve_host_effort("claude")[0])
        per_model["claude-opus-4-8"] = {"effortLevel": "high"}
        self.write_settings({"model": "opus", "modelSettings": per_model})
        effort, source = host_backend.resolve_host_effort("claude")
        self.assertEqual("", effort)
        self.assertIn("no effort selected in Claude Code", source)
        self.write_settings({"model": "opus", "modelSettings": "none", "effortLevel": 4})
        self.assertEqual("", host_backend.resolve_host_effort("claude")[0])


class HostCallUsesTheResolvedModelTests(_Hosts):
    CODEX_AUTH = {"kind": "subscription", "provider": "openai", "method": "chatgpt"}
    CLAUDE_AUTH = {"kind": "subscription", "provider": "anthropic", "method": "claude.ai"}

    def test_the_codex_command_names_the_inherited_model_and_ignores_user_config(self):
        self.write_config('model = "gpt-6.2-astra"\n')
        observed = {}

        def fake_run(command, prompt, cwd, env):
            observed["command"] = list(command)
            observed["env"] = dict(env)
            Path(command[command.index("-o") + 1]).write_text("OK", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="OK\n", stderr="")

        with patch.dict(os.environ, {"RRR_HOST_AUDIT": "0", "CODEX_THREAD_ID": THREAD}), \
                patch.object(host_backend, "_command_candidates", return_value=["codex"]), \
                patch.object(host_backend, "_run_process", side_effect=fake_run), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CODEX_AUTH):
            host_backend.host_chat(model="mistral-small:24b",
                                   messages=[{"role": "user", "content": "x"}])
        command = observed["command"]
        self.assertEqual("gpt-6.2-astra", command[command.index("--model") + 1])
        self.assertIn("--ignore-user-config", command)
        self.assertNotIn("CODEX_THREAD_ID", observed["env"])
        self.assertFalse([part for part in command if "model_reasoning_effort" in part])

    def test_the_codex_command_carries_the_inherited_effort_and_the_audit_records_it(self):
        self.write_config('model = "gpt-6.2-astra"\nmodel_reasoning_effort = "xhigh"\n')
        observed = {}

        def fake_run(command, prompt, cwd, env):
            observed["command"] = list(command)
            Path(command[command.index("-o") + 1]).write_text("OK", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="OK\n", stderr="")

        audit = Path(self.temp.name) / "audit"
        with patch.dict(os.environ, {"RRR_HOST_AUDIT": "1", "RRR_HOST_AUDIT_DIR": str(audit)}), \
                patch.object(host_backend, "_command_candidates", return_value=["codex"]), \
                patch.object(host_backend, "_run_process", side_effect=fake_run), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CODEX_AUTH):
            host_backend.host_chat(messages=[{"role": "user", "content": "x"}])
        command = observed["command"]
        setting = 'model_reasoning_effort="xhigh"'
        self.assertEqual("-c", command[command.index(setting) - 1])
        self.assertEqual("-", command[-1])
        records = [json.loads(path.read_text(encoding="utf-8")) for path in audit.glob("*.json")]
        self.assertEqual(
            [("gpt-6.2-astra", "xhigh")],
            [(record["requested_model"], record["requested_effort"]) for record in records],
        )

    def test_the_claude_command_names_the_inherited_model_and_loads_no_settings(self):
        self.write_settings({"model": "opus[1m]"})
        observed = {}

        def fake_run(command, prompt, cwd, env):
            observed["command"] = list(command)
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"result": "OK", "is_error": False}), stderr="",
            )

        with patch.dict(os.environ, {"RRR_HOST_AUDIT": "0", "RRR_HOST": "claude"}), \
                patch.object(host_backend, "_command_candidates", return_value=["claude"]), \
                patch.object(host_backend, "_run_process", side_effect=fake_run), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CLAUDE_AUTH):
            host_backend.host_chat(model="mistral-small:24b",
                                   messages=[{"role": "user", "content": "x"}])
        command = observed["command"]
        self.assertEqual("opus[1m]", command[command.index("--model") + 1])
        self.assertEqual("", command[command.index("--setting-sources") + 1])
        self.assertNotIn("--effort", command)
        self.write_settings({"model": "sonnet", "effortLevel": "high"})
        with patch.dict(os.environ, {"RRR_HOST_AUDIT": "0", "RRR_HOST": "claude"}), \
                patch.object(host_backend, "_command_candidates", return_value=["claude"]), \
                patch.object(host_backend, "_run_process", side_effect=fake_run), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CLAUDE_AUTH):
            host_backend.host_chat(messages=[{"role": "user", "content": "x"}])
        command = observed["command"]
        self.assertEqual("sonnet", command[command.index("--model") + 1])
        self.assertEqual("high", command[command.index("--effort") + 1])

    def test_host_doctor_reports_the_model_the_effort_and_their_sources(self):
        self.write_config('model = "gpt-6.2-astra"\nmodel_reasoning_effort = "xhigh"\n')
        with patch.object(host_backend, "_command_candidates", return_value=["codex"]), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CODEX_AUTH):
            status = host_backend.host_diagnostics()
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                host_doctor_main(["--host", "codex"])
            as_json = io.StringIO()
            with contextlib.redirect_stdout(as_json):
                host_doctor_main(["--host", "codex", "--json"])
        self.assertEqual("gpt-6.2-astra", status["model"])
        self.assertIn("the Codex configuration", status["model_source"])
        self.assertIn("[RRR] model: gpt-6.2-astra (from the Codex configuration, ", console.getvalue())
        self.assertIn("[RRR] effort: xhigh (from the Codex configuration, ", console.getvalue())
        report = json.loads(as_json.getvalue())
        self.assertEqual(("gpt-6.2-astra", "xhigh"), (report["model"], report["effort"]))
        self.assertIn("model_source", report)
        self.assertIn("effort_source", report)

    def test_host_doctor_names_the_environment_variable_when_it_decides(self):
        self.write_config('model = "gpt-6.2-astra"\n')
        console = io.StringIO()
        with patch.object(host_backend, "_command_candidates", return_value=["codex"]), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CODEX_AUTH), \
                patch.dict(os.environ, {"RRR_HOST_MODEL": "gpt-6.1-terra"}), \
                contextlib.redirect_stdout(console):
            host_doctor_main(["--host", "codex"])
        self.assertIn("[RRR] model: gpt-6.1-terra (from RRR_HOST_MODEL)", console.getvalue())
        self.assertIn(
            "[RRR] effort: the model's default (RRR_HOST_MODEL pins the model and "
            "RRR_HOST_EFFORT is not set)",
            console.getvalue(),
        )
        console = io.StringIO()
        with patch.object(host_backend, "_command_candidates", return_value=["codex"]), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CODEX_AUTH), \
                contextlib.redirect_stdout(console):
            host_doctor_main(["--host", "codex", "--model", "gpt-6.1-terra", "--effort", "low"])
        self.assertIn("[RRR] effort: low (from RRR_HOST_EFFORT)", console.getvalue())
        os.environ.pop("RRR_HOST_MODEL", None)
        os.environ.pop("RRR_HOST_EFFORT", None)

    def test_host_doctor_says_when_the_fallback_runs(self):
        console = io.StringIO()
        with patch.object(host_backend, "_command_candidates", return_value=["claude"]), \
                patch.object(host_backend, "_verify_subscription_auth", return_value=self.CLAUDE_AUTH), \
                contextlib.redirect_stdout(console):
            host_doctor_main(["--host", "claude"])
        self.assertIn(
            "[RRR] model: claude-opus-5-5 (from RRR fallback; the model selected in "
            "Claude Code could not be established)",
            console.getvalue(),
        )


class SkillTextTests(unittest.TestCase):
    def test_the_skill_describes_inheritance_and_the_two_fallbacks(self):
        root = Path(__file__).resolve().parents[2]
        for relative in (".agents/skills/rrr/SKILL.md", "plugins/rrr/skills/rrr/SKILL.md",
                         "skills/rrr/SKILL.md"):
            text = " ".join((root / relative).read_text(encoding="utf-8").split())
            self.assertIn(
                "A run inherits the model and the reasoning effort the user has selected in Codex",
                text,
            )
            self.assertIn(
                "A run inherits the model and the effort the user has selected in Claude Code", text,
            )
            self.assertIn(
                "`gpt-6.2-sol` at its default effort runs only when RRR finds no selection", text,
            )
            self.assertIn(
                "`claude-opus-5-5` at its default effort runs only when RRR finds no selection", text,
            )
            self.assertIn("`RRR_HOST_EFFORT`", text)
            self.assertNotIn("newest Sol-tier", text)
            self.assertNotIn("The default host model is", text)
            self.assertIn("GPT-5.5 or GPT-5.6 model", text)


if __name__ == "__main__":
    unittest.main()
