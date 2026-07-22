import copy
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rrr import api_backend
from rrr import cli
from rrr import language


class _FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _openai_module(response):
    responses = _FakeResponses(response)
    client = SimpleNamespace(responses=responses)
    module = SimpleNamespace(OpenAI=lambda: client)
    return module, responses


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _anthropic_module(response):
    messages = _FakeMessages(response)
    client = SimpleNamespace(messages=messages)
    module = SimpleNamespace(Anthropic=lambda: client)
    return module, messages


class ProviderBackendTests(unittest.TestCase):
    def test_openai_responses_prose_preserves_contract(self):
        response = SimpleNamespace(
            status="completed",
            output_text="Detta är ett svenskt svar.",
            output=[],
        )
        module, responses = _openai_module(response)
        messages = [
            {
                "role": "system",
                "content": "Write only from supplied evidence.",
            },
            {"role": "user", "content": "Respond in Swedish."},
        ]
        original = copy.deepcopy(messages)

        with (
            patch.dict(sys.modules, {"openai": module}),
            patch.dict(
                os.environ,
                {
                    "RRR_API_PROVIDER": "openai",
                    "RRR_API_MODEL": "gpt-5.6-sol",
                },
                clear=False,
            ),
        ):
            result = api_backend._openai_chat(
                "ignored",
                messages,
                {"temperature": 0.3, "num_predict": 1500},
                None,
            )

        self.assertEqual(
            {"message": {"content": "Detta är ett svenskt svar."}},
            result,
        )
        request = responses.calls[0]
        self.assertEqual("gpt-5.6-sol", request["model"])
        self.assertEqual(messages, request["input"])
        self.assertEqual(1500, request["max_output_tokens"])
        self.assertEqual(0.3, request["temperature"])
        self.assertEqual({"effort": "none"}, request["reasoning"])
        self.assertFalse(request["store"])
        self.assertNotIn("text", request)
        self.assertNotIn("messages", request)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("response_format", request)
        self.assertEqual(original, messages)

    def test_openai_responses_json_preserves_json_string(self):
        output = '{"probes":["institutioner"],"exclusions":[]}'
        response = SimpleNamespace(
            status="completed",
            output_text=output,
            output=[],
        )
        module, responses = _openai_module(response)
        messages = [
            {
                "role": "user",
                "content": "Return a JSON object with probes.",
            }
        ]
        original = copy.deepcopy(messages)

        with (
            patch.dict(sys.modules, {"openai": module}),
            patch.dict(
                os.environ,
                {
                    "RRR_API_PROVIDER": "openai",
                    "RRR_API_MODEL": "gpt-5.6-sol",
                },
                clear=False,
            ),
        ):
            result = api_backend._openai_chat(
                "ignored",
                messages,
                {"temperature": 0.0, "num_predict": 400},
                "json",
            )

        self.assertEqual(output, result["message"]["content"])
        self.assertEqual(
            {"probes": ["institutioner"], "exclusions": []},
            json.loads(result["message"]["content"]),
        )
        request = responses.calls[0]
        self.assertEqual(
            {"format": {"type": "json_object"}},
            request["text"],
        )
        self.assertEqual(400, request["max_output_tokens"])
        self.assertEqual({"effort": "none"}, request["reasoning"])
        self.assertTrue(
            request["input"][-1]["content"].endswith(
                api_backend._json_nudge("json")
            )
        )
        self.assertEqual(original, messages)

    def test_openai_empty_output_returns_empty_content(self):
        module, _responses = _openai_module(
            SimpleNamespace(status="completed", output_text=None, output=[])
        )
        with patch.dict(sys.modules, {"openai": module}):
            result = api_backend._openai_chat(
                "ignored",
                [{"role": "user", "content": "Answer."}],
                {},
                None,
            )
        self.assertEqual({"message": {"content": ""}}, result)

    def test_openai_refusal_uses_existing_failure_contract(self):
        refusal = SimpleNamespace(
            status="completed",
            output_text="Policy refusal",
            output=[
                SimpleNamespace(
                    content=[SimpleNamespace(type="refusal")]
                )
            ],
        )
        module, _responses = _openai_module(refusal)
        with patch.dict(sys.modules, {"openai": module}):
            result = api_backend._openai_chat(
                "ignored",
                [{"role": "user", "content": "Answer."}],
                {},
                None,
            )
        self.assertEqual("", result["message"]["content"])
        self.assertTrue(result["_rrr_refusal"])

    def test_openai_incomplete_response_fails_closed(self):
        incomplete = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output_text="partial text must not be released",
            output=[],
        )
        module, _responses = _openai_module(incomplete)
        with patch.dict(sys.modules, {"openai": module}):
            with self.assertRaisesRegex(
                RuntimeError,
                "incomplete.*max_output_tokens",
            ):
                api_backend._openai_chat(
                    "ignored",
                    [{"role": "user", "content": "Answer."}],
                    {},
                    None,
                )

    def test_openai_missing_status_fails_closed(self):
        response = SimpleNamespace(
            output_text="unverified output",
            output=[],
        )
        module, _responses = _openai_module(response)
        with patch.dict(sys.modules, {"openai": module}):
            with self.assertRaisesRegex(RuntimeError, "status None"):
                api_backend._openai_chat(
                    "ignored",
                    [{"role": "user", "content": "Answer."}],
                    {},
                    None,
                )

    def test_gpt_5_5_override_keeps_reasoning_explicit(self):
        response = SimpleNamespace(
            status="completed",
            output_text="ok",
            output=[],
        )
        module, responses = _openai_module(response)
        with (
            patch.dict(sys.modules, {"openai": module}),
            patch.dict(
                os.environ,
                {"RRR_API_MODEL": "gpt-5.5"},
                clear=False,
            ),
        ):
            api_backend._openai_chat(
                "ignored",
                [{"role": "user", "content": "Answer."}],
                {},
                None,
            )
        self.assertEqual(
            {"effort": "none"},
            responses.calls[0]["reasoning"],
        )

    def test_response_storage_can_be_enabled_explicitly(self):
        response = SimpleNamespace(
            status="completed",
            output_text="ok",
            output=[],
        )
        module, responses = _openai_module(response)
        with (
            patch.dict(sys.modules, {"openai": module}),
            patch.dict(
                os.environ,
                {"RRR_API_STORE": "1"},
                clear=False,
            ),
        ):
            api_backend._openai_chat(
                "ignored",
                [{"role": "user", "content": "Answer."}],
                {},
                None,
            )
        self.assertTrue(responses.calls[0]["store"])

    def test_unsupported_openai_override_is_rejected(self):
        with patch.dict(
            os.environ,
            {
                "RRR_API_PROVIDER": "openai",
                "RRR_API_MODEL": "gpt-4.1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "GPT-5.6 and GPT-5.5"):
                api_backend.api_model_name()

    def test_anthropic_completed_response_preserves_contract(self):
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Complete answer")],
        )
        module, messages = _anthropic_module(response)
        with (
            patch.dict(sys.modules, {"anthropic": module}),
            patch.dict(
                os.environ,
                {"RRR_API_MODEL": "claude-opus-4-8"},
                clear=False,
            ),
        ):
            result = api_backend._anthropic_chat(
                "ignored",
                [
                    {"role": "system", "content": "Use the evidence."},
                    {"role": "user", "content": "Answer."},
                ],
                {"num_predict": 120},
                None,
            )
        self.assertEqual(
            {"message": {"content": "Complete answer"}},
            result,
        )
        self.assertEqual(120, messages.calls[0]["max_tokens"])
        self.assertEqual("Use the evidence.", messages.calls[0]["system"])

    def test_anthropic_max_tokens_fails_closed(self):
        response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(type="text", text="partial")],
        )
        module, _messages = _anthropic_module(response)
        with patch.dict(sys.modules, {"anthropic": module}):
            with self.assertRaisesRegex(RuntimeError, "max_tokens"):
                api_backend._anthropic_chat(
                    "ignored",
                    [{"role": "user", "content": "Answer."}],
                    {"num_predict": 120},
                    None,
                )

    def test_anthropic_refusal_uses_existing_failure_contract(self):
        response = SimpleNamespace(stop_reason="refusal", content=[])
        module, _messages = _anthropic_module(response)
        with patch.dict(sys.modules, {"anthropic": module}):
            result = api_backend._anthropic_chat(
                "ignored",
                [{"role": "user", "content": "Answer."}],
                {},
                None,
            )
        self.assertEqual("", result["message"]["content"])
        self.assertTrue(result["_rrr_refusal"])

    def test_api_model_name_reports_new_openai_default(self):
        env = {"RRR_API_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual("gpt-5.6-sol", api_backend.api_model_name())

    def test_unknown_provider_is_rejected(self):
        with patch.dict(
            os.environ,
            {"RRR_API_PROVIDER": "typo"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "RRR_API_PROVIDER"):
                api_backend.api_chat(messages=[])
            with self.assertRaisesRegex(ValueError, "RRR_API_PROVIDER"):
                api_backend.api_model_name()

    def test_invalid_api_provider_cannot_fall_back_to_local_model(self):
        with patch.dict(
            os.environ,
            {"RRR_RUNTIME": "api", "RRR_API_PROVIDER": "typo"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "RRR_API_PROVIDER"):
                language.select_model("en")
            with self.assertRaisesRegex(ValueError, "RRR_API_PROVIDER"):
                cli._select_topic_runtime("economic growth and institutions")

    def test_language_router_bypasses_ollama_in_api_mode(self):
        with (
            patch.dict(
                os.environ,
                {"RRR_RUNTIME": "api", "RRR_API_PROVIDER": "openai"},
                clear=True,
            ),
            patch.object(
                language,
                "_list_ollama_models",
                side_effect=AssertionError("Ollama registry must not be read"),
            ),
        ):
            self.assertEqual("gpt-5.6-sol", language.select_model("en"))
            self.assertEqual("gpt-5.6-sol", language.select_model("zh"))
            self.assertEqual(
                "Respond in Chinese.",
                language.language_directive("zh"),
            )

    def test_llm_shim_routes_api_calls(self):
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
                    {"RRR_RUNTIME": "api"},
                    clear=False,
                ),
                patch(
                    "rrr.api_backend.api_chat",
                    return_value={"message": {"content": "api"}},
                ) as routed,
            ):
                result = fake_ollama.chat(
                    model="ignored",
                    messages=[{"role": "user", "content": "hello"}],
                )
        self.assertEqual("api", result["message"]["content"])
        routed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
