"""v15.13: frontier-API backend behind the ollama.chat interface.

When RRR_RUNTIME=api, rrr/llm.py routes every ollama.chat call here instead
of to a local Ollama model. This is the "desktop mode": running RRR as a
Codex / Claude Code skill (or anywhere the user has a frontier API key)
uses the host's Anthropic or OpenAI model instead of a local GPU.

The whole local-model apparatus (model routing between mistral/qwen, the
qwen3 thinking-mode shim, cross-language probe translation) becomes moot on
a frontier model — one model handles every language natively. The pivot
architecture still holds and only gets better: retrieval probes are still
emitted in the corpus language, and the writer still produces the topic
language, both of which a frontier model does effortlessly.

Design: translate Ollama's chat call — (model, messages, options, format) —
into the provider SDK call and return Ollama's response shape,
``{"message": {"content": "<text>"}}``, so all ~30 existing call sites work
unchanged.

Provider is chosen by RRR_API_PROVIDER (anthropic | openai, default
anthropic). Model by RRR_API_MODEL (default claude-opus-4-8 for anthropic,
gpt-4o for openai). Credentials resolve through the provider SDK's normal
chain (ANTHROPIC_API_KEY / `ant` profile; OPENAI_API_KEY) — we never take a
key as a parameter.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"
_OPENAI_DEFAULT_MODEL = "gpt-4o"


def _split_system(messages: List[dict]) -> Tuple[str, List[dict]]:
    """Anthropic takes the system prompt as a separate top-level arg, not a
    message. Pull every role=='system' message out, join them, and return
    (system_text, remaining_messages). The remaining messages keep their
    order and start with a user turn (RRR only ever puts system first)."""
    system_parts: List[str] = []
    rest: List[dict] = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
        else:
            rest.append(m)
    return "\n\n".join(system_parts), rest


def _max_tokens_from_options(options: Optional[dict], default: int = 8000) -> int:
    """Map Ollama's num_predict to the API's required max_tokens. Clamp to a
    sane floor/ceiling; RRR's largest stage (the writer) asks for ~8000, well
    under the non-streaming timeout ceiling."""
    n = None
    if isinstance(options, dict):
        n = options.get("num_predict")
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = default
    if n <= 0:
        n = default
    return max(1024, min(n, 16000))


def _json_nudge(fmt: Optional[str]) -> str:
    """RRR passes format='json' on structured stages. Frontier models don't
    need Ollama's format arg, but a short reinforcement keeps them from
    wrapping the object in prose or markdown fences. RRR's own prompts
    already demand JSON, so this is belt-and-suspenders."""
    if fmt == "json":
        return ("\n\nReturn ONLY the requested JSON object. No prose, no "
                "explanation, no markdown code fences.")
    return ""


def _anthropic_chat(model: str, messages: List[dict], options: Optional[dict],
                    fmt: Optional[str]) -> Dict[str, Any]:
    import anthropic

    api_model = os.environ.get("RRR_API_MODEL", _ANTHROPIC_DEFAULT_MODEL)
    system, rest = _split_system(messages)
    system = (system + _json_nudge(fmt)).strip()

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile
    # NB: opus-4.x reject temperature/top_p/top_k with a 400 — never pass
    # sampling params here. Thinking is left off (these are short structured
    # or single-pass prose calls; omitting `thinking` runs without it on
    # 4.7/4.8). max_tokens is required.
    kwargs: Dict[str, Any] = {
        "model": api_model,
        "max_tokens": _max_tokens_from_options(options),
        "messages": rest,
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)

    # A safety refusal (opus-4.x can return stop_reason='refusal' on an
    # HTTP 200) yields empty/partial content — surface empty text so RRR's
    # per-call failure handling takes over rather than crashing on an index.
    if getattr(resp, "stop_reason", None) == "refusal":
        return {"message": {"content": ""}, "_rrr_refusal": True}

    text = "".join(
        getattr(b, "text", "") for b in (resp.content or [])
        if getattr(b, "type", None) == "text"
    )
    return {"message": {"content": text}}


def _openai_chat(model: str, messages: List[dict], options: Optional[dict],
                 fmt: Optional[str]) -> Dict[str, Any]:
    import openai

    api_model = os.environ.get("RRR_API_MODEL", _OPENAI_DEFAULT_MODEL)
    client = openai.OpenAI()  # resolves OPENAI_API_KEY

    # OpenAI keeps system as a normal message, so pass messages through. If a
    # JSON stage, add the response_format hint (requires the literal word
    # "json" somewhere in the messages — RRR's prompts always contain it).
    msgs = list(messages or [])
    if fmt == "json" and msgs:
        # Reinforce on the last user message for models that ignore
        # response_format, harmless otherwise.
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i] = dict(msgs[i])
                msgs[i]["content"] = (msgs[i].get("content") or "") + _json_nudge(fmt)
                break

    kwargs: Dict[str, Any] = {
        "model": api_model,
        "messages": msgs,
        "max_tokens": _max_tokens_from_options(options),
    }
    temp = (options or {}).get("temperature")
    if temp is not None:
        kwargs["temperature"] = float(temp)
    if fmt == "json":
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    return {"message": {"content": text}}


def api_chat(model: str = "", messages: Optional[List[dict]] = None,
             options: Optional[dict] = None, format: Optional[str] = None,
             **_ignored) -> Dict[str, Any]:
    """ollama.chat-compatible entrypoint for the frontier-API runtime.
    Extra ollama kwargs (keep_alive, stream, etc.) are accepted and ignored.
    Returns {"message": {"content": <str>}}.
    """
    provider = os.environ.get("RRR_API_PROVIDER", "anthropic").strip().lower()
    if provider == "openai":
        return _openai_chat(model, messages or [], options, format)
    return _anthropic_chat(model, messages or [], options, format)


def api_model_name() -> str:
    """The model string select_model() reports in RRR_MODEL / metrics when
    running in API mode."""
    provider = os.environ.get("RRR_API_PROVIDER", "anthropic").strip().lower()
    default = _OPENAI_DEFAULT_MODEL if provider == "openai" else _ANTHROPIC_DEFAULT_MODEL
    return os.environ.get("RRR_API_MODEL", default)
