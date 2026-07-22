"""Provider-API backend behind the ``ollama.chat`` interface.

When ``RRR_RUNTIME=api``, ``rrr.llm`` routes every model call through a
separate Anthropic or OpenAI API account. A skill can select this runtime,
local Ollama, or the subscription-backed host adapter.

Provider models do not use RRR's local Mistral and Qwen routing. Retrieval
probes still follow the corpus language, and the writer follows the language
of the topic.

The adapter translates an Ollama-style chat call into the provider SDK call
and returns Ollama's response shape,
``{"message": {"content": "<text>"}}``, so all ~30 existing call sites work
unchanged.

Provider is chosen by RRR_API_PROVIDER (anthropic | openai, default
anthropic). Model by RRR_API_MODEL (default claude-opus-4-8 for Anthropic,
gpt-5.6-sol for OpenAI). Credentials resolve through the provider SDK's
normal chain (ANTHROPIC_API_KEY or OPENAI_API_KEY). We never
take a key as a parameter. OpenAI calls use the Responses API and disable
application-state response storage unless RRR_API_STORE is explicitly
enabled. Provider retention and abuse-monitoring policies still apply.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"
_OPENAI_DEFAULT_MODEL = "gpt-5.6-sol"


def _api_provider() -> str:
    provider = os.environ.get("RRR_API_PROVIDER", "anthropic").strip().lower()
    if provider not in {"anthropic", "openai"}:
        raise ValueError(
            "RRR_API_PROVIDER must be 'anthropic' or 'openai'; "
            f"received {provider!r}"
        )
    return provider


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
    """Map Ollama's num_predict to the provider output-token limit."""
    n = None
    if isinstance(options, dict):
        n = options.get("num_predict")
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = default
    if n <= 0:
        n = default
    return max(1, min(n, 16000))


def _json_nudge(fmt: Optional[str]) -> str:
    """RRR passes format='json' on structured stages. Frontier models don't
    need Ollama's format arg, but a short reinforcement keeps them from
    wrapping the object in prose or markdown fences. RRR's own prompts
    already demand JSON, so this is belt-and-suspenders."""
    if fmt == "json":
        return ("\n\nReturn ONLY the requested JSON object. No prose, no "
                "explanation, no markdown code fences.")
    return ""


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _openai_model_name() -> str:
    """Return a model whose parameter contract RRR supports explicitly."""
    model = os.environ.get("RRR_API_MODEL", _OPENAI_DEFAULT_MODEL).strip()
    lowered = model.lower()
    supported_family = lowered.startswith("gpt-5.6") or lowered.startswith(
        "gpt-5.5"
    )
    if not supported_family or "-pro" in lowered:
        raise ValueError(
            "The OpenAI adapter currently supports standard GPT-5.6 and "
            "GPT-5.5 models. Choose one of those models with RRR_API_MODEL."
        )
    return model


def _openai_refused(resp: Any) -> bool:
    """Return True when a Responses result contains a refusal block."""
    for item in getattr(resp, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "refusal":
                return True
    return False


def _anthropic_chat(model: str, messages: List[dict], options: Optional[dict],
                    fmt: Optional[str]) -> Dict[str, Any]:
    import anthropic

    api_model = os.environ.get("RRR_API_MODEL", _ANTHROPIC_DEFAULT_MODEL)
    system, rest = _split_system(messages)
    system = (system + _json_nudge(fmt)).strip()

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY
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

    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason != "end_turn":
        raise RuntimeError(
            "Anthropic Messages call did not complete cleanly; "
            f"stop_reason={stop_reason!r}"
        )

    text = "".join(
        getattr(b, "text", "") for b in (resp.content or [])
        if getattr(b, "type", None) == "text"
    )
    return {"message": {"content": text}}


def _openai_chat(model: str, messages: List[dict], options: Optional[dict],
                 fmt: Optional[str]) -> Dict[str, Any]:
    import openai

    api_model = _openai_model_name()
    client = openai.OpenAI()  # resolves OPENAI_API_KEY

    # Responses accepts the existing role-based message list. Work on a copy
    # because JSON reinforcement must never mutate the caller's prompt record.
    msgs = list(messages or [])
    if fmt == "json" and msgs:
        # Reinforce on the last user message for models that need an explicit
        # JSON instruction.
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i] = dict(msgs[i])
                msgs[i]["content"] = (msgs[i].get("content") or "") + _json_nudge(fmt)
                break

    kwargs: Dict[str, Any] = {
        "model": api_model,
        "input": msgs,
        "max_output_tokens": _max_tokens_from_options(options),
        "store": _env_flag("RRR_API_STORE", False),
    }
    # Supported GPT-5.5 and GPT-5.6 models default to medium reasoning. RRR's
    # former GPT-4o route had no hidden reasoning budget, so none preserves
    # that baseline and keeps call costs explicit.
    kwargs["reasoning"] = {"effort": "none"}
    temp = (options or {}).get("temperature")
    if temp is not None:
        kwargs["temperature"] = float(temp)
    if fmt == "json":
        # The central shim does not know each stage's schema. JSON-object mode
        # preserves the current contract while the stage parsers validate the
        # required keys and values.
        kwargs["text"] = {"format": {"type": "json_object"}}

    resp = client.responses.create(**kwargs)
    status = getattr(resp, "status", None)
    if status != "completed":
        details = getattr(resp, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(
            f"OpenAI Responses call ended with status {status!r}{suffix}"
        )
    text = getattr(resp, "output_text", "") or ""
    if _openai_refused(resp):
        return {"message": {"content": ""}, "_rrr_refusal": True}
    return {"message": {"content": text}}


def api_chat(model: str = "", messages: Optional[List[dict]] = None,
             options: Optional[dict] = None, format: Optional[str] = None,
             **_ignored) -> Dict[str, Any]:
    """ollama.chat-compatible entrypoint for the provider API runtime.
    Extra ollama kwargs (keep_alive, stream, etc.) are accepted and ignored.
    Returns {"message": {"content": <str>}}.
    """
    provider = _api_provider()
    if provider == "openai":
        return _openai_chat(model, messages or [], options, format)
    if provider == "anthropic":
        return _anthropic_chat(model, messages or [], options, format)
    raise AssertionError(f"unreachable provider: {provider}")


def api_model_name() -> str:
    """The model string select_model() reports in RRR_MODEL / metrics when
    running in API mode."""
    provider = _api_provider()
    default = _OPENAI_DEFAULT_MODEL if provider == "openai" else _ANTHROPIC_DEFAULT_MODEL
    if provider == "openai":
        return _openai_model_name()
    return os.environ.get("RRR_API_MODEL", default)
