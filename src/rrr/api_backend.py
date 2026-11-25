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
    if fmt == "json":
        return ("\n\nReturn ONLY the requested JSON object. No prose, no "
                "explanation, no markdown code fences.")
    return ""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _openai_model_name() -> str:
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

    client = anthropic.Anthropic()
    kwargs: Dict[str, Any] = {
        "model": api_model,
        "max_tokens": _max_tokens_from_options(options),
        "messages": rest,
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)

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
    client = openai.OpenAI()

    msgs = list(messages or [])
    if fmt == "json" and msgs:
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
    kwargs["reasoning"] = {"effort": "none"}
    temp = (options or {}).get("temperature")
    if temp is not None:
        kwargs["temperature"] = float(temp)
    if fmt == "json":
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
    provider = _api_provider()
    if provider == "openai":
        return _openai_chat(model, messages or [], options, format)
    if provider == "anthropic":
        return _anthropic_chat(model, messages or [], options, format)
    raise AssertionError(f"unreachable provider: {provider}")


def api_model_name() -> str:
    provider = _api_provider()
    default = _OPENAI_DEFAULT_MODEL if provider == "openai" else _ANTHROPIC_DEFAULT_MODEL
    if provider == "openai":
        return _openai_model_name()
    return os.environ.get("RRR_API_MODEL", default)
