from __future__ import annotations

import os


_DEFAULT_MARKERS = ("qwen3", "deepseek-r1", "r1")


def _markers() -> tuple:
    extra = os.environ.get("RRR_THINKING_MODELS", "").strip()
    if extra:
        return tuple(m.strip().lower() for m in extra.split(",") if m.strip())
    return _DEFAULT_MARKERS


def _is_thinking_model(model: str) -> bool:
    m = (model or "").lower()
    return any(marker in m for marker in _markers())


def _inject_no_think(messages):
    if not messages or not isinstance(messages, list):
        return messages
    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content") or ""
            if "/no_think" not in content:
                m["content"] = content + " /no_think"
            break
    return out


def install():
    try:
        import ollama
    except Exception:
        return False

    if getattr(ollama.chat, "_rrr_thinking_shim", False):
        return True

    _original = ollama.chat

    try:
        _timeout_s = float(os.environ.get("RRR_OLLAMA_TIMEOUT", "600") or 0)
    except ValueError:
        _timeout_s = 600.0
    _base_chat = _original
    if _timeout_s > 0:
        try:
            _base_chat = ollama.Client(timeout=_timeout_s).chat
        except Exception:
            _base_chat = _original

    def _patched(*args, **kwargs):
        runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
        if runtime == "api":
            from rrr.api_backend import api_chat
            if args:
                kwargs.setdefault("model", args[0] if len(args) > 0 else "")
                if len(args) > 1:
                    kwargs.setdefault("messages", args[1])
            return api_chat(**kwargs)
        if runtime == "host":
            from rrr.host_backend import host_chat
            if args:
                kwargs.setdefault("model", args[0] if len(args) > 0 else "")
                if len(args) > 1:
                    kwargs.setdefault("messages", args[1])
            return host_chat(**kwargs)
        if runtime not in {"", "local", "ollama"}:
            raise RuntimeError(
                "RRR_RUNTIME must be unset, 'local', 'ollama', 'api', or "
                f"'host'; received {runtime!r}"
            )

        model = kwargs.get("model") or (args[0] if args else "")
        kwargs.pop("_rrr_stage", None)
        if _is_thinking_model(model) and "think" not in kwargs:
            if "qwen3" in str(model).lower():
                kwargs = dict(kwargs)
                if "messages" in kwargs:
                    kwargs["messages"] = _inject_no_think(kwargs["messages"])
                elif len(args) > 1:
                    positional = list(args)
                    positional[1] = _inject_no_think(positional[1])
                    args = tuple(positional)
            try:
                return _base_chat(*args, think=False, **kwargs)
            except TypeError:
                if "messages" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["messages"] = _inject_no_think(kwargs["messages"])
                return _base_chat(*args, **kwargs)
        return _base_chat(*args, **kwargs)

    _patched._rrr_thinking_shim = True
    _patched._rrr_original = _original
    ollama.chat = _patched
    return True
