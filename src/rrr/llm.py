"""v15.17: central runtime router behind the ollama.chat interface.

qwen3 (our non-Latin routing tier) ships with "thinking" ON by default: the
model emits a long `<think>…</think>` reasoning trace into a separate channel
BEFORE the answer. With the small `num_predict` budgets the RRR stages use
(300-700 tokens), the thinking trace consumes the entire budget and the
actual `content` comes back EMPTY — which made every qwen3 stage fall to its
failure path (planner → heuristic_fallback, precheck → stage0_llm_failed,
writer → empty prose).

Rather than thread a `think=False` kwarg through all ~30 ollama.chat call
sites, we patch `ollama.chat` once. Every `import ollama` in the package
resolves to the same cached module object, so patching `ollama.chat` here —
imported once at reasoner entry — covers all call sites (planner, precheck,
cluster, posture, order, claim extraction, writer, prewarm, ingest).

The patch is idempotent and version-tolerant: if the installed ollama client
predates the `think=` kwarg it falls back to injecting a `/no_think` soft
switch into the last user message (qwen3 recognises it in-band).
"""
from __future__ import annotations

import os


# Substrings that identify a hybrid-thinking model whose reasoning trace must
# be suppressed for RRR's short-budget structured stages. Extendable via
# RRR_THINKING_MODELS (comma-separated substrings) without a code change.
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
    """Append ' /no_think' to the last user message so qwen3 disables
    thinking in-band (fallback for ollama clients without the think kwarg)."""
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
    """Idempotently patch ollama.chat to disable thinking for thinking models.
    Safe to call multiple times. Returns True if the patch is active, False
    if ollama isn't importable (e.g. local dev without the package)."""
    try:
        import ollama
    except Exception:
        return False

    if getattr(ollama.chat, "_rrr_thinking_shim", False):
        return True

    _original = ollama.chat

    # v15.14: request timeout. No ollama.chat call site passes one, so a hung
    # server stalled a run indefinitely (30m keep_alive, sequential writer).
    # Route every local call through a timeout-configured Client when the
    # installed client supports it. RRR_OLLAMA_TIMEOUT seconds, default 600;
    # 0 disables. Read at install time — the CLI sets its env before the
    # reasoner import that triggers install(), same contract as RRR_MODEL.
    try:
        _timeout_s = float(os.environ.get("RRR_OLLAMA_TIMEOUT", "600") or 0)
    except ValueError:
        _timeout_s = 600.0
    _base_chat = _original
    if _timeout_s > 0:
        try:
            _base_chat = ollama.Client(timeout=_timeout_s).chat
        except Exception:
            _base_chat = _original  # older client without timeout kwarg

    def _patched(*args, **kwargs):
        # API and host runtimes preserve the ollama.chat call contract while
        # changing only the model transport. Positional args map to
        # ollama.chat's (model, messages).
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
            # Some qwen3/Ollama combinations accept ``think=False`` yet still
            # emit a reasoning trace in the content channel. Apply qwen3's
            # in-band control as well as the transport flag. Copy both the
            # kwargs and messages so callers never observe prompt mutation.
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
                # Older ollama client: no think kwarg. Fall back to the
                # in-band /no_think switch on the messages.
                if "messages" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["messages"] = _inject_no_think(kwargs["messages"])
                return _base_chat(*args, **kwargs)
        return _base_chat(*args, **kwargs)

    _patched._rrr_thinking_shim = True
    _patched._rrr_original = _original
    ollama.chat = _patched
    return True
