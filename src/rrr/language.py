"""v15.11 language detection + model routing.

Two responsibilities:

1. detect_topic_language(text) — returns an ISO-639-1 code (or "unknown")
   for a user-provided topic. Uses langdetect with a deterministic seed so
   the same topic always resolves the same way (langdetect's Bayesian
   detector is otherwise non-deterministic per-process).

2. select_model(lang) returns the configured runtime model. Local Ollama uses
   language-aware Mistral or Qwen routing. Provider API and subscription-host
   modes return their explicitly configured model without consulting Ollama.

Both env vars are overridable so operators without the qwen model
installed can point RRR_MODEL_NONLATIN back at RRR_MODEL_LATIN. When the
routed model isn't in the local ollama registry, select_model falls back
to RRR_MODEL_LATIN with a stderr warning; a cold registry (no models
pulled at all) falls back to the RRR_MODEL / mistral-small:24b chain that the
pipeline was built on.

The whole module is import-safe — no LLM calls, no network. Both
functions must complete in <10ms on realistic inputs.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Model tiers (local Ollama runtime). Overridable via RRR_MODEL_LATIN /
# RRR_MODEL_NONLATIN; the constants below are the recommended defaults and
# the vetted alternatives, so the routing code itself documents what to run.
#
#   Latin (European) tier — RRR_MODEL_LATIN, default:
#       mistral-small:24b   dense 24B, ~15 GB q4, fits RTX 4090. Near-English
#                           quality on FR/PT/ES/DE/IT prose + JSON compliance.
#
#   Non-Latin tier — RRR_MODEL_NONLATIN, default:
#       qwen3:14b           dense 14B, ~9 GB q4, fits RTX 4090. PROVEN on the
#                           ZH cross-language smoke (49 docs admitted,
#                           ~90% Chinese review). The shipped default.
#
#   Non-Latin PREMIUM (dense, RTX 5090) — set RRR_MODEL_NONLATIN=qwen3:32b:
#       qwen3:32b           dense 32B, ~20 GB q4. The quality upgrade for the
#                           RTX 5090. Dense (not MoE), so it does NOT leak
#                           reasoning into content the way the a3b MoE does —
#                           clean JSON on RRR's tight-budget structured stages.
#                           Already in scripts/pod_pull_models.sh.
#
#   NOT RECOMMENDED:
#       qwen3:30b-a3b       30B MoE / 3B active. Verified (v15.13) to leak
#                           chain-of-thought into the *content* channel even
#                           with think=False, exhausting num_predict before
#                           emitting JSON → planner/precheck fail → refusal.
#                           Would need num_predict 2-3x across every structured
#                           stage. Use the dense qwen3:14b (4090) or
#                           qwen3:32b (5090) instead.
#
# All qwen3 variants are covered by the thinking-mode shim (rrr/llm.py,
# "qwen3" substring). API and subscription-host runtimes bypass this table
# because one configured model handles every language.
# ---------------------------------------------------------------------------
MODEL_LATIN_DEFAULT = "mistral-small:24b"
MODEL_NONLATIN_DEFAULT = "qwen3:14b"          # dense, RTX 4090, shipped default


# Latin-script European languages where mistral-small:24b handles
# prompt-following, JSON compliance, and prose generation at near-English
# quality. Everything else falls to the non-Latin router.
LATIN_LANGS = frozenset({
    "en", "fr", "es", "pt", "de", "it", "nl", "sv", "no", "da",
    "ro", "ca", "pl", "cs", "sk", "sl", "hu", "fi", "et", "lv",
    "lt", "hr", "af",
})


def detect_topic_language(text: str, default: str = "en") -> str:
    """Return an ISO-639-1 language code for the user's topic.

    Uses langdetect. Deterministic (seeded). If detection fails (short
    input, langdetect not installed, or language uncertain), returns
    ``default``.
    """
    if not text or not text.strip():
        return default
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(text)
        # langdetect returns "zh-cn"/"zh-tw" for Chinese variants; collapse.
        if lang.startswith("zh"):
            return "zh"
        return lang
    except Exception:
        # langdetect missing OR detection failure (LangDetectException).
        # Fall back to a codepoint-range hint so we still route non-Latin
        # inputs to the non-Latin model when langdetect is unavailable.
        return _fallback_script_detect(text, default)


def _fallback_script_detect(text: str, default: str) -> str:
    """Codepoint-range hint used only when langdetect can't be imported or
    raises. Returns coarse language tags: 'zh' (CJK), 'ar' (Arabic),
    'ru' (Cyrillic), 'hi' (Devanagari), 'ko' (Hangul), 'th' (Thai),
    'he' (Hebrew), or the default when the text looks Latin.
    """
    for c in text:
        cp = ord(c)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF:
            return "zh"  # CJK unified + hiragana/katakana (route as non-Latin)
        if 0xAC00 <= cp <= 0xD7AF:
            return "ko"  # Hangul syllables
        if 0x0E00 <= cp <= 0x0E7F:
            return "th"  # Thai
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return "ar"  # Arabic + supplement
        if 0x0590 <= cp <= 0x05FF:
            return "he"  # Hebrew
        if 0x0400 <= cp <= 0x04FF:
            return "ru"  # Cyrillic
        if 0x0900 <= cp <= 0x097F:
            return "hi"  # Devanagari
    return default


def select_model(topic_lang: str) -> str:
    """Return the ollama model name for the whole pipeline given a topic
    language.

    Routing:
      Latin European lang → RRR_MODEL_LATIN (default mistral-small:24b)
      other              → RRR_MODEL_NONLATIN (default qwen3:14b)

    If the routed model isn't found in the local ollama registry AND the
    other tier is available, falls back with a stderr warning so an
    operator without qwen3:14b installed still gets a run — just at
    degraded multilingual quality.

    A completely empty registry falls through to the legacy
    RRR_MODEL / 'mistral' chain so the caller can still boot and produce
    a clear ollama error rather than a mysterious import-time failure.
    """
    # API and subscription-host runtimes use one configured model for every
    # language, so neither path should inspect the local Ollama registry.
    runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
    if runtime == "api":
        from rrr.api_backend import api_model_name

        return api_model_name()
    if runtime == "host":
        from rrr.host_backend import host_model_name

        return host_model_name()

    latin = os.environ.get("RRR_MODEL_LATIN", MODEL_LATIN_DEFAULT)
    nonlatin = os.environ.get("RRR_MODEL_NONLATIN", MODEL_NONLATIN_DEFAULT)
    preferred = latin if topic_lang in LATIN_LANGS else nonlatin

    available = _list_ollama_models()
    if available is None:
        # ollama query failed (service down, network, etc.) — trust env.
        return preferred

    if _model_available(preferred, available):
        return preferred

    # Preferred tier missing. Try the other tier as fallback.
    other = nonlatin if preferred is latin else latin
    if _model_available(other, available):
        sys.stderr.write(
            f"[RRR] language={topic_lang}: preferred model {preferred!r} "
            f"not installed, falling back to {other!r}. Install with "
            f"'ollama pull {preferred}' for full multilingual quality.\n"
        )
        return other

    # Neither tier available. Fall through to legacy env-var chain so the
    # caller sees a clean ollama error at the first chat call.
    # v15.15: the ultimate default is MODEL_LATIN_DEFAULT (mistral-small:24b),
    # not mistral 7B. 7B is no longer a default anywhere — it stays reachable
    # for benchmarking via RRR_MODEL_LATIN=mistral:latest. Keep the stage-module
    # fallbacks in reasoner/writer/stance/outline/query_planner/cli in sync
    # with this value.
    legacy = os.environ.get(
        "RRR_REASONER_MODEL",
        os.environ.get("RRR_MODEL", MODEL_LATIN_DEFAULT),
    )
    sys.stderr.write(
        f"[RRR] language={topic_lang}: neither {preferred!r} nor {other!r} "
        f"installed; falling through to legacy chain {legacy!r}.\n"
    )
    return legacy


def _list_ollama_models() -> Optional[set]:
    """Return the set of installed ollama models (e.g. {'mistral:latest',
    'qwen3:14b'}) or None if ollama isn't reachable / not installed. The
    check is best-effort; a missing check simply skips fallback logic.
    """
    try:
        import ollama
        raw = ollama.list()
    except Exception:
        return None
    names = set()
    # ollama.list() shape varies by client version:
    #   old (dict):    {"models": [{"name": "mistral:latest"}, ...]}
    #   new (pydantic): ListResponse(models=[Model(model="mistral:latest"), ...])
    # Handle both. Individual model entries can be dicts OR pydantic Models
    # (Model.get(field) works via pydantic-compat, but attribute access is
    # the more portable path).
    if isinstance(raw, dict):
        models = raw.get("models", []) or []
    else:
        models = getattr(raw, "models", None) or []
    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model") or ""
        else:
            name = (getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or "")
        if name:
            names.add(name)
    return names


def _model_available(model: str, installed: set) -> bool:
    """True if ``model`` is in ``installed``, tolerating tag omission
    ('mistral' matches 'mistral:latest')."""
    if model in installed:
        return True
    if ":" not in model and f"{model}:latest" in installed:
        return True
    return False


def language_directive(topic_lang: str) -> str:
    """One-line prompt fragment appended to every LLM stage's user message
    so mistral/qwen respond in the user's language when it differs from
    English. Empty string for English (no directive needed; reduces prompt
    noise on the majority path).
    """
    if not topic_lang or topic_lang == "en":
        return ""
    return f"Respond in {language_name(topic_lang)}."


# ISO-639-1 → English-language name so prompts are readable by any
# instruction-tuned model. Curated set covering the languages the routing
# table actually supports.
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "de": "German", "it": "Italian", "nl": "Dutch",
    "sv": "Swedish", "no": "Norwegian", "da": "Danish",
    "ro": "Romanian", "ca": "Catalan", "pl": "Polish",
    "cs": "Czech", "sk": "Slovak", "hu": "Hungarian",
    "fi": "Finnish", "af": "Afrikaans", "sl": "Slovenian",
    "hr": "Croatian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "th": "Thai", "ar": "Arabic", "he": "Hebrew",
    "ru": "Russian", "uk": "Ukrainian", "hi": "Hindi",
}


def language_name(code: str) -> str:
    """ISO-639-1 code → human-readable English language name. Falls back to
    the raw code when unknown (so prompts remain intelligible)."""
    return LANGUAGE_NAMES.get((code or "").strip().lower(), code or "the source language")
