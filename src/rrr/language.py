from __future__ import annotations

import os
import sys
from typing import Optional


MODEL_LATIN_DEFAULT = "mistral-small:24b"
MODEL_NONLATIN_DEFAULT = "qwen3:14b"


LATIN_LANGS = frozenset({
    "en", "fr", "es", "pt", "de", "it", "nl", "sv", "no", "da",
    "ro", "ca", "pl", "cs", "sk", "sl", "hu", "fi", "et", "lv",
    "lt", "hr", "af",
})


def detect_topic_language(text: str, default: str = "en") -> str:
    if not text or not text.strip():
        return default
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(text)
        if lang.startswith("zh"):
            return "zh"
        return lang
    except Exception:
        return _fallback_script_detect(text, default)


def _fallback_script_detect(text: str, default: str) -> str:
    for c in text:
        cp = ord(c)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF:
            return "zh"
        if 0xAC00 <= cp <= 0xD7AF:
            return "ko"
        if 0x0E00 <= cp <= 0x0E7F:
            return "th"
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return "ar"
        if 0x0590 <= cp <= 0x05FF:
            return "he"
        if 0x0400 <= cp <= 0x04FF:
            return "ru"
        if 0x0900 <= cp <= 0x097F:
            return "hi"
    return default


def select_model(topic_lang: str) -> str:
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
        return preferred

    if _model_available(preferred, available):
        return preferred

    other = nonlatin if preferred is latin else latin
    if _model_available(other, available):
        sys.stderr.write(
            f"[RRR] language={topic_lang}: preferred model {preferred!r} "
            f"not installed, falling back to {other!r}. Install with "
            f"'ollama pull {preferred}' for full multilingual quality.\n"
        )
        return other

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
    try:
        import ollama
        raw = ollama.list()
    except Exception:
        return None
    names = set()
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
    if model in installed:
        return True
    if ":" not in model and f"{model}:latest" in installed:
        return True
    return False


def language_directive(topic_lang: str) -> str:
    if not topic_lang or topic_lang == "en":
        return ""
    return f"Respond in {language_name(topic_lang)}."


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
    return LANGUAGE_NAMES.get((code or "").strip().lower(), code or "the source language")
