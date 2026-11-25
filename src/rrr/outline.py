from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional

from rrr.paths import stage_cache_enabled, stage_cache_path
from rrr.utils import ensure_dir


_MODEL = os.environ.get(
    "RRR_OUTLINE_MODEL",
    os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b")),
)
_KEEP_ALIVE = "30m"

_PRECHECK_PROMPT_VERSION = "2026-11-25-v15.19.5-precheck"
_CLUSTER_PROMPT_VERSION = "2026-11-25-v15.19.5-cluster"
_POSTURE_PROMPT_VERSION = "2026-11-25-v15.19.5-posture"
_ORDER_PROMPT_VERSION = "2026-11-25-v15.19.5-order"


def _lang_sig() -> bytes:
    return (os.environ.get("RRR_CORPUS_LANG", "en") or "en").encode("utf-8")

_OPTIONS_PRECHECK = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 400}
_OPTIONS_CLUSTER = {"temperature": 0.0, "num_ctx": 12288, "num_predict": 1800}
_OPTIONS_POSTURE = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 600}
_OPTIONS_ORDER = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300}


_TOPIC_SHAPES = ("causal", "explanatory", "comparative", "descriptive")

_OUTCOME_SHAPES = ("causal", "explanatory")

_RELATIONS_BY_SHAPE: Dict[str, set] = {
    "causal": {
        "same_as_topic_cause",
        "upstream_of_topic_cause",
        "downstream_of_topic_cause",
        "rival_to_topic_cause",
        "scope_condition",
        "adjacent",
    },
    "explanatory": {
        "explains_outcome",
        "documents_outcome",
        "disputes_outcome",
        "adjacent",
    },
    "comparative": {
        "endorses_topic",
        "reverses_topic",
        "qualifies_topic",
        "methodological_critique",
        "adjacent",
    },
    "descriptive": {
        "confirms_description",
        "contradicts_description",
        "adds_nuance",
        "adjacent",
    },
}


def _sig_cluster(topic: str, doc_claims: List[Dict[str, str]],
                 topic_shape: str = "", topic_outcome: str = "") -> str:
    h = hashlib.sha256()
    h.update(_CLUSTER_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update(_lang_sig())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    h.update(b"\x04")
    h.update((topic_shape or "").encode("utf-8"))
    h.update(b"\x05")
    h.update((topic_outcome or "").encode("utf-8"))
    for entry in sorted(doc_claims, key=lambda d: d.get("doc_id", "")):
        h.update(b"\x01")
        h.update(str(entry.get("doc_id", "")).encode("utf-8"))
        h.update(b"\x02")
        h.update((entry.get("claim") or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _sig_posture(topic: str, topic_shape: str, cluster_doc_ids: List[str],
                 cluster_claims: List[str],
                 topic_cause: str = "", topic_outcome: str = "",
                 shared_thread: str = "", shared_cause: str = "",
                 cluster_evidence: Dict[str, List[str]] = None) -> str:
    h = hashlib.sha256()
    h.update(_POSTURE_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update(_lang_sig())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    h.update(b"\x01")
    h.update((topic_shape or "").encode())
    h.update(b"\x04")
    h.update((topic_cause or "").encode("utf-8"))
    h.update(b"\x05")
    h.update((topic_outcome or "").encode("utf-8"))
    h.update(b"\x06")
    h.update((shared_thread or "").encode("utf-8"))
    h.update(b"\x07")
    h.update((shared_cause or "").encode("utf-8"))
    for did, claim in sorted(zip(cluster_doc_ids, cluster_claims), key=lambda p: p[0]):
        h.update(b"\x02")
        h.update(str(did).encode("utf-8"))
        h.update(b"\x03")
        h.update((claim or "").encode("utf-8"))
    for did in sorted(cluster_evidence or {}):
        h.update(b"\x08")
        h.update(str(did).encode("utf-8"))
        for snip in (cluster_evidence or {}).get(did) or []:
            h.update(b"\x09")
            h.update((snip or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _sig_order(topic: str, cluster_summaries: List[Dict[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(_ORDER_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update(_lang_sig())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    for cs in sorted(cluster_summaries, key=lambda c: c.get("cluster_id", "")):
        h.update(b"\x01")
        h.update((cs.get("cluster_id") or "").encode())
        h.update(b"\x02")
        h.update((cs.get("relation") or "").encode())
        h.update(b"\x03")
        h.update((cs.get("elaboration") or "").encode("utf-8"))
        h.update(b"\x04")
        h.update((cs.get("shared_thread") or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(stage: str, sig: str) -> str:
    return str(stage_cache_path("outline", stage, f"{sig}.json"))


def _load_cache(stage: str, sig: str):
    if not stage_cache_enabled():
        return None
    try:
        with open(_cache_path(stage, sig), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(stage: str, sig: str, obj) -> None:
    ensure_dir(str(stage_cache_path("outline", stage)))
    path = _cache_path(stage, sig)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _sig_precheck(topic: str, doc_titles: List[str]) -> str:
    h = hashlib.sha256()
    h.update(_PRECHECK_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update(_lang_sig())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    for t in sorted(doc_titles):
        h.update(b"\x01")
        h.update((t or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _language_prefix() -> str:
    from rrr.language import language_directive
    directive = language_directive(os.environ.get("RRR_CORPUS_LANG", "en"))
    return f"{directive}\n\n" if directive else ""


def _build_precheck_prompt(topic: str, doc_titles: List[str]) -> str:
    _lang_pfx = _language_prefix()
    titles_block = "\n".join(f"  - {t}" for t in doc_titles)
    return (
        _lang_pfx +
        "You are doing a PRE-FLIGHT check for a literature-review generator. "
        "Two decisions, ONE JSON output.\n\n"
        f"TOPIC: {topic}\n\n"
        "CORPUS (one line per paper that survived initial retrieval):\n"
        f"{titles_block}\n\n"
        "## PART 1 — TOPIC SHAPE\n"
        "What SHAPE is this topic? Pick exactly one. Examples below span "
        "different academic domains on purpose — the SHAPE is determined by "
        "what the topic ASKS, not by its subject matter.\n\n"
        "  - causal: the topic NAMES a cause X and an outcome Y. It asserts "
        "that X CAUSES or EXPLAINS Y, or asks whether or how X affects Y. "
        "Hallmark: the topic's own words name the cause. A statement that "
        "names both a cause and its effect is causal.\n"
        "      Examples: \"Sleep deprivation impairs immune function.\" "
        "\"Catalyst surface area determines reaction rate.\" "
        "\"Coral bleaching resulted from warmer seas.\" "
        "\"How did the printing press change literacy in early modern Europe?\"\n\n"
        "  - why_question: the topic is a QUESTION. It names an outcome Y "
        "and asks WHY it happened, WHAT explains it, or HOW it came about, "
        "and it names NO cause: finding the cause is the question. Y may "
        "itself be a gap between two cases. A statement is NEVER a "
        "why_question, and a question that names the cause it asks about "
        "(\"How did X change Y?\") is causal.\n"
        "      Examples: \"Why are honeybee colonies declining?\" "
        "\"How did the Antarctic ozone hole come about?\" "
        "\"Why do songbirds in cities sing at a higher pitch than those in "
        "forests?\"\n\n"
        "  - comparative: the topic asserts that A differs from / is better "
        "than / is more X than B. Hallmark: two named entities being compared "
        "on some dimension.\n"
        "      Examples: \"Is Bayesian inference more sample-efficient than "
        "frequentist inference?\" \"Did silent reading replace oral reading "
        "in late medieval Europe?\"\n\n"
        "  - descriptive: the topic asks what something IS, what PATTERN is "
        "observed, what FEATURES X has, or what has been DOCUMENTED about X. "
        "There is NO single causal claim and NO comparison; the topic asks "
        "the literature to describe a state of affairs.\n"
        "      Examples: \"What patterns of code-switching have been "
        "documented in bilingual children?\" \"What are the morphological "
        "features of slime molds?\" \"How is voter intention measured in "
        "pre-election polls?\"\n\n"
        "COMMON TRAP: do NOT default to causal whenever the topic's SUBJECT "
        "matter has well-known causal stories attached. A topic that ASKS "
        "about patterns, features, or measurements is descriptive even when "
        "its subject is something normally studied causally. Read the topic "
        "as a QUESTION: does it ask 'does X cause Y?' (causal), 'why Y?' or "
        "'what causes Y?' (why_question), 'is A more X than B?' "
        "(comparative), or 'what is/are X?' (descriptive).\n\n"
        "SECOND TRAP: why_question is only for a question that names no "
        "cause. A topic that states its cause (\"Y resulted from X\", \"X "
        "shaped Y\") is causal, and X is its cause. Never supply a cause "
        "the topic does not name.\n\n"
        "## PART 2 — CORPUS FIT\n"
        "Look at the paper TITLES above. Are these papers PLAUSIBLY useful "
        "for a literature review on the topic? Be honest. Two outcomes:\n\n"
        "  - PROCEED: a substantial number of papers (say 5+) could "
        "reasonably contribute evidence to the topic, even if the rest are "
        "tangential. The topic and the corpus share a subject matter, "
        "methodology, or evidence base.\n\n"
        "  - REFUSE: there is no honest scholarly path from the papers in "
        "the corpus to the question the topic asks. This is the case when "
        "the corpus and topic come from intellectual domains with no shared "
        "subject matter, no shared methodology, and no shared evidence base "
        "— the papers simply cannot inform the topic.\n\n"
        "It is FAR BETTER to refuse than to invent connections between "
        "unrelated literatures. A refusal here triggers a polite refusal "
        "message to the user; a wrong PROCEED produces a HALLUCINATED "
        "review built from papers that do not honestly bear on the topic. "
        "Hallucination is the worst possible outcome.\n\n"
        "  - REFUSE (unintelligible topic): a special case of REFUSE. If the "
        "topic is not a coherent scholarly question — random characters, a "
        "keyboard slam, a grammatically-broken word-salad with no research "
        "question, or a fragment that no reasonable scholar could interpret "
        "— return REFUSE regardless of any surface-word overlap with the "
        "titles. Do NOT invent a research question the user did not ask. Do "
        "NOT force-fit a well-formed shape onto ill-formed input. Examples "
        "of REFUSE-worthy topics: \"asdfgh qwerty\", \"milk house getting "
        "asked whole table\", \"the of a and the\".\n\n"
        "Calibration: PROCEED is the right answer for the large majority of "
        "well-formed (topic, corpus) pairs a user is likely to submit, "
        "because users normally pair coherent topics with relevant corpora. "
        "REFUSE is the right answer when EITHER the mismatch is obvious "
        "from the titles OR the topic itself is not a coherent scholarly "
        "question. Do not refuse a coherent topic on weak fit; do not "
        "proceed on an incoherent topic just because the corpus is "
        "topically broad.\n\n"
        "## PART 3 — SLOTS (skip if topic_shape is comparative or descriptive)\n"
        "If you picked topic_shape=\"causal\" or \"why_question\", also "
        "identify the pinned slots that every downstream stage will "
        "reference verbatim:\n\n"
        "  - topic_outcome: a SHORT noun phrase (<=80 chars) for the "
        "phenomenon being EXPLAINED — the explanandum. Required for causal "
        "topics AND for a why_question.\n"
        "  - topic_cause: a SHORT noun phrase (<=80 chars) for the variable "
        "the topic claims is doing the EXPLAINING — the explanans. Causal "
        "topics only: copy the topic's own words. For a why_question "
        "return an EMPTY string. Do not guess a cause and do not write a "
        "placeholder such as \"unknown factors\".\n\n"
        "These slots are the most important fields you produce. Every "
        "downstream classification of an individual paper's relation to the "
        "topic is anchored on them. Be CONCRETE and use the topic's own "
        "wording where possible.\n\n"
        "Examples (cross-domain, on purpose):\n"
        "  Topic \"Sleep deprivation impairs immune function.\" -> "
        "topic_cause=\"sleep deprivation\", topic_outcome=\"impaired immune "
        "function\".\n"
        "  Topic \"Coral bleaching resulted from warmer seas.\" -> "
        "topic_cause=\"warmer seas\", topic_outcome=\"coral bleaching\".\n"
        "  Topic \"Why are honeybee colonies declining?\" -> "
        "topic_cause=\"\", topic_outcome=\"decline of honeybee colonies\".\n"
        "  Topic \"Why do songbirds in cities sing at a higher pitch than "
        "those in forests?\" -> topic_cause=\"\", topic_outcome=\"higher-"
        "pitched song of city songbirds\".\n\n"
        "For comparative or descriptive topics leave both fields as empty "
        "strings — they are not used.\n\n"
        "## OUTPUT\n"
        "Return ONE JSON object with EXACTLY these six keys:\n"
        "  topic_shape: \"causal\" | \"why_question\" | \"comparative\" | "
        "\"descriptive\"\n"
        "  topic_shape_rationale: one sentence (<=200 chars) defending the "
        "shape choice, quoting the topic's phrasing.\n"
        "  corpus_fit: \"PROCEED\" | \"REFUSE\"\n"
        "  corpus_fit_rationale: one sentence (<=240 chars) — if REFUSE, name "
        "the domain mismatch concretely.\n"
        "  topic_cause: noun phrase (<=80 chars) or empty string\n"
        "  topic_outcome: noun phrase (<=80 chars) or empty string\n\n"
        "Return ONLY the JSON object."
    )


_GENERIC_CAUSE_NOUNS = frozenset({
    "factor", "factors", "driver", "drivers", "cause", "causes", "reason",
    "reasons", "determinant", "determinants", "explanation", "explanations",
    "source", "sources", "origin", "origins", "root", "roots", "force",
    "forces", "influence", "influences", "mechanism", "mechanisms",
    "factores", "causas", "razones", "determinantes", "fatores", "razões",
    "razoes", "facteurs", "raisons", "faktoren", "ursachen", "gründe",
    "grunde", "fattori", "ragioni", "motivos", "motivi",
})
_PLACEHOLDER_QUALIFIERS = frozenset({
    "unspecified", "unknown", "unnamed", "unclear", "undetermined",
    "unidentified", "various", "several", "multiple", "many", "numerous",
    "underlying", "possible", "potential", "certain", "other", "some",
    "different", "main", "key", "major", "principal", "primary", "complex",
    "open", "debated", "contested", "competing",
})
_SLOT_FUNCTION_WORDS = frozenset({
    "the", "an", "of", "in", "on", "at", "to", "for", "by", "with", "and",
    "or", "but", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "their", "them",
    "we", "us", "our", "you", "your", "he", "she", "his", "her", "from",
    "into", "onto", "upon", "out", "over", "under", "between", "through",
    "via", "than", "then", "thus", "hence", "also", "not", "no", "if",
    "when", "where", "while", "because", "so", "more", "less", "most",
    "least", "any", "all", "each", "every", "such", "same", "do", "does",
    "did", "have", "has", "had", "having",
    "how", "what", "why", "who", "whom", "whose", "which", "after", "before",
    "during", "since", "about", "within", "across", "among", "around",
    "against", "toward", "towards", "behind", "per", "too", "very", "there",
    "here", "will", "would", "can", "could", "should", "may", "might",
    "must", "shall", "both", "either", "neither", "nor", "yet",
})


def _slot_words(text: str) -> List[str]:
    words = re.findall(r"[^\W\d_]+|\d+", (text or "").casefold())
    return [w for w in words if len(w) >= 2 and w not in _SLOT_FUNCTION_WORDS]


def _unspaced_script(text: str) -> bool:
    for ch in text or "":
        cp = ord(ch)
        if (0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
                or 0x0E00 <= cp <= 0x0EFF or 0x1000 <= cp <= 0x109F or 0x1780 <= cp <= 0x17FF):
            return True
    return False


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 4 and long_.startswith(short):
        return True
    return len(short) >= 5 and short[:5] == long_[:5]


def _word_in(word: str, words) -> bool:
    return any(_same_word(word, other) for other in words)


def _cause_check(cause: str, topic: str, outcome: str = "") -> str:
    cause_words = _slot_words(cause)
    topic_words = _slot_words(topic)
    if not topic_words or _unspaced_script(topic) or _unspaced_script(cause):
        return ""
    generic = _GENERIC_CAUSE_NOUNS | _PLACEHOLDER_QUALIFIERS
    for word in cause_words:
        if word not in generic:
            break
        if word in _GENERIC_CAUSE_NOUNS and not _word_in(word, topic_words):
            return "placeholder"
    core = [w for w in cause_words if w not in generic]
    if not core:
        return "placeholder"
    named = [w for w in core if _word_in(w, topic_words)]
    if len(named) * 2 < len(core):
        return "not_in_topic"
    outcome_words = _slot_words(outcome)
    if not [w for w in named if not _word_in(w, outcome_words)]:
        return "repeats_outcome"
    return ""


def _cause_words_comparable() -> bool:
    topic_lang = (os.environ.get("RRR_TOPIC_LANG", "en") or "en").strip().lower()
    corpus_lang = (os.environ.get("RRR_CORPUS_LANG", "en") or "en").strip().lower()
    return topic_lang == corpus_lang


_SHAPE_LABELS = {
    "why_question": "explanatory",
    "why-question": "explanatory",
    "why question": "explanatory",
    "why": "explanatory",
}

_QUESTION_OPENERS = frozenset({
    "why", "what", "how", "which", "when", "where", "who", "whom", "whose",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "should",
    "will", "would", "has", "have", "had", "to",
})


def _is_question(topic: str) -> bool:
    text = (topic or "").strip()
    if any(mark in text for mark in ("?", "？", "؟", ";", "¿")):
        return True
    first = re.findall(r"[^\W\d_]+", text.casefold())[:1]
    return bool(first) and first[0] in _QUESTION_OPENERS


def _validate_precheck(obj, topic: str = "", compare_words: bool = True,
                       final: bool = True) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    fit = str(obj.get("corpus_fit", "")).strip().upper()
    if fit not in ("PROCEED", "REFUSE"):
        return None
    shape = str(obj.get("topic_shape", "")).strip().lower()
    shape = _SHAPE_LABELS.get(shape, shape)
    if shape not in _TOPIC_SHAPES:
        if fit != "REFUSE":
            return None
        shape = "unclassified"
    topic_cause = str(obj.get("topic_cause", "") or "").strip()[:120]
    topic_outcome = str(obj.get("topic_outcome", "") or "").strip()[:120]
    model_shape, model_cause = shape, topic_cause
    adjusted = ""
    is_statement = bool(topic) and compare_words and not _is_question(topic)
    if fit == "PROCEED" and shape in _OUTCOME_SHAPES:
        if not topic_outcome:
            return None
        if shape == "explanatory":
            if topic_cause:
                adjusted = "cause_dropped_from_open_question"
                topic_cause = ""
        elif not topic_cause:
            shape = "explanatory"
            adjusted = "causal_without_cause"
        elif topic and compare_words and not is_statement:
            refused = _cause_check(topic_cause, topic, topic_outcome)
            if refused:
                shape = "explanatory"
                adjusted = f"cause_{refused}"
                topic_cause = ""
    if fit == "PROCEED" and shape == "explanatory" and is_statement:
        if model_shape == "causal" and not final:
            return None
        adjusted = ("statement_not_a_question" if model_shape == "explanatory"
                    else "statement_without_cause")
        shape = "descriptive"
        topic_cause = ""
        topic_outcome = ""
    result = {
        "topic_shape": shape,
        "topic_shape_rationale": str(obj.get("topic_shape_rationale", "") or "").strip()[:300],
        "corpus_fit": fit,
        "corpus_fit_rationale": str(obj.get("corpus_fit_rationale", "") or "").strip()[:400],
        "topic_cause": topic_cause,
        "topic_outcome": topic_outcome,
    }
    if adjusted:
        result["shape_adjusted"] = adjusted
        result["model_topic_shape"] = model_shape
        result["model_topic_cause"] = model_cause
    return result


_PRECHECK_RETRY_REMINDER = (
    "\n\nReminder: return ONLY the JSON object with the six keys. "
    "topic_shape must be one of the four values listed and corpus_fit must "
    "be PROCEED or REFUSE. A causal topic needs a topic_cause, in the "
    "topic's own words, and a topic_outcome. A why_question needs a "
    "topic_outcome and has an empty topic_cause."
)


def _precheck_failure(diagnostics: Optional[dict], kind: str, detail: str = "") -> None:
    if diagnostics is not None:
        diagnostics["failure"] = kind
        diagnostics["detail"] = (detail or "")[:400]
    return None


def precheck(topic: str, doc_summaries: List[dict], metrics=None,
             diagnostics: Optional[dict] = None) -> Optional[dict]:
    titles = []
    for d in doc_summaries:
        did = (d.get("doc_id") or "").strip()
        if not did:
            continue
        cite = (d.get("citation") or did).strip()
        cite = " ".join(cite.split())[:240]
        titles.append(cite)
    if not titles:
        return _precheck_failure(diagnostics, "no_titles")

    sig = _sig_precheck(topic, titles)
    cached = _load_cache("precheck", sig)
    if cached and isinstance(cached, dict) and cached.get("corpus_fit"):
        if metrics:
            metrics.cache_event("outline_precheck", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_precheck", "misses")

    prompt = _build_precheck_prompt(topic, titles)
    compare_words = _cause_words_comparable()
    result = None
    failure = ("model_call_error", "")
    for attempt, stage_name in enumerate(("outline_precheck", "outline_precheck_retry")):
        final = attempt == 1
        text = prompt + (_PRECHECK_RETRY_REMINDER if final else "")
        raw = ""
        try:
            import ollama
            start = time.perf_counter()
            res = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": text}],
                options=_OPTIONS_PRECHECK,
                keep_alive=_KEEP_ALIVE,
                format="json",
                stream=False,
            )
            raw = (res.get("message", {}).get("content") or "").strip()
            if metrics:
                metrics.record_llm(stage_name, _MODEL, options=_OPTIONS_PRECHECK,
                                   duration_s=time.perf_counter() - start,
                                   prompt_chars=len(text),
                                   response_chars=len(raw))
        except Exception as e:
            if metrics:
                metrics.record_llm(stage_name, _MODEL, options=_OPTIONS_PRECHECK,
                                   success=False, error=e)
            failure = ("model_call_error", f"{type(e).__name__}: {e}")
            continue
        if not raw:
            failure = ("empty_reply", "")
            continue
        obj = _parse_json_object(raw)
        if obj is None:
            failure = ("malformed_json", raw[:300])
            continue
        try:
            result = _validate_precheck(obj, topic, compare_words, final=final)
        except Exception:
            result = None
        if result is not None:
            break
        failure = ("invalid_answer", raw[:300])
    if result is None:
        return _precheck_failure(diagnostics, *failure)
    if result.get("shape_adjusted") and metrics:
        metrics.set("outline_shape_adjusted", result["shape_adjusted"])
    result["model"] = _MODEL
    result["prompt_version"] = _PRECHECK_PROMPT_VERSION
    _save_cache("precheck", sig, result)
    if metrics:
        metrics.cache_event("outline_precheck", "writes")
    return result


def _build_cluster_prompt(topic: str, doc_claims: List[Dict[str, str]],
                          topic_shape: str = "causal",
                          topic_outcome: str = "") -> str:
    if topic_shape == "explanatory":
        return _build_explanation_cluster_prompt(topic, doc_claims, topic_outcome)
    has_outcome = bool(topic_outcome) and topic_shape == "causal"
    outcome_line = (
        f"TOPIC OUTCOME: {topic_outcome}"
        if has_outcome else ""
    )
    cause_field_doc = (
        "    - shared_cause: 5-12 word noun phrase naming the cluster's "
        "distinctive cause (in the papers' own language). Required for "
        "causal topics; empty string for other shapes.\n"
        if has_outcome else ""
    )
    header_lines = [
        f"TOPIC: {topic}",
        f"TOPIC SHAPE: {topic_shape}",
    ]
    if outcome_line:
        header_lines.append(outcome_line)
    lines = header_lines + [
        "",
        "Group these papers into clusters that separate the distinct "
        "streams of literature in this corpus. Each cluster is one "
        "stream: papers developing a related line of argument.",
        "",
        "Aim for 3-6 clusters. This is a soft target, not a quota: prefer "
        "fewer, broader streams when papers cohere; only split further "
        "when a single label would obscure a real intellectual disagreement. "
        "Each cluster should typically hold 3+ papers; 1-2 paper clusters "
        "are acceptable only when a paper truly stands alone in the corpus. "
        "Over-splitting (10+ clusters from ~20 papers) produces redundant "
        "sections that say the same thing under different labels.",
        "",
        "Use `unassigned_doc_ids` for papers that address a genuinely "
        "different question. Every doc_id appears EXACTLY ONCE. Do NOT "
        "label relations or stances — Stage 2 handles those.",
        "",
        "PAPERS:",
    ]
    for entry in doc_claims:
        did = entry.get("doc_id", "")
        claim = (entry.get("claim") or "(no claim extracted)").strip()
        lines.append(f"  [{did}] {claim}")
    lines += [
        "",
        "Return ONE JSON object with EXACTLY these keys:",
        "  clusters: array of {cluster_id, doc_ids, shared_thread, shared_cause}",
        "    - cluster_id: a short tag like \"C1\", \"C2\", ...",
        "    - doc_ids: array of doc_id strings from the list above",
        "    - shared_thread: 5-10 word noun phrase summarising the cluster",
        cause_field_doc.rstrip("\n") if cause_field_doc else
        "    - shared_cause: empty string (only used for causal topics)",
        "  unassigned_doc_ids: array of doc_id strings",
        "",
        "Return ONLY the JSON object. No commentary.",
    ]
    return _language_prefix() + "\n".join(lines)


def _build_explanation_cluster_prompt(topic: str, doc_claims: List[Dict[str, str]],
                                      topic_outcome: str) -> str:
    lines = [
        f"TOPIC: {topic}",
        "TOPIC SHAPE: explanatory (the topic asks what explains the outcome "
        "below and names no cause)",
        f"TOPIC OUTCOME: {topic_outcome}",
        "",
        "Group these papers into clusters that separate the distinct "
        "streams of literature in this corpus, as they bear on the TOPIC "
        "OUTCOME:",
        "  - A paper that offers an EXPLANATION of the outcome goes with the "
        "papers that point to the same cause or the same kind of cause: one "
        "cluster per explanation.",
        "  - Papers that measure, date or describe the outcome without "
        "explaining it form a cluster of their own.",
        "  - Papers on a neighbouring question (a consequence of the "
        "outcome, its wider setting, the same outcome in another place) "
        "form clusters by subject.",
        "",
        "Aim for 3-6 clusters. This is a soft target, not a quota: keep two "
        "explanations apart when the papers treat them as competing answers, "
        "and merge papers that give the same answer in different words. Each "
        "cluster should typically hold 3+ papers; 1-2 paper clusters are "
        "acceptable only when a paper truly stands alone in the corpus.",
        "",
        "Use `unassigned_doc_ids` for papers that address a genuinely "
        "different question. Every doc_id appears EXACTLY ONCE. Do NOT "
        "label relations or stances — Stage 2 handles those. Take each "
        "explanation from the papers' claims; do not add one of your own.",
        "",
        "PAPERS:",
    ]
    for entry in doc_claims:
        did = entry.get("doc_id", "")
        claim = (entry.get("claim") or "(no claim extracted)").strip()
        lines.append(f"  [{did}] {claim}")
    lines += [
        "",
        "Return ONE JSON object with EXACTLY these keys:",
        "  clusters: array of {cluster_id, doc_ids, shared_thread, shared_cause}",
        "    - cluster_id: a short tag like \"C1\", \"C2\", ...",
        "    - doc_ids: array of doc_id strings from the list above",
        "    - shared_thread: 5-10 word noun phrase summarising the cluster",
        "    - shared_cause: 5-12 word noun phrase naming the explanation "
        "this cluster offers for the TOPIC OUTCOME (in the papers' own "
        "language); empty string for a cluster that offers none",
        "  unassigned_doc_ids: array of doc_id strings",
        "",
        "Return ONLY the JSON object. No commentary.",
    ]
    return _language_prefix() + "\n".join(lines)


def _validate_cluster_plan(obj, valid_doc_ids: set, topic_shape: str) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None

    raw_clusters = obj.get("clusters") or []
    if not isinstance(raw_clusters, list):
        return None

    seen = set()
    clusters = []
    for i, c in enumerate(raw_clusters):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("cluster_id", "") or f"C{i+1}").strip() or f"C{i+1}"
        thread = str(c.get("shared_thread", "") or "").strip()[:160]
        cause = str(c.get("shared_cause", "") or "").strip()[:160]
        raw_ids = c.get("doc_ids") or []
        if not isinstance(raw_ids, list):
            continue
        doc_ids = []
        for did in raw_ids:
            s = str(did).strip()
            if s in valid_doc_ids and s not in seen:
                doc_ids.append(s)
                seen.add(s)
        if doc_ids and thread:
            clusters.append({
                "cluster_id": cid,
                "doc_ids": doc_ids,
                "shared_thread": thread,
                "shared_cause": cause,
            })

    raw_unassigned = obj.get("unassigned_doc_ids") or []
    if not isinstance(raw_unassigned, list):
        raw_unassigned = []
    unassigned = []
    for did in raw_unassigned:
        s = str(did).strip()
        if s in valid_doc_ids and s not in seen:
            unassigned.append(s)
            seen.add(s)

    leftover = sorted(valid_doc_ids - seen)
    unassigned.extend(leftover)

    if not clusters:
        return None

    return {
        "topic_shape": topic_shape,
        "clusters": clusters,
        "unassigned_doc_ids": unassigned,
    }


def cluster_papers(topic: str, doc_summaries: List[dict],
                   topic_shape: str = "causal",
                   topic_outcome: str = "",
                   metrics=None,
                   diagnostics: Optional[dict] = None) -> Optional[dict]:
    doc_claims = [
        {"doc_id": d.get("doc_id", ""), "claim": (d.get("claim") or "").strip()}
        for d in doc_summaries
        if d.get("doc_id")
    ]
    if not doc_claims:
        return None

    valid_doc_ids = {d["doc_id"] for d in doc_claims}
    sig = _sig_cluster(topic, doc_claims, topic_shape=topic_shape, topic_outcome=topic_outcome)
    cached = _load_cache("cluster", sig)
    if cached and isinstance(cached, dict) and cached.get("clusters"):
        if metrics:
            metrics.cache_event("outline_cluster", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_cluster", "misses")

    prompt = _build_cluster_prompt(topic, doc_claims, topic_shape=topic_shape, topic_outcome=topic_outcome)
    raw = ""
    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OPTIONS_CLUSTER,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("outline_cluster", _MODEL, options=_OPTIONS_CLUSTER,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("outline_cluster", _MODEL, options=_OPTIONS_CLUSTER,
                               success=False, error=e)
        return _precheck_failure(diagnostics, "model_call_error", f"{type(e).__name__}: {e}")

    plan = _parse_and_validate(raw, lambda obj: _validate_cluster_plan(obj, valid_doc_ids, topic_shape))
    if plan is None:
        retry_prompt = prompt + "\n\nReminder: return ONLY the JSON object with the three keys topic_shape, clusters, unassigned_doc_ids."
        try:
            import ollama
            start = time.perf_counter()
            res = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": retry_prompt}],
                options=_OPTIONS_CLUSTER,
                keep_alive=_KEEP_ALIVE,
                format="json",
                stream=False,
            )
            raw = (res.get("message", {}).get("content") or "").strip()
            if metrics:
                metrics.record_llm("outline_cluster_retry", _MODEL, options=_OPTIONS_CLUSTER,
                                   duration_s=time.perf_counter() - start,
                                   prompt_chars=len(retry_prompt),
                                   response_chars=len(raw))
        except Exception as e:
            if metrics:
                metrics.record_llm("outline_cluster_retry", _MODEL, options=_OPTIONS_CLUSTER,
                                   success=False, error=e)
            return _precheck_failure(diagnostics, "model_call_error", f"{type(e).__name__}: {e}")
        plan = _parse_and_validate(raw, lambda obj: _validate_cluster_plan(obj, valid_doc_ids, topic_shape))
        if plan is None:
            return _precheck_failure(diagnostics, "no_valid_cluster_plan", raw[:300])

    plan["model"] = _MODEL
    plan["prompt_version"] = _CLUSTER_PROMPT_VERSION
    _save_cache("cluster", sig, plan)
    if metrics:
        metrics.cache_event("outline_cluster", "writes")
    return plan


def _scaffold_for_shape(topic_shape: str) -> str:
    if topic_shape == "causal":
        return (
            "Before choosing `relation`, populate the two QUOTE fields and "
            "then walk the tests IN ORDER. Take the FIRST test whose "
            "condition the cluster satisfies — do not keep testing once "
            "you have an answer.\n\n"
            "FOUNDATION — populate BOTH fields or relation is `adjacent`:\n"
            "  cluster_cause_quote: a VERBATIM substring (>=4 words) from "
            "the cluster's own claims or shared_cause naming the CAUSE the "
            "cluster proposes. Do not paraphrase. If you cannot find such "
            "a substring, the cluster is `adjacent` and you may stop.\n"
            "  cluster_outcome_quote: a VERBATIM substring (>=4 words) "
            "from the cluster's claims naming the OUTCOME the cluster "
            "explains. If you cannot find such a substring, the cluster "
            "is `adjacent` and you may stop.\n\n"
            "TEST C-THROUGH — CHAIN-THROUGH-TOPIC-CAUSE (apply FIRST):\n"
            "  Read your cluster_cause_quote. Does it name the topic's "
            "CAUSE (as quoted verbatim in TOPIC CAUSE above) as a "
            "DOWNSTREAM NOUN — i.e. as the thing the cluster's cause "
            "produces, establishes, shapes, leads to, causes, generates, "
            "or affects? Surface markers: cluster_cause_quote contains "
            "the topic-cause word(s) AFTER a verb like `led to`, "
            "`established`, `shaped`, `produced`, `caused`, `generated`, "
            "`affected`, `determined`, `gave rise to`. If yes, the "
            "cluster is arguing THROUGH the topic's cause — proceed to "
            "TEST S then TEST U; DO NOT consider rival at this point.\n\n"
            "TEST S — SAME-AS (re-expression, not rivalry):\n"
            "  Is the cluster's cause the topic's cause under a DIFFERENT "
            "LABEL or a more SPECIFIC INSTANCE of the same variable? "
            "(e.g. \"extractive institutions\" is an instance of "
            "\"institutions\"). If yes -> `same_as_topic_cause`.\n\n"
            "TEST U — UPSTREAM (cite the chain from your quotes):\n"
            "  Using cluster_cause_quote and cluster_outcome_quote, write "
            "the chain `cluster_cause -> topic_cause -> topic_outcome` in "
            "ONE sentence, with the bridge verb taken VERBATIM from "
            "cluster_cause_quote. If you can write that sentence without "
            "inventing words the cluster does not use -> "
            "`upstream_of_topic_cause`. If you cannot, this test FAILS; "
            "continue.\n\n"
            "TEST R — RIVAL (only after TEST C-THROUGH, S, U have failed):\n"
            "  Do the cluster's CAUSE and the topic's CAUSE both purport "
            "to EXPLAIN the SAME OUTCOME, with NEITHER cause acting "
            "THROUGH the other? Two causes are RIVAL when both are "
            "offered as competing answers to the same WHY-question at the "
            "same level (both fundamental, both proximate) AND the "
            "cluster's cause-quote does NOT name the topic's cause as a "
            "downstream conduit. If both conditions hold -> "
            "`rival_to_topic_cause`. If the cluster's cause acts through "
            "the topic's cause, you should have stopped at TEST "
            "C-THROUGH/S/U — do not pick rival here.\n\n"
            "TEST D — DOWNSTREAM:\n"
            "  Is the cluster's cause itself a CONSEQUENCE of the topic's "
            "cause (the cluster studies what happens AFTER the topic's "
            "cause acts)? -> `downstream_of_topic_cause`.\n\n"
            "TEST SC — SCOPE CONDITION:\n"
            "  Does the cluster accept the topic's claim but identify WHEN "
            "OR WHERE it holds vs fails (period limits, regional limits, "
            "measurement limits)? -> `scope_condition`.\n\n"
            "ADJACENT — fallback only:\n"
            "  If none of the tests above apply — the cluster talks about "
            "a different outcome or a different subject — -> `adjacent`."
        )
    if topic_shape == "explanatory":
        return (
            "Before choosing `relation`, populate the two QUOTE fields and "
            "then walk the tests IN ORDER. Take the FIRST test whose "
            "condition the cluster satisfies.\n\n"
            "QUOTES:\n"
            "  cluster_cause_quote: a VERBATIM substring (>=4 words) from "
            "the cluster's own claims or explanation naming the CAUSE the "
            "cluster proposes for the TOPIC OUTCOME. Do not paraphrase. "
            "Empty string if the cluster proposes none.\n"
            "  cluster_outcome_quote: a VERBATIM substring (>=4 words) "
            "from the cluster's claims naming what the cluster says about "
            "the TOPIC OUTCOME. If you cannot find such a substring, the "
            "cluster is `adjacent` and you may stop.\n\n"
            "TEST E — EXPLAINS:\n"
            "  Does the cluster name a cause and say that it produced, "
            "shaped or sustained the TOPIC OUTCOME? Your "
            "cluster_cause_quote must show that cause. If yes -> "
            "`explains_outcome`.\n\n"
            "TEST X — DISPUTES:\n"
            "  Does the cluster deny that the TOPIC OUTCOME holds as the "
            "topic states it, or report the opposite pattern? If yes -> "
            "`disputes_outcome`.\n\n"
            "TEST M — DOCUMENTS:\n"
            "  Does the cluster measure, date or describe the TOPIC "
            "OUTCOME (how large, since when, where, for whom) while "
            "proposing no cause for it? If yes -> `documents_outcome`.\n\n"
            "ADJACENT — fallback only:\n"
            "  If none of the tests apply — the cluster is about a "
            "different outcome or a different subject — -> `adjacent`.\n\n"
            "Judge only what the cluster's own claims and evidence say. Do "
            "not rank the explanations and do not add one of your own."
        )
    if topic_shape == "comparative":
        return (
            "Before choosing `relation`, write `reasoning_trace` answering "
            "these THREE questions in order:\n"
            "  (i) What COMPARISON does the topic make (A vs B, on what "
            "dimension)? (Quote the topic phrasing.)\n"
            "  (ii) What does this cluster's papers conclude about the same "
            "comparison? (Quote the language from at least one paper.)\n"
            "  (iii) Is the cluster: (A) confirming the topic's verdict "
            "(A really is more X than B) — `endorses_topic`; (B) reaching the "
            "OPPOSITE verdict — `reverses_topic`; (C) accepting the verdict "
            "only under conditions — `qualifies_topic`; (D) attacking the WAY "
            "the comparison is constructed (measurement, sample) without "
            "directly endorsing or reversing — `methodological_critique`; "
            "(E) about something else — `adjacent`."
        )
    return (
        "Before choosing `relation`, write `reasoning_trace` answering these "
        "THREE questions in order:\n"
        "  (i) What DESCRIPTION or PATTERN does the topic assert? "
        "(Quote the topic phrasing.)\n"
        "  (ii) What does this cluster's papers describe about the same "
        "subject? (Quote the language from at least one paper.)\n"
        "  (iii) Is the cluster's account: (A) consistent with the topic's "
        "description — `confirms_description`; (B) incompatible with it — "
        "`contradicts_description`; (C) accepting the description but "
        "refining its scope, mechanism, or magnitude — `adds_nuance`; "
        "(D) about something else — `adjacent`."
    )


def _allowed_relations_block(topic_shape: str) -> str:
    rels = sorted(_RELATIONS_BY_SHAPE.get(topic_shape, set()))
    return ", ".join(rels)


def _build_posture_prompt(topic: str, topic_shape: str, cluster: dict,
                          cluster_evidence: Dict[str, List[str]],
                          topic_cause: str = "",
                          topic_outcome: str = "") -> str:
    if topic_shape == "explanatory":
        return _build_explanation_posture_prompt(topic, cluster, cluster_evidence,
                                                 topic_outcome)
    pin_lines = []
    if topic_shape == "causal" and topic_cause:
        pin_lines.append(f"TOPIC CAUSE (explanans, pinned): {topic_cause}")
    if topic_shape == "causal" and topic_outcome:
        pin_lines.append(f"TOPIC OUTCOME (explanandum, pinned): {topic_outcome}")
    cluster_cause = (cluster.get("shared_cause") or "").strip()
    cluster_cause_line = (
        f"CLUSTER'S DISTINCTIVE CAUSE (from Stage 1): {cluster_cause}"
        if cluster_cause else ""
    )
    lines = [
        f"TOPIC: {topic}",
        f"TOPIC SHAPE (from Stage 0): {topic_shape}",
        *pin_lines,
        "",
        f"CLUSTER LABEL (from Stage 1): {cluster.get('shared_thread','')}",
        cluster_cause_line,
        "",
        "PAPERS IN THIS CLUSTER:",
    ]
    for did in cluster.get("doc_ids", []):
        claim = cluster.get("claims_by_doc", {}).get(did, "")
        lines.append(f"  [{did}]")
        if claim:
            lines.append(f"    CLAIM: {claim}")
        for snip in (cluster_evidence.get(did) or [])[:3]:
            lines.append(f"    EVIDENCE: {snip}")
    lines += [
        "",
        _scaffold_for_shape(topic_shape),
        "",
        "OUTPUT — return ONE JSON object with EXACTLY these keys:",
        "  cluster_cause_quote: VERBATIM substring (>=4 words) from the "
        "cluster's claims naming the cluster's cause. Required for "
        "causal topics; empty string allowed only when relation = "
        "`adjacent` (the cluster genuinely names no cause). For "
        "comparative / descriptive topics this field is empty.",
        "  cluster_outcome_quote: VERBATIM substring (>=4 words) from "
        "the cluster's claims naming the cluster's outcome. Same "
        "requirement as cluster_cause_quote.",
        "  reasoning_trace: a few sentences walking the test order. "
        "Reference your cluster_cause_quote when applying TEST C-THROUGH "
        "and TEST U.",
        f"  relation: one of [{_allowed_relations_block(topic_shape)}]",
        "  elaboration: a 1-2 sentence free-text posture (<=240 chars) in the "
        "literature's own terms — the writer renders this verbatim into the "
        "review section that introduces this cluster, so phrase it as a "
        "substantive claim about the world, not a meta-comment.",
        "  lead_doc_id: doc_id of the paper that best represents the cluster's "
        "argument (must be in the cluster's doc_ids).",
        "  internal_disagreement: <=200 chars or empty string. If members of "
        "this cluster reach DIFFERENT conclusions inside the same thread, "
        "name the disagreement; else leave empty.",
        "",
        "Return ONLY the JSON object.",
    ]
    return _language_prefix() + "\n".join(lines)


def _build_explanation_posture_prompt(topic: str, cluster: dict,
                                      cluster_evidence: Dict[str, List[str]],
                                      topic_outcome: str) -> str:
    explanation = (cluster.get("shared_cause") or "").strip()
    lines = [
        f"TOPIC: {topic}",
        "TOPIC SHAPE (from Stage 0): explanatory",
        f"TOPIC OUTCOME (explanandum, pinned): {topic_outcome}",
        "TOPIC CAUSE: none. The topic asks what explains the outcome.",
        "",
        f"CLUSTER LABEL (from Stage 1): {cluster.get('shared_thread','')}",
        f"CLUSTER'S EXPLANATION (from Stage 1): {explanation or '(none named)'}",
        "",
        "PAPERS IN THIS CLUSTER:",
    ]
    for did in cluster.get("doc_ids", []):
        claim = cluster.get("claims_by_doc", {}).get(did, "")
        lines.append(f"  [{did}]")
        if claim:
            lines.append(f"    CLAIM: {claim}")
        for snip in (cluster_evidence.get(did) or [])[:3]:
            lines.append(f"    EVIDENCE: {snip}")
    lines += [
        "",
        _scaffold_for_shape("explanatory"),
        "",
        "OUTPUT — return ONE JSON object with EXACTLY these keys:",
        "  cluster_cause_quote: VERBATIM substring (>=4 words) naming the "
        "cause this cluster proposes for the TOPIC OUTCOME. Required when "
        "relation = `explains_outcome`; empty string otherwise.",
        "  cluster_outcome_quote: VERBATIM substring (>=4 words) naming "
        "what the cluster says about the TOPIC OUTCOME. Empty string "
        "allowed only when relation = `adjacent`.",
        "  reasoning_trace: a few sentences walking the test order, "
        "referring to your quotes.",
        f"  relation: one of [{_allowed_relations_block('explanatory')}]",
        "  elaboration: a 1-2 sentence free-text posture (<=240 chars) in the "
        "literature's own terms — the writer renders this verbatim into the "
        "review section that introduces this cluster, so phrase it as a "
        "substantive claim about the world, not a meta-comment. For "
        "`explains_outcome`, state the explanation: which cause, acting "
        "how, produced the outcome.",
        "  lead_doc_id: doc_id of the paper that best represents the cluster's "
        "argument (must be in the cluster's doc_ids).",
        "  internal_disagreement: <=200 chars or empty string. If members of "
        "this cluster reach DIFFERENT conclusions inside the same thread, "
        "name the disagreement; else leave empty.",
        "",
        "Return ONLY the JSON object.",
    ]
    return _language_prefix() + "\n".join(lines)


def _explanation_grounded(quote: str, cluster: dict,
                          cluster_evidence: Dict[str, List[str]]) -> bool:
    own_text = " ".join(
        [cluster.get("shared_cause") or "", cluster.get("shared_thread") or ""]
        + [str(c or "") for c in (cluster.get("claims_by_doc") or {}).values()]
        + [str(s or "") for snips in (cluster_evidence or {}).values() for s in (snips or [])]
    )

    def _flat(text: str) -> str:
        return " ".join(str(text or "").casefold().split())

    if not _flat(quote):
        return False
    if _flat(quote) in _flat(own_text):
        return True
    quote_stems = _content_stems(quote)
    if not quote_stems:
        return True
    shared = quote_stems & _content_stems(own_text)
    needed = min(len(quote_stems), max(2, (len(quote_stems) + 1) // 2))
    return len(shared) >= needed


def _validate_posture(obj, topic_shape: str, valid_doc_ids: set) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    relation = str(obj.get("relation", "")).strip().lower()
    if relation not in _RELATIONS_BY_SHAPE.get(topic_shape, set()):
        return None
    elaboration = str(obj.get("elaboration", "") or "").strip()[:280]
    if not elaboration:
        return None
    reasoning = str(obj.get("reasoning_trace", "") or "").strip()[:1200]
    lead = str(obj.get("lead_doc_id", "") or "").strip()
    if lead and lead not in valid_doc_ids:
        lead = ""
    internal = str(obj.get("internal_disagreement", "") or "").strip()[:240]
    cluster_cause_quote = str(obj.get("cluster_cause_quote", "") or "").strip()[:400]
    cluster_outcome_quote = str(obj.get("cluster_outcome_quote", "") or "").strip()[:400]
    if (topic_shape == "causal"
            and relation != "adjacent"
            and (len(cluster_cause_quote) < 10 or len(cluster_outcome_quote) < 10)):
        return None
    if topic_shape == "explanatory":
        if relation == "explains_outcome" and len(cluster_cause_quote) < 10:
            return None
        if (relation in ("documents_outcome", "disputes_outcome")
                and len(cluster_outcome_quote) < 10):
            return None
    return {
        "relation": relation,
        "elaboration": elaboration,
        "reasoning_trace": reasoning,
        "lead_doc_id": lead,
        "internal_disagreement": internal,
        "cluster_cause_quote": cluster_cause_quote,
        "cluster_outcome_quote": cluster_outcome_quote,
    }


def posture_cluster(topic: str, topic_shape: str, cluster: dict,
                    cluster_evidence: Dict[str, List[str]],
                    topic_cause: str = "",
                    topic_outcome: str = "",
                    metrics=None) -> Optional[dict]:
    doc_ids = list(cluster.get("doc_ids") or [])
    claims = [cluster.get("claims_by_doc", {}).get(d, "") for d in doc_ids]
    sig = _sig_posture(
        topic, topic_shape, doc_ids, claims,
        topic_cause=topic_cause, topic_outcome=topic_outcome,
        shared_thread=(cluster.get("shared_thread") or ""),
        shared_cause=(cluster.get("shared_cause") or ""),
        cluster_evidence=cluster_evidence,
    )
    cached = _load_cache("posture", sig)
    if cached and isinstance(cached, dict) and cached.get("relation"):
        if metrics:
            metrics.cache_event("outline_posture", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_posture", "misses")

    prompt = _build_posture_prompt(topic, topic_shape, cluster, cluster_evidence,
                                   topic_cause=topic_cause, topic_outcome=topic_outcome)
    valid_doc_ids = set(doc_ids)
    raw = ""
    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OPTIONS_POSTURE,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("outline_posture", _MODEL, options=_OPTIONS_POSTURE,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("outline_posture", _MODEL, options=_OPTIONS_POSTURE,
                               success=False, error=e)
        return None

    posture = _parse_and_validate(raw, lambda obj: _validate_posture(obj, topic_shape, valid_doc_ids))
    if posture is None:
        retry_prompt = prompt + "\n\nReturn ONLY the JSON object with the five required keys."
        try:
            import ollama
            start = time.perf_counter()
            res = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": retry_prompt}],
                options=_OPTIONS_POSTURE,
                keep_alive=_KEEP_ALIVE,
                format="json",
                stream=False,
            )
            raw = (res.get("message", {}).get("content") or "").strip()
            if metrics:
                metrics.record_llm("outline_posture_retry", _MODEL, options=_OPTIONS_POSTURE,
                                   duration_s=time.perf_counter() - start,
                                   prompt_chars=len(retry_prompt),
                                   response_chars=len(raw))
        except Exception as e:
            if metrics:
                metrics.record_llm("outline_posture_retry", _MODEL, options=_OPTIONS_POSTURE,
                                   success=False, error=e)
            return None
        posture = _parse_and_validate(raw, lambda obj: _validate_posture(obj, topic_shape, valid_doc_ids))
        if posture is None:
            return None

    cluster_cause = (cluster.get("shared_cause") or "").strip()
    cluster_thread = (cluster.get("shared_thread") or "").strip()
    grounding_source = cluster_cause or cluster_thread
    if (topic_shape == "causal"
            and posture.get("relation") == "upstream_of_topic_cause"
            and grounding_source):
        reasoning = posture.get("reasoning_trace", "") or ""
        if _mechanism_grounding_fails(reasoning, grounding_source):
            posture["relation"] = "adjacent"
            posture["mechanism_grounding"] = "downgraded_upstream_to_adjacent"
            if metrics:
                metrics.inc("outline_posture_grounding_downgrades")

    quote_for_conduit = (posture.get("cluster_cause_quote") or "").strip() or grounding_source
    if (topic_shape == "causal"
            and posture.get("relation") in ("rival_to_topic_cause", "adjacent")
            and topic_cause
            and quote_for_conduit):
        if _names_topic_cause_as_conduit(quote_for_conduit, topic_cause):
            original = posture["relation"]
            posture["relation"] = "upstream_of_topic_cause"
            posture["mechanism_grounding"] = f"promoted_{original}_to_upstream_via_conduit_check"
            if metrics:
                metrics.inc("outline_posture_grounding_promotions")
                metrics.inc("outline_posture_conduit_promotions")

    if (topic_shape == "explanatory"
            and posture.get("relation") == "explains_outcome"
            and not _explanation_grounded(posture.get("cluster_cause_quote", ""),
                                          cluster, cluster_evidence)):
        posture["relation"] = "adjacent"
        posture["mechanism_grounding"] = "downgraded_explains_to_adjacent"
        if metrics:
            metrics.inc("outline_posture_grounding_downgrades")

    posture["model"] = _MODEL
    posture["prompt_version"] = _POSTURE_PROMPT_VERSION
    _save_cache("posture", sig, posture)
    if metrics:
        metrics.cache_event("outline_posture", "writes")
    return posture


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
    "and", "or", "but", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "they", "their",
    "them", "we", "us", "our", "i", "me", "my", "you", "your", "he", "she",
    "his", "her", "from", "into", "onto", "upon", "out", "over", "under",
    "between", "through", "via", "than", "then", "thus", "hence", "also",
    "not", "no", "yes", "if", "when", "where", "while", "because", "so",
    "more", "less", "most", "least", "some", "any", "all", "each", "every",
    "such", "same", "different", "other", "another", "one", "two", "first",
    "second", "do", "does", "did", "have", "has", "had", "having",
    "cause", "causes", "caused", "causing",
    "effect", "effects", "explain", "explains", "explained",
    "topic", "cluster", "paper", "papers", "literature", "stream",
    "argument", "claim", "claims", "argues", "argue",
})


def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("ational", "tional", "ization", "ising", "izing",
                "ation", "ition", "ments", "ness", "able", "ible",
                "ity", "ies", "ied", "ing", "ed", "es", "s", "y"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_stems(text: str) -> set:
    if not text:
        return set()
    toks = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    return {_stem(t) for t in toks if t not in _STOPWORDS and len(t) > 3}


def _mechanism_grounding_fails(reasoning: str, grounding_source: str) -> bool:
    r_stems = _content_stems(reasoning)
    g_stems = _content_stems(grounding_source)
    if not g_stems:
        return False
    overlap = r_stems & g_stems
    return len(overlap) < 2


_CONDUIT_VERBS = (
    "led to", "leads to", "leading to",
    "produced", "produces", "producing", "produce",
    "established", "establishes", "establishing",
    "shaped", "shapes", "shaping",
    "caused", "causes", "causing", "cause",
    "generated", "generates", "generating", "generate",
    "affected", "affects", "affecting",
    "determined", "determines", "determining",
    "gave rise to", "gives rise to", "giving rise to",
    "drives", "drove", "driving", "drive",
    "account for", "accounts for",
    "explain", "explains", "explaining",
    "explanation", "explanations",
)


def _names_topic_cause_as_conduit(grounding_source: str, topic_cause: str) -> bool:
    if not grounding_source or not topic_cause:
        return False
    text = grounding_source.lower()
    topic_stems = _content_stems(topic_cause)
    if not topic_stems:
        return False
    for verb in _CONDUIT_VERBS:
        idx = text.find(verb)
        while idx != -1:
            after = text[idx + len(verb):]
            after_stems = _content_stems(after)
            if topic_stems & after_stems:
                return True
            idx = text.find(verb, idx + 1)
    return False


def _build_order_prompt(topic: str, cluster_summaries: List[dict]) -> str:
    lines = [
        f"TOPIC: {topic}",
        "",
        "You have N literature-review sections to order. Each section already "
        "has a relation to the topic and a one-sentence posture. Pick a "
        "narrative order that makes the review read well: open with the most "
        "central streams, close with the most distant. Sections of the same "
        "relation type can be adjacent or interleaved as makes sense.",
        "",
        "SECTIONS:",
    ]
    for cs in cluster_summaries:
        lines.append(
            f"  [{cs.get('cluster_id')}] relation={cs.get('relation')} "
            f"thread={cs.get('shared_thread','')!r} posture={cs.get('elaboration','')!r}"
        )
    lines += [
        "",
        "Return ONE JSON object: { \"ordered_cluster_ids\": [\"C1\", \"C3\", ...] }",
        "Every cluster_id above must appear exactly once.",
    ]
    return _language_prefix() + "\n".join(lines)


def _validate_order(obj, valid_ids: set) -> Optional[List[str]]:
    if not isinstance(obj, dict):
        return None
    raw = obj.get("ordered_cluster_ids") or []
    if not isinstance(raw, list):
        return None
    seen = set()
    out = []
    for x in raw:
        s = str(x).strip()
        if s in valid_ids and s not in seen:
            out.append(s)
            seen.add(s)
    for cid in sorted(valid_ids - seen):
        out.append(cid)
    return out


def order_clusters(topic: str, cluster_summaries: List[dict],
                   metrics=None) -> List[str]:
    ids = [cs.get("cluster_id") for cs in cluster_summaries if cs.get("cluster_id")]
    if len(ids) <= 2:
        return ids

    sig = _sig_order(topic, cluster_summaries)
    cached = _load_cache("order", sig)
    if isinstance(cached, dict) and isinstance(cached.get("ordered_cluster_ids"), list):
        if metrics:
            metrics.cache_event("outline_order", "hits")
        return cached["ordered_cluster_ids"]
    if metrics:
        metrics.cache_event("outline_order", "misses")

    prompt = _build_order_prompt(topic, cluster_summaries)
    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OPTIONS_ORDER,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("outline_order", _MODEL, options=_OPTIONS_ORDER,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("outline_order", _MODEL, options=_OPTIONS_ORDER,
                               success=False, error=e)
        return ids

    parsed = _parse_and_validate(raw, lambda obj: _validate_order(obj, set(ids)))
    if not parsed:
        return ids
    _save_cache("order", sig, {"ordered_cluster_ids": parsed})
    if metrics:
        metrics.cache_event("outline_order", "writes")
    return parsed


def build_outline(topic: str, doc_summaries: List[dict], metrics=None,
                  precheck_result: Optional[dict] = None,
                  diagnostics: Optional[dict] = None) -> Optional[dict]:
    if metrics:
        metrics.set("outline_posture_grounding_downgrades", 0)
        metrics.set("outline_posture_grounding_promotions", 0)
        metrics.set("outline_posture_conduit_promotions", 0)

    pre = precheck_result if (isinstance(precheck_result, dict)
                              and precheck_result.get("corpus_fit")
                              and precheck_result.get("topic_shape")) else None
    if pre is None:
        pre = precheck(topic, doc_summaries, metrics=metrics, diagnostics=diagnostics)
    if not pre:
        if diagnostics is not None:
            diagnostics["stage"] = "stage0"
        return None

    topic_shape = pre["topic_shape"]
    topic_cause = pre.get("topic_cause", "") or ""
    topic_outcome = pre.get("topic_outcome", "") or ""
    if pre["corpus_fit"] == "REFUSE":
        return {
            "refused": True,
            "refusal_reason": "corpus_off_topic",
            "refusal_explanation": pre.get("corpus_fit_rationale", ""),
            "topic": topic,
            "topic_shape": topic_shape,
            "topic_shape_rationale": pre.get("topic_shape_rationale", ""),
            "topic_cause": topic_cause,
            "topic_outcome": topic_outcome,
            "admitted_total": len({d.get("doc_id") for d in doc_summaries if d.get("doc_id")}),
            "clusters": [],
            "unassigned_doc_ids": [d.get("doc_id") for d in doc_summaries if d.get("doc_id")],
            "ordered_cluster_ids": [],
            "relation_distribution": {},
            "unassigned_share": 1.0,
            "model": _MODEL,
            "precheck_prompt_version": _PRECHECK_PROMPT_VERSION,
        }

    plan = cluster_papers(topic, doc_summaries,
                          topic_shape=topic_shape,
                          topic_outcome=topic_outcome,
                          metrics=metrics,
                          diagnostics=diagnostics)
    if not plan:
        if diagnostics is not None:
            diagnostics["stage"] = "stage1"
            diagnostics.setdefault("failure", "no_valid_cluster_plan")
        return None
    summaries_by_doc = {
        d.get("doc_id"): d for d in doc_summaries if d.get("doc_id")
    }

    def _claim_of(did: str) -> str:
        d = summaries_by_doc.get(did) or {}
        return (d.get("claim") or "").strip()

    def _evidence_of(did: str) -> List[str]:
        d = summaries_by_doc.get(did) or {}
        out = []
        for q in (d.get("quotes") or [])[:3]:
            text = (q.get("text") or "").strip()
            if not text:
                continue
            clipped = (text[:220] + "...") if len(text) > 220 else text
            out.append(f"p.{q.get('page', '?')}: {clipped}")
        return out

    enriched_clusters = []
    for c in plan["clusters"]:
        claims_by_doc = {did: _claim_of(did) for did in c["doc_ids"]}
        cluster_for_call = dict(c)
        cluster_for_call["claims_by_doc"] = claims_by_doc
        evidence_by_doc = {did: _evidence_of(did) for did in c["doc_ids"]}
        posture = posture_cluster(topic, topic_shape, cluster_for_call,
                                  evidence_by_doc,
                                  topic_cause=topic_cause,
                                  topic_outcome=topic_outcome,
                                  metrics=metrics)
        if posture is None:
            posture = {
                "relation": "adjacent",
                "elaboration": c.get("shared_thread", "(posture failed)"),
                "reasoning_trace": "(posture call failed; defaulted to adjacent)",
                "lead_doc_id": c["doc_ids"][0] if c["doc_ids"] else "",
                "internal_disagreement": "",
                "posture_failed": True,
            }
        enriched_clusters.append({
            **c,
            "claims_by_doc": claims_by_doc,
            "relation": posture["relation"],
            "elaboration": posture["elaboration"],
            "reasoning_trace": posture.get("reasoning_trace", ""),
            "lead_doc_id": posture.get("lead_doc_id", ""),
            "internal_disagreement": posture.get("internal_disagreement", ""),
            "posture_failed": posture.get("posture_failed", False),
            "cluster_cause_quote": posture.get("cluster_cause_quote", ""),
            "cluster_outcome_quote": posture.get("cluster_outcome_quote", ""),
            "mechanism_grounding": posture.get("mechanism_grounding", ""),
        })

    cluster_summaries_for_ordering = [
        {
            "cluster_id": c["cluster_id"],
            "relation": c["relation"],
            "elaboration": c["elaboration"],
            "shared_thread": c["shared_thread"],
        }
        for c in enriched_clusters
    ]
    ordered_ids = order_clusters(topic, cluster_summaries_for_ordering, metrics=metrics)

    total_admitted = len({d.get("doc_id") for d in doc_summaries if d.get("doc_id")})
    n_unassigned = len(plan.get("unassigned_doc_ids") or [])
    relation_counts: Dict[str, int] = {}
    for c in enriched_clusters:
        relation_counts[c["relation"]] = relation_counts.get(c["relation"], 0) + len(c["doc_ids"])

    return {
        "topic": topic,
        "topic_shape": topic_shape,
        "topic_cause": topic_cause,
        "topic_outcome": topic_outcome,
        "topic_shape_rationale": pre.get("topic_shape_rationale", ""),
        **{key: pre[key] for key in ("shape_adjusted", "model_topic_shape",
                                     "model_topic_cause") if key in pre},
        "clusters": enriched_clusters,
        "unassigned_doc_ids": plan.get("unassigned_doc_ids") or [],
        "ordered_cluster_ids": ordered_ids,
        "admitted_total": total_admitted,
        "unassigned_share": round(n_unassigned / total_admitted, 3) if total_admitted else 0.0,
        "relation_distribution": relation_counts,
        "model": _MODEL,
        "precheck_prompt_version": _PRECHECK_PROMPT_VERSION,
        "cluster_prompt_version": _CLUSTER_PROMPT_VERSION,
        "posture_prompt_version": _POSTURE_PROMPT_VERSION,
        "order_prompt_version": _ORDER_PROMPT_VERSION,
    }


def _parse_and_validate(raw: str, validate_fn):
    obj = _parse_json_object(raw)
    if obj is None:
        return None
    try:
        return validate_fn(obj)
    except Exception:
        return None


def _parse_json_object(raw: str):
    if not raw:
        return None
    from rrr.utils import extract_first_json
    obj = extract_first_json(raw)
    if obj is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        payload = raw[start:end + 1]
        payload = re.sub(r",\s*}", "}", payload)
        payload = re.sub(r",\s*]", "]", payload)
        try:
            obj = json.loads(payload)
        except Exception:
            return None
    return obj
