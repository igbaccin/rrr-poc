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

_PRECHECK_PROMPT_VERSION = "2026-07-02-v15.14-precheck"
_CLUSTER_PROMPT_VERSION = "2026-07-02-v15.14-cluster"
_POSTURE_PROMPT_VERSION = "2026-07-02-v15.14-posture"
_ORDER_PROMPT_VERSION = "2026-07-02-v15.14-order"


def _lang_sig() -> bytes:
    return (os.environ.get("RRR_CORPUS_LANG", "en") or "en").encode("utf-8")

_OPTIONS_PRECHECK = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 400}
_OPTIONS_CLUSTER = {"temperature": 0.0, "num_ctx": 12288, "num_predict": 1800}
_OPTIONS_POSTURE = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 600}
_OPTIONS_ORDER = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300}


_TOPIC_SHAPES = ("causal", "comparative", "descriptive")

_RELATIONS_BY_SHAPE: Dict[str, set] = {
    "causal": {
        "same_as_topic_cause",
        "upstream_of_topic_cause",
        "downstream_of_topic_cause",
        "rival_to_topic_cause",
        "scope_condition",
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
        "  - causal: the topic asserts that some variable X CAUSES outcome Y, "
        "or that X is the fundamental EXPLANATION for some phenomenon. "
        "Hallmark: the topic names a cause and an effect.\n"
        "      Examples: \"Sleep deprivation impairs immune function.\" "
        "\"Catalyst surface area determines reaction rate.\" "
        "\"Phonemic awareness in early childhood predicts later reading skill.\"\n\n"
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
        "as a QUESTION: does it ask 'what causes X?' (causal), 'is A more X "
        "than B?' (comparative), or 'what is/are X?' (descriptive).\n\n"
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
        "## PART 3 — CAUSAL SLOTS (skip if topic_shape != causal)\n"
        "If you picked topic_shape=\"causal\", also identify the topic's "
        "EXPLANATORY structure as two pinned slots that every downstream "
        "stage will reference verbatim:\n\n"
        "  - topic_cause: a SHORT noun phrase (<=80 chars) for the variable "
        "the topic claims is doing the EXPLAINING — the explanans.\n"
        "  - topic_outcome: a SHORT noun phrase (<=80 chars) for the "
        "phenomenon the topic claims is being EXPLAINED — the explanandum.\n\n"
        "These slots are the most important fields you produce. Every "
        "downstream classification of an individual paper's relation to the "
        "topic is anchored on these two strings. Be CONCRETE and use the "
        "topic's own wording where possible.\n\n"
        "Examples (cross-domain, on purpose):\n"
        "  Topic \"Sleep deprivation impairs immune function.\" -> "
        "topic_cause=\"sleep deprivation\", topic_outcome=\"impaired immune "
        "function\".\n"
        "  Topic \"Phonemic awareness in early childhood predicts later "
        "reading skill.\" -> topic_cause=\"phonemic awareness in early "
        "childhood\", topic_outcome=\"later reading skill\".\n"
        "  Topic \"Mediterranean diet reduces cardiovascular disease risk.\" "
        "-> topic_cause=\"mediterranean diet\", topic_outcome=\"reduced "
        "cardiovascular disease risk\".\n\n"
        "For comparative or descriptive topics leave both fields as empty "
        "strings — they are not used.\n\n"
        "## OUTPUT\n"
        "Return ONE JSON object with EXACTLY these six keys:\n"
        "  topic_shape: \"causal\" | \"comparative\" | \"descriptive\"\n"
        "  topic_shape_rationale: one sentence (<=200 chars) defending the "
        "shape choice, quoting the topic's phrasing.\n"
        "  corpus_fit: \"PROCEED\" | \"REFUSE\"\n"
        "  corpus_fit_rationale: one sentence (<=240 chars) — if REFUSE, name "
        "the domain mismatch concretely.\n"
        "  topic_cause: noun phrase (<=80 chars) or empty string\n"
        "  topic_outcome: noun phrase (<=80 chars) or empty string\n\n"
        "Return ONLY the JSON object."
    )


def _validate_precheck(obj) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    shape = str(obj.get("topic_shape", "")).strip().lower()
    if shape not in _TOPIC_SHAPES:
        return None
    fit = str(obj.get("corpus_fit", "")).strip().upper()
    if fit not in ("PROCEED", "REFUSE"):
        return None
    topic_cause = str(obj.get("topic_cause", "") or "").strip()[:120]
    topic_outcome = str(obj.get("topic_outcome", "") or "").strip()[:120]
    if shape == "causal" and fit == "PROCEED":
        if not topic_cause or not topic_outcome:
            return None
    return {
        "topic_shape": shape,
        "topic_shape_rationale": str(obj.get("topic_shape_rationale", "") or "").strip()[:300],
        "corpus_fit": fit,
        "corpus_fit_rationale": str(obj.get("corpus_fit_rationale", "") or "").strip()[:400],
        "topic_cause": topic_cause,
        "topic_outcome": topic_outcome,
    }


def precheck(topic: str, doc_summaries: List[dict], metrics=None) -> Optional[dict]:
    titles = []
    for d in doc_summaries:
        did = (d.get("doc_id") or "").strip()
        if not did:
            continue
        cite = (d.get("citation") or did).strip()
        cite = " ".join(cite.split())[:240]
        titles.append(cite)
    if not titles:
        return None

    sig = _sig_precheck(topic, titles)
    cached = _load_cache("precheck", sig)
    if cached and isinstance(cached, dict) and cached.get("corpus_fit"):
        if metrics:
            metrics.cache_event("outline_precheck", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_precheck", "misses")

    prompt = _build_precheck_prompt(topic, titles)
    raw = ""
    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_OPTIONS_PRECHECK,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("outline_precheck", _MODEL, options=_OPTIONS_PRECHECK,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("outline_precheck", _MODEL, options=_OPTIONS_PRECHECK,
                               success=False, error=e)
        return None

    result = _parse_and_validate(raw, _validate_precheck)
    if result is None:
        return None
    result["model"] = _MODEL
    result["prompt_version"] = _PRECHECK_PROMPT_VERSION
    _save_cache("precheck", sig, result)
    if metrics:
        metrics.cache_event("outline_precheck", "writes")
    return result


def _build_cluster_prompt(topic: str, doc_claims: List[Dict[str, str]],
                          topic_shape: str = "causal",
                          topic_outcome: str = "") -> str:
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
                   metrics=None) -> Optional[dict]:
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
        return None

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
            return None
        plan = _parse_and_validate(raw, lambda obj: _validate_cluster_plan(obj, valid_doc_ids, topic_shape))
        if plan is None:
            return None

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


def build_outline(topic: str, doc_summaries: List[dict], metrics=None) -> Optional[dict]:
    if metrics:
        metrics.set("outline_posture_grounding_downgrades", 0)
        metrics.set("outline_posture_grounding_promotions", 0)
        metrics.set("outline_posture_conduit_promotions", 0)

    pre = precheck(topic, doc_summaries, metrics=metrics)
    if not pre:
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
                          metrics=metrics)
    if not plan:
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
    try:
        return validate_fn(obj)
    except Exception:
        return None
