"""v15: corpus-level outline builder.

Replaces v14.4's per-paper fused stance call with a three-stage corpus-level
pipeline:

  Stage 1 (CLUSTER): one LLM call over all admitted papers' claims. Returns
    `topic_shape` + clusters (papers grouped by what they argue, before any
    relation-to-topic judgement) + unassigned papers.

  Stage 2 (POSTURE): one LLM call per cluster. Returns the cluster's
    structural relation to the topic (from a per-shape enum), a free-text
    elaboration the writer uses verbatim, plus a `reasoning_trace` that
    forces the model to ground the relation tag in an explicit quote from
    the cluster's claims. No `supports`/`critiques`/`complicates` vocabulary
    survives into the writer's prose.

  Stage 3 (ORDER): one LLM call over the N cluster summaries to pick
    section order. Rule-based when there are <=2 clusters.

Topic shapes (3) and their relation enums:

  causal       — topic asserts X causes Y.
                 relations: same_as_topic_cause, upstream_of_topic_cause,
                 downstream_of_topic_cause, rival_to_topic_cause,
                 scope_condition, adjacent
  comparative  — topic asserts A differs from / is better than B.
                 relations: endorses_topic, reverses_topic, qualifies_topic,
                 methodological_critique, adjacent
  descriptive  — topic asserts X has property/pattern P.
                 relations: confirms_description, contradicts_description,
                 adds_nuance, adjacent

The 6-value causal enum is the load-bearing fix for the v14.4 failure
(AJR/Nunn upstream causes were rounded to `critiques`). Per-shape rubrics
keep the prompt topic-agnostic — same code works for gender-wage-gap,
monetary-policy, sociology-of-religion topics by simply landing in a
different shape.
"""

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

# Bump on prompt changes to invalidate downstream caches.
# v15.0.1: cluster prompt rewritten to encourage INCLUSIVE clustering.
# v15.1.0: shape detection + corpus-fit refusal moved OUT of Stage 1 into
#   a new Stage 0 (precheck). Stage 1 now consumes topic_shape as input
#   rather than re-detecting it, freeing it to specialise the clustering
#   prompt by shape AND removing the v15.0.2 failure modes:
#     - descriptive topics were misdetected as causal 3/3 (the menu was
#       inside the cluster prompt, where it lost attention)
#     - off-topic corpora wrote 1.4k-word hallucinated reviews because
#       inclusive-clustering force-fit everything into clusters and
#       unassigned_share never reached the refusal threshold
# v15.14: bumped — (a) the v15.12 _language_prefix rewrite changed prompt
# bytes without a version bump, so v15.11-keyed caches could serve stale
# postures; (b) the signatures below now cover ALL prompt-shaping inputs
# (corpus lang, topic_cause/outcome, shared_thread/cause, evidence), which
# changes key layout.
_PRECHECK_PROMPT_VERSION = "2026-07-02-v15.14-precheck"
_CLUSTER_PROMPT_VERSION = "2026-07-02-v15.14-cluster"
_POSTURE_PROMPT_VERSION = "2026-07-02-v15.14-posture"
_ORDER_PROMPT_VERSION = "2026-07-02-v15.14-order"


def _lang_sig() -> bytes:
    """v15.14: corpus language participates in every stage prompt via
    _language_prefix(), so it must participate in every cache key too —
    otherwise switching RRR_CORPUS_LANG replays caches built under a
    different prompt."""
    return (os.environ.get("RRR_CORPUS_LANG", "en") or "en").encode("utf-8")

# v15.1.0: Stage 0 (precheck) sees only the topic + paper titles, so
# context can be small. Cheap call.
_OPTIONS_PRECHECK = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 400}
# Ample num_predict on Stage 1 because the JSON may contain ~50 doc_ids
# distributed across 3-6 clusters.
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


# ---------------------------------------------------------------------------
# Cache plumbing — one cache directory per stage. Cache files live under
# runs/cache/outline/<stage>/<sig>.json. Keys are stable hashes; a prompt
# version bump invalidates lookups without disk cleanup.

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
    # v15.14: shape + outcome condition the cluster prompt (pin lines) and
    # were missing from the key — a re-run with a re-pinned Stage 0 silently
    # replayed the old clustering.
    h.update(b"\x04")
    h.update((topic_shape or "").encode("utf-8"))
    h.update(b"\x05")
    h.update((topic_outcome or "").encode("utf-8"))
    # Order-independent over docs: sort by doc_id.
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
    # v15.14: everything else the posture prompt renders now participates in
    # the key. Previously a re-run where Stage 0 pinned different
    # cause/outcome slots, or where evidence extraction changed, silently
    # reused the old posture.
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
        # v15.14: shared_thread is rendered into the order prompt.
        h.update(b"\x04")
        h.update((cs.get("shared_thread") or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(stage: str, sig: str) -> str:
    # v15.16: workspace-level (was runs_path("cache", ...) — per-run under
    # the v15.9 minted-run-id layout, so nothing was ever reused).
    return str(stage_cache_path("outline", stage, f"{sig}.json"))


def _load_cache(stage: str, sig: str):
    if not stage_cache_enabled():
        return None  # RRR_STAGE_CACHE=0: cold-run measurement mode
    try:
        with open(_cache_path(stage, sig), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(stage: str, sig: str, obj) -> None:
    # v15.14: atomic — a crash mid-write left a truncated cache entry
    # (self-healing on load, but pointlessly re-paying the LLM call).
    ensure_dir(str(stage_cache_path("outline", stage)))
    path = _cache_path(stage, sig)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage 0 — PRECHECK (corpus-fit refusal + topic shape detection)
#
# v15.1.0: pulled out of Stage 1 where it was a side-feature buried under
# clustering and routinely failed (descriptive 0/3, refusal 0/3 on the
# v15.0.2 smoke). Now its own focused call.

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
    """v15.12: internal outline stages (precheck, cluster, posture, order)
    run in the CORPUS/pivot language, not the topic language. This is the
    key fix for the v15.11 mixed-language bug: applying the topic-language
    directive to format=json stages made mistral emit half-English/half-
    French JSON values that then leaked into the writer. Internal reasoning
    now stays in the corpus language (where the model + evidence are
    strongest and JSON compliance is reliable); only the writer translates
    to the topic language for the user-facing review.

    Empty for an English corpus (byte-identical to pre-v15.11 prompts).
    """
    from rrr.language import language_directive
    directive = language_directive(os.environ.get("RRR_CORPUS_LANG", "en"))
    return f"{directive}\n\n" if directive else ""


def _build_precheck_prompt(topic: str, doc_titles: List[str]) -> str:
    _lang_pfx = _language_prefix()
    titles_block = "\n".join(f"  - {t}" for t in doc_titles)
    # NOTE on examples: every illustrative example below is drawn from a
    # domain DELIBERATELY unrelated to any topic this pipeline is
    # commonly tested on. The point of the examples is to demonstrate the
    # SHAPE PATTERN, not the subject matter. Adding examples whose subject
    # overlaps the user's actual topic would nudge the model's classification
    # in a non-agnostic way. Keep new examples cross-domain.
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
    # v15.2.0: causal topics MUST emit non-empty topic_cause + topic_outcome
    # (the pinned explanans/explanandum that Stage 1 outcome-anchors on and
    # Stage 2 pins as fixed reference points). Comparative/descriptive shapes
    # leave them empty until v15.3.0 extends per-shape slot extraction.
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
    """Stage 0. One cheap LLM call over topic + paper titles. Returns
    {topic_shape, corpus_fit, ...} or None on failure.

    Decoupling shape detection and refusal from clustering avoids two
    failure modes seen in the v15.0.2 smoke:
      * shape detection burying inside Stage 1 -> 0/3 descriptive detected
      * inclusive-clustering Stage 1 -> unassigned_share=0, refusal dead
    """
    titles = []
    for d in doc_summaries:
        did = (d.get("doc_id") or "").strip()
        if not did:
            continue
        cite = (d.get("citation") or did).strip()
        # Drop trailing whitespace/newlines and clip to keep prompt tight.
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


# ---------------------------------------------------------------------------
# Stage 1 — CLUSTER

def _build_cluster_prompt(topic: str, doc_claims: List[Dict[str, str]],
                          topic_shape: str = "causal",
                          topic_outcome: str = "") -> str:
    # v15.5: prompt stripped to a single qualitative principle. The
    # earlier versions accumulated quantitative pressures (3-6 cluster
    # target, 0-15% unassigned ceiling, "specific enough" thread rules,
    # "moderately broad" countervailing guidance) that conflicted and
    # forced the model to either over-split (v15.3.0 -> 12 narrow
    # clusters) or over-reject (v15.2.0 -> 0.5 unassigned share). The
    # new prompt asks the model to identify the streams of literature it
    # sees, and stops dictating count, coverage, or specificity.
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
                # v15.2.0: shared_cause is the cluster's distinctive
                # explanatory variable for causal topics; empty for other
                # shapes. Stage 2 uses this to pin the cluster's cause when
                # judging its relation to topic_cause.
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

    # Any doc_id that the model dropped (neither clustered nor unassigned) is
    # added to unassigned so the partition is total.
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
    """Stage 1. One LLM call. Returns ClusterPlan dict or None on failure.

    Input: doc_summaries with at least {doc_id, claim} per entry. Other
    fields are ignored. claims-only input keeps the prompt short so the
    model's attention is on the grouping decision, not on evidence weighing
    (which comes in Stage 2).
    """
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
        # One retry with an explicit reminder about the JSON contract.
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


# ---------------------------------------------------------------------------
# Stage 2 — POSTURE (per cluster)


def _scaffold_for_shape(topic_shape: str) -> str:
    if topic_shape == "causal":
        # v15.2.0: scaffold rebuilt to anchor on the EXPLANANDUM (the shared
        # outcome both causes purport to explain) and to put the rival test
        # FIRST. The v15.0.x scaffold gave upstream a positive mechanical
        # test ("flows INTO") but only a negative description for rival
        # ("REPLACES and does NOT operate through"), and added an asymmetric
        # "Key trap" warning that pushed every cluster away from rival. The
        # v15.0.2 GD smoke confirmed that bias: North_1989 was upstream 9/9
        # on a topic where it is the canonical rival, and AJR_2002 collapsed
        # to upstream in 3/9 runs. Rival is now Test R, applied first.
        # v15.2.1: scaffold reordered to make TEST U reachable. v15.2.0 put
        # TEST R first AND demanded TEST U cite cluster_cause_quote /
        # cluster_outcome_quote fields that the output schema never asked
        # for. Net effect: TEST U was mechanically unreachable, every
        # cluster fell through to TEST R, and clusters whose own
        # shared_cause string named the topic cause as a downstream noun
        # ("...led to extractive institutions" where topic_cause is
        # "institutions") were mislabelled rival. v15.2.1 fix:
        #   - The two quote fields are now REQUIRED output keys (enforced
        #     by _build_posture_prompt + _validate_posture).
        #   - New TEST C-THROUGH fires FIRST and asks "does the cluster's
        #     own cause quote name the topic cause as a downstream noun?"
        #     If yes, we are in the upstream/same-as zone and TEST R is
        #     off the table.
        #   - TEST R is preserved but only fires after TEST C-THROUGH /
        #     TEST S / TEST U have all failed — keeps the AJR-on-GD win
        #     intact (AJR's GD cluster names "institutions" as the
        #     conduit, not "Europe unique factor endowments", so
        #     TEST C-THROUGH does not fire and the cluster falls through
        #     to rival correctly).
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
    # descriptive
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
    # v15.2.0: pin TOPIC CAUSE and TOPIC OUTCOME at the prompt header for
    # causal topics. These are the explanans/explanandum extracted in
    # Stage 0 — every cluster's relation judgment must anchor on these
    # fixed strings, not re-derive them.
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
        # v15.2.1: cluster_cause_quote / cluster_outcome_quote are NOW
        # required output keys for causal topics. v15.2.0 mentioned them in
        # the scaffold but never asked for them in the schema, so they
        # arrived as NULL on every cluster and TEST U was mechanically
        # unreachable. Required for causal topics; for comparative /
        # descriptive shapes they remain empty strings.
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
    # v15.2.1: cluster_cause_quote + cluster_outcome_quote captured here
    # so they actually land in the ledger (v15.2.0 mentioned them in the
    # scaffold but never extracted them, so TEST U was unreachable).
    cluster_cause_quote = str(obj.get("cluster_cause_quote", "") or "").strip()[:400]
    cluster_outcome_quote = str(obj.get("cluster_outcome_quote", "") or "").strip()[:400]
    # For causal topics with a non-adjacent relation, require both quote
    # fields to be present (>=20 chars as a cheap "is this a real quote"
    # check). If empty, the posture call short-circuited; better to retry
    # than to ship a relation decision with no evidentiary anchor.
    # v15.2.2: lowered the quote-length minimum from 20 -> 10 chars.
    # The v15.2.1 smoke showed adjacent ballooning to 8/12 clusters on
    # causal topics because the model was producing short quotes that
    # failed the 20-char gate, then the validator rejected, retry hit the
    # same gate, and posture defaulted to adjacent. 10 chars still
    # excludes empty/single-word fields but admits realistic quote
    # phrases like "led to growth" or "shaped institutions".
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
    """Stage 2. One LLM call per cluster.

    cluster: {cluster_id, doc_ids, shared_thread, claims_by_doc}
    cluster_evidence: {doc_id: [snippet, ...]}
    """
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
        # One retry, JSON-only reminder.
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

    # v15.2.0: deterministic mechanism-grounding post-check. When the
    # model picks `upstream_of_topic_cause` for a causal topic, verify the
    # reasoning_trace actually shares enough content words with the
    # cluster's claims to defend the "flows INTO" chain. If the model is
    # template-filling rather than reading the cluster, downgrade to
    # `adjacent` (the safe option — admits uncertainty rather than create
    # false rivals). Only applies to causal topics where we have a
    # cluster_cause string to ground against.
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

    # v15.2.1 + v15.2.3: SYMMETRIC post-check. Promote a non-upstream
    # relation to upstream when the cluster's own quote names the
    # topic_cause as a DOWNSTREAM NOUN after a conduit verb. v15.2.1
    # caught the rival case; v15.2.3 widens to ALSO catch `adjacent`
    # because the v15.2.2 INST smoke showed mistral-small landing AJR in
    # adjacent (not rival) when its TEST C-THROUGH read failed on
    # literal-string mismatch ("extractive institutions" != "institutions"
    # to the model), so the rival-only post-check never fired. The
    # deterministic stem-overlap check correctly sees institutions in
    # extractive institutions.
    #
    # Grounding source: prefer the VERBATIM cluster_cause_quote the model
    # produced in Stage 2 over the Stage-1 shared_cause label, because
    # the quote is what the model actually based its decision on. Fall
    # back to shared_cause when the quote is empty (e.g. adjacent
    # responses that skipped the quote field).
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
                # v15.14: this counter was initialised in build_outline but
                # never incremented — batteries reading it always saw 0.
                metrics.inc("outline_posture_conduit_promotions")

    posture["model"] = _MODEL
    posture["prompt_version"] = _POSTURE_PROMPT_VERSION
    _save_cache("posture", sig, posture)
    if metrics:
        metrics.cache_event("outline_posture", "writes")
    return posture


# ---------------------------------------------------------------------------
# v15.2.0: mechanism-grounding sanity check (deterministic, no LLM).
# When the model picks `upstream_of_topic_cause`, its reasoning_trace
# must share at least 2 content-word stems with the cluster's
# shared_cause/shared_thread. Otherwise it's template-filling and gets
# downgraded to `adjacent` (safe — don't create false rivals).

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
    """Cheap suffix-stripper. Good enough to map plurals/tenses to the
    same key (institutions -> institut, slavery -> slaver, growing ->
    grow, mortality -> mortal). No NLP dependency."""
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
    """Return True when the model's reasoning_trace shares fewer than 2
    content-word stems with the cluster's shared_cause / shared_thread.
    A True return means the upstream claim is not grounded in the
    cluster's own language and should be downgraded to adjacent."""
    r_stems = _content_stems(reasoning)
    g_stems = _content_stems(grounding_source)
    if not g_stems:
        return False  # nothing to ground against; let the LLM call stand
    overlap = r_stems & g_stems
    return len(overlap) < 2


# v15.2.1: conduit verbs that, when a cluster's shared_cause text contains
# `<conduit_verb> ... <topic_cause_stem>`, signal the cluster is naming
# the topic cause as a DOWNSTREAM result of its own cause — i.e. the
# cluster is making an upstream argument. Detected post-Stage-2 to catch
# rival-template-fires where the cluster's own text names the through-
# chain explicitly (e.g. shared_cause="...led to the establishment of
# extractive institutions" with topic_cause="institutions"). The verb
# list is intentionally narrow and topic-agnostic.
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
    # v15.2.3: added "explain" family. The Sokoloff & Engerman cluster on
    # INST has shared_cause "distributional conflicts provide a better
    # explanation than efficiency for the core economic institutions" —
    # syntactically the topic-cause noun appears AFTER `explanation` even
    # though "explanation for" is not contiguous. Bare `explanation`
    # plus the stem-overlap check is enough — if topic_cause is mentioned
    # downstream of the word `explanation` in any phrasing, the cluster
    # is naming it as the explanandum. Topic-agnostic.
    "account for", "accounts for",
    "explain", "explains", "explaining",
    "explanation", "explanations",
)


def _names_topic_cause_as_conduit(grounding_source: str, topic_cause: str) -> bool:
    """Return True when `grounding_source` (the cluster's shared_cause or
    shared_thread) contains the topic_cause as a downstream noun after a
    conduit verb. Used to PROMOTE rival_to_topic_cause -> upstream_of_topic_cause
    when the cluster's own summary names the topic cause as the conduit
    of its own argument."""
    if not grounding_source or not topic_cause:
        return False
    text = grounding_source.lower()
    topic_stems = _content_stems(topic_cause)
    if not topic_stems:
        return False
    # Find any conduit verb in the cluster summary, then check whether
    # the text AFTER the verb contains any topic_cause stem.
    for verb in _CONDUIT_VERBS:
        idx = text.find(verb)
        while idx != -1:
            after = text[idx + len(verb):]
            after_stems = _content_stems(after)
            if topic_stems & after_stems:
                return True
            idx = text.find(verb, idx + 1)
    return False


# ---------------------------------------------------------------------------
# Stage 3 — ORDER


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
    # Fill in any leftovers in their original order so the partition is total.
    for cid in sorted(valid_ids - seen):
        out.append(cid)
    return out


def order_clusters(topic: str, cluster_summaries: List[dict],
                   metrics=None) -> List[str]:
    """Stage 3. Pick section order. Returns ordered cluster_ids.

    With <=2 clusters, returns the input order verbatim (no LLM call needed).
    """
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


# ---------------------------------------------------------------------------
# Top-level orchestrator


def build_outline(topic: str, doc_summaries: List[dict], metrics=None) -> Optional[dict]:
    """Run Stage 0 (precheck) + Stage 1 + Stage 2 + Stage 3.

    Returns:
      - {"refused": True, "refusal_reason": ..., "topic_shape": ...} when
        Stage 0 says corpus_fit=REFUSE. Caller should write a refusal
        manifest and stop.
      - Full outline plan on success.
      - None if Stage 0 or Stage 1 fails irrecoverably (LLM error).
    """
    # v15.7: initialise grounding-postcheck counters at 0 so the 9-battery
    # can distinguish "never fired" from "fired with zero hits". They were
    # previously absent from run_metrics.json when the postchecks took the
    # zero-hit branch.
    if metrics:
        metrics.set("outline_posture_grounding_downgrades", 0)
        metrics.set("outline_posture_grounding_promotions", 0)
        metrics.set("outline_posture_conduit_promotions", 0)

    # Stage 0: corpus-fit + shape detection.
    pre = precheck(topic, doc_summaries, metrics=metrics)
    if not pre:
        # Stage 0 LLM failure. Treat as a hard error rather than
        # silently defaulting to causal+PROCEED — that's how the v15.0.2
        # refusal failure happened.
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

    # Build claims_by_doc + evidence_by_doc once.
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
            # Conservative fallback: tag the cluster as `adjacent` with a
            # generic elaboration so the rest of the pipeline does not crash.
            # The smoke will surface the failure via metrics.outline_posture
            # error counts.
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
            # v15.2.2: carry the v15.2.1 quote fields + mechanism_grounding
            # signal into the ledger so the smoke harness can actually
            # SEE what Stage 2 produced. The v15.2.1 smoke had 0/6
            # quotes "populated" only because this propagation was
            # missing — the validator HAD extracted them, but they died
            # at this boundary.
            "cluster_cause_quote": posture.get("cluster_cause_quote", ""),
            "cluster_outcome_quote": posture.get("cluster_outcome_quote", ""),
            "mechanism_grounding": posture.get("mechanism_grounding", ""),
        })

    # Stage 3.
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


# ---------------------------------------------------------------------------
# JSON helpers


def _parse_and_validate(raw: str, validate_fn):
    """Tolerant JSON parsing: extract the first complete JSON object, parse
    it, run the validator. Returns whatever the validator returns, or None on
    any error.

    v15.13: uses the bracket-matching extractor (utils.extract_first_json) so
    verbose models that emit prose after the object — qwen3:30b-a3b, or a
    frontier API model appending an explanation — no longer break parsing.
    Falls back to the old first-{ / last-} slice + trailing-comma repair when
    the strict extractor finds nothing (handles the trailing-comma case the
    strict json.loads would reject).
    """
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
        # Strip trailing commas that the model sometimes emits.
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
