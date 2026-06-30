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
from typing import List, Dict, Any, Optional

from rrr.paths import runs_path
from rrr.utils import ensure_dir


_MODEL = os.environ.get(
    "RRR_OUTLINE_MODEL",
    os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral")),
)
_KEEP_ALIVE = "30m"

# Bump on prompt changes to invalidate downstream caches.
# v15.0.1: cluster prompt rewritten to encourage INCLUSIVE clustering
# (v15.0 smoke run-1 had unassigned_share=0.58 because the v15a prompt
# gave the model too much permission to dump borderline papers into
# unassigned). Threshold default also raised from 0.5 to 0.7 in
# reasoner.py to match.
_CLUSTER_PROMPT_VERSION = "2026-06-30-v15b-cluster-inclusive"
_POSTURE_PROMPT_VERSION = "2026-06-30-v15a-posture"
_ORDER_PROMPT_VERSION = "2026-06-30-v15a-order"

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

def _sig_cluster(topic: str, doc_claims: List[Dict[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(_CLUSTER_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    # Order-independent over docs: sort by doc_id.
    for entry in sorted(doc_claims, key=lambda d: d.get("doc_id", "")):
        h.update(b"\x01")
        h.update(str(entry.get("doc_id", "")).encode("utf-8"))
        h.update(b"\x02")
        h.update((entry.get("claim") or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _sig_posture(topic: str, topic_shape: str, cluster_doc_ids: List[str],
                 cluster_claims: List[str]) -> str:
    h = hashlib.sha256()
    h.update(_POSTURE_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    h.update(b"\x01")
    h.update((topic_shape or "").encode())
    for did, claim in sorted(zip(cluster_doc_ids, cluster_claims), key=lambda p: p[0]):
        h.update(b"\x02")
        h.update(str(did).encode("utf-8"))
        h.update(b"\x03")
        h.update((claim or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _sig_order(topic: str, cluster_summaries: List[Dict[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(_ORDER_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update((_MODEL or "").encode())
    h.update(b"\x00")
    h.update((topic or "").encode("utf-8"))
    for cs in sorted(cluster_summaries, key=lambda c: c.get("cluster_id", "")):
        h.update(b"\x01")
        h.update((cs.get("cluster_id") or "").encode())
        h.update(b"\x02")
        h.update((cs.get("relation") or "").encode())
        h.update(b"\x03")
        h.update((cs.get("elaboration") or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(stage: str, sig: str) -> str:
    return str(runs_path("cache", "outline", stage, f"{sig}.json"))


def _load_cache(stage: str, sig: str):
    try:
        with open(_cache_path(stage, sig), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(stage: str, sig: str, obj) -> None:
    ensure_dir(str(runs_path("cache", "outline", stage)))
    with open(_cache_path(stage, sig), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stage 1 — CLUSTER

def _build_cluster_prompt(topic: str, doc_claims: List[Dict[str, str]]) -> str:
    lines = [
        "You are organising a corpus of academic papers for a literature review.",
        "",
        f"TOPIC: {topic}",
        "",
        "Below are the central claims of every paper that survived initial "
        "retrieval. Your task is to (a) classify the TOPIC SHAPE and (b) group "
        "the papers into clusters by what they ARGUE.",
        "",
        "TOPIC SHAPES (choose one):",
        "  - causal: the topic asserts that some variable X causes outcome Y, "
        "or that X is the fundamental explanation for some phenomenon.",
        "  - comparative: the topic asserts that A differs from B, A is better "
        "than B, A is more X than B, or otherwise compares two things.",
        "  - descriptive: the topic asks what something is/looks like, what "
        "pattern is observed, what features X has — no single causal claim.",
        "",
        "CLUSTERING RULES:",
        "  - Aim for 3 to 6 clusters that TOGETHER cover the BULK of the "
        "corpus. At least 80% of papers should land in some cluster. Most "
        "academic literatures have substantial overlap; default to FITTING "
        "papers into the nearest cluster, not to excluding them.",
        "  - Group papers that argue SIMILAR things — same causal mechanism, "
        "same comparison verdict, same descriptive account. Do NOT cluster by "
        "their relationship to the topic (yet — that comes later).",
        "  - Threads can be BROAD. A broad thread that captures 6 papers (e.g. "
        "\"institutional persistence and long-run growth\") is BETTER than a "
        "narrow thread that captures 2 (e.g. \"colonial-era property rights "
        "in 19th-century West Africa\"). When in doubt, widen the thread.",
        "  - Each cluster gets a SHORT shared_thread label (5-10 words) that "
        "names what the cluster's papers have in common.",
        "  - Use `unassigned_doc_ids` ONLY for papers that genuinely address "
        "a completely different question — a different intellectual domain, "
        "a different subject matter, or a different unit of analysis. A "
        "paper that is RELEVANT but harder to place than others belongs in "
        "the nearest cluster. Do NOT use unassigned as an \"uncertain\" "
        "bucket.",
        "  - Every doc_id must appear EXACTLY ONCE (either inside one cluster "
        "or in unassigned_doc_ids).",
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
        "  topic_shape: one of causal|comparative|descriptive",
        "  clusters: array of {cluster_id, doc_ids, shared_thread}",
        "    - cluster_id: a short tag like \"C1\", \"C2\", ...",
        "    - doc_ids: array of doc_id strings from the list above",
        "    - shared_thread: 5-10 word noun phrase",
        "  unassigned_doc_ids: array of doc_id strings",
        "",
        "Return ONLY the JSON object. No commentary.",
    ]
    return "\n".join(lines)


def _validate_cluster_plan(obj, valid_doc_ids: set) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    topic_shape = str(obj.get("topic_shape", "")).strip().lower()
    if topic_shape not in _TOPIC_SHAPES:
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
    sig = _sig_cluster(topic, doc_claims)
    cached = _load_cache("cluster", sig)
    if cached and isinstance(cached, dict) and cached.get("clusters"):
        if metrics:
            metrics.cache_event("outline_cluster", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_cluster", "misses")

    prompt = _build_cluster_prompt(topic, doc_claims)
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

    plan = _parse_and_validate(raw, lambda obj: _validate_cluster_plan(obj, valid_doc_ids))
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
        plan = _parse_and_validate(raw, lambda obj: _validate_cluster_plan(obj, valid_doc_ids))
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
        return (
            "Before choosing `relation`, write `reasoning_trace` answering "
            "these THREE questions in order:\n"
            "  (i) What CAUSAL VARIABLE does the topic name as the cause? "
            "(Quote the topic phrasing.)\n"
            "  (ii) What CAUSAL VARIABLE do this cluster's papers name as the "
            "cause? (Quote the language from at least one paper's claim.)\n"
            "  (iii) Is the cluster's causal variable: (A) the topic's variable "
            "under a different label or a more specific instance — "
            "`same_as_topic_cause`; (B) something UPSTREAM that flows INTO the "
            "topic's variable as a trigger — `upstream_of_topic_cause`; "
            "(C) something the topic's variable itself produces — "
            "`downstream_of_topic_cause`; (D) a RIVAL cause that REPLACES the "
            "topic's variable and does NOT operate through it — "
            "`rival_to_topic_cause`; (E) a SCOPE CONDITION (the topic's claim "
            "holds in some settings but not others) — `scope_condition`; "
            "(F) unrelated subject matter — `adjacent`.\n\n"
            "Key trap: a paper that uses \"X, contra Y\" phrasing is NOT "
            "necessarily a rival. If X re-expresses the topic's cause, or X "
            "flows INTO the topic's cause, the relation is "
            "`same_as_topic_cause` or `upstream_of_topic_cause` — not "
            "`rival_to_topic_cause`. Decide on causal STRUCTURE, not surface "
            "rhetoric."
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
                          cluster_evidence: Dict[str, List[str]]) -> str:
    lines = [
        f"TOPIC: {topic}",
        f"TOPIC SHAPE (from Stage 1): {topic_shape}",
        "",
        f"CLUSTER LABEL (from Stage 1): {cluster.get('shared_thread','')}",
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
        "  reasoning_trace: a few sentences answering (i)-(iii) above with the "
        "required quotes.",
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
    return "\n".join(lines)


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
    return {
        "relation": relation,
        "elaboration": elaboration,
        "reasoning_trace": reasoning,
        "lead_doc_id": lead,
        "internal_disagreement": internal,
    }


def posture_cluster(topic: str, topic_shape: str, cluster: dict,
                    cluster_evidence: Dict[str, List[str]],
                    metrics=None) -> Optional[dict]:
    """Stage 2. One LLM call per cluster.

    cluster: {cluster_id, doc_ids, shared_thread, claims_by_doc}
    cluster_evidence: {doc_id: [snippet, ...]}
    """
    doc_ids = list(cluster.get("doc_ids") or [])
    claims = [cluster.get("claims_by_doc", {}).get(d, "") for d in doc_ids]
    sig = _sig_posture(topic, topic_shape, doc_ids, claims)
    cached = _load_cache("posture", sig)
    if cached and isinstance(cached, dict) and cached.get("relation"):
        if metrics:
            metrics.cache_event("outline_posture", "hits")
        return cached
    if metrics:
        metrics.cache_event("outline_posture", "misses")

    prompt = _build_posture_prompt(topic, topic_shape, cluster, cluster_evidence)
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

    posture["model"] = _MODEL
    posture["prompt_version"] = _POSTURE_PROMPT_VERSION
    _save_cache("posture", sig, posture)
    if metrics:
        metrics.cache_event("outline_posture", "writes")
    return posture


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
    return "\n".join(lines)


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
    """Run Stage 1 + Stage 2 (per cluster) + Stage 3.

    Returns the full outline plan or None if Stage 1 fails irrecoverably.
    """
    plan = cluster_papers(topic, doc_summaries, metrics=metrics)
    if not plan:
        return None

    topic_shape = plan["topic_shape"]
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
                                  evidence_by_doc, metrics=metrics)
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
        "clusters": enriched_clusters,
        "unassigned_doc_ids": plan.get("unassigned_doc_ids") or [],
        "ordered_cluster_ids": ordered_ids,
        "admitted_total": total_admitted,
        "unassigned_share": round(n_unassigned / total_admitted, 3) if total_admitted else 0.0,
        "relation_distribution": relation_counts,
        "model": _MODEL,
        "cluster_prompt_version": _CLUSTER_PROMPT_VERSION,
        "posture_prompt_version": _POSTURE_PROMPT_VERSION,
        "order_prompt_version": _ORDER_PROMPT_VERSION,
    }


# ---------------------------------------------------------------------------
# JSON helpers


def _parse_and_validate(raw: str, validate_fn):
    """Tolerant JSON parsing: find the first {...} block, parse it, run the
    validator. Returns whatever the validator returns, or None on any error.
    """
    if not raw:
        return None
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
