import hashlib
import json
import os
import re
from rrr.utils import ensure_dir, env_int
# v15.12: patch ollama.chat to disable qwen3 thinking mode BEFORE any stage
# runs. Every `import ollama` in the package resolves to the same module
# object, so this one call covers all ~30 call sites.
from rrr.llm import install as _install_llm_shim
_install_llm_shim()
# The corpus-level outline assigns free-text structural relations across the
# admitted literature. The retired per-paper stance path is no longer part of
# this module.
from rrr.outline import build_outline
from rrr.metrics import RunMetrics
from rrr.manifest import write_run_manifest
from rrr.paths import runs_path, stage_cache_path, stage_cache_enabled
from rapidfuzz import fuzz

# The active reasoner handles document admission, claim extraction, and the
# corpus-level outline. Model selection follows the shared reasoner fallback.
_MODEL = os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))
_KEEP_ALIVE = "30m"


# v13: _build_author_year_lookup and _collect_cited_docs promoted to
# render.py; both modules import them from there now.
from rrr.render import _build_author_year_lookup, _collect_cited_docs  # noqa: E402

def _clean_latex(s: str) -> str:
    """Clean LaTeX artifacts from BibTeX strings."""
    if not s:
        return s
    s = s.replace('{', '').replace('}', '')
    replacements = [
        (r"\\'e", 'é'), (r"\\`e", 'è'), (r'\\"e', 'ë'), (r'\\^e', 'ê'),
        (r"\\'a", 'á'), (r"\\`a", 'à'), (r'\\"a', 'ä'), (r'\\^a', 'â'),
        (r"\\'o", 'ó'), (r"\\`o", 'ò'), (r'\\"o', 'ö'), (r'\\^o', 'ô'),
        (r"\\'u", 'ú'), (r"\\`u", 'ù'), (r'\\"u', 'ü'), (r'\\^u', 'û'),
        (r"\\'i", 'í'), (r"\\`i", 'ì'), (r'\\"i', 'ï'), (r'\\^i', 'î'),
        (r'\\c{c}', 'ç'), (r'\\c{C}', 'Ç'),
        (r'\\c{s}', 'ş'), (r'\\c{S}', 'Ş'),
        (r'\\v{s}', 'š'), (r'\\v{S}', 'Š'),
        (r'\\~n', 'ñ'), (r'\\~N', 'Ñ'),
        (r'\\ss', 'ß'),
        (r"\\'", ''), (r'\\`', ''), (r'\\"', ''), (r'\\^', ''),
        (r'\\c', ''), (r'\\v', ''), (r'\\~', ''),
    ]
    for latex, char in replacements:
        s = s.replace(latex, char)
    s = re.sub(r'\\([a-zA-Z])', r'\1', s)
    return s.strip()

def _cite_harvard(row):
    """Format citation in Harvard (LUSEM) style."""
    def s(x):
        val = str(x).strip() if x is not None else ""
        return "" if val.lower() == "nan" else val

    def clean_num(x):
        val = s(x)
        if val.endswith('.0'):
            return val[:-2]
        return val

    author_full = _clean_latex(s(row.get("author_full")))
    authors_short = _clean_latex(s(row.get("authors")))
    title = _clean_latex(s(row.get("title")))
    year = s(row.get("year"))
    venue = _clean_latex(s(row.get("venue")))
    volume = clean_num(row.get("volume"))
    number = clean_num(row.get("number"))
    pages = s(row.get("pages"))

    if author_full:
        author_parts = [p.strip() for p in author_full.split(";") if p.strip()]
        formatted_authors = []
        for ap in author_parts:
            if "," in ap:
                surname, first = ap.split(",", 1)
                initials = "".join([n[0] + "." for n in first.strip().split() if n])
                formatted_authors.append(f"{surname.strip()}, {initials}")
            else:
                formatted_authors.append(ap)

        if len(formatted_authors) == 1:
            author_str = formatted_authors[0]
        elif len(formatted_authors) == 2:
            author_str = f"{formatted_authors[0]} and {formatted_authors[1]}"
        else:
            author_str = ", ".join(formatted_authors[:-1]) + f" and {formatted_authors[-1]}"
    else:
        author_str = authors_short or "[Unknown]"

    cite = f"{author_str} ({year})" if year else author_str

    if title:
        cite += f" '{title}'"

    if venue:
        cite += f", {venue}"
        if volume:
            cite += f", {volume}"
            if number:
                cite += f"({number})"
        if pages:
            cite += f", pp. {pages.replace('--', '-')}"

    cite += "."
    return cite


def _plan_probes(plan_obj: dict, topic: str, score_query: str) -> list:
    probes = []
    for item in plan_obj.get("probes", []) or []:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in probes:
            probes.append(text)
    for item in [score_query, topic]:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in probes:
            probes.append(text)
    return probes[:8] or [topic]


def _doc_admit_signature(topic: str, probes: list, doc_ids: list, settings: dict) -> str:
    h = hashlib.sha256()
    h.update((topic or "").encode("utf-8"))
    h.update(json.dumps(probes, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(doc_ids, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _doc_admit_cache_path(sig: str):
    # v15.16: workspace-level (was runs_path("cache", ...) — per-run
    # under the v15.9 minted-run-id layout, so replay never worked).
    return stage_cache_path("doc_admit", f"{sig}.json")


def _load_doc_admit_cache_obj(sig: str):
    if not stage_cache_enabled():
        return None  # RRR_STAGE_CACHE=0: cold-run measurement mode
    try:
        with open(_doc_admit_cache_path(sig), encoding="utf-8") as f:
            obj = json.load(f)
        docs = obj.get("docs", [])
        return obj if isinstance(docs, list) else None
    except Exception:
        return None


def _save_doc_admit_cache(sig: str, docs: list, meta: dict, rejections: list = None):
    ensure_dir(str(stage_cache_path("doc_admit")))
    with open(_doc_admit_cache_path(sig), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "docs": docs, "rejections": rejections or []}, f, indent=2, ensure_ascii=False)


def _best_probe_for_sentence(sentence: str, probes: list) -> str:
    if not probes:
        return ""
    return max(probes, key=lambda p: fuzz.token_set_ratio(sentence, p))


def _rerank_quotes_for_diversity(quotes: list, cap: int) -> list:
    if cap <= 0 or len(quotes) <= cap:
        return sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)
    remaining = sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)
    chosen = []
    seen_pages = set()
    seen_texts = []
    while remaining and len(chosen) < cap:
        best_idx = 0
        best_value = None
        for idx, q in enumerate(remaining):
            text = q.get("text", "")
            page_penalty = 8 if q.get("page") in seen_pages else 0
            similarity_penalty = max((fuzz.token_set_ratio(text, s) for s in seen_texts), default=0) * 0.10
            probe_bonus = 3 if q.get("best_probe") and q.get("best_probe") not in {x.get("best_probe") for x in chosen} else 0
            value = float(q.get("score", 0)) + probe_bonus - page_penalty - similarity_penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        q = remaining.pop(best_idx)
        chosen.append(q)
        seen_pages.add(q.get("page"))
        seen_texts.append(q.get("text", ""))
    return chosen


def _select_budget_docs(docs: list, budget: int, probes: list, metrics=None) -> list:
    if budget <= 0 or len(docs) <= budget:
        if metrics:
            metrics.set("doc_budget_requested", budget)
            metrics.set("doc_budget_selected", len(docs))
            metrics.set("doc_budget_exhaustive", True)
        return sorted(docs, key=lambda x: x.get("avg_score", 0), reverse=True)

    selected = []
    remaining = sorted(docs, key=lambda x: x.get("avg_score", 0), reverse=True)
    covered_probes = set()

    while remaining and len(selected) < budget:
        best_idx = 0
        best_value = None
        selected_doc_ids = {d.get("doc_id") for d in selected}
        for idx, doc in enumerate(remaining):
            doc_probes = set(doc.get("probe_hits", []))
            new_probe_bonus = 5 * len(doc_probes - covered_probes)
            score = float(doc.get("avg_score", 0))
            evidence_bonus = min(len(doc.get("quotes", [])), 8)
            duplicate_penalty = 20 if doc.get("doc_id") in selected_doc_ids else 0
            value = score + new_probe_bonus + evidence_bonus - duplicate_penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        doc = remaining.pop(best_idx)
        selected.append(doc)
        covered_probes.update(doc.get("probe_hits", []))

    selected = sorted(selected, key=lambda x: x.get("avg_score", 0), reverse=True)
    if metrics:
        metrics.set("doc_budget_requested", budget)
        metrics.set("doc_budget_selected", len(selected))
        metrics.set("doc_budget_exhaustive", False)
        metrics.set("doc_budget_probe_coverage", len(covered_probes))
        metrics.inc("docs_budget_dropped", max(0, len(docs) - len(selected)))
    return selected


def _assign_evidence_ids(doc_summaries: list):
    counter = 1
    for doc in sorted(doc_summaries, key=lambda d: d.get("doc_id", "")):
        for q in doc.get("quotes", []) or []:
            q["evidence_id"] = f"E{counter:04d}"
            counter += 1
    return counter - 1


def _mean_score(docs: list) -> float:
    vals = [float(d.get("avg_score", 0) or 0) for d in docs or []]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _compute_topic_fit(topic: str, probes: list, all_doc_ids: list, admitted_docs: list,
                       selected_docs: list, doc_summaries: list = None, rejections: list = None):
    total = len(all_doc_ids) or 1
    represented = doc_summaries or []
    admitted_probe_hits = set()
    selected_probe_hits = set()
    for doc in admitted_docs or []:
        admitted_probe_hits.update(doc.get("probe_hits", []) or [])
    for doc in selected_docs or []:
        selected_probe_hits.update(doc.get("probe_hits", []) or [])

    probe_count = len(probes) or 1
    rejection_counts = {}
    for item in rejections or []:
        reason = item.get("reason", "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    summary = {
        "topic": topic,
        "docs_total": len(all_doc_ids),
        "docs_admitted": len(admitted_docs or []),
        "docs_selected_for_llm": len(selected_docs or []),
        "docs_represented": len(represented),
        "admission_share": round(len(admitted_docs or []) / total, 4),
        "selection_share": round(len(selected_docs or []) / total, 4),
        "represented_share": round(len(represented) / total, 4),
        "admitted_probe_coverage": round(len(admitted_probe_hits) / probe_count, 4),
        "selected_probe_coverage": round(len(selected_probe_hits) / probe_count, 4),
        "mean_admitted_score": _mean_score(admitted_docs),
        "mean_selected_score": _mean_score(selected_docs),
        "rejection_counts": rejection_counts,
        "warnings": [],
    }

    if summary["admission_share"] < 0.25:
        summary["warnings"].append("low_admitted_document_share")
    if summary["selected_probe_coverage"] < 0.5:
        summary["warnings"].append("narrow_probe_coverage")
    if summary["mean_selected_score"] and summary["mean_selected_score"] < 45:
        summary["warnings"].append("low_mean_evidence_score")
    if represented and summary["represented_share"] < 0.15:
        summary["warnings"].append("low_represented_document_share")
    return summary


def _write_json_run(name: str, obj):
    ensure_dir(str(runs_path()))
    with open(runs_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _retrieve_doc_with_probes(retrieve_fn, doc_id: str, probes: list, topk: int, metrics=None) -> list:
    """v8 (R8): rank merged candidates by Reciprocal Rank Fusion across probes
    instead of max(bm25_score). Pages that rank moderately on every probe now
    outrank pages that spike on one probe. On multi-aspect topics (where
    different probes target distinct sub-claims that rarely co-occur on a
    single page), this preserves breadth-of-probe-agreement as a ranking
    signal. bm25_score is retained as a tiebreaker for downstream consumers.
    Constant k=60 is the textbook RRF default.
    """
    merged = {}
    per_probe_topk = max(1, topk)
    # v13: RRR_RRF_K retired; k=60 is the textbook RRF default and v8 R8 audit
    # decision baked into the source.
    rrf_k = 60.0
    for probe in probes:
        candidates = retrieve_fn(probe, topk=per_probe_topk, doc_id=doc_id)
        if metrics:
            metrics.inc("retrieval_probe_calls")
            metrics.inc("retrieved_pages_raw", len(candidates))
        for rank, c in enumerate(candidates):
            key = (c.get("doc_id"), c.get("page"))
            score = float(c.get("bm25_score", 0.0) or 0.0)
            rrf_contrib = 1.0 / (rrf_k + rank)
            existing = merged.get(key)
            if not existing:
                item = dict(c)
                item["matched_probes"] = [probe] if score > 0.0 else []
                item["rrf_score"] = rrf_contrib if score > 0.0 else 0.0
                item["bm25_score"] = score
                merged[key] = item
            else:
                if score > 0.0 and probe not in existing.get("matched_probes", []):
                    existing.setdefault("matched_probes", []).append(probe)
                if score > 0.0:
                    existing["rrf_score"] = float(existing.get("rrf_score", 0.0)) + rrf_contrib
                # keep bm25_score as the max for tiebreaking and downstream
                # diagnostic consumers that still look at it directly.
                if score > float(existing.get("bm25_score", 0.0) or 0.0):
                    existing["bm25_score"] = score
    # small bonus for breadth of probe agreement, capped so it can never
    # dominate a strong single-probe match
    for item in merged.values():
        item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 0.02 * len(set(item.get("matched_probes", [])))
    ranked = sorted(
        merged.values(),
        key=lambda x: (float(x.get("rrf_score", 0.0)), float(x.get("bm25_score", 0.0) or 0.0)),
        reverse=True,
    )
    return ranked[:max(1, topk)]


def _layered_t2_inner(args, meta_path):
    """
    Inner implementation of layered_t2.
    """
    import os, json, threading
    import pandas as pd
    from rrr.retrieve import retrieve
    from rrr.evidence_filter import select_sentences
    from rrr.validate import validate_evidence_verbose
    from rrr.utils import ensure_dir, env_int, write_run, normalize_space

    topic = args.topic
    metrics = RunMetrics("T2_LAYERED_GLOBAL", topic)

    # v15.11: language routing. CLI sets these before importing us; we
    # surface them on metrics for observability. Absence means the pipeline
    # is running via a non-CLI entry point (test/battery script) — fall back
    # to English so the majority path keeps its zero-directive prompts.
    topic_lang = os.environ.get("RRR_TOPIC_LANG", "en")
    selected_model = os.environ.get("RRR_MODEL", _MODEL)
    metrics.set("topic_lang", topic_lang)
    metrics.set("selected_model", selected_model)

    # v15.9 (#4): mint a per-invocation run_id so this call's artifacts land
    # in runs/<utc>_<slug>/ rather than the flat runs/ folder — concurrent
    # runs on the same corpus no longer collide.
    #
    # Env overrides for the smoke harness / battery:
    #   RRR_RUN_ID=<slug>  → use this exact slug (harness knows where to look)
    #   RRR_RUN_ID=""      → force flat layout (pre-v15.9 behaviour)
    #   unset              → mint one from timestamp + topic slug
    from rrr.paths import mint_run_id, set_default_run_id
    run_id_env = os.environ.get("RRR_RUN_ID")
    if run_id_env is None:
        run_id = mint_run_id(topic)
    elif run_id_env == "":
        run_id = None  # explicit opt-out: flat runs/ layout
    else:
        run_id = run_id_env
    set_default_run_id(run_id)
    metrics.set("run_id", run_id or "")

    # v8 (R12): fire-and-forget prewarm so the Ollama cold-load (~7s on first
    # call after model swap) overlaps with metadata loading + planning instead
    # of being charged to the planning stage. Background thread; no blocking.
    def _prewarm_ollama():
        try:
            import ollama
            ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": "."}],
                options={"num_predict": 1, "num_ctx": 4096, "temperature": 0.0},
                keep_alive=_KEEP_ALIVE,
                stream=False,
            )
        except Exception:
            pass
    runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
    if runtime not in {"api", "host"}:
        try:
            threading.Thread(
                target=_prewarm_ollama,
                name="rrr-ollama-prewarm",
                daemon=True,
            ).start()
        except Exception:
            pass

    with metrics.stage("load_metadata"):
        df = pd.read_csv(meta_path)
        df["doc_id"] = df["doc_id"].astype(str)

    refs = {str(r["doc_id"]): _cite_harvard(r) for _, r in df.iterrows()}
    all_doc_ids = df["doc_id"].tolist()
    metrics.set("metadata_path", str(meta_path))
    metrics.set("docs_total", len(all_doc_ids))

    # v15.9 (#1): populate the render-time metadata → author label lookup so
    # in-text citations and reference-list rendering can use metadata.csv as
    # the source of truth instead of parsing doc_ids via regex. When a
    # metadata row lacks display_label / first_author_surname columns, the
    # seed values come from the legacy regex — so the 50-paper hand-curated
    # corpus produces byte-identical output.
    from rrr.render import set_metadata_labels
    n_labels = set_metadata_labels(df.to_dict(orient="records"))
    metrics.set("metadata_labels_loaded", n_labels)
    print(f"[Reasoner] loaded {n_labels} metadata-driven author labels")

    # v15.9 (#5): corpus_fingerprint namespaces the claim cache; content_sha1
    # (per paper) becomes the primary cache key when populated by the ingest
    # cascade. Existing metadata.csv without those columns falls back to
    # doc_id keying (legacy flat cache is still readable via fallback path
    # in stance._claim_cache_path).
    from rrr.stance import compute_corpus_fingerprint
    corpus_fingerprint = compute_corpus_fingerprint(df)
    metrics.set("corpus_fingerprint", corpus_fingerprint)
    if "content_sha1" in df.columns:
        _content_sha_by_docid = {
            str(r["doc_id"]): str(r["content_sha1"]).strip()
            for _, r in df.iterrows()
            if str(r.get("content_sha1", "")).strip()
        }
    else:
        _content_sha_by_docid = {}
    metrics.set("content_sha1_populated", len(_content_sha_by_docid))

    # v15.11/v15.12: corpus language distribution + dominant corpus language.
    # The dominant corpus language is the retrieval/pivot language: the
    # planner emits BM25 probes in it (translating a cross-language topic),
    # internal reasoning runs in it, and the writer translates the final
    # review from it into the topic language. When metadata.csv has no lang
    # column we assume the corpus is English (the historical default) so the
    # feature is fully backward-compatible.
    from collections import Counter
    if "lang" in df.columns:
        corpus_lang_dist = dict(Counter(
            str(r.get("lang", "") or "unknown").strip().lower()
            for _, r in df.iterrows()
        ))
    else:
        corpus_lang_dist = {"en": len(all_doc_ids)}
    # Dominant corpus language = most frequent non-unknown lang, default en.
    _ranked = sorted(
        ((k, v) for k, v in corpus_lang_dist.items() if k and k != "unknown"),
        key=lambda kv: kv[1], reverse=True,
    )
    corpus_lang = _ranked[0][0] if _ranked else "en"
    metrics.set("corpus_lang_distribution", corpus_lang_dist)
    metrics.set("corpus_lang", corpus_lang)
    cross_lang = corpus_lang != topic_lang
    metrics.set("cross_lang_retrieval", cross_lang)
    # Expose the pivot language to prompt builders (writer output contract).
    os.environ["RRR_CORPUS_LANG"] = corpus_lang
    if cross_lang:
        print(f"[Reasoner] cross-language: topic_lang={topic_lang} "
              f"corpus_lang={corpus_lang} → planner emits {corpus_lang} probes, "
              f"writer outputs {topic_lang}.")

    ensure_dir(str(runs_path()))
    ensure_dir(str(runs_path("layered_docs")))

    # v15.14: env_int — a malformed env value degrades to the default with a
    # warning instead of an unhandled ValueError mid-run.
    PER_DOC_TOPK = env_int("RRR_PER_DOC_TOPK", 30)
    MAX_SENTS_PER_PAGE = env_int("RRR_MAX_SENTS_PAGE", 8)
    MIN_CHARS = env_int("RRR_MIN_SENT_CHARS", 20)
    MIN_DOC_SNIPS = env_int("RRR_MIN_DOC_SNIPS", 3)
    GLOBAL_MIN_DOCS = env_int("RRR_GLOBAL_MIN_DOCS", 5)
    MD_QUOTE_CAP = env_int("RRR_MD_QUOTE_CAP", 8)
    DOC_BUDGET = env_int("RRR_DOC_BUDGET", 24)
    # v13: RRR_DOC_ADMIT_CACHE retired (always on). The cache is cheap to
    # write, the v8 default was 1, and no battery script overrode it.
    DOC_ADMIT_CACHE = True
    DOC_ADMIT_REPLAY = os.environ.get("RRR_DOC_ADMIT_REPLAY", "0") == "1"
    EV_CAP = env_int("RRR_EV_PER_DOC_CAP", 8)
    metrics.set("doc_admit_cache_enabled", 1)
    metrics.set("doc_admit_replay", int(DOC_ADMIT_REPLAY))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rrr.query_planner import plan as plan_query
    with metrics.stage("planning"):
        plan_obj = plan_query(topic, metrics=metrics,
                              corpus_lang=corpus_lang, topic_lang=topic_lang)
    score_query = " ".join(plan_obj.get("keywords_must", []) + plan_obj.get("keywords_any", []))
    score_query = score_query.strip() or topic
    planner_mode = plan_obj.get("planner_meta", {}).get("mode", "unknown")
    # v8 (R9): when the LLM planner produced real phrase probes, do NOT inject
    # the bag-of-words concatenation as an additional probe. In baseline runs
    # the synthetic probe dominated max(token_set_ratio) scoring because it had
    # the most surface tokens, making the LLM-derived phrase probes do almost
    # no work. The synthetic remains as a fallback only when the planner fell
    # back to the heuristic mode.
    if planner_mode == "llm":
        probes = _plan_probes(plan_obj, topic, "")
    else:
        probes = _plan_probes(plan_obj, topic, score_query)
    plan_obj["active_probes"] = probes
    metrics.set("planner_mode", planner_mode)
    metrics.set("planner_probe_count", len(probes))
    print(f"[Layered-T2] score_query={score_query}")
    print(f"[Layered-T2] probes={len(probes)}")

    admit_settings = {
        "per_doc_topk": PER_DOC_TOPK,
        "max_sents_per_page": MAX_SENTS_PER_PAGE,
        "min_chars": MIN_CHARS,
        "min_doc_snips": MIN_DOC_SNIPS,
        "ev_cap": EV_CAP,
        "min_sent_score": os.environ.get("RRR_MIN_SENT_SCORE", "40"),
        "bypass_validation": os.environ.get("RRR_BYPASS_VALIDATION", "0"),
    }
    admit_sig = _doc_admit_signature(topic, probes, all_doc_ids, admit_settings)
    write_run_manifest(
        "T2_LAYERED_GLOBAL",
        topic,
        meta_path,
        _MODEL,
        plan=plan_obj,
        extra={"admit_settings": admit_settings},
    )

    MAX_WORKERS = env_int("RRR_CONCURRENCY", 4)
    admission_rejections = []
    rejection_lock = threading.Lock()

    def summarize_candidates(candidates):
        out = []
        for c in candidates[:10]:
            out.append({
                "page": int(c.get("page", 0) or 0),
                "bm25_score": round(float(c.get("bm25_score", 0.0) or 0.0), 4),
                "matched_probe_count": len(c.get("matched_probes", []) or []),
                "matched_probes": (c.get("matched_probes", []) or [])[:4],
            })
        return out

    def record_rejection(did, reason, candidates=None, **details):
        item = {
            "doc_id": did,
            "citation": refs.get(did, did),
            "reason": reason,
            "thresholds": {
                "min_doc_snips": MIN_DOC_SNIPS,
                "min_chars": MIN_CHARS,
                "min_sent_score": admit_settings["min_sent_score"],
                "per_doc_topk": PER_DOC_TOPK,
                "max_sents_per_page": MAX_SENTS_PER_PAGE,
            },
            "candidate_pages": summarize_candidates(candidates or []),
        }
        item.update(details)
        with rejection_lock:
            admission_rejections.append(item)

    print(f"[Layered-T2] starting evidence admission over {len(all_doc_ids)} docs (concurrency={MAX_WORKERS})")

    def process_doc(did):
        metrics.inc("docs_processed")
        candidates = _retrieve_doc_with_probes(retrieve, did, probes, PER_DOC_TOPK, metrics=metrics)
        metrics.inc("retrieved_pages_kept", len(candidates))
        if not candidates:
            record_rejection(did, "no_retrieved_pages", candidates=[])
            metrics.inc("docs_rejected_no_pages")
            return None

        quotes = []
        page_sentence_counts = {}
        for c in candidates:
            txt = c.get("text", "").strip()
            if not txt:
                page_sentence_counts[int(c.get("page", 0) or 0)] = 0
                continue
            # v13.2 FIX-SNIPPET-WIDEN: out_stats receives the per-page count
            # of context sentences stitched on (prev+next) so we can tally the
            # `evidence_context_sentences_added` run-level metric.
            sent_stats = {}
            scored_sents = select_sentences(
                txt,
                topic,
                max_sentences=MAX_SENTS_PER_PAGE,
                min_chars=MIN_CHARS,
                probes=probes,
                out_stats=sent_stats,
            )
            metrics.inc("evidence_context_sentences_added", int(sent_stats.get("context_sentences_added", 0)))
            page_sentence_counts[int(c.get("page", 0) or 0)] = len(scored_sents)
            for sent, score in scored_sents:
                s_norm = normalize_space(sent)
                if len(s_norm) < MIN_CHARS:
                    continue
                best_probe = _best_probe_for_sentence(s_norm, c.get("matched_probes") or probes)
                quotes.append({
                    "type": "quote",
                    "doc_id": did,
                    "page": int(c["page"]),
                    "text": s_norm,
                    "score": score,
                    "best_probe": best_probe,
                    "matched_probes": c.get("matched_probes", []),
                })
        metrics.inc("candidate_quotes", len(quotes))

        seen = {}
        for q in quotes:
            k = (q["page"], q["text"][:160])
            if k not in seen or q["score"] > seen[k]["score"]:
                seen[k] = q
        quotes = list(seen.values())

        if len(quotes) < MIN_DOC_SNIPS:
            metrics.inc("docs_rejected_min_quotes")
            record_rejection(
                did,
                "insufficient_candidate_sentences",
                candidates=candidates,
                selected_sentence_count=len(quotes),
                page_sentence_counts=page_sentence_counts,
            )
            return None

        verbose_val = validate_evidence_verbose(quotes, df)
        val = [{"item": v["item"], "ok": v["verdict"] in ("exact", "soft_ok", "bypass"), "reason": v["reason"]} for v in verbose_val]
        metrics.inc("validation_items", len(val))
        metrics.inc("validation_ok", sum(1 for v in val if v["ok"]))
        metrics.inc("validation_failed", sum(1 for v in val if not v["ok"]))
        valid_quotes = [v["item"] for v in val if v["ok"]]
        if len(valid_quotes) < MIN_DOC_SNIPS:
            metrics.inc("docs_rejected_validation")
            reason_counts = {}
            for v in verbose_val:
                reason = v.get("reason") or v.get("verdict") or "unknown"
                if v.get("verdict") not in ("exact", "soft_ok", "bypass"):
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            record_rejection(
                did,
                "insufficient_validated_quotes",
                candidates=candidates,
                selected_sentence_count=len(quotes),
                validation_items=len(val),
                validation_ok=len(valid_quotes),
                validation_failed=len(val) - len(valid_quotes),
                validation_reason_counts=reason_counts,
            )
            return None

        valid_quotes = _rerank_quotes_for_diversity(valid_quotes, EV_CAP)
        metrics.inc("valid_quotes_kept", len(valid_quotes))

        avg_score = sum(q.get("score", 0) for q in valid_quotes) / len(valid_quotes) if valid_quotes else 0
        probe_hits = []
        for q in valid_quotes:
            for probe in q.get("matched_probes", []) or [q.get("best_probe")]:
                if probe and probe not in probe_hits:
                    probe_hits.append(probe)

        metrics.inc("docs_admitted")
        return {
            "doc_id": did,
            "citation": refs.get(did, did),
            "quotes": valid_quotes,
            "avg_score": round(avg_score, 2),
            "probe_hits": probe_hits,
        }

    def enrich_doc(doc):
        # v15: per-doc enrichment SHRINKS to claim extraction only. The fused
        # stance+mechanism call (v14.4) is gone — corpus-level outline now
        # handles clustering and the per-cluster posture replaces per-doc
        # stance. Claim extraction stays (cached per paper, content-keyed) so
        # the outline.cluster_papers call has accurate inputs.
        did = doc["doc_id"]
        from rrr.stance import extract_paper_claim
        # v15.9 (#5): pass corpus_fingerprint + content_sha1 (when populated)
        # so this per-paper cache entry lands in the corpus-namespaced dir
        # keyed by PDF content, not by doc_id string.
        claim_info = extract_paper_claim(
            did, metrics=metrics,
            corpus_fingerprint=corpus_fingerprint,
            content_sha1=_content_sha_by_docid.get(did) or None,
        )
        paper_claim = claim_info.get("claim", "")
        metrics.inc("docs_kept")
        enriched = dict(doc)
        enriched["claim"] = paper_claim
        enriched["claim_source"] = claim_info.get("source", "")
        return enriched

    admitted_docs = None
    if DOC_ADMIT_CACHE and DOC_ADMIT_REPLAY:
        cached_admit = _load_doc_admit_cache_obj(admit_sig)
        if cached_admit is not None:
            admitted_docs = cached_admit.get("docs", [])
            admission_rejections.extend(cached_admit.get("rejections", []) or [])
            metrics.cache_event("doc_admit", "hits")
            metrics.set("doc_admit_cache_key", admit_sig)
        else:
            metrics.cache_event("doc_admit", "misses")
    elif DOC_ADMIT_CACHE:
        metrics.cache_event("doc_admit", "skips")
        metrics.set("doc_admit_cache_key", admit_sig)

    if admitted_docs is None:
        admitted_docs = []
        with metrics.stage("evidence_admission"):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_doc, did): did for did in all_doc_ids}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        admitted_docs.append(res)
        admitted_docs = sorted(admitted_docs, key=lambda x: x.get("avg_score", 0), reverse=True)
        admission_rejections = sorted(admission_rejections, key=lambda x: x.get("doc_id", ""))
        if DOC_ADMIT_CACHE:
            _save_doc_admit_cache(admit_sig, admitted_docs, admit_settings, rejections=admission_rejections)
            metrics.cache_event("doc_admit", "writes")
            metrics.set("doc_admit_cache_key", admit_sig)
    _write_json_run("admission_rejections.json", {
        "topic": topic,
        "cache_key": admit_sig,
        "rejections": admission_rejections,
    })
    metrics.set("docs_rejected_total", len(admission_rejections))

    metrics.set("docs_admitted_total", len(admitted_docs))
    selected_docs = _select_budget_docs(admitted_docs, DOC_BUDGET, probes, metrics=metrics)
    metrics.set("docs_selected_for_llm", len(selected_docs))
    topic_fit = _compute_topic_fit(
        topic,
        probes,
        all_doc_ids,
        admitted_docs,
        selected_docs,
        rejections=admission_rejections,
    )
    _write_json_run("topic_fit.json", topic_fit)
    metrics.set("topic_fit_warnings", topic_fit.get("warnings", []))
    metrics.set("topic_fit_admission_share", topic_fit.get("admission_share"))
    metrics.set("topic_fit_selected_probe_coverage", topic_fit.get("selected_probe_coverage"))
    print(f"[Layered-T2] admitted={len(admitted_docs)} selected_for_llm={len(selected_docs)} budget={DOC_BUDGET}")

    # v15.10 (#3+#4): early Stage 0 precheck BEFORE per_document_sweep. The
    # sweep's ~5s/doc claim-extraction LLM call is by far the most expensive
    # pre-writer stage on cold cache (~4 min for 50 docs). Stage 0 only reads
    # doc_id + citation, both available from the metadata refs dict without
    # needing claims. Running precheck here lets a genuine refusal skip the
    # sweep entirely; PROCEED falls through and the late build_outline() call
    # inside the outline stage hits the same-signature cache with zero LLM
    # cost. Also splits the previously-conflated 'outline_failed' refusal
    # reason into:
    #   - no_admitted_docs      (BM25/admission produced 0 docs)
    #   - stage0_llm_failed     (precheck LLM error or malformed JSON)
    #   - corpus_off_topic      (precheck returned REFUSE)
    # and leaves 'outline_failed' meaning only 'Stage 1 clustering returned
    # nothing' for the late-fall-through case.
    if not selected_docs:
        print("[Layered-T2] refusal=no_admitted_docs (BM25 admitted zero docs)")
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": 0},
                  {"refusal": True, "reason": "no_admitted_docs",
                   "explanation": "BM25 retrieval + evidence admission "
                                  "produced zero admitted documents. The "
                                  "topic likely has no vocabulary overlap "
                                  "with any corpus paper. See "
                                  "runs/admission_rejections.json."})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "no_admitted_docs")
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": "no_admitted_docs"},
        )
        metrics.save()
        return

    from rrr.outline import precheck as _early_precheck
    early_doc_summaries = [
        {"doc_id": d.get("doc_id"),
         "citation": refs.get(d.get("doc_id"), d.get("doc_id"))}
        for d in selected_docs if d.get("doc_id")
    ]
    print("[Layered-T2] early Stage 0 precheck (before per_document_sweep)...")
    with metrics.stage("outline_early_precheck"):
        early_pre = _early_precheck(topic, early_doc_summaries, metrics=metrics)
    if early_pre is None:
        print("[Layered-T2] refusal=stage0_llm_failed (Stage 0 precheck LLM "
              "error, malformed JSON, or invalid enum)")
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": 0},
                  {"refusal": True, "reason": "stage0_llm_failed",
                   "explanation": "Stage 0 precheck did not return a valid "
                                  "response. Causes: Ollama connection "
                                  "error, malformed JSON, or an invalid "
                                  "topic_shape/corpus_fit enum. Inspect "
                                  "metrics.llm_calls."})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "stage0_llm_failed")
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": "stage0_llm_failed"},
        )
        metrics.save()
        return
    if early_pre.get("corpus_fit") == "REFUSE":
        refusal_explanation = early_pre.get("corpus_fit_rationale", "") or (
            "Stage 0 precheck determined the topic and corpus come from "
            "different intellectual domains with no honest scholarly path.")
        print(f"[Layered-T2] refusal=corpus_off_topic (Stage 0 early precheck)")
        print(f"[Layered-T2] reason: {refusal_explanation}")
        # Persist a minimal outline_plan.json so debugging matches the late
        # refusal path's artefacts.
        _early_plan = {
            "refused": True,
            "refusal_reason": "corpus_off_topic",
            "refusal_explanation": refusal_explanation,
            "topic": topic,
            "topic_shape": early_pre.get("topic_shape"),
            "topic_shape_rationale": early_pre.get("topic_shape_rationale", ""),
            "topic_cause": early_pre.get("topic_cause", ""),
            "topic_outcome": early_pre.get("topic_outcome", ""),
            "admitted_total": len(early_doc_summaries),
            "clusters": [],
            "unassigned_doc_ids": [d["doc_id"] for d in early_doc_summaries],
            "ordered_cluster_ids": [],
            "relation_distribution": {},
            "unassigned_share": 1.0,
            "precheck_source": "early",
        }
        try:
            with open(runs_path("outline_plan.json"), "w", encoding="utf-8") as _f:
                json.dump(_early_plan, _f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": 0,
                   "topic_shape": early_pre.get("topic_shape")},
                  {"refusal": True, "reason": "corpus_off_topic",
                   "explanation": refusal_explanation})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "corpus_off_topic")
        metrics.set("outline_topic_shape", early_pre.get("topic_shape"))
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": "corpus_off_topic",
                   "outline_topic_shape": early_pre.get("topic_shape"),
                   "outline_topic_shape_rationale": early_pre.get("topic_shape_rationale", "")},
        )
        metrics.save()
        return
    # PROCEED — fall through. The late build_outline precheck call hits the
    # same-signature cache (sorted titles) with zero LLM cost.

    doc_summaries = []
    with metrics.stage("per_document_sweep"):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(enrich_doc, doc): doc.get("doc_id") for doc in selected_docs}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    doc_summaries.append(res)

    # v15: assign evidence IDs and persist per-doc artefacts FIRST so the
    # outline builder (and the writer downstream) can reference stable
    # evidence anchors. Outline is built next from the persisted doc
    # summaries (claim + quotes).
    evidence_id_count = _assign_evidence_ids(doc_summaries)
    metrics.set("evidence_ids_assigned", evidence_id_count)

    for entry in doc_summaries:
        did = entry["doc_id"]
        from rrr.utils import save_json as _save_json_atomic
        _save_json_atomic(entry, runs_path("layered_docs", f"{did}.json"))

    kept = len(doc_summaries)
    metrics.set("docs_represented", kept)
    topic_fit = _compute_topic_fit(
        topic,
        probes,
        all_doc_ids,
        admitted_docs,
        selected_docs,
        doc_summaries=doc_summaries,
        rejections=admission_rejections,
    )
    _write_json_run("topic_fit.json", topic_fit)
    metrics.set("topic_fit_warnings", topic_fit.get("warnings", []))
    metrics.set("topic_fit_represented_share", topic_fit.get("represented_share"))
    print(f"[Layered-T2] per-document sweep complete: {kept} docs summarised")

    # v15: corpus-level OUTLINE replaces per-paper stance + within-stance
    # cluster synthesis. Three LLM calls (Stage 1 cluster + Stage 2 posture
    # per cluster + Stage 3 order) produce an outline_plan with topic_shape,
    # clusters, postures, and the section order for the writer.
    print("[Layered-T2] building corpus-level outline (v15)...")
    with metrics.stage("outline"):
        outline_plan = build_outline(topic, doc_summaries, metrics=metrics)
    if outline_plan is None:
        # v15.10: early precheck already caught Stage 0 failures and empty
        # doc_summaries, so 'outline_failed' now means Stage 1 clustering
        # returned no valid plan. Renamed for observability.
        print("[Layered-T2] refusal=stage1_clustering_failed (Stage 1 clustering LLM returned no valid plan)")
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": kept},
                  {"refusal": True, "reason": "stage1_clustering_failed",
                   "explanation": "Stage 1 clustering LLM returned no valid "
                                  "plan after retry. Inspect metrics.llm_calls."})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "stage1_clustering_failed")
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": "stage1_clustering_failed"},
        )
        metrics.save()
        return

    # v15.1.0: Stage 0 corpus-fit refusal. The precheck call has explicit
    # authority to decline a (topic, corpus) pair that has no honest
    # scholarly path between them. This refusal fires BEFORE clustering and
    # is independent of unassigned_share (which the inclusive-clustering
    # prompt was structurally unable to drive above zero).
    if outline_plan.get("refused"):
        refusal_reason = outline_plan.get("refusal_reason", "corpus_off_topic")
        refusal_explanation = (
            outline_plan.get("refusal_explanation", "")
            or "Stage 0 precheck determined the topic and corpus come from "
               "different intellectual domains with no honest scholarly path "
               "between them."
        )
        print(f"[Layered-T2] refusal={refusal_reason} (Stage 0 precheck)")
        print(f"[Layered-T2] reason: {refusal_explanation}")
        # Persist the precheck plan for debugging.
        try:
            with open(runs_path("outline_plan.json"), "w", encoding="utf-8") as _f:
                json.dump(outline_plan, _f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": kept,
                   "topic_shape": outline_plan.get("topic_shape")},
                  {"refusal": True, "reason": refusal_reason,
                   "explanation": refusal_explanation})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", refusal_reason)
        metrics.set("outline_topic_shape", outline_plan.get("topic_shape"))
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": refusal_reason,
                   "outline_topic_shape": outline_plan.get("topic_shape"),
                   "outline_topic_shape_rationale": outline_plan.get("topic_shape_rationale", "")},
        )
        metrics.save()
        return

    # Propagate cluster_id and relation onto each doc_summary so existing
    # downstream code that walks doc_summaries (writer's evidence id map,
    # citation validators) keeps working. Unassigned docs get cluster_id=""
    # and relation="unassigned".
    cluster_id_by_doc = {}
    relation_by_doc = {}
    for c in outline_plan.get("clusters", []):
        for did in c.get("doc_ids", []):
            cluster_id_by_doc[did] = c["cluster_id"]
            relation_by_doc[did] = c["relation"]
    for d in doc_summaries:
        did = d.get("doc_id", "")
        d["cluster_id"] = cluster_id_by_doc.get(did, "")
        d["relation"] = relation_by_doc.get(did, "unassigned")

    # Surface relation distribution + topic shape on metrics for smoke
    # inspection. The smoke contract: AJR/Nunn must land in
    # `upstream_of_topic_cause` (not `rival_to_topic_cause`).
    metrics.set("outline_topic_shape", outline_plan.get("topic_shape"))
    metrics.set("relation_distribution", outline_plan.get("relation_distribution"))
    metrics.set("outline_unassigned_share", outline_plan.get("unassigned_share"))
    metrics.set("outline_clusters_count", len(outline_plan.get("clusters", [])))
    print(f"[Layered-T2] topic_shape={outline_plan.get('topic_shape')} "
          f"clusters={len(outline_plan.get('clusters', []))} "
          f"unassigned_share={outline_plan.get('unassigned_share')} "
          f"relations={outline_plan.get('relation_distribution')}")

    # Persist the outline plan to disk REGARDLESS of refusal so future
    # debugging has the cluster/relation/unassigned breakdown. The full
    # ledger writes later; this is a small standalone dump.
    try:
        with open(runs_path("outline_plan.json"), "w", encoding="utf-8") as _f:
            json.dump(outline_plan, _f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # v15: corpus-fit refusal — fires when too many admitted papers could not
    # be clustered into any stream. This is the structural analogue of
    # v14.4's tangential-fraction refusal but driven by the outline's own
    # unassigned bucket, which is a much sharper signal than per-paper
    # stance because Stage 1 sees the whole corpus at once. Threshold knob:
    # RRR_UNASSIGNED_REFUSAL_THRESHOLD. v15.0.1 default raised from 0.5 to
    # 0.7 — the v15.0 INST smoke had unassigned_share=0.58 on a clearly
    # on-topic corpus because Stage 1's prompt was too eager to dump
    # borderline papers; the v15b inclusive-clustering prompt + a more
    # permissive threshold gives the pipeline more room to proceed on
    # legitimately-on-topic corpora. Disable with >=1.0.
    try:
        unassigned_threshold = float(os.environ.get("RRR_UNASSIGNED_REFUSAL_THRESHOLD", "0.7"))
    except ValueError:
        unassigned_threshold = 0.7
    if unassigned_threshold <= 0.0 or unassigned_threshold > 1.0:
        unassigned_threshold = 1.01  # effectively disabled
    unassigned_share = float(outline_plan.get("unassigned_share") or 0.0)
    total_admitted = outline_plan.get("admitted_total") or kept
    if unassigned_share >= unassigned_threshold and total_admitted >= 5:
        print(f"[Layered-T2] refusal=corpus_off_topic "
              f"(unassigned {len(outline_plan.get('unassigned_doc_ids', []))}/"
              f"{total_admitted} = {unassigned_share:.0%} >= "
              f"threshold {unassigned_threshold:.0%})")
        write_run("T2_LAYERED_GLOBAL", topic,
                  {"docs_seen": len(all_doc_ids), "docs_represented": kept,
                   "unassigned_share": unassigned_share},
                  {"refusal": True, "reason": "corpus_off_topic",
                   "explanation": (
                       f"{len(outline_plan.get('unassigned_doc_ids', []))} of "
                       f"{total_admitted} admitted documents could not be "
                       f"clustered into any stream of literature about this "
                       f"topic. The corpus appears to be a poor fit for the "
                       f"question. Consider a different corpus or a different "
                       f"topic."
                   )})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "corpus_off_topic")
        write_run_manifest(
            "T2_LAYERED_GLOBAL", topic, meta_path, _MODEL, plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit,
                   "refusal": "corpus_off_topic",
                   "outline_topic_shape": outline_plan.get("topic_shape"),
                   "relation_distribution": outline_plan.get("relation_distribution"),
                   "unassigned_share": unassigned_share},
        )
        metrics.save()
        return

    if kept < GLOBAL_MIN_DOCS:
        print("[Layered-T2] refusal=insufficient_global_evidence")
        write_run("T2_LAYERED_GLOBAL", topic, {"docs_seen": len(all_doc_ids), "docs_represented": kept},
                  {"refusal": True, "reason": "insufficient_global_evidence"})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "insufficient_global_evidence")
        write_run_manifest(
            "T2_LAYERED_GLOBAL",
            topic,
            meta_path,
            _MODEL,
            plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit, "refusal": "insufficient_global_evidence"},
        )
        metrics.save()
        return

    import collections
    def _render_claim_verdict(topic, doc_summaries, meta_n_total, outline_plan, refs_by_docid):
        """v15.9 (T1 revival): render a claim-evaluator verdict.

        Reuses everything the T2 pipeline produced up to Stage 2 posture,
        but stops before the writer. Output: a markdown artifact grouping
        papers by their relation to the claim, with per-paper claim +
        top-quote so a reader can grade the corpus's aggregate stance
        without paying for the ~6-min writer run.
        """
        clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
        doc_by_id = {d.get("doc_id"): d for d in doc_summaries}
        relations = outline_plan.get("relation_distribution", {}) or {}
        topic_shape = outline_plan.get("topic_shape", "?")
        topic_cause = outline_plan.get("topic_cause", "") or ""
        topic_outcome = outline_plan.get("topic_outcome", "") or ""

        # Group by relation, preserving ordered_cluster_ids order within each
        by_relation = {}
        for cid in outline_plan.get("ordered_cluster_ids", []) or list(clusters_by_id.keys()):
            c = clusters_by_id.get(cid)
            if not c:
                continue
            rel = c.get("relation", "adjacent")
            by_relation.setdefault(rel, []).append(c)

        # Ordering: supporting first, then rival, then upstream/downstream,
        # then adjacent, then anything else. Makes the verdict readable
        # top-to-bottom as "the claim is supported by X, contested by Y..."
        rel_order = [
            "same_as_topic_cause", "same_as_topic_outcome",
            "supports", "supports_claim",
            "rival_to_topic_cause", "rival",
            "upstream_of_topic_cause", "upstream",
            "downstream_of_topic_cause", "downstream",
            "adjacent", "unassigned",
        ]
        rel_label = {
            "same_as_topic_cause": "Support the claim",
            "same_as_topic_outcome": "Support the claim (via same outcome)",
            "supports": "Support the claim",
            "supports_claim": "Support the claim",
            "rival_to_topic_cause": "Propose a rival explanation",
            "rival": "Propose a rival explanation",
            "upstream_of_topic_cause": "Identify an upstream mechanism",
            "upstream": "Identify an upstream mechanism",
            "downstream_of_topic_cause": "Trace a downstream implication",
            "downstream": "Trace a downstream implication",
            "adjacent": "Adjacent (related but not addressing the claim directly)",
        }
        seen = set()
        ordered_relations = [r for r in rel_order if r in by_relation and not seen.add(r)]
        ordered_relations += [r for r in by_relation if r not in seen]

        # Aggregate counts across DOCS (not clusters) for the headline verdict
        docs_by_rel = {rel: 0 for rel in by_relation}
        for cid, c in clusters_by_id.items():
            docs_by_rel.setdefault(c.get("relation", "adjacent"), 0)
            docs_by_rel[c["relation"]] = docs_by_rel.get(c["relation"], 0) + len(c.get("doc_ids", []))
        total_admitted = len(doc_summaries)
        n_support = sum(v for k, v in docs_by_rel.items() if k in ("same_as_topic_cause", "same_as_topic_outcome", "supports", "supports_claim"))
        n_rival = sum(v for k, v in docs_by_rel.items() if k in ("rival_to_topic_cause", "rival"))
        n_upstream = sum(v for k, v in docs_by_rel.items() if k in ("upstream_of_topic_cause", "upstream"))
        n_downstream = sum(v for k, v in docs_by_rel.items() if k in ("downstream_of_topic_cause", "downstream"))
        n_adjacent = sum(v for k, v in docs_by_rel.items() if k in ("adjacent",))

        def _headline_verdict():
            if total_admitted == 0:
                return "No corpus evidence"
            if n_support == 0 and n_rival == 0:
                return f"Under-engaged — the corpus does not address the claim directly ({n_adjacent} adjacent papers)"
            if n_rival >= 2 * max(1, n_support):
                return f"Contested — rival explanations dominate ({n_rival} rival vs {n_support} supporting)"
            if n_support >= 2 * max(1, n_rival):
                return f"Well-supported — supporting evidence dominates ({n_support} supporting vs {n_rival} rival)"
            return f"Mixed — {n_support} supporting, {n_rival} rival, {n_upstream} upstream, {n_downstream} downstream"

        lines: list = []
        lines.append(f"# Claim evaluation\n")
        lines.append(f"**Claim:** {topic}\n")
        lines.append(f"**Topic shape:** {topic_shape}"
                     + (f" (explanans: {topic_cause}; explanandum: {topic_outcome})" if topic_shape == "causal" and topic_cause else "")
                     + "\n")
        lines.append(f"**Corpus:** {total_admitted} of {meta_n_total} papers admitted ({100 * total_admitted // max(1, meta_n_total)}%).\n")
        lines.append(f"**Aggregate verdict:** {_headline_verdict()}.\n")
        lines.append(f"**Breakdown:** support={n_support}, rival={n_rival}, upstream={n_upstream}, downstream={n_downstream}, adjacent={n_adjacent}.\n")
        lines.append("---\n")

        for rel in ordered_relations:
            clusters = by_relation.get(rel, [])
            if not clusters:
                continue
            n_docs = sum(len(c.get("doc_ids", [])) for c in clusters)
            lines.append(f"## {rel_label.get(rel, rel.replace('_', ' ').title())} ({n_docs} papers across {len(clusters)} cluster{'s' if len(clusters) != 1 else ''})\n")
            for c in clusters:
                lines.append(f"### {c.get('shared_thread', c.get('cluster_id', 'cluster'))}")
                elab = (c.get("elaboration") or "").strip()
                if elab:
                    lines.append(f"*{elab}*\n")
                for did in c.get("doc_ids", []):
                    d = doc_by_id.get(did) or {}
                    cite = d.get("citation") or refs_by_docid.get(did) or did
                    lines.append(f"- **{cite}**")
                    claim = (d.get("claim") or "").strip()
                    if claim:
                        lines.append(f"    - Claim: {claim}")
                    for q in (d.get("quotes") or [])[:1]:
                        text = str(q.get("text", "") or "").strip()
                        page = q.get("page")
                        if text and page:
                            snippet = text if len(text) < 220 else text[:220].rstrip() + "…"
                            lines.append(f'    - Quote (p.{page}): "{snippet}"')
                lines.append("")

        unassigned_ids = outline_plan.get("unassigned_doc_ids", []) or []
        if unassigned_ids:
            lines.append(f"## Unassigned ({len(unassigned_ids)} papers)\n")
            lines.append("Papers admitted by BM25 retrieval but not clustered into any stream by Stage 1. Typically they address a different question or use very different vocabulary.\n")
            for did in unassigned_ids:
                d = doc_by_id.get(did) or {}
                cite = d.get("citation") or refs_by_docid.get(did) or did
                lines.append(f"- {cite}")
            lines.append("")

        return "\n".join(lines)


    def _render_review_narrative(topic, doc_summaries, meta_n_total, outline_plan):
        # v15: narrative summary is the outline plan rendered as markdown.
        # No stance buckets. Sections are streams ordered by Stage 3.
        clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
        doc_by_id = {d.get("doc_id"): d for d in doc_summaries}
        lines = []
        lines.append("# Literature review\n")
        lines.append(f"**Topic:** {topic}\n")
        lines.append(f"**Topic shape:** {outline_plan.get('topic_shape','?')}\n")
        lines.append(f"**Coverage:** {len(doc_summaries)} of {meta_n_total} documents "
                     f"({len(clusters_by_id)} streams, "
                     f"{len(outline_plan.get('unassigned_doc_ids', []))} unassigned).\n")
        relations = outline_plan.get("relation_distribution", {}) or {}
        if relations:
            lines.append("**Relation distribution:** " +
                         ", ".join(f"{k}: {v}" for k, v in sorted(relations.items())) + "\n")
        lines.append("---\n")
        for cid in outline_plan.get("ordered_cluster_ids", []) or list(clusters_by_id.keys()):
            c = clusters_by_id.get(cid)
            if not c:
                continue
            lines.append(f"## {c.get('shared_thread', cid)}  ({cid} — {c.get('relation','?')})\n")
            elab = (c.get("elaboration") or "").strip()
            if elab:
                lines.append(f"**Stream posture:** {elab}\n")
            disagreement = (c.get("internal_disagreement") or "").strip()
            if disagreement:
                lines.append(f"**Internal disagreement:** {disagreement}\n")
            citations = []
            for did in c.get("doc_ids", []):
                d = doc_by_id.get(did) or {}
                cite = d.get("citation") or did
                citations.append((cite, d.get("avg_score", 0)))
            citations.sort(key=lambda t: t[1], reverse=True)
            if citations:
                lines.append("**Sources:**")
                lines += [f"- {c0} - avg score {s0:.1f}" for c0, s0 in citations] + [""]
        unassigned_ids = outline_plan.get("unassigned_doc_ids", []) or []
        if unassigned_ids:
            lines.append(f"## Unassigned ({len(unassigned_ids)})\n")
            for did in unassigned_ids:
                d = doc_by_id.get(did) or {}
                lines.append(f"- {d.get('citation', did)}")
            lines.append("")
        return "\n".join(lines)

    ensure_dir(str(runs_path()))

    # v15.9 (#6): propagate doc-level provenance into the ledger so the
    # writer can build citations.json without re-reading metadata.csv.
    _pdf_paths_by_docid = {}
    if "pdf_path" in df.columns:
        _pdf_paths_by_docid = {
            str(r["doc_id"]): str(r["pdf_path"]).strip()
            for _, r in df.iterrows()
            if str(r.get("pdf_path", "")).strip()
        }
    _pdf_page_offsets = {}
    if "pdf_page_offset" in df.columns:
        for _, r in df.iterrows():
            try:
                _pdf_page_offsets[str(r["doc_id"])] = int(r["pdf_page_offset"] or 0)
            except Exception:
                pass
    _dois_by_docid = {}
    if "doi_or_url" in df.columns:
        _dois_by_docid = {
            str(r["doc_id"]): str(r["doi_or_url"]).strip()
            for _, r in df.iterrows()
            if str(r.get("doi_or_url", "")).strip()
        }
    ledger_data = {
        "topic": topic,
        "plan": plan_obj,
        "bypass_condition": os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1",
        "topic_fit": topic_fit,
        "admission": {
            "cache_key": admit_sig,
            "settings": admit_settings,
            "docs_admitted": len(admitted_docs),
            "docs_rejected": len(admission_rejections),
            "docs_selected_for_llm": len(selected_docs),
        },
        "docs": doc_summaries,
        # v15.9 (#6): provenance data for citations.json
        "pdf_paths_by_docid": _pdf_paths_by_docid,
        "pdf_page_offsets": _pdf_page_offsets,
        "dois_by_docid": _dois_by_docid,
        # v15: outline_plan is now the corpus-level structure the writer
        # consumes. It replaces the v11-C cluster_syntheses keyed by
        # "stance::cluster_label" — clusters are top-level, not nested under
        # discrete stance buckets.
        "outline_plan": outline_plan,
    }
    with metrics.stage("write_ledger"):
        # v15.14: atomic via save_json — the ledger is the writer's input; a
        # crash mid-write left a truncated file that broke any replay.
        from rrr.utils import save_json as _save_json_atomic
        _save_json_atomic(ledger_data, runs_path("review_ledger.json"))
        _save_json_atomic(plan_obj, runs_path("plan.json"))

    narrative_md = _render_review_narrative(topic, doc_summaries, len(all_doc_ids), outline_plan)
    with open(runs_path("review_narrative.md"), "w", encoding="utf-8") as f:
        f.write(narrative_md)

    if not getattr(args, "narrative_only", False):
        md_lines = [f"# Literature Review\n", f"**Topic:** {topic}\n",
                    f"\n**Coverage:** {kept} of {len(all_doc_ids)} documents.\n",
                    f"\n**Topic shape:** {outline_plan.get('topic_shape','?')}\n",
                    "\n---\n"]
        # v15: appendix groups docs by their cluster + relation, in the
        # outline's ordered_cluster_ids. Unassigned docs are listed last.
        clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
        doc_by_id = {d.get("doc_id"): d for d in doc_summaries}
        for cid in outline_plan.get("ordered_cluster_ids", []) or list(clusters_by_id.keys()):
            c = clusters_by_id.get(cid)
            if not c:
                continue
            md_lines.append(f"## {c.get('shared_thread', cid)}  ({cid} — {c.get('relation','?')})\n")
            elab = (c.get("elaboration") or "").strip()
            if elab:
                md_lines.append(f"**Stream posture:** {elab}\n")
            for did in c.get("doc_ids", []):
                entry = doc_by_id.get(did)
                if not entry:
                    continue
                md_lines.append(f"### {entry.get('citation', did)}")
                md_lines.append(f"**Relation:** {entry.get('relation', '?')} | "
                                f"**Cluster:** {entry.get('cluster_id', '')} | "
                                f"**Relevance:** {entry.get('avg_score', 0):.1f}")
                claim = (entry.get("claim") or "").strip()
                if claim:
                    md_lines.append(f"**Claim:** {claim}")
                if entry.get("quotes"):
                    md_lines.append("**Quotes (page-level, with scores):**")
                    for q in entry["quotes"][:MD_QUOTE_CAP]:
                        md_lines.append(f"- p.{q['page']} [score={q.get('score',0):.0f}]: \"{q['text']}\"")
                md_lines.append("")
        unassigned_ids = outline_plan.get("unassigned_doc_ids", []) or []
        if unassigned_ids:
            md_lines.append(f"## Unassigned ({len(unassigned_ids)})\n")
            for did in unassigned_ids:
                entry = doc_by_id.get(did) or {}
                md_lines.append(f"- {entry.get('citation', did)}")
            md_lines.append("")
        with open(runs_path("T2_review.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    # v13: RRR_WRITE_REVIEW retired. The writer is the headline architecture
    # output; the narrative-only short summary is preserved but only printed
    # when narrative_only is requested. Composition now runs unconditionally
    # so the composed review always lands on disk.
    print("\n[Layered-T2] wrote: runs/review_narrative.md and runs/review_ledger.json")
    if not getattr(args, "narrative_only", False):
        print("[Layered-T2] appendix: runs/T2_review.md")

    # v15.9 (T1 revival): claim-eval mode. Stop after Stage 0/1/2 + ledger
    # write. Emit runs/claim_verdict.md with per-cluster + aggregate.
    # Reuses all upstream invariants (retrieval, admission, quote-verify)
    # but skips the ~6-min writer + validation chain.
    if getattr(args, "t1_only", False) or getattr(args, "claim_only", False):
        try:
            verdict_md = _render_claim_verdict(topic, doc_summaries, len(all_doc_ids), outline_plan, refs)
            with open(runs_path("claim_verdict.md"), "w", encoding="utf-8") as f:
                f.write(verdict_md)
            print("\n" + "=" * 80)
            print("CLAIM VERDICT")
            print("=" * 80 + "\n")
            print(verdict_md)
            print("\n" + "=" * 80 + "\n")
            print("[Layered-T1] claim_verdict.md written")
            metrics.set("claim_verdict_written", True)
            metrics.save()
            write_run_manifest(
                "T1_CLAIM_EVAL",
                topic,
                meta_path,
                _MODEL,
                plan=plan_obj,
                extra={
                    "admit_settings": admit_settings,
                    "topic_fit": topic_fit,
                    "outputs": ["claim_verdict.md", "review_ledger.json", "outline_plan.json"],
                    "mode": "t1_claim_eval",
                },
            )
            return
        except Exception as e:
            print(f"[Layered-T1] verdict render failed, falling through to T2: {e}")

    if True:  # v13: writer composition is unconditional (was RRR_WRITE_REVIEW gate)
        # v14: RRR_WRITER_MODE dispatches between the single-pass (default) and
        # chunked writer paths. The chunked path is preserved as a measurement
        # fallback so v13.1.1 5-topic smokes remain reproducible.
        from rrr.writer import compose_review
        print("[Layered-T2] composing long-form literature review...")
        writer_error_msg = None
        try:
            with metrics.stage("writing"):
                composed_path = compose_review(str(runs_path("review_ledger.json")), metrics=metrics)
            print(f"[Layered-T2] composed review written at: {composed_path}")

            with open(composed_path, "r", encoding="utf-8") as f:
                long_form = f.read()

            print("\n" + "="*80)
            print("LONG-FORM LITERATURE REVIEW")
            print("="*80 + "\n")
            print(long_form)
            print("\n" + "="*80 + "\n")

            allowed_docs = set()
            allowed_pages_by_doc = {}
            for d in doc_summaries:
                did = str(d.get("doc_id", "")).strip()
                if not did:
                    continue
                allowed_docs.add(did)
                allowed_pages_by_doc.setdefault(did, set())
                for q in (d.get("quotes") or []):
                    qdid = str(q.get("doc_id", did)).strip() or did
                    try:
                        pg = int(q.get("page", 0) or 0)
                    except Exception:
                        pg = 0
                    if qdid and pg > 0:
                        allowed_docs.add(qdid)
                        allowed_pages_by_doc.setdefault(qdid, set()).add(pg)

            cited_docs_path = str(runs_path("review_cited_docs.json"))
            if os.path.isfile(cited_docs_path):
                with open(cited_docs_path, "r", encoding="utf-8") as f:
                    cited_docids = json.load(f)
            else:
                author_year_to_docid = _build_author_year_lookup(allowed_docs)
                cited_docs = _collect_cited_docs(long_form, allowed_docs, author_year_to_docid)
                cited_docids = list(cited_docs)

            if not cited_docids:
                ensure_dir(str(runs_path()))
                with open(runs_path("review_reference_build.failures.txt"), "w", encoding="utf-8") as f:
                    f.write("No valid citations found in review_composed.md.\n")
                print("\n" + "="*80)
                print("REFERENCES (cited in review)")
                print("="*80 + "\n")
                print("[REFUSAL] No citations found. See runs/review_reference_build.failures.txt")
                print("\n" + "="*80 + "\n")
                metrics.set("refusal", True)
                metrics.set("refusal_reason", "no_citations_found")
                write_run_manifest(
                    "T2_LAYERED_GLOBAL",
                    topic,
                    meta_path,
                    _MODEL,
                    plan=plan_obj,
                    extra={"admit_settings": admit_settings, "topic_fit": topic_fit, "refusal": "no_citations_found"},
                )
                metrics.save()
                return

            raw_ref_lines = [(did, refs.get(did, did)) for did in cited_docids]

            def sort_key(item):
                did = item[0]
                clean = did.replace("EtAl", "").replace("&", "")
                parts = clean.split("_")
                return parts[0].lower() if parts else did.lower()

            raw_ref_lines = sorted(raw_ref_lines, key=sort_key)

            seen_refs = set()
            ref_lines = []
            for did, rline in raw_ref_lines:
                if rline not in seen_refs:
                    ref_lines.append(rline)
                    seen_refs.add(rline)

            print("\n" + "="*80)
            print("REFERENCES (cited in review)")
            print("="*80 + "\n")
            for i, rline in enumerate(ref_lines, start=1):
                print(f"{i}. {rline}")
            print("\n" + "="*80 + "\n")

            ensure_dir(str(runs_path()))
            with open(runs_path("review_references.txt"), "w", encoding="utf-8") as f:
                for i, rline in enumerate(ref_lines, start=1):
                    f.write(f"{i}. {rline}\n")

            # v15.9: append the reference list to review_composed.md so the
            # single-file artifact is self-contained (the reader doesn't need
            # to open a second file to see what was cited). The separate
            # review_references.txt is preserved for backward compat with
            # scripts that read it directly.
            composed_md = runs_path("review_composed.md")
            if composed_md.is_file():
                try:
                    with open(composed_md, "a", encoding="utf-8") as f:
                        f.write("\n\n---\n\n## References\n\n")
                        for i, rline in enumerate(ref_lines, start=1):
                            f.write(f"{i}. {rline}\n")
                    print(f"[Layered-T2] appended {len(ref_lines)} references to review_composed.md")
                except Exception as ref_e:
                    print(f"[Layered-T2] failed to append references to review_composed.md: {ref_e}")

        except Exception as e:
            print(f"[Layered-T2] writer failed: {e}")
            metrics.set("writer_error", str(e))
            writer_error_msg = str(e)

    metrics.set("refusal", False)
    # v15.14: the manifest previously recorded an unconditional success shape
    # even when the writer crashed — downstream automation reading it saw a
    # healthy run with no review_composed.md. Record writer_failed explicitly
    # and only list outputs that actually exist on disk.
    outputs = [
        "review_ledger.json",
        "review_narrative.md",
        "topic_fit.json",
        "admission_rejections.json",
        "run_metrics.json",
    ]
    if os.path.exists(str(runs_path("review_composed.md"))):
        outputs.append("review_composed.md")
    write_run_manifest(
        "T2_LAYERED_GLOBAL",
        topic,
        meta_path,
        _MODEL,
        plan=plan_obj,
        extra={
            "admit_settings": admit_settings,
            "topic_fit": topic_fit,
            "writer_failed": bool(writer_error_msg),
            "writer_error": writer_error_msg,
            "outputs": outputs,
        },
    )
    metrics.save()


def layered_t2(args, meta_path):
    """Run the current layered T2 pipeline once."""
    return _layered_t2_inner(args, meta_path)
