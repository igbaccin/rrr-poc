import hashlib
import json
import os
import time
from rrr.paths import claim_cache_path, page_text_path, require_page_text_dir

# Claim extraction is a reasoning stage, so it uses the reasoner model and
# falls back to the shared model.
_MODEL = os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))

# v14.4 Shape B: paper-claim extraction prompt version. Bump to invalidate the
# claims cache on prompt changes. Cache key composes (doc_id, model,
# prompt_version, abstract_hash, conclusion_hash) so unrelated cache entries
# stay valid.
_CLAIM_PROMPT_VERSION = "2026-06-29-v1-claim"


def _get_abstract(doc_id: str) -> str:
    """Get first page as abstract proxy."""
    require_page_text_dir()
    path = page_text_path(doc_id, 1)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return f.read()[:1500]
    return ""


def _get_conclusion(doc_id: str, max_chars: int = 1500) -> str:
    """v14.4 Shape B: read the LAST surviving content page as a conclusion
    proxy. The cleanup pipeline may have dropped pages (e.g. publisher
    splashes), so we scan for the highest-numbered page_text file rather than
    assuming a contiguous range. Returns up to `max_chars` from the end of
    that page (the conclusion usually sits at the END of the last content
    page, so we tail the text rather than head it).
    """
    require_page_text_dir()
    page_dir = page_text_path(doc_id, 1).parent
    highest = 0
    for entry in page_dir.glob(f"{doc_id}_page_*.txt"):
        try:
            n = int(entry.stem.rsplit("_page_", 1)[1])
        except (ValueError, IndexError):
            continue
        if n > highest:
            highest = n
    # Single-page paper: the "conclusion" is the abstract, so return empty to
    # avoid feeding the same text to the LLM under two different labels.
    # The claim-extraction prompt explicitly handles "(no conclusion available)".
    if highest <= 1:
        return ""
    path = page_text_path(doc_id, highest)
    if not path.is_file():
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text[-max_chars:] if len(text) > max_chars else text


def _claim_cache_path(paper_key: str, sig: str, corpus_fingerprint: str = None):
    """Return Path to the claim-cache entry for this paper.

    v15.9 (#5): key by content_sha1 when available, doc_id as fallback,
    and namespace by corpus_fingerprint. Two corpora with a shared doc_id
    ('Ogilvie_2007' in Corpus A and 'Ogilvie_2007' in Corpus B, DIFFERENT
    papers) no longer poison each other's cache. Same PDF ingested in two
    corpora hits the shared entry (via content_sha1) without duplicating
    work.

    Layout:
      <root>/                         (legacy, flat — read-only fallback)
      <root>/<corpus_fp>/             (v15.9 primary; corpus_fp is 16 hex)

    File name is `<paper_key>_<sig>.json` in both layouts. Legacy entries
    remain readable via the fall-through in extract_paper_claim.

    v15.0.2 rationale still applies: cache lives OUTSIDE runs/ so the per-
    topic smoke harness cannot evict it; RRR_CLAIM_CACHE_DIR points at a
    persistent volume on pods.
    """
    if corpus_fingerprint:
        cache_dir = claim_cache_path(corpus_fingerprint)
    else:
        cache_dir = claim_cache_path()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{paper_key}_{sig}.json"


def _claim_sig(doc_id: str, abstract: str, conclusion: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(_CLAIM_PROMPT_VERSION.encode())
    h.update(b"\x00")
    h.update(model.encode())
    h.update(b"\x00")
    h.update(abstract.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(conclusion.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def compute_corpus_fingerprint(df) -> str:
    """v15.9 (#5): stable fingerprint of a metadata.csv DataFrame used to
    namespace the claim cache. Prefers content_sha1 (when populated by the
    ingest cascade) so identical PDFs across corpora share fingerprint
    components; falls back to sorted doc_id list when content_sha1 is
    missing (existing 50-paper hand-curated corpus). Returns 16 hex chars.
    """
    h = hashlib.sha256()
    h.update(b"rrr-corpus-fp-v1\x00")
    keys: list = []
    if df is not None and "content_sha1" in df.columns:
        col = df["content_sha1"].fillna("").astype(str)
        vals = [v for v in col.tolist() if v]
        if vals:
            keys = sorted(vals)
            h.update(b"via=content_sha1\x00")
    if not keys and df is not None and "doc_id" in df.columns:
        keys = sorted(str(x) for x in df["doc_id"].fillna("").astype(str).tolist() if x)
        h.update(b"via=doc_id\x00")
    for k in keys:
        h.update(k.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def extract_paper_claim(doc_id: str, metrics=None, *,
                         corpus_fingerprint: str = None,
                         content_sha1: str = None) -> dict:
    """v14.4 Shape B: extract the paper's central claim from its abstract +
    conclusion using one cheap LLM call. Cached PER PAPER (not per topic) so
    a corpus is processed once across all topics — ~50 calls × ~5s = ~4 min
    one-time, then every subsequent topic against the same corpus is free.

    Returns {claim: str, source: str, duration_s: float}. On empty
    abstract+conclusion or LLM failure, returns an empty claim with a
    descriptive `source` field. The corpus-level outline can then work from
    validated evidence alone.

    The claim is intentionally short (1-2 sentences, ~120-220 chars) so it
    can be supplied to the corpus-level clustering prompt without bloating
    the context window.
    """
    abstract = _get_abstract(doc_id)
    conclusion = _get_conclusion(doc_id)
    if not abstract and not conclusion:
        return {"claim": "", "source": "no_text", "duration_s": 0.0}

    sig = _claim_sig(doc_id, abstract, conclusion, _MODEL)
    # v15.9 (#5): try three cache locations in priority order:
    #   1. namespaced content_sha1: <root>/<corpus_fp>/<content_sha1>_<sig>.json
    #   2. namespaced doc_id:      <root>/<corpus_fp>/<doc_id>_<sig>.json
    #   3. legacy flat doc_id:     <root>/<doc_id>_<sig>.json
    # First hit wins. Writes always go to the primary (namespaced,
    # content_sha1-preferred) so new entries land in the v15.9 layout.
    paper_key = content_sha1 or doc_id
    primary_path = _claim_cache_path(paper_key, sig, corpus_fingerprint=corpus_fingerprint)
    fallback_paths = []
    if content_sha1 and doc_id != content_sha1:
        fallback_paths.append(_claim_cache_path(doc_id, sig, corpus_fingerprint=corpus_fingerprint))
    fallback_paths.append(_claim_cache_path(doc_id, sig, corpus_fingerprint=None))
    for cp in [primary_path, *fallback_paths]:
        if cp.is_file():
            try:
                cached = json.loads(cp.read_text(encoding="utf-8"))
                # v15.14: truthy check, not `is not None` — pre-fix cache
                # files hold claim: "" (a cached FAILURE) and were served
                # forever; empty claims should re-extract like any miss.
                if isinstance(cached, dict) and str(cached.get("claim") or "").strip():
                    cached.setdefault("source", "cache")
                    cached.setdefault("duration_s", 0.0)
                    if metrics:
                        metrics.cache_event("paper_claim", "hits")
                    return cached
            except Exception:
                pass  # corrupt cache entry; try the next fallback
    cache_path = primary_path  # writes below always go here

    if metrics:
        metrics.cache_event("paper_claim", "misses")

    prompt = (
        "You are reading a single academic paper. Your task is to identify "
        "the paper's CENTRAL CLAIM in ONE sentence (max 220 chars).\n\n"
        "A central claim is what the paper ARGUES, not what it describes. "
        "It should answer: \"what does this author try to convince the "
        "reader is true?\" — using the author's own framing.\n\n"
        "Write it in the form: \"The paper argues X, contra Y.\" or "
        "\"The paper argues X.\" Be concrete about X (the claim) and, "
        "when the paper attacks a rival position, Y (the rival).\n\n"
        f"ABSTRACT (first page of the paper):\n{abstract or '(no abstract available)'}\n\n"
        f"CONCLUSION (last page of the paper):\n{conclusion or '(no conclusion available)'}\n\n"
        "Reply with ONLY the central-claim sentence. No preamble, no "
        "quotation marks, no labels."
    )

    options = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 120}
    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive="30m",
            stream=False,
        )
        duration = time.perf_counter() - start
        raw = (res.get("message", {}).get("content") or "").strip()
        # Strip enclosing quotes the model sometimes adds despite instruction.
        raw = raw.strip("`'\"")
        # Clip to a hard char budget so a runaway response doesn't poison the
        # downstream outline prompt.
        claim = raw[:240].strip()
        result = {
            "claim": claim,
            "source": "llm" if claim else "llm_empty",
            "duration_s": round(duration, 3),
            "model": _MODEL,
            "prompt_version": _CLAIM_PROMPT_VERSION,
            "abstract_chars": len(abstract),
            "conclusion_chars": len(conclusion),
        }
        # Only cache successful (non-empty) extractions. An empty LLM response
        # for a paper with non-empty abstract+conclusion is most likely a
        # transient model failure; not caching it means the next run retries.
        if claim:
            try:
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                if metrics:
                    metrics.cache_event("paper_claim", "writes")
            except Exception:
                pass  # cache write failure is non-fatal
        if metrics:
            metrics.record_llm(
                "paper_claim", _MODEL, options=options,
                duration_s=duration,
                prompt_chars=len(prompt),
                response_chars=len(raw),
            )
        return result
    except Exception as e:
        if metrics:
            metrics.record_llm("paper_claim", _MODEL, options=options,
                               success=False, error=e)
        return {"claim": "", "source": "error", "error": str(e), "duration_s": 0.0}
