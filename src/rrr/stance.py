import hashlib
import json
import os
from functools import lru_cache
import time
from rrr.paths import page_text_path, require_page_text_dir, runs_path

# v11.2 lever 2: stance is logically part of reasoning, so prefer
# RRR_REASONER_MODEL if set; fall back to RRR_MODEL otherwise.
_MODEL = os.environ.get("RRR_STANCE_MODEL", os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral")))
_STANCE_TOKENS = {"supports", "critiques", "complicates", "tangential"}

# v14.4 Shape B: paper-claim extraction prompt version. Bump to invalidate the
# claims cache on prompt changes. Cache key composes (doc_id, model,
# prompt_version, abstract_hash, conclusion_hash) so unrelated cache entries
# stay valid.
_CLAIM_PROMPT_VERSION = "2026-06-29-v1-claim"


def _parse_stance_token(raw: str):
    token = (raw or "").strip().lower()
    token = token.strip("`'\" .,:;()[]{}")
    return token if token in _STANCE_TOKENS else None

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


def _claim_cache_path(doc_id: str, sig: str):
    """Return Path to runs/cache/claims/<doc_id>_<sig>.json. The directory is
    created on demand."""
    cache_dir = runs_path("cache", "claims")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_id}_{sig}.json"


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


def extract_paper_claim(doc_id: str, metrics=None) -> dict:
    """v14.4 Shape B: extract the paper's central claim from its abstract +
    conclusion using one cheap LLM call. Cached PER PAPER (not per topic) so
    a corpus is processed once across all topics — ~50 calls × ~5s = ~4 min
    one-time, then every subsequent topic against the same corpus is free.

    Returns {claim: str, source: str, duration_s: float}. On empty
    abstract+conclusion or LLM failure, returns an empty claim with a
    descriptive `source` field so the consumer can fall back to evidence-only
    classification.

    The claim is intentionally short (1-2 sentences, ~120-220 chars) so it
    can be injected verbatim into the fused-stance prompt without bloating
    the context window.
    """
    abstract = _get_abstract(doc_id)
    conclusion = _get_conclusion(doc_id)
    if not abstract and not conclusion:
        return {"claim": "", "source": "no_text", "duration_s": 0.0}

    sig = _claim_sig(doc_id, abstract, conclusion, _MODEL)
    cache_path = _claim_cache_path(doc_id, sig)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("claim") is not None:
                cached.setdefault("source", "cache")
                cached.setdefault("duration_s", 0.0)
                if metrics:
                    metrics.cache_event("paper_claim", "hits")
                return cached
        except Exception:
            pass  # corrupt cache; fall through and recompute

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
        # downstream stance prompt.
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

@lru_cache(maxsize=256)
def _classify_stance_cached(doc_id: str, topic: str, model: str):
    abstract = _get_abstract(doc_id)
    if len(abstract) < 100:
        return {"stance": "tangential", "source": "no_abstract", "duration_s": 0.0}
    
    prompt = f"""Classify this paper's position on: "{topic}"

ABSTRACT:
{abstract}

Reply with ONE word only: SUPPORTS, CRITIQUES, COMPLICATES, or TANGENTIAL"""

    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(model=model, messages=[{"role":"user","content":prompt}],
                          options={"temperature":0.0,"num_ctx":2048,"num_predict":20},
                          keep_alive="30m", stream=False)
        duration_s = time.perf_counter() - start
        raw = res["message"]["content"].strip()
        stance = _parse_stance_token(raw)
        if stance:
            return {"stance": stance, "source": "llm", "duration_s": duration_s, "response_chars": len(raw)}
        retry_prompt = prompt + "\n\nYour previous reply was invalid. Reply with exactly one token."
        retry_start = time.perf_counter()
        retry = ollama.chat(model=model, messages=[{"role":"user","content":retry_prompt}],
                            options={"temperature":0.0,"num_ctx":2048,"num_predict":8},
                            keep_alive="30m", stream=False)
        duration_s += time.perf_counter() - retry_start
        raw_retry = retry["message"]["content"].strip()
        stance = _parse_stance_token(raw_retry)
        if stance:
            return {"stance": stance, "source": "llm_retry", "duration_s": duration_s, "response_chars": len(raw) + len(raw_retry)}
        return {"stance": "tangential", "source": "invalid_reply", "duration_s": duration_s, "raw": raw_retry}
    except Exception as e:
        return {"stance": "tangential", "source": "error", "error": str(e), "duration_s": 0.0}
    return {"stance": "tangential", "source": "fallback", "duration_s": 0.0}


def classify_stance(doc_id: str, topic: str, metrics=None) -> str:
    """Classify stance from the first page proxy. Cached by (doc_id, topic, model)."""
    before = _classify_stance_cached.cache_info()
    info = _classify_stance_cached(doc_id, topic, _MODEL)
    after = _classify_stance_cached.cache_info()
    if metrics:
        if after.hits > before.hits:
            metrics.cache_event("stance", "hits")
        else:
            metrics.cache_event("stance", "misses")
            if info.get("source") == "llm":
                metrics.record_llm(
                    "stance", _MODEL,
                    options={"temperature":0.0,"num_ctx":2048,"num_predict":20},
                    duration_s=info.get("duration_s"),
                    response_chars=info.get("response_chars"),
                )
            elif info.get("source") == "error":
                metrics.record_llm(
                    "stance", _MODEL,
                    options={"temperature":0.0,"num_ctx":2048,"num_predict":20},
                    success=False,
                    error=info.get("error"),
                )
    return info.get("stance", "tangential")


def classify_evidence_stance(doc_id: str, topic: str, quotes, metrics=None) -> str:
    """Classify stance from validated quote evidence, falling back to first-page stance."""
    snippets = []
    for q in quotes or []:
        text = str(q.get("text", "")).strip()
        if text:
            snippets.append(f"- p.{q.get('page')}: {text[:500]}")
        if len(snippets) >= 6:
            break
    if not snippets:
        return classify_stance(doc_id, topic, metrics=metrics)

    prompt = f"""Classify this document's position on: "{topic}"

Use only these validated excerpts:
{chr(10).join(snippets)}

Reply with exactly one token:
SUPPORTS
CRITIQUES
COMPLICATES
TANGENTIAL"""

    options = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 8}
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
        raw = res["message"]["content"].strip()
        stance = _parse_stance_token(raw)
        duration_s = time.perf_counter() - start
        if not stance:
            retry = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt + "\n\nInvalid reply. Return one token only."}],
                options=options,
                keep_alive="30m",
                stream=False,
            )
            raw_retry = retry["message"]["content"].strip()
            stance = _parse_stance_token(raw_retry)
            duration_s = time.perf_counter() - start
            raw = raw + " | retry: " + raw_retry
        if metrics:
            metrics.record_llm(
                "stance_evidence",
                _MODEL,
                options=options,
                duration_s=duration_s,
                prompt_chars=len(prompt),
                response_chars=len(raw),
                success=bool(stance),
                error=None if stance else f"invalid stance reply: {raw}",
            )
        return stance or classify_stance(doc_id, topic, metrics=metrics)
    except Exception as e:
        if metrics:
            metrics.record_llm("stance_evidence", _MODEL, options=options, success=False, error=e)
        return classify_stance(doc_id, topic, metrics=metrics)
