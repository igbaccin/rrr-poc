import os, json, hashlib
from functools import lru_cache
import time
from rrr.paths import page_text_path, require_page_text_dir

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_STANCE_TOKENS = {"supports", "critiques", "complicates", "tangential"}


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
