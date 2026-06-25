import os, json, hashlib
from functools import lru_cache
import time
from rrr.paths import page_text_path, require_page_text_dir

_MODEL = os.environ.get("RRR_MODEL", "mistral")

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
        raw = res["message"]["content"].strip().lower()
        for s in ["supports","critiques","complicates","tangential"]:
            if s in raw:
                return {"stance": s, "source": "llm", "duration_s": duration_s, "response_chars": len(raw)}
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
