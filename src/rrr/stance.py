import os, json, hashlib
from functools import lru_cache

_MODEL = os.environ.get("RRR_MODEL", "mistral")

def _get_abstract(doc_id: str) -> str:
    """Get first page as abstract proxy."""
    path = f"data/page_text/{doc_id}_page_1.txt"
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()[:1500]
    return ""

@lru_cache(maxsize=256)
def classify_stance(doc_id: str, topic: str) -> str:
    """Classify stance from abstract. Cached by (doc_id, topic)."""
    abstract = _get_abstract(doc_id)
    if len(abstract) < 100:
        return "tangential"
    
    prompt = f"""Classify this paper's position on: "{topic}"

ABSTRACT:
{abstract}

Reply with ONE word only: SUPPORTS, CRITIQUES, COMPLICATES, or TANGENTIAL"""

    try:
        import ollama
        res = ollama.chat(model=_MODEL, messages=[{"role":"user","content":prompt}],
                          options={"temperature":0.0,"num_ctx":2048,"num_predict":20},
                          keep_alive="30m", stream=False)
        raw = res["message"]["content"].strip().lower()
        for s in ["supports","critiques","complicates","tangential"]:
            if s in raw:
                return s
    except:
        pass
    return "tangential"
