from rrr.utils import normalize_space, jaccard
import os

def _norm(s: str) -> str:
    s = normalize_space(s or "")
    # normalise soft hyphens and curly quotes so "exact" survives trivial OCR differences
    s = s.replace("\u00AD","")
    s = s.replace("’","'")
    s = s.replace("“",'"').replace("”",'"')
    return s

# ---------- Phase B3: cached page-text loader ----------
from functools import lru_cache

@lru_cache(maxsize=1024)
def load_page_text(doc_id, page:int):
    """Load and memoize page text to avoid redundant disk reads."""
    path = f"data/page_text/{doc_id}_page_{page}.txt"
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def quote_exact(page_text: str, snippet: str) -> bool:
    return _norm(page_text).find(_norm(snippet)) != -1

def quote_soft(page_text: str, snippet: str, threshold: float = 0.78):
    a = set(_norm(snippet).lower().split())
    b = set(_norm(page_text).lower().split())
    inter = len(a & b); union = len(a | b) or 1
    score = inter / union
    return (score >= threshold, score)

def validate_evidence_verbose(evidence, metadata_df, soft_threshold: float = 0.78):
    """
    Returns per-item dicts with:
      verdict: "exact" | "soft_ok" | "fail"
      exact: bool, soft: bool, soft_score: float, reason: str
    """
    results = []
    known = set(str(x) for x in metadata_df["doc_id"])
    for it in evidence:
        doc_id = str(it.get("doc_id",""))
        page   = int(it.get("page", 1))
        text   = it.get("text") or it.get("quote","") or ""
        if doc_id not in known:
            results.append({"item": it, "verdict":"fail", "reason":"unknown_doc", "exact":False, "soft":False, "soft_score":0.0})
            continue
        page_text = load_page_text(doc_id, page)
        if page_text is None:
            results.append({"item": it, "verdict":"fail", "reason":"page_out_of_range", "exact":False, "soft":False, "soft_score":0.0})
            continue
        if quote_exact(page_text, text):
            results.append({"item": it, "verdict":"exact", "reason":"", "exact":True, "soft":True, "soft_score":1.0})
            continue
        ok_soft, score = quote_soft(page_text, text, threshold=soft_threshold)
        if ok_soft:
            results.append({"item": it, "verdict":"soft_ok", "reason":"", "exact":False, "soft":True, "soft_score":score})
        else:
            results.append({"item": it, "verdict":"fail", "reason":"quote_not_found", "exact":False, "soft":False, "soft_score":score})
    return results

# Backwards-compat wrapper (treat exact or soft_ok as ok)
def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):
    out = []
    for r in validate_evidence_verbose(evidence, metadata_df, soft_threshold=soft_threshold):
        ok = r["verdict"] in ("exact","soft_ok")
        out.append({"item": r["item"], "ok": ok, "reason": r["reason"]})
    return out
