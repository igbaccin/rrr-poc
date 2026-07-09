import pickle

import numpy as np
from functools import lru_cache
from rrr.paths import indices_path, page_text_path, require_file, require_indices_dir, require_page_text_dir
from rrr.text import tokenize_query
from rrr.utils import env_int

@lru_cache(maxsize=1)
def _load_bm25_and_ids():
    require_indices_dir()
    with open(require_file(indices_path("bm25.pkl"), "BM25 index"),"rb") as f:
        bm = pickle.load(f)
    page_ids = np.load(require_file(indices_path("page_ids.npy"), "BM25 page-id index"), allow_pickle=True).tolist()
    return bm, page_ids

def _split_pid(pid):
    if "_page_" in pid:
        did, page = pid.split("_page_")
        try: return did, int(page)
        except: return did, 1
    return pid, 1

# ---------- Phase B2: cache BM25 scoring per topic ----------
# v16.16: maxsize was 8 but the planner issues up to _PROBE_CAP=12 distinct
# probe queries per run (query_planner.py). An 8-slot LRU cycled by >8 distinct
# keys thrashes to ~100% miss -> a full-corpus BM25 rescan per (probe, doc).
# 32 holds every probe for the whole run; byte-identical output, fewer rescans.
@lru_cache(maxsize=32)
def _scores_for_query_cached(query: str):
    """Compute BM25 scores once per topic, then reuse.

    v8: uses tokenize_query (preserves hyphens, smaller stoplist) so multi-word
    concepts like 'rule of law', 'long-run growth' don't collapse before scoring.
    """
    bm, page_ids = _load_bm25_and_ids()
    toks = tokenize_query(query)
    scores = bm.get_scores(toks)
    return bm, page_ids, toks, scores


def retrieve(query: str, topk=20, doc_id=None):
    # Legacy entry point: keep working by delegating
    if doc_id:
        return retrieve_doc_pages(query, doc_id, pages_per_doc=topk)
    else:
        return retrieve_breadth(query, docs_k=max(10, topk//2), pages_per_doc=2)

def retrieve_doc_pages(query: str, doc_id: str, pages_per_doc=4):
    require_page_text_dir()
    bm, page_ids, toks, scores = _scores_for_query_cached(query)
    pairs = [(i, scores[i]) for i, pid in enumerate(page_ids) if pid.startswith(f"{doc_id}_page_")]
    pairs.sort(key=lambda x: x[1], reverse=True)
    # v8: drop zero-score pages so they don't occupy candidate slots that
    # downstream filters then waste LLM calls trying to extract signal from.
    pairs = [p for p in pairs if p[1] > 0.0]
    out = []
    for i, score in pairs[:max(1, pages_per_doc)]:
        pid = page_ids[i]; did, page = _split_pid(pid)
        txt_path = page_text_path(did, page)
        snippet = ""
        if txt_path.exists():
            with open(txt_path, encoding="utf-8") as f:
                snippet = f.read()
        out.append({"doc_id": did, "page": page, "text": snippet, "bm25_score": float(score)})
    return out

def retrieve_breadth(query: str, docs_k=20, pages_per_doc=2):
    require_page_text_dir()
    bm, page_ids, toks, scores = _scores_for_query_cached(query)

    # Build doc -> [(page_index, score)]
    per_doc = {}
    for i, pid in enumerate(page_ids):
        did, _ = _split_pid(pid)
        per_doc.setdefault(did, []).append((i, scores[i]))

    # Rank docs by best page
    doc_rank = sorted(per_doc.items(), key=lambda kv: max(x[1] for x in kv[1]), reverse=True)
    doc_rank = doc_rank[:max(1, docs_k)]

    # Page cap per doc
    per_doc_cap = env_int("RRR_PAGES_PER_DOC_CAP", pages_per_doc)
    out = []
    for did, entries in doc_rank:
        entries.sort(key=lambda x: x[1], reverse=True)
        take = min(pages_per_doc, per_doc_cap)
        for i, score in entries[:max(1, take)]:
            pid = page_ids[i]; _did, page = _split_pid(pid)
            txt_path = page_text_path(_did, page)
            snippet = ""
            if txt_path.exists():
                with open(txt_path, encoding="utf-8") as f:
                    snippet = f.read()
            out.append({"doc_id": _did, "page": page, "text": snippet, "bm25_score": float(score)})
    return out
