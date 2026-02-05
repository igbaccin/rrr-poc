import os, pickle, numpy as np, re
from functools import lru_cache

_tok = re.compile(r"[A-Za-z0-9]+")
def _tokenize(txt: str):
    return _tok.findall((txt or "").lower())

@lru_cache(maxsize=1)
def _load_bm25_and_ids():
    with open("indices/bm25.pkl","rb") as f:
        bm = pickle.load(f)
    page_ids = np.load("indices/page_ids.npy", allow_pickle=True).tolist()
    return bm, page_ids

def _split_pid(pid):
    if "_page_" in pid:
        did, page = pid.split("_page_")
        try: return did, int(page)
        except: return did, 1
    return pid, 1

# ---------- Phase B2: cache BM25 scoring per topic ----------
from functools import lru_cache

@lru_cache(maxsize=8)
def _scores_for_query_cached(query: str):
    """Compute BM25 scores once per topic, then reuse."""
    bm, page_ids = _load_bm25_and_ids()
    toks = _tokenize(query)
    scores = bm.get_scores(toks)
    return bm, page_ids, toks, scores


def retrieve(query: str, topk=20, doc_id=None):
    # Legacy entry point: keep working by delegating
    if doc_id:
        return retrieve_doc_pages(query, doc_id, pages_per_doc=topk)
    else:
        return retrieve_breadth(query, docs_k=max(10, topk//2), pages_per_doc=2)

def retrieve_doc_pages(query: str, doc_id: str, pages_per_doc=4):
    bm, page_ids, toks, scores = _scores_for_query_cached(query)
    pairs = [(i, scores[i]) for i, pid in enumerate(page_ids) if pid.startswith(f"{doc_id}_page_")]
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, _ in pairs[:max(1, pages_per_doc)]:
        pid = page_ids[i]; did, page = _split_pid(pid)
        txt_path = f"data/page_text/{did}_page_{page}.txt"
        snippet = ""
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                snippet = f.read()
        out.append({"doc_id": did, "page": page, "text": snippet})
    return out

def retrieve_breadth(query: str, docs_k=20, pages_per_doc=2):
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
    per_doc_cap = int(os.environ.get("RRR_PAGES_PER_DOC_CAP", str(pages_per_doc)))
    out = []
    for did, entries in doc_rank:
        entries.sort(key=lambda x: x[1], reverse=True)
        take = min(pages_per_doc, per_doc_cap)
        for i, _s in entries[:max(1, take)]:
            pid = page_ids[i]; _did, page = _split_pid(pid)
            txt_path = f"data/page_text/{_did}_page_{page}.txt"
            snippet = ""
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    snippet = f.read()
            out.append({"doc_id": _did, "page": page, "text": snippet})
    return out
