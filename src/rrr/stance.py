import hashlib
import json
import os
import time
from rrr.paths import claim_cache_path, page_text_path, require_page_text_dir

_MODEL = os.environ.get("RRR_REASONER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))

_CLAIM_PROMPT_VERSION = "2026-06-29-v1-claim"


def _get_abstract(doc_id: str) -> str:
    require_page_text_dir()
    path = page_text_path(doc_id, 1)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return f.read()[:1500]
    return ""


def _get_conclusion(doc_id: str, max_chars: int = 1500) -> str:
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
    if highest <= 1:
        return ""
    path = page_text_path(doc_id, highest)
    if not path.is_file():
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text[-max_chars:] if len(text) > max_chars else text


def _claim_cache_path(paper_key: str, sig: str, corpus_fingerprint: str = None):
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
    abstract = _get_abstract(doc_id)
    conclusion = _get_conclusion(doc_id)
    if not abstract and not conclusion:
        return {"claim": "", "source": "no_text", "duration_s": 0.0}

    sig = _claim_sig(doc_id, abstract, conclusion, _MODEL)
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
                if isinstance(cached, dict) and str(cached.get("claim") or "").strip():
                    cached.setdefault("source", "cache")
                    cached.setdefault("duration_s", 0.0)
                    if metrics:
                        metrics.cache_event("paper_claim", "hits")
                    return cached
            except Exception:
                pass
    cache_path = primary_path

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
        raw = raw.strip("`'\"")
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
        if claim:
            try:
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                if metrics:
                    metrics.cache_event("paper_claim", "writes")
            except Exception:
                pass
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
