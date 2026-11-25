import os, re
from rapidfuzz import fuzz
from rrr.text import normalize_text, sentence_spans
from rrr.utils import env_int

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _query_list(claim: str, probes=None):
    queries = []
    for item in probes or []:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in queries:
            queries.append(text)
    fallback = re.sub(r"\s+", " ", str(claim or "").strip())
    if fallback and fallback not in queries:
        queries.append(fallback)
    return queries


def _score(sentence: str, claim: str, probes=None) -> float:
    queries = _query_list(claim, probes)
    if not queries:
        return 0.0
    return max(fuzz.token_set_ratio(sentence, q) for q in queries)

def _is_biblio(s: str) -> bool:
    s_lower = s.lower()

    if re.search(r'\d+\s*\(\d{4}\)\s*,?\s*\d+[-–]\d+', s):
        return True

    if re.search(r'\bpp?\.\s*\d+[-–]?\d*', s_lower):
        return True

    if re.search(r'\([A-Z][a-z]+:\s*[A-Z][a-z]+.*?,\s*\d{4}\)', s):
        return True

    if len(re.findall(r'\(\d{4}\)', s)) >= 3:
        return True

    if re.search(r'^\s*(references|bibliography|works cited)\s*$', s_lower):
        return True

    if len(re.findall(r'[A-Z][a-z]+,?\s+\(\d{4}\)', s)) >= 3:
        return True

    if re.search(r'\beds\.?\b', s_lower) and re.search(r'\d{4}|\d+[-–]\d+', s):
        return True

    return False


_SPACED_LETTERS_RE = re.compile(r"(?:\b[A-Za-z]\s+){4,}[A-Za-z]\b")
_RUNON_TOKEN_RE = re.compile(r"\b[A-Za-z]{26,}\b")

def _quote_corruption_signals(s: str) -> int:
    if not s:
        return 0
    signals = 0
    if _SPACED_LETTERS_RE.search(s):
        signals += 1
    if _RUNON_TOKEN_RE.search(s):
        signals += 1
    return signals


def select_sentences(page_text: str, claim: str, max_sentences: int = 6, min_chars: int = 40, probes=None, out_stats=None):
    all_spans = sentence_spans(page_text, min_chars=1)
    all_sentences = [s["text"].strip() for s in all_spans]
    spans = [s for s in all_spans if len(s["text"].strip()) >= min_chars]
    sentences = [s["text"].strip() for s in spans]
    if not sentences:
        sentences = [normalize_text(s).strip() for s in _SENT_SPLIT.split(page_text) if len(s.strip()) >= min_chars]
        all_sentences = sentences[:]
    if not sentences:
        return []

    sentences = [s for s in sentences if not _is_biblio(s)]
    if not sentences:
        return []

    quote_quality_min = env_int("RRR_WRITER_QUOTE_QUALITY_MIN", 1)
    if quote_quality_min > 0:
        sentences = [s for s in sentences if _quote_corruption_signals(s) < quote_quality_min]
        if not sentences:
            return []

    min_score = env_int("RRR_MIN_SENT_SCORE", 40)

    scored = []
    for s in sentences:
        sc = _score(s, claim, probes=probes)
        if sc >= min_score:
            scored.append((s, sc))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    diversity_weight = float(os.environ.get("RRR_SENT_DIVERSITY_WEIGHT", "0.15"))
    chosen = []
    seen_texts = []
    remaining = scored[:]
    while remaining and len(chosen) < max_sentences:
        best_idx = 0
        best_value = None
        for idx, (s, sc) in enumerate(remaining):
            similarity_penalty = max((fuzz.token_set_ratio(s, t) for t in seen_texts), default=0)
            value = sc - diversity_weight * similarity_penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        s, sc = remaining.pop(best_idx)
        if all(fuzz.token_set_ratio(s, t) < 92 for t in seen_texts):
            chosen.append((s, sc))
            seen_texts.append(s)
        if len(chosen) >= max_sentences:
            break

    n_context = env_int("RRR_EVIDENCE_CONTEXT_SENTENCES", 1)
    if n_context <= 0 or not all_sentences:
        return chosen

    pos_by_text = {}
    for i, s in enumerate(all_sentences):
        pos_by_text.setdefault(s, i)

    widened = []
    context_added = 0
    for s, sc in chosen:
        idx = pos_by_text.get(s)
        if idx is None:
            widened.append((s, sc))
            continue
        lo = max(0, idx - n_context)
        hi = min(len(all_sentences), idx + n_context + 1)
        window = all_sentences[lo:hi]
        added = (idx - lo) + (hi - 1 - idx)
        context_added += added
        widened.append((" ".join(window).strip(), sc))

    if isinstance(out_stats, dict):
        out_stats["context_sentences_added"] = context_added

    return widened
