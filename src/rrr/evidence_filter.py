import os, re
from rapidfuzz import fuzz
from rrr.text import normalize_text, sentence_spans

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
    """Reject sentences that look like bibliography/reference list entries."""
    s_lower = s.lower()
    
    # Volume(year), page-page pattern: "117 (2002), 1231-1294"
    if re.search(r'\d+\s*\(\d{4}\)\s*,?\s*\d+[-–]\d+', s):
        return True
    
    # Explicit page ranges: "pp. 45-67" or "p. 45"
    if re.search(r'\bpp?\.\s*\d+[-–]?\d*', s_lower):
        return True
    
    # Publisher pattern: "(City: Publisher, Year)" e.g., "(London: Heinemann, 1975)"
    if re.search(r'\([A-Z][a-z]+:\s*[A-Z][a-z]+.*?,\s*\d{4}\)', s):
        return True
    
    # Multiple years in parentheses (dense citation clusters)
    if len(re.findall(r'\(\d{4}\)', s)) >= 3:
        return True
    
    # Explicit reference section headers
    if re.search(r'^\s*(references|bibliography|works cited)\s*$', s_lower):
        return True
    
    # Dense author-year sequences: "Smith (2001), Jones (2003), Brown (2005)"
    if len(re.findall(r'[A-Z][a-z]+,?\s+\(\d{4}\)', s)) >= 3:
        return True
    
    # "eds." with year or page info
    if re.search(r'\beds\.?\b', s_lower) and re.search(r'\d{4}|\d+[-–]\d+', s):
        return True
    
    return False

def select_sentences(page_text: str, claim: str, max_sentences: int = 6, min_chars: int = 40, probes=None):
    """
    Returns list of (sentence, score) tuples, sorted by score descending.
    """
    spans = sentence_spans(page_text, min_chars=min_chars)
    sentences = [s["text"].strip() for s in spans]
    if not sentences:
        sentences = [normalize_text(s).strip() for s in _SENT_SPLIT.split(page_text) if len(s.strip()) >= min_chars]
    if not sentences:
        return []

    # Filter out bibliographic text before scoring
    sentences = [s for s in sentences if not _is_biblio(s)]
    if not sentences:
        return []

    min_score = int(os.environ.get("RRR_MIN_SENT_SCORE", "40"))

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

    return chosen
