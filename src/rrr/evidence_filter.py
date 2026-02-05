import os, re
from rapidfuzz import fuzz

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _score(sentence: str, claim: str) -> float:
    return fuzz.token_set_ratio(sentence, claim)

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

def select_sentences(page_text: str, claim: str, max_sentences: int = 6, min_chars: int = 40):
    """
    Returns list of (sentence, score) tuples, sorted by score descending.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(page_text) if len(s.strip()) >= min_chars]
    if not sentences:
        return []

    # Filter out bibliographic text before scoring
    sentences = [s for s in sentences if not _is_biblio(s)]
    if not sentences:
        return []

    min_score = int(os.environ.get("RRR_MIN_SENT_SCORE", "40"))

    scored = []
    for s in sentences:
        sc = _score(s, claim)
        if sc >= min_score:
            scored.append((s, sc))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    chosen = []
    seen_texts = []
    for s, sc in scored:
        if all(fuzz.token_set_ratio(s, t) < 92 for t in seen_texts):
            chosen.append((s, sc))  # Return tuple with score
            seen_texts.append(s)
        if len(chosen) >= max_sentences:
            break

    return chosen
