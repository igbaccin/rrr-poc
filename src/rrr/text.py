import re


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# v8: query-time token RE preserves hyphens so multi-word concepts like
# "long-run", "rule-of-law", "factor-endowments" survive tokenization for retrieval.
QUERY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with",
}
# v8: query-time stoplist trimmed — keep "of","the","that","for" tokenizable
# so phrases like "rule of law", "theory of growth", "test for unit root" don't
# silently collapse to a unigram bag before BM25 sees them.
QUERY_STOPWORDS = STOPWORDS - {"of", "the", "that", "for"}


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str):
    tokens = []
    for tok in TOKEN_RE.findall(normalize_text(text).lower()):
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        tokens.append(_stem(tok))
    return tokens


def tokenize_query(text: str):
    """v8: query-time tokenizer that preserves hyphens and a smaller stoplist.

    Used by retrieve() to keep multi-word concepts intact. Doc-time tokenize()
    is unchanged so the existing BM25 index does not need rebuilding.
    Hyphenated tokens are also emitted as their components so they still match
    the doc-time tokens in the index.
    """
    tokens = []
    for tok in QUERY_TOKEN_RE.findall(normalize_text(text).lower()):
        if tok in QUERY_STOPWORDS:
            continue
        if tok.isdigit():
            continue
        if "-" in tok:
            for piece in tok.split("-"):
                if piece and piece not in QUERY_STOPWORDS and not piece.isdigit():
                    tokens.append(_stem(piece))
        else:
            tokens.append(_stem(tok))
    return tokens


def _stem(tok: str) -> str:
    for suffix in ("ization", "isation", "fulness", "iveness", "ments", "ment", "ing", "ies", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 3:
            if suffix == "ies":
                return tok[:-3] + "y"
            return tok[: -len(suffix)]
    return tok


def sentence_spans(text: str, min_chars: int = 20):
    normalized = normalize_text(text)
    spans = []
    pos = 0
    for part in SENTENCE_RE.split(normalized):
        sent = part.strip()
        if not sent:
            pos += len(part)
            continue
        start = normalized.find(sent, pos)
        if start < 0:
            start = pos
        end = start + len(sent)
        if len(sent) >= min_chars:
            spans.append({"start": start, "end": end, "text": sent})
        pos = end
    return spans


def page_sort_key(path_or_name: str):
    name = str(path_or_name)
    match = re.search(r"_page_(\d+)", name)
    if match:
        return int(match.group(1))
    return 0
