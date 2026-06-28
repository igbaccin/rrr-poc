import re


# Legacy canonical surface kept for backwards compatibility during the v10
# transition (postprocess accepts either surface and rewrites to display).
CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

# v10 display surface — two shapes the model produces interchangeably:
#   1. BARE  : "Author (Year, p.N)"          -- user's preferred display form
#   2. PAREN : "(Author Year, p.N)"          -- common academic shorthand
# Both must parse to the same canonical (doc_id, page). The final-assembly
# step rewrites shape 2 back to shape 1 so the user-facing output is uniform.
# v13.1.1: the particle alternation is case-insensitive so "Van Zanden" /
# "De La X" capitalised forms — which the model occasionally produces at
# sentence-start — match into the particle path and capture the full label.
# Without this, capital-V "Van" fell through to the single-token branch and
# the downstream display_lookup resolved against "van" only, producing a
# false E1 in the FIX-F detector. The lookup itself is already lowercased,
# so the case-insensitive widening here is non-destructive.
_DISPLAY_UNIT = (
    r"(?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
)
_DISPLAY_LABEL = (
    r"(?:" + _DISPLAY_UNIT +
    r"(?:\s+(?:and\s+" + _DISPLAY_UNIT + r"|et\s+al\.))?)"
)
DISPLAY_CITE_RE = re.compile(
    r"(?<!\w)(" + _DISPLAY_LABEL + r")\s+"
    r"\((\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)
# Paren-shape variant: outer parens around BOTH label and year. We accept the
# label-year separator as either a space or comma+space because the model
# uses both, and the year-page separator as either a comma or whitespace so
# the comma-less form `(Author Year p.N)` parses too (v10.3).
DISPLAY_PAREN_CITE_RE = re.compile(
    r"\((" + _DISPLAY_LABEL + r")(?:,?\s+)(\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)


def _doc_id_to_author_label(doc_id: str) -> str:
    """v10: derive the human-readable author label from a canonical doc_id.

    Conventions used by the corpus filenames (and therefore doc_ids):
      Author_Year                 -> "Author"
      AuthorEtAl_Year             -> "Author et al."
      Author1&Author2_Year        -> "Author1 and Author2"
      Author1&Author2&Author3_Year -> "Author1 et al."   (3+ defaults to et al.)
    The doc_id may carry a trailing letter (e.g. 1989a) which we strip for the
    year. Surnames containing camelCase particles (e.g. vanZanden) are split.
    """
    if not doc_id:
        return ""
    m = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(doc_id))
    if not m:
        return str(doc_id)
    name_part = m.group(1)
    year = m.group(2)

    def _normalise_surname(s: str) -> str:
        # Split lowercase particle prefixes ("vanZanden" -> "van Zanden").
        return re.sub(r"\b(van|von|de|del|der)([A-Z])", r"\1 \2", s)

    if "EtAl" in name_part:
        head = name_part.replace("EtAl", "")
        return f"{_normalise_surname(head)} et al. ({year})"
    if "&" in name_part:
        parts = [_normalise_surname(p) for p in name_part.split("&") if p]
        if len(parts) == 1:
            return f"{parts[0]} ({year})"
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]} ({year})"
        return f"{parts[0]} et al. ({year})"
    return f"{_normalise_surname(name_part)} ({year})"


def render_citation(doc_id, page) -> str:
    """v10: emit the new display surface 'Author (Year, p.N)'.

    Callers that need the old canonical surface for parsing legacy text should
    use render_citation_canonical(); render_citation() is the canonical
    user-facing renderer from v10 onward.
    """
    base = _doc_id_to_author_label(str(doc_id).strip())
    if not base.endswith(")"):
        return f"{base} (p.{int(page)})"
    # Inject ", p.N" before the closing paren of the author/year part.
    return base[:-1] + f", p.{int(page)})"


def render_citation_canonical(doc_id, page) -> str:
    """Legacy '(Doc_Year: p.N)' surface — kept for tests and migrators."""
    return f"({str(doc_id).strip()}: p.{int(page)})"


def _build_author_year_lookup(allowed_docs):
    """Build reverse lookup: (author, year) -> doc_id for academic citation
    matching. Shared by writer.py (final citation collection) and reasoner.py
    (fallback when runs/review_cited_docs.json is missing). v13: promoted to
    render.py to retire the byte-identical copies that previously lived in
    both modules.
    """
    author_year_to_docid = {}
    for did in allowed_docs:
        clean = did.replace("EtAl", "").replace("&", "")
        parts = clean.split("_")
        if len(parts) >= 2:
            author = parts[0].lower()
            year = parts[-1].rstrip('abcdefgh')
            author_year_to_docid[(author, year)] = did
            if "EtAl" in did:
                author_year_to_docid[(author + " et al", year)] = did
    return author_year_to_docid


def _collect_cited_docs(text: str, allowed_docs, author_year_to_docid):
    """Collect cited doc_ids from canonical, bare-canonical, and legacy
    author-year forms. v13: promoted from writer.py + reasoner.py. The writer
    also extends the display-form scan (DISPLAY_CITE_RE / DISPLAY_PAREN_CITE_RE);
    that branch lives in writer.py and adds to the set this function returns.
    """
    cited_docs = set()

    for m in CITE_RE.finditer(text or ""):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)

    for m in re.finditer(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)", text or ""):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)

    for m in re.finditer(r"\(([A-Za-z&]+(?:\s+et\s+al\.?)?)[,\s]+(\d{4})\)", text or ""):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)

    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)", text or ""):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)

    return cited_docs


def _build_display_lookup(allowed_doc_ids):
    """Map (lowercase author-label, year) -> canonical doc_id.

    Built per call site from the allowed doc_id set so we never resolve to a
    document outside the validated corpus. Ambiguous keys (two doc_ids that
    produce the identical surface form) are dropped from the lookup so the
    downstream parser can flag the ambiguity rather than picking arbitrarily.
    """
    lookup = {}
    collisions = set()
    for did in allowed_doc_ids or []:
        label = _doc_id_to_author_label(str(did))
        m = re.match(r"^(.*?)\s*\((\d{4})\)$", label)
        if not m:
            continue
        key = (m.group(1).strip().lower(), m.group(2))
        if key in lookup and lookup[key] != did:
            collisions.add(key)
        else:
            lookup[key] = did
    for k in collisions:
        lookup.pop(k, None)
    return lookup


def parse_citations(text: str):
    """Iterate citations in either the legacy or the v10 display surface.

    Yields {doc_id_or_label, year (optional), page, start, end, raw, surface}.
    Validators downstream do their own canonical lookup against the corpus.
    """
    for match in CITE_RE.finditer(text or ""):
        yield {
            "doc_id": match.group(1),
            "page": int(match.group(2)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
            "surface": "canonical",
        }
    for match in DISPLAY_CITE_RE.finditer(text or ""):
        yield {
            "doc_id": None,  # caller resolves via _build_display_lookup
            "label": match.group(1).strip(),
            "year": match.group(2),
            "page": int(match.group(3)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
            "surface": "display",
        }


def rewrite_canonical_to_display(text: str) -> tuple:
    """Replace every (Doc_Year: p.N) in the text with the v10 display form.

    Returns (text, count). Safe to run unconditionally — the regex only matches
    the canonical surface, so already-display citations are untouched.
    """
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return render_citation(m.group(1), int(m.group(2)))

    return CITE_RE.sub(repl, text or ""), count


# v11-A: detect narrative citations sitting in non-narrative positions and
# rewrite them to parenthetical form. The clear failure mode is a citation at
# the end of a sentence whose author label is NOT the grammatical subject —
# e.g. "...conducive to long-run economic growth North and Weingast (1989, p.4)."
# That reads as broken. Conservative rewriter: only fires when (a) the citation
# is followed immediately by a sentence-end punctuation, AND (b) the author
# label is not at the start of its clause. Mid-sentence narrative cites where
# the author is not the subject are left alone — too easy to false-positive.
_NARRATIVE_REWRITE_BACKSCAN = 320

def rewrite_misplaced_narrative_citations(text: str) -> tuple:
    """Find Author (Year, p.N) tucked at the end of a sentence whose subject is
    not the author, and rewrite to (Author Year, p.N). Returns (text, count)."""
    if not text:
        return text, 0
    rewrites = 0
    matches = list(DISPLAY_CITE_RE.finditer(text))
    if not matches:
        return text, 0
    out_parts = []
    last_end = 0
    for m in matches:
        label = m.group(1).strip()
        year = m.group(2)
        page = m.group(3)
        start, end = m.start(), m.end()
        # (a) Next non-whitespace char must be sentence-ending punct.
        tail = text[end:end + 6]
        tail_stripped = tail.lstrip()
        if not tail_stripped or tail_stripped[0] not in ".!?;":
            continue
        # (b) Author label must NOT be at the start of its clause.
        prefix = text[max(0, start - _NARRATIVE_REWRITE_BACKSCAN):start]
        cb = max(
            prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"),
            prefix.rfind(";"), prefix.rfind(":"), prefix.rfind("\n"),
            prefix.rfind("("),
        )
        before_label = prefix[cb + 1:].strip() if cb >= 0 else prefix.strip()
        if not before_label:
            # Author label is sentence-initial — narrative form is correct.
            continue
        # Skip leading attribution connectives ("As", "In", "Following", "After",
        # "Per", "According to") — these still mark the author as the speaker
        # even though the label is not at column 0.
        if re.match(
            r"^(as|in|following|after|per|according to)\b",
            before_label,
            re.IGNORECASE,
        ):
            continue
        out_parts.append(text[last_end:start])
        out_parts.append(f"({label} {year}, p.{page})")
        last_end = end
        rewrites += 1
    if rewrites == 0:
        return text, 0
    out_parts.append(text[last_end:])
    return "".join(out_parts), rewrites


# v11.1: collapse the nested-narrative-inside-paren multi-cite form that
# mistral-small:24b produces almost every time it puts two narrative cites
# inside an outer paren wrapper. Input: "(Author (Year, p.N); Author
# (Year, p.N))" -> "(Author Year, p.N; Author Year, p.N)". Conservative —
# only fires on the (outer-paren ... narrative ... ; ... narrative ...
# outer-paren) shape; single narratives wrapped in an extra paren are also
# collapsed because the same pattern catches them.
_NESTED_NARRATIVE_OUTER_RE = re.compile(
    r"\(\s*"
    r"(?:" + _DISPLAY_LABEL + r")\s+\(\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+\)"
    r"(?:\s*;\s*(?:" + _DISPLAY_LABEL + r")\s+\(\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+\))*"
    r"\s*\)"
)
_INNER_NARRATIVE_CITE_RE = re.compile(
    r"(" + _DISPLAY_LABEL + r")\s+\((\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)


def collapse_nested_narrative_multicite(text: str) -> tuple:
    """v11.1: rewrite '(Author (Year, p.N); Author (Year, p.N))' to
    '(Author Year, p.N; Author Year, p.N)'. Returns (text, count)."""
    if not text:
        return text, 0
    rewrites = 0

    def repl_outer(m):
        nonlocal rewrites
        inner = m.group(0)[1:-1]  # strip outer parens
        # Rewrite each inner narrative to the bare 'Author Year, p.N' form.
        new_inner, n_inner = _INNER_NARRATIVE_CITE_RE.subn(
            lambda im: f"{im.group(1).strip()} {im.group(2)}, p.{im.group(3)}",
            inner,
        )
        if n_inner == 0:
            return m.group(0)
        rewrites += 1
        return f"({new_inner.strip()})"

    new_text = _NESTED_NARRATIVE_OUTER_RE.sub(repl_outer, text)
    return new_text, rewrites


