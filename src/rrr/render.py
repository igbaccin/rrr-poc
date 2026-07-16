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

# A grouped parenthetical shares one pair of outer parentheses across two or
# more citation members, for example ``(Austin 2008, p.1; North 1989, p.2)``.
# The standalone patterns above deliberately retain their historical contract:
# they match a complete citation with its own parentheses.  Group parsing is a
# separate, conservative path so callers that use those public regexes directly
# keep seeing exactly the same matches.
_GROUPED_PAREN_RE = re.compile(r"\((?P<body>[^();]*(?:;[^();]*)+)\)")
_GROUPED_SURNAME = (
    r"(?:(?:(?i:van|von|de|del|der)\s+)?"
    r"[A-Z][A-Za-z'\u2019\-]+)"
)
_GROUPED_DISPLAY_LABEL = (
    _GROUPED_SURNAME
    + r"(?:\s+(?:(?:and|&)\s+" + _GROUPED_SURNAME
    + r"|et\s+al\.?))?"
)
_GROUPED_CANONICAL_ITEM_RE = re.compile(
    r"(?P<doc_id>[A-Za-z0-9_&.\-]+)\s*:\s*p\.\s*(?P<page>\d+)"
)
_GROUPED_DISPLAY_ITEM_RE = re.compile(
    r"(?P<label>" + _GROUPED_DISPLAY_LABEL + r")(?:,?\s+)"
    r"(?P<year>\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(?P<page>\d+)"
)
_GROUPED_NARRATIVE_FIRST_RE = re.compile(
    r"(?P<year>\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(?P<page>\d+)"
)
_GROUPED_NARRATIVE_LABEL_RE = re.compile(
    r"(?<!\w)(?P<label>" + _GROUPED_DISPLAY_LABEL + r")\s*$"
)
_GROUPED_CITATION_SHAPED_RE = re.compile(
    r"(?:\bp{1,2}\.\s*\d+|\b\d{4}[a-z]?\b)", re.IGNORECASE
)


# v15.9 (#1): metadata-driven author label lookup.
#
# Historically _doc_id_to_author_label parsed doc_id filenames via regex
# (Author_Year / AuthorEtAl_Year / Author1&Author2_Year). That convention
# only works when the user names PDFs to match it. To support arbitrary
# filenames, we now consult a module-level dict populated from
# metadata.csv at pipeline entry (reasoner._layered_t2_inner calls
# set_metadata_labels). If a doc_id is not in the dict, we fall back to
# the regex — preserving byte-identical behaviour on legacy corpora that
# don't have display_label / first_author_surname columns.
_METADATA_LABEL_LOOKUP: dict = {}


def set_metadata_labels(rows, *, clear: bool = True) -> int:
    """Populate the module-level lookup from metadata.csv rows.

    Each row can be a dict (from pandas.DataFrame.to_dict(orient='records'))
    or any mapping with 'doc_id', 'authors', 'year' at minimum.
    Optional columns override the derived values:
      - display_label: 'Acemoglu et al. (2001)' (used as-is)
      - first_author_surname: 'Acemoglu' (used as-is)

    When display_label / first_author_surname are missing, the seed values
    come from the regex-based _regex_doc_id_to_author_label / _regex_author_surnames_only
    so the pre-v15.9 output is preserved byte-for-byte on the 50-paper
    hand-curated corpus.

    Returns the number of entries loaded.
    """
    global _METADATA_LABEL_LOOKUP
    if clear:
        _METADATA_LABEL_LOOKUP = {}
    count = 0
    for row in rows or []:
        try:
            doc_id = str(row.get("doc_id", "") or "").strip()
        except AttributeError:
            continue
        if not doc_id:
            continue
        display_label = str(row.get("display_label", "") or "").strip()
        first_surname = str(row.get("first_author_surname", "") or "").strip()
        year = str(row.get("year", "") or "").strip()
        # Seed from regex when explicit columns are missing (backward compat)
        if not display_label:
            display_label = _regex_doc_id_to_author_label(doc_id)
        if not first_surname:
            first_surname = _regex_author_surnames_only(display_label).split(" and ")[0].split(",")[0].strip()
        _METADATA_LABEL_LOOKUP[doc_id] = {
            "display_label": display_label,
            "first_author_surname": first_surname,
            "year": year,
            "surnames": _regex_author_surnames_only(display_label),
        }
        count += 1
    return count


def _regex_author_surnames_only(label: str) -> str:
    """Strip trailing ' (Year)' from an 'Acemoglu et al. (2001)' style label."""
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


def _regex_doc_id_to_author_label(doc_id: str) -> str:
    """Legacy regex-based label derivation. Preserved for corpora that
    follow the Author_YYYY filename convention and don't yet have a
    metadata-driven lookup populated. See _doc_id_to_author_label below
    for the composed path."""
    if not doc_id:
        return ""
    m = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(doc_id))
    if not m:
        return str(doc_id)
    name_part = m.group(1)
    year = m.group(2)

    def _normalise_surname(s: str) -> str:
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


def _doc_id_to_author_label(doc_id: str) -> str:
    """v15.9 (#1): metadata lookup first, regex fallback.

    Conventions used by the corpus filenames (regex path):
      Author_Year                 -> "Author"
      AuthorEtAl_Year             -> "Author et al."
      Author1&Author2_Year        -> "Author1 and Author2"
      Author1&Author2&Author3_Year -> "Author1 et al."
    The doc_id may carry a trailing letter (e.g. 1989a) which we strip for the
    year. Surnames containing camelCase particles (e.g. vanZanden) are split.
    """
    if not doc_id:
        return ""
    entry = _METADATA_LABEL_LOOKUP.get(str(doc_id).strip())
    if entry and entry.get("display_label"):
        return entry["display_label"]
    return _regex_doc_id_to_author_label(doc_id)


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
    (fallback when runs/review_cited_docs.json is missing).

    v15.9 (#1): metadata lookup first (via _METADATA_LABEL_LOOKUP); regex
    parse of the doc_id string as fallback. This lets arbitrary-filename
    corpora work — as long as metadata.csv is populated, we don't need
    the doc_id string to encode the author.
    """
    author_year_to_docid = {}
    for did in allowed_docs:
        entry = _METADATA_LABEL_LOOKUP.get(str(did).strip())
        if entry and entry.get("first_author_surname") and entry.get("year"):
            author = entry["first_author_surname"].lower()
            year = entry["year"].rstrip("abcdefgh")
            author_year_to_docid[(author, year)] = did
            # 'Author et al.' variant if the label ends with 'et al.'
            surnames = entry.get("surnames", "")
            if surnames.endswith(" et al.") or " et al." in surnames:
                author_year_to_docid[(author + " et al", year)] = did
            continue
        # Regex fallback (legacy Author_YYYY convention)
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


def parse_citations(text: str, display_lookup: dict = None):
    """Iterate citations across all production surfaces in source order.

    Yields {doc_id, label, year, page, start, end, raw, surface}.

    v15.7: iterates DISPLAY_PAREN_CITE_RE in addition to CITE_RE and
    DISPLAY_CITE_RE. When display_lookup is provided (via
    _build_display_lookup(allowed_docs)), the display-form yields
    populate doc_id by looking up (label_lowercase, year). When no
    lookup is provided OR the (label, year) is unresolvable (collision
    or unknown author), doc_id is None and the caller must handle it.

    Validators that need (doc_id, page) tuples across all surfaces
    (coverage audit, redundancy drop, invalid-cite removal) should pass
    the display_lookup so display surfaces resolve consistently.

    A semicolon group is accepted when it contains at least one complete
    canonical or display citation and every remaining member is citation
    shaped.  Malformed members remain untouched and complete members are
    yielded independently.  A narrative first member is also accepted, as in
    ``Austin (2008, p.1; North 1989, p.2)``.  Each complete member receives its
    own exact source span.  Shared parentheses and semicolon delimiters remain
    outside those spans so provenance linkification can wrap members cleanly.
    """
    source = text or ""
    hits = []
    grouped_spans = []

    def _resolve_display(label, year):
        if not display_lookup:
            return None
        return display_lookup.get((label.lower(), year))

    # Parse grouped parentheticals first.  Splitting on semicolons is safe only
    # after the outer matcher has ruled out nested parentheses.  At least one
    # member must match an anchored citation pattern, and any malformed member
    # must still be citation shaped.  This keeps ordinary prose parentheticals
    # outside the citation parser while preserving valid neighbours of errors.
    for group_match in _GROUPED_PAREN_RE.finditer(source):
        body = group_match.group("body")
        body_start = group_match.start("body")
        pieces = body.split(";")
        if len(pieces) < 2:
            continue

        parsed = []
        cursor = 0
        narrative_label_match = None
        citation_shaped_group = True
        for index, piece in enumerate(pieces):
            left_trim = len(piece) - len(piece.lstrip())
            right_trim = len(piece.rstrip())
            item_start = body_start + cursor + left_trim
            item_end = body_start + cursor + right_trim
            item_raw = source[item_start:item_end]

            canonical = _GROUPED_CANONICAL_ITEM_RE.fullmatch(item_raw)
            if canonical:
                parsed.append({
                    "doc_id": canonical.group("doc_id"),
                    "label": None,
                    "year": None,
                    "page": int(canonical.group("page")),
                    "start": item_start,
                    "end": item_end,
                    "raw": item_raw,
                    "surface": "canonical",
                })
                cursor += len(piece) + 1
                continue

            display = _GROUPED_DISPLAY_ITEM_RE.fullmatch(item_raw)
            if display:
                label = display.group("label").strip()
                year = display.group("year")
                parsed.append({
                    "doc_id": _resolve_display(label, year),
                    "label": label,
                    "year": year,
                    "page": int(display.group("page")),
                    "start": item_start,
                    "end": item_end,
                    "raw": item_raw,
                    "surface": "display_paren",
                })
                cursor += len(piece) + 1
                continue

            narrative = (
                _GROUPED_NARRATIVE_FIRST_RE.fullmatch(item_raw)
                if index == 0 else None
            )
            if narrative:
                prefix = source[:group_match.start()]
                narrative_label_match = _GROUPED_NARRATIVE_LABEL_RE.search(prefix)
            if narrative and narrative_label_match:
                label = narrative_label_match.group("label").strip()
                year = narrative.group("year")
                parsed.append({
                    "doc_id": _resolve_display(label, year),
                    "label": label,
                    "year": year,
                    "page": int(narrative.group("page")),
                    "start": item_start,
                    "end": item_end,
                    "raw": item_raw,
                    "surface": "display_narrative",
                })
                cursor += len(piece) + 1
                continue

            # A malformed citation-shaped neighbour must not hide complete
            # members from provenance and validation.  Leave the malformed
            # text untouched and keep scanning.  Ordinary prose marks the
            # parenthetical as non-citation content, so reject that container
            # wholesale to retain the conservative classification boundary.
            if not item_raw or _GROUPED_CITATION_SHAPED_RE.search(item_raw):
                cursor += len(piece) + 1
                continue
            citation_shaped_group = False
            break

        if not citation_shaped_group or not parsed:
            continue
        group_start = (
            narrative_label_match.start("label")
            if narrative_label_match else group_match.start()
        )
        grouped_spans.append((group_start, group_match.end()))
        hits.extend(parsed)

    def _overlaps_group(start, end):
        return any(start < group_end and group_start < end
                   for group_start, group_end in grouped_spans)

    for match in CITE_RE.finditer(source):
        if _overlaps_group(match.start(), match.end()):
            continue
        hits.append({
            "doc_id": match.group(1),
            "label": None,
            "year": None,
            "page": int(match.group(2)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
            "surface": "canonical",
        })
    for match in DISPLAY_CITE_RE.finditer(source):
        if _overlaps_group(match.start(), match.end()):
            continue
        label = match.group(1).strip()
        year = match.group(2)
        hits.append({
            "doc_id": _resolve_display(label, year),
            "label": label,
            "year": year,
            "page": int(match.group(3)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
            "surface": "display_narrative",
        })
    for match in DISPLAY_PAREN_CITE_RE.finditer(source):
        if _overlaps_group(match.start(), match.end()):
            continue
        label = match.group(1).strip()
        year = match.group(2)
        hits.append({
            "doc_id": _resolve_display(label, year),
            "label": label,
            "year": year,
            "page": int(match.group(3)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
            "surface": "display_paren",
        })

    hits.sort(key=lambda citation: (citation["start"], citation["end"]))
    yield from hits


# Citation rendering now has one active path. The writer emits evidence IDs,
# and its context-aware renderer converts them directly to display citations.

