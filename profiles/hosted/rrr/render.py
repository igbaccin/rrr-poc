import re


CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

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
DISPLAY_PAREN_CITE_RE = re.compile(
    r"\((" + _DISPLAY_LABEL + r")(?:,?\s+)(\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)

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


_METADATA_LABEL_LOOKUP: dict = {}


def set_metadata_labels(rows, *, clear: bool = True) -> int:
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
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


def _regex_doc_id_to_author_label(doc_id: str) -> str:
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
    if not doc_id:
        return ""
    entry = _METADATA_LABEL_LOOKUP.get(str(doc_id).strip())
    if entry and entry.get("display_label"):
        return entry["display_label"]
    return _regex_doc_id_to_author_label(doc_id)


def render_citation(doc_id, page) -> str:
    base = _doc_id_to_author_label(str(doc_id).strip())
    if not base.endswith(")"):
        return f"{base} (p.{int(page)})"
    return base[:-1] + f", p.{int(page)})"


def render_citation_canonical(doc_id, page) -> str:
    return f"({str(doc_id).strip()}: p.{int(page)})"


def _build_author_year_lookup(allowed_docs):
    author_year_to_docid = {}
    for did in allowed_docs:
        entry = _METADATA_LABEL_LOOKUP.get(str(did).strip())
        if entry and entry.get("first_author_surname") and entry.get("year"):
            author = entry["first_author_surname"].lower()
            year = entry["year"].rstrip("abcdefgh")
            author_year_to_docid[(author, year)] = did
            surnames = entry.get("surnames", "")
            if surnames.endswith(" et al.") or " et al." in surnames:
                author_year_to_docid[(author + " et al", year)] = did
            continue
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
    cited_docs = set()

    display_lookup = _build_display_lookup(allowed_docs)
    for citation in parse_citations(text, display_lookup=display_lookup):
        did = citation.get("doc_id")
        if did in allowed_docs:
            cited_docs.add(did)

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
    source = text or ""
    hits = []
    grouped_spans = []

    def _resolve_display(label, year):
        if not display_lookup:
            return None
        return display_lookup.get((label.lower(), year))

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
