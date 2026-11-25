import re


CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

NAME_PARTICLES = (
    "van", "von", "de", "del", "della", "der", "den", "la", "le", "du", "di",
    "da", "dos", "das", "ten", "ter", "zu", "zur",
)
_PARTICLE_ALTERNATION = "|".join(sorted(NAME_PARTICLES, key=len, reverse=True))
_PARTICLE_CHAIN = r"(?:(?i:" + _PARTICLE_ALTERNATION + r")\s+){1,3}"
_DISPLAY_UNIT = (
    r"(?:" + _PARTICLE_CHAIN + r"[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
)
_DISPLAY_LABEL = (
    r"(?:" + _DISPLAY_UNIT +
    r"(?:\s+(?:and\s+" + _DISPLAY_UNIT + r"|et\s+al\.))?)"
)
DISPLAY_CITE_RE = re.compile(
    r"(?<!\w)(" + _DISPLAY_LABEL + r")\s+"
    r"\((\d{4}[a-z]?)(?:,\s*|\s+)p\.\s*(\d+)\)"
)
DISPLAY_PAREN_CITE_RE = re.compile(
    r"\((" + _DISPLAY_LABEL + r")(?:,?\s+)(\d{4}[a-z]?)(?:,\s*|\s+)p\.\s*(\d+)\)"
)

_GROUPED_PAREN_RE = re.compile(r"\((?P<body>[^();]*(?:;[^();]*)+)\)")
_GROUPED_SURNAME = (
    r"(?:(?:(?i:" + _PARTICLE_ALTERNATION + r")\s+){0,3}"
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
    r"(?P<year>\d{4}[a-z]?)(?:,\s*|\s+)p\.\s*(?P<page>\d+)"
)
_GROUPED_NARRATIVE_FIRST_RE = re.compile(
    r"(?P<year>\d{4}[a-z]?)(?:,\s*|\s+)p\.\s*(?P<page>\d+)"
)
_GROUPED_NARRATIVE_LABEL_RE = re.compile(
    r"(?<!\w)(?P<label>" + _GROUPED_DISPLAY_LABEL + r")\s*$"
)
_GROUPED_CITATION_SHAPED_RE = re.compile(
    r"(?:\bp{1,2}\.\s*\d+|\b\d{4}[a-z]?\b)", re.IGNORECASE
)


_METADATA_LABEL_LOOKUP: dict = {}

_RENDERED_CITATIONS: dict = {}

_LABEL_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4}[a-z]?)\)\s*$")


def _year_cell(value) -> str:
    match = re.fullmatch(r"\s*(\d{4})(?:\.0+)?([a-z]?)\s*", str(value if value is not None else ""))
    return (match.group(1) + match.group(2)) if match else ""


def _label_with_year(label: str, year: str) -> str:
    match = _LABEL_YEAR_RE.match(label or "")
    if not match or not year or match.group(2)[:4] != year[:4]:
        return label
    return f"{match.group(1)} ({year})"


def note_rendered_citation(doc_id, page) -> None:
    doc_id = str(doc_id or "").strip()
    match = _LABEL_YEAR_RE.match(_doc_id_to_author_label(doc_id))
    if not doc_id or not match:
        return
    try:
        key = (match.group(1).strip().lower(), match.group(2), int(page))
    except (TypeError, ValueError):
        return
    _RENDERED_CITATIONS.setdefault(key, set()).add(doc_id)


def set_metadata_labels(rows, *, clear: bool = True) -> int:
    global _METADATA_LABEL_LOOKUP, _RENDERED_CITATIONS
    if clear:
        _METADATA_LABEL_LOOKUP = {}
        _RENDERED_CITATIONS = {}
    count = 0
    derived = set()
    for row in rows or []:
        try:
            doc_id = str(row.get("doc_id", "") or "").strip()
        except AttributeError:
            continue
        if not doc_id:
            continue
        display_label = str(row.get("display_label", "") or "").strip()
        first_surname = str(row.get("first_author_surname", "") or "").strip()
        if first_surname.lower() == "nan":
            first_surname = ""
        if display_label.lower() == "nan":
            display_label = ""
        year = _year_cell(row.get("year", "")) or str(row.get("year", "") or "").strip()
        if not display_label:
            display_label = _label_with_year(
                _regex_doc_id_to_author_label(doc_id), _year_cell(year),
            )
            derived.add(doc_id)
        if not first_surname:
            first_surname = _regex_author_surnames_only(display_label).split(" and ")[0].split(",")[0].strip()
        _METADATA_LABEL_LOOKUP[doc_id] = {
            "display_label": display_label,
            "first_author_surname": first_surname,
            "year": year,
            "surnames": _regex_author_surnames_only(display_label),
        }
        count += 1

    by_label = {}
    for doc_id, entry in _METADATA_LABEL_LOOKUP.items():
        by_label.setdefault(entry["display_label"].lower(), []).append(doc_id)
    for doc_ids in by_label.values():
        if len(doc_ids) < 2:
            continue
        for doc_id in doc_ids:
            if doc_id not in derived:
                continue
            lettered = _regex_doc_id_to_author_label(doc_id, keep_letter=True)
            entry = _METADATA_LABEL_LOOKUP[doc_id]
            if lettered != entry["display_label"]:
                entry["display_label"] = lettered
                entry["surnames"] = _regex_author_surnames_only(lettered)
    return count


def _regex_author_surnames_only(label: str) -> str:
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


def _split_leading_particles(name: str) -> str:
    if not name or not name[0].islower():
        return name

    def split(rest: str, depth: int):
        if rest[:1].isupper():
            return [rest]
        if depth >= 3:
            return None
        for particle in sorted(NAME_PARTICLES, key=len, reverse=True):
            if rest.startswith(particle) and len(rest) > len(particle):
                tail = split(rest[len(particle):], depth + 1)
                if tail is not None:
                    return [particle] + tail
        return None

    parts = split(name, 0)
    if parts is None:
        return name[0].upper() + name[1:]
    return " ".join(parts)


def _regex_doc_id_to_author_label(doc_id: str, keep_letter: bool = False) -> str:
    if not doc_id:
        return ""
    m = re.match(r"^(.+?)_(\d{4})([a-z]?)$", str(doc_id))
    if not m:
        return str(doc_id)
    name_part = m.group(1)
    year = m.group(2) + (m.group(3) if keep_letter else "")

    def _normalise_surname(s: str) -> str:
        s = _split_leading_particles(s)
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
    note_rendered_citation(doc_id, page)
    if not base.endswith(")"):
        return f"{base} (p.{int(page)})"
    return base[:-1] + f", p.{int(page)})"


def render_citation_canonical(doc_id, page) -> str:
    return f"({str(doc_id).strip()}: p.{int(page)})"


def _build_author_year_lookup(allowed_docs):
    author_year_to_docid = {}

    def add(author, year_full, did, et_al):
        for year in dict.fromkeys((year_full, year_full.rstrip("abcdefghijklmnopqrstuvwxyz"))):
            author_year_to_docid[(author, year)] = did
            if et_al:
                author_year_to_docid[(author + " et al", year)] = did

    for did in allowed_docs:
        entry = _METADATA_LABEL_LOOKUP.get(str(did).strip())
        if entry and entry.get("first_author_surname") and entry.get("year"):
            author = entry["first_author_surname"].lower()
            surnames = entry.get("surnames", "")
            year_full = _year_cell(entry["year"]) or entry["year"]
            label_year = _LABEL_YEAR_RE.match(entry.get("display_label", "") or "")
            if (
                label_year
                and len(label_year.group(2)) == 5
                and label_year.group(2)[:4] == year_full[:4]
            ):
                year_full = label_year.group(2)
            add(
                author, year_full, did,
                surnames.endswith(" et al.") or " et al." in surnames,
            )
            continue
        clean = did.replace("EtAl", "").replace("&", "")
        parts = clean.split("_")
        if len(parts) >= 2:
            add(parts[0].lower(), parts[-1], did, "EtAl" in did)
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

    def _pageless(author, year):
        return (
            author_year_to_docid.get((author, year))
            or author_year_to_docid.get((author, year[:4]))
        )

    for m in re.finditer(r"\(([A-Za-z&]+(?:\s+et\s+al\.?)?)[,\s]+(\d{4}[a-z]?)\)", text or ""):
        author = m.group(1).lower().strip().rstrip('.')
        did = _pageless(author, m.group(2))
        if did:
            cited_docs.add(did)

    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4}[a-z]?)\)", text or ""):
        author = m.group(1).lower().strip().rstrip('.')
        did = _pageless(author, m.group(2))
        if did:
            cited_docs.add(did)

    return cited_docs


class DisplayLookup(dict):
    def __init__(self):
        super().__init__()
        self.ambiguous = {}
        self._lettered_bases = set()

    def __bool__(self):
        return bool(len(self) or self.ambiguous)

    def _direct(self, label: str, year: str):
        key = (label, year)
        doc_id = dict.get(self, key)
        if doc_id is not None:
            return doc_id, ()
        candidates = self.ambiguous.get(key)
        if candidates:
            return None, candidates
        if len(year) == 5 and (label, year[:4]) not in self._lettered_bases:
            base = (label, year[:4])
            doc_id = dict.get(self, base)
            if doc_id is not None:
                return doc_id, ()
            return None, self.ambiguous.get(base, ())
        return None, ()

    def resolve(self, label, year, page=None):
        label = str(label or "").strip().lower()
        year = str(year or "").strip()
        doc_id, candidates = self._direct(label, year)
        if doc_id is None and not candidates:
            words = label.split()
            while len(words) > 1 and words[0] in NAME_PARTICLES:
                words = words[1:]
                doc_id, candidates = self._direct(" ".join(words), year)
                if doc_id is not None or candidates:
                    break
        if doc_id is not None or not candidates:
            return doc_id, ()
        rendered = set()
        if page is not None:
            for candidate in candidates:
                entry = _LABEL_YEAR_RE.match(_doc_id_to_author_label(candidate))
                if not entry:
                    continue
                noted = _RENDERED_CITATIONS.get(
                    (entry.group(1).strip().lower(), entry.group(2), int(page)), ()
                )
                if candidate in noted:
                    rendered.add(candidate)
        pool = sorted(rendered) if rendered else list(candidates)
        return pool[0], tuple(candidates)

    def get(self, key, default=None):
        if isinstance(key, tuple) and len(key) == 2:
            doc_id, _ = self._direct(
                str(key[0] or "").strip().lower(), str(key[1] or "").strip(),
            )
            return default if doc_id is None else doc_id
        return dict.get(self, key, default)


def _build_display_lookup(allowed_doc_ids):
    lookup = DisplayLookup()
    exact = {}
    by_base = {}
    for did in allowed_doc_ids or []:
        label = _doc_id_to_author_label(str(did))
        m = _LABEL_YEAR_RE.match(label)
        if not m:
            continue
        name = m.group(1).strip().lower()
        year = m.group(2)
        exact.setdefault((name, year), [])
        if did not in exact[(name, year)]:
            exact[(name, year)].append(did)
        by_base.setdefault((name, year[:4]), [])
        if did not in by_base[(name, year[:4])]:
            by_base[(name, year[:4])].append(did)
        if len(year) == 5:
            lookup._lettered_bases.add((name, year[:4]))
    for key, doc_ids in exact.items():
        if len(doc_ids) == 1:
            lookup[key] = doc_ids[0]
        else:
            lookup.ambiguous[key] = tuple(sorted(doc_ids))
    for base, doc_ids in by_base.items():
        if base in exact:
            continue
        if len(doc_ids) == 1:
            lookup[base] = doc_ids[0]
        else:
            lookup.ambiguous[base] = tuple(sorted(doc_ids))
    return lookup


def citation_inside(citation, allowed_docs, allowed_pairs=None):
    documents = tuple(citation.get("candidates") or ())
    if not documents and citation.get("doc_id") is not None:
        documents = (citation["doc_id"],)
    canonical = {str(doc).lower(): doc for doc in allowed_docs or ()}
    pairs = set(allowed_pairs or ())
    page = citation.get("page")
    ordered = sorted(
        documents, key=lambda doc: doc != citation.get("doc_id")
    )
    for doc in ordered:
        match = canonical.get(str(doc).lower())
        if match is None:
            continue
        if pairs and (match, int(page)) not in pairs:
            continue
        return match
    return None


def parse_citations(text: str, display_lookup: dict = None):
    source = text or ""
    hits = []
    grouped_spans = []

    def _display_hit(label, year, page, start, end, raw, surface):
        doc_id, candidates = None, ()
        if display_lookup:
            if isinstance(display_lookup, DisplayLookup):
                doc_id, candidates = display_lookup.resolve(label, year, page)
            else:
                doc_id = display_lookup.get((label.lower(), year))
                if doc_id is None and len(year) == 5:
                    doc_id = display_lookup.get((label.lower(), year[:4]))
        hit = {
            "doc_id": doc_id,
            "label": label,
            "year": year,
            "page": page,
            "start": start,
            "end": end,
            "raw": raw,
            "surface": surface,
        }
        if candidates:
            hit["candidates"] = candidates
        return hit

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
                parsed.append(_display_hit(
                    display.group("label").strip(),
                    display.group("year"),
                    int(display.group("page")),
                    item_start, item_end, item_raw, "display_paren",
                ))
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
                parsed.append(_display_hit(
                    narrative_label_match.group("label").strip(),
                    narrative.group("year"),
                    int(narrative.group("page")),
                    item_start, item_end, item_raw, "display_narrative",
                ))
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
        hits.append(_display_hit(
            match.group(1).strip(), match.group(2), int(match.group(3)),
            match.start(), match.end(), match.group(0), "display_narrative",
        ))
    for match in DISPLAY_PAREN_CITE_RE.finditer(source):
        if _overlaps_group(match.start(), match.end()):
            continue
        hits.append(_display_hit(
            match.group(1).strip(), match.group(2), int(match.group(3)),
            match.start(), match.end(), match.group(0), "display_paren",
        ))

    hits.sort(key=lambda citation: (citation["start"], citation["end"]))
    yield from hits
