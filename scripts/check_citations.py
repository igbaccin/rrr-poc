#!/usr/bin/env python3
"""
check_citations.py — citation + quote integrity checker.

Parses a composed review (.md) and checks every citation surface against the
corpus (metadata.csv + per-doc page counts in data/*.json), then verifies any
direct-quoted prose against the actual page_text on disk.

v14.2 error taxonomy (expanded from the original E1/E2/E3):
    E1  Out-of-corpus citation — cited identity does not resolve to a
                                  document in metadata.csv
    E2  Invalid page           — cited page exceeds doc's content-page count
    E3  Format violation       — citation-like pattern that misses strict
                                  format. Now HARD-only: page-only (p.5),
                                  doc-without-page, multi-page citation,
                                  square-bracket dump. Author-year-text
                                  surfaces (e.g. "Hopkins (2009)" without a
                                  page) are no longer E3 — they appear in
                                  e3_soft for visibility but do NOT count
                                  toward the hard failure score.
    E4  Unverified quote       — quoted words do not occur on any retained
                                  page of the documents cited in their sentence.
    E5  Mis-attributed quote   — quoted words occur in a cited document, but
                                  on a retained page other than the page cited
                                  in their sentence.

CLI:
    python3 scripts/check_citations.py runs/review_composed.md
    python3 scripts/check_citations.py runs/review_composed.md --json results.json
    python3 scripts/check_citations.py runs/review_composed.md --json -   # stdout
    python3 scripts/check_citations.py runs/review_composed.md --no-quote-check

Import:
    from check_citations import check_file, check_review
"""

from functools import lru_cache
import os, re, json, glob, sys
import unicodedata

# v13: import CITE_RE from the canonical home so the script and the writer
# always agree on the citation surface. v13.1 (FIX-F): also import the
# display-form patterns and the author/year lookup builders so we can resolve
# user-facing surfaces ("Author (Year, p.N)" / "(Author Year, p.N)") against
# the corpus and bump E1 when the display label refers to a non-existent doc.
# Fall back to local copies only when the rrr package isn't importable (e.g.
# ad-hoc CLI use without PYTHONPATH set).
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from rrr.render import (  # type: ignore
        CITE_RE,
        DISPLAY_CITE_RE,
        DISPLAY_PAREN_CITE_RE,
        _build_author_year_lookup,
        _build_display_lookup,
        _doc_id_to_author_label,
    )
except Exception:
    CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")
    # Re-implement the display-form patterns inline so the checker keeps
    # working when run outside the package context. Keep these byte-identical
    # with render.py so the writer and checker agree on what counts as a
    # display citation.
    # v13.1.1: particle group case-insensitive so "Van Zanden" capital-V matches
    # the particle branch and the full label gets captured (mirrors render.py).
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

    def _doc_id_to_author_label(doc_id):  # type: ignore
        if not doc_id:
            return ""
        m = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(doc_id))
        if not m:
            return str(doc_id)
        name_part, year = m.group(1), m.group(2)

        def _ns(s):
            return re.sub(r"\b(van|von|de|del|der)([A-Z])", r"\1 \2", s)

        if "EtAl" in name_part:
            return f"{_ns(name_part.replace('EtAl', ''))} et al. ({year})"
        if "&" in name_part:
            parts = [_ns(p) for p in name_part.split("&") if p]
            if len(parts) == 1:
                return f"{parts[0]} ({year})"
            if len(parts) == 2:
                return f"{parts[0]} and {parts[1]} ({year})"
            return f"{parts[0]} et al. ({year})"
        return f"{_ns(name_part)} ({year})"

    def _build_author_year_lookup(allowed_docs):  # type: ignore
        out = {}
        for did in allowed_docs:
            clean = did.replace("EtAl", "").replace("&", "")
            parts = clean.split("_")
            if len(parts) >= 2:
                author = parts[0].lower()
                year = parts[-1].rstrip("abcdefgh")
                out[(author, year)] = did
                if "EtAl" in did:
                    out[(author + " et al", year)] = did
        return out

    def _build_display_lookup(allowed_doc_ids):  # type: ignore
        lookup, collisions = {}, set()
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

_AUTHOR_NAME_RE = r"(?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)[A-Z][A-Za-z&.\-]+)"

# Citation groups use semicolons to place two or more complete citation units
# inside one pair of parentheses.  The renderer's standalone patterns cannot
# see the inner units because only the whole group has parentheses, so the
# checker parses group containers before scanning standalone citations.
_GROUP_SURNAME_RE = (
    r"(?:(?:(?i:van|von|de|del|der)\s+)?"
    r"[A-Z][A-Za-z'\u2019\-]+)"
)
_GROUP_DISPLAY_LABEL_RE = (
    _GROUP_SURNAME_RE
    + r"(?:\s+(?:(?:and|&)\s+" + _GROUP_SURNAME_RE
    + r"|et\s+al\.?))?"
)
_GROUP_CANONICAL_UNIT_RE = re.compile(
    r"^\s*(?P<doc_id>[A-Za-z0-9_&.\-]+)\s*:\s*"
    r"p\.\s*(?P<page>\d+)\s*$"
)
_GROUP_DISPLAY_UNIT_RE = re.compile(
    r"^\s*(?P<label>" + _GROUP_DISPLAY_LABEL_RE + r")"
    r"(?:,?\s+)(?P<year>\d{4})[a-z]?"
    r"(?:,\s*|\s+)p\.\s*(?P<page>\d+)\s*$"
)
_GROUP_NARRATIVE_FIRST_RE = re.compile(
    r"^\s*(?P<year>\d{4})[a-z]?"
    r"(?:,\s*|\s+)p\.\s*(?P<page>\d+)\s*$"
)
_GROUP_PREFIX_RE = re.compile(
    r"(?<!\w)(?P<label>" + _GROUP_DISPLAY_LABEL_RE + r")\s*$"
)
_SEMICOLON_CONTAINER_RE = re.compile(
    r"\((?P<body>[^()]*;[^()]*)\)"
)
_PAREN_CONTAINER_RE = re.compile(r"\((?P<body>[^()]*)\)")
_RECOVERABLE_PAGE_SPEC = (
    r"\d+(?:\s*(?:[-\u2013\u2014,;]|\band\b)\s*"
    r"(?:p{1,2}\.\s*)?\d+)+"
)
_RECOVERABLE_CANONICAL_UNIT_RE = re.compile(
    r"^\s*(?P<doc_id>[A-Za-z0-9_&.\-]+)\s*:\s*"
    r"p\.\s*(?P<pages>" + _RECOVERABLE_PAGE_SPEC + r")\s*$",
    re.IGNORECASE,
)
_RECOVERABLE_DISPLAY_UNIT_RE = re.compile(
    r"^\s*(?P<label>" + _GROUP_DISPLAY_LABEL_RE + r")"
    r"(?:,?\s+)(?P<year>\d{4})[a-z]?"
    r"(?:,\s*|\s+)p\.\s*(?P<pages>"
    + _RECOVERABLE_PAGE_SPEC + r")\s*$"
)
_RECOVERABLE_NARRATIVE_UNIT_RE = re.compile(
    r"^\s*(?P<year>\d{4})[a-z]?"
    r"(?:,\s*|\s+)p\.\s*(?P<pages>"
    + _RECOVERABLE_PAGE_SPEC + r")\s*$"
)
_PAGE_ONLY_SPEC_RE = re.compile(
    r"^\s*p{1,2}\.\s*(?P<pages>\d+(?:\s*"
    r"(?:[-\u2013\u2014,;]|\band\b)\s*(?:p{1,2}\.\s*)?\d+)*)\s*$",
    re.IGNORECASE,
)


def _span_overlaps(start, end, spans):
    return any(start < span_end and end > span_start
               for span_start, span_end in spans)


def _expand_cited_pages(value):
    """Expand a malformed page list or short range into explicit pages."""
    cleaned = re.sub(r"p{1,2}\.\s*", "", str(value), flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:,|;|\band\b)\s*", cleaned)
    pages = []
    for part in parts:
        if not part:
            continue
        range_match = re.fullmatch(
            r"(\d+)\s*[-\u2013\u2014]\s*(\d+)", part)
        if range_match:
            first, last = map(int, range_match.groups())
            if last < first or last - first > 500:
                continue
            pages.extend(range(first, last + 1))
            continue
        if re.fullmatch(r"\d+", part):
            pages.append(int(part))
    return list(dict.fromkeys(pages))


def _recover_quote_only_citation_units(
        text, citation_units, group_errors, group_spans):
    """Recover named pages from malformed citations for quotation checks.

    These units never enter citation totals or document-breadth metrics. Their
    citation form remains an E3 event. Recovery only prevents that format
    event from causing a second, false quotation event when the quoted words
    occur on one of the pages the malformed citation explicitly names.
    """
    recovered = []

    def _append(template, pages, start, end, container_start, container_end,
                group_key, raw, surface):
        for page in pages:
            unit = dict(template)
            unit.update({
                "page": page,
                "start": start,
                "end": end,
                "container_start": container_start,
                "container_end": container_end,
                "group_key": group_key,
                "raw": raw,
                "surface": surface,
                "quote_only": True,
            })
            recovered.append(unit)

    def _explicit_template(value):
        canonical = _RECOVERABLE_CANONICAL_UNIT_RE.fullmatch(value)
        if canonical:
            return ({
                "kind": "canonical",
                "doc_id": canonical.group("doc_id"),
            }, _expand_cited_pages(canonical.group("pages")))
        display = _RECOVERABLE_DISPLAY_UNIT_RE.fullmatch(value)
        if display:
            return ({
                "kind": "display",
                "label": display.group("label"),
                "year": display.group("year"),
            }, _expand_cited_pages(display.group("pages")))
        return None, []

    # Standalone ranges and multi-page citations have one parenthetical
    # container. A narrative display citation takes its author label from the
    # text immediately before that container.
    for container in _PAREN_CONTAINER_RE.finditer(text):
        body = container.group("body")
        template, pages = _explicit_template(body)
        container_start = container.start()
        if template is None:
            narrative = _RECOVERABLE_NARRATIVE_UNIT_RE.fullmatch(body)
            if narrative:
                prefix_window_start = max(0, container.start() - 160)
                prefix = _GROUP_PREFIX_RE.search(
                    text[prefix_window_start:container.start()])
                if prefix:
                    template = {
                        "kind": "display",
                        "label": prefix.group("label"),
                        "year": narrative.group("year"),
                    }
                    pages = _expand_cited_pages(narrative.group("pages"))
                    container_start = (
                        prefix_window_start + prefix.start("label"))
        if template is None or len(pages) < 2:
            continue
        matching_group = next((
            (start, end) for start, end in group_spans
            if start <= container.start() and end >= container.end()
        ), None)
        if matching_group:
            group_key = ("group", *matching_group)
            effective_start, effective_end = matching_group
        else:
            effective_start, effective_end = container_start, container.end()
            group_key = (
                "quote_only", effective_start, effective_end)
        _append(
            template,
            pages,
            effective_start,
            effective_end,
            effective_start,
            effective_end,
            group_key,
            text[effective_start:effective_end],
            "malformed_multi_page",
        )

    # In a citation group, a page-only member conventionally carries forward
    # the document named by the preceding member. Preserve that association
    # solely for quotation verification while retaining the member's E3.
    for detail in group_errors:
        page_only = _PAGE_ONLY_SPEC_RE.fullmatch(detail["raw"])
        if not page_only:
            template, pages = _explicit_template(detail["raw"])
            if template is None or not pages:
                continue
        else:
            prior = [
                unit for unit in citation_units
                if unit["group_key"][0] == "group"
                and unit["container_start"] == detail["container_start"]
                and unit["container_end"] == detail["container_end"]
                and unit.get("member_index", -1) < detail["member_index"]
            ]
            if not prior:
                continue
            source = max(prior, key=lambda unit: unit["member_index"])
            template = {"kind": source["kind"]}
            if source["kind"] == "canonical":
                template["doc_id"] = source["doc_id"]
            else:
                template["label"] = source["label"]
                template["year"] = source["year"]
            pages = _expand_cited_pages(page_only.group("pages"))
        group_key = (
            "group", detail["container_start"], detail["container_end"])
        _append(
            template,
            pages,
            detail["start"],
            detail["end"],
            detail["container_start"],
            detail["container_end"],
            group_key,
            detail["raw"],
            "malformed_group_member",
        )

    seen = set()
    unique = []
    for unit in recovered:
        identity = (
            unit["kind"], unit.get("doc_id"), unit.get("label"),
            unit.get("year"), unit["page"], unit["group_key"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(unit)
    return unique


def _classify_malformed_group_unit(raw, narrative_label=None):
    """Return a hard-E3 reason for a citation-shaped malformed member."""
    value = str(raw or "").strip()
    if not value:
        return "group_empty_unit"
    expanded = f"{narrative_label} {value}" if narrative_label else value
    if re.fullmatch(r"pp?\.\s*\d+(?:\s*[-\u2013\u2014,]\s*(?:pp?\.\s*)?\d+)?",
                    value, re.IGNORECASE):
        return "group_page_only"
    if re.search(
        r"\bp\.\s*\d+\s*[-\u2013\u2014]\s*(?:p\.\s*)?\d+",
        expanded,
                 re.IGNORECASE):
        return "group_page_range"
    if re.search(r"\bp\.\s*\d+\s*,\s*(?:p\.\s*)?\d+", expanded,
                 re.IGNORECASE):
        return "group_multi_page"
    if re.fullmatch(
        r"\s*" + _GROUP_DISPLAY_LABEL_RE
        + r"(?:,?\s+)\d{4}[a-z]?\s*",
        expanded,
    ):
        return "group_author_year_without_page"
    if re.fullmatch(r"[A-Za-z0-9_&.\-]+_\d{4}[a-z]?", expanded):
        return "group_doc_without_page"
    if re.search(r"\bp{1,2}\.\s*\d+", expanded, re.IGNORECASE):
        return "group_malformed_unit"
    return None


def _parse_citation_units(text):
    """Parse grouped and standalone citation units without resolving docs.

    Every syntactically complete member becomes one unit.  Group containers
    are discovered first so their members cannot be counted again by the
    standalone patterns.  The returned group errors are already one-per-unit
    hard E3 events.
    """
    units = []
    group_errors = []
    group_spans = []

    for container in _SEMICOLON_CONTAINER_RE.finditer(text):
        body_start = container.start("body")
        fragments = []
        relative_start = 0
        for value in container.group("body").split(";"):
            fragments.append((value, relative_start))
            relative_start += len(value) + 1
        if len(fragments) < 2:
            continue

        prefix_window_start = max(0, container.start() - 160)
        prefix_match = _GROUP_PREFIX_RE.search(
            text[prefix_window_start:container.start()])
        narrative_label = None
        narrative_start = None
        if prefix_match:
            first_value = fragments[0][0]
            if (_GROUP_NARRATIVE_FIRST_RE.fullmatch(first_value)
                    or re.fullmatch(r"\s*\d{4}[a-z]?\s*", first_value)):
                narrative_label = prefix_match.group("label")
                narrative_start = prefix_window_start + prefix_match.start("label")

        parsed_members = []
        malformed_members = []
        all_citation_shaped = True
        for member_index, (raw, fragment_start) in enumerate(fragments):
            trimmed_left = len(raw) - len(raw.lstrip())
            trimmed_right = len(raw.rstrip())
            member_start = body_start + fragment_start + trimmed_left
            member_end = body_start + fragment_start + trimmed_right
            parsed = None

            canonical = _GROUP_CANONICAL_UNIT_RE.fullmatch(raw)
            if canonical:
                parsed = {
                    "kind": "canonical",
                    "surface": "canonical_group",
                    "doc_id": canonical.group("doc_id"),
                    "page": int(canonical.group("page")),
                }
            else:
                display = _GROUP_DISPLAY_UNIT_RE.fullmatch(raw)
                if display:
                    parsed = {
                        "kind": "display",
                        "surface": "display_group",
                        "label": display.group("label"),
                        "year": display.group("year"),
                        "page": int(display.group("page")),
                    }
                elif member_index == 0 and narrative_label:
                    narrative = _GROUP_NARRATIVE_FIRST_RE.fullmatch(raw)
                    if narrative:
                        parsed = {
                            "kind": "display",
                            "surface": "display_narrative_group",
                            "label": narrative_label,
                            "year": narrative.group("year"),
                            "page": int(narrative.group("page")),
                        }
                        member_start = narrative_start

            if parsed is not None:
                parsed.update({
                    "start": member_start,
                    "end": member_end,
                    "raw": text[member_start:member_end],
                    "member_index": member_index,
                })
                parsed_members.append(parsed)
                continue

            reason = _classify_malformed_group_unit(
                raw,
                narrative_label=(
                    narrative_label if member_index == 0 else None),
            )
            if reason is None:
                all_citation_shaped = False
                break
            malformed_members.append({
                "reason": reason,
                "raw": raw.strip(),
                "start": member_start,
                "end": member_end,
                "member_index": member_index,
            })

        if not all_citation_shaped:
            continue
        # A semicolon parenthesis containing only unpaged author-year prose
        # remains outside grouped-citation parsing.  At least one complete
        # citation establishes the container; malformed neighbours then
        # receive unit-level E3 treatment.
        if not parsed_members:
            continue

        container_start = (
            narrative_start if narrative_start is not None
            else container.start()
        )
        container_end = container.end()
        group_key = ("group", container_start, container_end)
        group_spans.append((container_start, container_end))
        for parsed in parsed_members:
            parsed.update({
                "container_start": container_start,
                "container_end": container_end,
                "group_key": group_key,
            })
            units.append(parsed)
        for malformed in malformed_members:
            malformed.update({
                "container_start": container_start,
                "container_end": container_end,
            })
            group_errors.append(malformed)

    seen = {
        (unit["kind"], unit["start"], unit["end"])
        for unit in units
    }

    def _add_standalone(match, kind, surface):
        start, end = match.span()
        if _span_overlaps(start, end, group_spans):
            return
        key = (kind, start, end)
        if key in seen:
            return
        unit = {
            "kind": kind,
            "surface": surface,
            "start": start,
            "end": end,
            "raw": match.group(0),
            "container_start": start,
            "container_end": end,
            "group_key": ("standalone", start, end),
            "member_index": 0,
            "page": int(match.group(2 if kind == "canonical" else 3)),
        }
        if kind == "canonical":
            unit["doc_id"] = match.group(1)
        else:
            unit["label"] = match.group(1)
            unit["year"] = match.group(2)
        units.append(unit)
        seen.add(key)

    for match in CITE_RE.finditer(text):
        _add_standalone(match, "canonical", "canonical")
    for match in DISPLAY_CITE_RE.finditer(text):
        _add_standalone(match, "display", "display_bare")
    for match in DISPLAY_PAREN_CITE_RE.finditer(text):
        _add_standalone(match, "display", "display_paren")

    units.sort(key=lambda unit: (
        unit["start"], unit["end"], unit.get("member_index", 0)))
    group_spans.sort()
    return units, group_errors, group_spans

# v13.1 (FIX-F): loose display form the model emits without a page when it is
# uncertain — "Author & Author Year" or "Author and Author Year". These are
# NOT matched by DISPLAY_CITE_RE/DISPLAY_PAREN_CITE_RE (which require a page),
# but they still assert a (author, year) tuple that must resolve to a corpus
# document. We capture the surface, the first-author surname (everything left
# of the year), and the year, then resolve the same way as the paged forms.
_LOOSE_AUTHOR_YEAR_RE = re.compile(
    r"(?<!\w)("
    r"(?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"(?:\s*(?:&|and)\s*(?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+)?"
    r"(?:\s+et\s+al\.?)?"
    r")\s+(\d{4})[a-z]?(?!\s*[:,]?\s*p\.)"
)

# E3 (HARD) — citation-shaped strings that are clearly wrong:
#   - canonical-form cite missing a page
#   - bare page-only cite, e.g. (p.5)
#   - page range or multi-page packed citation, e.g. (Foo: p.5, p.7)
#   - square-bracket evidence-id dump that survived the renderer
_E3_HARD_PATTERNS = [
    ("doc_without_page", re.compile(r"\((?=[^)]*[A-Za-z0-9_&.\-]+_\d{4})(?![^)]*:\s*p\.)[^)]*\)")),
    ("page_only", re.compile(r"\((?:pp?\.)\s*\d+(?:\s*(?:,|-|and)\s*(?:pp?\.)?\s*\d+)*\)", re.IGNORECASE)),
    ("multi_page_citation", re.compile(r"\([A-Za-z0-9_&.\-]+:\s*p\.\d+\s*,\s*p\.\d+[^)]*\)")),
    ("square_bracket_dump", re.compile(
        r"^\s*\[[^\]\n]*[A-Za-z0-9_&.\-]+:\s*p\.\d+[^\]\n]*(?:;\s*[A-Za-z0-9_&.\-]+:\s*p\.\d+|,\s*p\.\d+)[^\]\n]*\]\s*$",
        re.MULTILINE,
    )),
]

# E3 (SOFT) — display-form narrative mentions WITHOUT a page number. These are
# legitimate prose surfaces in many cases (e.g. "Hopkins (2009) argues that..."
# followed by a fully-paged citation later in the paragraph), and the writer
# system prompt does already require pages on every cite. We count them for
# visibility (`e3_soft` in the result) but do NOT count them in `e3` so the
# v14.1 false-positive over-counting against well-formed reviews is gone.
_E3_SOFT_PATTERNS = [
    ("author_year_text", re.compile(rf"\b{_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?\s*\(\d{{4}}\)")),
    ("author_year_possessive", re.compile(rf"\b{_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?'s\s*\(\d{{4}}\)")),
    ("author_year_parenthetical", re.compile(rf"\({_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?,\s*\d{{4}}\)")),
]

# E4 / E5 helpers — quoted spans of >=20 chars; nearby canonical (doc: p.N)
# cite within +/- 200 chars.
# v16.8 precision fix: (1) match smart quotes as well as straight ones — LLM
# essays routinely use “ ” for real quotations, which the old straight-only
# regex missed while stray straight " marks paired across prose; (2) cap the
# span length so a mispaired quote can't swallow half a paragraph. A real
# inline quotation is short; a 500-char "quote" is a pairing error.
_QUOTED_SPAN_RE = re.compile(r'["“]([^"”\n]{20,500})["”]')
# v16.8: a clean verbatim quotation never EMBEDS an in-text citation or an
# evidence-id bracket. When the span matcher pairs the wrong quote marks it
# swallows prose + citations; such spans are not checkable quotations and
# would inflate E4/E5 with false quotation errors. Used to reject them.
_SPAN_HAS_CITATION_RE = re.compile(
    # v16.9: page RANGES (`p.1-2`, en-dash too) count as citations — models
    # write them, and a range defeated the v16.8 pattern, letting a mispaired
    # span containing "(AcemogluEtAl_2002: p.1-2)" through as a "quote".
    r"\([A-Za-z0-9_&.\-]+:\s*p\.\s*\d+(?:\s*[-–]\s*\d+)?\)"        # canonical (doc: p.N[-M])
    r"|\([^()]*\b\d{4}[a-z]?\s*,\s*p\.\s*\d+(?:\s*[-–]\s*\d+)?\)"  # display (Author Year, p.N[-M])
    r"|\([^()\n]*(?:\b\d{4}[a-z]?\b|\bp\.\s*\d+)[^()\n]*;"
    r"[^()\n]*(?:\b\d{4}[a-z]?\b|\bp\.\s*\d+)[^()\n]*\)"           # grouped citation container
    r"|\[E\d{3,5}\]"                                               # unrendered evidence id
)
_NEARBY_CANONICAL_CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")
# Display-form cite for quote attribution lookup when the canonical form is
# absent (e.g. when checking the FINAL user-facing review where cites have
# already been display-rewritten). We re-use the existing DISPLAY_CITE_RE /
# DISPLAY_PAREN_CITE_RE patterns when checking display-form reviews.
_QUOTE_WINDOW_CHARS = 200
_SENTENCE_TERMINATOR_RE = re.compile(
    r'(?P<terminal>(?:(?<!\.)\.(?!\.)|[!?])'
    r'(?:["\u201d\u2019\'\)\]]*))'
    r'(?P<spacing>[ \t\r\n]+)'
)
_PARAGRAPH_BREAK_RE = re.compile(r"\r?\n[ \t]*\r?\n+")
_PERIOD_ABBREVIATIONS = {
    "al", "cf", "ch", "chap", "dr", "ed", "eds", "eq", "etc",
    "fig", "figs", "jr", "mr", "mrs", "ms", "no", "nos", "p",
    "pp", "prof", "rev", "sec", "sr", "st", "trans", "vol", "vols",
    "vs",
}
_ALWAYS_CONTINUING_ABBREVIATIONS = {
    "ch", "chap", "dr", "eq", "fig", "figs", "jr", "mr", "mrs", "ms",
    "no", "nos", "p", "pp", "prof", "sec", "sr", "st", "vol", "vols",
}


def _abbreviation_has_continuation(text: str, dot_index: int) -> bool:
    """Distinguish an internal abbreviation from one ending a sentence."""
    tail = text[dot_index + 1:]
    next_char = re.search(r"\S", tail)
    if not next_char:
        return False
    char = next_char.group(0)
    return char.islower() or char.isdigit() or char in "(["


def _period_ends_abbreviation(text: str, dot_index: int) -> bool:
    """Return True for common periods that do not close a sentence."""
    prefix = text[:dot_index + 1]
    if re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix):
        return _abbreviation_has_continuation(text, dot_index)
    if re.search(r"\b[A-Z]\.$", prefix):
        return True
    token = re.search(r"([A-Za-z]+)\.$", prefix)
    if not token:
        return False
    value = token.group(1).lower()
    if value in _ALWAYS_CONTINUING_ABBREVIATIONS:
        return True
    return (
        value in _PERIOD_ABBREVIATIONS
        and _abbreviation_has_continuation(text, dot_index)
    )


def _sentence_breaks(text: str):
    """Yield ``(sentence_end, next_sentence_start)`` boundary pairs."""
    boundaries = {
        (match.start(), match.end())
        for match in _PARAGRAPH_BREAK_RE.finditer(text)
    }
    for match in _SENTENCE_TERMINATOR_RE.finditer(text):
        terminal_start = match.start("terminal")
        if text[terminal_start] == "." and _period_ends_abbreviation(
                text, terminal_start):
            continue
        boundaries.add((match.end("terminal"), match.end()))
    yield from sorted(boundaries)


def _sentence_bounds(text: str, start: int, end: int, boundaries=None):
    """Return the sentence-like span governing ``text[start:end]``.

    Terminal punctuation followed by whitespace closes a sentence regardless
    of how the next sentence begins. Common abbreviations and initials are
    excluded, and blank lines remain hard boundaries.
    """
    left = 0
    right = len(text)
    if boundaries is None:
        boundaries = _sentence_breaks(text)
    for sentence_end, next_sentence_start in boundaries:
        if next_sentence_start <= start:
            left = max(left, next_sentence_start)
            continue
        if sentence_end >= end:
            right = sentence_end
            break
    return left, right


def _normalise_for_quote_match(s: str) -> str:
    """Normalise for verbatim-quote substring matching.

    Normalization is symmetric across quoted words and page text. It handles
    OCR hyphenation, Unicode ligatures and dashes, soft hyphens, and the narrow
    currency-marker substitution observed in the retained corpus. Ellipses and
    editorial brackets are evaluated separately as ordered quotation gaps.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    s = "".join(
        "-" if unicodedata.category(char) == "Pd" else char
        for char in s
    )
    s = s.lower()
    s = re.sub(r"[£ł?](?=\d)", "", s)
    s = re.sub(r"[\"'`“”‘’«»]", "", s)
    s = s.replace("…", " ")
    s = re.sub(r"\.{2,}", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"([a-z])-([a-z])", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalise_quote_needle(s: str) -> str:
    """Normalize quoted words and drop punctuation attached to quote marks."""
    return _normalise_for_quote_match(s).strip(" ,.;:!?")


_QUOTE_GAP_RE = re.compile(r"(?:\.{2,}|\u2026|\[[^\[\]]*\])")


def _quote_matches_normalised_page(quote: str, page_text: str) -> bool:
    """Match a quote, treating ellipses and editorial brackets as gaps."""
    needle = _normalise_quote_needle(quote)
    if needle and needle in page_text:
        return True
    if not _QUOTE_GAP_RE.search(quote):
        return False

    fragments = [
        _normalise_quote_needle(fragment)
        for fragment in _QUOTE_GAP_RE.split(quote)
    ]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return False

    cursor = 0
    for fragment in fragments:
        position = page_text.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment)
    return True


@lru_cache(maxsize=4096)
def _read_page_text(path: str):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


@lru_cache(maxsize=512)
def _page_text_paths(doc_id: str, page_text_dir: str):
    return tuple(sorted(
        glob.glob(os.path.join(page_text_dir, f"{doc_id}_page_*.txt"))
    ))


@lru_cache(maxsize=4096)
def _normalised_page_text(path: str):
    return _normalise_for_quote_match(_read_page_text(path))


def _load_page_text(doc_id: str, page: int, page_text_dir: str):
    """Read data/page_text/{doc_id}_page_{N}.txt; return None on miss."""
    try:
        p = os.path.join(page_text_dir, f"{doc_id}_page_{int(page)}.txt")
        if os.path.isfile(p):
            return _read_page_text(p)
    except Exception:
        pass
    return None


def _find_other_pages_with_quote(doc_id: str, quote: str, page_text_dir: str, exclude_page: int):
    """E5 detection: scan ALL pages of a doc for the normalised quote span.
    Returns the page number(s) where the quote DOES appear (excluding the
    cited page). An empty list means the wording remains unverified within
    the cited document, so it is not a page-misattribution case."""
    if not os.path.isdir(page_text_dir):
        return []
    if not _normalise_quote_needle(quote):
        return []
    matches = []
    for path in _page_text_paths(doc_id, page_text_dir):
        fname = os.path.basename(path)
        m = re.match(rf"^{re.escape(doc_id)}_page_(\d+)\.txt$", fname)
        if not m:
            continue
        page = int(m.group(1))
        if page == exclude_page:
            continue
        try:
            if _quote_matches_normalised_page(
                    quote, _normalised_page_text(path)):
                matches.append(page)
        except Exception:
            continue
    return sorted(matches)


def _load_valid_docs(metadata_path):
    """Load set of valid doc_ids from metadata CSV (stdlib csv — no pandas
    dependency so the checker works in minimal environments)."""
    if not os.path.isfile(metadata_path):
        return set()
    import csv as _csv
    out = set()
    with open(metadata_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            did = row.get("doc_id")
            if did:
                out.add(str(did))
    return out


def _load_citation_metadata(metadata_path):
    """Load the citation fields needed to resolve display labels.

    The checker normally works with corpus doc_ids that encode author and
    year.  Newer corpora may use arbitrary doc_ids and carry the display
    label in metadata instead, so resolution must consult both sources.
    """
    if not os.path.isfile(metadata_path):
        return {}
    import csv as _csv
    out = {}
    with open(metadata_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            did = str(row.get("doc_id", "") or "").strip()
            if not did:
                continue
            out[did] = {
                "display_label": str(
                    row.get("display_label", "") or "").strip(),
                "first_author_surname": str(
                    row.get("first_author_surname", "") or "").strip(),
                "authors": str(row.get("authors", "") or "").strip(),
                "authors_short": str(
                    row.get("authors_short", "") or "").strip(),
                "year": str(row.get("year", "") or "").strip(),
            }
    return out


def _load_doc_max_pages(data_dir):
    """Load max content-page count per doc_id from preprocessing metadata."""
    doc_max = {}
    for jpath in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            with open(jpath, encoding="utf-8") as f:
                meta = json.load(f)
            did = meta.get("doc_id", "")
            pc = meta.get("page_count", 0)
            if did and pc:
                doc_max[did] = pc
        except Exception:
            continue

    # Fallback: count page_text files
    if not doc_max:
        ptdir = os.path.join(data_dir, "page_text")
        if os.path.isdir(ptdir):
            seen = {}
            for fn in os.listdir(ptdir):
                if "_page_" in fn and fn.endswith(".txt"):
                    did = fn.rsplit("_page_", 1)[0]
                    seen[did] = seen.get(did, 0) + 1
            doc_max = seen

    return doc_max


def check_review(text, metadata_path="metadata.csv", data_dir="data", quote_check=True):
    """
    Check a review text string for E1/E2/E3/E4/E5 errors.

    Returns dict with:
      n_citations, e1, e2, e3, e3_soft, e4, e5,
      docs_cited, n_docs_cited, word_count,
      e1_details, e2_details, e3_details, e3_soft_details,
      e4_details, e5_details, quotes_checked, quotes_verified.

    `quote_check` toggles the E4/E5 scan (cheap; off only for backwards
    compatibility with callers that don't want disk reads of page_text).
    """
    citation_metadata = _load_citation_metadata(metadata_path)
    valid_docs = set(citation_metadata)
    doc_max = _load_doc_max_pages(data_dir)
    page_text_dir = os.path.join(data_dir, "page_text")

    # v13.1 (FIX-F): build the author/year lookups from the corpus so we can
    # resolve display-form citations the writer emits ("Author (Year, p.N)"
    # and "(Author Year, p.N)") and bump E1 when the (author, year) tuple
    # doesn't match any corpus document. Without this layer, the strict
    # CITE_RE check above misses display-form fabrications entirely — the
    # v13 smoke surfaced "Dalrymple-Smith & Frankema 2017" with e1=0.
    ignored_name_tokens = {
        "and", "et", "al", "van", "von", "de", "del", "der",
    }

    def _normalise_year(value):
        match = re.search(r"\b(\d{4})", str(value or ""))
        return match.group(1) if match else ""

    def _normalise_label(value):
        return str(value or "").strip().lower().rstrip(".")

    def _split_display_label(value, fallback_year=""):
        label = str(value or "").strip()
        match = re.match(r"^(.*?)\s*\((\d{4})[a-z]?\)\s*$", label)
        if match:
            return match.group(1).strip(), match.group(2)
        return label, _normalise_year(fallback_year)

    def _signature_tokens(value, *, doc_id_surface=False):
        tokens = set()
        for raw in re.findall(r"[A-Za-z\-]+", str(value or "")):
            lowered = raw.lower()
            if lowered in ignored_name_tokens:
                continue
            if doc_id_surface:
                particle = re.match(
                    r"^(van|von|de|del|der)([A-Z].+)$", raw)
                if particle:
                    lowered = particle.group(2).lower()
            tokens.add(lowered)
        return frozenset(tokens)

    # Build a collision-safe exact-label index from metadata when present,
    # with the established doc_id renderer as the fallback. Keep full
    # co-author signatures from both representations because arbitrary
    # metadata doc_ids cannot be decoded for author names, while legacy
    # corpora may omit display_label from metadata.
    display_lookup = {}
    display_collisions = set()
    citation_labels = {}
    full_signatures = {}
    for _did in valid_docs:
        _row = citation_metadata.get(_did, {})
        _metadata_label = _row.get("display_label", "")
        _metadata_authors = (
            _row.get("authors_short", "") or _row.get("authors", ""))
        _derived_label = _doc_id_to_author_label(str(_did))
        _label = _metadata_label or _derived_label
        _author_label, _year = _split_display_label(
            _label, _row.get("year", ""))
        if not _year:
            _doc_match = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(_did))
            if _doc_match:
                _year = _doc_match.group(2)
        citation_labels[_did] = (_author_label, _year)

        if _author_label and _year:
            _key = (_normalise_label(_author_label), _year)
            if _key in display_lookup and display_lookup[_key] != _did:
                display_collisions.add(_key)
            else:
                display_lookup[_key] = _did

        _signatures = set()
        if re.search(r"(?:\s+(?:and|&)\s+|\s*;\s*)",
                     _metadata_authors, re.IGNORECASE):
            _signature = _signature_tokens(_metadata_authors)
            if len(_signature) > 1:
                _signatures.add(_signature)
        if (
            re.search(r"\s+(?:and|&)\s+", _author_label, re.IGNORECASE)
            and not re.search(r"\bet\s+al\.?\b", _author_label,
                              re.IGNORECASE)
        ):
            _signature = _signature_tokens(_author_label)
            if len(_signature) > 1:
                _signatures.add(_signature)
        _doc_match = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(_did))
        if (
            _doc_match
            and _doc_match.group(2) == _year
            and "&" in _doc_match.group(1)
            and "EtAl" not in _doc_match.group(1)
        ):
            _signature = _signature_tokens(
                _doc_match.group(1).replace("&", " and "),
                doc_id_surface=True,
            )
            if len(_signature) > 1:
                _signatures.add(_signature)
        full_signatures[_did] = _signatures
    for _key in display_collisions:
        display_lookup.pop(_key, None)

    # The shared author-year helper preserves one value per key and therefore
    # cannot expose collisions. Build a checker-local fallback that drops
    # every first-author/year key shared by more than one corpus work. Exact
    # display labels and full co-author matches are attempted before this map.
    unique_author_year_lookup = {}
    _first_author_collisions = set()
    for _did in valid_docs:
        _author_label, _year = citation_labels.get(_did, ("", ""))
        if not _author_label or not _year:
            continue
        _row = citation_metadata.get(_did, {})
        _first_author = _normalise_label(
            _row.get("first_author_surname", ""))
        if not _first_author:
            _metadata_authors = (
                _row.get("authors_short", "") or _row.get("authors", ""))
            _first_author = _normalise_label(re.split(
                r"\s+(?:&|and)\s+|\s*;\s*",
                _metadata_authors,
                maxsplit=1,
            )[0])
        if not _first_author:
            _first_author = re.split(
                r"\s+(?:&|and)\s+|\s+et\s+al\.?",
                _normalise_label(_author_label),
                maxsplit=1,
            )[0].strip()
        _keys = [(_first_author, _year)]
        if " et al" in _normalise_label(_author_label):
            _keys.append((_first_author + " et al", _year))
        for _key in _keys:
            if _key in unique_author_year_lookup and unique_author_year_lookup[_key] != _did:
                _first_author_collisions.add(_key)
            else:
                unique_author_year_lookup[_key] = _did
    for _key in _first_author_collisions:
        unique_author_year_lookup.pop(_key, None)

    def _resolve_display_label(label, year):
        """Return the canonical doc_id for a display (label, year) pair, or
        None if it doesn't match any corpus document. Exact labels and full
        co-author signatures take precedence. A single surname or an
        ``et al.`` surface may then use a unique first-author/year fallback."""
        label_norm = _normalise_label(label)
        did = display_lookup.get((label_norm, year))
        if did:
            return did

        # A supplied co-author pair is an explicit signature. It must match
        # in full and cannot degrade to either surname. Token comparison lets
        # a display label omit a particle retained in a legacy doc_id.
        if re.search(r"\s+(?:and|&)\s+", label_norm):
            label_signature = _signature_tokens(label_norm)
            coauthor_matches = [
                candidate for candidate in valid_docs
                if citation_labels.get(candidate, ("", ""))[1] == year
                and label_signature in full_signatures.get(candidate, set())
            ]
            return coauthor_matches[0] if len(coauthor_matches) == 1 else None

        # The only degraded form accepted is a unique first-author/year
        # surface, including the conventional first-author ``et al.`` label.
        first_author = re.split(
            r"\s+et\s+al\.?", label_norm, maxsplit=1)[0]
        first_author = first_author.strip()
        did = unique_author_year_lookup.get((first_author, year))
        if did:
            return did
        did = unique_author_year_lookup.get((first_author + " et al", year))
        return did

    # One parsed unit list drives every citation metric and downstream check.
    # Full members of semicolon groups remain independent citations, while a
    # malformed member contributes one hard E3 event and inherits nothing
    # from its neighbours.
    citation_units, group_e3_details, group_spans = (
        _parse_citation_units(text))
    e1, e2 = 0, 0
    e1_details, e2_details = [], []
    e1_loose_advisory = []  # v16.8: bare "Word Year" prose, advisory only
    docs_cited = set()

    for citation in citation_units:
        page = citation["page"]
        ctx = text[
            max(0, citation["container_start"] - 80):
            citation["container_end"] + 80
        ].replace("\n", " ").strip()

        if citation["kind"] == "canonical":
            did = citation["doc_id"]
        else:
            did = _resolve_display_label(
                citation["label"], citation["year"])
        citation["resolved_doc_id"] = did

        if did is None or did not in valid_docs:
            e1 += 1
            detail = {
                "doc_id": did,
                "page": page,
                "raw": citation["raw"],
                "context": ctx,
                "surface": citation["surface"],
            }
            if citation["kind"] == "canonical":
                detail["doc_id"] = citation["doc_id"]
                detail["reason"] = "canonical_doc_not_in_corpus"
            else:
                detail.update({
                    "label": citation["label"],
                    "year": citation["year"],
                    "reason": "display_label_not_in_corpus",
                })
            e1_details.append(detail)
            continue

        docs_cited.add(did)
        mx = doc_max.get(did)
        if mx is not None and page > mx:
            e2 += 1
            e2_details.append({
                "doc_id": did,
                "cited_page": page,
                "max_valid_page": mx,
                "overshoot": page - mx,
                "context": ctx,
                "surface": citation["surface"],
            })

    quote_only_units = _recover_quote_only_citation_units(
        text, citation_units, group_e3_details, group_spans)
    for citation in quote_only_units:
        if citation["kind"] == "canonical":
            did = citation["doc_id"]
        else:
            did = _resolve_display_label(
                citation["label"], citation["year"])
        citation["resolved_doc_id"] = did

    # Loose form: "Author & Author Year" / "Author and Author Year" without a
    # page. These don't get an E2 check (no page asserted) but they DO assert
    # a corpus document and so bump E1 when unmatched. Skip anything that
    # overlaps a span we've already classified to avoid double-counting.
    citation_spans = [
        (citation["container_start"], citation["container_end"])
        for citation in citation_units
    ]
    all_spans = sorted(set(citation_spans + group_spans))
    for m in _LOOSE_AUTHOR_YEAR_RE.finditer(text):
        if _span_overlaps(m.start(), m.end(), all_spans):
            continue
        label, year = m.group(1), m.group(2)
        # Skip pure-year false positives (e.g. "the 1960s" can't match
        # because the regex requires a leading capitalised surname, but
        # belt-and-braces: require at least one alpha char in the label).
        if not re.search(r"[A-Za-z]", label):
            continue
        did = _resolve_display_label(label, year)
        if did is None:
            # v16.8: bare "Word Year" (no parentheses, no page) is
            # STRUCTURALLY AMBIGUOUS with ordinary historical prose —
            # "England 1066", "Parliament 1688", "In 1850" all match this
            # pattern. Counting them as fabricated citations produced
            # systematic false positives, especially on unrestricted /
            # off-the-shelf essays full of "ProperNoun Year" prose. A hard
            # fabrication metric must not fire on ambiguous surfaces, so
            # these are recorded as an ADVISORY signal (e1_loose_advisory)
            # and NOT summed into E1. Genuine out-of-corpus citations in
            # canonical `(doc: p.N)` or display `Author (Year, p.N)` form —
            # which ARE structurally unambiguous — still count toward E1.
            ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
            e1_loose_advisory.append({
                "doc_id": None, "label": label.strip(), "year": year,
                "page": None, "raw": m.group(0), "context": ctx,
                "surface": "display_loose",
                "reason": "bare_author_year_ambiguous_with_prose",
            })
        else:
            docs_cited.add(did)

    # E3 HARD: format violations that are unambiguously broken (page-only,
    # canonical cite missing a page, multi-page packed cite, bracket dump).
    # v13.1 (FIX-F): exclude spans already classified by the display-form scan
    # so a legitimate "Author (Year, p.N)" surface is not double-counted.
    e3_details = []
    for detail in group_e3_details:
        ctx = text[
            max(0, detail["container_start"] - 80):
            detail["container_end"] + 80
        ].replace("\n", " ").strip()
        e3_details.append({
            "reason": detail["reason"],
            "raw": detail["raw"],
            "context": ctx,
        })
    quote_only_e3_spans = []
    recovered_containers = {}
    for citation in quote_only_units:
        if citation["group_key"][0] != "quote_only":
            continue
        recovered_containers.setdefault(citation["group_key"], citation)
    for citation in recovered_containers.values():
        raw = citation["raw"]
        reason = (
            "page_range_citation"
            if re.search(r"\d+\s*[-\u2013\u2014]\s*\d+", raw)
            else "multi_page_citation"
        )
        ctx = text[
            max(0, citation["container_start"] - 80):
            citation["container_end"] + 80
        ].replace("\n", " ").strip()
        e3_details.append({"reason": reason, "raw": raw, "context": ctx})
        quote_only_e3_spans.append((
            citation["container_start"], citation["container_end"]))
    e3 = len(e3_details)
    standalone_citation_spans = [
        (citation["container_start"], citation["container_end"])
        for citation in citation_units
        if citation["group_key"][0] == "standalone"
    ]

    def _e3_match_is_accounted_for(match):
        # A parsed group owns its complete container because malformed
        # members have already received unit-level E3 treatment. For a
        # standalone citation, suppress only patterns that begin inside the
        # valid citation. A malformed outer wrapper can contain a valid inner
        # display citation and must remain visible as its own E3 event.
        if _span_overlaps(match.start(), match.end(), group_spans):
            return True
        if _span_overlaps(
                match.start(), match.end(), quote_only_e3_spans):
            return True
        return any(
            span_start <= match.start() < span_end
            for span_start, span_end in standalone_citation_spans
        )

    for reason, pat in _E3_HARD_PATTERNS:
        for m in pat.finditer(text):
            if not _e3_match_is_accounted_for(m):
                e3 += 1
                ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
                e3_details.append({"reason": reason, "raw": m.group(0), "context": ctx})

    # E3 SOFT: display-form author-year mentions without a page number. These
    # do not count toward the hard E3 score but are surfaced for visibility.
    # The writer prompt asks for pages on every cite — a high e3_soft count
    # signals the writer is dropping pages, which is a prose-quality concern
    # rather than a fabrication.
    e3_soft = 0
    e3_soft_details = []
    for reason, pat in _E3_SOFT_PATTERNS:
        for m in pat.finditer(text):
            if not _e3_match_is_accounted_for(m):
                e3_soft += 1
                ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
                e3_soft_details.append({"reason": reason, "raw": m.group(0), "context": ctx})

    # E4 / E5: quote-text integrity. Scan every "..." span >=20 chars and
    # associate it with all actionable citations in the same sentence. When
    # the sentence has no citation, fall back to the single nearest citation
    # in the surrounding window. This prevents a closer citation in an
    # adjacent sentence from displacing the citation that governs the quote.
    e4 = 0
    e5 = 0
    e4_details = []
    e5_details = []
    quotes_checked = 0
    quotes_verified = 0
    if quote_check and os.path.isdir(page_text_dir):
        sentence_boundaries = tuple(_sentence_breaks(text))
        for qm in _QUOTED_SPAN_RE.finditer(text):
            quote = qm.group(1)
            # v16.8: reject mispaired spans that embed a citation / evidence
            # id — they are not verbatim quotations and would false-positive
            # as E4/E5.
            if _SPAN_HAS_CITATION_RE.search(quote):
                continue
            # v16.9: reject mispaired scare-quote spans structurally. A
            # quoted phrase SHORTER than the 20-char span floor (e.g.
            # "reversal of fortune", 19 chars) cannot match as a quote, so
            # the matcher pairs its CLOSING mark with the NEXT phrase's
            # opening mark and captures the interstitial PROSE as a phantom
            # quote — which then false-positives as E4. A genuine quotation
            # never begins or ends with whitespace inside its marks, so edge
            # whitespace is the mispairing signature. Production audit
            # (80 Claude-arm essays): 16/151 Sonnet and 71/262 Opus E4s
            # carried it; every inspected case was interstitial prose.
            if quote != quote.strip():
                continue
            q_start, q_end = qm.span()
            win_start = max(0, q_start - _QUOTE_WINDOW_CHARS)
            win_end = min(len(text), q_end + _QUOTE_WINDOW_CHARS)

            def _candidate_from_unit(citation):
                return {
                    "doc_id": citation["resolved_doc_id"],
                    "page": citation["page"],
                    "start": citation["start"],
                    "end": citation["end"],
                    "surface": citation["surface"],
                    "group_key": citation["group_key"],
                    "container_start": citation["container_start"],
                    "container_end": citation["container_end"],
                }

            actionable_units = []
            actionable_seen = set()
            for citation in citation_units + quote_only_units:
                if citation.get("resolved_doc_id") not in valid_docs:
                    continue
                identity = (
                    citation["resolved_doc_id"], citation["page"],
                    citation["group_key"],
                )
                if identity in actionable_seen:
                    continue
                actionable_seen.add(identity)
                actionable_units.append(citation)

            def _groups_overlapping(region_start, region_end):
                keys = {
                    citation["group_key"]
                    for citation in actionable_units
                    if citation["container_start"] < region_end
                    and citation["container_end"] > region_start
                }
                grouped = {}
                for citation in actionable_units:
                    if citation["group_key"] in keys:
                        grouped.setdefault(citation["group_key"], []).append(
                            _candidate_from_unit(citation))
                return grouped

            sentence_start, sentence_end = _sentence_bounds(
                text, q_start, q_end, sentence_boundaries)
            sentence_groups = _groups_overlapping(
                sentence_start, sentence_end)
            candidates = [
                candidate
                for group in sorted(
                    sentence_groups.values(),
                    key=lambda members: members[0]["container_start"],
                )
                for candidate in group
            ]
            candidate_scope = "same_sentence"
            if not candidates:
                nearby_groups = _groups_overlapping(win_start, win_end)
            else:
                nearby_groups = {}
            if nearby_groups:
                quote_center = (q_start + q_end) / 2
                nearest_group = min(
                    nearby_groups.values(),
                    key=lambda members: abs(
                        (members[0]["container_start"]
                         + members[0]["container_end"]) / 2
                        - quote_center),
                )
                fallback_sentence = _sentence_bounds(
                    text,
                    nearest_group[0]["container_start"],
                    nearest_group[0]["container_end"],
                    sentence_boundaries,
                )
                fallback_sentence_groups = _groups_overlapping(
                    *fallback_sentence)
                candidates = [
                    candidate
                    for group in sorted(
                        fallback_sentence_groups.values(),
                        key=lambda members: members[0]["container_start"],
                    )
                    for candidate in group
                ]
                candidate_scope = "nearest_sentence_fallback"

            ctx = text[max(0, q_start - 80):q_end + 80].replace("\n", " ").strip()
            if not candidates:
                # No actionable attribution, so the quote cannot be checked.
                continue
            quotes_checked += 1
            norm_q = _normalise_quote_needle(quote)
            if not norm_q:
                continue

            cited_pages = []
            all_pages_available = True
            for candidate in candidates:
                page_txt = _load_page_text(
                    candidate["doc_id"], candidate["page"], page_text_dir)
                if page_txt is None:
                    all_pages_available = False
                cited_pages.append((candidate, page_txt))

            if any(
                page_txt is not None
                and _quote_matches_normalised_page(
                    quote, _normalise_for_quote_match(page_txt))
                for candidate, page_txt in cited_pages
            ):
                quotes_verified += 1
                continue

            # A missing cited page is already an E2 event. It cannot support a
            # reliable E4/E5 classification, so avoid double-counting it.
            if not all_pages_available:
                continue

            e5_hit = None
            for candidate in candidates:
                other_pages = _find_other_pages_with_quote(
                    candidate["doc_id"], quote, page_text_dir,
                    exclude_page=candidate["page"],
                )
                if other_pages:
                    e5_hit = (candidate, other_pages)
                    break
            candidate_labels = [
                f'{candidate["doc_id"]} p.{candidate["page"]}'
                for candidate in candidates
            ]
            if e5_hit:
                candidate, other_pages = e5_hit
                e5 += 1
                e5_details.append({
                    "doc_id": candidate["doc_id"],
                    "cited_page": candidate["page"],
                    "found_on_pages": other_pages,
                    "candidate_scope": candidate_scope,
                    "candidate_cites": candidate_labels,
                    "quote": quote[:240], "context": ctx,
                })
            else:
                e4 += 1
                candidate = candidates[0]
                e4_details.append({
                    "doc_id": candidate["doc_id"],
                    "cited_page": candidate["page"],
                    "candidate_scope": candidate_scope,
                    "candidate_cites": candidate_labels,
                    "quote": quote[:240], "context": ctx,
                })

    word_count = len(re.findall(r"\b\w+\b", text))

    # v14.2: n_citations now sums canonical + display + loose. Final
    # user-facing reviews are always in display form (the canonical form is
    # only the internal validation surface), so counting canonical-only made
    # the checker report n_citations=0 on every real review — a confusing
    # gap that masked review quality.
    n_canonical = sum(
        citation["kind"] == "canonical" for citation in citation_units)
    n_display = sum(
        citation["kind"] == "display" for citation in citation_units)
    n_total_citations = n_canonical + n_display

    return {
        "n_citations": n_total_citations,
        "n_canonical_citations": n_canonical,
        "n_display_citations": n_display,
        "e1": e1, "e2": e2, "e3": e3,
        "e3_soft": e3_soft,
        "e4": e4, "e5": e5,
        "docs_cited": sorted(docs_cited),
        "n_docs_cited": len(docs_cited),
        "word_count": word_count,
        "quotes_checked": quotes_checked,
        "quotes_verified": quotes_verified,
        "e1_details": e1_details,
        "e1_loose_advisory": e1_loose_advisory,
        "e1_loose_advisory_count": len(e1_loose_advisory),
        "e2_details": e2_details,
        "e3_details": e3_details,
        "e3_soft_details": e3_soft_details,
        "e4_details": e4_details,
        "e5_details": e5_details,
    }


def _empty_result(reason):
    return {
        "n_citations": 0, "e1": 0, "e2": 0, "e3": 0, "e3_soft": 0,
        "e4": 0, "e5": 0,
        "docs_cited": [], "n_docs_cited": 0, "word_count": 0,
        "quotes_checked": 0, "quotes_verified": 0,
        "e1_details": [], "e2_details": [], "e3_details": [],
        "e3_soft_details": [], "e4_details": [], "e5_details": [],
        "refusal": True, "reason": reason,
    }


def check_file(path, metadata_path="metadata.csv", data_dir="data", quote_check=True):
    """Check a review file by path. Handles missing/empty files as refusal."""
    if not os.path.isfile(path):
        return _empty_result("file_not_found")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) < 100:
        return _empty_result("empty_output")
    r = check_review(text, metadata_path, data_dir, quote_check=quote_check)
    r["refusal"] = False
    return r


# ── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check citations in a review file")
    ap.add_argument("review", help="Path to review markdown file")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-quote-check", action="store_true",
                    help="Skip E4/E5 (quoted-text verification against page_text)")
    ap.add_argument("--json", nargs="?", const="-", default=None,
                    help="Output JSON (path, or '-' for stdout)")
    args = ap.parse_args()

    result = check_file(args.review, args.metadata, args.data_dir,
                        quote_check=not args.no_quote_check)

    if args.json is not None:
        out = json.dumps(result, indent=2, ensure_ascii=False)
        if args.json == "-":
            print(out)
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(out)
    else:
        print(f"Citations: {result['n_citations']} "
              f"(canonical={result.get('n_canonical_citations', 0)}, "
              f"display={result.get('n_display_citations', 0)})   "
              f"Words: {result.get('word_count', 0)}")
        print(f"E1 (out-of-corpus cite):   {result['e1']}")
        print(f"E2 (invalid page):         {result['e2']}")
        print(f"E3 (hard format):          {result['e3']}")
        print(f"  e3_soft (no-page mention): {result.get('e3_soft', 0)}")
        print(f"E4 (unverified quote):     {result.get('e4', 0)}")
        print(f"E5 (mis-attributed quote): {result.get('e5', 0)}")
        print(f"  quotes checked / verified: {result.get('quotes_checked', 0)} / "
              f"{result.get('quotes_verified', 0)}")
        print(f"Docs cited:                {result['n_docs_cited']}")
        if result.get("refusal"):
            print(f"REFUSAL: {result.get('reason', 'unknown')}")
        if result["e1_details"]:
            print("\nE1 details:")
            for d in result["e1_details"]:
                # v13.1: display-form entries carry a label/year/surface
                # instead of a canonical doc_id. Print whichever shape we have.
                if d.get("doc_id"):
                    print(f"  {d['doc_id']}: p.{d['page']}")
                else:
                    page = d.get("page")
                    page_str = f", p.{page}" if page is not None else ""
                    surface = d.get("surface", "display")
                    print(f"  [{surface}] {d.get('label')} {d.get('year')}{page_str}")
        if result["e2_details"]:
            print("\nE2 details:")
            for d in result["e2_details"]:
                print(f"  {d['doc_id']}: cited p.{d['cited_page']}, "
                      f"max p.{d['max_valid_page']} (+{d['overshoot']})")
        if result.get("e3_details"):
            print("\nE3 (hard) details:")
            for d in result["e3_details"][:20]:
                print(f"  {d['reason']}: {d['raw']}")
        if result.get("e3_soft_details"):
            print(f"\ne3_soft details (first 5 of {len(result['e3_soft_details'])}):")
            for d in result["e3_soft_details"][:5]:
                print(f"  {d['reason']}: {d['raw']}")
        if result.get("e4_details"):
            print("\nE4 (UNVERIFIED QUOTE) details:")
            for d in result["e4_details"]:
                print(f"  {d['doc_id']} p.{d['cited_page']}: \"{d['quote'][:120]}...\"")
        if result.get("e5_details"):
            print("\nE5 (MIS-ATTRIBUTED QUOTE) details:")
            for d in result["e5_details"]:
                pages = ", ".join(f"p.{p}" for p in d["found_on_pages"][:5])
                print(f"  {d['doc_id']} cited p.{d['cited_page']} -> actually on {pages}: "
                      f"\"{d['quote'][:120]}...\"")
