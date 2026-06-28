#!/usr/bin/env python3
"""
check_citations.py — E1/E2/E3 citation integrity checker.

Parses a composed review (.md) and checks every (DocId: p.N) citation against
the corpus metadata.csv and per-document page counts in data/*.json.

Error taxonomy:
    E1  Fabricated document — doc_id not in metadata.csv
    E2  Invalid page       — cited page exceeds doc's content-page count
    E3  Format violation   — citation-like pattern that misses strict format

CLI:
    python3 scripts/check_citations.py runs/review_composed.md
    python3 scripts/check_citations.py runs/review_composed.md --json results.json
    python3 scripts/check_citations.py runs/review_composed.md --json -   # stdout

Import:
    from check_citations import check_file, check_review
"""

import os, re, json, glob, sys

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

# E3 patterns: citation-like strings that fail the strict format
_E3_PATTERNS = [
    ("doc_without_page", re.compile(r"\((?=[^)]*[A-Za-z0-9_&.\-]+_\d{4})(?![^)]*:\s*p\.)[^)]*\)")),
    ("page_only", re.compile(r"\((?:pp?\.)\s*\d+(?:\s*(?:,|-|and)\s*(?:pp?\.)?\s*\d+)*\)", re.IGNORECASE)),
    ("author_year_text", re.compile(rf"\b{_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?\s*\(\d{{4}}\)")),
    ("author_year_possessive", re.compile(rf"\b{_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?'s\s*\(\d{{4}}\)")),
    ("author_year_parenthetical", re.compile(rf"\({_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?,\s*\d{{4}}\)")),
    ("multi_page_citation", re.compile(r"\([A-Za-z0-9_&.\-]+:\s*p\.\d+\s*,\s*p\.\d+[^)]*\)")),
    ("square_bracket_dump", re.compile(
        r"^\s*\[[^\]\n]*[A-Za-z0-9_&.\-]+:\s*p\.\d+[^\]\n]*(?:;\s*[A-Za-z0-9_&.\-]+:\s*p\.\d+|,\s*p\.\d+)[^\]\n]*\]\s*$",
        re.MULTILINE,
    )),
]


def _load_valid_docs(metadata_path):
    """Load set of valid doc_ids from metadata CSV."""
    if not os.path.isfile(metadata_path):
        return set()
    import pandas as pd
    df = pd.read_csv(metadata_path)
    return set(str(x) for x in df["doc_id"])


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


def check_review(text, metadata_path="metadata.csv", data_dir="data"):
    """
    Check a review text string for E1/E2/E3 errors.

    Returns dict with n_citations, e1, e2, e3, docs_cited, details, word_count.
    """
    valid_docs = _load_valid_docs(metadata_path)
    doc_max = _load_doc_max_pages(data_dir)

    # v13.1 (FIX-F): build the author/year lookups from the corpus so we can
    # resolve display-form citations the writer emits ("Author (Year, p.N)"
    # and "(Author Year, p.N)") and bump E1 when the (author, year) tuple
    # doesn't match any corpus document. Without this layer, the strict
    # CITE_RE check above misses display-form fabrications entirely — the
    # v13 smoke surfaced "Dalrymple-Smith & Frankema 2017" with e1=0.
    author_year_lookup = _build_author_year_lookup(valid_docs)
    display_lookup = _build_display_lookup(valid_docs)

    # v13.1 (FIX-F): extra fallback for "ShortLabel Year" surfaces where the
    # writer dropped a co-author — e.g. corpus has "Nunn&Wantchekon_2011" but
    # the prose says "Wantchekon 2011". The display lookup keys on the full
    # label only, and the author-year lookup keys on the first surname only,
    # so we build a third map keyed on EVERY surname token in each doc_id.
    # Ambiguous keys (two docs sharing a surname+year) are dropped so we
    # never resolve incorrectly.
    any_author_year_lookup = {}
    _collisions = set()
    for _did in valid_docs:
        _m = re.match(r"^(.+?)_(\d{4})[a-z]?$", str(_did))
        if not _m:
            continue
        _name, _year = _m.group(1), _m.group(2)
        _name = _name.replace("EtAl", "")
        _tokens = [t for t in re.split(r"&|\s+", _name) if t and t.lower() not in {"van", "von", "de", "del", "der"}]
        for _t in _tokens:
            _k = (_t.lower(), _year)
            if _k in any_author_year_lookup and any_author_year_lookup[_k] != _did:
                _collisions.add(_k)
            else:
                any_author_year_lookup[_k] = _did
    for _k in _collisions:
        any_author_year_lookup.pop(_k, None)

    # Parse all strict-format citations
    citations = []
    for m in CITE_RE.finditer(text):
        citations.append({
            "doc_id": m.group(1), "page": int(m.group(2)),
            "start": m.start(), "end": m.end(), "raw": m.group(0)
        })

    e1, e2 = 0, 0
    e1_details, e2_details = [], []
    docs_cited = set()

    for c in citations:
        did, page = c["doc_id"], c["page"]
        ctx = text[max(0, c["start"] - 80):c["end"] + 80].replace("\n", " ").strip()

        # E1: fabricated document
        if did not in valid_docs:
            e1 += 1
            e1_details.append({
                "doc_id": did, "page": page, "context": ctx,
                "surface": "canonical",
            })
            continue

        docs_cited.add(did)

        # E2: invalid page
        mx = doc_max.get(did)
        if mx is not None and page > mx:
            e2 += 1
            e2_details.append({
                "doc_id": did, "cited_page": page,
                "max_valid_page": mx, "overshoot": page - mx, "context": ctx
            })

    # v13.1 (FIX-F): scan display-form citations and resolve against the
    # corpus lookups. Unmatched display labels are E1 (the writer asserted a
    # paper that isn't in the corpus). Matched-but-page-over-max are E2.
    # Spans are recorded so the loose author-year scan below doesn't
    # double-count them.
    strict_spans = [(c["start"], c["end"]) for c in citations]
    display_spans = []

    def _resolve_display_label(label, year):
        """Return the canonical doc_id for a display (label, year) pair, or
        None if it doesn't match any corpus document. Tries the display
        lookup (full label) first, then the author-year lookup (first
        surname only) as a fallback for legacy/loose surfaces."""
        label_norm = label.strip().lower().rstrip(".")
        did = display_lookup.get((label_norm, year))
        if did:
            return did
        # Fallback: strip "et al." / co-author tail and try the leading
        # surname against the author-year lookup. Handles "Author & Co Year"
        # and "Author and Co Year" surfaces the display lookup doesn't key.
        first_author = re.split(r"\s+(?:&|and)\s+|\s+et\s+al\.?", label_norm, maxsplit=1)[0]
        first_author = first_author.strip()
        did = author_year_lookup.get((first_author, year))
        if did:
            return did
        # Last-ditch: "et al" key in the author-year lookup for EtAl docs.
        did = author_year_lookup.get((first_author + " et al", year))
        if did:
            return did
        # v13.1: try every surname token across the label, in order. Handles
        # "Wantchekon 2011" -> "Nunn&Wantchekon_2011" (writer dropped the
        # first author). Returns the first unambiguous corpus match.
        for tok in re.findall(r"[A-Za-z\-]+", label_norm):
            if tok in {"and", "et", "al", "van", "von", "de", "del", "der"}:
                continue
            did = any_author_year_lookup.get((tok, year))
            if did:
                return did
        return None

    def _scan_display(pat, surface):
        nonlocal e1, e2
        for m in pat.finditer(text):
            # Skip if this span overlaps a strict canonical citation already
            # accounted for. (Shouldn't happen given the patterns, but cheap
            # insurance.)
            if any(s <= m.start() < e for s, e in strict_spans):
                continue
            label, year, page = m.group(1), m.group(2), int(m.group(3))
            display_spans.append((m.start(), m.end()))
            ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
            did = _resolve_display_label(label, year)
            if did is None:
                e1 += 1
                e1_details.append({
                    "doc_id": None, "label": label, "year": year, "page": page,
                    "raw": m.group(0), "context": ctx, "surface": surface,
                    "reason": "display_label_not_in_corpus",
                })
                continue
            docs_cited.add(did)
            mx = doc_max.get(did)
            if mx is not None and page > mx:
                e2 += 1
                e2_details.append({
                    "doc_id": did, "cited_page": page,
                    "max_valid_page": mx, "overshoot": page - mx,
                    "context": ctx, "surface": surface,
                })

    _scan_display(DISPLAY_CITE_RE, "display_bare")
    _scan_display(DISPLAY_PAREN_CITE_RE, "display_paren")

    # Loose form: "Author & Author Year" / "Author and Author Year" without a
    # page. These don't get an E2 check (no page asserted) but they DO assert
    # a corpus document and so bump E1 when unmatched. Skip anything that
    # overlaps a span we've already classified to avoid double-counting.
    all_spans = strict_spans + display_spans
    for m in _LOOSE_AUTHOR_YEAR_RE.finditer(text):
        if any(s <= m.start() < e for s, e in all_spans):
            continue
        label, year = m.group(1), m.group(2)
        # Skip pure-year false positives (e.g. "the 1960s" can't match
        # because the regex requires a leading capitalised surname, but
        # belt-and-braces: require at least one alpha char in the label).
        if not re.search(r"[A-Za-z]", label):
            continue
        did = _resolve_display_label(label, year)
        if did is None:
            e1 += 1
            ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
            e1_details.append({
                "doc_id": None, "label": label.strip(), "year": year,
                "page": None, "raw": m.group(0), "context": ctx,
                "surface": "display_loose",
                "reason": "display_label_not_in_corpus",
            })
        else:
            docs_cited.add(did)

    # E3: loose citation patterns outside strict format. v13.1 (FIX-F): also
    # exclude spans that were already classified by the display-form scan
    # above — otherwise legitimate "Author (Year, p.N)" surfaces get
    # double-counted as E3 (author_year_text/author_year_parenthetical).
    e3 = 0
    e3_details = []
    accounted_spans = strict_spans + display_spans
    for reason, pat in _E3_PATTERNS:
        for m in pat.finditer(text):
            if not any(s <= m.start() < e for s, e in accounted_spans):
                e3 += 1
                ctx = text[max(0, m.start() - 80):m.end() + 80].replace("\n", " ").strip()
                e3_details.append({"reason": reason, "raw": m.group(0), "context": ctx})

    word_count = len(re.findall(r"\b\w+\b", text))

    return {
        "n_citations": len(citations),
        "e1": e1, "e2": e2, "e3": e3,
        "docs_cited": sorted(docs_cited),
        "n_docs_cited": len(docs_cited),
        "word_count": word_count,
        "e1_details": e1_details,
        "e2_details": e2_details,
        "e3_details": e3_details,
    }


def check_file(path, metadata_path="metadata.csv", data_dir="data"):
    """Check a review file by path. Handles missing/empty files as refusal."""
    if not os.path.isfile(path):
        return {
            "n_citations": 0, "e1": 0, "e2": 0, "e3": 0,
            "docs_cited": [], "n_docs_cited": 0, "word_count": 0,
            "e1_details": [], "e2_details": [],
            "refusal": True, "reason": "file_not_found",
        }
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) < 100:
        return {
            "n_citations": 0, "e1": 0, "e2": 0, "e3": 0,
            "docs_cited": [], "n_docs_cited": 0, "word_count": 0,
            "e1_details": [], "e2_details": [],
            "refusal": True, "reason": "empty_output",
        }
    r = check_review(text, metadata_path, data_dir)
    r["refusal"] = False
    return r


# ── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check citations in a review file")
    ap.add_argument("review", help="Path to review markdown file")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--json", nargs="?", const="-", default=None,
                    help="Output JSON (path, or '-' for stdout)")
    args = ap.parse_args()

    result = check_file(args.review, args.metadata, args.data_dir)

    if args.json is not None:
        out = json.dumps(result, indent=2, ensure_ascii=False)
        if args.json == "-":
            print(out)
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(out)
    else:
        print(f"Citations: {result['n_citations']}   Words: {result.get('word_count', 0)}")
        print(f"E1 (fabricated doc):  {result['e1']}")
        print(f"E2 (invalid page):   {result['e2']}")
        print(f"E3 (format):         {result['e3']}")
        print(f"Docs cited:          {result['n_docs_cited']}")
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
            print("\nE3 details:")
            for d in result["e3_details"][:20]:
                print(f"  {d['reason']}: {d['raw']}")
