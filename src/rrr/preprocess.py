import argparse, os, pandas as pd, re
import pymupdf
from rrr.utils import sha256_file, ensure_dir, save_json
from rrr.paths import data_path
from rrr.text import normalize_text, sentence_spans
from multiprocessing import Pool, cpu_count

# v14.3: extractor swapped pdfminer.six -> PyMuPDF after the v14.3 extraction
# audit (workflow wf_b750f0f0-eeb) showed PyMuPDF: 14-44x faster, 0 fi/fl
# ligatures vs pdfminer's 3186 across 488 pages, 0 double-space runs (vs
# pervasive), preserves per-character font/superscript metadata. Cleanup
# pipeline in scripts/clean_page_text.py handles residual artifacts.

# Reference section markers (case-insensitive check done separately).
# v14.3: dropped the leading `\b` boundary on the inline REFERENCES match so
# glued headers like `control.REFERENCESAlatas` (observed in Stoler_1989 p.23,
# Austin_2007 p.24, Austin_2008 p.26, Allen_2001 p.31, Kuznets_1973 p.12,
# Peters_2004 p.38) get detected. The line-anchored patterns stay strict.
_REF_HEADERS = [
    r'^\s*REFERENCES\s*$',
    r'^\s*References\s*$',
    r'^\s*BIBLIOGRAPHY\s*$',
    r'^\s*Bibliography\s*$',
    r'^\s*Works\s+Cited\s*$',
    r'^\s*WORKS\s+CITED\s*$',
    r'^\s*Reference\s+List\s*$',
    r'^\s*REFERENCE\s+LIST\s*$',
    r'^\s*Literature\s+Cited\s*$',
    r'^\s*LITERATURE\s+CITED\s*$',
    r'^\s*Sources\s*$',
    r'^\s*SOURCES\s*$',
]

# v14.3: inline matcher catches glued reference-headers (no leading word
# boundary). Returns a re.Match so we can use the match's position to split
# the page.
_INLINE_REFERENCES_RE = re.compile(r'REFERENCES(?=[A-Z]|\s|$)', re.IGNORECASE)

# v16.12: a STRONG glued reference header — ALL-CAPS 'REFERENCES' only (the
# JSTOR 'control.REFERENCESAlatas' run-on). Distinct from the mere word
# 'references' in body prose. Header detection uses this to take precedence
# over the density fallback; the IGNORECASE matcher above stays the WEAK
# signal, honoured only when its own page is also reference-dense.
_STRONG_INLINE_REFERENCES_RE = re.compile(r'REFERENCES(?=[A-Z]|\s|$)')

# v16.13: JSTOR cover-page boilerplate — "REFERENCES / Linked references are
# available on JSTOR for this article ...". Metadata noise on the cover, NOT
# the article bibliography; a page carrying it is never a reference boundary
# and is exempt from the post-extraction leak invariant.
_JSTOR_COVER_RE = re.compile(r'Linked references are available on JSTOR', re.IGNORECASE)

# A bibliographic entry line: "Surname, I. ..." / "Surname, Firstname ..." at
# line start, followed by SUBSTANTIAL content (author + title/journal). The
# trailing-length requirement distinguishes a real reference entry from a short
# "City, Country" data-table row (Nunn_2008 Table I: "Valencia, Spain"), which
# would otherwise false-positive. Digit-independent, so it survives OCR that
# drops the year/pages (Koepke&Baten: "ALLEN, R. (). Economic structure ...").
_BIB_ENTRY_LINE_RE = re.compile(
    r'(?m)^\s*[A-Z][A-Za-z\'.\-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+).{20,}')

# A standalone reference-section heading on its own line (for the invariant).
_STANDALONE_REF_HEADER_RE = re.compile(
    r'(?m)^\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|WORKS CITED|Works Cited)\s*$')


def _truncate_at_reference_header(page_text: str) -> tuple:
    """v14.3: when a reference header appears mid-page (line-anchored or
    glued inline), return (text_before_header, True). When no header,
    return (page_text, False).

    Captures the v14.3 audit finding that pages like Allen_2001 p.31 have
    `control.REFERENCESAlatas, S. (1977)...` — the page's pre-header content
    is real main text and should be kept; everything after is the start of
    the reference list and should be dropped along with subsequent pages.
    """
    earliest = len(page_text)
    found = False
    for pattern in _REF_HEADERS:
        m = re.search(pattern, page_text, re.MULTILINE)
        if m and m.start() < earliest:
            earliest = m.start()
            found = True
    m_inline = _INLINE_REFERENCES_RE.search(page_text)
    if m_inline and m_inline.start() < earliest:
        earliest = m_inline.start()
        found = True
    if found:
        return page_text[:earliest].rstrip(), True
    return page_text, False

def _is_reference_dense(page_text: str) -> bool:
    """Check if page has dense reference-list patterns."""
    if len(page_text) < 200:
        return False

    # Count reference indicators
    # Author, Initial. (Year) pattern
    author_year = len(re.findall(r'[A-Z][a-z]+,?\s+[A-Z]\..*?\(\d{4}\)', page_text))
    # Journal of / Review of / Quarterly patterns
    journals = len(re.findall(r'\b(Journal of|Review of|Quarterly|Economic History|American Economic)\b', page_text, re.IGNORECASE))
    # Page ranges: 123-456 or 123–456
    page_ranges = len(re.findall(r'\b\d{1,4}[-–]\d{1,4}\b', page_text))
    # Publisher names
    publishers = len(re.findall(r'(University Press|Cambridge|Oxford|Princeton|MIT Press|Wiley|Elsevier)', page_text, re.IGNORECASE))
    # Volume/number patterns: Vol. 5, No. 2 or 117(4)
    vol_num = len(re.findall(r'(\bVol\.?\s*\d|\bNo\.?\s*\d|\d+\s*\(\d+\))', page_text))
    # "eds." or "ed." patterns
    editors = len(re.findall(r'\beds?\.', page_text))
    # DOI patterns
    dois = len(re.findall(r'doi:|DOI:', page_text))
    # v16.13: reference-entry lines are a DIGIT-INDEPENDENT signal — they
    # survive OCR that mangles years/pages (Koepke&Baten's "()" / "pp. -"),
    # which used to zero out every numeric indicator and hide a dense
    # bibliography page from the density fallback.
    entry_lines = len(_BIB_ENTRY_LINE_RE.findall(page_text))

    total_indicators = (author_year + journals + page_ranges + publishers
                        + vol_num + editors + dois + entry_lines)

    # Calculate density per 500 chars
    density = total_indicators / (len(page_text) / 500)

    return density > 3  # More than 3 indicators per 500 chars = likely references

def _is_strong_reference_header(page_text: str) -> bool:
    """A STRUCTURAL reference header: a line-anchored heading (REFERENCES,
    BIBLIOGRAPHY, WORKS CITED, ...) or a glued ALL-CAPS 'REFERENCES' run-on.
    High-confidence, unlike the bare word 'references' which also occurs in
    body prose."""
    for pattern in _REF_HEADERS:
        if re.search(pattern, page_text, re.MULTILINE):
            return True
    return bool(_STRONG_INLINE_REFERENCES_RE.search(page_text))


def _is_reference_start_page(pages: list, i: int) -> bool:
    """True if page i begins the reference section.

    A strong structural header counts when this page OR the next is
    reference-dense (covers a header sitting at the very bottom of a page).
    A weak, case-insensitive 'references' match (e.g. the glued lowercase
    'Footnote references Allen, R. C., ...' in Bolt&vanZanden_2014) counts
    ONLY when its OWN page is reference-dense — so the word 'references' in
    body prose (Bryant_2006's coda) does not trigger a premature cut."""
    page_text = pages[i]
    n = len(pages)
    # JSTOR cover boilerplate is metadata noise, never a reference boundary.
    if _JSTOR_COVER_RE.search(page_text):
        return False
    if _is_strong_reference_header(page_text):
        # v16.13: a TAIL-positioned structural REFERENCES/BIBLIOGRAPHY heading
        # is strong evidence on its own — cut regardless of density (that
        # density requirement was what let Kuznets_1973, Peters_2004 and
        # Koepke&Baten_2005 leak their bibliographies: their numbers were
        # mangled in extraction or the header sat on the last page with no next
        # page to corroborate). An EARLY header (intro/body) still needs
        # corroboration to avoid a false cut. Boundary truncation
        # (_truncate_at_reference_header) keeps any main text preceding the
        # header on the same page.
        if i >= n * 0.5:
            return True
        return _is_reference_dense(page_text) or (
            i + 1 < n and _is_reference_dense(pages[i + 1]))
    # A weak lowercase 'references' counts only when its own page is dense.
    if _INLINE_REFERENCES_RE.search(page_text) and _is_reference_dense(page_text):
        return True
    return False


def _find_reference_start(pages: list) -> int:
    """Find the page index where the reference section starts (-1 if none).

    v16.12: header detection takes PRECEDENCE over the density fallback. The
    old single pass interleaved them per page, so a short 'CONCLUSION' page or
    a data appendix (dense with year-ranges) could be returned BEFORE the
    document's real reference header — dropping genuine body text (Allen_2001
    lost its conclusion). Pass 1 now finds the first genuine reference-section
    page anywhere in the document; the density-only fallback (pass 2) fires
    solely when no reference header exists at all."""
    n = len(pages)
    # Pass 1: first genuine reference-section page (header-anchored).
    for i in range(n):
        if _is_reference_start_page(pages, i):
            return i
    # Pass 2: no header anywhere — trailing density run in the last 40%.
    for i in range(n):
        if i > n * 0.6 and _is_reference_dense(pages[i]) and \
                i + 1 < n and _is_reference_dense(pages[i + 1]):
            return i
    return -1

def extract_pages(pdf_path: str):
    """v14.3: extract per-page text using PyMuPDF (was pdfminer.six).

    PyMuPDF returns a Document where each page exposes `get_text()`. Unlike
    pdfminer's high-level `extract_text` (which splits on form-feed and
    leaves ligatures + double-spaces from column-flowed layouts), PyMuPDF
    reads the PDF's text objects directly and produces cleaner output.
    Empty pages (commonly the JSTOR splash or a between-section blank) are
    dropped post-strip. normalize_text still applies for unicode hyphen /
    quote canonicalisation; the scripts/clean_page_text.py pass adds the
    heavier-weight cleanup steps.
    """
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    pages = [p.strip() for p in pages if p.strip()]
    pages = [normalize_text(p) for p in pages]
    return pages

def _process_one(row_dict):
    pdf = row_dict.get("pdf_path")
    doc_id = str(row_dict.get("doc_id"))

    # Clear old pages for this doc before writing new ones — page_text AND
    # page_meta. v16.13: clearing only page_text left stale page_meta files
    # whenever a doc's page count DECREASED (e.g. Peters_2004 39->38,
    # Koepke&Baten_2005 35->31 once their leaked bibliographies were stripped),
    # breaking the page_text/page_meta consistency invariant.
    import glob
    for _pat in (data_path("page_text", f"{doc_id}_page_*.txt"),
                 data_path("page_meta", f"{doc_id}_page_*.json")):
        for old in glob.glob(str(_pat)):
            os.remove(old)

    if not (isinstance(pdf, str) and os.path.isfile(pdf)):
        return {"doc_id": doc_id, "ok": False, "reason": "missing_pdf"}
    try:
        h = sha256_file(pdf)
        pages = extract_pages(pdf)

        # Find where references start
        ref_start = _find_reference_start(pages)

        # Determine which pages to keep
        if ref_start > 0:
            content_pages = pages[:ref_start]
            ref_pages = len(pages) - ref_start
        else:
            content_pages = pages
            ref_pages = 0

        # v14.3: when the FIRST reference page contains main-text content
        # before the header (e.g. Allen_2001 p.31: "...political control.
        # REFERENCES Alatas, S. (1977)..."), truncate that page so the main
        # text is kept and the references portion is dropped. Without this,
        # the entire page (with its mixed content) was dropped, losing real
        # main-text claims; or kept whole, surfacing the reference list to BM25.
        if ref_start > 0:
            boundary_page = pages[ref_start]
            trimmed, truncated = _truncate_at_reference_header(boundary_page)
            if truncated and trimmed:
                content_pages = content_pages + [trimmed]
                ref_pages -= 1

        # Write only content pages
        for i, ptxt in enumerate(content_pages, start=1):
            outp = data_path("page_text", f"{doc_id}_page_{i}.txt")
            ensure_dir(os.path.dirname(str(outp)))
            with open(outp, "w", encoding="utf-8") as f:
                f.write(ptxt)
            meta_dir = data_path("page_meta")
            ensure_dir(str(meta_dir))
            save_json(
                {
                    "doc_id": doc_id,
                    "page": i,
                    "page_type": "content",
                    "char_count": len(ptxt),
                    "sentences": sentence_spans(ptxt),
                },
                str(meta_dir / f"{doc_id}_page_{i}.json"),
            )

        meta = {
            "doc_id": doc_id,
            "pdf_path": pdf,
            "hash": h,
            "page_count": len(content_pages),
            "total_pages": len(pages),
            "ref_pages_excluded": ref_pages
        }
        save_json(meta, str(data_path(f"{doc_id}.json")))

        return {"doc_id": doc_id, "ok": True, "pages": len(content_pages), "hash": h[:12], "ref_excluded": ref_pages}
    except Exception as e:
        return {"doc_id": doc_id, "ok": False, "reason": str(e)}

def scan_reference_leaks(page_text_dir):
    """Post-extraction invariant (v16.13): NO retained non-cover page may still
    contain a reference section. Returns a list of (filename, reason) for each
    violation. JSTOR cover pages (metadata boilerplate) are exempt. A hard guard
    against silently reintroducing the reference-leak bug class — it would have
    caught Kuznets_1973 / Peters_2004 / Koepke&Baten_2005 immediately."""
    import glob as _glob
    leaks = []
    for fp in sorted(_glob.glob(os.path.join(str(page_text_dir), "*_page_*.txt"))):
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
        if _JSTOR_COVER_RE.search(txt):
            continue  # cover metadata, exempt
        n_entries = len(_BIB_ENTRY_LINE_RE.findall(txt))
        if _STANDALONE_REF_HEADER_RE.search(txt) and n_entries >= 3:
            leaks.append((os.path.basename(fp), f"standalone ref header + {n_entries} entry lines"))
        elif n_entries >= 10:
            leaks.append((os.path.basename(fp), f"{n_entries} bib-entry lines (no header)"))
    return leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()-1))
    args = ap.parse_args()
    df = pd.read_csv(args.metadata)
    rows = [r.to_dict() for _, r in df.iterrows()]

    total_ref_excluded = 0
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_process_one, rows, chunksize=1):
            if res.get("ok"):
                ref_note = f" (excl. {res['ref_excluded']} ref pages)" if res.get('ref_excluded', 0) > 0 else ""
                print(f"[ok] {res['doc_id']}: {res['pages']} pages{ref_note}, hash={res['hash']}...")
                total_ref_excluded += res.get('ref_excluded', 0)
            else:
                print(f"[skip] {res['doc_id']}: {res.get('reason')}")

    print(f"[done] preprocessing - excluded {total_ref_excluded} reference pages total")

    # v16.13 invariant: fail loudly if any retained page still leaks a
    # bibliography into the corpus (BM25 must never be fed reference lists).
    leaks = scan_reference_leaks(data_path("page_text"))
    if leaks:
        print(f"[FATAL] reference-leak invariant FAILED — {len(leaks)} retained "
              f"page(s) still contain a bibliography:")
        for fn, why in leaks:
            print(f"    {fn}: {why}")
        raise SystemExit(1)
    print("[invariant] OK - no reference-section leaks in retained page_text.")

if __name__ == "__main__":
    main()
