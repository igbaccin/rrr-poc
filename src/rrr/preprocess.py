import argparse, os, pandas as pd, re
import pymupdf
from rrr.utils import sha256_file, ensure_dir, save_json
from rrr.paths import data_path
from rrr.text import normalize_text, sentence_spans
from multiprocessing import Pool, cpu_count

_REF_HEADER_END = r'\s*[:.]?\s*$'
_REF_HEADERS = [
    r'^\s*REFERENCES' + _REF_HEADER_END,
    r'^\s*References' + _REF_HEADER_END,
    r'^\s*BIBLIOGRAPHY' + _REF_HEADER_END,
    r'^\s*Bibliography' + _REF_HEADER_END,
    r'^\s*Works\s+Cited' + _REF_HEADER_END,
    r'^\s*WORKS\s+CITED' + _REF_HEADER_END,
    r'^\s*Reference\s+List' + _REF_HEADER_END,
    r'^\s*REFERENCE\s+LIST' + _REF_HEADER_END,
    r'^\s*Literature\s+Cited' + _REF_HEADER_END,
    r'^\s*LITERATURE\s+CITED' + _REF_HEADER_END,
    r'^\s*Sources' + _REF_HEADER_END,
    r'^\s*SOURCES' + _REF_HEADER_END,
]

_INLINE_REFERENCES_RE = re.compile(r'REFERENCES(?=[A-Z]|\s|$)', re.IGNORECASE)

_STRONG_INLINE_REFERENCES_RE = re.compile(r'REFERENCES(?=[A-Z]|\s|[:.]|$)')

_JSTOR_COVER_RE = re.compile(r'Linked references are available on JSTOR', re.IGNORECASE)

_BIB_ENTRY_LINE_RE = re.compile(
    r'(?m)^\s*[A-Z][A-Za-z\'.\-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+).{20,}')

_STANDALONE_REF_HEADER_RE = re.compile(
    r'(?m)^\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|WORKS CITED|Works Cited)'
    r'\s*[:.]?\s*$')


def _truncate_at_reference_header(page_text: str) -> tuple:
    earliest = len(page_text)
    found = False
    for pattern in _REF_HEADERS:
        m = re.search(pattern, page_text, re.MULTILINE)
        if m and m.start() < earliest:
            earliest = m.start()
            found = True
    for inline in (_INLINE_REFERENCES_RE, _STRONG_INLINE_REFERENCES_RE):
        m_inline = inline.search(page_text)
        if m_inline and m_inline.start() < earliest:
            earliest = m_inline.start()
            found = True
    if found:
        return page_text[:earliest].rstrip(), True
    return page_text, False


def _column_text_beside_reference_header(page, text_before_header: str) -> str:
    try:
        blocks = [b for b in page.get_text("blocks") if len(b) >= 7 and b[6] == 0]
    except Exception:
        return ""
    heading_index = None
    heading_top = None
    before_parts = []
    for index, block in enumerate(blocks):
        text = block[4]
        match = None
        for pattern in _REF_HEADERS:
            found = re.search(pattern, text, re.MULTILINE)
            if found and (match is None or found.start() < match.start()):
                match = found
        if match is None:
            before_parts.append(text)
            continue
        lead = text[:match.start()]
        before_parts.append(lead)
        lines = max(1, text.count("\n"))
        heading_top = block[1] + (block[3] - block[1]) * (lead.count("\n") / lines)
        heading_index = index
        break
    if heading_index is None:
        return ""
    if normalize_text("".join(before_parts).strip()).rstrip() != text_before_header:
        return ""

    heading_block = blocks[heading_index]
    column_right = heading_block[2]
    for block in blocks[:heading_index]:
        if block[0] <= heading_block[0] + 5 and block[2] > column_right:
            column_right = block[2]
    extra = [
        block[4] for block in blocks[heading_index + 1:]
        if block[3] <= heading_top + 2 and block[0] >= column_right - 5
    ]
    text = normalize_text("".join(extra).strip())
    if len(text) < 200:
        return ""
    if _BIB_ENTRY_LINE_RE.search(text) or _is_reference_dense(text):
        return ""
    return text

def _is_reference_dense(page_text: str) -> bool:
    if len(page_text) < 200:
        return False

    author_year = len(re.findall(r'[A-Z][a-z]+,?\s+[A-Z]\..*?\(\d{4}\)', page_text))
    journals = len(re.findall(r'\b(Journal of|Review of|Quarterly|Economic History|American Economic)\b', page_text, re.IGNORECASE))
    page_ranges = len(re.findall(r'\b\d{1,4}[-–]\d{1,4}\b', page_text))
    publishers = len(re.findall(r'(University Press|Cambridge|Oxford|Princeton|MIT Press|Wiley|Elsevier)', page_text, re.IGNORECASE))
    vol_num = len(re.findall(r'(\bVol\.?\s*\d|\bNo\.?\s*\d|\d+\s*\(\d+\))', page_text))
    editors = len(re.findall(r'\beds?\.', page_text))
    dois = len(re.findall(r'doi:|DOI:', page_text))
    entry_lines = len(_BIB_ENTRY_LINE_RE.findall(page_text))

    total_indicators = (author_year + journals + page_ranges + publishers
                        + vol_num + editors + dois + entry_lines)

    density = total_indicators / (len(page_text) / 500)

    return density > 3

def _is_strong_reference_header(page_text: str) -> bool:
    for pattern in _REF_HEADERS:
        if re.search(pattern, page_text, re.MULTILINE):
            return True
    return bool(_STRONG_INLINE_REFERENCES_RE.search(page_text))


def _is_reference_start_page(pages: list, i: int) -> bool:
    page_text = pages[i]
    n = len(pages)
    if _JSTOR_COVER_RE.search(page_text):
        return False
    if _is_strong_reference_header(page_text):
        if i >= n * 0.5:
            return True
        return _is_reference_dense(page_text) or (
            i + 1 < n and _is_reference_dense(pages[i + 1]))
    if _INLINE_REFERENCES_RE.search(page_text) and _is_reference_dense(page_text):
        return True
    return False


def _find_reference_start(pages: list) -> int:
    n = len(pages)
    for i in range(n):
        if _is_reference_start_page(pages, i):
            return i
    for i in range(n):
        if i > n * 0.6 and _is_reference_dense(pages[i]) and \
                i + 1 < n and _is_reference_dense(pages[i + 1]):
            return i
    return -1

def extract_pages_with_numbers(pdf_path: str):
    with pymupdf.open(pdf_path) as doc:
        raw = [page.get_text() for page in doc]
    pages, numbers = [], []
    for number, text in enumerate(raw):
        if text.strip():
            pages.append(normalize_text(text.strip()))
            numbers.append(number)
    return pages, numbers


def extract_pages(pdf_path: str):
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    pages = [p.strip() for p in pages if p.strip()]
    pages = [normalize_text(p) for p in pages]
    return pages

def _process_one(row_dict):
    pdf = row_dict.get("pdf_path")
    doc_id = str(row_dict.get("doc_id"))

    import glob
    for _pat in (data_path("page_text", f"{doc_id}_page_*.txt"),
                 data_path("page_meta", f"{doc_id}_page_*.json")):
        for old in glob.glob(str(_pat)):
            os.remove(old)

    if not (isinstance(pdf, str) and os.path.isfile(pdf)):
        return {"doc_id": doc_id, "ok": False, "reason": "missing_pdf"}
    try:
        h = sha256_file(pdf)
        pages, page_numbers = extract_pages_with_numbers(pdf)

        ref_start = _find_reference_start(pages)

        if ref_start > 0:
            content_pages = pages[:ref_start]
            ref_pages = len(pages) - ref_start
        else:
            content_pages = pages
            ref_pages = 0

        if ref_start > 0:
            boundary_page = pages[ref_start]
            trimmed, truncated = _truncate_at_reference_header(boundary_page)
            if truncated and trimmed:
                with pymupdf.open(pdf) as doc:
                    beside = _column_text_beside_reference_header(
                        doc[page_numbers[ref_start]], trimmed,
                    )
                if beside:
                    if re.search(r"[A-Za-z]-$", trimmed) and re.match(r"[a-z]", beside):
                        trimmed = trimmed[:-1] + beside
                    else:
                        trimmed = trimmed + "\n" + beside
                content_pages = content_pages + [trimmed]
                ref_pages -= 1

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
    import glob as _glob
    leaks = []
    for fp in sorted(_glob.glob(os.path.join(str(page_text_dir), "*_page_*.txt"))):
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
        if _JSTOR_COVER_RE.search(txt):
            continue
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
