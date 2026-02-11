import argparse, os, pandas as pd, re
from pdfminer.high_level import extract_text
from rrr.utils import sha256_file, ensure_dir, save_json
from multiprocessing import Pool, cpu_count

# Reference section markers (case-insensitive check done separately)
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

def _has_reference_header(page_text: str) -> bool:
    """Check if page contains a reference section header."""
    for pattern in _REF_HEADERS:
        if re.search(pattern, page_text, re.MULTILINE):
            return True
    # Also check for header followed by typical reference formatting
    if re.search(r'\bREFERENCES\b', page_text, re.IGNORECASE):
        return True
    return False

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
    
    total_indicators = author_year + journals + page_ranges + publishers + vol_num + editors + dois
    
    # Calculate density per 500 chars
    density = total_indicators / (len(page_text) / 500)
    
    return density > 3  # More than 3 indicators per 500 chars = likely references

def _find_reference_start(pages: list) -> int:
    """
    Find the page index where references section starts.
    Returns -1 if no reference section detected.
    """
    for i, page_text in enumerate(pages):
        # Check for explicit header
        if _has_reference_header(page_text):
            # Verify it's followed by reference-dense content
            # Check this page or next page
            if _is_reference_dense(page_text):
                return i
            if i + 1 < len(pages) and _is_reference_dense(pages[i + 1]):
                return i
        
        # For pages without header but very high reference density
        # (handles cases where header is at bottom of previous page)
        if i > len(pages) * 0.6:  # Only check last 40% of document
            if _is_reference_dense(page_text):
                # Check if next page is also reference-dense
                if i + 1 < len(pages) and _is_reference_dense(pages[i + 1]):
                    return i
    
    return -1

def extract_pages(pdf_path: str):
    text = extract_text(pdf_path)
    pages = text.split("\x0c")
    pages = [p.strip() for p in pages if p.strip()]
    # clean CID / control characters
    pages = [re.sub(r'\(cid:\d+\)', '', p) for p in pages]
    pages = [re.sub(r'[\x00-\x1f\x7f-\x9f]', '', p) for p in pages]
    return pages

def _process_one(row_dict):
    pdf = row_dict.get("pdf_path")
    doc_id = str(row_dict.get("doc_id"))

    # Clear old pages for this doc before writing new ones
    import glob
    old_pages = glob.glob(f"data/page_text/{doc_id}_page_*.txt")
    for old in old_pages:
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
        
        # Write only content pages
        for i, ptxt in enumerate(content_pages, start=1):
            outp = os.path.join("data/page_text", f"{doc_id}_page_{i}.txt")
            ensure_dir(os.path.dirname(outp))
            with open(outp, "w", encoding="utf-8") as f:
                f.write(ptxt)
        
        meta = {
            "doc_id": doc_id,
            "pdf_path": pdf,
            "hash": h,
            "page_count": len(content_pages),
            "total_pages": len(pages),
            "ref_pages_excluded": ref_pages
        }
        save_json(meta, f"data/{doc_id}.json")
        
        status_note = f" (excl. {ref_pages} ref pages)" if ref_pages > 0 else ""
        return {"doc_id": doc_id, "ok": True, "pages": len(content_pages), "hash": h[:12], "ref_excluded": ref_pages}
    except Exception as e:
        return {"doc_id": doc_id, "ok": False, "reason": str(e)}

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

if __name__ == "__main__":
    main()
