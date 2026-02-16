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

CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

# E3 patterns: citation-like strings that fail the strict format
_E3_PATTERNS = [
    re.compile(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)"),              # (Author_Year) no page
    re.compile(r"[A-Z][a-z]+\s+et\s+al\.?\s*\(\d{4}\)"),           # Author et al. (Year)
    re.compile(r"\([A-Z][a-z]+,\s*\d{4}\)"),                        # (Author, Year)
    re.compile(r"\([A-Za-z0-9_&.\-]+:\s*p\.\d+\s*,\s*p\.\d+\)"),  # multi-page in one cite
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
            e1_details.append({"doc_id": did, "page": page, "context": ctx})
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

    # E3: loose citation patterns outside strict format
    e3 = 0
    strict_spans = [(c["start"], c["end"]) for c in citations]
    for pat in _E3_PATTERNS:
        for m in pat.finditer(text):
            if not any(s <= m.start() < e for s, e in strict_spans):
                e3 += 1

    word_count = len(re.findall(r"\b\w+\b", text))

    return {
        "n_citations": len(citations),
        "e1": e1, "e2": e2, "e3": e3,
        "docs_cited": sorted(docs_cited),
        "n_docs_cited": len(docs_cited),
        "word_count": word_count,
        "e1_details": e1_details,
        "e2_details": e2_details,
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
                print(f"  {d['doc_id']}: p.{d['page']}")
        if result["e2_details"]:
            print("\nE2 details:")
            for d in result["e2_details"]:
                print(f"  {d['doc_id']}: cited p.{d['cited_page']}, "
                      f"max p.{d['max_valid_page']} (+{d['overshoot']})")
