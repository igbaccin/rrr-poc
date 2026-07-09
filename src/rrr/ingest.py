"""Auto-metadata cascade for arbitrary-filename PDF corpora (v15.8 / Improvement #2).

Runs, per PDF:
  1. content_sha1(pdf)         — stable cache key
  2. extract_pages_text        — first N pages via PyMuPDF; skip if <200 chars total (needs OCR)
  3. bib sidecar match         — key match, fuzzy title match, author+year match
  4. filename heuristic        — Author_YYYY pattern → hint only (not final metadata)
  5. DOI regex + CrossRef      — very high confidence when hit
  6. LLM extraction            — Ollama (or configurable) on first ~4000 chars, structured JSON
  7. validation gate           — extracted year AND first-author surname must appear in text
  8. OpenAlex title lookup     — fills venue/volume/DOI when missing (best-effort)
  9. generate_doc_id           — collision-safe Author[EtAl]_YYYY convention (backward compat)

The output is a metadata.csv row that the rest of the pipeline consumes unchanged
(same 11 legacy columns + 5 new v15.8 columns: content_sha1, first_author_surname,
confidence, source, notes).

This module is intentionally standalone — no imports from rrr.writer / rrr.reasoner
so it can be smoke-tested against the existing 50-paper corpus without spinning up
the whole review pipeline.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ExtractedMeta:
    doc_id: str = ""
    title: str = ""
    authors_short: str = ""            # "Acemoglu & Johnson & Robinson"
    author_full: str = ""              # "Acemoglu, Daron; Johnson, Simon; Robinson, James"
    first_author_surname: str = ""
    year: str = ""
    venue: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    doi_or_url: str = ""
    pdf_path: str = ""
    content_sha1: str = ""
    lang: str = "en"                   # v15.12: ISO-639-1 detected from body text
    confidence: str = "low"            # high / medium / low / failed
    source: str = "none"               # bib_sidecar / filename_regex / doi_crossref / llm_extraction / llm+crossref / needs_ocr / none
    notes: str = ""


METADATA_CSV_COLUMNS = [
    "doc_id", "title", "authors", "author_full", "year", "venue", "volume",
    "number", "pages", "doi_or_url", "pdf_path",
    # v15.8 productisation columns:
    "content_sha1", "first_author_surname", "confidence", "source", "notes",
    # v15.12 multilingual: detected body-text language (ISO-639-1).
    "lang",
]


def detect_pdf_language(text: str, default: str = "en") -> str:
    """v15.12: detect the body-text language of a PDF (ISO-639-1). Used by
    the ingest cascade to populate metadata.csv[lang], which the reasoner
    reads to pick the corpus/pivot language for cross-language retrieval.
    Deterministic (seeded). Falls back to ``default`` on short text or when
    langdetect is unavailable.
    """
    body = (text or "").strip()
    if len(body) < 40:
        return default
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(body[:4000])
        return "zh" if lang.startswith("zh") else lang
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def content_sha1(pdf_path: Path) -> str:
    h = hashlib.sha1()
    with pdf_path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_pages_text(pdf_path: Path, max_pages: int = 3) -> str:
    """First N pages concatenated. Returns '' on any error (missing lib,
    corrupt PDF, image-only PDF).
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    parts = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                parts.append(page.get_text())
    except Exception:
        return ""
    return "\n".join(parts)


DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)", re.IGNORECASE)


def find_doi(text: str) -> Optional[str]:
    m = DOI_RE.search(text or "")
    if not m:
        return None
    doi = m.group(1)
    # Trim trailing punctuation the regex tolerated
    while doi and doi[-1] in ".,;:)]}'\"":
        doi = doi[:-1]
    return doi or None


# ---------------------------------------------------------------------------
# CrossRef + OpenAlex
# ---------------------------------------------------------------------------

_UA = "rrr-ingest/0.1 (mailto:rrr@localhost)"


def crossref_lookup(doi: str, timeout: float = 6.0) -> Optional[dict]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def openalex_by_title(title: str, timeout: float = 6.0) -> Optional[dict]:
    if not title or len(title) < 6:
        return None
    q = urllib.parse.quote(title[:200])
    url = f"https://api.openalex.org/works?search={q}&per-page=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", []) or []
        return results[0] if results else None
    except Exception:
        return None


def _crossref_to_fields(cr: dict) -> dict:
    msg = cr.get("message", {}) if isinstance(cr, dict) else {}
    authors = msg.get("author", []) or []
    short_names, full_names, first_surname = [], [], ""
    for a in authors:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if not family:
            continue
        short_names.append(family)
        full_names.append(f"{family}, {given}" if given else family)
        if not first_surname:
            first_surname = family
    title_list = msg.get("title", []) or []
    title = title_list[0].strip() if title_list else ""
    year = ""
    for key in ("issued", "published-print", "published-online"):
        parts = (msg.get(key) or {}).get("date-parts", [])
        if parts and parts[0]:
            year = str(parts[0][0])
            break
    venue_list = msg.get("container-title", []) or []
    venue = venue_list[0].strip() if venue_list else ""
    volume = str(msg.get("volume", "") or "")
    number = str(msg.get("issue", "") or "")
    pages = str(msg.get("page", "") or "").replace("-", "--")
    doi = (msg.get("DOI") or "").strip()
    return {
        "title": title, "year": year, "venue": venue,
        "volume": volume, "number": number, "pages": pages,
        "doi_or_url": doi,
        "authors_short": " & ".join(short_names),
        "author_full": "; ".join(full_names),
        "first_author_surname": first_surname,
    }


# ---------------------------------------------------------------------------
# Bib sidecar
# ---------------------------------------------------------------------------

def parse_bib_sidecar(bib_path: Optional[Path]) -> list:
    if not bib_path or not bib_path.is_file():
        return []
    try:
        from pybtex.database import parse_file
        db = parse_file(str(bib_path))
    except Exception:
        return []
    entries = []
    for key, entry in db.entries.items():
        fields = entry.fields
        persons = entry.persons.get("author", []) or []
        short_names, full_names = [], []
        for p in persons:
            first = " ".join(p.first_names)
            last = " ".join(p.last_names)
            if last:
                short_names.append(last)
                full_names.append(f"{last}, {first}" if first else last)
        entries.append({
            "key": key,
            "title": (fields.get("title") or "").strip("{} "),
            "year": (fields.get("year") or "").strip(),
            "venue": (fields.get("journal") or fields.get("booktitle") or "").strip(),
            "volume": (fields.get("volume") or "").strip(),
            "number": (fields.get("number") or "").strip(),
            "pages": (fields.get("pages") or "").strip(),
            "doi": (fields.get("doi") or "").strip(),
            "authors_short": " & ".join(short_names),
            "author_full": "; ".join(full_names),
            "first_surname": short_names[0] if short_names else "",
        })
    return entries


def _fuzzy_title_match(title: str, text: str, threshold: float = 0.9) -> bool:
    """STRICT title match: require BOTH signals to be present:
      (a) >= threshold fraction of 5+ char title tokens appear as WHOLE WORDS
          (\\b-anchored) in the lowercased text — no substring inflation.
      (b) at least one 4-word verbatim chunk from the title appears as a
          contiguous substring of the (lowercased, whitespace-collapsed) text.

    The old version used 4+ char tokens with substring matching and threshold
    0.75 — that produced false positives like matching every economics paper
    to whichever bib entry had the most-common vocabulary. This one is
    conservative on purpose; when it says True, it's an actual match.
    """
    if not title or not text:
        return False
    title_lower = re.sub(r"\s+", " ", title.lower())
    text_lower = re.sub(r"\s+", " ", text.lower())
    tokens = re.findall(r"[a-z0-9]{5,}", title_lower)
    if len(tokens) < 3:
        return False
    hits = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text_lower))
    if (hits / len(tokens)) < threshold:
        return False
    # 4-word verbatim chunk
    words = re.findall(r"[a-z0-9]+", title_lower)
    for i in range(len(words) - 3):
        chunk = " ".join(words[i:i + 4])
        if chunk in text_lower:
            return True
    return False


def match_bib_entry(pdf_path: Path, text: str, sidecar: list) -> Optional[dict]:
    """Match a PDF to a bib entry with a strict cascade:
      1. Exact filename-stem == bib key
      2. Parse filename via Author_YYYY convention → find bib entry with same
         first-author surname AND same year (this is the load-bearing path
         for corpora that already follow Author_YYYY naming).
      3. Strict fuzzy title: high threshold + verbatim 4-word chunk +
         first-surname + year both present in text (guards against generic
         economics vocabulary inflating the match).
    Returns the matching entry dict, or None.
    """
    stem = pdf_path.stem
    stem_lower = stem.lower()

    # 1) exact key match
    for e in sidecar:
        if e["key"].lower() == stem_lower:
            return e

    # 2) filename Author_YYYY → bib first-surname + year
    fn = parse_filename_heuristic(pdf_path)
    if fn:
        fn_year = fn["year"]
        # normalise filename name-part to lowercase, drop EtAl / & markers to
        # match against bib first-surname
        fn_name = fn["name"].lower()
        fn_name_first = re.sub(r"etal.*|&.*", "", fn_name)  # 'acemogluetal' → 'acemoglu'
        candidates = []
        for e in sidecar:
            surname = (e.get("first_surname") or "").lower()
            year = (e.get("year") or "").strip()
            if not surname or year != fn_year:
                continue
            # match if filename first-name starts with surname, OR surname
            # contains the filename name (handles hyphenated / camelCase)
            if fn_name_first.startswith(surname) or surname.startswith(fn_name_first):
                candidates.append(e)
        # If exactly one candidate, take it. If several, disambiguate via
        # verbatim title chunk match against text.
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for e in candidates:
                if _fuzzy_title_match(e["title"], text, threshold=0.85):
                    return e
            # Ambiguous — return the first (keeps behaviour deterministic
            # even if not ideal; the downstream doc_id collision logic will
            # surface duplicates).
            return candidates[0]

    # 3) strict fuzzy title (surname + year in text AND title in text)
    text_lower = text.lower()
    for e in sidecar:
        surname = (e.get("first_surname") or "").lower()
        year = (e.get("year") or "").strip()
        if not surname or not year:
            continue
        if surname not in text_lower or year not in text:
            continue
        if _fuzzy_title_match(e["title"], text, threshold=0.9):
            return e
    return None


# ---------------------------------------------------------------------------
# Filename heuristic
# ---------------------------------------------------------------------------

_FILENAME_AUTHOR_YEAR_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9&_.\-]*?)_(?P<year>\d{4})[a-z]?$"
)


def parse_filename_heuristic(pdf_path: Path) -> Optional[dict]:
    """Detect Author_YYYY / AuthorEtAl_YYYY / Author1&Author2_YYYY conventions.
    Returns {'name': ..., 'year': ...} or None."""
    stem = pdf_path.stem
    m = _FILENAME_AUTHOR_YEAR_RE.match(stem)
    if not m:
        return None
    return {"name": m.group("name"), "year": m.group("year")}


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are extracting bibliographic metadata from the first pages of an academic paper.

FIRST PAGES TEXT (may include cover pages):
---
{text}
---

Extract these fields, using ONLY what is clearly present in the text. If a field cannot be identified with high confidence, use null (do NOT guess).

Return ONE JSON object with EXACTLY these keys:
  "authors": ["Surname, Given" for each author, in the order they appear; use empty list if unknown]
  "year": "YYYY" (4-digit publication year present in the text) or null
  "title": "..." (the paper title as it appears) or null
  "doi": "..." (if a DOI is visible) or null

Return ONLY the JSON object. No commentary."""


def llm_extract_metadata(text: str, model: Optional[str] = None,
                          timeout: float = 60.0) -> Optional[dict]:
    """Call the configured LLM (Ollama) to extract metadata. Returns the
    parsed dict or None on any error. Model defaults to
    RRR_INGEST_MODEL env or 'mistral-small:24b'.
    """
    if not text or len(text.strip()) < 50:
        return None
    try:
        import ollama
    except Exception:
        return None
    model = model or os.environ.get("RRR_INGEST_MODEL", "mistral-small:24b")
    text_slice = text[:4000]
    prompt = _LLM_PROMPT.format(text=text_slice)
    try:
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 500},
            keep_alive="30m",
            stream=False,
        )
        raw = (res.get("message") or {}).get("content", "")
        i = raw.find("{")
        j = raw.rfind("}")
        if i < 0 or j <= i:
            return None
        return json.loads(raw[i:j + 1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(authors: list, year: str, text: str) -> tuple:
    """Returns (is_valid, reasons_list). Requires:
       - year is a 4-digit string and appears in text
       - first author's surname appears in text
    """
    reasons = []
    year_s = str(year or "").strip()
    if not year_s or not re.match(r"^\d{4}$", year_s):
        reasons.append("no_valid_year")
    elif year_s not in text:
        reasons.append("year_not_in_text")
    if not authors:
        reasons.append("no_authors")
    else:
        first = str(authors[0] or "")
        surname = first.split(",")[0].strip() if "," in first else (first.strip().split()[-1] if first.strip() else "")
        if not surname:
            reasons.append("no_first_surname")
        elif surname.lower() not in text.lower():
            reasons.append("first_surname_not_in_text")
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# doc_id generation
# ---------------------------------------------------------------------------

_LATIN_PARTICLES = {"van", "von", "de", "del", "der", "la", "le", "du", "di", "da"}


def _clean_surname_token(s: str) -> str:
    """Strip non-letters, drop 'et al' markers, lowercase-known-particles.

    v15.8.1: bib may pass 'Van Zanden' (capital V because BibTeX preserved
    case); the RRR doc_id convention uses lowercase particles: 'vanZanden'.
    We split on whitespace, lowercase any token in _LATIN_PARTICLES, then
    concatenate. 'Van Zanden' → 'vanZanden'; 'de La Croix' → 'delaCroix'.
    """
    if not s:
        return ""
    s = re.sub(r"\bet\s+al\.?", "", s, flags=re.IGNORECASE)
    parts = re.split(r"\s+", s.strip())
    out = []
    for part in parts:
        clean = re.sub(r"[^A-Za-z]", "", part)
        if not clean:
            continue
        if clean.lower() in _LATIN_PARTICLES:
            out.append(clean.lower())
        else:
            out.append(clean)
    return "".join(out)


def generate_doc_id(surnames: list, year: str, existing: set = None) -> str:
    existing = existing or set()
    year = str(year or "").strip()
    if not year or not re.match(r"^\d{4}[a-z]?$", year):
        return ""
    clean = [_clean_surname_token(s) for s in surnames]
    clean = [c for c in clean if c]
    if not clean:
        return ""
    if len(clean) == 1:
        base = f"{clean[0]}_{year}"
        stem = f"{clean[0]}_{year}"
    elif len(clean) == 2:
        base = f"{clean[0]}&{clean[1]}_{year}"
        stem = f"{clean[0]}&{clean[1]}_{year}"
    else:
        base = f"{clean[0]}EtAl_{year}"
        stem = f"{clean[0]}EtAl_{year}"
    if base not in existing:
        return base
    for suffix in "abcdefghijklmnop":
        cand = stem.replace(f"_{year}", f"_{year}{suffix}")
        if cand not in existing:
            return cand
    n = 1
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

def _fill_from_bib(meta: ExtractedMeta, entry: dict) -> None:
    meta.title = entry["title"]
    meta.year = entry["year"]
    meta.venue = entry["venue"]
    meta.volume = entry["volume"]
    meta.number = entry["number"]
    meta.pages = entry["pages"]
    meta.doi_or_url = entry["doi"]
    meta.authors_short = entry["authors_short"]
    meta.author_full = entry["author_full"]
    meta.first_author_surname = entry["first_surname"]


def _fill_from_dict(meta: ExtractedMeta, d: dict) -> None:
    for f in ("title", "year", "venue", "volume", "number", "pages",
              "doi_or_url", "authors_short", "author_full", "first_author_surname"):
        v = d.get(f)
        if v:
            setattr(meta, f, v)


def _surnames_from_meta(meta: ExtractedMeta) -> list:
    if not meta.authors_short:
        return []
    return [s.strip() for s in meta.authors_short.split("&") if s.strip()]


def _bib_surnames_list(entry: dict) -> list:
    return [s.strip() for s in (entry.get("authors_short") or "").split("&") if s.strip()]


def cascade(pdf_path: Path,
             sidecar_bib_entries: Optional[list] = None,
             existing_doc_ids: Optional[set] = None,
             use_llm: bool = True,
             use_crossref: bool = True,
             use_openalex: bool = True,
             llm_model: Optional[str] = None) -> ExtractedMeta:
    """Run the full cascade on ONE PDF. Returns an ExtractedMeta."""
    existing_doc_ids = set(existing_doc_ids or ())
    meta = ExtractedMeta(pdf_path=str(pdf_path.resolve()))

    # Step 1: content hash (also basic sanity check)
    try:
        meta.content_sha1 = content_sha1(pdf_path)
    except Exception as e:
        meta.confidence = "failed"
        meta.source = "none"
        meta.notes = f"content_sha1_failed:{e}"
        return meta

    # Step 2: page text
    text = extract_pages_text(pdf_path, max_pages=3)
    if len(text.strip()) < 200:
        meta.confidence = "low"
        meta.source = "needs_ocr"
        meta.notes = f"page_text_only_{len(text.strip())}_chars"
        meta.lang = detect_pdf_language(text)
        return meta

    # v15.12: detect body-text language once, before any early-return path so
    # every cascade outcome (bib_sidecar / crossref / llm / etc.) carries it.
    meta.lang = detect_pdf_language(text)

    # Step 3: bib sidecar
    if sidecar_bib_entries:
        entry = match_bib_entry(pdf_path, text, sidecar_bib_entries)
        if entry:
            _fill_from_bib(meta, entry)
            meta.source = "bib_sidecar"
            meta.confidence = "high"
            meta.doc_id = generate_doc_id(_bib_surnames_list(entry), entry["year"], existing_doc_ids)
            return meta

    # Step 4: filename heuristic — hint only, not final metadata
    fn = parse_filename_heuristic(pdf_path)
    if fn:
        meta.notes = f"filename_hint:{fn['name']}_{fn['year']}"

    # Step 5: DOI + CrossRef
    doi = find_doi(text)
    if doi and use_crossref:
        cr = crossref_lookup(doi)
        if cr:
            fields = _crossref_to_fields(cr)
            _fill_from_dict(meta, fields)
            meta.source = "doi_crossref"
            valid, reasons = validate_extraction(
                [meta.first_author_surname] if meta.first_author_surname else [],
                meta.year, text,
            )
            meta.confidence = "high" if valid else "medium"
            if reasons:
                meta.notes = ("crossref_" + ",".join(reasons))
            meta.doc_id = generate_doc_id(_surnames_from_meta(meta), meta.year, existing_doc_ids)
            return meta

    # Step 6: LLM extraction
    if use_llm:
        llm = llm_extract_metadata(text, model=llm_model)
        if llm and (llm.get("authors") or llm.get("year")):
            authors = llm.get("authors") or []
            year = str(llm.get("year") or "").strip()
            title = str(llm.get("title") or "").strip()
            doi = str(llm.get("doi") or "").strip()

            short_names, full_names, first_surname = [], [], ""
            for a in authors:
                a_str = str(a or "").strip()
                if not a_str:
                    continue
                surname = a_str.split(",")[0].strip() if "," in a_str else (
                    a_str.strip().split()[-1] if a_str.strip() else "")
                if not first_surname and surname:
                    first_surname = surname
                short_names.append(surname)
                full_names.append(a_str)
            meta.title = title
            meta.year = year
            meta.authors_short = " & ".join(short_names)
            meta.author_full = "; ".join(full_names)
            meta.first_author_surname = first_surname
            meta.doi_or_url = doi
            meta.source = "llm_extraction"

            valid, reasons = validate_extraction(authors, year, text)
            meta.confidence = "high" if valid else "medium"
            if reasons:
                meta.notes = ((meta.notes + "; ") if meta.notes else "") + "llm_" + ",".join(reasons)

            # Optional CrossRef upgrade if LLM gave a DOI
            if doi and use_crossref and valid:
                cr = crossref_lookup(doi)
                if cr:
                    fields = _crossref_to_fields(cr)
                    for f in ("venue", "volume", "number", "pages"):
                        if not getattr(meta, f):
                            setattr(meta, f, fields.get(f, ""))
                    if fields.get("title") and not meta.title:
                        meta.title = fields["title"]
                    meta.source = "llm+crossref"

            # Optional OpenAlex title upgrade if venue still missing
            if valid and use_openalex and not meta.venue and meta.title:
                oa = openalex_by_title(meta.title)
                if oa:
                    if not meta.venue:
                        meta.venue = ((oa.get("host_venue") or {}).get("display_name") or
                                      (oa.get("primary_location") or {}).get("source", {}).get("display_name", "")) or ""
                    if not meta.doi_or_url and oa.get("doi"):
                        meta.doi_or_url = (oa.get("doi") or "").replace("https://doi.org/", "")
                    if meta.source == "llm_extraction":
                        meta.source = "llm+openalex"

            meta.doc_id = generate_doc_id(short_names, year, existing_doc_ids)
            return meta

    # All extractors failed
    meta.confidence = "failed"
    meta.source = "none"
    return meta


# ---------------------------------------------------------------------------
# Corpus-wide ingest
# ---------------------------------------------------------------------------

def ingest_corpus(corpus_dir: Path,
                   output_csv: Optional[Path] = None,
                   sidecar_bib: Optional[Path] = None,
                   use_llm: bool = True,
                   use_crossref: bool = True,
                   use_openalex: bool = True,
                   llm_model: Optional[str] = None,
                   progress: bool = True) -> list:
    """Walk corpus_dir for *.pdf; run cascade on each; write metadata.csv.
    Returns list of ExtractedMeta.
    """
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if progress:
        print(f"[ingest] found {len(pdfs)} PDFs in {corpus_dir}")
    sidecar = parse_bib_sidecar(sidecar_bib) if sidecar_bib else []
    if progress and sidecar_bib:
        print(f"[ingest] loaded {len(sidecar)} entries from {sidecar_bib}")
    existing_doc_ids = set()
    results = []
    for pdf in pdfs:
        t0 = time.perf_counter()
        meta = cascade(
            pdf,
            sidecar_bib_entries=sidecar,
            existing_doc_ids=existing_doc_ids,
            use_llm=use_llm,
            use_crossref=use_crossref,
            use_openalex=use_openalex,
            llm_model=llm_model,
        )
        results.append(meta)
        if meta.doc_id:
            existing_doc_ids.add(meta.doc_id)
        dt = time.perf_counter() - t0
        if progress:
            print(f"  [{dt:5.1f}s] {pdf.name:44s}  →  doc_id={meta.doc_id!r:34s} "
                  f"src={meta.source} conf={meta.confidence}"
                  + (f"  notes={meta.notes}" if meta.notes else ""))
    if output_csv:
        _write_metadata_csv(results, output_csv)
        if progress:
            print(f"[ingest] wrote {len(results)} rows to {output_csv}")
    return results


def _write_metadata_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "doc_id": r.doc_id,
                "title": r.title,
                "authors": r.authors_short,
                "author_full": r.author_full,
                "year": r.year,
                "venue": r.venue,
                "volume": r.volume,
                "number": r.number,
                "pages": r.pages,
                "doi_or_url": r.doi_or_url,
                "pdf_path": r.pdf_path,
                "content_sha1": r.content_sha1,
                "first_author_surname": r.first_author_surname,
                "confidence": r.confidence,
                "source": r.source,
                "notes": r.notes,
                "lang": r.lang,
            })
