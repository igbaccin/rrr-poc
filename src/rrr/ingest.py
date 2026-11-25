from __future__ import annotations

import codecs
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rrr.llm import install as _install_llm_shim


_install_llm_shim()


@dataclass
class ExtractedMeta:
    doc_id: str = ""
    title: str = ""
    authors_short: str = ""
    author_full: str = ""
    first_author_surname: str = ""
    year: str = ""
    venue: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    doi_or_url: str = ""
    pdf_path: str = ""
    content_sha1: str = ""
    lang: str = "en"
    confidence: str = "low"
    source: str = "none"
    notes: str = ""
    display_label: str = ""
    surnames: list = field(default_factory=list)
    bib_key: str = ""
    year_letter_source: str = ""


METADATA_CSV_COLUMNS = [
    "doc_id", "title", "authors", "author_full", "year", "venue", "volume",
    "number", "pages", "doi_or_url", "pdf_path",
    "content_sha1", "first_author_surname", "confidence", "source", "notes",
    "lang",
    "display_label",
]


def detect_pdf_language(text: str, default: str = "en") -> str:
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
    try:
        import pymupdf
    except Exception:
        return ""
    parts = []
    try:
        with pymupdf.open(pdf_path) as doc:
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
    while doi and doi[-1] in ".,;:)]}'\"":
        doi = doi[:-1]
    return doi or None


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


_LATEX_ACCENTS = {
    "'": "\N{COMBINING ACUTE ACCENT}",
    "`": "\N{COMBINING GRAVE ACCENT}",
    '"': "\N{COMBINING DIAERESIS}",
    "^": "\N{COMBINING CIRCUMFLEX ACCENT}",
    "~": "\N{COMBINING TILDE}",
    "c": "\N{COMBINING CEDILLA}",
    "v": "\N{COMBINING CARON}",
}
_LATEX_ACCENT_RE = re.compile(
    r"\\(?P<command>['`\"^~cv])\s*"
    r"(?:\{\s*\\?(?P<braced>[A-Za-z\u0131])\s*\}|(?P<bare>[A-Za-z]))"
)


def _bibtex_plain(value) -> str:
    text = str(value or "")
    if "\\" in text:
        try:
            import latexcodec
            text = codecs.decode(text, "ulatex")
        except Exception:
            pass

        def accent(match: re.Match) -> str:
            letter = match.group("braced") or match.group("bare")
            letter = "i" if letter == "\u0131" else letter
            return unicodedata.normalize(
                "NFC", letter + _LATEX_ACCENTS[match.group("command")]
            )

        text = _LATEX_ACCENT_RE.sub(accent, text)
        text = re.sub(r"\\ss\b", "ß", text)
        text = re.sub(r"\\([&%$#_])", r"\1", text)
        text = re.sub(r"\\[A-Za-z]+\s*", "", text)
        text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _person_surname(person) -> str:
    parts = list(person.prelast_names) + list(person.last_names)
    return _bibtex_plain(" ".join(parts))


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
            first = _bibtex_plain(" ".join(p.first_names))
            last = _person_surname(p)
            if last:
                short_names.append(last)
                full_names.append(f"{last}, {first}" if first else last)
        entries.append({
            "key": key,
            "title": _bibtex_plain(fields.get("title")),
            "year": (fields.get("year") or "").strip(),
            "venue": _bibtex_plain(fields.get("journal") or fields.get("booktitle")),
            "volume": (fields.get("volume") or "").strip(),
            "number": (fields.get("number") or "").strip(),
            "pages": (fields.get("pages") or "").strip(),
            "doi": (fields.get("doi") or "").strip(),
            "authors_short": " & ".join(short_names),
            "author_full": "; ".join(full_names),
            "first_surname": short_names[0] if short_names else "",
            "surnames": list(short_names),
            "files": _bib_file_names(fields.get("file")),
        })
    return entries


def _match_key(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii_fold(value).casefold())


def _file_name_key(name) -> str:
    return unicodedata.normalize("NFC", str(name or "")).casefold().strip()


def _bib_file_names(value) -> list:
    names = []
    for part in re.split(r"(?<!\\);", str(value or "")):
        text = part.strip()
        if not text:
            continue
        if "\\\\" in text or "\\:" in text:
            text = text.replace("\\\\", "/")
            text = re.sub(r"\\(.)", r"\1", text)
        else:
            text = text.replace("\\", "/")
        last = None
        for last in re.finditer(r"\.pdf", text, flags=re.IGNORECASE):
            pass
        if last is None:
            continue
        base = re.split(r"[/:]", text[:last.end()])[-1].strip()
        if base and base not in names:
            names.append(base)
    return names


def _year_parts(value) -> tuple:
    match = re.fullmatch(r"\s*(\d{4})([a-z]?)\s*", str(value or ""))
    return (match.group(1), match.group(2)) if match else ("", "")


def _entry_letter(entry: dict) -> str:
    year, letter = _year_parts(entry.get("year"))
    if letter or not year:
        return letter
    key = re.search(rf"(?<!\d){year}([a-z])$", str(entry.get("key") or ""))
    return key.group(1) if key else ""


def _match_text(value) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _ascii_fold(value).lower()))


def _fuzzy_title_match(title: str, text: str, threshold: float = 0.9) -> bool:
    if not title or not text:
        return False
    title_words = _match_text(title).split()
    text_plain = _match_text(text)
    return _title_words_match(title_words, text_plain, set(text_plain.split()), threshold)


def _title_words_match(title_words: list, text_plain: str, text_words: set,
                       threshold: float) -> bool:
    tokens = [word for word in title_words if len(word) >= 5]
    if len(tokens) < 3:
        return False
    hits = sum(1 for token in tokens if token in text_words)
    if (hits / len(tokens)) < threshold:
        return False
    padded = f" {text_plain} "
    for i in range(len(title_words) - 3):
        if f" {' '.join(title_words[i:i + 4])} " in padded:
            return True
    return False


@dataclass
class BibMatch:
    entry: Optional[dict] = None
    route: str = ""
    note: str = ""
    ambiguous: bool = False


def _entry_fits_file_name(entry: dict, parsed: dict) -> str:
    year, _ = _year_parts(entry.get("year"))
    if not year or year != parsed["year"]:
        return ""
    first = _match_key(entry.get("first_surname"))
    if not first:
        return ""
    if _match_key(parsed["whole"]) == first:
        return "exact"
    if not parsed["surnames"]:
        return ""
    name = _match_key(parsed["surnames"][0])
    if not name:
        return ""
    without_particles = _match_key(" ".join(
        word for word in str(entry.get("first_surname") or "").split()
        if word.lower() not in _LATIN_PARTICLES
    ))
    if name in (first, without_particles):
        return "exact"
    if name.startswith(first) or first.startswith(name):
        return "loose"
    return ""


def _narrow_by_co_authors(candidates: list, parsed: dict) -> list:
    wanted = [_match_key(name) for name in parsed["surnames"]]

    def agrees(entry: dict) -> bool:
        surnames = [_match_key(name) for name in entry.get("surnames") or []]
        if not surnames:
            return False
        if parsed["et_al"]:
            return len(surnames) >= 3
        if len(wanted) >= 2:
            return surnames[1:len(wanted)] == wanted[1:]
        return len(surnames) == 1

    agreeing = [entry for entry in candidates if agrees(entry)]
    return agreeing or candidates


def match_bib_entry_detailed(pdf_path: Path, text: str, sidecar: list) -> BibMatch:
    name_key = _file_name_key(pdf_path.name)
    text_plain = _match_text(text)
    text_words = set(text_plain.split())
    padded = f" {text_plain} "

    listed = [
        entry for entry in sidecar
        if any(_file_name_key(name) == name_key for name in entry.get("files") or [])
    ]
    if len(listed) == 1:
        return BibMatch(entry=listed[0], route="file_field")
    shared_file_note = ""
    if len(listed) > 1:
        shared_file_note = (
            "file field: " + ", ".join(entry["key"] for entry in listed)
            + " all name this file"
        )

    stem_key = _file_name_key(pdf_path.stem)
    for entry in sidecar:
        if _file_name_key(entry["key"]) == stem_key:
            return BibMatch(entry=entry, route="key")

    notes = [shared_file_note] if shared_file_note else []
    readings = parse_filename_readings(pdf_path)
    ambiguous = None
    for parsed in readings:
        fits = [(entry, _entry_fits_file_name(entry, parsed)) for entry in sidecar]
        candidates = [entry for entry, fit in fits if fit == "exact"]
        if not candidates:
            candidates = [
                entry for entry, fit in fits
                if fit == "loose"
                and (
                    f" {_match_text(entry.get('first_surname'))} " in padded
                    or (parsed["letter"] and _entry_letter(entry) == parsed["letter"])
                )
            ]
        if len(candidates) > 1:
            candidates = _narrow_by_co_authors(candidates, parsed)
        if len(candidates) > 1:
            by_letter = [
                entry for entry in candidates
                if parsed["letter"] and _entry_letter(entry) == parsed["letter"]
            ]
            if by_letter:
                candidates = by_letter
        if len(candidates) > 1:
            by_title = [
                entry for entry in candidates
                if _fuzzy_title_match(entry["title"], text, threshold=0.85)
            ]
            if len(by_title) == 1:
                candidates = by_title
        if len(candidates) == 1:
            return BibMatch(entry=candidates[0], route="file_name")
        if len(candidates) > 1 and ambiguous is None:
            who = parsed["whole"] + (" et al." if parsed["et_al"] else "")
            keys = ", ".join(entry["key"] for entry in candidates)
            ambiguous = BibMatch(
                ambiguous=True,
                note=(
                    f"bibliography: {len(candidates)} entries fit the file name "
                    f"({who}, {parsed['year']}{parsed['letter']}): {keys}. Add "
                    "the year letter to the file name or a file field to the entry"
                ),
            )
    if ambiguous is not None:
        return ambiguous
    if readings:
        first = readings[0]
        who = first["whole"] + (" et al." if first["et_al"] else "")
        notes.append(
            f"file name read as {who} ({first['year']}{first['letter']}), and no "
            "bibliography entry has that first author and year"
        )
    else:
        notes.append("file name has no author-and-year form")

    for entry in sidecar:
        surname = _match_text(entry.get("first_surname"))
        year, _ = _year_parts(entry.get("year"))
        if not surname or not year:
            continue
        if f" {surname} " not in padded or year not in text_words:
            continue
        if _title_words_match(_match_text(entry["title"]).split(), text_plain,
                              text_words, 0.9):
            return BibMatch(entry=entry, route="title")
    notes.append("no bibliography title found on the first pages")
    return BibMatch(note="bibliography: " + "; ".join(notes))


def match_bib_entry(pdf_path: Path, text: str, sidecar: list) -> Optional[dict]:
    return match_bib_entry_detailed(pdf_path, text, sidecar).entry


_FILENAME_AUTHOR_YEAR_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9&_.\-]*?)_(?P<year>\d{4})[a-z]?$"
)

_FILENAME_YEAR = r"(?P<year>1[4-9]\d\d|20\d\d)(?P<letter>[a-z]?)"
_FILENAME_DASH = r"\s+[-\u2013\u2014]\s+"
_FILENAME_FORMS = (
    re.compile(r"^(?P<names>.*?[^\W\d_].*?)\s*[(\[]\s*" + _FILENAME_YEAR + r"\s*[)\]]\s*$"),
    re.compile(r"^(?P<names>.*?[^\W\d_].*?)" + _FILENAME_DASH + _FILENAME_YEAR
               + _FILENAME_DASH + r".+$"),
    re.compile(r"^(?P<names>.*?[^\W\d_].*?)\s*[(\[]\s*" + _FILENAME_YEAR
               + r"\s*[)\]]\s*[-\u2013\u2014_,.:]?\s*\S.*$"),
    re.compile(r"^(?P<names>.*?[^\W\d_].*?)[\s_\-,.]+" + _FILENAME_YEAR + r"\s*$"),
    re.compile(r"^(?P<names>[^\W\d_]+(?:[&\-][^\W\d_]+)*)" + _FILENAME_YEAR + r"$"),
    re.compile(r"^" + _FILENAME_YEAR + r"\s*[-\u2013\u2014_ ]\s*"
               r"(?P<names>[^\W\d_][^\-\u2013\u2014_]*?)"
               r"(?:\s*[-\u2013\u2014_]\s*.+)?$"),
)
_FILENAME_ET_AL_RE = re.compile(
    r"(?:(?i:[\s_,]+et[\s_.]*al\.?)|(?<=[a-z])EtAl)\s*$"
)


def parse_filename_readings(pdf_path: Path) -> list:
    stem = unicodedata.normalize("NFC", Path(pdf_path).stem).strip()
    legacy = _FILENAME_AUTHOR_YEAR_RE.match(stem)
    readings = []
    seen = set()
    for form in _FILENAME_FORMS:
        match = form.match(stem)
        if not match:
            continue
        raw = match.group("names").strip().strip(",;").strip()
        if not raw:
            continue
        names = raw
        et_al = bool(_FILENAME_ET_AL_RE.search(names))
        if et_al:
            names = _FILENAME_ET_AL_RE.sub("", names).strip().strip(",;").strip()
        whole = re.sub(r"[_\s]+", " ", names).strip()
        surnames = [
            part.strip()
            for part in re.split(r"\s*&\s*|\s+and\s+|\s*,\s*|\s*\+\s*", whole)
            if part.strip()
        ]
        key = (whole.casefold(), match.group("year"), match.group("letter"), et_al)
        if not whole or key in seen:
            continue
        seen.add(key)
        readings.append({
            "name": legacy.group("name") if legacy else raw,
            "year": match.group("year"),
            "letter": match.group("letter"),
            "whole": whole,
            "surnames": surnames,
            "et_al": et_al,
        })
    return readings


def parse_filename_heuristic(pdf_path: Path) -> Optional[dict]:
    readings = parse_filename_readings(pdf_path)
    return readings[0] if readings else None


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


def validate_extraction(authors: list, year: str, text: str) -> tuple:
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


from rrr.render import NAME_PARTICLES as _RENDER_PARTICLES

_LATIN_PARTICLES = set(_RENDER_PARTICLES)

_ASCII_SPELLINGS = {
    "ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th",
    "Þ": "Th", "ı": "i",
}


def _ascii_fold(s: str) -> str:
    spelled = "".join(_ASCII_SPELLINGS.get(ch, ch) for ch in str(s or ""))
    decomposed = unicodedata.normalize("NFKD", spelled)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _clean_surname_token(s: str) -> str:
    if not s:
        return ""
    s = _ascii_fold(s)
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


_YEAR_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_DOC_ID_PARTS_RE = re.compile(r"^(?P<name>.+)_(?P<year>\d{4})(?P<letter>[a-z]?)$")


def _year_letter_hint(meta: ExtractedMeta, year: str) -> tuple:
    cell = re.fullmatch(r"\s*(\d{4})([a-z])\s*", str(meta.year or ""))
    if cell and cell.group(1) == year:
        return cell.group(2), "bibliography year"
    key = re.search(rf"(?<!\d){year}([a-z])$", str(meta.bib_key or ""))
    if key:
        return key.group(1), "bibliography key"
    stem = Path(str(meta.pdf_path or "")).stem
    name = re.search(rf"(?<!\d){year}([a-z])(?![A-Za-z0-9])", stem)
    if name:
        return name.group(1), "file name"
    return "", ""


def assign_year_letters(rows: list) -> list:
    from rrr.render import _regex_doc_id_to_author_label

    groups: dict = {}
    for meta in rows:
        parts = _DOC_ID_PARTS_RE.match(str(getattr(meta, "doc_id", "") or ""))
        if not parts:
            continue
        key = (parts.group("name").casefold(), parts.group("year"))
        groups.setdefault(key, []).append((meta, parts.group("name")))

    report = []
    for (_, year), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda item: (
            Path(str(item[0].pdf_path or "")).name.casefold(),
            str(item[0].content_sha1 or ""),
        ))
        name = members[0][1]
        taken: dict = {}
        pending = []
        for meta, _ in members:
            letter, source = _year_letter_hint(meta, year)
            if letter and letter not in taken:
                taken[letter] = (meta, source)
            else:
                pending.append((meta, letter))
        free = [letter for letter in _YEAR_LETTERS if letter not in taken]
        for meta, wanted in pending:
            if not free:
                meta.notes = _join_notes(meta.notes, "same_author_year:no_free_letter")
                continue
            letter = free.pop(0)
            source = "assigned" if not wanted else f"assigned ({wanted} already in use)"
            taken[letter] = (meta, source)
        record = {
            "label": _regex_doc_id_to_author_label(f"{name}_{year}"),
            "members": [],
        }
        for letter in sorted(taken):
            meta, source = taken[letter]
            meta.year = f"{year}{letter}"
            meta.doc_id = f"{name}_{year}{letter}"
            meta.year_letter_source = source
            record["members"].append({
                "doc_id": meta.doc_id,
                "pdf": meta.pdf_path,
                "year": meta.year,
                "letter_source": source,
            })
        report.append(record)
    return report


def year_letter_groups(rows: list) -> list:
    from rrr.render import _regex_doc_id_to_author_label

    groups: dict = {}
    for meta in rows:
        if not getattr(meta, "year_letter_source", ""):
            continue
        parts = _DOC_ID_PARTS_RE.match(str(meta.doc_id or ""))
        if not parts or not parts.group("letter"):
            continue
        key = (parts.group("name"), parts.group("year"))
        groups.setdefault(key, []).append(meta)
    report = []
    for (name, year), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        report.append({
            "label": _regex_doc_id_to_author_label(f"{name}_{year}"),
            "members": [
                {
                    "doc_id": meta.doc_id,
                    "pdf": meta.pdf_path,
                    "year": meta.year,
                    "letter_source": meta.year_letter_source,
                }
                for meta in sorted(members, key=lambda meta: meta.doc_id)
            ],
        })
    return report


def _join_notes(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def _display_name(name) -> str:
    text = unicodedata.normalize("NFC", _bibtex_plain(name))
    for dash in "\u2010\u2011\u2012\u2013\u2014":
        text = text.replace(dash, "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[()\[\];,]", " ", text)
    text = re.sub(r"\bet\s+al\.?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[^\W\d_]", text) else ""


def build_display_label(surnames: list, year) -> str:
    names = [_display_name(name) for name in surnames or []]
    cell = re.fullmatch(r"\s*(\d{4}[a-z]?)\s*", str(year or ""))
    if not names or not all(names) or not cell:
        return ""
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} ({cell.group(1)})"


def assign_display_labels(rows: list) -> None:
    for meta in rows:
        if not getattr(meta, "doc_id", ""):
            continue
        surnames = list(getattr(meta, "surnames", None) or []) or _surnames_from_meta(meta)
        meta.display_label = build_display_label(surnames, meta.year)


def hold_repeated_bib_entries(rows: list) -> list:
    strength = {"file_field": 0, "key": 1, "file_name": 2, "title": 3}
    by_key: dict = {}
    for meta in rows:
        if getattr(meta, "source", "") == "bib_sidecar" and getattr(meta, "bib_key", ""):
            by_key.setdefault(meta.bib_key, []).append(meta)
    held = []
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda meta: (
            strength.get(str(meta.notes or "").partition("bib_match:")[2], 9),
            Path(str(meta.pdf_path or "")).name.casefold(),
        ))
        keeper = members[0]
        for meta in members[1:]:
            same_file = bool(meta.content_sha1) and meta.content_sha1 == keeper.content_sha1
            meta.notes = (
                f"bibliography entry {key} is already matched to "
                f"{Path(str(keeper.pdf_path)).name}"
                + ("; this PDF is an identical copy of it" if same_file else
                   "; remove the second copy, or give this PDF its own entry")
            )
            meta.doc_id = ""
            meta.confidence = "low"
            meta.source = "bib_repeated"
            held.append(meta)
    return held


def _fill_from_bib(meta: ExtractedMeta, entry: dict) -> None:
    meta.bib_key = entry.get("key", "")
    meta.surnames = list(entry.get("surnames") or [])
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
    if entry.get("surnames"):
        return list(entry["surnames"])
    return [s.strip() for s in (entry.get("authors_short") or "").split("&") if s.strip()]


def cascade(pdf_path: Path,
             sidecar_bib_entries: Optional[list] = None,
             existing_doc_ids: Optional[set] = None,
             use_llm: bool = True,
             use_crossref: bool = True,
             use_openalex: bool = True,
             llm_model: Optional[str] = None) -> ExtractedMeta:
    existing_doc_ids = set(existing_doc_ids or ())
    meta = ExtractedMeta(pdf_path=str(pdf_path.resolve()))

    try:
        meta.content_sha1 = content_sha1(pdf_path)
    except Exception as e:
        meta.confidence = "failed"
        meta.source = "none"
        meta.notes = f"content_sha1_failed:{e}"
        return meta

    text = extract_pages_text(pdf_path, max_pages=3)
    if len(text.strip()) < 200:
        meta.confidence = "low"
        meta.source = "needs_ocr"
        meta.notes = (
            f"page_text_only_{len(text.strip())}_chars; the first pages carry "
            "too little text to identify the document, so the PDF needs OCR"
        )
        meta.lang = detect_pdf_language(text)
        return meta

    meta.lang = detect_pdf_language(text)

    reasons = []

    if sidecar_bib_entries:
        match = match_bib_entry_detailed(pdf_path, text, sidecar_bib_entries)
        if match.entry:
            entry = match.entry
            _fill_from_bib(meta, entry)
            meta.source = "bib_sidecar"
            meta.confidence = "high"
            meta.notes = f"bib_match:{match.route}"
            meta.doc_id = generate_doc_id(_bib_surnames_list(entry), entry["year"], existing_doc_ids)
            return meta
        if match.ambiguous:
            meta.source = "bib_ambiguous"
            meta.confidence = "low"
            meta.notes = match.note
            return meta
        reasons.append(match.note)
    else:
        reasons.append("no bibliography file was supplied")

    fn = parse_filename_heuristic(pdf_path)
    if fn:
        meta.notes = f"filename_hint:{fn['name']}_{fn['year']}"

    doi = find_doi(text)
    if not doi:
        reasons.append("no DOI on the first pages")
    elif not use_crossref:
        reasons.append(f"DOI {doi} found, Crossref lookup switched off")
    if doi and use_crossref:
        cr = crossref_lookup(doi)
        if not cr:
            reasons.append(f"DOI {doi} found, Crossref returned no record")
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

    if not use_llm:
        reasons.append("model extraction not requested")
    if use_llm:
        llm = llm_extract_metadata(text, model=llm_model)
        if not (llm and (llm.get("authors") or llm.get("year"))):
            reasons.append("model extraction returned no author or year")
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

    meta.confidence = "failed"
    meta.source = "none"
    meta.notes = _join_notes(meta.notes, "unmatched: " + "; ".join(reasons))
    return meta


def ingest_corpus(corpus_dir: Path,
                   output_csv: Optional[Path] = None,
                   sidecar_bib: Optional[Path] = None,
                   use_llm: bool = True,
                   use_crossref: bool = True,
                   use_openalex: bool = True,
                   llm_model: Optional[str] = None,
                   progress: bool = True) -> list:
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
            print(f"  [{dt:5.1f}s] {pdf.name:44s}  ->  doc_id={meta.doc_id!r:34s} "
                  f"src={meta.source} conf={meta.confidence}"
                  + (f"  notes={meta.notes}" if meta.notes else ""))
    for meta in hold_repeated_bib_entries(results):
        if progress:
            print(f"[ingest] held for review: {Path(meta.pdf_path).name}: {meta.notes}")
    for group in assign_year_letters(results):
        if progress:
            members = "; ".join(
                f"{member['doc_id']} <- {Path(member['pdf']).name} "
                f"(letter: {member['letter_source']})"
                for member in group["members"]
            )
            print(f"[ingest] same author and year, {group['label']}: {members}")
    assign_display_labels(results)
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
                "display_label": getattr(r, "display_label", ""),
            })
