import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from rrr.utils import ensure_dir
from rrr.paths import runs_path
from rrr.render import CITE_RE, parse_citations, render_citation

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": float(os.environ.get("RRR_WRITER_T", "0.50")),  # v7: increased from 0.30
    "num_ctx": int(os.environ.get("RRR_WRITER_CTX", "32768")),
    "num_predict": int(os.environ.get("RRR_WRITER_PRED", "2000")),
    "top_p": float(os.environ.get("RRR_WRITER_TOPP", "0.9")),
}

_TAIL_CHARS = int(os.environ.get("RRR_WRITER_TAIL_CHARS", "250"))

_PAGE_ONLY_RE = re.compile(r"\((?:pp?\.)\s*\d+(?:\s*(?:,|-|and)\s*(?:pp?\.)?\s*\d+)*\)", re.IGNORECASE)
_AUTHOR_NAME_RE = r"(?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)[A-Z][A-Za-z&.\-]+)"
_DOC_WITHOUT_PAGE_RE = re.compile(r"\((?=[^)]*[A-Za-z0-9_&.\-]+_\d{4})(?![^)]*:\s*p\.)[^)]*\)")
_AUTHOR_YEAR_PAREN_RE = re.compile(rf"\(({_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?),\s*(\d{{4}})\)")
_AUTHOR_YEAR_TEXT_RE = re.compile(rf"\b({_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?)\s+\((\d{{4}})\)")
_AUTHOR_YEAR_POSSESSIVE_RE = re.compile(rf"\b({_AUTHOR_NAME_RE}(?:\s+et\s+al\.?)?)'s\s+\((\d{{4}})\)")
_MULTIPAGE_CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.\d+\s*,\s*p\.\d+[^)]*\)")
_GENERIC_STYLE_RE = re.compile(
    r"\b("
    r"complex interplay|valuable insights|policy-making|future research|further research|"
    r"further investigation|nuanced perspective|the stakes are high|this analysis will|"
    r"delving deeper|underscores? the need|further exploration|ongoing research|"
    r"shed light|complex and influenced by various factors"
    r")\b",
    re.IGNORECASE,
)
_MIN_SECTION_CITED_DOCS = int(os.environ.get("RRR_WRITER_MIN_SECTION_CITED_DOCS", "2"))
_ENFORCE_COVERAGE = os.environ.get("RRR_WRITER_ENFORCE_COVERAGE", "1") != "0"

# v7: Streamlined system instruction - citation rules only, prose guidance moved to prompts
_SYSTEM_CITATION_INSTRUCTION = (
    "CITATION FORMAT: (AuthorName_Year: p.N)\n"
    "- Single author: (Smith_1990: p.12)\n"
    "- Multiple authors: (North&Weingast_1989: p.28)\n"
    "- Three+ authors: (AcemogluEtAl_2002: p.4)\n\n"
    "RULES:\n"
    "1. Only cite documents and pages from the evidence provided\n"
    "2. Copy document IDs exactly as shown\n"
    "3. One page per citation\n"
    "4. Never write page-only citations such as (p.3)\n"
    "5. Never write author-year citations such as Author (1990) or (Author, 1990)\n"
    "6. If unsure, omit the citation entirely\n"
)


def _get_cluster(d):
    return d.get("cluster", "Other") or "Other"


def _get_stance(d):
    return d.get("stance", "tangential") or "tangential"


def _score_doc(d) -> float:
    return d.get("avg_score", 0)


def _clip(s: str, n=260) -> str:
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "...") if len(s) > n else s


def _format_quote(q) -> str:
    did = str(q.get("doc_id", "")).strip()
    pg = int(q.get("page", 0) or 0)
    tx = _clip(q.get("text", ""), n=260)
    eid = str(q.get("evidence_id", "")).strip()
    prefix = f"[{eid}] " if eid else ""
    return f'{prefix}"{tx}" {render_citation(did, pg)}'


def _format_doc_entry(d) -> str:
    # Format doc entry with stance label for writer context.
    did = str(d.get("doc_id", "")).strip()
    stance = d.get("stance", "tangential")
    lines = [f"[{did}] [STANCE: {stance.upper()}]"]
    mechanisms = [str(m).strip() for m in d.get("mechanisms", []) if str(m).strip()]
    if mechanisms:
        lines.append("  Mechanisms:")
        for m in mechanisms[:2]:
            lines.append(f"  - {_clip(m, n=180)}")
    qs = d.get("quotes") or []
    for q in qs[:4]:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    # Create explicit list of allowed citations for this chunk.
    lines = []
    evidence_lines = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            page = int(q.get("page", 0) or 0)
            if eid and did and page:
                evidence_lines.append(f"  - [{eid}] -> {render_citation(did, page)}")
        if evidence_lines:
            continue
        pages = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pages:
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - {did}: {page_str}")
    if evidence_lines:
        return "\n".join(evidence_lines[:32])
    return "\n".join(lines)


def _build_evidence_id_map(docs):
    evidence = {}
    for d in docs:
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", "")).strip()
            page = int(q.get("page", 0) or 0)
            if eid and did and page:
                evidence[eid] = {"doc_id": did, "page": page}
    return evidence


def _render_evidence_id_citations(text: str, evidence_map: dict) -> tuple:
    replacements = 0

    def repl(match):
        nonlocal replacements
        eid = match.group(1).upper()
        ev = evidence_map.get(eid)
        if not ev:
            return match.group(0)
        replacements += 1
        return render_citation(ev["doc_id"], ev["page"])

    rendered = re.sub(r"\[([Ee]\d{4})\]", lambda m: repl(m), text or "")
    return rendered, replacements


def _max_clusters_for_stance(n_docs: int) -> int:
    # Determine max clusters based on evidence density.
    if n_docs >= 15:
        return 3
    elif n_docs >= 8:
        return 2
    else:
        return 1


def _strip_wrapping(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def _strip_placeholder_citations(text: str) -> str:
    # Remove placeholder citations that leaked from system prompt.
    text = re.sub(r'\s*\(DocId_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(AuthorName_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(AuthorEtAl_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(FirstAuthor&SecondAuthor_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(DocId_\d{4}:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(AuthorName_YYYY:\s*p\.N\)', '', text)
    text = re.sub(r'\s*\(FirstAuthor&SecondAuthor_YYYY:\s*p\.N\)', '', text)
    text = re.sub(r'\s*\(FirstAuthorEtAl_YYYY:\s*p\.N\)', '', text)
    text = re.sub(r'\s*\(Smith_1990:\s*p\.12\)', '', text)  # v7: catch example from system prompt
    return text


def _split_sentences_for_cleanup(line: str):
    sentinel = "__RRR_DOT__"

    def protect(m):
        return m.group(0).replace(".", sentinel)

    protected = re.sub(r"\bet\s+al\.", protect, line, flags=re.IGNORECASE)
    protected = re.sub(r"\b(?:e\.g|i\.e|cf)\.", protect, protected, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace(sentinel, ".") for p in parts if p.strip()]


def _fix_ajr_abbreviation(text: str) -> tuple:
    # Fix AJR abbreviation to AcemogluEtAl. Returns (fixed_text, fix_count).
    fix_count = 0
    
    def replacer(m):
        nonlocal fix_count
        year = m.group(1)
        page = m.group(2)
        fix_count += 1
        return f"(AcemogluEtAl_{year}: p.{page})"
    
    text = re.sub(r'\(AJR_(\d{4}):\s*p\.(\d+)\)', replacer, text)
    
    def replacer_no_page(m):
        nonlocal fix_count
        year = m.group(1)
        fix_count += 1
        return f"(AcemogluEtAl_{year})"
    
    text = re.sub(r'\(AJR_(\d{4})\)', replacer_no_page, text)
    
    return text, fix_count


def _normalize_citation_case(text: str, allowed_docs: set) -> tuple:
    # Normalize citation case to match corpus. Returns (fixed_text, fix_count).
    lower_to_canonical = {did.lower(): did for did in allowed_docs}
    
    fix_count = 0
    
    def replacer(m):
        nonlocal fix_count
        full_match = m.group(0)
        doc_id = m.group(1)
        page = m.group(2)
        
        doc_lower = doc_id.lower()
        if doc_lower in lower_to_canonical:
            canonical = lower_to_canonical[doc_lower]
            if canonical != doc_id:
                fix_count += 1
                return f"({canonical}: p.{page})"
        return full_match
    
    text = re.sub(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', replacer, text)
    
    return text, fix_count


def _remove_invalid_citations(text: str, allowed_docs: set, allowed_pairs=None) -> tuple:
    # Remove sentences containing citations outside the validated evidence set.
    allowed_lower = {did.lower() for did in allowed_docs}
    lower_to_canonical = {did.lower(): did for did in allowed_docs}
    allowed_pairs = set(allowed_pairs or [])
    
    removed = []
    
    def find_invalid_citations_in_text(txt):
        invalid = []
        strict_spans = []
        for m in CITE_RE.finditer(txt):
            strict_spans.append((m.start(), m.end()))
            doc_id = m.group(1)
            page = int(m.group(2))
            if doc_id.lower() not in allowed_lower:
                invalid.append((m.start(), m.end(), doc_id, page, "unknown_doc"))
                continue
            canonical = lower_to_canonical.get(doc_id.lower(), doc_id)
            if allowed_pairs and (canonical, page) not in allowed_pairs:
                invalid.append((m.start(), m.end(), canonical, page, "invalid_page"))

        def outside_strict(match):
            return not any(s <= match.start() < e for s, e in strict_spans)

        loose_patterns = [
            (_MULTIPAGE_CITE_RE, "multi_page_citation"),
            (_PAGE_ONLY_RE, "page_only_citation"),
            (_DOC_WITHOUT_PAGE_RE, "doc_without_page"),
            (_AUTHOR_YEAR_PAREN_RE, "author_year_citation"),
            (_AUTHOR_YEAR_TEXT_RE, "author_year_citation"),
            (_AUTHOR_YEAR_POSSESSIVE_RE, "author_year_citation"),
        ]
        for pattern, reason in loose_patterns:
            for m in pattern.finditer(txt):
                if outside_strict(m):
                    invalid.append((m.start(), m.end(), m.group(0), 0, reason))
        return invalid
    
    invalid_citations = find_invalid_citations_in_text(text)
    
    if not invalid_citations:
        return text, []
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_invalid = find_invalid_citations_in_text(line)
        
        if not line_invalid:
            cleaned_lines.append(line)
            continue
        
        sentences = _split_sentences_for_cleanup(line)
        
        kept_sentences = []
        for sent in sentences:
            sent_invalid = find_invalid_citations_in_text(sent)
            if sent_invalid:
                for _, _, doc_id, page, reason in sent_invalid:
                    removed.append({
                        'doc_id': doc_id,
                        'page': page,
                        'reason': reason,
                        'sentence': sent[:100] + '...' if len(sent) > 100 else sent
                    })
            else:
                kept_sentences.append(sent)
        
        if kept_sentences:
            cleaned_lines.append(' '.join(kept_sentences))
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    
    return cleaned_text, removed


def _remove_style_violations(text: str) -> tuple:
    removed = []
    cleaned_lines = []
    for line in text.split('\n'):
        if not _GENERIC_STYLE_RE.search(line):
            cleaned_lines.append(line)
            continue

        kept = []
        for sent in _split_sentences_for_cleanup(line):
            if _GENERIC_STYLE_RE.search(sent):
                removed.append(_clip(sent, n=180))
            else:
                kept.append(sent)
        if kept:
            cleaned_lines.append(' '.join(kept))

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    return cleaned_text.strip(), removed


def _strip_orphaned_citations(text: str) -> str:
    # Remove lines that are ONLY a citation.
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\([A-Za-z0-9_&.\-]+:\s*p\.\d+\)$', stripped):
            continue
        if re.match(r'^\([A-Za-z0-9_&]+_\d{4}[a-z]?\)$', stripped):
            continue
        if re.match(r'^\([A-Za-z]+\s+et\s+al\.?,?\s*\d{4}\)$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def _strip_continuation_markers(text: str) -> str:
    # Remove to be continued and similar markers.
    text = re.sub(r'(?im)^\s*Coverage repair:\s*$', '', text)
    text = re.sub(r'(?i)\bThe previous draft failed the citation coverage rule\.\s*', '', text)
    text = re.sub(r'(?i)\bWrite the section again using only the allowed citations above\.\s*', '', text)
    text = re.sub(r'(?im)^\s*(Requirements|Previous draft|Rewrite):\s*$', '', text)
    patterns = [
        r'\.\.\.?\s*to be continued.*?\n*',
        r'\(to be continued.*?\)',
        r'The next section will.*?\.',
        r'In the next section.*?\.',
        r'we will delve deeper.*?\.',
        r'\.\.\.to be continued in the next section\.',
        r'Continued in next message\.\.\.?\s*',
        r'The discussion will continue.*?\.',
        r'In the following section[s,]*\s*$',
        r'This review will continue.*?\.',
        r'As we continue our exploration.*?\.',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _strip_conclusion(text: str) -> str:
    # Remove conclusion paragraphs.
    patterns = [
        r'\n\s*In conclusion[,.].*$',
        r'\n\s*To conclude[,.].*$',
        r'\n\s*In summary[,.].*$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _extract_citation_dumps(text: str):
    # Extract citation dump lines and return (cleaned_text, dump_citations).
    lines = text.split('\n')
    cleaned = []
    dump_citations = []
    
    for line in lines:
        stripped = line.strip()

        if stripped.startswith('[') and stripped.endswith(']'):
            inner = stripped[1:-1]
            cite_matches = re.findall(r'([A-Za-z0-9_&.\-]+):\s*p\.\d+', inner)
            page_refs = re.findall(r'\bp\.\d+\b', inner)
            if cite_matches and (len(cite_matches) >= 2 or len(page_refs) >= 3):
                dump_citations.extend(cite_matches)
                continue
        
        if stripped.startswith('(') and stripped.endswith(')') and ',' in stripped:
            inner = stripped[1:-1]
            cite_matches = re.findall(r'[A-Za-z0-9_&.\-]+_\d{4}[a-z]?:\s*p\.\d+', inner)
            if len(cite_matches) >= 2:
                for m in cite_matches:
                    did = m.split(':')[0]
                    dump_citations.append(did)
                continue
        
        if stripped.startswith('(') and stripped.endswith(')'):
            inner = stripped[1:-1]
            page_refs = re.findall(r'p\.\d+', inner)
            if len(page_refs) >= 3:
                doc_match = re.match(r'([A-Za-z0-9_&]+)', inner)
                if doc_match:
                    dump_citations.append(doc_match.group(1))
                continue
        
        paren_count = len(re.findall(r'\([A-Za-z]', stripped))
        if paren_count >= 3 and len(stripped) < 600:
            without_citations = re.sub(r'\([^)]+\)', '', stripped)
            prose_ratio = len(without_citations.strip()) / max(len(stripped), 1)
            if prose_ratio < 0.3:
                for m in re.finditer(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', stripped):
                    dump_citations.append(m.group(1))
                for m in re.finditer(r'\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)', stripped):
                    dump_citations.append(m.group(1))
                continue
        
        cleaned.append(line)
    
    return '\n'.join(cleaned), dump_citations


def _strip_references_section(text: str) -> str:
    # Remove formal References/Bibliography sections.
    patterns = [
        r'\n\s*References\s*:?\s*\n.*$',
        r'\n\s*Bibliography\s*:?\s*\n.*$',
        r'\n\s*Works Cited\s*:?\s*\n.*$',
        r'\n\s*\(References:.*?\).*$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _writer_enforcement_enabled() -> bool:
    return _ENFORCE_COVERAGE and os.environ.get("RRR_BYPASS_VALIDATION", "0") != "1"


def _writer_parallel_workers(n_chunks: int) -> int:
    if n_chunks <= 1 or os.environ.get("RRR_WRITER_PARALLEL", "1") == "0":
        return 1
    raw = os.environ.get("RRR_WRITER_PARALLELISM") or os.environ.get("RRR_CONCURRENCY") or "2"
    try:
        workers = int(raw)
    except Exception:
        workers = 1
    return max(1, min(workers, n_chunks))


def _strict_cited_doc_ids(text: str, allowed_pairs=None) -> set:
    allowed_pairs = set(allowed_pairs or [])
    cited = set()
    for c in parse_citations(text):
        pair = (c["doc_id"], c["page"])
        if not allowed_pairs or pair in allowed_pairs:
            cited.add(c["doc_id"])
    return cited


def _coverage_requirement(chunk_docs, section_kind: str) -> int:
    n_docs = len([d for d in chunk_docs if d.get("doc_id")])
    if n_docs <= 0:
        return 0
    if section_kind == "closing":
        return 1
    return min(max(1, _MIN_SECTION_CITED_DOCS), n_docs)


def _audit_section_coverage(text: str, chunk_docs, section_kind: str):
    chunk_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
    cited = _strict_cited_doc_ids(text, allowed_pairs=chunk_pairs)
    required = _coverage_requirement(chunk_docs, section_kind)
    return {
        "section": section_kind,
        "required_cited_docs": required,
        "cited_doc_count": len(cited),
        "cited_docs": sorted(cited),
        "provided_doc_count": len(chunk_allowed_docs),
        "ok": len(cited) >= required,
    }


def _append_coverage_fallback(text: str, chunk_docs, required_docs: int, allowed_pairs=None) -> tuple:
    allowed_pairs = set(allowed_pairs or [])
    cited = _strict_cited_doc_ids(text, allowed_pairs=allowed_pairs)
    if len(cited) >= required_docs:
        return text, 0

    additions = []
    for d in chunk_docs:
        did = str(d.get("doc_id", "")).strip()
        if not did or did in cited:
            continue
        quote = None
        for q in d.get("quotes", []) or []:
            pg = int(q.get("page", 0) or 0)
            if pg and (not allowed_pairs or (did, pg) in allowed_pairs):
                quote = q
                break
        if not quote:
            continue
        pg = int(quote.get("page", 0) or 0)
        tx = _clip(quote.get("text", ""), n=180)
        additions.append(f'A further source records "{tx}" {render_citation(did, pg)}.')
        cited.add(did)
        if len(cited) >= required_docs:
            break

    if not additions:
        return text, 0

    patched = (text or "").rstrip()
    if patched:
        patched += "\n\n"
    patched += " ".join(additions)
    return patched.strip(), len(additions)


def _coverage_retry_prompt(original_prompt: str, prior_chunk: str, required_docs: int) -> str:
    return f"""{original_prompt}

Coverage repair:
The previous draft failed the citation coverage rule. Write the section again using only the allowed citations above.

Requirements:
- Cite at least {required_docs} different provided documents when that many are available.
- Every paragraph must include at least one strict citation in the form (DocId: p.N).
- Do not use page-only citations, author-year citations, or citations without pages.
- Preserve the same substantive role and word range.

Previous draft:
{prior_chunk}

Rewrite:"""


def _build_allowed_citations(docs):
    allowed_pairs = set()
    allowed_docs = set()
    allowed_pages_by_doc = defaultdict(set)
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        if not did:
            continue
        allowed_docs.add(did)
        for q in d.get("quotes", []) or []:
            qdid = str(q.get("doc_id", did)).strip() or did
            try:
                pg = int(q.get("page", 0) or 0)
            except Exception:
                pg = 0
            if qdid and pg > 0:
                allowed_pairs.add((qdid, pg))
                allowed_pages_by_doc[qdid].add(pg)
    return allowed_pairs, allowed_docs, allowed_pages_by_doc


def _build_year_to_docid(docs):
    # Build year -> doc_id mapping. Only includes unambiguous mappings.
    year_to_docs = defaultdict(list)
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        if not did:
            continue
        m = re.search(r'_(\d{4})[a-z]?$', did)
        if m:
            year = m.group(1)
            year_to_docs[year].append(did)
    return {year: docs[0] for year, docs in year_to_docs.items() if len(docs) == 1}


def _repair_year_only_citations(text: str, year_to_docid: dict) -> tuple:
    # Repair (YEAR: p.X) -> (DocId_Year: p.X) using context.
    repair_count = 0
    
    def replacer(m):
        nonlocal repair_count
        year = m.group(1)
        page = m.group(2)
        if year in year_to_docid:
            repair_count += 1
            return f"({year_to_docid[year]}: p.{page})"
        return m.group(0)
    
    repaired = re.sub(r'\((\d{4}):\s*p\.(\d+)\)', replacer, text)
    return repaired, repair_count


# =============================================================================
# v7: LEANER PROMPTS - Claims about phenomena, not scholars
# =============================================================================

_PROSE_DIRECTIVE = (
    "Write in the register of historical demography and economic history, with attention to population processes, "
    "institutions, labor regimes, prices, measurement, state capacity, and source limits. "
    "Do not begin sentences with author names, and do not write 'X argues', 'X demonstrates', 'X highlights', or 'X supports'. "
    "State the substantive claim, then cite a validated page. "
    "Avoid generic survey phrases such as 'the literature suggests', 'complex interplay', 'future research', and 'further investigation'. "
    "Do not use em dashes. "
    "You may cite an evidence ID such as [E0001]; it will be rendered into a validated page citation. "
    "Otherwise all citations must use the (Author_Year: p.N) format. No other citation style."
)

def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Literature review on: {topic}

{stance_summary}

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Write 200-300 words. Establish the central question and its stakes. Introduce the main positions. End mid-thought—the argument continues.

Begin:"""


def _build_supports_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster} — these sources SUPPORT the thesis.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Develop the argument directly. 200-300 words. End mid-thought.

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster} — these sources CHALLENGE the thesis.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Present the counterargument directly. 200-300 words. End mid-thought.

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster} — these sources ADD NUANCE to the thesis.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Show how these qualifications reshape the argument. 200-300 words. End mid-thought.

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Close this literature review on: {topic}

Previous ending:
...{previous_tail}

ALLOWED CITATIONS:
{allowed_list}

Remaining evidence:
{evidence}

{_PROSE_DIRECTIVE}

Write 150-200 words. Identify where scholars converge and diverge. State what remains unresolved. 
Do NOT write "In conclusion" or "To summarize."

Continue:"""


# Batch-one prompt definitions. These names intentionally shadow the earlier
# prompt builders so the writer uses mechanisms and stance-specific structures.
def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Literature review on: {topic}

{stance_summary}

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Write 200-300 words. Establish the central question and its stakes. Introduce the main positions. End mid-thought, since the argument continues.

Begin:"""


def _build_supports_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources SUPPORT the thesis through corroborating mechanisms, measurements, or historical conditions.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Lead with the shared mechanism, then connect the strongest quoted evidence to the historical claim. 200-300 words. End mid-thought.

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources CHALLENGE the thesis by identifying weak assumptions, alternative causes, or contrary evidence.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Name the vulnerable assumption, develop the counter-evidence, and keep citations tied to concrete pages. 200-300 words. End mid-thought.

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources COMPLICATE the thesis by setting scope conditions, contingencies, or measurement limits.

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

DO NOT restate the thesis. DO NOT mention "future research" or "further investigation."
Show how the scope condition changes the interpretation without treating the evidence as a simple rejection. 200-300 words. End mid-thought.

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Close this literature review on: {topic}

Previous ending:
...{previous_tail}

ALLOWED CITATIONS:
{allowed_list}

Section claims and supporting evidence:
{evidence}

{_PROSE_DIRECTIVE}

Write 150-200 words. Identify where scholars converge and diverge. State what remains unresolved.
Do NOT write "In conclusion" or "To summarize."

Continue:"""


def _ollama_chat(prompt: str, metrics=None, stage="writer"):
    import ollama
    import time
    start = time.perf_counter()
    try:
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_CITATION_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            options=_DEFAULT_CHAT_OPTIONS,
            keep_alive=_KEEP_ALIVE,
            stream=False,
        )
    except Exception as e:
        if metrics:
            metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                               success=False, duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), error=e)
        raise
    out = (res.get("message", {}).get("content") or "").strip()
    if metrics:
        metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                           duration_s=time.perf_counter() - start,
                           prompt_chars=len(prompt), response_chars=len(out))
    return out


def _build_author_year_lookup(allowed_docs):
    # Build reverse lookup: (author, year) -> doc_id for academic citation matching.
    author_year_to_docid = {}
    for did in allowed_docs:
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
    # Collect cited doc_ids from both correct and academic citation formats.
    cited_docs = set()
    
    for m in re.finditer(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    for m in re.finditer(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    for m in re.finditer(r"\(([A-Za-z&]+(?:\s+et\s+al\.?)?)[,\s]+(\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    return cited_docs


def compose_from_ledger(ledger_path=None, metrics=None):
    ledger_path = ledger_path or str(runs_path("review_ledger.json"))
    if not os.path.isfile(ledger_path):
        raise SystemExit(f"Ledger not found: {ledger_path}")

    with open(ledger_path, encoding="utf-8") as f:
        data = json.load(f)

    topic = data.get("topic", "(no topic)")
    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise SystemExit("Ledger empty or malformed (no docs).")

    allowed_pairs, allowed_docs, allowed_pages_by_doc = _build_allowed_citations(docs)
    if not allowed_pairs:
        raise SystemExit("No allowed citations found in ledger.")

    author_year_to_docid = _build_author_year_lookup(allowed_docs)
    evidence_id_map = _build_evidence_id_map(docs)

    # Bucket by stance first, then by cluster
    stance_buckets = defaultdict(lambda: defaultdict(list))
    for d in docs:
        stance = _get_stance(d)
        cluster = _get_cluster(d)
        stance_buckets[stance][cluster].append(d)

    stance_counts = {s: sum(len(cl) for cl in clusters.values()) 
                     for s, clusters in stance_buckets.items()}
    
    stance_summary = f"Of {len(docs)} sources, {stance_counts.get('supports', 0)} support the thesis, " \
                     f"{stance_counts.get('critiques', 0)} offer critiques, and " \
                     f"{stance_counts.get('complicates', 0)} add nuance or qualifications."
    
    print(f"[Writer] Stance distribution: {dict(stance_counts)}")

    # Build chunk sequence with adaptive cluster count
    chunk_plan = []
    
    def rank_clusters(stance):
        if stance not in stance_buckets:
            return []
        clusters = stance_buckets[stance]
        ranked = sorted(
            clusters.items(),
            key=lambda kv: sum(_score_doc(x) for x in kv[1]),
            reverse=True
        )
        n_docs = stance_counts.get(stance, 0)
        max_clusters = _max_clusters_for_stance(n_docs)
        return ranked[:max_clusters]
    
    for cluster, cluster_docs in rank_clusters("supports"):
        chunk_plan.append(("supports", cluster, cluster_docs, _build_supports_prompt))
    
    for cluster, cluster_docs in rank_clusters("critiques"):
        chunk_plan.append(("critiques", cluster, cluster_docs, _build_critiques_prompt))
    
    for cluster, cluster_docs in rank_clusters("complicates"):
        chunk_plan.append(("complicates", cluster, cluster_docs, _build_complicates_prompt))
    
    if not chunk_plan:
        raise SystemExit("No documents to write about.")

    supports_clusters = sum(1 for s, _, _, _ in chunk_plan if s == "supports")
    critiques_clusters = sum(1 for s, _, _, _ in chunk_plan if s == "critiques")
    complicates_clusters = sum(1 for s, _, _, _ in chunk_plan if s == "complicates")
    print(f"[Writer] Adaptive clusters: supports={supports_clusters}, critiques={critiques_clusters}, complicates={complicates_clusters}")
    print(f"[Writer] Generating {len(chunk_plan) + 2} sections (opening + {len(chunk_plan)} stance sections + closing)...")
    if metrics:
        metrics.set("writer_chunk_plan", {
            "supports": supports_clusters,
            "critiques": critiques_clusters,
            "complicates": complicates_clusters,
            "total_stance_sections": len(chunk_plan),
        })

    chunks = []
    section_claims = []
    all_dump_citations = []
    total_repairs = 0
    total_placeholders_stripped = 0
    total_ajr_fixes = 0
    total_removed_citations = 0
    total_style_removed = 0
    total_coverage_fallbacks = 0
    total_evidence_id_renders = 0
    section_coverage = []
    
    def postprocess_chunk(chunk, chunk_docs):
        nonlocal total_repairs, total_placeholders_stripped, all_dump_citations, total_ajr_fixes
        nonlocal total_removed_citations, total_style_removed, total_evidence_id_renders
        
        chunk = _strip_wrapping(chunk)
        chunk, evidence_renders = _render_evidence_id_citations(chunk, evidence_id_map)
        total_evidence_id_renders += evidence_renders
        
        chunk_before = chunk
        chunk = _strip_placeholder_citations(chunk)
        placeholders_stripped = chunk_before.count('DocId_Year') + chunk_before.count('AuthorName_Year')
        total_placeholders_stripped += placeholders_stripped
        
        chunk, ajr_fixes = _fix_ajr_abbreviation(chunk)
        total_ajr_fixes += ajr_fixes
        
        year_to_docid = _build_year_to_docid(chunk_docs)
        chunk, repair_count = _repair_year_only_citations(chunk, year_to_docid)
        total_repairs += repair_count
        
        chunk, dump_cites = _extract_citation_dumps(chunk)
        all_dump_citations.extend(dump_cites)
        
        chunk = _strip_orphaned_citations(chunk)
        chunk = _strip_references_section(chunk)
        chunk = _strip_continuation_markers(chunk)
        chunk = _strip_conclusion(chunk)
        if _writer_enforcement_enabled():
            chunk_allowed_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
            chunk, removed = _remove_invalid_citations(
                chunk,
                chunk_allowed_docs,
                allowed_pairs=chunk_allowed_pairs,
            )
            total_removed_citations += len(removed)

        chunk, style_removed = _remove_style_violations(chunk)
        total_style_removed += len(style_removed)
        
        return chunk, repair_count, placeholders_stripped, ajr_fixes, len(style_removed)

    def finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage):
        nonlocal total_coverage_fallbacks
        chunk, repairs, placeholders, ajr, style_removed = postprocess_chunk(raw, chunk_docs)
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        if audit["ok"] or not _writer_enforcement_enabled():
            return chunk, repairs, placeholders, ajr, style_removed, audit
        if metrics:
            metrics.inc("writer_section_coverage_retries")
        retry_prompt = _coverage_retry_prompt(prompt, chunk, audit["required_cited_docs"])
        raw = _ollama_chat(retry_prompt, metrics=metrics, stage=f"{stage}_coverage_retry")
        chunk, repairs2, placeholders2, ajr2, style_removed2 = postprocess_chunk(raw, chunk_docs)
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        if not audit["ok"]:
            chunk_allowed_pairs, _, _ = _build_allowed_citations(chunk_docs)
            chunk, fallback_count = _append_coverage_fallback(
                chunk,
                chunk_docs,
                audit["required_cited_docs"],
                allowed_pairs=chunk_allowed_pairs,
            )
            if fallback_count:
                total_coverage_fallbacks += fallback_count
                if metrics:
                    metrics.inc("writer_section_coverage_fallbacks", fallback_count)
                audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        if not audit["ok"]:
            raise ValueError(
                f"citation coverage failed for {section_kind}: "
                f"{audit['cited_doc_count']}/{audit['required_cited_docs']} cited docs"
            )
        return (
            chunk,
            repairs + repairs2,
            placeholders + placeholders2,
            ajr + ajr2,
            style_removed + style_removed2,
            audit,
        )

    def generate_covered_chunk(prompt, chunk_docs, section_kind, stage):
        raw = _ollama_chat(prompt, metrics=metrics, stage=stage)
        return finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage)

    # Generate opening
    opening_docs = []
    for stance in ["supports", "complicates", "critiques"]:
        for cluster, cluster_docs in stance_buckets[stance].items():
            opening_docs.extend(sorted(cluster_docs, key=_score_doc, reverse=True)[:2])
    opening_docs = sorted(opening_docs, key=_score_doc, reverse=True)[:6]
    
    allowed_list = _list_allowed_citations(opening_docs, allowed_pages_by_doc)
    evidence = "\n\n".join(_format_doc_entry(d) for d in opening_docs)
    
    prompt = _build_opening_prompt(topic, stance_summary, evidence, allowed_list)
    
    try:
        chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
            prompt,
            opening_docs,
            "opening",
            "writer_opening",
        )
        word_count = _count_words(chunk)
        print(f"[Writer] Opening: {word_count} words; cited_docs={audit['cited_doc_count']}")
        section_coverage.append(audit)
        chunks.append(chunk)
        if metrics:
            metrics.inc("writer_sections_succeeded")
    except Exception as e:
        print(f"[Writer] Opening failed: {e}")
        if metrics:
            metrics.inc("writer_sections_failed")

    # Generate stance sections
    stance_jobs = []
    parallel_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    for i, (stance, cluster, cluster_docs, prompt_builder) in enumerate(chunk_plan):
        cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, parallel_tail)
        stance_jobs.append({
            "index": i,
            "stance": stance,
            "cluster": cluster,
            "docs": cluster_docs_sorted,
            "prompt": prompt,
            "stage": f"writer_{stance}",
        })

    def record_stance_chunk(job, chunk, word_count):
        top_mechs = []
        for d in job["docs"]:
            for m in d.get("mechanisms", []) or []:
                m = str(m).strip()
                if m and m not in top_mechs:
                    top_mechs.append(m)
        section_claims.append({
            "stance": job["stance"],
            "cluster": job["cluster"],
            "docs": [d.get("doc_id") for d in job["docs"]],
            "mechanisms": top_mechs[:4],
            "word_count": word_count,
        })
        chunks.append(chunk)

    def log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit):
        notes = []
        if repairs > 0:
            notes.append(f"repaired {repairs}")
        if placeholders > 0:
            notes.append(f"stripped {placeholders}")
        if ajr > 0:
            notes.append(f"AJR fixed {ajr}")
        if style_removed > 0:
            notes.append(f"style stripped {style_removed}")
        if audit.get("cited_doc_count", 0) > 0:
            notes.append(f"cited {audit['cited_doc_count']}")
        note_str = f" ({', '.join(notes)})" if notes else ""
        print(f"[Writer] {job['stance'].upper()}/{job['cluster']}: {word_count} words{note_str}")

    parallel_workers = _writer_parallel_workers(len(stance_jobs))
    if metrics:
        metrics.set("writer_parallel_workers", parallel_workers)

    if parallel_workers > 1:
        print(f"[Writer] Parallel stance chunks: workers={parallel_workers}")
        raw_by_index = {}
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {
                pool.submit(_ollama_chat, job["prompt"], metrics=metrics, stage=job["stage"]): job["index"]
                for job in stance_jobs
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_by_index[idx] = future.result()
                except Exception as e:
                    raw_by_index[idx] = e

        for job in stance_jobs:
            try:
                raw = raw_by_index.get(job["index"])
                if isinstance(raw, Exception):
                    raise raw
                chunk, repairs, placeholders, ajr, style_removed, audit = finalize_covered_chunk(
                    raw,
                    job["prompt"],
                    job["docs"],
                    job["stance"],
                    job["stage"],
                )
                section_coverage.append(audit)
                word_count = _count_words(chunk)
                log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit)
                record_stance_chunk(job, chunk, word_count)
                if metrics:
                    metrics.inc("writer_sections_succeeded")
            except Exception as e:
                print(f"[Writer] {job['stance']}/{job['cluster']} failed: {e}")
                if metrics:
                    metrics.inc("writer_sections_failed")
    else:
        for i, (stance, cluster, cluster_docs, prompt_builder) in enumerate(chunk_plan):
            cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
            allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
            evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
            previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
            prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail)
            job = {
                "index": i,
                "stance": stance,
                "cluster": cluster,
                "docs": cluster_docs_sorted,
                "prompt": prompt,
                "stage": f"writer_{stance}",
            }
            try:
                chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
                    job["prompt"],
                    job["docs"],
                    job["stance"],
                    job["stage"],
                )
                section_coverage.append(audit)
                word_count = _count_words(chunk)
                log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit)
                record_stance_chunk(job, chunk, word_count)
                if metrics:
                    metrics.inc("writer_sections_succeeded")
            except Exception as e:
                print(f"[Writer] {job['stance']}/{job['cluster']} failed: {e}")
                if metrics:
                    metrics.inc("writer_sections_failed")

    # Generate closing
    used_doc_ids = []
    for claim in section_claims:
        for did in claim.get("docs", []):
            if did and did not in used_doc_ids:
                used_doc_ids.append(did)
    closing_docs = [d for d in docs if d.get("doc_id") in set(used_doc_ids)]
    closing_docs = sorted(closing_docs, key=_score_doc, reverse=True)[:6]
    
    allowed_list = _list_allowed_citations(closing_docs, allowed_pages_by_doc)
    if not allowed_list:
        allowed_list = "(Use only citations already present in the preceding text.)"
    claim_lines = []
    for claim in section_claims:
        mechs = "; ".join(claim.get("mechanisms", [])[:3]) or "no mechanism recorded"
        docs_line = ", ".join(str(d) for d in claim.get("docs", [])[:5])
        claim_lines.append(
            f"- {claim['stance'].upper()} / {claim['cluster']}: mechanisms: {mechs}. Documents: {docs_line}."
        )
    evidence_parts = ["Section claims:"] + claim_lines
    if closing_docs:
        evidence_parts.append("\nRepresentative evidence:")
        evidence_parts.extend(_format_doc_entry(d) for d in closing_docs)
    evidence = "\n".join(evidence_parts)
    
    previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    prompt = _build_closing_prompt(topic, evidence, allowed_list, previous_tail)
    
    try:
        chunk, repairs, placeholders, ajr, style_removed, audit = generate_covered_chunk(
            prompt,
            closing_docs,
            "closing",
            "writer_closing",
        )
        word_count = _count_words(chunk)
        print(f"[Writer] Closing: {word_count} words; cited_docs={audit['cited_doc_count']}")
        section_coverage.append(audit)
        chunks.append(chunk)
        if metrics:
            metrics.inc("writer_sections_succeeded")
    except Exception as e:
        print(f"[Writer] Closing failed: {e}")
        if metrics:
            metrics.inc("writer_sections_failed")

    # Final assembly
    full_text = "\n\n".join(chunks)
    full_text, final_evidence_renders = _render_evidence_id_citations(full_text, evidence_id_map)
    total_evidence_id_renders += final_evidence_renders
    
    global_year_to_docid = _build_year_to_docid(docs)
    full_text, final_repairs = _repair_year_only_citations(full_text, global_year_to_docid)
    total_repairs += final_repairs
    
    full_text, final_ajr = _fix_ajr_abbreviation(full_text)
    total_ajr_fixes += final_ajr
    
    full_text = _strip_placeholder_citations(full_text)
    full_text, final_dump_cites = _extract_citation_dumps(full_text)
    all_dump_citations.extend(final_dump_cites)
    
    full_text, case_fixes = _normalize_citation_case(full_text, allowed_docs)
    if case_fixes > 0:
        print(f"[Writer] Case normalized: {case_fixes} citations")
    
    if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        removed_citations = []
        print("[Writer] Citation removal skipped because RRR_BYPASS_VALIDATION=1")
        if metrics:
            metrics.set("writer_bypass_validation", True)
    else:
        full_text, removed_citations = _remove_invalid_citations(full_text, allowed_docs, allowed_pairs=allowed_pairs)
        total_removed_citations += len(removed_citations)
    if removed_citations:
        print(f"[Writer] Removed {len(removed_citations)} invalid citation(s):")
        for r in removed_citations:
            if r.get("page"):
                print(f"         - {r['doc_id']}: p.{r['page']} ({r.get('reason', 'invalid')})")
            else:
                print(f"         - {r['doc_id']} ({r.get('reason', 'invalid')})")

    full_text, final_style_removed = _remove_style_violations(full_text)
    total_style_removed += len(final_style_removed)
    
    full_text = _strip_orphaned_citations(full_text)
    full_text = _strip_references_section(full_text)
    full_text = _strip_continuation_markers(full_text)
    
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    cited_docs = _collect_cited_docs(full_text, allowed_docs, author_year_to_docid)
    for did in all_dump_citations:
        if did in allowed_docs:
            cited_docs.add(did)
    cited_docids = sorted(cited_docs)

    total_words = _count_words(full_text)

    ensure_dir(str(runs_path()))
    out_path = runs_path("review_composed.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(runs_path("review_cited_docs.json"), "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    print(f"[Writer] review_composed.md written ({total_words} words).")
    print(
        f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} "
        f"repairs={total_repairs} AJR={total_ajr_fixes} case={case_fixes} "
        f"removed={total_removed_citations} style_removed={total_style_removed} "
        f"coverage_fallbacks={total_coverage_fallbacks} evidence_id_renders={total_evidence_id_renders}"
    )
    if metrics:
        metrics.set("writer_stats", {
            "chunks_written": len(chunks),
            "distinct_docs_cited": len(cited_docids),
            "word_count": total_words,
            "repairs": total_repairs,
            "ajr_fixes": total_ajr_fixes,
            "case_fixes": case_fixes,
            "removed_citations": total_removed_citations,
            "style_sentences_removed": total_style_removed,
            "coverage_fallbacks": total_coverage_fallbacks,
            "evidence_id_renders": total_evidence_id_renders,
            "citation_dump_docs": len(all_dump_citations),
            "section_claims": section_claims,
            "section_coverage": section_coverage,
        })
        metrics.inc("writer_citation_repairs", total_repairs)
        metrics.inc("writer_removed_citations", total_removed_citations)
        metrics.inc("writer_style_sentences_removed", total_style_removed)
        metrics.inc("writer_evidence_id_renders", total_evidence_id_renders)
    
    return str(out_path)


if __name__ == "__main__":
    compose_from_ledger()
