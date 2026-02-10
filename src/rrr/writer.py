import os
import json
import re
from collections import Counter, defaultdict
from rrr.utils import ensure_dir

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": float(os.environ.get("RRR_WRITER_T", "0.30")),  # Reduced from 0.35
    "num_ctx": int(os.environ.get("RRR_WRITER_CTX", "32768")),
    "num_predict": int(os.environ.get("RRR_WRITER_PRED", "2000")),
    "top_p": float(os.environ.get("RRR_WRITER_TOPP", "0.9")),
}

_TAIL_CHARS = int(os.environ.get("RRR_WRITER_TAIL_CHARS", "250"))

_CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

_SYSTEM_CITATION_INSTRUCTION = """CITATION RULES — MANDATORY:

You MUST cite using EXACTLY this format: (DocId_Year: p.X)

Examples of CORRECT citations:
- (North_1989: p.9)
- (AcemogluEtAl_2001: p.19)
- (North&Weingast_1989: p.28)
- (Broadberry&Gupta_2006: p.9)

WRONG formats (NEVER use):
- (2002: p.4) — missing author
- Kuznets (1973) — must be (Kuznets_1973: p.X)
- (AJR_2001) — no abbreviations, use full doc_id
- (North_1981: p.X) — WRONG if North_1981 is not in evidence
- Author et al. (Year) — WRONG format
- (Author_Year: p.1, p.2) — only ONE page per citation

CRITICAL — DO NOT FABRICATE:
1. ONLY cite documents that appear in the evidence provided below
2. ONLY cite page numbers that appear in the evidence provided below
3. If you cannot find a citation in the evidence, DO NOT INVENT ONE
4. It is better to write a shorter paragraph than to fabricate a citation
5. Never abbreviate document IDs (no "AJR" — use "AcemogluEtAl")
6. Never cite documents not explicitly listed in the evidence
7. Copy document IDs CHARACTER-FOR-CHARACTER from the evidence

PROSE QUALITY — Avoid overused phrases:
- "This perspective underscores..."
- "sheds light on..."
- "It is worth noting..."
- "provides valuable insights..."
- "a pivotal role..."

Make direct statements. Say what the evidence shows."""


def _get_cluster(d):
    return d.get("cluster", "Other") or "Other"


def _get_stance(d):
    return d.get("stance", "tangential") or "tangential"


def _score_doc(d) -> float:
    return d.get("avg_score", 0)


def _clip(s: str, n=260) -> str:
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "…") if len(s) > n else s


def _format_quote(q) -> str:
    did = str(q.get("doc_id", "")).strip()
    pg = int(q.get("page", 0) or 0)
    tx = _clip(q.get("text", ""), n=260)
    return f'"{tx}" ({did}: p.{pg})'


def _format_doc_entry(d) -> str:
    """Format doc entry with stance label for writer context."""
    did = str(d.get("doc_id", "")).strip()
    stance = d.get("stance", "tangential")
    lines = [f"[{did}] [STANCE: {stance.upper()}]"]
    qs = d.get("quotes") or []
    for q in qs[:4]:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    """Create explicit list of allowed citations for this chunk."""
    lines = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        pages = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pages:
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - {did}: {page_str}")
    return "\n".join(lines)


def _strip_wrapping(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def _strip_placeholder_citations(text: str) -> str:
    """Remove placeholder citations that leaked from system prompt."""
    text = re.sub(r'\s*\(DocId_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(AuthorName_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(AuthorEtAl_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(FirstAuthor&SecondAuthor_Year:\s*p\.[X\d]+\)', '', text)
    text = re.sub(r'\s*\(DocId_\d{4}:\s*p\.[X\d]+\)', '', text)
    return text


# ============================================================
# NEW: TIER 1 — AJR HARDCODE FIX (per-chunk, no corpus needed)
# ============================================================

def _fix_ajr_abbreviation(text: str) -> tuple:
    """
    Fix AJR abbreviation to AcemogluEtAl.
    Returns (fixed_text, fix_count).
    """
    fix_count = 0
    
    def replacer(m):
        nonlocal fix_count
        year = m.group(1)
        page = m.group(2)
        fix_count += 1
        return f"(AcemogluEtAl_{year}: p.{page})"
    
    # Pattern: (AJR_YYYY: p.X)
    text = re.sub(r'\(AJR_(\d{4}):\s*p\.(\d+)\)', replacer, text)
    
    # Also catch without page: (AJR_YYYY)
    def replacer_no_page(m):
        nonlocal fix_count
        year = m.group(1)
        fix_count += 1
        return f"(AcemogluEtAl_{year})"
    
    text = re.sub(r'\(AJR_(\d{4})\)', replacer_no_page, text)
    
    return text, fix_count


# ============================================================
# NEW: TIER 2 — CASE NORMALIZATION (needs allowed_docs)
# ============================================================

def _normalize_citation_case(text: str, allowed_docs: set) -> tuple:
    """
    Normalize citation case to match corpus.
    e.g., VanZanden_2009 → vanZanden_2009
    Returns (fixed_text, fix_count).
    """
    # Build case-insensitive lookup
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
            if canonical != doc_id:  # Case differs
                fix_count += 1
                return f"({canonical}: p.{page})"
        return full_match
    
    text = re.sub(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', replacer, text)
    
    return text, fix_count


# ============================================================
# NEW: TIER 3 — INVALID CITATION REMOVAL (needs allowed_docs)
# ============================================================

def _remove_invalid_citations(text: str, allowed_docs: set) -> tuple:
    """
    Remove sentences containing citations to documents not in corpus.
    Returns (cleaned_text, list_of_removed).
    """
    # Build case-insensitive lookup for validation
    allowed_lower = {did.lower() for did in allowed_docs}
    
    removed = []
    
    # Split into sentences (rough but effective)
    # We use a pattern that splits on . ! ? followed by space and capital letter
    # But we need to be careful with abbreviations like "p." and "et al."
    
    def find_invalid_citations_in_text(txt):
        """Find all invalid citations in text."""
        invalid = []
        for m in re.finditer(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', txt):
            doc_id = m.group(1)
            if doc_id.lower() not in allowed_lower:
                invalid.append((m.start(), m.end(), doc_id, m.group(2)))
        return invalid
    
    # Find all invalid citations
    invalid_citations = find_invalid_citations_in_text(text)
    
    if not invalid_citations:
        return text, []
    
    # For each invalid citation, find and remove the containing sentence
    # Work backwards to preserve indices
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Check if this line contains any invalid citation
        line_invalid = find_invalid_citations_in_text(line)
        
        if not line_invalid:
            cleaned_lines.append(line)
            continue
        
        # Split line into sentences and filter
        # Simple sentence split: look for ". " followed by capital letter
        # But preserve the line if only part is invalid
        
        # For simplicity: if line contains invalid citation, try to remove just that sentence
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        
        kept_sentences = []
        for sent in sentences:
            sent_invalid = find_invalid_citations_in_text(sent)
            if sent_invalid:
                for _, _, doc_id, page in sent_invalid:
                    removed.append({
                        'doc_id': doc_id,
                        'page': page,
                        'sentence': sent[:100] + '...' if len(sent) > 100 else sent
                    })
            else:
                kept_sentences.append(sent)
        
        if kept_sentences:
            cleaned_lines.append(' '.join(kept_sentences))
        # If no sentences kept, line is dropped entirely
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Clean up any double spaces or awkward gaps
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    
    return cleaned_text, removed


def _strip_orphaned_citations(text: str) -> str:
    """Remove lines that are ONLY a citation."""
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
    """Remove 'to be continued' and similar markers."""
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
    """Remove conclusion paragraphs."""
    patterns = [
        r'\n\s*In conclusion[,.].*$',
        r'\n\s*To conclude[,.].*$',
        r'\n\s*In summary[,.].*$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _extract_citation_dumps(text: str):
    """Extract citation dump lines and return (cleaned_text, dump_citations)."""
    lines = text.split('\n')
    cleaned = []
    dump_citations = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('(') and stripped.endswith(')') and ',' in stripped:
            inner = stripped[1:-1]
            cite_matches = re.findall(r'[A-Za-z0-9_&]+_\d{4}[a-z]?:\s*p\.\d+', inner)
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
    """Remove formal References/Bibliography sections."""
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


def _build_example_citations(docs, allowed_pages_by_doc):
    """Build example citations from docs — real examples from evidence."""
    examples = []
    for d in docs[:3]:
        did = str(d.get("doc_id", "")).strip()
        pgs = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pgs:
            examples.append(f"({did}: p.{pgs[0]})")
    return ", ".join(examples) if examples else "(AuthorName_Year: p.X)"


def _build_year_to_docid(docs):
    """Build year -> doc_id mapping. Only includes unambiguous mappings."""
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
    """Repair (YEAR: p.X) -> (DocId_Year: p.X) using context."""
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


# ============================================================
# STANCE-AWARE PROMPTS WITH STRICT EVIDENCE CONSTRAINTS
# ============================================================

def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Write the opening section of a literature review on: {topic}

This review examines a scholarly debate. {stance_summary}

ALLOWED CITATIONS — You may ONLY cite from this list:
{allowed_list}

DO NOT cite any document or page not in this list. If unsure, omit the citation.

Evidence to synthesize:
{evidence}

Requirements:
- 300-400 words (shorter is fine if evidence is limited)
- Frame the central question and why it matters
- Introduce the key positions scholars take
- ONLY cite documents and pages from the allowed list above
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Begin:"""


def _build_supports_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present SUPPORTING arguments for the thesis. Theme: {cluster}

ALLOWED CITATIONS — You may ONLY cite from this list:
{allowed_list}

DO NOT cite any document or page not in this list. DO NOT invent citations.

Evidence to synthesize (these scholars SUPPORT the thesis):
{evidence}

Requirements:
- 250-350 words (shorter is fine if evidence is limited)
- Present the mechanisms and evidence these scholars offer
- Connect smoothly to previous text
- ONLY cite documents and pages from the allowed list above
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present COUNTERARGUMENTS to the thesis. Theme: {cluster}

ALLOWED CITATIONS — You may ONLY cite from this list:
{allowed_list}

DO NOT cite any document or page not in this list. DO NOT invent citations.

Evidence to synthesize (these scholars CHALLENGE or CRITIQUE the thesis):
{evidence}

Requirements:
- 250-350 words (shorter is fine if evidence is limited)
- Present the objections, alternative explanations, or empirical challenges
- Frame as counterarguments: "However...", "Against this view...", "Critics argue..."
- Connect smoothly to previous text
- ONLY cite documents and pages from the allowed list above
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present NUANCES and QUALIFICATIONS to the thesis. Theme: {cluster}

ALLOWED CITATIONS — You may ONLY cite from this list:
{allowed_list}

DO NOT cite any document or page not in this list. DO NOT invent citations.

Evidence to synthesize (these scholars ADD NUANCE or COMPLICATE the thesis):
{evidence}

Requirements:
- 250-350 words (shorter is fine if evidence is limited)
- Present conditional factors, scope conditions, or contextual variations
- Frame as refinements: "The relationship proves more complex when...", "Context matters because..."
- Connect smoothly to previous text
- ONLY cite documents and pages from the allowed list above
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    return f"""Write the closing section of this literature review on: {topic}

Previous text ended with:
...{previous_tail}

ALLOWED CITATIONS — You may ONLY cite from this list:
{allowed_list}

DO NOT cite any document or page not in this list. DO NOT invent citations.

Remaining evidence to integrate:
{evidence}

Requirements:
- 200-300 words
- Synthesize the debate: where do scholars agree, where do they diverge?
- Identify gaps in the literature or unresolved questions
- End with directions for future research
- ONLY cite documents and pages from the allowed list above
- Write in flowing prose, no bullet points or headers
- Do NOT write "In conclusion" or similar

Continue:"""


def _ollama_chat(prompt: str):
    import ollama
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
    return (res.get("message", {}).get("content") or "").strip()


def _build_author_year_lookup(allowed_docs):
    """Build reverse lookup: (author, year) -> doc_id for academic citation matching."""
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
    """Collect cited doc_ids from both correct and academic citation formats."""
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


def compose_from_ledger(ledger_path="runs/review_ledger.json"):
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

    # ============================================================
    # BUCKET BY STANCE FIRST, THEN BY CLUSTER
    # ============================================================
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

    # ============================================================
    # BUILD CHUNK SEQUENCE: OPENING → SUPPORTS → CRITIQUES → COMPLICATES → CLOSING
    # ============================================================
    
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
        max_per_stance = int(os.environ.get("RRR_WRITER_MAX_CLUSTERS_PER_STANCE", "3"))
        return ranked[:max_per_stance]
    
    for cluster, cluster_docs in rank_clusters("supports"):
        chunk_plan.append(("supports", cluster, cluster_docs, _build_supports_prompt))
    
    for cluster, cluster_docs in rank_clusters("critiques"):
        chunk_plan.append(("critiques", cluster, cluster_docs, _build_critiques_prompt))
    
    for cluster, cluster_docs in rank_clusters("complicates"):
        chunk_plan.append(("complicates", cluster, cluster_docs, _build_complicates_prompt))
    
    if not chunk_plan:
        raise SystemExit("No documents to write about.")

    print(f"[Writer] Generating {len(chunk_plan) + 2} sections (opening + {len(chunk_plan)} stance sections + closing)...")

    chunks = []
    all_dump_citations = []
    total_repairs = 0
    total_placeholders_stripped = 0
    total_ajr_fixes = 0
    
    def postprocess_chunk(chunk, chunk_docs):
        """Per-chunk postprocessing (before join)."""
        nonlocal total_repairs, total_placeholders_stripped, all_dump_citations, total_ajr_fixes
        
        chunk = _strip_wrapping(chunk)
        
        # Strip placeholder citations
        chunk_before = chunk
        chunk = _strip_placeholder_citations(chunk)
        placeholders_stripped = chunk_before.count('DocId_Year') + chunk_before.count('AuthorName_Year')
        total_placeholders_stripped += placeholders_stripped
        
        # TIER 1: Fix AJR abbreviation (hardcoded, no corpus needed)
        chunk, ajr_fixes = _fix_ajr_abbreviation(chunk)
        total_ajr_fixes += ajr_fixes
        
        # Year-only repairs
        year_to_docid = _build_year_to_docid(chunk_docs)
        chunk, repair_count = _repair_year_only_citations(chunk, year_to_docid)
        total_repairs += repair_count
        
        # Extract citation dumps
        chunk, dump_cites = _extract_citation_dumps(chunk)
        all_dump_citations.extend(dump_cites)
        
        # Basic cleanup
        chunk = _strip_orphaned_citations(chunk)
        chunk = _strip_references_section(chunk)
        chunk = _strip_continuation_markers(chunk)
        chunk = _strip_conclusion(chunk)
        
        return chunk, repair_count, placeholders_stripped, ajr_fixes

    # ============================================================
    # GENERATE OPENING
    # ============================================================
    opening_docs = []
    for stance in ["supports", "complicates", "critiques"]:
        for cluster, cluster_docs in stance_buckets[stance].items():
            opening_docs.extend(sorted(cluster_docs, key=_score_doc, reverse=True)[:2])
    opening_docs = sorted(opening_docs, key=_score_doc, reverse=True)[:6]
    
    allowed_list = _list_allowed_citations(opening_docs, allowed_pages_by_doc)
    evidence = "\n\n".join(_format_doc_entry(d) for d in opening_docs)
    
    prompt = _build_opening_prompt(topic, stance_summary, evidence, allowed_list)
    
    try:
        chunk = _ollama_chat(prompt)
        chunk, repairs, placeholders, ajr = postprocess_chunk(chunk, opening_docs)
        word_count = _count_words(chunk)
        print(f"[Writer] Opening: {word_count} words")
        chunks.append(chunk)
    except Exception as e:
        print(f"[Writer] Opening failed: {e}")

    # ============================================================
    # GENERATE STANCE SECTIONS
    # ============================================================
    for i, (stance, cluster, cluster_docs, prompt_builder) in enumerate(chunk_plan):
        cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        
        allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        
        previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail)
        
        try:
            chunk = _ollama_chat(prompt)
            chunk, repairs, placeholders, ajr = postprocess_chunk(chunk, cluster_docs_sorted)
            word_count = _count_words(chunk)
            notes = []
            if repairs > 0:
                notes.append(f"repaired {repairs}")
            if placeholders > 0:
                notes.append(f"stripped {placeholders}")
            if ajr > 0:
                notes.append(f"AJR fixed {ajr}")
            note_str = f" ({', '.join(notes)})" if notes else ""
            print(f"[Writer] {stance.upper()}/{cluster}: {word_count} words{note_str}")
            chunks.append(chunk)
        except Exception as e:
            print(f"[Writer] {stance}/{cluster} failed: {e}")

    # ============================================================
    # GENERATE CLOSING
    # ============================================================
    closing_docs = []
    for d in docs:
        if _get_stance(d) == "tangential":
            closing_docs.append(d)
    closing_docs = sorted(closing_docs, key=_score_doc, reverse=True)[:4]
    
    if closing_docs:
        allowed_list = _list_allowed_citations(closing_docs, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in closing_docs)
    else:
        allowed_list = "(No additional citations for closing)"
        evidence = "(No additional evidence for closing)"
    
    previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    prompt = _build_closing_prompt(topic, evidence, allowed_list, previous_tail)
    
    try:
        chunk = _ollama_chat(prompt)
        chunk, repairs, placeholders, ajr = postprocess_chunk(chunk, closing_docs)
        word_count = _count_words(chunk)
        print(f"[Writer] Closing: {word_count} words")
        chunks.append(chunk)
    except Exception as e:
        print(f"[Writer] Closing failed: {e}")

    # ============================================================
    # FINAL ASSEMBLY
    # ============================================================
    full_text = "\n\n".join(chunks)
    
    # Global year-only repair
    global_year_to_docid = _build_year_to_docid(docs)
    full_text, final_repairs = _repair_year_only_citations(full_text, global_year_to_docid)
    total_repairs += final_repairs
    
    # Final AJR fix (in case any slipped through)
    full_text, final_ajr = _fix_ajr_abbreviation(full_text)
    total_ajr_fixes += final_ajr
    
    full_text = _strip_placeholder_citations(full_text)
    full_text, final_dump_cites = _extract_citation_dumps(full_text)
    all_dump_citations.extend(final_dump_cites)
    
    # ============================================================
    # TIER 2: CASE NORMALIZATION (needs full allowed_docs)
    # ============================================================
    full_text, case_fixes = _normalize_citation_case(full_text, allowed_docs)
    if case_fixes > 0:
        print(f"[Writer] Case normalized: {case_fixes} citations")
    
    # ============================================================
    # TIER 3: REMOVE INVALID CITATIONS (needs full allowed_docs)
    # ============================================================
    full_text, removed_citations = _remove_invalid_citations(full_text, allowed_docs)
    if removed_citations:
        print(f"[Writer] Removed {len(removed_citations)} invalid citation(s):")
        for r in removed_citations:
            print(f"         - {r['doc_id']}: p.{r['page']}")
    
    # Final cleanup
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

    ensure_dir("runs")
    out_path = "runs/review_composed.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open("runs/review_cited_docs.json", "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    print(f"[Writer] review_composed.md written ({total_words} words).")
    print(f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} repairs={total_repairs} AJR={total_ajr_fixes} case={case_fixes} removed={len(removed_citations)}")
    
    return out_path


if __name__ == "__main__":
    compose_from_ledger()

