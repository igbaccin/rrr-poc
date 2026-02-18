import os
import json
import re
from collections import Counter, defaultdict
from rrr.utils import ensure_dir

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": float(os.environ.get("RRR_WRITER_T", "0.50")),  # v7: increased from 0.30
    "num_ctx": int(os.environ.get("RRR_WRITER_CTX", "32768")),
    "num_predict": int(os.environ.get("RRR_WRITER_PRED", "2000")),
    "top_p": float(os.environ.get("RRR_WRITER_TOPP", "0.9")),
}

_TAIL_CHARS = int(os.environ.get("RRR_WRITER_TAIL_CHARS", "250"))

_CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")

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
    "4. If unsure, omit the citation entirely\n"
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
    return f'"{tx}" ({did}: p.{pg})'


def _format_doc_entry(d) -> str:
    # Format doc entry with stance label for writer context.
    did = str(d.get("doc_id", "")).strip()
    stance = d.get("stance", "tangential")
    lines = [f"[{did}] [STANCE: {stance.upper()}]"]
    qs = d.get("quotes") or []
    for q in qs[:4]:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    # Create explicit list of allowed citations for this chunk.
    lines = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        pages = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pages:
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - {did}: {page_str}")
    return "\n".join(lines)


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


def _remove_invalid_citations(text: str, allowed_docs: set) -> tuple:
    # Remove sentences containing citations to documents not in corpus.
    allowed_lower = {did.lower() for did in allowed_docs}
    
    removed = []
    
    def find_invalid_citations_in_text(txt):
        invalid = []
        for m in re.finditer(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', txt):
            doc_id = m.group(1)
            if doc_id.lower() not in allowed_lower:
                invalid.append((m.start(), m.end(), doc_id, m.group(2)))
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
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    
    return cleaned_text, removed


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
    "NEVER begin a sentence with an author name. "
    "NEVER write 'X argues', 'X demonstrates', 'X highlights', 'X supports'. "
    "State the claim, then cite: 'Structural change correlates with growth (Author_Year: p.N).' "
    "All citations must use the (Author_Year: p.N) format. No other citation style."
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

    chunks = []
    all_dump_citations = []
    total_repairs = 0
    total_placeholders_stripped = 0
    total_ajr_fixes = 0
    
    def postprocess_chunk(chunk, chunk_docs):
        nonlocal total_repairs, total_placeholders_stripped, all_dump_citations, total_ajr_fixes
        
        chunk = _strip_wrapping(chunk)
        
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
        
        return chunk, repair_count, placeholders_stripped, ajr_fixes

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
        chunk = _ollama_chat(prompt)
        chunk, repairs, placeholders, ajr = postprocess_chunk(chunk, opening_docs)
        word_count = _count_words(chunk)
        print(f"[Writer] Opening: {word_count} words")
        chunks.append(chunk)
    except Exception as e:
        print(f"[Writer] Opening failed: {e}")

    # Generate stance sections
    for i, (stance, cluster, cluster_docs, prompt_builder) in enumerate(chunk_plan):
        cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        
        allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        
        previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail)

        # DEBUG: Check Austin pages
        if any('Austin_2008' in str(d.get('doc_id', '')) for d in cluster_docs_sorted):
            print("=== DEBUG: AUSTIN IN THIS CHUNK ===")
            print(f"Stance: {stance}, Cluster: {cluster}")
            print(f"Allowed list:\n{allowed_list}")
            print("=== END DEBUG ===")
        
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

    # Generate closing
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

    # Final assembly
    full_text = "\n\n".join(chunks)
    
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
    
    full_text, removed_citations = _remove_invalid_citations(full_text, allowed_docs)
    if removed_citations:
        print(f"[Writer] Removed {len(removed_citations)} invalid citation(s):")
        for r in removed_citations:
            print(f"         - {r['doc_id']}: p.{r['page']}")
    
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
