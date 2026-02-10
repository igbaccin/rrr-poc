import os
import json
import re
from collections import Counter, defaultdict
from rrr.utils import ensure_dir

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": float(os.environ.get("RRR_WRITER_T", "0.35")),
    "num_ctx": int(os.environ.get("RRR_WRITER_CTX", "32768")),
    "num_predict": int(os.environ.get("RRR_WRITER_PRED", "2000")),
    "top_p": float(os.environ.get("RRR_WRITER_TOPP", "0.9")),
}

_TAIL_CHARS = int(os.environ.get("RRR_WRITER_TAIL_CHARS", "250"))

_CITE_RE = re.compile(r"([A-Za-z0-9_&.\-]+):\s*p\.(\d+)")

_SYSTEM_CITATION_INSTRUCTION = """CITATION FORMAT IS MANDATORY. You MUST cite using EXACTLY this format: (DocId_Year: p.X)

Examples of CORRECT citations:
- (North_1989: p.9)
- (AcemogluEtAl_2001: p.19)
- (North&Weingast_1989: p.28)
- (Broadberry&Gupta_2006: p.9)

WRONG formats (NEVER use these):
- (2002: p.4) — WRONG, missing author
- Kuznets (1973) — WRONG, must be (Kuznets_1973: p.X)
- (Temin 2002: p.4) — WRONG, missing underscore
- Author et al. (Year) — WRONG
- (Author et al., Year) — WRONG
- (Broadberry & Gardner 2022: p.21) — WRONG, use (Broadberry&Gardner_2022: p.21)
- (Author_Year: p.1, p.2) — WRONG, only ONE page per citation

CRITICAL RULES:
1. Copy document IDs EXACTLY as they appear in the evidence snippets
2. ALWAYS include underscore between name and year: Author_Year
3. ALWAYS include page number with p. prefix
4. Do NOT invent document IDs — only cite what is in the evidence
5. Each citation must have exactly ONE page number
6. Do NOT end paragraphs with bare citations — integrate them into sentences

PROSE QUALITY — Avoid these overused phrases:
- "This perspective underscores..."
- "This finding suggests..."
- "sheds light on..."
- "It is worth noting..."
- "It is important to note..."
- "This assertion is supported by..."
- "In this regard..."
- "In a similar vein..."
- "This notion aligns with..."
- "This sentiment is echoed by..."
- "lends support to..."
- "provides valuable insights..."
- "offers compelling insights..."
- "delves into..."
- "a crucial factor..."
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
# STANCE-AWARE PROMPTS WITH STRICT CITATION FORMAT
# ============================================================

def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, example_str: str):
    return f"""Write the opening section of a literature review on: {topic}

This review examines a scholarly debate. {stance_summary}

CITATION FORMAT — You MUST use this exact format: {example_str}
- Format: (AuthorName_Year: p.PageNumber)
- Copy document IDs EXACTLY from the evidence below
- Every citation needs underscore and page number

Evidence to synthesize:
{evidence}

Requirements:
- 400-500 words
- Frame the central question and why it matters
- Introduce the key positions scholars take
- Cite using EXACT format: (DocId_Year: p.X)
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Begin:"""


def _build_supports_prompt(topic: str, cluster: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present SUPPORTING arguments for the thesis. Theme: {cluster}

CITATION FORMAT — You MUST use this exact format: {example_str}
- Format: (AuthorName_Year: p.PageNumber)
- Copy document IDs EXACTLY from the evidence below
- Every citation needs underscore and page number

Evidence to synthesize (these scholars SUPPORT the thesis):
{evidence}

Requirements:
- 350-450 words
- Present the mechanisms and evidence these scholars offer
- Connect smoothly to previous text
- Cite using EXACT format: (DocId_Year: p.X)
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present COUNTERARGUMENTS to the thesis. Theme: {cluster}

CITATION FORMAT — You MUST use this exact format: {example_str}
- Format: (AuthorName_Year: p.PageNumber)
- Copy document IDs EXACTLY from the evidence below
- Every citation needs underscore and page number

Evidence to synthesize (these scholars CHALLENGE or CRITIQUE the thesis):
{evidence}

Requirements:
- 350-450 words
- Present the objections, alternative explanations, or empirical challenges
- Frame as counterarguments: "However...", "Against this view...", "Critics argue..."
- Connect smoothly to previous text
- Cite using EXACT format: (DocId_Year: p.X)
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Now present NUANCES and QUALIFICATIONS to the thesis. Theme: {cluster}

CITATION FORMAT — You MUST use this exact format: {example_str}
- Format: (AuthorName_Year: p.PageNumber)
- Copy document IDs EXACTLY from the evidence below
- Every citation needs underscore and page number

Evidence to synthesize (these scholars ADD NUANCE or COMPLICATE the thesis):
{evidence}

Requirements:
- 350-450 words
- Present conditional factors, scope conditions, or contextual variations
- Frame as refinements: "The relationship proves more complex when...", "Context matters because..."
- Connect smoothly to previous text
- Cite using EXACT format: (DocId_Year: p.X)
- Write in flowing prose, no bullet points or headers
- End mid-thought for continuation

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Write the closing section of this literature review on: {topic}

Previous text ended with:
...{previous_tail}

CITATION FORMAT — You MUST use this exact format: {example_str}
- Format: (AuthorName_Year: p.PageNumber)
- Copy document IDs EXACTLY from the evidence below
- Every citation needs underscore and page number

Remaining evidence to integrate:
{evidence}

Requirements:
- 300-400 words
- Synthesize the debate: where do scholars agree, where do they diverge?
- Identify gaps in the literature or unresolved questions
- End with directions for future research
- Cite using EXACT format: (DocId_Year: p.X)
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
    
    def postprocess_chunk(chunk, chunk_docs):
        nonlocal total_repairs, total_placeholders_stripped, all_dump_citations
        
        chunk = _strip_wrapping(chunk)
        
        chunk_before = chunk
        chunk = _strip_placeholder_citations(chunk)
        placeholders_stripped = chunk_before.count('DocId_Year') + chunk_before.count('AuthorName_Year')
        total_placeholders_stripped += placeholders_stripped
        
        year_to_docid = _build_year_to_docid(chunk_docs)
        chunk, repair_count = _repair_year_only_citations(chunk, year_to_docid)
        total_repairs += repair_count
        
        chunk, dump_cites = _extract_citation_dumps(chunk)
        all_dump_citations.extend(dump_cites)
        
        chunk = _strip_orphaned_citations(chunk)
        chunk = _strip_references_section(chunk)
        chunk = _strip_continuation_markers(chunk)
        chunk = _strip_conclusion(chunk)
        
        return chunk, repair_count, placeholders_stripped

    # ============================================================
    # GENERATE OPENING
    # ============================================================
    opening_docs = []
    for stance in ["supports", "complicates", "critiques"]:
        for cluster, cluster_docs in stance_buckets[stance].items():
            opening_docs.extend(sorted(cluster_docs, key=_score_doc, reverse=True)[:2])
    opening_docs = sorted(opening_docs, key=_score_doc, reverse=True)[:6]
    
    example_str = _build_example_citations(opening_docs, allowed_pages_by_doc)
    evidence = "\n\n".join(_format_doc_entry(d) for d in opening_docs)
    
    prompt = _build_opening_prompt(topic, stance_summary, evidence, example_str)
    
    try:
        chunk = _ollama_chat(prompt)
        chunk, repairs, placeholders = postprocess_chunk(chunk, opening_docs)
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
        
        example_str = _build_example_citations(cluster_docs_sorted, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        
        previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
        prompt = prompt_builder(topic, cluster, evidence, example_str, previous_tail)
        
        try:
            chunk = _ollama_chat(prompt)
            chunk, repairs, placeholders = postprocess_chunk(chunk, cluster_docs_sorted)
            word_count = _count_words(chunk)
            notes = []
            if repairs > 0:
                notes.append(f"repaired {repairs}")
            if placeholders > 0:
                notes.append(f"stripped {placeholders}")
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
        example_str = _build_example_citations(closing_docs, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in closing_docs)
    else:
        example_str = ""
        evidence = "(No additional evidence for closing)"
    
    previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    prompt = _build_closing_prompt(topic, evidence, example_str, previous_tail)
    
    try:
        chunk = _ollama_chat(prompt)
        chunk, repairs, placeholders = postprocess_chunk(chunk, closing_docs)
        word_count = _count_words(chunk)
        print(f"[Writer] Closing: {word_count} words")
        chunks.append(chunk)
    except Exception as e:
        print(f"[Writer] Closing failed: {e}")

    # ============================================================
    # FINAL ASSEMBLY
    # ============================================================
    full_text = "\n\n".join(chunks)
    
    global_year_to_docid = _build_year_to_docid(docs)
    full_text, final_repairs = _repair_year_only_citations(full_text, global_year_to_docid)
    total_repairs += final_repairs
    
    full_text = _strip_placeholder_citations(full_text)
    full_text, final_dump_cites = _extract_citation_dumps(full_text)
    all_dump_citations.extend(final_dump_cites)
    
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
    print(f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} repairs={total_repairs} placeholders={total_placeholders_stripped}")
    
    return out_path


if __name__ == "__main__":
    compose_from_ledger()

