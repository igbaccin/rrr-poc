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

_SYSTEM_CITATION_INSTRUCTION = """You MUST cite using EXACTLY this format: (DocId_Year: p.X)

Examples of CORRECT format:
- (NorthEtAl_2006: p.17)
- (Broadberry&Gardner_2022: p.1)
- (Acemoglu&Johnson_2005: p.23)
- (AcemogluEtAl_2002: p.49)

WRONG formats (NEVER use these):
- (2002: p.4) — WRONG, missing author prefix
- (2006: p.17) — WRONG, missing author prefix
- North et al. (2006) — WRONG
- (North et al., 2006) — WRONG
- (North et al. 2006: 17) — WRONG
- Broadberry and Gardner (2022) — WRONG
- (NorthEtAl_2006: p.3, p.17, p.4) — WRONG (only cite ONE page per parenthesis)

CRITICAL: The document ID MUST include the author name(s), not just the year.
WRONG: (2002: p.4)
CORRECT: (AcemogluEtAl_2002: p.4)

Copy the document ID exactly as shown in the evidence snippets."""


def _get_cluster(d):
    return d.get("cluster", "Other") or "Other"


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
    did = str(d.get("doc_id", "")).strip()
    lines = [f"[{did}]"]
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
        
        # Pattern 1: Single parenthesis with multiple comma-separated citations
        if stripped.startswith('(') and stripped.endswith(')') and ',' in stripped:
            inner = stripped[1:-1]
            cite_matches = re.findall(r'[A-Za-z0-9_&]+_\d{4}[a-z]?:\s*p\.\d+', inner)
            if len(cite_matches) >= 2:
                for m in cite_matches:
                    did = m.split(':')[0]
                    dump_citations.append(did)
                continue
            cite_matches_academic = re.findall(r'[A-Za-z&\s]+_\d{4}[a-z]?:\s*p\.\d+', inner)
            if len(cite_matches_academic) >= 2:
                for m in cite_matches_academic:
                    did = m.split(':')[0].strip()
                    dump_citations.append(did)
                continue
        
        # Pattern 2: Single doc with multiple pages
        if stripped.startswith('(') and stripped.endswith(')'):
            inner = stripped[1:-1]
            page_refs = re.findall(r'p\.\d+', inner)
            if len(page_refs) >= 3:
                doc_match = re.match(r'([A-Za-z0-9_&]+)', inner)
                if doc_match:
                    dump_citations.append(doc_match.group(1))
                continue
        
        # Pattern 3: Multiple separate parenthetical citations on one line
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


def _build_example_citations_for_chunk(cluster_docs, allowed_pages_by_doc):
    """Build example citations from THIS chunk's docs only."""
    examples = []
    for d in cluster_docs[:3]:
        did = str(d.get("doc_id", "")).strip()
        pgs = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pgs:
            examples.append(f"({did}: p.{pgs[0]})")
    return ", ".join(examples) if examples else "(DocId_Year: p.X)"


def _build_year_to_docid_for_chunk(cluster_docs):
    """
    Build year -> doc_id mapping for a chunk.
    Only includes unambiguous mappings (one doc per year).
    """
    year_to_docs = defaultdict(list)
    for d in cluster_docs:
        did = str(d.get("doc_id", "")).strip()
        if not did:
            continue
        # Extract year from doc_id (last 4 digits before optional letter suffix)
        m = re.search(r'_(\d{4})[a-z]?$', did)
        if m:
            year = m.group(1)
            year_to_docs[year].append(did)
    
    # Only return unambiguous mappings
    return {year: docs[0] for year, docs in year_to_docs.items() if len(docs) == 1}


def _repair_year_only_citations(text: str, year_to_docid: dict) -> tuple:
    """
    Repair (YEAR: p.X) -> (DocId_Year: p.X) using chunk context.
    
    Returns (repaired_text, repair_count).
    """
    repair_count = 0
    
    def replacer(m):
        nonlocal repair_count
        year = m.group(1)
        page = m.group(2)
        if year in year_to_docid:
            repair_count += 1
            return f"({year_to_docid[year]}: p.{page})"
        return m.group(0)  # Leave unchanged if ambiguous
    
    repaired = re.sub(r'\((\d{4}):\s*p\.(\d+)\)', replacer, text)
    return repaired, repair_count


def _build_opening_prompt(topic: str, cluster: str, evidence: str, example_str: str):
    return f"""Write the opening section of a literature review on: {topic}

Theme: {cluster}

CITATION FORMAT — Copy document IDs exactly as shown: {example_str}
WRONG: (2002: p.4) — Never use year-only citations!

Evidence to synthesize:
{evidence}

Requirements:
- 400-500 words
- Cite using EXACT document IDs from the evidence: (AuthorName_Year: p.X)
- Focus on the topic, not author names as sentence subjects
- No headings, no bullets
- End mid-thought for continuation

Begin:"""


def _build_continuation_prompt(topic: str, cluster: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Continue this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Next theme: {cluster}

CITATION FORMAT — Copy document IDs exactly as shown: {example_str}
WRONG: (2002: p.4) — Never use year-only citations!

Evidence to synthesize:
{evidence}

Requirements:
- 400-500 words
- Cite using EXACT document IDs from the evidence: (AuthorName_Year: p.X)
- Connect smoothly to previous text
- Focus on the topic, not author names as sentence subjects
- No headings, no bullets
- End mid-thought for continuation

Continue:"""


def _build_closing_prompt(topic: str, cluster: str, evidence: str, example_str: str, previous_tail: str):
    return f"""Write the final section of this literature review on: {topic}

Previous text ended with:
...{previous_tail}

Final theme: {cluster}

CITATION FORMAT — Copy document IDs exactly as shown: {example_str}
WRONG: (2002: p.4) — Never use year-only citations!

Evidence to synthesize:
{evidence}

Requirements:
- 400-500 words
- Cite using EXACT document IDs from the evidence: (AuthorName_Year: p.X)
- Connect smoothly to previous text
- Focus on the topic, not author names as sentence subjects
- No headings, no bullets
- End with open questions for future research
- Do NOT write "In conclusion"

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
    
    # Correct format: (DocId: p.X)
    for m in re.finditer(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    # Format without page: (DocId_Year)
    for m in re.finditer(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    # Academic format (parenthetical): (Author et al., Year)
    for m in re.finditer(r"\(([A-Za-z&]+(?:\s+et\s+al\.?)?)[,\s]+(\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    # Academic format (inline): Author et al. (Year)
    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    # Also catch inline academic with page: Author (Year: p.X)
    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4}):\s*p\.\d+\)", text):
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

    # Bucket docs by cluster
    buckets = defaultdict(list)
    for d in docs:
        buckets[_get_cluster(d)].append(d)

    # Rank clusters by total relevance score
    cluster_rank = sorted(
        buckets.items(),
        key=lambda kv: sum(_score_doc(x) for x in kv[1]),
        reverse=True
    )

    # Limit to top clusters
    max_clusters = int(os.environ.get("RRR_WRITER_MAX_CLUSTERS", "8"))
    cluster_rank = cluster_rank[:max_clusters]

    if not cluster_rank:
        raise SystemExit("No clusters found.")

    print(f"[Writer] Generating {len(cluster_rank)} chunks (one per theme)...")

    chunks = []
    all_dump_citations = []
    total_repairs = 0
    
    for i, (cluster, cluster_docs) in enumerate(cluster_rank):
        cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        
        # Build examples and year->docid mapping for THIS chunk
        example_str = _build_example_citations_for_chunk(cluster_docs_sorted, allowed_pages_by_doc)
        year_to_docid = _build_year_to_docid_for_chunk(cluster_docs_sorted)
        
        evidence_lines = []
        for d in cluster_docs_sorted:
            evidence_lines.append(_format_doc_entry(d))
        evidence = "\n\n".join(evidence_lines)

        if i == 0:
            prompt = _build_opening_prompt(topic, cluster, evidence, example_str)
        elif i == len(cluster_rank) - 1:
            previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
            prompt = _build_closing_prompt(topic, cluster, evidence, example_str, previous_tail)
        else:
            previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
            prompt = _build_continuation_prompt(topic, cluster, evidence, example_str, previous_tail)

        try:
            chunk = _ollama_chat(prompt)
        except Exception as e:
            print(f"[Writer] Chunk {i+1} failed: {e}")
            continue

        chunk = _strip_wrapping(chunk)
        
        # REPAIR year-only citations using chunk context
        chunk, repair_count = _repair_year_only_citations(chunk, year_to_docid)
        total_repairs += repair_count
        
        # Extract citation dumps before stripping
        chunk, dump_cites = _extract_citation_dumps(chunk)
        all_dump_citations.extend(dump_cites)
        
        # Strip formal references sections
        chunk = _strip_references_section(chunk)
        
        # Strip conclusions from non-final chunks
        if i < len(cluster_rank) - 1:
            chunk = _strip_conclusion(chunk)

        word_count = _count_words(chunk)
        repair_note = f", repaired {repair_count}" if repair_count > 0 else ""
        print(f"[Writer] Chunk {i+1}/{len(cluster_rank)} ({cluster}): {word_count} words{repair_note}")

        chunks.append(chunk)

    # Concatenate chunks
    full_text = "\n\n".join(chunks)
    
    # Build global year->docid for final pass (using all docs)
    global_year_to_docid = {}
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        if did:
            m = re.search(r'_(\d{4})[a-z]?$', did)
            if m:
                year = m.group(1)
                # Only add if not already present (avoid ambiguity)
                if year not in global_year_to_docid:
                    global_year_to_docid[year] = did
    
    # Final repair pass on full text
    full_text, final_repairs = _repair_year_only_citations(full_text, global_year_to_docid)
    total_repairs += final_repairs
    
    # Final cleanup passes
    full_text, final_dump_cites = _extract_citation_dumps(full_text)
    all_dump_citations.extend(final_dump_cites)
    
    full_text = _strip_references_section(full_text)
    full_text = _strip_continuation_markers(full_text)
    full_text = _strip_conclusion(full_text)
    
    # Clean up extra whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Collect all cited docs
    cited_docs = _collect_cited_docs(full_text, allowed_docs, author_year_to_docid)
    
    # Add dump citations that are in allowed_docs
    for did in all_dump_citations:
        if did in allowed_docs:
            cited_docs.add(did)

    # Dedupe and sort
    cited_docids = sorted(cited_docs)

    # Stats
    total_words = _count_words(full_text)

    ensure_dir("runs")
    out_path = "runs/review_composed.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Save cited docs for reference builder
    with open("runs/review_cited_docs.json", "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    print(f"[Writer] review_composed.md written ({total_words} words).")
    print(f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} citations_repaired={total_repairs}")
    
    return out_path


if __name__ == "__main__":
    compose_from_ledger()

