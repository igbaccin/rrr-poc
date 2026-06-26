import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from rrr.utils import ensure_dir
from rrr.paths import runs_path
from rrr.render import (
    CITE_RE,
    DISPLAY_CITE_RE,
    DISPLAY_PAREN_CITE_RE,
    parse_citations,
    render_citation,
    render_citation_canonical,
    render_narrative_citation,
    render_parenthetical_citation,
    rewrite_canonical_to_display,
    rewrite_misplaced_narrative_citations,
    _build_display_lookup,
    _doc_id_to_author_label,
)

_MODEL = os.environ.get("RRR_MODEL", "mistral")
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": float(os.environ.get("RRR_WRITER_T", "0.50")),  # v7: increased from 0.30
    # v8: writer ctx default 12288 (was 32768). Largest observed writer prompt
    # is ~2900 tokens; 32k KV-cache allocation slows generation 15-25% with no
    # quality gain. Override RRR_WRITER_CTX if a topic needs a much larger
    # allowed-citation list.
    "num_ctx": int(os.environ.get("RRR_WRITER_CTX", "12288")),
    # v8: writer pred default 1500 (was 2000). Largest observed output is
    # ~620 tokens; 1500 leaves 2.4x slack.
    "num_predict": int(os.environ.get("RRR_WRITER_PRED", "1500")),
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
    r"shed light|complex and influenced by various factors|"
    # v8: NGO/consulting register that leaks through current directives
    r"holds significance|potential implications|strategies aimed at|"
    r"sustainable development|alleviating poverty|this investigation|"
    r"this analysis|this study|this examination|plays a pivotal role|"
    r"contributing significantly to|fundamental factor|crucial factor|"
    r"essential factor|interplay between|in this regard|in this context|"
    r"by acknowledging these variables|presents a complex picture|"
    r"thereby contributing"
    r")\b",
    re.IGNORECASE,
)

# v8 (R5) / v9.1: catch nested citation wraps that escape every other cleanup
# because CITE_RE matches only the single-paren form. Two forms observed:
#   1. ((Doc: p.N))              — simple double-wrap (v8)
#   2. ((Doc: p.X): p.Y)         — doubled page-suffix; happens when the LLM
#      writes ([E0001]: p.X) and _render_evidence_id_citations turns [E0001]
#      into (Doc: p.X), producing the nested form. v9 introduced two of these
#      because the fused-call writer prompt provides richer evidence context
#      that the model wraps in parens. (v9.1 fix.)
_DOUBLE_PAREN_CITE_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\)\)"
)
_NESTED_PAGE_SUFFIX_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+):\s*p\.\d+\):\s*p\.\d+\)"
)

# v10: detect-then-LLM-rewrite style enforcement. We deliberately do NOT
# substitute forbidden words mechanically because the right replacement is
# sense-dependent (e.g. "sharp distinction" vs "sharply declined"). Sentences
# containing any of these patterns are routed to a single batched LLM rewrite
# call per writer section.
_FORBIDDEN_LEXEME_RE = re.compile(r"\b(?:legible|sharp(?:ly)?)\b", re.IGNORECASE)
_ADVERSATIVE_RE = re.compile(
    r"\b("
    r"not\s+\S+(?:\s+\S+){0,5}\s+but(?:\s+rather)?\s+"
    r"|rather than\b"
    r"|in (?:sharp\s+)?contrast(?:\s+to)?\b"
    r"|instead of\b"
    r"|unlike\b"
    r")",
    re.IGNORECASE,
)
# v10 (rule 6): trailing significance clauses — these ARE safely strippable
# because the head of the sentence carries the substantive claim.
_TRAILING_SIGNIFICANCE_RE = re.compile(
    r"[,.;]?\s+"
    r"(?:[Aa]nd that matters"
    r"|[Ww]hich matters because[^.!?]*"
    r"|[Tt]his is important because[^.!?]*"
    r"|[Tt]his matters because[^.!?]*)"
    r"[.!?]?",
)
# v10 (rule 8): colon-heavy academic setup ("A long stem clause: A new clause").
_COLON_SETUP_RE = re.compile(r"[A-Z][^.!?:\n]{20,}:\s+[A-Z]")
# v10 (rule 1): em dash + en dash. Both go through the LLM rewriter because the
# right replacement varies (comma, period, parens, semicolon).
_TYPOGRAPHIC_DASH_RE = re.compile("[–—]")
# v10 options for the short style-rewrite LLM call.
_OPTIONS_STYLE_REWRITE = {
    "temperature": 0.1,
    "num_ctx": int(os.environ.get("RRR_STYLE_CTX", "4096")),
    "num_predict": int(os.environ.get("RRR_STYLE_PRED", "1500")),
}

# v8 (R5): detect "Author argues / posits / emphasizes ..." openings that the
# system prompt forbids but the model keeps producing. Used to flag, then
# optionally rewrite, sections that violate.
_AUTHOR_VERB_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)\s*[A-Z][A-Za-z&.\-]+)(?:\s+et\s+al\.?)?(?:\s*\(?[A-Za-z_]*\d{4}[A-Za-z_]*\)?)?)\s+"
    r"(argues|emphasizes|emphasises|demonstrates|highlights|posits|suggests|"
    r"claims|notes|observes|maintains|asserts|contends|shows|writes|states|"
    r"finds|points out|underscores|argues that|notes that)\b",
    re.IGNORECASE,
)
_MIN_SECTION_CITED_DOCS = int(os.environ.get("RRR_WRITER_MIN_SECTION_CITED_DOCS", "2"))
_ENFORCE_COVERAGE = os.environ.get("RRR_WRITER_ENFORCE_COVERAGE", "1") != "0"

# v10: cite as 'Author (Year, p.N)'. Underscore-keyed canonical form retired.
_SYSTEM_CITATION_INSTRUCTION = (
    "CITATION FORMATS: two forms, picked by sentence role.\n"
    "  NARRATIVE — author is the grammatical subject of the sentence:\n"
    "    Author (Year, p.N)\n"
    "    e.g. 'North and Weingast (1989, p.2) argue that institutions...'\n"
    "  PARENTHETICAL — whole reference in parens, when the author is NOT\n"
    "    the subject (citing as an aside, or citing multiple sources for a\n"
    "    shared claim):\n"
    "    (Author Year, p.N)\n"
    "    e.g. '...credible commitment underpins growth (North and Weingast 1989, p.2).'\n"
    "    Multi-source: '...established across the literature (North 1989, p.9; "
    "Acemoglu et al. 2001, p.27).'\n\n"
    "Use the narrative form when the author is the speaker in the sentence; "
    "use the parenthetical form otherwise.\n\n"
    "ALWAYS WRONG (never emit these):\n"
    "  (North, 1989)             # missing page\n"
    "  North (1989) p.9          # page outside the parens\n"
    "  (p.5)                     # page-only, no author/year\n\n"
    "Rules:\n"
    "1. Copy each author label EXACTLY as it appears in the ALLOWED CITATIONS list below.\n"
    "2. Use only the page numbers shown next to that author in the list.\n"
    "3. Every citation must include a page number; never write a citation without a page.\n"
    "4. One page per citation; never write 'pp.5-7' or 'pp.5, 7'.\n"
    "5. Never write page-only citations such as (p.5).\n"
    "6. If a claim is not supported by the allowed evidence, state it without a citation; do not invent one.\n"
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
    # v9 (R3): when the fused stance+mechanism call produced rationale /
    # mechanism / contested fields, surface them so the writer prompt has
    # substantive per-doc anchors instead of just the stance label. Falls back
    # silently when those fields are absent (legacy two-call path).
    # v10: include the human-readable author-year label so the writer copies
    # 'Acemoglu et al. (2002, p.5)' directly instead of inventing surface form.
    did = str(d.get("doc_id", "")).strip()
    stance = d.get("stance", "tangential")
    author_label = _doc_id_to_author_label(did)
    header = f"[{did}] cite as {author_label!s}  [STANCE: {stance.upper()}]"
    lines = [header]
    lead_mechanism = str(d.get("mechanism", "") or "").strip()
    if lead_mechanism:
        lines.append(f"  Lead mechanism: {_clip(lead_mechanism, n=180)}")
    rationale = str(d.get("rationale", "") or "").strip()
    if rationale:
        lines.append(f"  Why this stance: {_clip(rationale, n=220)}")
    contested = str(d.get("contested", "") or "").strip()
    if contested:
        lines.append(f"  What this contests: {_clip(contested, n=220)}")
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
    # v10: emit display form 'Author (Year, p.N)' so the writer can copy-paste
    # directly. Each line maps an evidence_id to its rendered display citation.
    # When the ledger lacks evidence_ids we fall back to a per-doc summary:
    # 'Acemoglu et al. (2002): pp.5, 27, 33'.
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
            author_label = _doc_id_to_author_label(did)
            # strip trailing ')' from the label and re-emit with pages
            base = author_label[:-1] if author_label.endswith(")") else author_label
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - {base}, {page_str})")
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


def _collapse_double_parens(text: str) -> tuple:
    # v8 (R5): collapse ((DocId: p.N)) -> (DocId: p.N). Returns (text, count).
    # v9.1: also collapse ((DocId: p.X): p.Y) -> (DocId: p.X). The inner page
    # wins because it was the model's own evidence-id-rendered citation, while
    # the outer ": p.Y" is the model's redundant attempt at a page suffix.
    count = 0

    def repl_double(m):
        nonlocal count
        count += 1
        return "(" + m.group(1) + ")"

    def repl_nested_suffix(m):
        nonlocal count
        count += 1
        # m.group(1) is just the DocId; we need to preserve the original inner
        # page so reconstruct it by re-running the inner regex on the match.
        inner = re.match(r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\):\s*p\.\d+\)", m.group(0))
        if inner:
            return "(" + inner.group(1) + ")"
        return m.group(0)

    text, _ = (_DOUBLE_PAREN_CITE_RE.subn(repl_double, text))
    text, _ = (_NESTED_PAGE_SUFFIX_RE.subn(repl_nested_suffix, text))
    return text, count


def _count_author_led_openings(text: str) -> int:
    # v8 (R5): count sentences that open with "Author argues / posits / ...".
    # v10: kept as metric only; user explicitly ditched the prompt rule because
    # mistral-small:24b and above produce them sparingly, in natural academic
    # register, not as a parade pattern.
    count = 0
    for sentence in _split_sentences_for_cleanup(text):
        s = sentence.strip()
        if not s:
            continue
        head = re.sub(r"^[\(\[\"\'\s]+", "", s)[:120]
        if _AUTHOR_VERB_RE.search(head):
            count += 1
    return count


# ============================================================================
# v10: STYLE ENFORCEMENT — detect-then-LLM-rewrite for ambiguous violations.
# Strippable patterns (trailing significance) are handled mechanically first;
# the remaining sentences with em-dashes, forbidden lexemes, adversative
# framings, or colon setups are routed to a single batched LLM rewrite call
# per writer postprocess pass.
# ============================================================================

def _strip_trailing_significance(text: str) -> tuple:
    # v10 (rule 6): mechanically strip "and that matters" / "which matters
    # because ..." trailing clauses. These have a stable shape and live at end
    # of sentence; removing them does not change the substantive claim head.
    matches = list(_TRAILING_SIGNIFICANCE_RE.finditer(text or ""))
    if not matches:
        return text or "", 0
    cleaned = _TRAILING_SIGNIFICANCE_RE.sub("", text)
    # Recompose terminal punctuation if the strip removed it.
    cleaned = re.sub(r"([A-Za-z0-9_\)\]])(?=\n|$)", r"\1.", cleaned)
    return cleaned, len(matches)


def _classify_sentence_violations(sentence: str) -> list:
    reasons = []
    if _TYPOGRAPHIC_DASH_RE.search(sentence):
        reasons.append("em_dash")
    if _FORBIDDEN_LEXEME_RE.search(sentence):
        reasons.append("forbidden_lexeme")
    if _ADVERSATIVE_RE.search(sentence):
        reasons.append("adversative")
    if _COLON_SETUP_RE.search(sentence):
        reasons.append("colon_setup")
    return reasons


def _collect_style_violations(text: str):
    """Return list of (sentence_index, sentence_text, [reasons]) for the
    sentences in `text` that warrant LLM rewrite. Paragraph structure is
    preserved by walking sentences in order; the rewriter splices results back
    by index.
    """
    sentences = _split_sentences_for_cleanup(text)
    violations = []
    for idx, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue
        reasons = _classify_sentence_violations(s)
        if reasons:
            violations.append((idx, s, reasons))
    return sentences, violations


def _build_style_rewrite_prompt(violations) -> str:
    """One short prompt covering all flagged sentences from this section."""
    parts = [
        "You are revising sentences from a literature review. Rewrite each "
        "numbered sentence below so that it expresses the same meaning while "
        "observing ALL of the following constraints:",
        "",
        "- Do not use em dashes or en dashes. Use commas, periods, semicolons, "
        "or parentheses.",
        "- Do not use the words 'legible', 'sharp', or 'sharply'.",
        "- Do not use adversative framings as the default prose engine. Avoid "
        "'not X but Y', 'rather than', 'unlike', 'in contrast', and "
        "contrastive 'instead of'.",
        "- Do not introduce a claim with a colon followed by an explanatory "
        "clause. Use a full sentence.",
        "- Preserve every 'Author (Year, p.N)' citation exactly as written.",
        "- Preserve the substantive meaning. Do not add or remove evidence.",
        "- Keep prose natural and varied. Do not produce staccato short "
        "sentences in place of the originals.",
        "",
        "Sentences:",
    ]
    for n, (_, sent, _) in enumerate(violations, start=1):
        parts.append(f"{n}. {sent}")
    parts.append("")
    parts.append(
        "Return ONLY a JSON object with one key 'rewritten' whose value is a "
        "JSON array of strings, the rewritten sentences in the same order. "
        "Do not include any commentary."
    )
    return "\n".join(parts)


def _rewrite_style_violations(sentences, violations, metrics=None):
    """Run one batched LLM call to rewrite all flagged sentences.

    Returns a new list of sentences (same length as input). On any failure the
    original sentences are returned unchanged.
    """
    if not violations:
        return sentences, 0, "no_violations"
    prompt = _build_style_rewrite_prompt(violations)
    try:
        import ollama
        import time
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system",
                 "content": "You revise academic prose. Be concise. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            options=_OPTIONS_STYLE_REWRITE,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("style_rewrite", _MODEL, options=_OPTIONS_STYLE_REWRITE,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("style_rewrite", _MODEL, options=_OPTIONS_STYLE_REWRITE,
                               success=False, error=e)
        return sentences, 0, f"llm_call_failed:{type(e).__name__}"

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return sentences, 0, "json_braces_missing"
        obj = json.loads(raw[start:end + 1])
        rewritten = obj.get("rewritten", [])
        if not isinstance(rewritten, list) or len(rewritten) != len(violations):
            return sentences, 0, "wrong_count"
    except Exception:
        return sentences, 0, "json_parse_failed"

    new_sentences = list(sentences)
    rewrites_applied = 0
    for (idx, original, _reasons), new_text in zip(violations, rewritten):
        new_text = (new_text or "").strip()
        if not new_text:
            continue
        # Refuse a rewrite that REINTRODUCES the same violation (model loop).
        if _classify_sentence_violations(new_text):
            continue
        # Refuse a rewrite that drops or changes any citation token. We compare
        # the set of '(Year, p.N)' tuples present before and after.
        original_cites = set(re.findall(r"\((\d{4})[a-z]?,\s*p\.\s*(\d+)\)", original))
        new_cites = set(re.findall(r"\((\d{4})[a-z]?,\s*p\.\s*(\d+)\)", new_text))
        if original_cites != new_cites:
            continue
        new_sentences[idx] = new_text
        rewrites_applied += 1

    return new_sentences, rewrites_applied, "ok"


def _splice_sentences_back(original_text: str, new_sentences) -> str:
    """Reconstruct text from sentence array, preserving paragraph breaks.

    The original splitter is sentence-level only; we use paragraph breaks from
    the input to chunk sentences back into the same shape.
    """
    if not original_text:
        return ""
    paragraphs = original_text.split("\n\n")
    out_paragraphs = []
    cursor = 0
    for para in paragraphs:
        original_sents = _split_sentences_for_cleanup(para)
        n = len(original_sents)
        replacement = new_sentences[cursor:cursor + n]
        cursor += n
        if replacement:
            out_paragraphs.append(" ".join(s.strip() for s in replacement if s.strip()))
        else:
            out_paragraphs.append(para)
    return "\n\n".join(out_paragraphs).strip()


def _apply_style_enforcement(text: str, metrics=None):
    """v10 single entry point for prose-style cleanup.

    Order matters:
      1. Strip the trailing-significance clause shape mechanically. Safe.
      2. Detect remaining violations (em-dash, lexeme, adversative, colon).
      3. Batch-LLM-rewrite the flagged sentences and splice back.

    Returns (text, stats_dict). Stats keys: 'trailing_stripped', 'violations',
    'rewrites_applied', 'fallback_reason'.
    """
    if not text or os.environ.get("RRR_STYLE_ENFORCE", "1") == "0":
        return text, {"trailing_stripped": 0, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "disabled"}

    text, trailing_stripped = _strip_trailing_significance(text)

    sentences, violations = _collect_style_violations(text)
    if not violations:
        return text, {"trailing_stripped": trailing_stripped, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "no_violations"}

    new_sentences, applied, reason = _rewrite_style_violations(
        sentences, violations, metrics=metrics,
    )
    if applied == 0:
        return text, {"trailing_stripped": trailing_stripped,
                      "violations": len(violations),
                      "rewrites_applied": 0,
                      "fallback_reason": reason}
    rewritten_text = _splice_sentences_back(text, new_sentences)
    return rewritten_text, {"trailing_stripped": trailing_stripped,
                            "violations": len(violations),
                            "rewrites_applied": applied,
                            "fallback_reason": reason}


# v9 (R6): word-level content tokens for "shares >=K content tokens with an
# earlier sentence" check. Strips punctuation and short stopwords so the
# overlap measure reflects substantive content rather than function-word
# coincidence. Tuned to be permissive — we only drop on STRICT-SUBSET citation
# overlap PLUS strong content overlap, so a high bar here is fine.
_REDUNDANCY_STOPWORDS = frozenset({
    "the", "and", "of", "in", "to", "a", "an", "is", "are", "was", "were",
    "for", "on", "by", "with", "as", "at", "from", "that", "this", "these",
    "those", "it", "its", "their", "they", "we", "be", "been", "has", "have",
    "had", "or", "but", "not", "no", "nor", "if", "than", "then", "so", "such",
    "into", "out", "over", "under", "between", "among", "while", "when",
    "where", "which", "who", "whom", "whose", "what", "how", "why", "also",
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "instance", "example", "p", "pp",
})


def _redundancy_tokens(sentence: str) -> set:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]+", sentence.lower())
    return {t for t in raw if len(t) >= 4 and t not in _REDUNDANCY_STOPWORDS}


def _sentence_citation_pairs(sentence: str):
    pairs = set()
    for m in re.finditer(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)", sentence):
        try:
            pairs.add((m.group(1), int(m.group(2))))
        except Exception:
            continue
    return pairs


def _drop_cross_section_redundancy(text: str, min_token_overlap: int = 4):
    """v9 (R6): walk the assembled review sentence by sentence; drop a sentence
    when BOTH:
      (a) every (doc, page) citation it contains has already been cited earlier
          in the document (strict-subset rule); AND
      (b) it shares >= min_token_overlap content tokens with at least one
          earlier kept sentence (high content overlap).
    Sentences with at least one novel citation are always kept; sentences
    citing already-seen evidence are kept if they introduce a genuinely
    distinct content angle. Returns (cleaned_text, removed_examples).
    """
    if not text:
        return text, []
    # Process paragraph by paragraph so we don't merge cross-paragraph text.
    paragraphs = text.split("\n\n")
    seen_pairs: set = set()
    seen_tokens: list = []  # list of token-sets per kept sentence
    removed_examples = []
    new_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        kept_lines = []
        for line in lines:
            sentences = _split_sentences_for_cleanup(line)
            if not sentences:
                kept_lines.append(line)
                continue
            kept_sentences = []
            for sent in sentences:
                sent_pairs = _sentence_citation_pairs(sent)
                # No citation: keep (we never drop uncited prose here)
                if not sent_pairs:
                    kept_sentences.append(sent)
                    continue
                # Any novel citation: keep, register pairs
                if not sent_pairs.issubset(seen_pairs):
                    kept_sentences.append(sent)
                    seen_pairs |= sent_pairs
                    seen_tokens.append(_redundancy_tokens(sent))
                    continue
                # All citations already seen: check token overlap
                this_tokens = _redundancy_tokens(sent)
                max_overlap = max((len(this_tokens & prev) for prev in seen_tokens), default=0)
                if max_overlap >= min_token_overlap:
                    snippet = re.sub(r"\s+", " ", sent.strip())[:140]
                    removed_examples.append({
                        "snippet": snippet + ("…" if len(snippet) == 140 else ""),
                        "overlap_tokens": int(max_overlap),
                        "cited_pairs": sorted(list(sent_pairs)),
                    })
                    continue
                # Same citation set but distinct content: keep
                kept_sentences.append(sent)
                seen_tokens.append(this_tokens)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences))
        if kept_lines:
            new_paragraphs.append("\n".join(kept_lines))
    cleaned = "\n\n".join(new_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed_examples


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
    "Write in a scholarly register suited to historical methods. "
    "Argue in direct positive claims; do not lean on contrastive framings such as "
    "'not X but Y', 'rather than', 'unlike', or 'in contrast'. "
    "Vary sentence length; do not produce runs of short staccato sentences. "
    "Introduce claims as full sentences rather than via colon setups. "
    "You may cite an evidence ID such as [E0001]; it will be rendered into a validated page citation. "
    "When multiple sources in the evidence support the same point, cite them together in one "
    "parenthetical reference rather than presenting each as a separate witness — e.g. "
    "'the institutional turn is established across the literature ([E0001]; [E0007]; [E0014]).' "
    "Save the named, narrative form 'Author (Year, p.N) argues that...' for moments when one "
    "source's specific argument is the subject of the sentence."
)

# v8 (R4): single canonical set of prompt builders. The previous file had two
# definitions per builder — the first was dead code shadowed by the second.
# Each stance builder now has a structurally DISTINCT instruction so the
# resulting paragraphs read as substantively different arguments rather than
# three parallel templates with one verb swapped.

def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Literature review on: {topic}

{stance_summary}

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Open the review. State the substantive question and what is at stake intellectually (not in NGO terms). Name the live disagreement: what positions exist, where they conflict, without listing them as a survey. 200-300 words. End mid-thought; the argument continues.

Begin:"""


def _format_cluster_synthesis_block(synth, doc_to_evidence_ids) -> str:
    """v11-C: render a cluster synthesis dict into a prompt-ready block.

    Translates supporting/qualifying doc_ids into evidence IDs ([E####]) so the
    writer can emit a multi-citation parenthetical when stating the shared
    claim. Returns an empty string when there is no synthesis (singleton
    cluster, "Other" bucket, RRR_CLUSTER_SYNTHESIS=0, or LLM failure)."""
    if not synth or not isinstance(synth, dict):
        return ""

    def _eids(doc_ids):
        out = []
        for did in doc_ids or []:
            ids = doc_to_evidence_ids.get(did) or []
            if ids:
                out.append(ids[0])
        return out

    supporting_eids = _eids(synth.get("supporting_doc_ids", []))
    qualifying_eids = _eids(synth.get("qualifying_doc_ids", []))
    shared = (synth.get("shared_mechanism") or "").strip()
    contested = (synth.get("contested_dimension") or "").strip()

    if not shared or len(supporting_eids) < 2:
        return ""

    cite_block = "; ".join(f"[{e}]" for e in supporting_eids[:5])
    lines = [
        "CLUSTER SYNTHESIS (use ONE parenthetical multi-citation when stating the shared claim):",
        f"  Shared mechanism: {shared}",
        f"  Supporting sources to cite together: ({cite_block})",
    ]
    if qualifying_eids:
        qual_block = "; ".join(f"[{e}]" for e in qualifying_eids[:3])
        lines.append(
            f"  Sources that qualify or scope the claim (cite as a follow-up "
            f"sentence, not in the main multi-citation): ({qual_block})"
        )
    if contested:
        lines.append(f"  Internal disagreement to acknowledge: {contested}")
    return "\n".join(lines)


def _build_supports_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                           previous_tail: str, synthesis_block: str = ""):
    # v8 (R4) — distinct shape: single chain of inference, ~260-300 words.
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources SUPPORT the thesis through corroborating mechanisms, measurements, or historical conditions.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Build one chain of inference. Open by stating the shared mechanism the cluster holds in common and cite the supporting sources together in one parenthetical (per the CLUSTER SYNTHESIS above if present, otherwise the strongest 2-3 sources in Evidence). Then narrow to one source for a specific quoted phrase that establishes the mechanism; show how a second source confirms it under different conditions (period, region, measurement); then trace one downstream implication that follows. Do not list. Do not restate the thesis. 260-300 words. End mid-thought.

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                            previous_tail: str, synthesis_block: str = ""):
    # v8 (R4) — distinct shape: force a choice. Shorter, sharper.
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources CHALLENGE the thesis by identifying weak assumptions, alternative causes, or contrary evidence.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Identify the specific assumption the supporting case rests on, then quote the evidence that breaks it inside the same sentence, not after it. When the CLUSTER SYNTHESIS above identifies a shared critical move across multiple sources, name it once with a multi-citation parenthetical. Force the reader to weigh the assumption against the counter-evidence; do not list qualifications. Do not restate the thesis. 200-240 words. End mid-thought.

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                              previous_tail: str, synthesis_block: str = ""):
    # v8 (R4) — distinct shape: pick the binding scope condition and develop it.
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    return f"""Continue this literature review on: {topic}

Previous ending:
...{previous_tail}

Theme: {cluster}. These sources COMPLICATE the thesis by setting scope conditions, contingencies, or measurement limits.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Do not list qualifications. Pick the single scope condition that most narrows the supporting claim, and use one extended example to show what the thesis can and cannot explain under that condition. If the CLUSTER SYNTHESIS above names a scope condition shared across multiple sources, cite them together in one parenthetical when introducing it. Treat the qualification as reshaping the question, not as a rejection. Do not restate the thesis. 220-260 words. End mid-thought.

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    # v8 (R4/R6): closing must synthesise without restating prior section claims;
    # do not paraphrase the same two mechanisms that opened.
    return f"""Close this literature review on: {topic}

Previous ending:
...{previous_tail}

ALLOWED CITATIONS:
{allowed_list}

Section claims already made (do NOT restate the specific mechanisms below; synthesise across them):
{evidence}

{_PROSE_DIRECTIVE}

Without restating the specific mechanisms or evidence already developed, name the underlying point of agreement, the precise remaining disagreement, and what would have to be measured to resolve it. 150-200 words. Do not write "In conclusion" or "To summarize".

Continue:"""


def _dump_writer_prompt(stage: str, system: str, user: str) -> None:
    """v10: when RRR_DEBUG_WRITER_PROMPTS=1 (or an explicit dir), write the
    exact system+user pair sent to the model into runs/prompts/. One file per
    stage call; later calls of the same stage suffix with a counter.
    """
    target = os.environ.get("RRR_WRITER_PROMPT_DUMP_DIR")
    if not target and os.environ.get("RRR_DEBUG_WRITER_PROMPTS", "0") == "1":
        target = str(runs_path("prompts"))
    if not target:
        return
    try:
        os.makedirs(target, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", stage)
        # disambiguate repeated stages (e.g. writer_complicates fires N times)
        idx = 0
        while True:
            fn = os.path.join(target, f"{slug}_{idx:02d}.txt")
            if not os.path.exists(fn):
                break
            idx += 1
        payload = (
            "===== SYSTEM =====\n" + (system or "") + "\n\n"
            "===== USER =====\n" + (user or "") + "\n"
        )
        with open(fn, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        # debug dump must never break a writer call
        pass


def _ollama_chat(prompt: str, metrics=None, stage="writer"):
    import ollama
    import time
    _dump_writer_prompt(stage, _SYSTEM_CITATION_INSTRUCTION, prompt)
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


def _apply_cross_section_stitch(chunks, topic: str, metrics=None):
    """v11-B: rewrite the first sentence of each interior section so the
    review flows from section to section instead of each restating the topic
    framing. One batched LLM call. Falls back to the original chunks on any
    parse / verification failure. Toggle via RRR_WRITER_STITCH (default ON).

    Verification: the rewritten opener must contain the SAME set of canonical
    citations (Doc_Year: p.N) as the original; any addition or removal causes
    that opener to be discarded. Only the opening sentence is replaced; the
    rest of each section is untouched.
    """
    import time
    if os.environ.get("RRR_WRITER_STITCH", "1") != "1":
        return chunks
    # Need at least opening + 2 interior sections + closing to make stitching
    # meaningful. (Opening alone doesn't need stitching; closing already
    # synthesises across sections.)
    if not chunks or len(chunks) < 4:
        return chunks

    interior_indices = list(range(1, len(chunks) - 1))
    extracts = []
    for i in interior_indices:
        sents = _split_sentences_for_cleanup(chunks[i])
        if not sents:
            continue
        opener = sents[0].strip()
        prev_sents = _split_sentences_for_cleanup(chunks[i - 1])
        prev_tail = prev_sents[-1].strip() if prev_sents else ""
        if not opener:
            continue
        extracts.append({"index": i, "opener": opener, "prev_tail": prev_tail})

    if len(extracts) < 2:
        return chunks

    parts = [
        f"Topic of the literature review: {topic}",
        "",
        "Below are the OPENING sentence of several adjacent sections in the "
        "same review. Each currently restates the topic framing on its own, "
        "which reads as several mini-essays stitched together. Rewrite each "
        "OPENING so it flows from the previous section's last sentence — pick "
        "up a thread, contrast a finding, follow a scope condition — rather "
        "than re-introducing the topic.",
        "",
        "STRICT RULES:",
        "- Keep every (Doc_Year: p.N) citation token UNCHANGED.",
        "- Do NOT add or remove any citation.",
        "- Do NOT exceed the original sentence length by more than ~20%.",
        "- Return ONLY a single JSON object: "
        '{"openings": [{"index": <int>, "rewritten": "<sentence>"}, ...]}',
        "",
        "Sections to stitch:",
    ]
    for e in extracts:
        parts.append(f"\n--- section index={e['index']} ---")
        if e["prev_tail"]:
            parts.append(f"Previous section ended: {e['prev_tail']}")
        parts.append(f"Current opening (rewrite this): {e['opener']}")
    prompt = "\n".join(parts)

    try:
        import ollama
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": float(os.environ.get("RRR_WRITER_STITCH_T", "0.3")),
                "num_ctx": int(os.environ.get("RRR_WRITER_STITCH_CTX", "8192")),
                "num_predict": int(os.environ.get("RRR_WRITER_STITCH_PRED", "1500")),
            },
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        if metrics:
            metrics.record_llm("cross_section_stitch", _MODEL,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), response_chars=len(raw))
    except Exception as e:
        if metrics:
            metrics.record_llm("cross_section_stitch", _MODEL,
                               success=False, error=e)
        return chunks

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx < 0 or end_idx <= start_idx:
            return chunks
        obj = json.loads(raw[start_idx:end_idx + 1])
    except Exception:
        return chunks
    rewrites = obj.get("openings") if isinstance(obj, dict) else None
    if not isinstance(rewrites, list):
        return chunks

    new_chunks = list(chunks)
    applied = 0
    skipped_citation = 0
    skipped_empty = 0
    interior_set = set(interior_indices)
    cite_token_re = re.compile(r"\([A-Za-z0-9_&.\-]+:\s*p\.\d+\)")
    for r in rewrites:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("index"))
        except (TypeError, ValueError):
            continue
        rewritten = str(r.get("rewritten", "") or "").strip()
        if not rewritten or idx not in interior_set:
            skipped_empty += 1
            continue
        sents = _split_sentences_for_cleanup(new_chunks[idx])
        if not sents:
            continue
        orig_opener = sents[0]
        orig_cites = sorted(cite_token_re.findall(orig_opener))
        new_cites = sorted(cite_token_re.findall(rewritten))
        if orig_cites != new_cites:
            skipped_citation += 1
            continue
        sents[0] = rewritten
        new_chunks[idx] = " ".join(sents)
        applied += 1

    if metrics:
        metrics.set("writer_stitch_applied", applied)
        metrics.set("writer_stitch_skipped_citation_change", skipped_citation)
        metrics.set("writer_stitch_skipped_empty", skipped_empty)
    if applied:
        print(f"[Writer] Cross-section stitch: rewrote {applied} section "
              f"opener(s) (skipped {skipped_citation} for citation drift)")
    return new_chunks


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
    """Collect cited doc_ids from all surfaces the writer may have emitted.

    v10: the user-facing surface is 'Author (Year, p.N)'. We still scan the
    legacy canonical and bare doc_id forms in case any slipped through.
    """
    cited_docs = set()

    # Canonical: (Doc_Year: p.N)
    for m in re.finditer(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)

    # Bare canonical: (Doc_Year)
    for m in re.finditer(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)

    # v10 display surfaces: both 'Author (Year, p.N)' and '(Author Year, p.N)'
    display_lookup = _build_display_lookup(allowed_docs)
    for m in DISPLAY_CITE_RE.finditer(text):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)
    for m in DISPLAY_PAREN_CITE_RE.finditer(text):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)

    # Legacy author-year academic forms: (Author, Year) and Author (Year)
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

    # v11-C: read cluster syntheses from the ledger and build the doc -> [evidence_ids]
    # reverse lookup we need to render multi-citation parentheticals. Each
    # synthesis is keyed as "stance::cluster_label" in the ledger; unpack into a
    # tuple-keyed dict for lookup during chunk_plan construction.
    cluster_syntheses_raw = data.get("cluster_syntheses", {}) or {}
    cluster_syntheses = {}
    for key, synth in cluster_syntheses_raw.items():
        if "::" not in key or not isinstance(synth, dict):
            continue
        stance_part, _, cluster_part = key.partition("::")
        cluster_syntheses[(stance_part, cluster_part)] = synth
    doc_to_evidence_ids = defaultdict(list)
    for eid, ev in evidence_id_map.items():
        did = (ev.get("doc_id") or "").strip()
        if did:
            doc_to_evidence_ids[did].append(eid)
    # Sort each doc's evidence IDs so the first one is deterministic (E0001 < E0002).
    for did in list(doc_to_evidence_ids.keys()):
        doc_to_evidence_ids[did] = sorted(doc_to_evidence_ids[did])
    if cluster_syntheses:
        print(f"[Writer] cluster syntheses loaded: {len(cluster_syntheses)}")
        if metrics:
            metrics.set("writer_cluster_syntheses_loaded", len(cluster_syntheses))

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
    
    def _synth_block(stance, cluster_label):
        synth = cluster_syntheses.get((stance, cluster_label))
        return _format_cluster_synthesis_block(synth, doc_to_evidence_ids)

    for cluster, cluster_docs in rank_clusters("supports"):
        chunk_plan.append(("supports", cluster, cluster_docs,
                           _build_supports_prompt, _synth_block("supports", cluster)))

    for cluster, cluster_docs in rank_clusters("critiques"):
        chunk_plan.append(("critiques", cluster, cluster_docs,
                           _build_critiques_prompt, _synth_block("critiques", cluster)))

    for cluster, cluster_docs in rank_clusters("complicates"):
        chunk_plan.append(("complicates", cluster, cluster_docs,
                           _build_complicates_prompt, _synth_block("complicates", cluster)))
    
    if not chunk_plan:
        raise SystemExit("No documents to write about.")

    supports_clusters = sum(1 for s, *_ in chunk_plan if s == "supports")
    critiques_clusters = sum(1 for s, *_ in chunk_plan if s == "critiques")
    complicates_clusters = sum(1 for s, *_ in chunk_plan if s == "complicates")
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
    # v8 (R5): observability for new postproc surfaces
    total_double_paren_collapsed = 0
    total_author_led_openings = 0
    section_coverage = []
    
    # v10: display->canonical lookup built once per run. The writer is asked
    # for display surface 'Author (Year, p.N)' but the existing per-section
    # validators (citation removal, coverage audit, redundancy detection) all
    # speak canonical '(Doc_Year: p.N)'. We rewrite display->canonical at the
    # START of postprocess_chunk so the canonical machinery keeps working
    # unchanged; the FINAL assembly converts everything back to display.
    chunk_display_lookup = _build_display_lookup(allowed_docs)

    def _chunk_display_to_canonical(text):
        def repl(m):
            label = m.group(1).strip().lower()
            year = m.group(2)
            page = int(m.group(3))
            did = chunk_display_lookup.get((label, year))
            if did:
                return render_citation_canonical(did, page)
            return m.group(0)
        # v10.2: rewrite BOTH 'Author (Year, p.N)' AND '(Author Year, p.N)' to
        # the canonical surface. The paren-shape was the silent failure mode in
        # the v10.1 smoke (model mixed the two; we caught only one).
        text = DISPLAY_CITE_RE.sub(repl, text)
        text = DISPLAY_PAREN_CITE_RE.sub(repl, text)
        return text

    def postprocess_chunk(chunk, chunk_docs):
        nonlocal total_repairs, total_placeholders_stripped, all_dump_citations, total_ajr_fixes
        nonlocal total_removed_citations, total_style_removed, total_evidence_id_renders
        nonlocal total_double_paren_collapsed, total_author_led_openings

        chunk = _strip_wrapping(chunk)
        chunk, evidence_renders = _render_evidence_id_citations(chunk, evidence_id_map)
        total_evidence_id_renders += evidence_renders

        # v10: rewrite Author (Year, p.N) -> (Doc_Year: p.N) so every existing
        # validator below sees the canonical surface it understands.
        chunk = _chunk_display_to_canonical(chunk)

        # v8 (R5): collapse ((Doc: p.N)) before any other citation pass — those
        # forms otherwise survive every downstream regex because CITE_RE matches
        # single-paren only.
        chunk, double_collapsed = _collapse_double_parens(chunk)
        total_double_paren_collapsed += double_collapsed

        # v8 (R5): observe author-led-opening violations and count them.
        total_author_led_openings += _count_author_led_openings(chunk)

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

        # v8 (R11): try the deterministic fallback FIRST — it's free, uses the
        # same allowed-pairs provenance as the LLM retry would, and resolves
        # the small-shortfall case (which dominates) without a ~2s LLM call.
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
        if audit["ok"]:
            return chunk, repairs, placeholders, ajr, style_removed, audit

        # Deterministic fallback couldn't reach the required doc count. Escalate
        # to LLM coverage retry as the last resort.
        if metrics:
            metrics.inc("writer_section_coverage_retries")
        retry_prompt = _coverage_retry_prompt(prompt, chunk, audit["required_cited_docs"])
        raw = _ollama_chat(retry_prompt, metrics=metrics, stage=f"{stage}_coverage_retry")
        chunk, repairs2, placeholders2, ajr2, style_removed2 = postprocess_chunk(raw, chunk_docs)
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        if not audit["ok"]:
            # Final deterministic top-up in case the LLM retry made partial progress.
            chunk, fallback_count2 = _append_coverage_fallback(
                chunk,
                chunk_docs,
                audit["required_cited_docs"],
                allowed_pairs=chunk_allowed_pairs,
            )
            if fallback_count2:
                total_coverage_fallbacks += fallback_count2
                if metrics:
                    metrics.inc("writer_section_coverage_fallbacks", fallback_count2)
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
    for i, (stance, cluster, cluster_docs, prompt_builder, synthesis_block) in enumerate(chunk_plan):
        cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, parallel_tail,
                                synthesis_block)
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
        for i, (stance, cluster, cluster_docs, prompt_builder, synthesis_block) in enumerate(chunk_plan):
            cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
            allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
            evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
            previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
            prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail,
                                    synthesis_block)
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

    # v11-B: one batched LLM call that rewrites the opening sentence of each
    # interior section so the review flows section-to-section instead of each
    # restating the topic framing. Operates on the chunks list (each chunk is
    # one section). Verifies citation tokens are preserved per opener; any
    # opener whose citation set drifts is silently reverted to the original.
    if metrics:
        with metrics.stage("writer_stitch"):
            chunks = _apply_cross_section_stitch(chunks, topic, metrics=metrics)
    else:
        chunks = _apply_cross_section_stitch(chunks, topic, metrics=None)

    # Final assembly
    full_text = "\n\n".join(chunks)
    full_text, final_evidence_renders = _render_evidence_id_citations(full_text, evidence_id_map)
    total_evidence_id_renders += final_evidence_renders

    # v10: bring everything to the CANONICAL surface for validation. The model
    # is asked for display form 'Author (Year, p.N)'; we map it back to the
    # canonical '(Doc_Year: p.N)' so the existing validator (which knows about
    # allowed_pairs in canonical form) can do its work unchanged.
    display_lookup = _build_display_lookup(allowed_docs)
    display_to_canonical_count = 0

    def _display_to_canonical(text):
        nonlocal display_to_canonical_count

        def repl(m):
            nonlocal display_to_canonical_count
            label = m.group(1).strip().lower()
            year = m.group(2)
            page = int(m.group(3))
            did = display_lookup.get((label, year))
            if did:
                display_to_canonical_count += 1
                return render_citation_canonical(did, page)
            return m.group(0)

        # v10.2: handle both display shapes
        text = DISPLAY_CITE_RE.sub(repl, text)
        text = DISPLAY_PAREN_CITE_RE.sub(repl, text)
        return text

    full_text = _display_to_canonical(full_text)
    if display_to_canonical_count and metrics:
        metrics.inc("writer_display_to_canonical", display_to_canonical_count)

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

    # v9 (R6): final-assembly cross-section redundancy drop. Conservative —
    # only fires when a sentence's citations are a strict subset of those
    # already cited AND it has substantial content overlap with an earlier
    # sentence. Configurable via env (set 0 to disable, or raise the overlap
    # threshold to be even more conservative).
    if os.environ.get("RRR_WRITER_DROP_REDUNDANCY", "1") != "0":
        overlap_threshold = int(os.environ.get("RRR_WRITER_REDUNDANCY_OVERLAP", "4"))
        full_text, redundancy_drops = _drop_cross_section_redundancy(
            full_text, min_token_overlap=overlap_threshold,
        )
        if redundancy_drops:
            print(f"[Writer] Dropped {len(redundancy_drops)} redundant sentence(s) at final assembly:")
            for r in redundancy_drops[:5]:
                print(f"         - overlap={r['overlap_tokens']} cited={r['cited_pairs']}: {r['snippet']}")
            if metrics:
                metrics.inc("writer_redundancy_drops", len(redundancy_drops))
                metrics.set("writer_redundancy_examples", redundancy_drops[:10])
    else:
        redundancy_drops = []

    full_text = _strip_orphaned_citations(full_text)
    full_text = _strip_references_section(full_text)
    full_text = _strip_continuation_markers(full_text)

    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # v10: every surviving canonical-form citation '(Doc_Year: p.N)' is
    # rewritten to the user-facing display form 'Author (Year, p.N)'.
    # Validation is already complete at this point, so the rewrite is purely
    # cosmetic; the corpus boundary check ran against the canonical pair.
    full_text, canonical_to_display_count = rewrite_canonical_to_display(full_text)
    if canonical_to_display_count and metrics:
        metrics.inc("writer_canonical_to_display", canonical_to_display_count)

    # v11-A: rewrite narrative citations 'Author (Year, p.N)' sitting at the
    # end of a sentence whose subject is not the author to the parenthetical
    # form '(Author Year, p.N)'. Closes the v10 known edge case where the
    # writer produced 'X conducive to growth North and Weingast (1989, p.4).'
    full_text, narrative_rewrites = rewrite_misplaced_narrative_citations(full_text)
    if narrative_rewrites and metrics:
        metrics.inc("writer_narrative_to_paren", narrative_rewrites)
    if narrative_rewrites:
        print(f"[Writer] Narrative->paren rewrites: {narrative_rewrites}")

    # v10: detect-then-LLM-rewrite style enforcement (one batched call). Runs
    # AFTER validation so the rewriter sees the final citation surface and is
    # forbidden from changing any (Year, p.N) tokens. Refuses to apply rewrites
    # that would reintroduce a violation or alter a citation.
    full_text, style_stats = _apply_style_enforcement(full_text, metrics=metrics)
    if metrics:
        metrics.set("writer_style_enforcement", style_stats)
    if style_stats.get("violations"):
        print(f"[Writer] Style: {style_stats['violations']} flagged, "
              f"{style_stats['rewrites_applied']} rewritten "
              f"(trailing_significance stripped: {style_stats['trailing_stripped']}, "
              f"fallback_reason: {style_stats['fallback_reason']})")
    elif style_stats.get("trailing_stripped"):
        print(f"[Writer] Style: stripped {style_stats['trailing_stripped']} "
              "trailing-significance phrase(s); no other violations.")

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
        f"coverage_fallbacks={total_coverage_fallbacks} evidence_id_renders={total_evidence_id_renders} "
        f"double_paren_collapsed={total_double_paren_collapsed} author_led_openings={total_author_led_openings} "
        f"redundancy_drops={len(redundancy_drops)}"
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
            "double_paren_collapsed": total_double_paren_collapsed,
            "author_led_openings": total_author_led_openings,
            "redundancy_drops": len(redundancy_drops),
            "citation_dump_docs": len(all_dump_citations),
            "section_claims": section_claims,
            "section_coverage": section_coverage,
        })
        metrics.inc("writer_citation_repairs", total_repairs)
        metrics.inc("writer_removed_citations", total_removed_citations)
        metrics.inc("writer_style_sentences_removed", total_style_removed)
        metrics.inc("writer_evidence_id_renders", total_evidence_id_renders)
        metrics.inc("writer_double_paren_collapsed", total_double_paren_collapsed)
        metrics.inc("writer_author_led_openings", total_author_led_openings)
    
    return str(out_path)


if __name__ == "__main__":
    compose_from_ledger()
