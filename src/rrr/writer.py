import os
import json
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from rrr.utils import ensure_dir, env_int
from rrr.paths import runs_path
from rrr.render import (
    CITE_RE,
    DISPLAY_CITE_RE,
    DISPLAY_PAREN_CITE_RE,
    parse_citations,
    render_citation,
    citation_inside,
    note_rendered_citation,
    render_citation_canonical,
    _build_author_year_lookup,
    _build_display_lookup,
    _collect_cited_docs as _shared_collect_cited_docs,
    _doc_id_to_author_label,
)
from rrr.llm import install as _install_llm_shim
_install_llm_shim()
_MODEL = os.environ.get("RRR_WRITER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))
_KEEP_ALIVE = "30m"

_DEFAULT_CHAT_OPTIONS = {
    "temperature": 0.50,
    "num_ctx": 12288,
    "num_predict": 1500,
    "top_p": 0.9,
}

_TAIL_CHARS = 600

_GENERIC_STYLE_RE = re.compile(
    r"\b("
    r"complex interplay|valuable insights|policy-making|future research|further research|"
    r"further investigation|nuanced perspective|the stakes are high|this analysis will|"
    r"delving deeper|underscores? the need|further exploration|ongoing research|"
    r"shed light|complex and influenced by various factors|"
    r"holds significance|potential implications|strategies aimed at|"
    r"sustainable development|alleviating poverty|this investigation|"
    r"this analysis|this study|this examination|plays a pivotal role|"
    r"contributing significantly to|fundamental factor|crucial factor|"
    r"essential factor|interplay between|in this regard|in this context|"
    r"by acknowledging these variables|presents a complex picture|"
    r"thereby contributing|"
    r"complex interactions between|"
    r"play(?:s|ed)?\s+a\s+significant\s+role|"
    r"significantly\s+influenc(?:e|ed|es)|"
    r"a\s+more\s+nuanced\s+understanding|"
    r"it\s+is\s+crucial\s+to\s+examine|"
    r"are\s+pivotal\s+in|"
    r"pivotal\s+(?:question|role|issue|factor|moment|area|importance|insight)|"
    r"(?:is|are|stands\s+as|remains|becomes)\s+(?:a\s+|the\s+)?pivotal"
    r")\b",
    re.IGNORECASE,
)

_DOUBLE_PAREN_CITE_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\)\)"
)
_NESTED_PAGE_SUFFIX_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+):\s*p\.\d+\):\s*p\.\d+\)"
)
_DOUBLE_PAREN_DISPLAY_RE = re.compile(
    r"\(\(((?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"(?:\s+(?:and\s+(?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"|et\s+al\.))?"
    r"\s+\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+)\)\)"
)
_DOUBLE_PAREN_DISPLAY_GROUP_RE = re.compile(
    r"\(((?:\([^()]+?\)[\s;,]*){2,})\)"
)
_ASYM_OUTER_PAREN_RE = re.compile(
    r"\(\(([^()]+\))(?!\))"
)

_FORBIDDEN_LEXEME_RE = re.compile(r"\b(?:legible|sharp(?:ly)?)\b", re.IGNORECASE)
_ADVERSATIVE_RE = re.compile(
    r"(?:\b("
    r"not\s+(?:\S+\s+){0,15}but(?:\s+(?:rather|also))?\s+"
    r"|rather than\b"
    r"|in (?:sharp\s+)?contrast(?:\s+to)?\b"
    r"|instead of\b"
    r"|unlike\b"
    r")\b)|(?:(?<=[.!?]\s)(?:Conversely|Yet|However),)|(?:^(?:Conversely|Yet|However),)",
    re.IGNORECASE | re.MULTILINE,
)
_TRAILING_SIGNIFICANCE_RE = re.compile(
    r"[,.;]?\s+"
    r"(?:[Aa]nd that matters"
    r"|[Ww]hich matters because[^.!?]*"
    r"|[Tt]his is important because[^.!?]*"
    r"|[Tt]his matters because[^.!?]*)"
    r"[.!?]?",
)
_COLON_SETUP_RE = re.compile(r"[A-Z][^.!?:\n]{20,}:\s+[A-Z]")
_TYPOGRAPHIC_DASH_RE = re.compile("[–—]")
_OPTIONS_STYLE_REWRITE = {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1500}

_AUTHOR_VERB_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)\s*[A-Z][A-Za-z&.\-]+)(?:\s+et\s+al\.?)?(?:\s*\(?[A-Za-z_]*\d{4}[A-Za-z_]*\)?)?)\s+"
    r"(argues|emphasizes|emphasises|demonstrates|highlights|posits|suggests|"
    r"claims|notes|observes|maintains|asserts|contends|shows|writes|states|"
    r"finds|points out|underscores|argues that|notes that)\b",
    re.IGNORECASE,
)
_MIN_SECTION_CITED_DOCS = env_int("RRR_WRITER_MIN_SECTION_CITED_DOCS", 2)
_MIN_CLOSING_WORDS = env_int("RRR_WRITER_MIN_CLOSING_WORDS", 120)
_CLOSING_START_MARKER = "[[RRR-CLOSING-START]]."
_ENFORCE_COVERAGE = True


def _writer_system_prompt() -> str:
    from rrr.language import language_name
    topic_lang = os.environ.get("RRR_TOPIC_LANG", "en")
    corpus_lang = os.environ.get("RRR_CORPUS_LANG", "en")
    if topic_lang == corpus_lang:
        return _SYSTEM_CITATION_INSTRUCTION
    tl = language_name(topic_lang)
    cl = language_name(corpus_lang)
    contract = (
        f"OUTPUT LANGUAGE — READ FIRST. The source evidence, cluster "
        f"syntheses, and section notes below are in {cl}. Write the review "
        f"in {tl}, with this rule:\n"
        f"  - YOUR OWN WORDS — all narration, analysis, synthesis, topic "
        f"sentences, transitions, everything that is not a quotation — MUST "
        f"be in {tl}. Do NOT copy {cl} sentences from the evidence into your "
        f"prose as if they were your narration; read the {cl} evidence, "
        f"understand it, and express the point in {tl}.\n"
        f"  - VERBATIM QUOTATIONS may stay in {cl}. A verbatim quote is "
        f"scholarly evidence and keeps its ORIGINAL language — do NOT "
        f"translate a quote. But it MUST be a genuine short excerpt "
        f"(<=12 words) inside quotation marks (\" \") followed by its "
        f"[E####] citation. Anything in {cl} that is NOT inside quotation "
        f"marks is a mistake — rewrite it in {tl}.\n"
        f"  - Author surnames and years inside citation markers stay as-is.\n"
        f"Net: a reader sees {tl} prose throughout, with occasional clearly-"
        f"marked {cl} quotations. No unquoted {cl} sentences anywhere."
    )
    return f"{contract}\n\n{_SYSTEM_CITATION_INSTRUCTION}"


_SYSTEM_CITATION_INSTRUCTION = (
    "CITE only via evidence IDs: every citation is one [E####] marker. "
    "Examples: 'North and Weingast [E0001] argue that...' OR 'credible "
    "commitment underpins growth [E0001].' Multi-source: 'shared finding "
    "[E0001] [E0007] [E0014].' DO NOT write '(Author Year, p.N)', "
    "'Author (Year, p.N)', '(p.5)' or any other citation surface — the "
    "renderer converts every [E####] to the right surface for you.\n\n"
    "BOUNDARY RULES:\n"
    "1. Use only [E####] IDs from the ALLOWED CITATIONS list.\n"
    "2. If a claim is not supported by allowed evidence, state it WITHOUT "
    "a citation. Do not invent evidence IDs.\n"
    "3. QUOTED TEXT: paraphrase by default. Reserve quotation marks (\" \") "
    "for SHORT phrases (≤12 words) you copy-paste verbatim from the "
    "Evidence section. Multi-sentence quotes, ellipsis-truncated quotes, "
    "and paraphrased-but-quoted material are fabrications and get "
    "stripped.\n"
    "4. Finish every paragraph and every section with a complete, "
    "standalone sentence. Section transitions are thematic. Never leave a "
    "clause or sentence for the next section to complete.\n"
    "5. Return only the requested review prose. Begin with its first "
    "sentence. Do not include planning, analysis, a checklist, headings, "
    "notes, a word count, draft labels, or commentary.\n"
)

_OUTPUT_ONLY_DIRECTIVE = (
    "Return only the requested review prose. Begin immediately with its "
    "first sentence. Do not include planning, analysis, a checklist, "
    "headings, notes, a word count, draft labels, or commentary."
)

_QWEN_PROSE_SCHEMA = {
    "type": "object",
    "properties": {"prose": {"type": "string"}},
    "required": ["prose"],
    "additionalProperties": False,
}
_QWEN_PROSE_TRANSPORT_INSTRUCTION = (
    "For transport, return exactly one JSON object with one key named "
    "prose. Put only the requested review prose in that string."
)


def _uses_qwen_prose_transport(model: str) -> bool:
    runtime = os.environ.get("RRR_RUNTIME", "").strip().lower()
    return "qwen3" in str(model or "").lower() and runtime in {
        "", "local", "ollama",
    }


def _decode_qwen_prose_transport(raw: str) -> str:
    try:
        payload = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Qwen prose transport returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"prose"}:
        raise ValueError("Qwen prose transport must contain only 'prose'")
    prose = payload.get("prose")
    if not isinstance(prose, str) or not prose.strip():
        raise ValueError("Qwen prose transport returned empty prose")
    return prose.strip()


def _score_doc(d) -> float:
    return d.get("avg_score", 0)


def _clip(s: str, n=260) -> str:
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "...") if len(s) > n else s


def _writer_passage_text(q) -> str:
    return re.sub(r"\s+", " ", str(q.get("text", "") or "")).strip()


def _format_quote(q) -> str:
    did = str(q.get("doc_id", "")).strip()
    pg = int(q.get("page", 0) or 0)
    tx = _writer_passage_text(q)
    eid = str(q.get("evidence_id", "")).strip()
    prefix = f"[{eid}] " if eid else ""
    suffix = f" (from {did}, p.{pg})" if did and pg else ""
    return f'{prefix}"{tx}"{suffix}'


def _writer_quotes_per_doc() -> int:
    return max(1, env_int("RRR_WRITER_QUOTES_PER_DOC", 2))


def _select_call_evidence(docs, quotes_per_doc=None):
    limit = max(1, int(quotes_per_doc or _writer_quotes_per_doc()))
    packet = []
    for source_doc in docs or []:
        d = dict(source_doc)
        selected = list(source_doc.get("quotes") or [])[:limit]
        if not selected:
            raise ValueError(
                f"writer call document {source_doc.get('doc_id')!r} has no evidence passages"
            )
        for q in selected:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", source_doc.get("doc_id", ""))).strip()
            page = int(q.get("page", 0) or 0)
            if not eid or not did or page <= 0:
                raise ValueError(
                    f"writer call document {source_doc.get('doc_id')!r} contains "
                    "a selected passage without evidence_id, doc_id, or page"
                )
        d["quotes"] = selected
        packet.append(d)
    return packet


def _format_doc_entry(d) -> str:
    did = str(d.get("doc_id", "")).strip()
    author_label = _doc_id_to_author_label(did)
    lines = [f"[{did}] cite as {author_label!s}"]
    claim = str(d.get("claim", "") or "").strip()
    if claim:
        lines.append(f"  Paper's central claim: {_clip(claim, n=240)}")
    qs = d.get("quotes") or []
    for q in qs:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    lines = []
    evidence_lines = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            page = int(q.get("page", 0) or 0)
            if eid and did and page:
                surnames = _author_surnames_only(_doc_id_to_author_label(did))
                evidence_lines.append(
                    f"  - [{eid}] (paper {did}, page {page}; author surname: {surnames})"
                )
        if evidence_lines:
            continue
        pages = sorted(list(allowed_pages_by_doc.get(did, set())))
        if pages:
            surnames = _author_surnames_only(_doc_id_to_author_label(did))
            page_str = ", ".join(f"p.{p}" for p in pages[:6])
            lines.append(f"  - (paper {did}, pages {page_str}; author surname: {surnames})")
    if evidence_lines:
        return "\n".join(evidence_lines)
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


def _author_label_pattern(author: str) -> str:
    parts = re.split(r"\s+(?:and|&)\s+", author or "", flags=re.IGNORECASE)
    escaped = [
        r"\s+".join(re.escape(word) for word in part.split())
        for part in parts
        if part.strip()
    ]
    return r"\s+(?:and|&)\s+".join(escaped)


_CONJOINED_NAME_BEFORE_LABEL_RE = re.compile(
    r"(?<![\w.])"
    r"[A-Z][\w'\u2019\-]*(?:\s+(?:et\s+al\.?|[A-Z][\w'\u2019\-]*))*"
    r"\s*,?\s*(?:and|&)\s*$"
)


def _label_preceded_by_conjoined_name(prefix: str) -> bool:
    return bool(_CONJOINED_NAME_BEFORE_LABEL_RE.search((prefix or "").rstrip()))


def _repair_unresolved_display_citations(text: str, display_lookup: dict) -> tuple:
    if not display_lookup:
        return text or "", 0
    replacements = []
    for citation in parse_citations(text or "", display_lookup=display_lookup):
        if citation.get("doc_id") is not None:
            continue
        label = str(citation.get("label") or "").strip()
        year = str(citation.get("year") or "")
        if not label or not year or citation.get("page") is None:
            continue
        units = [u.strip() for u in re.split(r"\s+(?:and|&)\s+", label) if u.strip()]
        if len(units) < 2:
            continue
        tail = units[-1]
        if display_lookup.get((tail.lower(), year)) is None:
            continue
        raw = citation.get("raw") or ""
        tail_match = re.search(
            r"\(\s*" + re.escape(year) + r"[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)\s*$", raw,
        )
        if not tail_match:
            continue
        rebound = raw[:tail_match.start()] + f"({tail} {year}, p.{tail_match.group(1)})"
        replacements.append((citation["start"], citation["end"], rebound))
    out = text or ""
    for start, end, rebound in sorted(replacements, reverse=True):
        out = out[:start] + rebound + out[end:]
    return out, len(replacements)


def _apply_observed_capitalisation(canonical: str, observed: str) -> str:
    if observed[:1].isupper() and canonical[:1].islower():
        return canonical[:1].upper() + canonical[1:]
    return canonical


def _repair_allowed_author_year_mentions(
    text: str,
    docs,
    *,
    stage: str = "",
) -> tuple:
    source = text or ""
    evidence_map = _build_evidence_id_map(docs)
    surface_entries = defaultdict(list)

    for doc in docs or []:
        doc_id = str(doc.get("doc_id", "") or "").strip()
        label = _doc_id_to_author_label(doc_id)
        label_match = re.fullmatch(r"(.+?)\s*\((\d{4}[a-z]?)\)", label or "")
        if not doc_id or not label_match:
            continue
        selected = None
        for quote in doc.get("quotes", []) or []:
            eid = str(quote.get("evidence_id", "") or "").strip()
            quote_doc = str(quote.get("doc_id", "") or "").strip()
            page = int(quote.get("page", 0) or 0)
            mapped = evidence_map.get(eid)
            if (
                eid and quote_doc == doc_id and page > 0 and mapped
                and mapped.get("doc_id") == doc_id
                and int(mapped.get("page", 0) or 0) == page
            ):
                selected = {"evidence_id": eid, "page": page}
                break
        if not selected:
            continue
        author = label_match.group(1).strip()
        year = label_match.group(2)
        common = {
            "doc_id": doc_id,
            "evidence_id": selected["evidence_id"],
            "page": selected["page"],
        }
        surface_entries[("narrative", author, year)].append(common)
        surface_entries[("parenthetical", author, year)].append(common)

    unique_surfaces = {
        key: entries[0]
        for key, entries in surface_entries.items()
        if len({entry["doc_id"] for entry in entries}) == 1
    }
    collision_skips = sum(
        1
        for entries in surface_entries.values()
        if len({entry["doc_id"] for entry in entries}) > 1
    )

    quoted_spans = [
        match.span()
        for pattern in (
            r'"[^"\n]*"',
            r"\u201c[^\u201d\n]*\u201d",
            r"'[^'\n]*'",
            r"\u2018[^\u2019\n]*\u2019",
            r"(?m)^\s*>[^\n]*$",
        )
        for match in re.finditer(pattern, source)
    ]

    def is_quoted(start: int, end: int) -> bool:
        return any(start < quote_end and quote_start < end
                   for quote_start, quote_end in quoted_spans)

    proposed = []
    occupied = []
    for (surface_kind, author, year), entry in sorted(
        unique_surfaces.items(), key=lambda item: len(item[0][1]), reverse=True,
    ):
        label_pattern = _author_label_pattern(author)
        if not label_pattern:
            continue
        escaped_year = re.escape(year)
        if surface_kind == "narrative":
            pattern = re.compile(
                rf"(?<!\w)(?P<author>{label_pattern})"
                r"(?P<possessive>['’]s)?\s*"
                rf"\({escaped_year}\)(?!\w)"
                r"(?![\s,;:]*\[[Ee]\d{1,5}\])",
                flags=re.IGNORECASE,
            )
        else:
            pattern = re.compile(
                rf"\(\s*(?P<author>{label_pattern}),?\s+{escaped_year}\s*\)"
                r"(?![\s,;:]*\[[Ee]\d{1,5}\])",
                flags=re.IGNORECASE,
            )
        for match in pattern.finditer(source):
            if is_quoted(match.start(), match.end()):
                continue
            if any(match.start() < end and start < match.end()
                   for start, end in occupied):
                continue
            if surface_kind == "narrative":
                display_author = _apply_observed_capitalisation(
                    author, match.group("author"),
                )
                possessive = match.group("possessive") or ""
                replacement = (
                    f"{display_author}{possessive} [{entry['evidence_id']}]"
                )
            else:
                replacement = f"[{entry['evidence_id']}]"
            proposed.append({
                "start": match.start(),
                "end": match.end(),
                "replacement": replacement,
                "stage": stage,
                "phase": "after_retry",
                "surface_kind": surface_kind,
                "matched_text": match.group(0),
                "doc_id": entry["doc_id"],
                "evidence_id": entry["evidence_id"],
                "page": entry["page"],
                "selection_policy": "first_call_packet_passage",
            })
            occupied.append(match.span())

    repaired = source
    for record in sorted(proposed, key=lambda item: item["start"], reverse=True):
        repaired = (
            repaired[:record["start"]]
            + record["replacement"]
            + repaired[record["end"]:]
        )
    public_records = [
        {key: value for key, value in record.items()
         if key not in {"start", "end", "replacement"}}
        for record in sorted(proposed, key=lambda item: item["start"])
    ]
    return repaired, public_records, collision_skips


def _group_evidence_ids_by_doc(docs):
    grouped = defaultdict(list)
    for eid, ev in _build_evidence_id_map(docs).items():
        did = str(ev.get("doc_id", "") or "").strip()
        if did:
            grouped[did].append(eid)
    return {
        did: sorted(eids)
        for did, eids in grouped.items()
    }


def _build_call_contract(stage: str, docs, allowed_list: str, prompt: str) -> dict:
    evidence_map = _build_evidence_id_map(docs)
    allowed_pairs, allowed_docs, _ = _build_allowed_citations(docs)
    displayed_ids = set(evidence_map)
    listed_ids = set(re.findall(r"\[(E\d{4})\]", allowed_list or ""))
    prompt_ids = set(re.findall(r"\[(E\d{4})\]", prompt or ""))

    if displayed_ids != listed_ids:
        raise AssertionError(
            f"{stage}: displayed evidence IDs differ from the allowed citation list"
        )
    if not prompt_ids.issubset(displayed_ids):
        extra = sorted(prompt_ids - displayed_ids)
        raise AssertionError(
            f"{stage}: prompt contains evidence IDs outside the call packet: {extra}"
        )

    passages = []
    for d in docs:
        source_did = str(d.get("doc_id", "")).strip()
        for q in d.get("quotes", []) or []:
            eid = str(q.get("evidence_id", "")).strip()
            did = str(q.get("doc_id", source_did)).strip() or source_did
            page = int(q.get("page", 0) or 0)
            source_text = str(q.get("text", "") or "")
            display_text = _writer_passage_text(q)
            passages.append({
                "evidence_id": eid,
                "doc_id": did,
                "page": page,
                "display_text": display_text,
                "display_text_sha256": hashlib.sha256(
                    display_text.encode("utf-8")
                ).hexdigest(),
                "source_text_sha256": hashlib.sha256(
                    source_text.encode("utf-8")
                ).hexdigest(),
            })

    displayed_texts_present = all(
        passage["display_text"] in (prompt or "") for passage in passages
    )
    if not displayed_texts_present:
        raise AssertionError(
            f"{stage}: a selected passage is absent or truncated in the prompt"
        )

    return {
        "stage": stage,
        "document_ids": sorted(allowed_docs),
        "evidence_ids": sorted(displayed_ids),
        "allowed_doc_page_pairs": [
            {"doc_id": did, "page": page}
            for did, page in sorted(allowed_pairs)
        ],
        "prompt_sha256": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt or ""),
        "passages": passages,
        "invariants": {
            "displayed_ids_equal_listed_ids": displayed_ids == listed_ids,
            "displayed_passage_texts_present": displayed_texts_present,
            "prompt_ids_within_call_packet": prompt_ids.issubset(displayed_ids),
            "renderer_ids_equal_displayed_ids": set(evidence_map) == displayed_ids,
            "validator_pairs_equal_displayed_pairs": set(allowed_pairs) == {
                (v["doc_id"], int(v["page"])) for v in evidence_map.values()
            },
        },
    }


def _author_surnames_only(label: str) -> str:
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


_EID_LOOKBACK_CHARS = 80


def _render_evidence_id_citations(text: str, evidence_map: dict) -> tuple:
    stats = {
        "replacements": 0,
        "unknown_eids": 0,
        "unknown_eid_snippets": [],
        "attribution_mismatches": 0,
        "attribution_mismatch_snippets": [],
        "conjoined_subject_parentheticals": 0,
    }
    if not text:
        return text or "", stats
    grouped_marker_re = re.compile(
        r"\[(?P<body>[Ee]\d{1,5}"
        r"(?:\s*[;,]\s*[Ee]\d{1,5})+)\]"
    )

    def expand_grouped_marker(match):
        marker_ids = re.findall(r"[Ee]\d{1,5}", match.group("body"))
        return "; ".join(f"[{marker_id}]" for marker_id in marker_ids)

    text, grouped_markers_expanded = grouped_marker_re.subn(
        expand_grouped_marker,
        text,
    )
    stats["grouped_markers_expanded"] = grouped_markers_expanded
    pattern = re.compile(r"\[([Ee]\d{1,5})\]")
    out_parts: list = []
    last_end = 0
    known_surnames = set()
    for _ev in evidence_map.values():
        _did = _ev.get("doc_id", "")
        _lbl = _author_surnames_only(_doc_id_to_author_label(_did))
        if _lbl:
            known_surnames.add(_lbl.lower())
    marker_matches = list(pattern.finditer(text))

    def normalized_eid(marker_match):
        raw = marker_match.group(1).upper()
        marker_digits = raw[1:].lstrip("0") or "0"
        return "E" + marker_digits.zfill(4)

    def cluster_surnames(marker_index):
        left = marker_index
        right = marker_index
        while left > 0 and re.fullmatch(
            r"[\s;,()]*",
            text[marker_matches[left - 1].end():marker_matches[left].start()],
        ):
            left -= 1
        while right + 1 < len(marker_matches) and re.fullmatch(
            r"[\s;,()]*",
            text[marker_matches[right].end():marker_matches[right + 1].start()],
        ):
            right += 1
        labels = set()
        for clustered_match in marker_matches[left:right + 1]:
            clustered = evidence_map.get(normalized_eid(clustered_match))
            if not clustered:
                continue
            clustered_label = _author_surnames_only(
                _doc_id_to_author_label(clustered.get("doc_id", ""))
            )
            if clustered_label:
                labels.add(clustered_label.lower())
        return labels

    for marker_index, match in enumerate(marker_matches):
        raw_eid = match.group(1).upper()
        digits = raw_eid[1:].lstrip("0") or "0"
        eid = "E" + digits.zfill(4)
        ev = evidence_map.get(eid)
        if not ev:
            stats["unknown_eids"] += 1
            if len(stats["unknown_eid_snippets"]) < 10:
                lo = max(0, match.start() - 60)
                hi = min(len(text), match.end() + 60)
                stats["unknown_eid_snippets"].append(text[lo:hi].replace("\n", " "))
            out_parts.append(text[last_end:match.start()])
            last_end = match.end()
            continue
        doc_id = ev["doc_id"]
        page = int(ev["page"])
        note_rendered_citation(doc_id, page)
        full_label = _doc_id_to_author_label(doc_id)
        surnames = _author_surnames_only(full_label)
        ym = re.match(r"^.*?\((\d{4}[a-z]?)\)\s*$", full_label)
        year = ym.group(1) if ym else ""

        prefix = text[max(0, match.start() - _EID_LOOKBACK_CHARS):match.start()]
        immediate = prefix.rstrip()
        immediate = re.sub(r"['’]s$", "", immediate).rstrip()
        author_immediately_before = False
        prose_surname_mismatch = None
        if surnames:
            escaped = re.escape(surnames)
            adjacency_re = re.compile(rf"(?<![A-Za-z0-9_]){escaped}$", re.IGNORECASE)
            author_immediately_before = (
                immediate.lower().endswith(surnames.lower())
                and (
                    immediate.lower() == surnames.lower()
                    or bool(adjacency_re.search(immediate))
                )
            )
            if author_immediately_before and _label_preceded_by_conjoined_name(
                immediate[:len(immediate) - len(surnames)]
            ):
                author_immediately_before = False
                stats["conjoined_subject_parentheticals"] += 1
            if not author_immediately_before:
                for other_surnames in known_surnames:
                    if other_surnames == surnames.lower():
                        continue
                    other_esc = re.escape(other_surnames)
                    other_re = re.compile(
                        rf"(?<![A-Za-z0-9_]){other_esc}$", re.IGNORECASE
                    )
                    if other_re.search(immediate):
                        prose_surname_mismatch = other_surnames
                        break
            if (
                prose_surname_mismatch
                and prose_surname_mismatch in cluster_surnames(marker_index)
            ):
                prose_surname_mismatch = None

        if prose_surname_mismatch:
            stats["attribution_mismatches"] += 1
            if len(stats["attribution_mismatch_snippets"]) < 20:
                stats["attribution_mismatch_snippets"].append({
                    "snippet": (immediate[-80:] + match.group(0)).replace("\n", " "),
                    "prose_surname": prose_surname_mismatch,
                    "expected_surnames": surnames,
                    "eid": eid,
                    "doc_id": doc_id,
                    "page": page,
                })
            if surnames and year:
                rendered = f"({surnames} {year}, p.{page})"
            else:
                rendered = render_citation(doc_id, page)
        elif author_immediately_before and year:
            rendered = f"({year}, p.{page})"
        elif surnames and year:
            rendered = f"({surnames} {year}, p.{page})"
        else:
            rendered = render_citation(doc_id, page)

        out_parts.append(text[last_end:match.start()])
        out_parts.append(rendered)
        last_end = match.end()
        stats["replacements"] += 1
    out_parts.append(text[last_end:])
    return "".join(out_parts), stats


_YEAR_ONLY_PAREN_RE = re.compile(r"\((?P<year>(?:18|19|20)\d{2}[a-z]?)\)")


def _find_unsupported_year_only_citations(text: str, chunk_docs) -> list:
    author_labels = []
    for doc in chunk_docs or []:
        doc_id = str(doc.get("doc_id", "") or "")
        label = _author_surnames_only(_doc_id_to_author_label(doc_id))
        if label:
            author_labels.append(label)

    faults = []
    for match in _YEAR_ONLY_PAREN_RE.finditer(text or ""):
        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        prefix = text[sentence_start + 1:match.start()]
        immediate = prefix.rstrip()
        followed_by_terminal = bool(
            re.match(r"\s*[.!?]", text[match.end():])
        )
        linked_author = None
        for label in author_labels:
            escaped = re.escape(label)
            if re.search(
                rf"(?<!\w){escaped}\s*$",
                immediate,
                flags=re.IGNORECASE,
            ):
                linked_author = label
                break
            if followed_by_terminal and re.search(
                rf"(?<!\w){escaped}(?!\w)",
                prefix,
                flags=re.IGNORECASE,
            ):
                linked_author = label
                break
        if not linked_author:
            continue
        lo = max(sentence_start + 1, match.start() - 120)
        hi = min(len(text), match.end() + 30)
        faults.append({
            "year": match.group("year"),
            "author": linked_author,
            "snippet": re.sub(r"\s+", " ", text[lo:hi]).strip(),
        })
    return faults


_BRACKETED_DOC_ID_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9&.\-]*_\d{4}[a-z]?)\]")


def _render_bracketed_doc_ids(text: str, allowed_docs, doc_to_eids) -> tuple:
    if not text:
        return text or "", 0
    allowed = set(allowed_docs or ())
    count = 0

    def repl(m):
        nonlocal count
        did = m.group(1)
        if did not in allowed:
            return m.group(0)
        label = _doc_id_to_author_label(did)
        if not label:
            return m.group(0)
        count += 1
        return label

    rendered = _BRACKETED_DOC_ID_RE.sub(repl, text)
    return rendered, count


_DOC_ID_TOKEN = r"[A-Za-z][A-Za-z0-9&.\-]*_\d{4}[a-z]?"


def _docid_paren_label(doc_id: str) -> str:
    label = _doc_id_to_author_label(doc_id)
    if not label:
        return doc_id
    return re.sub(r"\((\d{4}[a-z]?)\)", r"\1", label).strip()


def _normalize_leaked_doc_id_citations(text: str, allowed_docs) -> tuple:
    if not text:
        return text or "", 0
    allowed = set(allowed_docs or ())
    count = 0

    def known(token: str) -> bool:
        return token in allowed

    grounded = r"\([^()]*?\d{4}[a-z]?,\s*p\.?\s*\d+[^()]*?\)"

    def a_repl(m):
        nonlocal count
        if not known(m.group(1)):
            return m.group(0)
        count += 1
        return m.group(2)
    text = re.sub(rf"\(\s*({_DOC_ID_TOKEN})\s*({grounded})\s*\)", a_repl, text)

    junk = rf"\((?:paper\s+|from\s+)?{_DOC_ID_TOKEN}[^()]*\)"

    def b_repl_after(m):
        nonlocal count
        tok = re.search(_DOC_ID_TOKEN, m.group(0))
        if not tok or not known(tok.group(0)):
            return m.group(0)
        count += 1
        return m.group(1)
    text = re.sub(rf"({grounded})\s*{junk}", b_repl_after, text)
    text = re.sub(rf"{junk}\s*({grounded})", b_repl_after, text)

    def c_repl(m):
        nonlocal count
        inner = m.group(0)[1:-1]
        dids = re.findall(_DOC_ID_TOKEN, inner)
        if not dids or not all(known(d) for d in dids):
            return m.group(0)
        labels = []
        for d in dids:
            lab = _docid_paren_label(d)
            if lab not in labels:
                labels.append(lab)
        count += 1
        return "(" + "; ".join(labels) + ")"
    text = re.sub(
        rf"\((?=[^)]*{_DOC_ID_TOKEN})(?![^)]*:\s*p\.)[^)]*\)", c_repl, text
    )

    text = re.sub(r"\(\(([^()]*)\)\)", r"(\1)", text)
    return text, count


def _strip_wrapping(text: str) -> str:
    t = (text or "").strip()
    leaked_think_boundaries = list(
        re.finditer(r"</think\s*>", t, flags=re.IGNORECASE)
    )
    if leaked_think_boundaries:
        t = t[leaked_think_boundaries[-1].end():].strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def _emit_citations_manifest(
    text: str,
    allowed_docs: set,
    display_lookup: dict,
    pdf_paths_by_docid: dict,
    pdf_page_offsets: dict,
    dois_by_docid: dict,
    linkify: bool = False,
):
    from rrr.render import parse_citations
    citations = []
    seen_docs = set()
    for c in parse_citations(text, display_lookup=display_lookup):
        doc_id = c.get("doc_id")
        page = int(c.get("page") or 0) or None
        if doc_id and doc_id in allowed_docs:
            seen_docs.add(doc_id)
        pdf_path = pdf_paths_by_docid.get(doc_id) if doc_id else None
        offset = int(pdf_page_offsets.get(doc_id, 0) or 0) if doc_id else 0
        pdf_page = (page + offset) if (page is not None) else None
        doi = dois_by_docid.get(doc_id) if doc_id else None
        citations.append({
            "index": len(citations),
            "cite_text": c["raw"],
            "doc_id": doc_id,
            "page": page,
            "pdf_path": pdf_path,
            "pdf_page": pdf_page,
            "doi_or_url": doi or None,
            "start": c["start"],
            "end": c["end"],
            "surface": c["surface"],
        })

    if not citations:
        return text, {"citations": [], "distinct_docs": 0, "linkify": False}

    out_text = text
    if linkify:
        for cite in sorted(citations, key=lambda x: -x["start"]):
            if not cite["pdf_path"] or not cite["pdf_page"]:
                continue
            raw = cite["cite_text"]
            url = _pdf_page_url(cite["pdf_path"], cite["pdf_page"])
            link = f"[{raw}]({url})"
            out_text = out_text[:cite["start"]] + link + out_text[cite["end"]:]

    return out_text, {
        "citations": citations,
        "distinct_docs": len(seen_docs),
        "linkify": linkify,
    }


def _pdf_page_url(pdf_path: str, page: int) -> str:
    from urllib.parse import quote
    p = str(pdf_path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        url = "file:///" + quote(p, safe="/:")
    elif p.startswith("/"):
        url = "file://" + quote(p, safe="/")
    else:
        url = "file://" + quote(p, safe="/")
    return f"{url}#page={int(page)}"


def _citation_identity(citation: dict) -> tuple:
    doc_id = str(citation.get("doc_id") or "").strip().lower()
    if doc_id:
        return ("doc", doc_id, int(citation.get("page") or 0))
    return (
        "display",
        str(citation.get("label") or "").strip().lower(),
        str(citation.get("year") or ""),
        int(citation.get("page") or 0),
    )


def _dedupe_grouped_citation_identities(text: str, display_lookup=None) -> tuple:
    source = text or ""
    group_re = re.compile(r"\((?P<body>[^()\n;]+(?:;[^()\n;]+)+)\)")
    citations = list(parse_citations(source, display_lookup=display_lookup))
    replacements = []
    removed = 0

    for group in group_re.finditer(source):
        body = group.group("body")
        body_start = group.start("body")
        pieces = body.split(";")
        piece_records = []
        cursor = 0
        valid_group = True
        for piece in pieces:
            start = body_start + cursor
            end = start + len(piece)
            hits = [
                citation for citation in citations
                if citation["start"] >= start and citation["end"] <= end
            ]
            if len(hits) != 1:
                valid_group = False
                break
            piece_records.append((piece.strip(), _citation_identity(hits[0])))
            cursor += len(piece) + 1
        if not valid_group:
            continue

        seen = set()
        unique_pieces = []
        for piece, identity in piece_records:
            if identity in seen:
                removed += 1
                continue
            seen.add(identity)
            unique_pieces.append(piece)
        if len(unique_pieces) != len(piece_records):
            replacements.append((
                group.start(), group.end(),
                "(" + "; ".join(unique_pieces) + ")",
            ))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, removed


def _merge_adjacent_paren_cites(text: str, display_lookup=None,
                                stats: dict = None) -> tuple:
    if not text:
        if stats is not None:
            stats["duplicate_identities_removed"] = 0
        return text or "", 0
    matches = list(DISPLAY_PAREN_CITE_RE.finditer(text))
    groups: list = []
    current: list = [matches[0]] if matches else []
    separator_re = re.compile(r"\A(?:\s+|\s*;\s*)\Z")
    for m in matches[1:]:
        between = text[current[-1].end():m.start()]
        if separator_re.match(between):
            current.append(m)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [m]
    if len(current) > 1:
        groups.append(current)
    out: list = []
    cursor = 0
    merged_pairs = 0
    for grp in groups:
        out.append(text[cursor:grp[0].start()])
        inners = [m.group(0)[1:-1] for m in grp]
        out.append("(" + "; ".join(inners) + ")")
        cursor = grp[-1].end()
        merged_pairs += len(grp) - 1
    out.append(text[cursor:])
    merged = "".join(out) if groups else text
    deduped, duplicate_count = _dedupe_grouped_citation_identities(
        merged, display_lookup=display_lookup,
    )
    if stats is not None:
        stats["duplicate_identities_removed"] = duplicate_count
    return deduped, merged_pairs


def _collapse_redundant_narrative_labels(text: str) -> tuple:
    replacements = []
    for match in DISPLAY_PAREN_CITE_RE.finditer(text or ""):
        inner_label = match.group(1)
        label_variants = {
            inner_label,
            inner_label.rstrip("."),
            re.sub(r"\s+and\s+", " & ", inner_label, flags=re.IGNORECASE),
        }
        prefix = text[:match.start()]
        for variant in label_variants:
            label_match = re.search(
                rf"(?<!\w)({re.escape(variant)})(?:['\u2019]s)?\s*$",
                prefix,
                flags=re.IGNORECASE,
            )
            if not label_match:
                continue
            if _label_preceded_by_conjoined_name(prefix[:label_match.start(1)]):
                break
            replacement = (
                f"{inner_label} ({match.group(2)}, p.{match.group(3)})"
            )
            replacements.append((
                label_match.start(1),
                match.end(),
                replacement,
            ))
            break

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text, len(replacements)


def _citation_label_key(label: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"\s+&\s+", " and ", label or "", flags=re.IGNORECASE),
    ).strip().lower()


def _merge_same_author_mixed_citations(text: str) -> tuple:
    replacements = []
    for narrative in DISPLAY_CITE_RE.finditer(text or ""):
        separator = re.match(
            r"(?:\s+and\s+|\s*;\s*|\s+)",
            text[narrative.end():],
            flags=re.IGNORECASE,
        )
        if not separator:
            continue
        second_start = narrative.end() + separator.end()
        parenthetical = DISPLAY_PAREN_CITE_RE.match(text, second_start)
        if not parenthetical:
            continue
        if _citation_label_key(narrative.group(1)) != _citation_label_key(
            parenthetical.group(1)
        ):
            continue
        replacement = (
            f"{narrative.group(1)} "
            f"({narrative.group(2)}, p.{narrative.group(3)}; "
            f"{parenthetical.group(2)}, p.{parenthetical.group(3)})"
        )
        replacements.append((
            narrative.start(),
            parenthetical.end() + (
                1 if text[parenthetical.end():].startswith(")") else 0
            ),
            replacement,
        ))

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text, len(replacements)


_SEMICOLON_CITATION_CONTAINER_RE = re.compile(
    r"\((?P<body>[^();]*(?:;[^();]*)+)\)"
)


def _collapse_redundant_grouped_narrative_labels(text: str) -> tuple:
    source = text or ""
    citations = list(parse_citations(source))
    replacements = []

    for container in _SEMICOLON_CITATION_CONTAINER_RE.finditer(source):
        members = [
            citation for citation in citations
            if citation.get("label")
            and citation["start"] >= container.start()
            and citation["end"] <= container.end()
        ]
        if len(members) < 2:
            continue
        label_keys = {
            _citation_label_key(citation["label"]) for citation in members
        }
        if len(label_keys) != 1:
            continue
        label = str(members[0]["label"]).strip()
        prefix = source[:container.start()]
        variants = {
            label,
            label.rstrip("."),
            re.sub(r"\s+and\s+", " & ", label, flags=re.IGNORECASE),
        }
        outer_match = None
        for variant in variants:
            outer_match = re.search(
                rf"(?<!\w){re.escape(variant)}(?:['\u2019]s)?\s*$",
                prefix,
                flags=re.IGNORECASE,
            )
            if outer_match:
                break
        if not outer_match:
            continue

        compact_members = "; ".join(
            f"{citation['year']}, p.{citation['page']}"
            for citation in members
        )
        replacement = f"{label} ({compact_members})"
        if source[outer_match.start():container.end()] == replacement:
            continue
        replacements.append((
            outer_match.start(),
            container.end(),
            replacement,
        ))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, len(replacements)


_PAIR_AUTHOR_LABEL = (
    r"[A-Z][A-Za-z'\u2019\-]+"
    r"(?:\s+(?:(?:and|&)\s+[A-Z][A-Za-z'\u2019\-]+|et\s+al\.))?"
)
_REDUNDANT_AUTHOR_PAIR_RE = re.compile(
    rf"(?P<label1>{_PAIR_AUTHOR_LABEL})\s+\((?P<year1>\d{{4}})\)"
    rf"\s+and\s+"
    rf"(?P<label2>{_PAIR_AUTHOR_LABEL})\s+\((?P<year2>\d{{4}})\)"
    rf"\s+\("
    rf"(?P<cite_label1>{_PAIR_AUTHOR_LABEL})\s+(?P<cite_year1>\d{{4}}),"
    rf"\s*p\.(?P<page1>\d+);\s*"
    rf"(?P<cite_label2>{_PAIR_AUTHOR_LABEL})\s+(?P<cite_year2>\d{{4}}),"
    rf"\s*p\.(?P<page2>\d+)"
    rf"\)"
)


def _collapse_redundant_author_pair(text: str) -> tuple:
    count = 0

    def replace(match):
        nonlocal count
        pairs_match = (
            _citation_label_key(match.group("label1"))
            == _citation_label_key(match.group("cite_label1"))
            and _citation_label_key(match.group("label2"))
            == _citation_label_key(match.group("cite_label2"))
            and match.group("year1") == match.group("cite_year1")
            and match.group("year2") == match.group("cite_year2")
        )
        if not pairs_match:
            return match.group(0)
        count += 1
        return (
            f"{match.group('label1')} "
            f"({match.group('year1')}, p.{match.group('page1')}) and "
            f"{match.group('label2')} "
            f"({match.group('year2')}, p.{match.group('page2')})"
        )

    return _REDUNDANT_AUTHOR_PAIR_RE.sub(replace, text or ""), count


def _remove_redundant_author_year_mentions(text: str) -> tuple:
    source = text or ""
    pieces = re.split(r"(\n[ \t]*\n)", source)
    count = 0

    for index in range(0, len(pieces), 2):
        paragraph = pieces[index]
        if not paragraph.strip():
            continue
        identities = {
            (
                _citation_label_key(citation.get("label")),
                str(citation.get("year") or ""),
                str(citation.get("label") or "").strip(),
            )
            for citation in parse_citations(paragraph)
            if citation.get("label")
            and citation.get("year")
            and citation.get("page") is not None
        }
        cleaned = paragraph
        for _, year, label in identities:
            narrative_pattern = re.compile(
                rf"(?<!\w)(?P<label>{re.escape(label)})\s+"
                rf"\({re.escape(year)}\)",
                flags=re.IGNORECASE,
            )
            cleaned, removed = narrative_pattern.subn(
                lambda match: match.group("label"),
                cleaned,
            )
            count += removed
            parenthetical_pattern = re.compile(
                rf"\s*\(\s*{re.escape(label)}(?:,?\s+)"
                rf"{re.escape(year)}\s*\)",
                flags=re.IGNORECASE,
            )
            cleaned, removed = parenthetical_pattern.subn("", cleaned)
            count += removed
        pieces[index] = cleaned

    return "".join(pieces), count


def _reattach_stranded_parenthetical_citations(text: str) -> tuple:
    source = text or ""
    containers = list(_SEMICOLON_CITATION_CONTAINER_RE.finditer(source))
    grouped_spans = [(match.start(), match.end()) for match in containers]
    for match in DISPLAY_PAREN_CITE_RE.finditer(source):
        if any(
            start <= match.start() and match.end() <= end
            for start, end in grouped_spans
        ):
            continue
        containers.append(match)

    replacements = []
    for container in sorted(containers, key=lambda match: match.start()):
        if not list(parse_citations(container.group(0))):
            continue
        left = re.search(r"(?P<punct>[.!?])(?P<space>\s+)$",
                         source[:container.start()])
        right = re.match(r"(?P<space>\s+)(?=[A-Z])",
                         source[container.end():])
        if not left or not right:
            continue
        replacements.append((
            left.start("punct"),
            container.end(),
            f" {container.group(0)}{left.group('punct')}",
        ))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, len(replacements)


def _known_label_occurrences(text: str, display_lookup: dict) -> list:
    candidates = []
    labels = {
        str(label).strip()
        for label, _year in (display_lookup or {}).keys()
        if str(label).strip()
    }
    for label in sorted(labels, key=len, reverse=True):
        variants = {
            label,
            re.sub(r"\s+and\s+", " & ", label, flags=re.IGNORECASE),
        }
        for variant in variants:
            for match in re.finditer(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                text,
                flags=re.IGNORECASE,
            ):
                candidates.append((
                    match.start(),
                    match.end(),
                    _citation_label_key(label),
                    match.group(0),
                ))

    selected = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -item[1])):
        if any(
            candidate[0] < end and start < candidate[1]
            for start, end, *_ in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected)


_JOINT_SUBJECT_VERB_RE = re.compile(
    r"(?P<lead>\s*,?\s*)"
    r"(?P<verb>"
    r"argue|argues|assert|asserts|claim|claims|contend|contends|"
    r"demonstrate|demonstrates|document|documents|emphasise|emphasises|"
    r"emphasize|emphasizes|find|finds|highlight|highlights|"
    r"indicate|indicates|maintain|maintains|note|notes|observe|observes|"
    r"report|reports|reveal|reveals|show|shows|suggest|suggests|"
    r"underscore|underscores|write|writes"
    r")\b",
    flags=re.IGNORECASE,
)


def _realize_mixed_narrative_subject_chains(
    text: str,
    display_lookup: dict,
) -> tuple:
    if not display_lookup:
        return text or "", 0
    source = text or ""
    replacements = []
    lookup_keys = {
        (_citation_label_key(label), str(year))
        for label, year in display_lookup
    }

    cursor = 0
    for sentence in _split_sentences_for_cleanup(source):
        sentence_start = source.find(sentence, cursor)
        if sentence_start < 0:
            continue
        cursor = sentence_start + len(sentence)
        leading = re.match(r"\s*", sentence).end()
        first = DISPLAY_CITE_RE.match(sentence, leading)
        if not first:
            continue

        members = [{
            "label": first.group(1),
            "year": first.group(2),
            "page": first.group(3),
        }]
        if (
            _citation_label_key(first.group(1)),
            first.group(2),
        ) not in lookup_keys:
            continue

        position = first.end()
        while True:
            separator = re.match(
                r"(?:\s*;\s*|\s+)",
                sentence[position:],
            )
            if not separator:
                break
            candidate_start = position + separator.end()
            parenthetical = DISPLAY_PAREN_CITE_RE.match(
                sentence,
                candidate_start,
            )
            if not parenthetical:
                break
            if (
                _citation_label_key(parenthetical.group(1)),
                parenthetical.group(2),
            ) not in lookup_keys:
                break
            members.append({
                "label": parenthetical.group(1),
                "year": parenthetical.group(2),
                "page": parenthetical.group(3),
            })
            position = parenthetical.end()

        if len(members) < 2:
            continue
        verb = _JOINT_SUBJECT_VERB_RE.match(sentence, position)
        if not verb:
            continue

        grouped_members = []
        by_label = {}
        for member in members:
            key = _citation_label_key(member["label"])
            if key not in by_label:
                by_label[key] = []
                grouped_members.append(key)
            by_label[key].append(member)
        if len(grouped_members) < 2:
            continue

        realized_subjects = []
        for key in grouped_members:
            label_members = by_label[key]
            compact = "; ".join(
                f"{member['year']}, p.{member['page']}"
                for member in label_members
            )
            realized_subjects.append(
                f"{label_members[0]['label']} ({compact})"
            )
        replacement = " and ".join(realized_subjects) + " "
        replacements.append((
            sentence_start + leading,
            sentence_start + verb.end("lead"),
            replacement,
        ))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, len(replacements)


def _realize_leading_author_chains(text: str, display_lookup: dict) -> tuple:
    if not display_lookup:
        return text or "", 0
    source = text or ""
    replacements = []

    cursor = 0
    for sentence in _split_sentences_for_cleanup(source):
        sentence_start = source.find(sentence, cursor)
        if sentence_start < 0:
            continue
        cursor = sentence_start + len(sentence)
        containers = list(_SEMICOLON_CITATION_CONTAINER_RE.finditer(sentence))
        containers.extend(DISPLAY_PAREN_CITE_RE.finditer(sentence))
        containers = sorted(containers, key=lambda match: match.start())
        for container in containers:
            if container.start() > 180:
                break
            members = list(
                parse_citations(
                    container.group(0),
                    display_lookup=display_lookup,
                )
            )
            if not members or any(not member.get("label") for member in members):
                continue
            chain_end = container.end()
            while True:
                separator = re.match(r"(?:\s*;\s*|\s+)", sentence[chain_end:])
                if not separator:
                    break
                following = DISPLAY_PAREN_CITE_RE.match(
                    sentence, chain_end + separator.end(),
                )
                if not following:
                    break
                more = list(parse_citations(
                    following.group(0), display_lookup=display_lookup,
                ))
                if not more or any(not member.get("label") for member in more):
                    break
                members.extend(more)
                chain_end = following.end()
            prefix = sentence[:container.start()].strip()
            occurrences = _known_label_occurrences(prefix, display_lookup)
            if not occurrences:
                continue
            residue = list(prefix)
            for start, end, *_ in occurrences:
                residue[start:end] = " " * (end - start)
            residue_text = "".join(residue)
            if re.sub(
                r"(?i)\balong\s+with\b|\b(?:and|both|jointly)\b|[\s,&]",
                "",
                residue_text,
            ):
                continue
            subject_keys = {item[2] for item in occurrences}
            cited_keys = {
                _citation_label_key(member["label"]) for member in members
            }
            if not subject_keys.intersection(cited_keys):
                continue

            grouped_members = []
            by_label = {}
            for member in members:
                key = _citation_label_key(member["label"])
                if key not in by_label:
                    by_label[key] = []
                    grouped_members.append(key)
                by_label[key].append(member)
            realized_subjects = []
            for key in grouped_members:
                label_members = by_label[key]
                label = str(label_members[0]["label"]).strip()
                compact = "; ".join(
                    f"{member['year']}, p.{member['page']}"
                    for member in label_members
                )
                realized_subjects.append(f"{label} ({compact})")
            replacement = " and ".join(realized_subjects)
            original = sentence[:chain_end]
            if original == replacement:
                continue
            replacement_end = sentence_start + chain_end
            if sentence[chain_end:].startswith(","):
                replacement_end += 1
            replacements.append((
                sentence_start,
                replacement_end,
                replacement,
            ))
            break

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, len(replacements)


_BLOCKING_CITATION_SURFACE_FAULTS = frozenset({
    "unrendered_evidence_marker",
    "unresolved_display_citation",
})


def _find_citation_surface_faults(text: str, display_lookup: dict) -> list:
    source = text or ""
    faults = []
    for match in re.finditer(r"\[[^\]\n]*\b[Ee]\d{1,5}[^\]\n]*\]", source):
        faults.append({
            "type": "unrendered_evidence_marker",
            "snippet": match.group(0),
        })
    if display_lookup:
        for citation in parse_citations(source, display_lookup=display_lookup):
            if (
                citation.get("doc_id") is None
                and citation.get("label")
                and citation.get("page") is not None
            ):
                faults.append({
                    "type": "unresolved_display_citation",
                    "label": citation.get("label"),
                    "year": citation.get("year"),
                    "snippet": citation.get("raw"),
                })
    renormalized, pending_changes = _collapse_double_parens(
        source,
        display_lookup=display_lookup,
    )
    if pending_changes or renormalized != source:
        faults.append({
            "type": "noncanonical_citation_surface",
            "count": pending_changes,
            "snippet": _clip(source, n=220),
        })

    seen_spans = set()
    for (label, year), _doc_id in (display_lookup or {}).items():
        if not label or not year:
            continue
        variants = {
            label,
            re.sub(r"\s+and\s+", " & ", label, flags=re.IGNORECASE),
        }
        for variant in variants:
            patterns = (
                re.compile(
                    rf"(?<!\w){re.escape(variant)}\s+"
                    rf"\({re.escape(year)}\)",
                    flags=re.IGNORECASE,
                ),
                re.compile(
                    rf"\(\s*{re.escape(variant)}(?:,?\s+)"
                    rf"{re.escape(year)}\s*\)",
                    flags=re.IGNORECASE,
                ),
            )
            for pattern in patterns:
                for match in pattern.finditer(source):
                    span = (match.start(), match.end())
                    if span in seen_spans:
                        continue
                    seen_spans.add(span)
                    faults.append({
                        "type": "page_free_author_year",
                        "label": label,
                        "year": year,
                        "snippet": match.group(0),
                    })

    cursor = 0
    for sentence in _split_sentences_for_cleanup(source):
        sentence_start = source.find(sentence, cursor)
        if sentence_start < 0:
            continue
        cursor = sentence_start + len(sentence)
        containers = list(_SEMICOLON_CITATION_CONTAINER_RE.finditer(sentence))
        containers.extend(DISPLAY_PAREN_CITE_RE.finditer(sentence))
        for container in sorted(containers, key=lambda match: match.start()):
            members = list(
                parse_citations(
                    container.group(0),
                    display_lookup=display_lookup,
                )
            )
            if not members or any(not member.get("label") for member in members):
                continue
            prefix = sentence[:container.start()].strip()
            occurrences = _known_label_occurrences(prefix, display_lookup)
            if len(occurrences) < 2:
                continue
            residue = list(prefix)
            for start, end, *_ in occurrences:
                residue[start:end] = " " * (end - start)
            if re.sub(
                r"(?i)\balong\s+with\b|\b(?:and|both|jointly)\b|[\s,&]",
                "",
                "".join(residue),
            ):
                continue
            subject_keys = {item[2] for item in occurrences}
            cited_keys = {
                _citation_label_key(member["label"]) for member in members
            }
            unsupported = sorted(subject_keys - cited_keys)
            if unsupported:
                faults.append({
                    "type": "unsupported_leading_author",
                    "authors": unsupported,
                    "snippet": _clip(sentence, n=220),
                })
            break

    return faults


def _normalize_wrapped_narrative_groups(text: str) -> tuple:
    stack = []
    spans = []
    for index, char in enumerate(text or ""):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            spans.append((stack.pop(), index + 1))

    replacements = []
    occupied = []
    for start, end in sorted(spans, key=lambda span: (span[0], -span[1])):
        inner = text[start + 1:end - 1]
        if "(" not in inner or ")" not in inner:
            continue

        members = []
        member_start = 0
        depth = 0
        for index, char in enumerate(inner):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == ";" and depth == 0:
                members.append(inner[member_start:index].strip())
                member_start = index + 1
        members.append(inner[member_start:].strip())

        normalized = []
        for member in members:
            candidate = re.sub(
                r"\s+&\s+",
                " and ",
                member,
                flags=re.IGNORECASE,
            )
            citation = DISPLAY_CITE_RE.fullmatch(candidate)
            if not citation:
                normalized = []
                break
            normalized.append(
                f"{citation.group(1)} {citation.group(2)}, "
                f"p.{citation.group(3)}"
            )
        if not normalized:
            continue
        if any(start >= left and end <= right for left, right in occupied):
            continue

        if len(normalized) == 1:
            replacement = "(" + normalized[0] + ")"
        else:
            replacement = "(" + "; ".join(normalized) + ")"
        replacements.append((start, end, replacement))
        occupied.append((start, end))

    for start, end, replacement in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        text = text[:start] + replacement + text[end:]
    return text, len(replacements)


def _collapse_double_parens(text: str, display_lookup=None) -> tuple:
    count = 0

    def repl_inner_only(m):
        nonlocal count
        count += 1
        return "(" + m.group(1) + ")"

    def repl_nested_suffix(m):
        nonlocal count
        count += 1
        inner = re.match(r"\(\(([A-Za-z0-9_&.\-]+:\s*p\.\d+)\):\s*p\.\d+\)", m.group(0))
        if inner:
            return "(" + inner.group(1) + ")"
        return m.group(0)

    def repl_group(m):
        nonlocal count
        inner = m.group(1).strip().rstrip(";,").strip()
        if re.search(r"[A-Za-z][A-Za-z0-9]{2,}", re.sub(r"\([^()]*\)", "", inner)):
            return m.group(0)
        count += 1
        cites = re.findall(r"\([^()]+?\)", inner)
        return "; ".join(cites)

    def repl_asym(m):
        nonlocal count
        count += 1
        return "(" + m.group(1)

    text, _ = (_DOUBLE_PAREN_CITE_RE.subn(repl_inner_only, text))
    text, _ = (_NESTED_PAGE_SUFFIX_RE.subn(repl_nested_suffix, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_RE.subn(repl_inner_only, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_GROUP_RE.subn(repl_group, text))
    text, _ = (_ASYM_OUTER_PAREN_RE.subn(repl_asym, text))
    text, redundant_labels = _collapse_redundant_narrative_labels(text)
    count += redundant_labels
    text, redundant_grouped_labels = (
        _collapse_redundant_grouped_narrative_labels(text)
    )
    count += redundant_grouped_labels
    text, mixed_author_groups = _merge_same_author_mixed_citations(text)
    count += mixed_author_groups
    text, redundant_author_pairs = _collapse_redundant_author_pair(text)
    count += redundant_author_pairs
    text, wrapped_narratives = _normalize_wrapped_narrative_groups(text)
    count += wrapped_narratives
    text, redundant_year_mentions = _remove_redundant_author_year_mentions(text)
    count += redundant_year_mentions
    text, stranded_citations = _reattach_stranded_parenthetical_citations(text)
    count += stranded_citations
    text, mixed_subject_chains = _realize_mixed_narrative_subject_chains(
        text, display_lookup or {},
    )
    count += mixed_subject_chains
    text, author_chain_repairs = _realize_leading_author_chains(
        text, display_lookup or {},
    )
    count += author_chain_repairs
    return text, count


def _count_author_led_openings(text: str) -> int:
    count = 0
    for sentence in _split_sentences_for_cleanup(text):
        s = sentence.strip()
        if not s:
            continue
        head = re.sub(r"^[\(\[\"\'\s]+", "", s)[:120]
        if _AUTHOR_VERB_RE.search(head):
            count += 1
    return count


def _strip_trailing_significance(text: str) -> tuple:
    matches = list(_TRAILING_SIGNIFICANCE_RE.finditer(text or ""))
    if not matches:
        return text or "", 0
    cleaned = _TRAILING_SIGNIFICANCE_RE.sub("", text)
    cleaned = re.sub(r"([A-Za-z0-9_\)\]])(?=\n|$)", r"\1.", cleaned)
    return cleaned, len(matches)


def _mechanical_dash_replace(text: str) -> tuple:
    if not text:
        return text or "", 0
    count = 0

    def _dash_sub(m):
        nonlocal count
        s, e = m.start(), m.end()
        prev_c = text[s - 1] if s > 0 else ""
        next_c = text[e] if e < len(text) else ""
        if prev_c.isdigit() and next_c.isdigit():
            return m.group(0)
        count += 1
        return ","

    cleaned = _TYPOGRAPHIC_DASH_RE.sub(_dash_sub, text)
    if count == 0:
        return text, 0
    cleaned = re.sub(r"[ \t]*,[ \t]*", ", ", cleaned)
    cleaned = re.sub(r"([.,;:!?])[ \t]*,", r"\1", cleaned)
    cleaned = re.sub(r",[ \t]*([.,;:!?])", r"\1", cleaned)
    return cleaned, count


def _drop_zero_citation_paragraphs(
    text: str,
    keep_closing: bool = False,
    allowed_pairs=None,
    display_lookup=None,
) -> tuple:
    if not text:
        return text or "", [], None
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text or "", [], None
    last_idx = len(paragraphs) - 1
    kept = []
    removed = []
    kept_closing_snippet = None
    _UNRENDERED_EID_RE = re.compile(r"\[[Ee]\w*\]")
    for i, para in enumerate(paragraphs):
        has_canonical = bool(CITE_RE.search(para))
        has_display_narrative = bool(DISPLAY_CITE_RE.search(para))
        has_display_paren = bool(DISPLAY_PAREN_CITE_RE.search(para))
        has_resolved_display = bool(
            _strict_cited_doc_ids(
                para,
                allowed_pairs=allowed_pairs,
                display_lookup=display_lookup,
            )
        )
        has_unrendered_eid = bool(_UNRENDERED_EID_RE.search(para))
        if (
            has_canonical
            or has_display_narrative
            or has_display_paren
            or has_resolved_display
            or has_unrendered_eid
        ):
            kept.append(para)
            continue
        snippet = re.sub(r"\s+", " ", para.strip())[:200]
        snippet = snippet + ("…" if len(para.strip()) > 200 else "")
        if keep_closing and i == last_idx:
            kept.append(para)
            kept_closing_snippet = snippet
            continue
        removed.append(snippet)
    cleaned = "\n\n".join(kept)
    return cleaned, removed, kept_closing_snippet


def _apply_zero_citation_gate(
    text: str,
    enforce_citation_integrity: bool,
    allowed_pairs=None,
    display_lookup=None,
):
    if not enforce_citation_integrity:
        return text or "", [], None
    return _drop_zero_citation_paragraphs(
        text,
        keep_closing=False,
        allowed_pairs=allowed_pairs,
        display_lookup=display_lookup,
    )


def _apply_residual_duplicate_gate(final_structure: dict,
                                   residual_duplicates: int,
                                   enforce_citation_integrity: bool) -> dict:
    result = dict(final_structure)
    result["residual_duplicate_citation_identities"] = residual_duplicates
    result["ok"] = bool(
        result.get("ok")
        and (
            not enforce_citation_integrity
            or residual_duplicates == 0
        )
    )
    return result


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


_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?](?:[\"'\u2019\u201d)\]]*)$")
_TERMINAL_TOKEN_RE = re.compile(r"[.!?](?:[\"'\u2019\u201d)\]]*)")
_EVIDENCE_ID_AUDIT_RE = re.compile(r"\[[Ee]\d{1,5}\]")
_CITATION_SUFFIX_DELIMITER_RE = re.compile(r"[\s;,()\[\]]*")
_NONTERMINAL_ABBREVIATION_RES = (
    re.compile(r"\bet\s+al\.", re.IGNORECASE),
    re.compile(r"\b(?:e\.g|i\.e|cf)\.", re.IGNORECASE),
    re.compile(r"\bpp?\.(?=\s*\d)", re.IGNORECASE),
)


def _prose_paragraphs(text: str) -> list:
    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_RE.split(text or "")
        if paragraph.strip()
    ]


def _sentences_paragraph_local(text: str) -> list:
    sentences = []
    for paragraph in _prose_paragraphs(text):
        sentences.extend(_split_sentences_for_cleanup(paragraph))
    return sentences


def _ends_with_terminal_punctuation(text: str) -> bool:
    candidate = (text or "").strip()

    def mask_nonterminal_abbreviation_periods(value: str) -> str:
        masked = list(value)
        for pattern in _NONTERMINAL_ABBREVIATION_RES:
            for match in pattern.finditer(value):
                for index in range(match.start(), match.end()):
                    if masked[index] == ".":
                        masked[index] = " "
        return "".join(masked)

    if _TERMINAL_PUNCTUATION_RE.search(
        mask_nonterminal_abbreviation_periods(candidate)
    ):
        return True

    citations = list(parse_citations(candidate))
    spans = [
        (citation["start"], citation["end"])
        for citation in citations
    ]
    for citation in citations:
        if citation.get("surface") != "display_narrative":
            continue
        if not re.match(r"^\d{4}", citation.get("raw", "")):
            continue
        label = str(citation.get("label", "") or "").strip()
        if not label:
            continue
        prefix = candidate[:citation["start"]]
        label_match = re.search(
            rf"(?<!\w)({re.escape(label)})\s*\($",
            prefix,
            flags=re.IGNORECASE,
        )
        if label_match:
            spans.append(label_match.span(1))
    spans.extend(
        (match.start(), match.end())
        for match in _EVIDENCE_ID_AUDIT_RE.finditer(candidate)
    )
    if not spans:
        return False

    masked = list(candidate)
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    masked_text = "".join(masked)

    masked_text = mask_nonterminal_abbreviation_periods(masked_text)

    def balanced_delimiters(suffix: str) -> bool:
        stack = []
        pairs = {")": "(", "]": "["}
        for char in suffix:
            if char in "([":
                stack.append(char)
            elif char in ")]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return not stack

    for terminal in reversed(list(_TERMINAL_TOKEN_RE.finditer(masked_text))):
        suffix = masked_text[terminal.end():]
        if not _CITATION_SUFFIX_DELIMITER_RE.fullmatch(suffix):
            continue
        if not balanced_delimiters(suffix):
            continue
        if any(start >= terminal.end() for start, _ in spans):
            return True
    return False


def _audit_section_structure(text: str, section_kind: str) -> dict:
    paragraphs = _prose_paragraphs(text)
    incomplete = [
        {
            "paragraph_index": index,
            "snippet": _clip(paragraph, n=180),
        }
        for index, paragraph in enumerate(paragraphs)
        if not _ends_with_terminal_punctuation(paragraph)
    ]
    word_count = _count_words(text)
    closing_word_floor_ok = (
        section_kind != "closing" or word_count >= _MIN_CLOSING_WORDS
    )
    return {
        "paragraph_count": len(paragraphs),
        "word_count": word_count,
        "incomplete_paragraph_count": len(incomplete),
        "incomplete_paragraphs": incomplete,
        "closing_word_floor": (
            _MIN_CLOSING_WORDS if section_kind == "closing" else 0
        ),
        "closing_word_floor_ok": closing_word_floor_ok,
        "closing_word_ceiling": None,
        "closing_word_ceiling_ok": True,
        "closing_paragraph_count_ok": True,
        "ok": (
            bool(paragraphs)
            and not incomplete
            and closing_word_floor_ok
        ),
    }


def _collect_style_violations(text: str):
    sentences = _sentences_paragraph_local(text)
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


_EVIDENCE_ID_TOKEN_RE = re.compile(r"\[E\d{4,5}\]")
_STYLE_CITATION_GUARD_VERSION = "all-surfaces-v1"


def _citation_fingerprints(text: str):
    fingerprints = []
    for citation in parse_citations(text or ""):
        label = (
            citation.get("label") or citation.get("doc_id") or ""
        ).strip().lower()
        fingerprints.append((
            label,
            str(citation.get("year") or ""),
            int(citation.get("page") or 0),
        ))
    fingerprints.extend(
        ("__eid__", token, 0)
        for token in _EVIDENCE_ID_TOKEN_RE.findall(text or "")
    )
    return sorted(fingerprints)


def _rewrite_style_violations(sentences, violations, metrics=None):
    if not violations:
        return sentences, 0, "no_violations"
    prompt = _build_style_rewrite_prompt(violations)
    system = "You revise academic prose. Be concise. Return JSON only."
    prompt_path = _dump_writer_prompt("style_rewrite", system, prompt)
    try:
        import ollama
        import time
        start = time.perf_counter()
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options=_OPTIONS_STYLE_REWRITE,
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        _dump_writer_response(prompt_path, response=raw)
        if metrics:
            metrics.record_llm("style_rewrite", _MODEL, options=_OPTIONS_STYLE_REWRITE,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))
    except Exception as e:
        _dump_writer_response(prompt_path, error=e)
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
        if not isinstance(new_text, str):
            continue
        new_text = new_text.strip()
        if not new_text:
            continue
        if "\n" in new_text or "\r" in new_text:
            continue
        if (
            len(_split_sentences_for_cleanup(new_text)) != 1
            or not _ends_with_terminal_punctuation(new_text)
        ):
            continue
        if _classify_sentence_violations(new_text):
            continue
        original_cites = _citation_fingerprints(original)
        new_cites = _citation_fingerprints(new_text)
        if original_cites != new_cites:
            continue
        new_sentences[idx] = new_text
        rewrites_applied += 1

    return new_sentences, rewrites_applied, "ok"


def _splice_sentences_back_checked(original_text: str, new_sentences) -> tuple:
    if not original_text:
        return "", "empty"

    parts = re.split(r"(\n[ \t]*\n)", original_text)
    paragraph_indices = list(range(0, len(parts), 2))
    expected_counts = [
        len(_split_sentences_for_cleanup(parts[index]))
        for index in paragraph_indices
    ]
    if sum(expected_counts) != len(new_sentences):
        return original_text, "sentence_count_mismatch"
    for sentence in new_sentences:
        if not isinstance(sentence, str) or "\n" in sentence or "\r" in sentence:
            return original_text, "paragraph_injection"
        if len(_split_sentences_for_cleanup(sentence.strip())) != 1:
            return original_text, "replacement_sentence_count_mismatch"

    out_parts = list(parts)
    cursor = 0
    for part_index, n in zip(paragraph_indices, expected_counts):
        para = parts[part_index]
        if n == 0:
            out_parts[part_index] = para
            continue
        replacement = new_sentences[cursor:cursor + n]
        cursor += n
        if len(replacement) != n or any(not s.strip() for s in replacement):
            return original_text, "empty_replacement"
        out_parts[part_index] = " ".join(s.strip() for s in replacement)

    rewritten = "".join(out_parts).strip()
    if len(_prose_paragraphs(rewritten)) != len(_prose_paragraphs(original_text)):
        return original_text, "paragraph_count_mismatch"
    return rewritten, "ok"


def _splice_sentences_back(original_text: str, new_sentences) -> str:
    rewritten, _reason = _splice_sentences_back_checked(
        original_text, new_sentences,
    )
    return rewritten


def _apply_style_enforcement(text: str, metrics=None):
    if not text:
        return text, {"trailing_stripped": 0, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "empty",
                      "mechanical_dashes_replaced": 0,
                      "paragraphs_before": 0, "paragraphs_after": 0,
                      "paragraph_layout_preserved": True,
                      "layout_fallback_reason": "empty",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}

    text, trailing_stripped = _strip_trailing_significance(text)
    paragraphs_before = len(_prose_paragraphs(text))

    sentences, violations = _collect_style_violations(text)
    if not violations:
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "no_violations",
                      "mechanical_dashes_replaced": mechanical_replaced,
                      "paragraphs_before": paragraphs_before,
                      "paragraphs_after": len(_prose_paragraphs(text)),
                      "paragraph_layout_preserved": (
                          len(_prose_paragraphs(text)) == paragraphs_before
                      ),
                      "layout_fallback_reason": "no_rewrite",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}

    new_sentences, applied, reason = _rewrite_style_violations(
        sentences, violations, metrics=metrics,
    )
    if applied == 0:
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped,
                      "violations": len(violations),
                      "rewrites_applied": 0,
                      "fallback_reason": reason,
                      "mechanical_dashes_replaced": mechanical_replaced,
                      "paragraphs_before": paragraphs_before,
                      "paragraphs_after": len(_prose_paragraphs(text)),
                      "paragraph_layout_preserved": (
                          len(_prose_paragraphs(text)) == paragraphs_before
                      ),
                      "layout_fallback_reason": "no_usable_rewrite",
                      "citation_fingerprint_guard":
                          _STYLE_CITATION_GUARD_VERSION}
    rewritten_text, layout_reason = _splice_sentences_back_checked(
        text, new_sentences,
    )
    if layout_reason != "ok":
        applied = 0
    rewritten_text, mechanical_replaced = _mechanical_dash_replace(rewritten_text)
    return rewritten_text, {"trailing_stripped": trailing_stripped,
                            "violations": len(violations),
                            "rewrites_applied": applied,
                            "fallback_reason": reason,
                            "mechanical_dashes_replaced": mechanical_replaced,
                            "paragraphs_before": paragraphs_before,
                            "paragraphs_after": len(_prose_paragraphs(rewritten_text)),
                            "paragraph_layout_preserved": (
                                len(_prose_paragraphs(rewritten_text))
                                == paragraphs_before
                            ),
                            "layout_fallback_reason": layout_reason,
                            "citation_fingerprint_guard":
                                _STYLE_CITATION_GUARD_VERSION}


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


def _sentence_citation_pairs(sentence: str, display_lookup=None):
    pairs = set()
    for c in parse_citations(sentence, display_lookup=display_lookup):
        if c["doc_id"] is None:
            continue
        pairs.add((c["doc_id"], c["page"]))
    return pairs


def _sentence_doc_ids(sentence: str, display_lookup=None) -> set:
    return {p[0] for p in _sentence_citation_pairs(sentence, display_lookup=display_lookup)}


def _drop_would_strand_surname(prev_text: str, sent: str, allowed_docs,
                                 display_lookup=None) -> bool:
    if not prev_text or not sent:
        return False
    tail = prev_text[-120:]
    sent_doc_ids = _sentence_doc_ids(sent, display_lookup=display_lookup)
    if not sent_doc_ids:
        return False
    covered_doc_ids = _sentence_doc_ids(prev_text, display_lookup=display_lookup)
    surname_to_docids: dict = {}
    for did in (allowed_docs or set()):
        surnames = _author_surnames_only(_doc_id_to_author_label(did)).lower()
        if surnames:
            surname_to_docids.setdefault(surnames, set()).add(did)
    tail_lower = tail.lower()
    for surnames, dids in surname_to_docids.items():
        if surnames in tail_lower and (dids & sent_doc_ids) and not (dids & covered_doc_ids):
            return True
    return False


def _drop_cross_section_redundancy(text: str, min_token_overlap: int = 4,
                                     display_lookup=None, allowed_docs=None):
    if not text:
        return text, []
    paragraphs = text.split("\n\n")
    seen_pairs: set = set()
    seen_tokens: list = []
    removed_examples = []
    new_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        kept_lines = []
        para_removed = []
        para_first_sentence = None
        para_sentence_count = 0
        for line in lines:
            sentences = _split_sentences_for_cleanup(line)
            if not sentences:
                kept_lines.append(line)
                continue
            if para_first_sentence is None:
                para_first_sentence = sentences[0]
            para_sentence_count += len(sentences)
            kept_sentences = []
            for sent in sentences:
                sent_pairs = _sentence_citation_pairs(sent, display_lookup=display_lookup)
                if not sent_pairs:
                    kept_sentences.append(sent)
                    continue
                if not sent_pairs.issubset(seen_pairs):
                    kept_sentences.append(sent)
                    seen_pairs |= sent_pairs
                    seen_tokens.append(_redundancy_tokens(sent))
                    continue
                this_tokens = _redundancy_tokens(sent)
                max_overlap = max((len(this_tokens & prev) for prev in seen_tokens), default=0)
                if max_overlap >= min_token_overlap:
                    prev_text = " ".join(kept_sentences)
                    if _drop_would_strand_surname(
                        prev_text, sent, allowed_docs, display_lookup=display_lookup,
                    ):
                        kept_sentences.append(sent)
                        seen_tokens.append(this_tokens)
                        continue
                    snippet = re.sub(r"\s+", " ", sent.strip())[:140]
                    para_removed.append({
                        "sentence": sent,
                        "snippet": snippet + ("…" if len(snippet) == 140 else ""),
                        "overlap_tokens": int(max_overlap),
                        "cited_pairs": sorted(list(sent_pairs)),
                    })
                    continue
                kept_sentences.append(sent)
                seen_tokens.append(this_tokens)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences))
        if not kept_lines and para_sentence_count > 1:
            kept_lines.append(para_first_sentence)
            seen_tokens.append(_redundancy_tokens(para_first_sentence))
            para_removed = [
                record for record in para_removed
                if record["sentence"] != para_first_sentence
            ]
        removed_examples.extend(
            {k: v for k, v in record.items() if k != "sentence"}
            for record in para_removed
        )
        if kept_lines:
            new_paragraphs.append("\n".join(kept_lines))
    cleaned = "\n\n".join(new_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed_examples


def _drop_body_redundancy_preserving_closing(
        text: str, min_token_overlap: int = 4,
        display_lookup=None, allowed_docs=None) -> tuple:
    split = _split_marked_closing(text)
    if not split["ok"]:
        raise ValueError("closing identity missing before redundancy cleanup")
    body, removed = _drop_cross_section_redundancy(
        split["body"],
        min_token_overlap=min_token_overlap,
        display_lookup=display_lookup,
        allowed_docs=allowed_docs,
    )
    joined = (
        body.rstrip()
        + "\n\n"
        + _CLOSING_START_MARKER
        + " "
        + split["closing"].lstrip()
    )
    return joined, removed, _count_words(split["closing"])


def _split_sentences_for_cleanup(line: str):
    sentinel = "__RRR_DOT__"

    def protect(m):
        return m.group(0).replace(".", sentinel)

    protected = re.sub(r"\bet\s+al\.", protect, line, flags=re.IGNORECASE)
    protected = re.sub(r"\b(?:e\.g|i\.e|cf)\.", protect, protected, flags=re.IGNORECASE)
    protected = re.sub(r"\bpp?\.(?=\s*\d)", protect, protected, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace(sentinel, ".") for p in parts if p.strip()]


def _remove_invalid_citations(text: str, allowed_docs: set, allowed_pairs=None,
                                display_lookup=None) -> tuple:
    allowed_lower = {did.lower() for did in allowed_docs}
    lower_to_canonical = {did.lower(): did for did in allowed_docs}
    allowed_pairs = set(allowed_pairs or [])

    removed = []

    def find_invalid_citations_in_text(txt):
        invalid = []
        for c in parse_citations(txt, display_lookup=display_lookup):
            page = int(c["page"])
            doc_id = c["doc_id"]
            if doc_id is None:
                invalid.append((
                    c["start"], c["end"],
                    c.get("label") or "?",
                    page,
                    "unresolved_display",
                ))
                continue
            if c.get("candidates") and citation_inside(
                c, allowed_docs, allowed_pairs,
            ):
                continue
            if doc_id.lower() not in allowed_lower:
                invalid.append((c["start"], c["end"], doc_id, page, "unknown_doc"))
                continue
            canonical = lower_to_canonical.get(doc_id.lower(), doc_id)
            if allowed_pairs and (canonical, page) not in allowed_pairs:
                invalid.append((c["start"], c["end"], canonical, page, "invalid_page"))
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


_META_COMMENTARY_RE = re.compile(
    r"\b("
    r"the literature reviewed (?:here|above)?"
    r"|this (?:review|paper|essay|article|analysis|discussion|investigation)"
    r"|the (?:sources|works|studies|papers) (?:cited|reviewed|discussed|surveyed)"
    r"|as (?:discussed|noted|shown|argued|established) (?:above|earlier)"
    r"|in (?:this|the (?:foregoing|preceding)) (?:review|paper|essay|article|section|discussion)"
    r"|we (?:will|shall) (?:examine|explore|investigate|consider|argue|show|discuss|analyse|analyze)"
    r"|this (?:section|paragraph) (?:examines|explores|investigates|considers|argues|shows|discusses)"
    r"|(?:applies|illustrates how|qualification(?:s)? to|adds nuance to|in support of|against) the thesis"
    r"|this is (?:an? )?important (?:qualification|implication|consequence|insight|point) (?:to|for|of) the thesis"
    r")\b",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> tuple:
    if not text:
        return text, []
    removed = []
    cleaned_paragraphs = []
    for para in text.split("\n\n"):
        kept_sents = []
        for sent in _split_sentences_for_cleanup(para):
            if _META_COMMENTARY_RE.search(sent):
                removed.append(_clip(sent, n=200))
            else:
                kept_sents.append(sent)
        if kept_sents:
            cleaned_paragraphs.append(" ".join(kept_sents))
    cleaned = "\n\n".join(cleaned_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), removed


def _strip_orphaned_citations(text: str) -> str:
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


def _strip_tail_after(text: str, patterns, min_frac: float = 0.6) -> str:
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            if m.start() >= len(text) * min_frac:
                text = text[:m.start()]
                break
    return text.strip()


def _strip_conclusion(text: str) -> str:
    return _strip_tail_after(text, [
        r'\n\s*In conclusion[,.].*$',
        r'\n\s*To conclude[,.].*$',
        r'\n\s*In summary[,.].*$',
    ])


def _strip_references_section(text: str) -> str:
    return _strip_tail_after(text, [
        r'\n\s*References\s*:?\s*\n.*$',
        r'\n\s*Bibliography\s*:?\s*\n.*$',
        r'\n\s*Works Cited\s*:?\s*\n.*$',
        r'\n\s*\(References:.*?\).*$',
    ])


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _writer_enforcement_enabled() -> bool:
    return _ENFORCE_COVERAGE and os.environ.get("RRR_BYPASS_VALIDATION", "0") != "1"


def _writer_parallel_workers(n_chunks: int) -> int:
    if n_chunks <= 1 or os.environ.get("RRR_WRITER_PARALLEL", "0") == "0":
        return 1
    raw = os.environ.get("RRR_WRITER_PARALLELISM") or os.environ.get("RRR_CONCURRENCY") or "2"
    try:
        workers = int(raw)
    except Exception:
        workers = 1
    return max(1, min(workers, n_chunks))


def _strict_cited_doc_ids(text: str, allowed_pairs=None, display_lookup=None) -> set:
    allowed_pairs = set(allowed_pairs or [])
    cited = set()
    for c in parse_citations(text, display_lookup=display_lookup):
        if c["doc_id"] is None:
            continue
        pair = (c["doc_id"], c["page"])
        if not allowed_pairs or pair in allowed_pairs:
            cited.add(c["doc_id"])
    return cited


def _coverage_requirement(chunk_docs, section_kind: str) -> int:
    n_docs = len([d for d in chunk_docs if d.get("doc_id")])
    if n_docs <= 0:
        return 0
    if section_kind == "closing":
        return min(2, n_docs)
    return min(max(1, _MIN_SECTION_CITED_DOCS), n_docs)


def _audit_section_coverage(text: str, chunk_docs, section_kind: str):
    chunk_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
    chunk_display_lookup_local = _build_display_lookup(chunk_allowed_docs)
    cited = _strict_cited_doc_ids(
        text, allowed_pairs=chunk_pairs, display_lookup=chunk_display_lookup_local,
    )
    required = _coverage_requirement(chunk_docs, section_kind)
    structure = _audit_section_structure(text, section_kind)
    uncited_closing_paragraphs = []
    if section_kind == "closing":
        for index, paragraph in enumerate(_prose_paragraphs(text)):
            paragraph_cited = _strict_cited_doc_ids(
                paragraph,
                allowed_pairs=chunk_pairs,
                display_lookup=chunk_display_lookup_local,
            )
            if not paragraph_cited:
                uncited_closing_paragraphs.append(index)
    coverage_ok = len(cited) >= required
    closing_paragraph_citations_ok = not uncited_closing_paragraphs
    return {
        "section": section_kind,
        "required_cited_docs": required,
        "cited_doc_count": len(cited),
        "cited_docs": sorted(cited),
        "provided_doc_count": len(chunk_allowed_docs),
        "coverage_ok": coverage_ok,
        "paragraph_count": structure["paragraph_count"],
        "word_count": structure["word_count"],
        "incomplete_paragraph_count": structure["incomplete_paragraph_count"],
        "incomplete_paragraphs": structure["incomplete_paragraphs"],
        "closing_word_floor": structure["closing_word_floor"],
        "closing_word_floor_ok": structure["closing_word_floor_ok"],
        "closing_word_ceiling": structure["closing_word_ceiling"],
        "closing_word_ceiling_ok": structure["closing_word_ceiling_ok"],
        "closing_paragraph_count_ok": structure[
            "closing_paragraph_count_ok"
        ],
        "uncited_closing_paragraphs": uncited_closing_paragraphs,
        "closing_paragraph_citations_ok": closing_paragraph_citations_ok,
        "structure_ok": structure["ok"],
        "ok": (
            coverage_ok
            and structure["ok"]
            and closing_paragraph_citations_ok
        ),
    }


def _section_candidate_passes(audit: dict, render_stats: dict,
                              section_kind: str = "closing",
                              structural_only: bool = False) -> bool:
    if structural_only:
        return bool(audit.get("structure_ok"))
    if not audit.get("ok"):
        return False
    return bool(
        int(render_stats.get("unknown_eids", 0) or 0) == 0
        and int(render_stats.get("attribution_mismatches", 0) or 0) == 0
        and int(
            render_stats.get("unsupported_year_only_citations", 0) or 0
        ) == 0
    )


def _section_quality_faults(audit: dict, render_stats: dict,
                            section_kind: str,
                            structural_only: bool = False) -> list:
    faults = []
    if int(audit.get("paragraph_count", 0) or 0) == 0:
        faults.append("The output contained no prose paragraph.")
    incomplete = int(audit.get("incomplete_paragraph_count", 0) or 0)
    if incomplete:
        faults.append(
            f"{incomplete} paragraph(s) ended without a complete sentence."
        )
    if section_kind == "closing" and not audit.get(
        "closing_word_floor_ok", True
    ):
        faults.append(
            "The closing was below the minimum length: "
            f"{int(audit.get('word_count', 0) or 0)} words supplied, "
            f"{int(audit.get('closing_word_floor', 0) or 0)} required."
        )
    if not structural_only and not audit.get("coverage_ok", False):
        faults.append(
            "Citation coverage was insufficient: "
            f"{int(audit.get('cited_doc_count', 0) or 0)} distinct papers "
            f"cited, {int(audit.get('required_cited_docs', 0) or 0)} required."
        )
    uncited = (
        [] if structural_only
        else list(audit.get("uncited_closing_paragraphs") or [])
    )
    if uncited:
        faults.append(
            "Closing paragraph(s) without a valid citation: "
            + ", ".join(str(index + 1) for index in uncited) + "."
        )
    unknown = 0 if structural_only else int(
        render_stats.get("unknown_eids", 0) or 0
    )
    if unknown:
        faults.append(
            f"The output used {unknown} unknown or malformed evidence ID(s)."
        )
    mismatches = 0 if structural_only else int(
        render_stats.get("attribution_mismatches", 0) or 0
    )
    if mismatches:
        faults.append(
            f"The output contained {mismatches} author/evidence attribution "
            "mismatch(es)."
        )
    unsupported_years = 0 if structural_only else int(
        render_stats.get("unsupported_year_only_citations", 0) or 0
    )
    if unsupported_years:
        faults.append(
            f"The output used {unsupported_years} author-linked year-only "
            "citation(s) without an evidence page."
        )
    return faults or ["The output failed the section quality contract."]


def _section_retry_reason(audit: dict, render_stats: dict,
                          section_kind: str,
                          structural_only: bool = False) -> str:
    if structural_only:
        return "structure"
    if int(render_stats.get("attribution_mismatches", 0) or 0):
        return "attribution"
    if (
        int(render_stats.get("unknown_eids", 0) or 0)
        or int(render_stats.get("unsupported_year_only_citations", 0) or 0)
    ):
        return "evidence"
    if (
        not audit.get("structure_ok", False)
        or (
            section_kind == "closing"
            and not audit.get("closing_paragraph_citations_ok", False)
        )
    ):
        return "structure"
    return "coverage"


def _section_retry_stage(stage: str, retry_reason: str, attempt: int,
                         section_kind: str) -> str:
    base = f"{stage}_{retry_reason}_retry"
    return f"{base}_{attempt}" if section_kind == "closing" else base


def _resolve_nonclosing_retry(prior_state: dict, retry_state: dict,
                               retry_reason: str,
                               section_kind: str = "stream"):
    if section_kind == "closing":
        return retry_state, "unresolved"
    prior_mismatches = int(
        prior_state["render_stats"].get("attribution_mismatches", 0) or 0
    )
    retry_mismatches = int(
        retry_state["render_stats"].get("attribution_mismatches", 0) or 0
    )
    attribution_improved = (
        retry_mismatches == 0
        or (
            retry_reason == "attribution"
            and 0 < retry_mismatches < prior_mismatches
        )
    )
    if retry_state["audit"].get("ok") and attribution_improved:
        accepted = dict(retry_state)
        accepted["_nonclosing_compat_accepted"] = True
        return accepted, "retry_accepted"
    prior_unknown = int(
        prior_state["render_stats"].get("unknown_eids", 0) or 0
    )
    if (
        retry_reason == "attribution"
        and prior_state["audit"].get("ok")
        and prior_unknown == 0
        and prior_mismatches > 0
    ):
        retained = dict(prior_state)
        retained["_nonclosing_compat_accepted"] = True
        return retained, "original_retained"
    return retry_state, "unresolved"


def _run_bounded_section_retries(initial_state: dict, section_kind: str,
                                   retry_candidate,
                                   structural_only: bool = False):
    state = initial_state
    if _section_candidate_passes(
        state["audit"], state["render_stats"], section_kind,
        structural_only,
    ):
        return state, 0, True
    max_retries = 2 if section_kind == "closing" else 1
    for attempt in range(1, max_retries + 1):
        state = retry_candidate(state, attempt)
        if (
            state.get("_nonclosing_compat_accepted", False)
            or _section_candidate_passes(
                state["audit"], state["render_stats"], section_kind,
                structural_only,
            )
        ):
            return state, attempt, True
    return state, max_retries, False


def _split_marked_closing(text: str) -> dict:
    source = text or ""
    marker_count = source.count(_CLOSING_START_MARKER)
    if marker_count != 1:
        return {
            "ok": False,
            "marker_count": marker_count,
            "marker_at_paragraph_start": False,
            "body": source,
            "closing": "",
            "without_marker": source,
        }
    marker_start = source.index(_CLOSING_START_MARKER)
    prefix = source[:marker_start]
    marker_at_paragraph_start = (
        not prefix.strip()
        or bool(re.search(r"\n[ \t]*\n[ \t]*$", prefix))
    )
    body = prefix.rstrip()
    closing = source[marker_start + len(_CLOSING_START_MARKER):].lstrip()
    if body and closing:
        without_marker = body + "\n\n" + closing
    else:
        without_marker = body or closing
    return {
        "ok": marker_at_paragraph_start and bool(closing),
        "marker_count": marker_count,
        "marker_at_paragraph_start": marker_at_paragraph_start,
        "body": body,
        "closing": closing,
        "without_marker": without_marker.strip(),
    }


def _audit_final_structure(text: str, closing_docs,
                           enforce_citation_integrity: bool = True) -> dict:
    split = _split_marked_closing(text)
    whole = _audit_section_structure(split["without_marker"], "full_review")
    closing_audit = _audit_section_coverage(
        split["closing"], closing_docs, "closing",
    )
    closing_ok = (
        (closing_audit["coverage_ok"] and closing_audit["structure_ok"])
        if enforce_citation_integrity
        else closing_audit["structure_ok"]
    )
    return {
        "ok": bool(split["ok"] and whole["ok"] and closing_ok),
        "citation_integrity_enforced": enforce_citation_integrity,
        "closing_uncited_paragraphs_recorded": list(
            closing_audit.get("uncited_closing_paragraphs") or []
        ),
        "closing_marker_count": split["marker_count"],
        "closing_marker_at_paragraph_start": split["marker_at_paragraph_start"],
        "paragraph_count": whole["paragraph_count"],
        "incomplete_paragraph_count": whole["incomplete_paragraph_count"],
        "incomplete_paragraphs": whole["incomplete_paragraphs"],
        "closing": closing_audit,
        "text_without_marker": split["without_marker"],
    }


_COVERAGE_PATCH_SENTINEL = "[[COV-PATCH]] "


def _complete_coverage_excerpt(text: str, max_chars: int = 360) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    candidates = []
    for sentence in _split_sentences_for_cleanup(normalized):
        candidate = sentence.strip().strip("\"'“”‘’ ")
        if not candidate or len(candidate) > max_chars:
            continue
        if not re.search(r"[.!?][\"'”’]?$", candidate):
            continue
        candidates.append(candidate)

    if not candidates:
        return ""
    meta_opening = re.compile(
        r"^(?:our argument|this (?:paper|article|chapter|study)|"
        r"we (?:argue|show|examine|investigate|document))\b",
        flags=re.IGNORECASE,
    )
    return next(
        (candidate for candidate in candidates if not meta_opening.search(candidate)),
        candidates[0],
    )


def _append_coverage_fallback(text: str, chunk_docs, required_docs: int,
                              allowed_pairs=None, display_lookup=None) -> tuple:
    allowed_pairs = set(allowed_pairs or [])
    cited = _strict_cited_doc_ids(
        text, allowed_pairs=allowed_pairs, display_lookup=display_lookup,
    )
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
        excerpt = _complete_coverage_excerpt(quote.get("text", ""))
        if not excerpt:
            continue
        author_label = _author_surnames_only(
            _doc_id_to_author_label(did)
        )
        reporting_verb = (
            "report"
            if re.search(r"(?:\bet\s+al\.$|\band\b)", author_label)
            else "reports"
        )
        additions.append(
            f"{_COVERAGE_PATCH_SENTINEL}{render_citation(did, pg)} "
            f"{reporting_verb}, “{excerpt}”"
        )
        cited.add(did)
        if len(cited) >= required_docs:
            break

    if not additions:
        return text, 0

    patched = (text or "").rstrip()
    if patched:
        patched += " "
    patched += " ".join(additions)
    return patched.strip(), len(additions)


def _strip_coverage_patches_when_safe(text: str, allowed_docs, allowed_pairs,
                                        display_lookup=None) -> tuple:
    if not text or _COVERAGE_PATCH_SENTINEL not in text:
        return text, 0, 0
    allowed_pairs = set(allowed_pairs or [])
    n_dropped = 0
    n_kept = 0
    out_paras = []
    for para in text.split("\n\n"):
        if _COVERAGE_PATCH_SENTINEL not in para:
            out_paras.append(para)
            continue
        sentences = _split_sentences_for_cleanup(para)
        non_patch_sents = [s for s in sentences if _COVERAGE_PATCH_SENTINEL not in s]
        joined_non_patch = " ".join(non_patch_sents)
        non_patch_cited = _strict_cited_doc_ids(
            joined_non_patch,
            allowed_pairs=allowed_pairs,
            display_lookup=display_lookup,
        )
        if len(non_patch_cited) >= 2:
            new_para = " ".join(non_patch_sents).strip()
            if new_para:
                out_paras.append(new_para)
                n_dropped += sum(
                    1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
                )
            else:
                cleaned = para.replace(_COVERAGE_PATCH_SENTINEL, "")
                out_paras.append(cleaned)
                n_kept += sum(
                    1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
                )
        else:
            cleaned = para.replace(_COVERAGE_PATCH_SENTINEL, "")
            out_paras.append(cleaned)
            n_kept += sum(
                1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
            )
    cleaned_text = "\n\n".join(out_paras)
    return cleaned_text, n_dropped, n_kept


def _coverage_retry_prompt(original_prompt: str, prior_chunk: str,
                           required_docs: int, section_kind: str = "",
                           retry_reason: str = "coverage",
                           faults=None) -> str:
    reason_text = {
        "structure": (
            "The previous output contained an incomplete paragraph, an invalid "
            "paragraph boundary, or an undersized closing."
        ),
        "attribution": (
            "The previous output attached an evidence marker to an incompatible "
            "author attribution."
        ),
        "coverage": (
            "The previous output did not cite enough distinct papers from the "
            "allowed evidence."
        ),
        "evidence": (
            "The previous output used an unknown or malformed evidence ID."
        ),
    }.get(retry_reason, "The previous output failed a section quality rule.")
    fault_lines = "\n".join(
        f"- {fault}" for fault in (faults or [reason_text])
    )
    clean_room = section_kind == "closing"
    prior_block = "" if clean_room else f"""

Previous draft:
{prior_chunk}
"""
    repair_mode = (
        "Generate a fresh closing from the allowed evidence above. Do not reuse "
        "the rejected output."
        if clean_room else
        "Write the section again using the allowed evidence above."
    )
    return f"""{original_prompt}

Section repair:
The previous output failed the checks listed below:
{fault_lines}

{repair_mode} Use only [E####] markers drawn from the ALLOWED CITATIONS list at the top of this prompt.

Requirements:
- Cite at least {required_docs} DIFFERENT papers when that many are listed. (One paper can contribute multiple markers; the count is of distinct papers.)
- Every paragraph must include at least one [E####] marker.
- Use ONLY [E####] markers from the ALLOWED CITATIONS list. Copy every allowed marker character-for-character. Do not invent IDs. Do not write '(Author Year, p.N)', 'Author (Year, p.N)', '(DocId: p.N)' or any other citation surface — the renderer expands every [E####] into the correct surface.
- Preserve the same substantive role and word range.
- Finish every paragraph with a complete standalone sentence and terminal punctuation.
{prior_block}

{_OUTPUT_ONLY_DIRECTIVE}

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


_PROSE_DIRECTIVE = (
    "Write a scholarly literature review organised around streams of "
    "related scholarship. Each section presents ONE stream and synthesises "
    "across its sources. Argue in direct positive claims; avoid contrastive "
    "framings such as 'not X but Y', 'rather than', 'unlike', or 'in "
    "contrast'. Vary sentence length; introduce claims as full sentences "
    "rather than via colon setups. "
    "Each paragraph must integrate at least two DISTINCT documents; do not "
    "cite the same document for more than two consecutive citations. "
    "Finish every paragraph and section with a complete standalone sentence. "
    "Create continuity by carrying a substantive theme across sections. "
    "Make claims directly about the substantive question. Do NOT write meta-"
    "statements about the review itself ('this review', 'the literature "
    "reviewed here', 'as discussed above', 'we will examine'). "
    "Do not restate the topic or the thesis."
)

def _prev_tail_block(previous_tail: str) -> str:
    if not previous_tail:
        return ""
    block = f"Previous ending:\n...{previous_tail}\n"
    if os.environ.get("RRR_WRITER_PREV_TAIL_STRONG", "1") == "1":
        block += (
            "\nIMPORTANT: The Previous ending is a complete sentence. Open "
            "with another complete, standalone sentence that develops one of "
            "its themes, findings, or scope conditions. Do not continue its "
            "grammar, and do not restate the topic on its own.\n"
        )
    return block


def _closing_continuity_block(previous_tail: str) -> str:
    if not previous_tail:
        return ""
    return (
        "CONTINUITY CONTEXT ONLY. The excerpt below grants no citation or "
        "evidentiary authority. Do not copy its citations. Ground every closing "
        "claim only in the ALLOWED CITATIONS and CALL-LOCAL EVIDENCE that "
        "follow. Its sole purpose is to help the closing continue the prior "
        "section's theme.\n"
        f"Previous ending:\n...{previous_tail}\n"
    )


def _build_opening_prompt(topic: str, stance_summary: str, evidence: str, allowed_list: str):
    return f"""Literature review on: {topic}

{stance_summary}

ALLOWED CITATIONS:
{allowed_list}

Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Open the review. State the substantive question and what is at stake intellectually (not in NGO terms). Name the live disagreement: what positions exist, where they conflict, without listing them as a survey. 200-300 words. End with a complete standalone sentence whose substantive theme can carry into the next section.

{_OUTPUT_ONLY_DIRECTIVE}

Begin:"""


_DISPLAY_NAMED_AUTHOR_RE = re.compile(
    r"\b((?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"(?:\s+and\s+(?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"|\s+et\s+al\.)?)\s+\(\d{4}[a-z]?(?:,|\s)"
)


def _extract_named_surnames(chunk: str) -> list:
    if not chunk:
        return []
    seen = []
    for m in _DISPLAY_NAMED_AUTHOR_RE.finditer(chunk):
        label = m.group(1).strip()
        if label.lower() in {"the", "this", "that", "these", "those", "many", "some",
                             "such", "their", "they", "his", "her", "our", "your"}:
            continue
        if label not in seen:
            seen.append(label)
    for m in CITE_RE.finditer(chunk):
        did = m.group(1)
        label = _doc_id_to_author_label(did)
        label = re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()
        if label and label not in seen:
            seen.append(label)
    return seen


def _format_claims_so_far(section_claims) -> str:
    if not section_claims:
        return ""
    lines = [
        "CLAIMS ALREADY MADE IN PRIOR SECTIONS (do NOT repeat these openers or "
        "the same mechanism phrasing, and do NOT re-introduce an author already "
        "named in prose below with a fresh \"X argues / X shows...\" sentence — "
        "cite them parenthetically instead):"
    ]
    all_named: list = []
    for i, c in enumerate(section_claims, 1):
        mechs = "; ".join((c.get("mechanisms") or [])[:2]) or "(no mechanism recorded)"
        docs = ", ".join(str(d) for d in (c.get("docs") or [])[:4])
        stance = (c.get("stance") or "").upper()
        cluster = c.get("cluster") or ""
        lines.append(f"  Section {i} [{stance}/{cluster}]: {mechs} (citing {docs})")
        for n in (c.get("surnames_named") or []):
            if n not in all_named:
                all_named.append(n)
    if all_named:
        lines.append(f"  Already named in prose (cite parenthetically, do not "
                     f"re-introduce): {', '.join(all_named[:20])}")
    return "\n".join(lines)


_RELATION_OPENERS = {
    "same_as_topic_cause":
        "this stream develops the topic's claim using the same causal "
        "variable under different phrasing or as a more specific instance",
    "upstream_of_topic_cause":
        "this stream identifies an ANTECEDENT that flows INTO the topic's "
        "causal variable as a trigger; the stream's account complements "
        "rather than competes with the topic's claim",
    "downstream_of_topic_cause":
        "this stream studies a DOWNSTREAM consequence of the topic's causal "
        "variable; the topic's claim is taken as given and the stream adds "
        "what follows from it",
    "rival_to_topic_cause":
        "this stream proposes a RIVAL cause that does NOT operate through "
        "the topic's variable and offers a genuinely competing explanation",
    "scope_condition":
        "this stream identifies WHEN the topic's claim holds and when it "
        "does not — a contingency or scope restriction on the topic's account",
    "adjacent":
        "this stream addresses a related but distinct question and engages "
        "the topic only obliquely",
    "endorses_topic":
        "this stream AFFIRMS the topic's comparative judgment with "
        "convergent evidence",
    "reverses_topic":
        "this stream reaches the OPPOSITE comparative verdict and reads the "
        "evidence as supporting the inverse claim",
    "qualifies_topic":
        "this stream ACCEPTS the topic's comparison only under specific "
        "conditions and identifies when the comparison breaks down",
    "methodological_critique":
        "this stream challenges the WAY the comparison is constructed — "
        "the measurement, sample, or framing — without directly endorsing "
        "or reversing the topic's verdict",
    "confirms_description":
        "this stream's account is CONSISTENT with the topic's description "
        "and adds corroborating evidence for the same pattern",
    "contradicts_description":
        "this stream's account is INCOMPATIBLE with the topic's "
        "description and reports a different pattern in the same subject",
    "adds_nuance":
        "this stream ACCEPTS the topic's description but refines its scope, "
        "magnitude, or mechanism",
    "explains_outcome":
        "this stream offers an EXPLANATION of the outcome the topic asks "
        "about: it names a cause and the mechanism through which that cause "
        "produced the outcome. It is one of the answers the literature "
        "gives, to be set beside the others",
    "documents_outcome":
        "this stream ESTABLISHES the outcome itself (how large it is, since "
        "when, where and for whom) and proposes no cause for it; it fixes "
        "what the explanations have to account for",
    "disputes_outcome":
        "this stream DENIES that the outcome holds as the topic states it, "
        "or reports the opposite pattern; it questions the premise of the "
        "topic's question",
}


def _explanation_summary(outline_plan: dict, n_docs: int) -> str:
    outline_plan = outline_plan or {}
    clusters = {c.get("cluster_id"): c for c in outline_plan.get("clusters", []) or []}
    order = outline_plan.get("ordered_cluster_ids") or list(clusters.keys())
    explanations, documents, disputes = [], 0, 0
    for cid in order:
        c = clusters.get(cid)
        if not c:
            continue
        n = len(c.get("doc_ids", []) or [])
        relation = c.get("relation", "")
        if relation == "explains_outcome":
            label = (c.get("shared_cause") or c.get("shared_thread") or cid).strip()
            explanations.append(f"{label} ({n} source(s))")
        elif relation == "documents_outcome":
            documents += n
        elif relation == "disputes_outcome":
            disputes += n
    outcome = (outline_plan.get("topic_outcome") or "").strip() or "the outcome it names"
    parts = [f"Topic shape: explanatory. The topic asks what explains {outcome} "
             f"and names no cause."]
    if explanations:
        parts.append(f"Across {n_docs} sources the literature offers these "
                     f"explanations: " + "; ".join(explanations) + ".")
        parts.append("Set them side by side as answers to the same question. "
                     "Add no explanation that is absent from this list.")
    else:
        parts.append(f"No stream among the {n_docs} sources proposes a cause. "
                     f"Report what the sources establish about the outcome.")
    if documents:
        parts.append(f"{documents} source(s) establish the outcome itself "
                     f"(its size, timing or place).")
    if disputes:
        parts.append(f"{disputes} source(s) dispute that the outcome holds as "
                     f"the topic states it.")
    return " ".join(parts)


def _format_outline_block(cluster: dict, doc_to_evidence_ids) -> str:
    if not cluster or not isinstance(cluster, dict):
        return ""

    def _eids(doc_ids):
        out = []
        for did in doc_ids or []:
            ids = doc_to_evidence_ids.get(did) or []
            if ids:
                out.append(ids[0])
        return out

    eids = _eids(cluster.get("doc_ids", []))
    elaboration = (cluster.get("elaboration") or "").strip()
    disagreement = (cluster.get("internal_disagreement") or "").strip()
    if not elaboration:
        return ""

    lines = [
        "STREAM POSTURE (use this as the section's substantive claim about "
        "the world; do NOT print it verbatim, but anchor the section's "
        "argument to it):",
        f"  {elaboration}",
    ]
    if len(eids) >= 2:
        cite_block = "; ".join(f"[{e}]" for e in eids[:5])
        lines.append(
            f"Cite these stream members together once when stating the shared "
            f"posture: ({cite_block})"
        )
    if disagreement:
        lines.append(f"Internal disagreement to acknowledge in passing: {disagreement}")
    return "\n".join(lines)


def _build_stream_prompt(topic: str, cluster_label: str, evidence: str,
                         allowed_list: str, previous_tail: str,
                         outline_block: str = "",
                         claims_so_far: str = "",
                         relation: str = "",
                         topic_shape: str = "",
                         lead_surname: str = "",
                         forbidden_opener_surnames: list = None,
                         explanation: str = ""):
    outline_section = (outline_block + "\n\n") if outline_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    relation_gloss = _RELATION_OPENERS.get(relation or "", "")
    relation_line = (
        f"Structural relation of this stream to the topic ({relation or 'unspecified'}): "
        f"{relation_gloss}.\n"
        if relation_gloss else ""
    )
    if explanation and relation == "explains_outcome":
        relation_line += (
            f"This stream's explanation of the outcome: {explanation}. "
            f"Present it as one answer to the topic's question, with the "
            f"mechanism through which it works.\n"
        )
    forbid_line = ""
    if forbidden_opener_surnames:
        forbid_list = "; ".join(sorted(set(forbidden_opener_surnames)))
        forbid_line = (
            f"OPENER GUARD: Do NOT open this section with any of these author "
            f"surnames as the topic-sentence subject (they are the lead authors "
            f"of OTHER sections): {forbid_list}. They may still be cited inside "
            f"the section if their evidence is in your ALLOWED CITATIONS list, "
            f"but the OPENING SENTENCE must be led by a different surname.\n"
        )
    lead_line = ""
    if lead_surname:
        lead_line = (
            f"SECTION LEAD: Open with a sentence whose grammatical subject "
            f"names '{lead_surname}' (this cluster's lead author). The cite "
            f"that anchors the opener must resolve to {lead_surname}'s paper.\n"
        )
    return f"""Continue this literature review on: {topic}

{prev_block}
{prior_section}Stream: {cluster_label}.
{relation_line}{lead_line}{forbid_line}
ALLOWED CITATIONS:
{allowed_list}

{outline_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Present this stream of literature. Open with a direct, substantive claim grounded in the STREAM POSTURE above. Never name the stream's structural relation in the prose (no 'this stream supports/critiques/complicates'); express the relation by the shape of the argument and the sources you cite together. Cite the stream's lead and one supporting source in the same sentence when stating the shared claim. Develop the claim through one specific mechanism, condition, or pattern the sources jointly establish; trace one downstream implication that another source in the stream qualifies. 220-300 words. End with a complete standalone sentence whose substantive theme can carry into the next section.

{_OUTPUT_ONLY_DIRECTIVE}

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    prev_block = _closing_continuity_block(previous_tail)
    return f"""Close this literature review on: {topic}

{prev_block}
ALLOWED CITATIONS:
{allowed_list}

CALL-LOCAL EVIDENCE. Only these representative entries and the ALLOWED CITATIONS list above grant evidentiary authority:
{evidence}

{_PROSE_DIRECTIVE}

Write a synthesis closing grounded in the cited sources. Begin with the underlying point of agreement and ground it with an [E####] marker from at least one paper. Then name the precise remaining disagreement and ground it with an [E####] marker from a distinct paper, so the closing cites at least two documents. Finish by identifying the specific historical evidence or empirical comparison that settles the disagreement, drawing on a measurement type or case that the cited sources have already used.

STRICT RULES:
- Do NOT write methodology-stub prose: avoid "this would involve measuring", "future research should", "further work would assess", "would provide a nuanced understanding", "remains an open question", "remains a subject of debate".
- v13.1 FIX-E: also forbidden — "would be necessary", "would illuminate", "would clarify", "would resolve", "would shed light", "would inform", "would allow us to", "thorough comparison", "thorough examination". More generally: do NOT attach the modal "would" to a methodology verb such as comparison, examination, assessment, analysis, or investigation.
- Use present-tense indicative statements about what the cited sources SHOW, not conditional-tense statements about what future work WOULD show.
- Do NOT write "In conclusion" or "To summarize".
- Tie every claim to [E####] markers from the allowed list. The renderer supplies the correct author and page surface. Closing paragraphs without evidence markers are forbidden.
- Aim for roughly 250 words. This is a target, not a limit. Use the paragraphing that best serves the synthesis, and finish each paragraph with a complete standalone sentence.

{_OUTPUT_ONLY_DIRECTIVE}

Continue:"""


def _dump_writer_prompt(stage: str, system: str, user: str):
    target = os.environ.get("RRR_WRITER_PROMPT_DUMP_DIR")
    if not target and os.environ.get("RRR_DEBUG_WRITER_PROMPTS", "0") == "1":
        target = str(runs_path("prompts"))
    if not target:
        return
    try:
        os.makedirs(target, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", stage)
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
        return fn
    except Exception:
        return None


def _dump_writer_response(prompt_path, response: str = "", error=None) -> None:
    if not prompt_path:
        return
    try:
        response_path = os.path.splitext(prompt_path)[0] + "_response.txt"
        payload = response or ""
        if error is not None:
            payload += f"\n\n===== ERROR =====\n{error}\n"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass


def _ollama_chat(prompt: str, metrics=None, stage="writer"):
    import ollama
    import time
    _system = _writer_system_prompt()
    qwen_prose_transport = _uses_qwen_prose_transport(_MODEL)
    if qwen_prose_transport:
        _system += "\n\n" + _QWEN_PROSE_TRANSPORT_INSTRUCTION
    prompt_path = _dump_writer_prompt(stage, _system, prompt)
    start = time.perf_counter()
    chat_kwargs = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _system},
            {"role": "user", "content": prompt},
        ],
        "options": _DEFAULT_CHAT_OPTIONS,
        "keep_alive": _KEEP_ALIVE,
        "stream": False,
    }
    if qwen_prose_transport:
        chat_kwargs["format"] = _QWEN_PROSE_SCHEMA
    try:
        res = ollama.chat(**chat_kwargs)
    except Exception as e:
        _dump_writer_response(prompt_path, error=e)
        if metrics:
            metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                               success=False, duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), error=e)
        raise
    raw_out = (res.get("message", {}).get("content") or "").strip()
    try:
        out = (
            _decode_qwen_prose_transport(raw_out)
            if qwen_prose_transport
            else raw_out
        )
    except ValueError as exc:
        _dump_writer_response(prompt_path, response=raw_out, error=exc)
        if metrics:
            metrics.record_llm(
                stage,
                _MODEL,
                options=_DEFAULT_CHAT_OPTIONS,
                success=False,
                duration_s=time.perf_counter() - start,
                prompt_chars=len(prompt),
                response_chars=len(raw_out),
                error=exc,
            )
        raise
    _dump_writer_response(prompt_path, response=raw_out)
    if metrics:
        metrics.record_llm(stage, _MODEL, options=_DEFAULT_CHAT_OPTIONS,
                           duration_s=time.perf_counter() - start,
                           prompt_chars=len(prompt), response_chars=len(raw_out))
    return out


_FABRICATED_QUOTE_RE = re.compile(
    r'"([^"\u201c\u201d\n]{20,})"|\u201c([^"\u201c\u201d\n]{20,})\u201d'
)
_COVERAGE_QUOTE_INTRODUCER_RE = re.compile(r"\(\d{4}, p\.\d+\) reports?, $")


def _is_period_inside_paren(text: str, idx: int) -> bool:
    if idx < 0 or idx >= len(text):
        return False
    depth = 0
    i = idx - 1
    while i >= 0 and text[i] != "\n":
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                return True
            depth -= 1
        i -= 1
    return False


def _normalise_for_quote_match(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[\"'`“”‘’«»]", "", s)
    s = s.replace("…", " ")
    s = re.sub(r"\.{2,}", " ", s)
    s = re.sub(r"([a-z])-\s*([a-z])", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _quote_is_coverage_patch(full_text: str, q_start: int) -> bool:
    line_start = full_text.rfind("\n", 0, q_start) + 1
    sentence_start = line_start
    for idx in range(q_start - 1, line_start - 1, -1):
        if full_text[idx] in ".?!" and not _is_period_inside_paren(full_text, idx):
            sentence_start = idx + 1
            break
    sentence_head = full_text[sentence_start:q_start]
    if _COVERAGE_PATCH_SENTINEL.strip() in sentence_head:
        return True
    return bool(_COVERAGE_QUOTE_INTRODUCER_RE.search(sentence_head))


def _strip_fabricated_quotes(full_text: str, allowed_docs, metrics=None,
                              display_lookup=None):
    import time
    from rrr.paths import page_text_path

    stats = {
        "enabled": True,
        "checked_quotes": 0,
        "coverage_quotes_skipped": 0,
        "verified_real": 0,
        "fabricated_stripped": 0,
        "fabricated_kept_no_citation": 0,
        "fabricated_kept_doc_not_in_corpus": 0,
        "fabricated_kept_page_not_readable": 0,
        "fabrications": [],
        "duration_s": 0.0,
        "fallback_reason": "ok",
    }
    if os.environ.get("RRR_QUOTE_VERIFY", "1") != "1":
        stats["enabled"] = False
        stats["fallback_reason"] = "disabled"
        return full_text, stats
    if not full_text:
        stats["fallback_reason"] = "empty_text"
        return full_text, stats

    allowed_doc_set = set(allowed_docs or ())
    start = time.perf_counter()
    page_text_cache: dict = {}

    def _load_page_text(doc_id: str, page: int):
        key = (doc_id, page)
        if key in page_text_cache:
            return page_text_cache[key]
        try:
            p = page_text_path(doc_id, page)
            if p.is_file():
                txt = p.read_text(encoding="utf-8", errors="replace")
                page_text_cache[key] = txt
                return txt
        except Exception:
            pass
        page_text_cache[key] = None
        return None

    spans_to_strip = []
    spans_to_dequote = []
    for qm in _FABRICATED_QUOTE_RE.finditer(full_text):
        quote = qm.group(1) or qm.group(2)
        q_start, q_end = qm.span()
        if _quote_is_coverage_patch(full_text, q_start):
            stats["coverage_quotes_skipped"] += 1
            continue
        stats["checked_quotes"] += 1

        win_start = max(0, q_start - 200)
        win_end = min(len(full_text), q_end + 200)
        window = full_text[win_start:win_end]
        cites = list(parse_citations(window, display_lookup=display_lookup))
        cites = [c for c in cites if c["doc_id"] is not None]
        if not cites:
            stats["fabricated_kept_no_citation"] += 1
            continue

        q_center = (q_start + q_end) / 2 - win_start

        def _dist(c, _q=q_center):
            return abs((c["start"] + c["end"]) / 2 - _q)

        nearest = min(cites, key=_dist)
        doc_id = nearest["doc_id"]
        page = int(nearest["page"])

        if doc_id not in allowed_doc_set:
            stats["fabricated_kept_doc_not_in_corpus"] += 1
            continue

        page_txt = _load_page_text(doc_id, page)
        if not page_txt:
            stats["fabricated_kept_page_not_readable"] += 1
            continue

        norm_q = _normalise_for_quote_match(quote)
        norm_page = _normalise_for_quote_match(page_txt)
        if norm_q and norm_q in norm_page:
            stats["verified_real"] += 1
            continue

        sent_start = q_start
        while sent_start > 0:
            ch = full_text[sent_start - 1]
            if ch == "\n":
                break
            if ch in ".?!" and not _is_period_inside_paren(full_text, sent_start - 1):
                break
            sent_start -= 1
        while sent_start < q_start and full_text[sent_start] in " \t":
            sent_start += 1
        sent_end = q_end
        while sent_end < len(full_text):
            prev_ch = full_text[sent_end - 1]
            if prev_ch == "\n":
                break
            if prev_ch in ".?!" and not _is_period_inside_paren(full_text, sent_end - 1):
                break
            sent_end += 1
        sent_text = full_text[sent_start:sent_end]
        prev_text = full_text[max(0, sent_start - 200):sent_start]
        if _drop_would_strand_surname(
            prev_text, sent_text, allowed_doc_set, display_lookup=display_lookup,
        ):
            stats.setdefault("strand_guard_kept", 0)
            stats["strand_guard_kept"] += 1
            spans_to_dequote.append((q_start, q_end, quote, doc_id, page, sent_start, sent_end))
            continue
        spans_to_strip.append((sent_start, sent_end, quote, doc_id, page))

    edits = []
    seen_strips = set()
    for sent_start, sent_end, quote, doc_id, page in spans_to_strip:
        if (sent_start, sent_end) in seen_strips:
            continue
        seen_strips.add((sent_start, sent_end))
        edits.append((sent_start, sent_end, "sentence_stripped", quote, doc_id, page, sent_start, sent_end))
    for q_start, q_end, quote, doc_id, page, sent_start, sent_end in spans_to_dequote:
        if any(s <= q_start < e for s, e in seen_strips):
            continue
        edits.append((q_start, q_end, "quotation_marks_removed", quote, doc_id, page, sent_start, sent_end))
    if edits:
        edits.sort(key=lambda x: -x[0])
        out = full_text
        for start, end, action, quote, doc_id, page, sent_start, sent_end in edits:
            if action == "sentence_stripped":
                out = out[:start] + out[end:]
                stats["fabricated_stripped"] += 1
            else:
                out = out[:start] + out[start + 1:end - 1] + out[end:]
                stats.setdefault("strand_guard_dequoted", 0)
                stats["strand_guard_dequoted"] += 1
                seg_end = sent_end - 2
                seg = out[sent_start:seg_end]
                if seg.count('"') % 2 or seg.count("\u201c") != seg.count("\u201d"):
                    out = out[:sent_start] + re.sub(r'["\u201c\u201d]', "", seg) + out[seg_end:]
                    stats.setdefault("strand_guard_marks_cleared", 0)
                    stats["strand_guard_marks_cleared"] += 1
            stats["fabrications"].append({
                "quote": quote[:240],
                "doc_id": doc_id,
                "page": page,
                "action": action,
            })
        out = re.sub(r"  +", " ", out)
        out = re.sub(r"\s+\.", ".", out)
        out = re.sub(r"\n[ \t]+\n", "\n\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
    else:
        out = full_text

    stats["duration_s"] = round(time.perf_counter() - start, 3)
    return out, stats


def _first_paragraph_sentence_span(text: str, sentence_count: int):
    source = text or ""
    paragraph_break = _PARAGRAPH_BREAK_RE.search(source)
    first_paragraph_end = (
        paragraph_break.start() if paragraph_break else len(source)
    )
    first_paragraph = source[:first_paragraph_end]
    sentences = _split_sentences_for_cleanup(first_paragraph)
    selected = sentences[:max(1, sentence_count)]
    if not selected:
        return None

    cursor = 0
    first_start = None
    last_end = None
    for sentence in selected:
        needle = sentence.strip()
        start = first_paragraph.find(needle, cursor)
        if start < 0:
            return None
        if first_start is None:
            first_start = start
        last_end = start + len(needle)
        cursor = last_end
    return {
        "start": first_start,
        "end": last_end,
        "text": first_paragraph[first_start:last_end],
        "sentence_count": len(selected),
    }


def _replace_first_paragraph_opener(text: str, sentence_count: int,
                                    rewritten: str) -> tuple:
    span = _first_paragraph_sentence_span(text, sentence_count)
    if not span:
        return text, "opener_span_missing"
    candidate = (rewritten or "").strip()
    if not candidate or "\n" in candidate or "\r" in candidate:
        return text, "paragraph_injection"
    original_length = max(1, len(span["text"].strip()))
    if len(candidate) > int(original_length * 1.2) + 10:
        return text, "length_growth"
    candidate_sentences = _split_sentences_for_cleanup(candidate)
    if len(candidate_sentences) != span["sentence_count"]:
        return text, "sentence_count_mismatch"
    if any(
        not _ends_with_terminal_punctuation(sentence)
        for sentence in candidate_sentences
    ):
        return text, "incomplete_rewrite"
    rewritten_text = text[:span["start"]] + candidate + text[span["end"]:]
    if len(_prose_paragraphs(rewritten_text)) != len(_prose_paragraphs(text)):
        return text, "paragraph_count_mismatch"
    return rewritten_text, "ok"


def _apply_cross_section_stitch(chunks, topic: str, metrics=None):
    import time
    if not chunks or len(chunks) < 4:
        return chunks

    try:
        stitch_n = max(1, int(os.environ.get("RRR_WRITER_STITCH_SENTENCES", "1")))
    except ValueError:
        stitch_n = 1

    interior_indices = list(range(1, len(chunks) - 1))
    extracts = []
    for i in interior_indices:
        span = _first_paragraph_sentence_span(chunks[i], stitch_n)
        if not span:
            continue
        opener = span["text"].strip()
        prev_sents = _sentences_paragraph_local(chunks[i - 1])
        prev_tail = prev_sents[-1].strip() if prev_sents else ""
        if not opener:
            continue
        extracts.append({
            "index": i,
            "opener": opener,
            "opener_sent_count": span["sentence_count"],
            "prev_tail": prev_tail,
        })

    if len(extracts) < 2:
        return chunks

    opener_unit = "sentence" if stitch_n == 1 else f"first {stitch_n} sentences"
    parts = [
        f"Topic of the literature review: {topic}",
        "",
        f"Below are the OPENING {opener_unit} of several adjacent sections in "
        "the same review. Each currently restates the topic framing on its "
        "own, which reads as several mini-essays stitched together. Rewrite "
        f"each OPENING ({opener_unit}) as a complete standalone sentence "
        "that develops a theme, finding, or scope condition from the "
        "previous section. The previous section already ends with a complete "
        "sentence. Do not continue its grammar and do not re-introduce the topic.",
        "",
        "STRICT RULES:",
        "- Keep every citation token — 'Author (Year, p.N)', '(Author Year, p.N)', "
        "'(Doc_Year: p.N)' or '[E####]' — exactly UNCHANGED, character for character.",
        "- Do NOT add or remove any citation.",
        "- Do NOT exceed the original block length by more than ~20%.",
        "- Keep the original sentence count. Every rewritten sentence must "
        "end with terminal punctuation.",
        "- Return each rewritten opener as one line with no paragraph break.",
        f"- Return ONLY a single JSON object whose 'rewritten' field is the "
        f"{opener_unit} (joined into one string, original sentence-boundary "
        f"order preserved): "
        '{"openings": [{"index": <int>, "rewritten": "<text>"}, ...]}',
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
        prompt_path = _dump_writer_prompt("cross_section_stitch", "", prompt)
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 1500},
            keep_alive=_KEEP_ALIVE,
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        _dump_writer_response(prompt_path, response=raw)
        if metrics:
            metrics.record_llm("cross_section_stitch", _MODEL,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), response_chars=len(raw))
    except Exception as e:
        if "prompt_path" in locals():
            _dump_writer_response(prompt_path, error=e)
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
    skipped_topic_paraphrase = 0
    skipped_tail_echo = 0
    skipped_layout_change = 0
    skipped_incomplete = 0
    interior_set = set(interior_indices)
    prev_tail_by_index = {e["index"]: e["prev_tail"] for e in extracts}

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "for", "on", "with", "as", "by", "at", "from",
        "this", "that", "these", "those", "and", "or", "but", "if", "then",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "how", "their", "there", "here", "it", "its", "such", "any", "all",
        "no", "not", "do", "does", "did", "have", "has", "had", "can", "could",
        "may", "might", "will", "would", "shall", "should", "must",
        "more", "most", "less", "least", "very", "much", "many", "some",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "among", "across", "over", "under",
        "we", "you", "they", "i", "me", "us", "them", "he", "she", "him", "her",
        "our", "your", "my",
    }

    def _content_tokens(s: str) -> set:
        toks = re.findall(r"[A-Za-z][A-Za-z\-]+", (s or "").lower())
        return {t for t in toks if t not in _STOPWORDS and len(t) >= 3}

    topic_tokens = _content_tokens(topic)
    paraphrase_threshold = max(4, int(0.66 * len(topic_tokens)))

    opener_sent_count_by_index = {e["index"]: e["opener_sent_count"] for e in extracts}
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
        span = _first_paragraph_sentence_span(
            new_chunks[idx], opener_sent_count_by_index.get(idx, 1),
        )
        if not span:
            skipped_layout_change += 1
            continue
        n_open = opener_sent_count_by_index.get(idx, 1)
        orig_opener_block = span["text"].strip()
        orig_cites = _citation_fingerprints(orig_opener_block)
        new_cites = _citation_fingerprints(rewritten)
        if orig_cites != new_cites:
            skipped_citation += 1
            continue
        rewrite_tokens = _content_tokens(rewritten)
        overlap = len(rewrite_tokens & topic_tokens)
        if topic_tokens and overlap >= paraphrase_threshold:
            skipped_topic_paraphrase += 1
            continue
        prev_tail = prev_tail_by_index.get(idx, "")
        if prev_tail:
            prev_tail_tokens = _content_tokens(prev_tail)
            tail_overlap = len(rewrite_tokens & prev_tail_tokens)
            if (tail_overlap >= 6) or (
                tail_overlap >= 4 and rewrite_tokens and rewrite_tokens.issubset(prev_tail_tokens)
            ):
                skipped_tail_echo += 1
                continue
        rewritten_chunk, layout_reason = _replace_first_paragraph_opener(
            new_chunks[idx], n_open, rewritten,
        )
        if layout_reason != "ok":
            if layout_reason == "incomplete_rewrite":
                skipped_incomplete += 1
            else:
                skipped_layout_change += 1
            continue
        new_chunks[idx] = rewritten_chunk
        applied += 1

    if metrics:
        metrics.set("writer_stitch_applied", applied)
        metrics.set("writer_stitch_skipped_citation_change", skipped_citation)
        metrics.set("writer_stitch_skipped_empty", skipped_empty)
        metrics.set("writer_stitch_skipped_topic_paraphrase", skipped_topic_paraphrase)
        metrics.set("writer_stitch_skipped_tail_echo", skipped_tail_echo)
        metrics.set("writer_stitch_skipped_layout_change", skipped_layout_change)
        metrics.set("writer_stitch_skipped_incomplete", skipped_incomplete)
        metrics.set(
            "writer_stitch_paragraph_counts_preserved",
            all(
                len(_prose_paragraphs(before)) == len(_prose_paragraphs(after))
                for before, after in zip(chunks, new_chunks)
            ),
        )
    if applied or skipped_topic_paraphrase or skipped_tail_echo:
        print(f"[Writer] Cross-section stitch: rewrote {applied} section "
              f"opener(s) (skipped {skipped_citation} citation drift, "
              f"{skipped_topic_paraphrase} topic paraphrase, "
              f"{skipped_tail_echo} tail echo)")
    return new_chunks


def _collect_cited_docs(text: str, allowed_docs, author_year_to_docid):
    return _shared_collect_cited_docs(text, allowed_docs, author_year_to_docid)


def compose_from_ledger(ledger_path=None, metrics=None):
    ledger_path = ledger_path or str(runs_path("review_ledger.json"))
    if not os.path.isfile(ledger_path):
        raise SystemExit(f"Ledger not found: {ledger_path}")

    with open(ledger_path, encoding="utf-8") as f:
        data = json.load(f)

    topic_display = data.get("topic", "(no topic)")
    topic = (data.get("plan", {}) or {}).get("topic_question") or topic_display
    topic_dimensions = (data.get("plan", {}) or {}).get("topic_dimensions") or []
    if topic != topic_display:
        print(f"[Writer] using reformulated topic: '{topic}'")
        if topic_dimensions:
            print(f"[Writer] topic dimensions: {topic_dimensions}")
    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise SystemExit("Ledger empty or malformed (no docs).")

    pdf_paths_by_docid = data.get("pdf_paths_by_docid", {}) or {}
    pdf_page_offsets = data.get("pdf_page_offsets", {}) or {}
    dois_by_docid = data.get("dois_by_docid", {}) or {}

    ledger_allowed_pairs, ledger_allowed_docs, _ = _build_allowed_citations(docs)
    if not ledger_allowed_pairs:
        raise SystemExit("No allowed citations found in ledger.")

    outline_plan = data.get("outline_plan") or {}
    if not outline_plan or not outline_plan.get("clusters"):
        raise SystemExit("Ledger missing outline_plan or outline produced no clusters.")
    topic_shape = outline_plan.get("topic_shape") or "causal"
    clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
    ordered_cluster_ids = (
        outline_plan.get("ordered_cluster_ids")
        or list(clusters_by_id.keys())
    )

    docs_by_id = {d.get("doc_id"): d for d in docs}

    relation_dist = outline_plan.get("relation_distribution") or {}
    relation_summary_parts = []
    for relation in sorted(relation_dist.keys()):
        n = relation_dist[relation]
        gloss = _RELATION_OPENERS.get(relation, relation.replace("_", " "))
        relation_summary_parts.append(
            f"{n} source(s) in streams whose relation to the topic is: {gloss}"
        )
    stance_summary = (
        f"Topic shape: {topic_shape}. Of {len(docs)} sources, " +
        "; ".join(relation_summary_parts) + "." if relation_summary_parts
        else f"Topic shape: {topic_shape}. {len(docs)} sources across "
             f"{len(clusters_by_id)} streams."
    )
    if topic_shape == "explanatory":
        stance_summary = _explanation_summary(outline_plan, len(docs))
    print(f"[Writer] topic_shape={topic_shape} clusters={len(clusters_by_id)} "
          f"relations={relation_dist}")

    from functools import partial
    cluster_lead_surnames: dict = {}
    for cid in ordered_cluster_ids:
        c = clusters_by_id.get(cid)
        if not c:
            continue
        lead_did = (c.get("lead_doc_id") or "").strip()
        if not lead_did:
            doc_ids = c.get("doc_ids") or []
            lead_did = doc_ids[0] if doc_ids else ""
        if lead_did:
            label = _doc_id_to_author_label(lead_did)
            cluster_lead_surnames[cid] = _author_surnames_only(label)
        else:
            cluster_lead_surnames[cid] = ""
    all_lead_surnames = {s for s in cluster_lead_surnames.values() if s}

    chunk_plan = []
    for cid in ordered_cluster_ids:
        c = clusters_by_id.get(cid)
        if not c or not c.get("doc_ids"):
            continue
        cluster_docs = [docs_by_id[did] for did in c["doc_ids"] if did in docs_by_id]
        if not cluster_docs:
            continue
        this_lead = cluster_lead_surnames.get(cid, "")
        forbidden = sorted(s for s in all_lead_surnames if s and s != this_lead)
        builder = partial(_build_stream_prompt,
                          relation=c.get("relation", ""),
                          topic_shape=topic_shape,
                          lead_surname=this_lead,
                          forbidden_opener_surnames=forbidden,
                          explanation=((c.get("shared_cause") or "").strip()
                                       if topic_shape == "explanatory" else ""))
        chunk_plan.append((
            c.get("relation", "stream"),
            c.get("shared_thread") or cid,
            cluster_docs,
            builder,
            c,
        ))

    if not chunk_plan:
        raise SystemExit("No clusters produced any documents to write about.")

    print(f"[Writer] Generating {len(chunk_plan) + 2} sections "
          f"(opening + {len(chunk_plan)} stream sections + closing)...")
    if metrics:
        metrics.set("writer_chunk_plan", {
            "topic_shape": topic_shape,
            "clusters_total": len(chunk_plan),
            "relation_distribution": dict(relation_dist),
        })

    chunks = []
    section_claims = []
    total_removed_citations = 0
    total_style_removed = 0
    total_coverage_fallbacks = 0
    total_evidence_id_renders = 0
    total_double_paren_collapsed = 0
    total_author_led_openings = 0
    total_bracket_id_rewrites = 0
    total_unknown_eids = 0
    total_attribution_mismatches = 0
    total_unsupported_year_only_citations = 0
    total_attribution_retries = 0
    total_structure_retries = 0
    total_closing_model_retries = 0
    unknown_eid_snippets: list = []
    attribution_mismatch_snippets: list = []
    unsupported_year_only_snippets: list = []
    structure_failure_records: list = []
    author_year_repair_stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "replacements_proposed": 0,
        "replacements_accepted": 0,
        "distinct_docs_rescued": 0,
        "collision_skips": 0,
        "final_assembly_replacements": 0,
    }
    author_year_repair_records: list = []
    section_coverage = []
    call_contracts = []
    call_allowed_pairs = set()
    call_allowed_docs = set()
    call_evidence_id_map = {}

    def register_call(stage, call_docs, allowed_list, prompt):
        contract = _build_call_contract(stage, call_docs, allowed_list, prompt)
        contract["call_index"] = len(call_contracts)
        call_contracts.append(contract)
        pairs, doc_ids, _ = _build_allowed_citations(call_docs)
        call_allowed_pairs.update(pairs)
        call_allowed_docs.update(doc_ids)
        call_evidence_id_map.update(_build_evidence_id_map(call_docs))
        ensure_dir(str(runs_path()))
        with open(runs_path("writer_call_contracts.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "corrected-writer-v17",
                    "quotes_per_doc": _writer_quotes_per_doc(),
                    "calls": call_contracts,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def register_retry_call(stage, call_docs, prompt):
        _, _, pages_by_doc = _build_allowed_citations(call_docs)
        retry_allowed_list = _list_allowed_citations(call_docs, pages_by_doc)
        register_call(stage, call_docs, retry_allowed_list, prompt)

    chunk_display_lookup = _build_display_lookup(ledger_allowed_docs)

    def _chunk_display_to_canonical(text):
        def repl(m):
            label = m.group(1).strip().lower()
            year = m.group(2)
            page = int(m.group(3))
            did = chunk_display_lookup.get((label, year))
            if did:
                return render_citation_canonical(did, page)
            return m.group(0)
        text = DISPLAY_CITE_RE.sub(repl, text)
        text = DISPLAY_PAREN_CITE_RE.sub(repl, text)
        return text

    def postprocess_chunk(chunk, chunk_docs, stage):
        nonlocal total_removed_citations, total_style_removed, total_evidence_id_renders
        nonlocal total_double_paren_collapsed, total_author_led_openings
        nonlocal total_bracket_id_rewrites
        nonlocal total_unknown_eids, total_attribution_mismatches
        nonlocal total_unsupported_year_only_citations

        chunk = _strip_wrapping(chunk)
        chunk, author_year_repairs, author_year_collision_skips = (
            _repair_allowed_author_year_mentions(chunk, chunk_docs, stage=stage)
        )
        chunk_evidence_id_map = _build_evidence_id_map(chunk_docs)
        chunk, render_stats = _render_evidence_id_citations(
            chunk, chunk_evidence_id_map
        )
        render_stats["author_year_repair_records"] = author_year_repairs
        render_stats["author_year_collision_skips"] = author_year_collision_skips
        total_evidence_id_renders += render_stats["replacements"]
        total_unknown_eids += render_stats["unknown_eids"]
        unknown_eid_snippets.extend(render_stats["unknown_eid_snippets"])
        total_attribution_mismatches += render_stats["attribution_mismatches"]
        attribution_mismatch_snippets.extend(render_stats["attribution_mismatch_snippets"])

        chunk, bracket_rewrites = _render_bracketed_doc_ids(
            chunk,
            {ev["doc_id"] for ev in chunk_evidence_id_map.values()},
            {
                did: sorted(
                    eid
                    for eid, ev in chunk_evidence_id_map.items()
                    if ev["doc_id"] == did
                )
                for did in {ev["doc_id"] for ev in chunk_evidence_id_map.values()}
            },
        )
        total_bracket_id_rewrites += bracket_rewrites

        chunk, dp_collapsed = _collapse_double_parens(chunk)
        total_double_paren_collapsed += dp_collapsed
        unsupported_year_only = _find_unsupported_year_only_citations(
            chunk, chunk_docs,
        )
        render_stats["unsupported_year_only_citations"] = len(
            unsupported_year_only
        )
        render_stats["unsupported_year_only_citation_records"] = (
            unsupported_year_only
        )
        total_unsupported_year_only_citations += len(unsupported_year_only)
        unsupported_year_only_snippets.extend(
            record["snippet"] for record in unsupported_year_only
        )
        total_author_led_openings += _count_author_led_openings(chunk)

        chunk = _strip_orphaned_citations(chunk)
        chunk = _strip_references_section(chunk)
        chunk = _strip_continuation_markers(chunk)
        chunk = _strip_conclusion(chunk)
        if _writer_enforcement_enabled():
            chunk_allowed_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
            chunk_display_lookup_local = _build_display_lookup(chunk_allowed_docs)
            chunk, removed = _remove_invalid_citations(
                chunk,
                chunk_allowed_docs,
                allowed_pairs=chunk_allowed_pairs,
                display_lookup=chunk_display_lookup_local,
            )
            total_removed_citations += len(removed)

        style_removed = []

        return chunk, 0, 0, 0, len(style_removed), render_stats

    def record_author_year_repair_outcome(render_stats, *, accepted: bool):
        records = list(render_stats.get("author_year_repair_records", []) or [])
        collision_skips = int(
            render_stats.get("author_year_collision_skips", 0) or 0
        )
        if not records and not collision_skips:
            return
        author_year_repair_stats["attempts"] += 1
        author_year_repair_stats["collision_skips"] += collision_skips
        author_year_repair_stats["replacements_proposed"] += len(records)
        if accepted:
            author_year_repair_stats["accepted"] += 1
            author_year_repair_stats["replacements_accepted"] += len(records)
            author_year_repair_stats["distinct_docs_rescued"] += len(
                {record["doc_id"] for record in records}
            )
        else:
            author_year_repair_stats["rejected"] += 1
        for record in records:
            author_year_repair_records.append({
                **record,
                "outcome": "accepted" if accepted else "rejected",
            })

    def finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage):
        nonlocal total_coverage_fallbacks, total_attribution_retries
        nonlocal total_structure_retries, total_closing_model_retries
        local_failure_records = []
        chunk, repairs, placeholders, ajr, style_removed, render_stats = postprocess_chunk(
            raw, chunk_docs, stage,
        )
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        structural_only = not _writer_enforcement_enabled()

        chunk_allowed_pairs, chunk_allowed_doc_ids, _ = _build_allowed_citations(chunk_docs)
        chunk_fallback_dl = _build_display_lookup(chunk_allowed_doc_ids)

        def maybe_add_nonclosing_fallback(state):
            nonlocal total_coverage_fallbacks
            current_audit = state["audit"]
            if not (
                section_kind != "closing"
                and not structural_only
                and current_audit["structure_ok"]
                and not current_audit["coverage_ok"]
            ):
                return state
            patched, fallback_count = _append_coverage_fallback(
                state["chunk"], chunk_docs,
                current_audit["required_cited_docs"],
                allowed_pairs=chunk_allowed_pairs,
                display_lookup=chunk_fallback_dl,
            )
            if fallback_count:
                total_coverage_fallbacks += fallback_count
                if metrics:
                    metrics.inc("writer_section_coverage_fallbacks", fallback_count)
                state = dict(state)
                state["chunk"] = patched
                state["audit"] = _audit_section_coverage(
                    patched, chunk_docs, section_kind,
                )
            return state

        def record_failed_candidate(state, phase, attempt):
            record_author_year_repair_outcome(
                state["render_stats"], accepted=False,
            )
            current_audit = state["audit"]
            current_render = state["render_stats"]
            record = {
                "stage": stage,
                "phase": phase,
                "attempt": attempt,
                "section": section_kind,
                "word_count": current_audit["word_count"],
                "paragraph_count": current_audit["paragraph_count"],
                "closing_paragraph_count_ok": current_audit[
                    "closing_paragraph_count_ok"
                ],
                "cited_doc_count": current_audit["cited_doc_count"],
                "required_cited_docs": current_audit["required_cited_docs"],
                "incomplete_paragraph_count": current_audit[
                    "incomplete_paragraph_count"
                ],
                "closing_word_floor_ok": current_audit[
                    "closing_word_floor_ok"
                ],
                "closing_word_ceiling_ok": current_audit[
                    "closing_word_ceiling_ok"
                ],
                "uncited_closing_paragraphs": current_audit[
                    "uncited_closing_paragraphs"
                ],
                "unknown_eids": current_render["unknown_eids"],
                "attribution_mismatches": current_render[
                    "attribution_mismatches"
                ],
                "unsupported_year_only_citations": current_render.get(
                    "unsupported_year_only_citations", 0,
                ),
                "faults": _section_quality_faults(
                    current_audit, current_render, section_kind,
                    structural_only,
                ),
            }
            local_failure_records.append(record)
            structure_failure_records.append(record)

        state = maybe_add_nonclosing_fallback({
            "chunk": chunk,
            "repairs": repairs,
            "placeholders": placeholders,
            "ajr": ajr,
            "style_removed": style_removed,
            "audit": audit,
            "render_stats": render_stats,
        })
        if not _section_candidate_passes(
            state["audit"], state["render_stats"], section_kind,
            structural_only,
        ):
            record_failed_candidate(state, "generated", 0)

        def retry_candidate(prior_state, attempt):
            nonlocal total_attribution_retries, total_structure_retries
            nonlocal total_closing_model_retries
            retry_reason = _section_retry_reason(
                prior_state["audit"], prior_state["render_stats"], section_kind,
                structural_only,
            )
            retry_stage = _section_retry_stage(
                stage, retry_reason, attempt, section_kind,
            )
            if section_kind == "closing":
                total_closing_model_retries += 1
                if metrics:
                    metrics.inc("writer_closing_model_retries")
            if retry_reason == "structure":
                total_structure_retries += 1
                if metrics:
                    metrics.inc("writer_structure_retries")
            elif retry_reason == "attribution":
                total_attribution_retries += 1
                if metrics:
                    metrics.inc("writer_attribution_retries")
            elif retry_reason == "evidence":
                if metrics:
                    metrics.inc("writer_evidence_integrity_retries")
            elif metrics:
                metrics.inc("writer_section_coverage_retries")

            retry_prompt = _coverage_retry_prompt(
                prompt,
                prior_state["chunk"],
                prior_state["audit"]["required_cited_docs"],
                section_kind=section_kind,
                retry_reason=retry_reason,
                faults=_section_quality_faults(
                    prior_state["audit"],
                    prior_state["render_stats"],
                    section_kind,
                    structural_only,
                ),
            )
            register_retry_call(retry_stage, chunk_docs, retry_prompt)
            raw_retry = _ollama_chat(
                retry_prompt, metrics=metrics, stage=retry_stage,
            )
            (
                retry_chunk, repairs2, placeholders2, ajr2, style_removed2,
                retry_render_stats,
            ) = postprocess_chunk(raw_retry, chunk_docs, retry_stage)
            retry_state = {
                "chunk": retry_chunk,
                "repairs": prior_state["repairs"] + repairs2,
                "placeholders": prior_state["placeholders"] + placeholders2,
                "ajr": prior_state["ajr"] + ajr2,
                "style_removed": prior_state["style_removed"] + style_removed2,
                "audit": _audit_section_coverage(
                    retry_chunk, chunk_docs, section_kind,
                ),
                "render_stats": retry_render_stats,
            }
            retry_state = maybe_add_nonclosing_fallback(retry_state)
            nonclosing_resolution = None
            if section_kind != "closing" and not structural_only:
                retry_state, nonclosing_resolution = _resolve_nonclosing_retry(
                    prior_state, retry_state, retry_reason, section_kind,
                )

            if (
                metrics and retry_reason == "attribution"
                and (
                    nonclosing_resolution == "retry_accepted"
                    or _section_candidate_passes(
                        retry_state["audit"],
                        retry_state["render_stats"],
                        section_kind,
                        structural_only,
                    )
                )
            ):
                metrics.inc("writer_attribution_retry_accepted")
            elif metrics and retry_reason == "attribution":
                metrics.inc("writer_attribution_retry_rejected")
            if not (
                retry_state.get("_nonclosing_compat_accepted", False)
                or _section_candidate_passes(
                    retry_state["audit"],
                    retry_state["render_stats"],
                    section_kind,
                    structural_only,
                )
            ):
                record_failed_candidate(retry_state, "after_retry", attempt)
            return retry_state

        state, retries_used, accepted = _run_bounded_section_retries(
            state, section_kind, retry_candidate, structural_only,
        )
        if not accepted:
            failure_record = {
                "schema_version": "v15.18",
                "error": "section_quality_validation_failed",
                "stage": stage,
                "section": section_kind,
                "model_retries": retries_used,
                "audit": state["audit"],
                "render_stats": state["render_stats"],
                "candidate_failures": local_failure_records,
                "faults": _section_quality_faults(
                    state["audit"], state["render_stats"], section_kind,
                    structural_only,
                ),
            }
            try:
                ensure_dir(str(runs_path()))
                with open(
                    runs_path("writer_section_failure.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(failure_record, f, indent=2, ensure_ascii=False)
            except Exception as artifact_error:
                print(
                    "[Writer] Could not write writer_section_failure.json: "
                    f"{artifact_error}"
                )
            raise ValueError(
                f"section quality failed for {section_kind}: "
                f"coverage={state['audit']['cited_doc_count']}/"
                f"{state['audit']['required_cited_docs']}, "
                f"paragraphs={state['audit']['paragraph_count']}, "
                f"incomplete_paragraphs="
                f"{state['audit']['incomplete_paragraph_count']}, "
                f"unknown_eids={state['render_stats']['unknown_eids']}, "
                f"attribution_mismatches="
                f"{state['render_stats']['attribution_mismatches']}, "
                f"words={state['audit']['word_count']}"
            )
        record_author_year_repair_outcome(
            state["render_stats"], accepted=True,
        )
        return (
            state["chunk"], state["repairs"], state["placeholders"],
            state["ajr"], state["style_removed"], state["audit"],
        )

    def generate_covered_chunk(prompt, chunk_docs, section_kind, stage):
        raw = _ollama_chat(prompt, metrics=metrics, stage=stage)
        return finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage)

    opening_docs_unbounded = []
    for relation, cluster_label, cluster_docs, _builder, _block in chunk_plan:
        opening_docs_unbounded.extend(
            sorted(cluster_docs, key=_score_doc, reverse=True)[:2]
        )
    opening_docs_unbounded = sorted(
        opening_docs_unbounded, key=_score_doc, reverse=True
    )[:6]
    opening_docs = _select_call_evidence(opening_docs_unbounded)

    _, _, opening_pages_by_doc = _build_allowed_citations(opening_docs)
    allowed_list = _list_allowed_citations(opening_docs, opening_pages_by_doc)
    evidence = "\n\n".join(_format_doc_entry(d) for d in opening_docs)

    prompt = _build_opening_prompt(topic, stance_summary, evidence, allowed_list)
    register_call("writer_opening", opening_docs, allowed_list, prompt)

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
        raise RuntimeError(
            "writer opening failed structural or evidence validation"
        ) from e

    stance_jobs = []
    parallel_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    for i, (stance, cluster, cluster_docs, prompt_builder, cluster_plan) in enumerate(chunk_plan):
        cluster_docs_sorted = _select_call_evidence(
            sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
        )
        _, _, cluster_pages_by_doc = _build_allowed_citations(cluster_docs_sorted)
        allowed_list = _list_allowed_citations(
            cluster_docs_sorted, cluster_pages_by_doc
        )
        evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
        synthesis_block = _format_outline_block(
            cluster_plan, _group_evidence_ids_by_doc(cluster_docs_sorted)
        )
        prompt = prompt_builder(topic, cluster, evidence, allowed_list, parallel_tail,
                                synthesis_block)
        job = {
            "index": i,
            "stance": stance,
            "cluster": cluster,
            "docs": cluster_docs_sorted,
            "prompt": prompt,
            "stage": f"writer_{stance}",
            "allowed_list": allowed_list,
        }
        stance_jobs.append(job)

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
            "surnames_named": _extract_named_surnames(chunk),
        })
        chunks.append(chunk)

    def log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit):
        del repairs, placeholders, ajr
        notes = []
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
        for job in stance_jobs:
            register_call(
                job["stage"], job["docs"], job["allowed_list"], job["prompt"]
            )
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
                raise RuntimeError(
                    f"writer section failed for {job['stance']}/{job['cluster']}"
                ) from e
    else:
        print(f"[Writer] Sequential stance chunks: claims-so-far context enabled")
        for i, (stance, cluster, cluster_docs, prompt_builder, cluster_plan) in enumerate(chunk_plan):
            cluster_docs_sorted = _select_call_evidence(
                sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
            )
            _, _, cluster_pages_by_doc = _build_allowed_citations(cluster_docs_sorted)
            allowed_list = _list_allowed_citations(
                cluster_docs_sorted, cluster_pages_by_doc
            )
            evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
            synthesis_block = _format_outline_block(
                cluster_plan, _group_evidence_ids_by_doc(cluster_docs_sorted)
            )
            previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
            claims_so_far = _format_claims_so_far(section_claims)
            prompt = prompt_builder(topic, cluster, evidence, allowed_list, previous_tail,
                                    synthesis_block, claims_so_far)
            job = {
                "index": i,
                "stance": stance,
                "cluster": cluster,
                "docs": cluster_docs_sorted,
                "prompt": prompt,
                "stage": f"writer_{stance}",
            }
            register_call(job["stage"], job["docs"], allowed_list, prompt)
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
                raise RuntimeError(
                    f"writer section failed for {job['stance']}/{job['cluster']}"
                ) from e

    used_doc_ids = []
    for claim in section_claims:
        for did in claim.get("docs", []):
            if did and did not in used_doc_ids:
                used_doc_ids.append(did)
    closing_docs_unbounded = [
        d for d in docs if d.get("doc_id") in set(used_doc_ids)
    ]
    closing_seen = {
        d.get("doc_id") for d in closing_docs_unbounded if d.get("doc_id")
    }
    if len(closing_seen) < 2:
        for candidate in sorted(docs, key=_score_doc, reverse=True):
            candidate_id = candidate.get("doc_id")
            if not candidate_id or candidate_id in closing_seen:
                continue
            closing_docs_unbounded.append(candidate)
            closing_seen.add(candidate_id)
            if len(closing_seen) >= 2:
                break
    closing_docs = _select_call_evidence(
        sorted(closing_docs_unbounded, key=_score_doc, reverse=True)[:6]
    )

    _, _, closing_pages_by_doc = _build_allowed_citations(closing_docs)
    allowed_list = _list_allowed_citations(closing_docs, closing_pages_by_doc)
    if not allowed_list:
        allowed_list = "(No call-local citations are available.)"
    evidence = "\n\n".join(_format_doc_entry(d) for d in closing_docs)

    previous_tail = chunks[-1][-_TAIL_CHARS:] if chunks else ""
    prompt = _build_closing_prompt(topic, evidence, allowed_list, previous_tail)
    register_call("writer_closing", closing_docs, allowed_list, prompt)

    closing_generated_audit = None
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
        closing_generated_audit = dict(audit)
        chunks.append(chunk)
        if metrics:
            metrics.inc("writer_sections_succeeded")
    except Exception as e:
        print(f"[Writer] Closing failed: {e}")
        if metrics:
            metrics.inc("writer_sections_failed")
        raise RuntimeError(
            "writer closing failed structural or evidence validation"
        ) from e

    expected_chunk_count = len(chunk_plan) + 2
    if len(chunks) != expected_chunk_count:
        raise RuntimeError(
            "writer section count mismatch: "
            f"expected {expected_chunk_count}, found {len(chunks)}"
        )

    if metrics:
        with metrics.stage("writer_stitch"):
            chunks = _apply_cross_section_stitch(chunks, topic, metrics=metrics)
    else:
        chunks = _apply_cross_section_stitch(chunks, topic, metrics=None)

    if closing_generated_audit is None or not chunks:
        raise RuntimeError("writer lost closing section identity before assembly")
    chunks[-1] = f"{_CLOSING_START_MARKER} {chunks[-1].lstrip()}"

    if not chunks or not any((c or "").strip() for c in chunks):
        raise RuntimeError(
            "writer produced no usable sections (all chunk generations failed); "
            "refusing to write an empty review_composed.md"
        )

    full_text = "\n\n".join(chunks)
    full_text, final_render_stats = _render_evidence_id_citations(full_text, {})
    total_evidence_id_renders += final_render_stats["replacements"]
    total_unknown_eids += final_render_stats["unknown_eids"]
    unknown_eid_snippets.extend(final_render_stats["unknown_eid_snippets"])
    total_attribution_mismatches += final_render_stats["attribution_mismatches"]
    attribution_mismatch_snippets.extend(final_render_stats["attribution_mismatch_snippets"])

    full_text, final_bracket_rewrites = _render_bracketed_doc_ids(
        full_text, call_allowed_docs, call_evidence_id_map,
    )
    total_bracket_id_rewrites += final_bracket_rewrites

    full_text, leaked_docid_normalizations = _normalize_leaked_doc_id_citations(
        full_text, call_allowed_docs,
    )
    if metrics:
        metrics.set(
            "writer_leaked_docid_normalizations", leaked_docid_normalizations,
        )

    allowed_pairs = call_allowed_pairs
    allowed_docs = call_allowed_docs
    author_year_to_docid = _build_author_year_lookup(allowed_docs)
    display_lookup = _build_display_lookup(allowed_docs)

    if metrics:
        with metrics.stage("writer_quote_verify"):
            full_text, quote_verify_stats = _strip_fabricated_quotes(
                full_text, allowed_docs, metrics=metrics,
                display_lookup=display_lookup,
            )
    else:
        full_text, quote_verify_stats = _strip_fabricated_quotes(
            full_text, allowed_docs, metrics=None,
            display_lookup=display_lookup,
        )
    if metrics:
        metrics.set("writer_quote_verification", quote_verify_stats)
        if quote_verify_stats.get("fabricated_stripped"):
            metrics.inc(
                "writer_fabricated_quotes_stripped",
                int(quote_verify_stats["fabricated_stripped"]),
            )
    if quote_verify_stats.get("fabricated_stripped"):
        print(
            f"[Writer] Quote verifier: {quote_verify_stats['fabricated_stripped']} "
            f"fabricated quote(s) stripped from "
            f"{quote_verify_stats['checked_quotes']} checked "
            f"(verified_real={quote_verify_stats['verified_real']}, "
            f"kept_no_cite={quote_verify_stats['fabricated_kept_no_citation']}, "
            f"kept_doc_not_in_corpus={quote_verify_stats['fabricated_kept_doc_not_in_corpus']}, "
            f"kept_page_unreadable={quote_verify_stats['fabricated_kept_page_not_readable']})"
        )
        for fab in quote_verify_stats["fabrications"][:5]:
            print(
                f"         - ({fab['doc_id']}: p.{fab['page']}) "
                f"\"{fab['quote'][:120]}...\""
            )
    elif quote_verify_stats.get("checked_quotes"):
        print(
            f"[Writer] Quote verifier: {quote_verify_stats['verified_real']}/"
            f"{quote_verify_stats['checked_quotes']} quotes verified as real "
            f"(0 fabrications)"
        )

    if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        removed_citations = []
        print("[Writer] Citation removal skipped because RRR_BYPASS_VALIDATION=1")
        if metrics:
            metrics.set("writer_bypass_validation", True)
    else:
        full_text, removed_citations = _remove_invalid_citations(
            full_text, allowed_docs, allowed_pairs=allowed_pairs,
            display_lookup=display_lookup,
        )
        total_removed_citations += len(removed_citations)
    if removed_citations:
        print(f"[Writer] Removed {len(removed_citations)} invalid citation(s):")
        for r in removed_citations:
            if r.get("page"):
                print(f"         - {r['doc_id']}: p.{r['page']} ({r.get('reason', 'invalid')})")
            else:
                print(f"         - {r['doc_id']} ({r.get('reason', 'invalid')})")

    final_style_removed = []

    closing_words_before_redundancy = _count_words(
        _split_marked_closing(full_text)["closing"]
    )
    full_text, redundancy_drops, _closing_words_after_redundancy = (
        _drop_body_redundancy_preserving_closing(
            full_text,
            min_token_overlap=4,
            display_lookup=display_lookup,
            allowed_docs=allowed_docs,
        )
    )
    if redundancy_drops:
        print(f"[Writer] Dropped {len(redundancy_drops)} redundant sentence(s) at final assembly:")
        for r in redundancy_drops[:5]:
            print(f"         - overlap={r['overlap_tokens']} cited={r['cited_pairs']}: {r['snippet']}")
        if metrics:
            metrics.inc("writer_redundancy_drops", len(redundancy_drops))
            metrics.set("writer_redundancy_examples", redundancy_drops[:10])

    full_text = _strip_orphaned_citations(full_text)
    full_text = _strip_references_section(full_text)
    full_text = _strip_continuation_markers(full_text)

    meta_removed = []
    if meta_removed and metrics:
        metrics.inc("writer_meta_commentary_stripped", len(meta_removed))
    if meta_removed:
        print(f"[Writer] Stripped {len(meta_removed)} meta-commentary sentence(s):")
        for s in meta_removed[:3]:
            print(f"         - {s}")

    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

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

    full_text, final_citation_normalizations = _collapse_double_parens(
        full_text,
        display_lookup=display_lookup,
    )
    total_double_paren_collapsed += final_citation_normalizations
    if metrics:
        metrics.set(
            "writer_final_citation_normalizations",
            final_citation_normalizations,
        )

    union_docs_by_id: dict = {}
    for union_eid, union_ev in sorted(call_evidence_id_map.items()):
        union_doc_id = str(union_ev.get("doc_id", "") or "").strip()
        union_page = int(union_ev.get("page", 0) or 0)
        if not union_doc_id or union_page <= 0:
            continue
        entry = union_docs_by_id.setdefault(
            union_doc_id, {"doc_id": union_doc_id, "quotes": []},
        )
        entry["quotes"].append({
            "evidence_id": union_eid,
            "doc_id": union_doc_id,
            "page": union_page,
        })
    full_text, final_author_year_repairs, final_author_year_collisions = (
        _repair_allowed_author_year_mentions(
            full_text,
            list(union_docs_by_id.values()),
            stage="final_assembly",
        )
    )
    if final_author_year_repairs:
        full_text, final_repair_render_stats = _render_evidence_id_citations(
            full_text, call_evidence_id_map,
        )
        total_evidence_id_renders += final_repair_render_stats["replacements"]
        total_unknown_eids += final_repair_render_stats["unknown_eids"]
        unknown_eid_snippets.extend(
            final_repair_render_stats["unknown_eid_snippets"]
        )
        total_attribution_mismatches += (
            final_repair_render_stats["attribution_mismatches"]
        )
        attribution_mismatch_snippets.extend(
            final_repair_render_stats["attribution_mismatch_snippets"]
        )
    author_year_repair_stats["collision_skips"] += final_author_year_collisions
    author_year_repair_stats["replacements_proposed"] += len(
        final_author_year_repairs
    )
    author_year_repair_stats["replacements_accepted"] += len(
        final_author_year_repairs
    )
    author_year_repair_stats["final_assembly_replacements"] = len(
        final_author_year_repairs
    )
    for record in final_author_year_repairs:
        author_year_repair_records.append({
            **record,
            "phase": "final_assembly",
            "outcome": "accepted",
        })
    if metrics:
        metrics.set(
            "writer_final_author_year_repairs",
            len(final_author_year_repairs),
        )

    full_text, n_patches_dropped, n_patches_kept = _strip_coverage_patches_when_safe(
        full_text, allowed_docs, allowed_pairs, display_lookup=display_lookup,
    )
    if metrics:
        metrics.set("writer_coverage_patches_dropped", n_patches_dropped)
        metrics.set("writer_coverage_patches_shipped", n_patches_kept)
    if n_patches_dropped or n_patches_kept:
        print(
            f"[Writer] Coverage patches: dropped={n_patches_dropped} "
            f"shipped={n_patches_kept}"
        )

    citation_integrity_enforced = _writer_enforcement_enabled()
    zero_cite_dropped = []
    zc_kept_closing = None

    full_text, unresolved_display_repairs = _repair_unresolved_display_citations(
        full_text, display_lookup,
    )
    if metrics:
        metrics.set("writer_unresolved_display_repairs", unresolved_display_repairs)

    citation_merge_stats = {}
    full_text, adjacent_paren_merges = _merge_adjacent_paren_cites(
        full_text,
        display_lookup=display_lookup,
        stats=citation_merge_stats,
    )
    duplicate_citations_removed = int(
        citation_merge_stats.get("duplicate_identities_removed", 0)
    )
    if metrics:
        metrics.set("writer_adjacent_paren_merges", adjacent_paren_merges)
        metrics.set(
            "writer_duplicate_citation_identities_removed",
            duplicate_citations_removed,
        )
    if adjacent_paren_merges:
        print(f"[Writer] Merged {adjacent_paren_merges} adjacent paren-cite pair(s).")
    if duplicate_citations_removed:
        print(
            f"[Writer] Removed {duplicate_citations_removed} repeated "
            "citation identity or identities from grouped citations."
        )

    full_text, post_merge_citation_normalizations = _collapse_double_parens(
        full_text,
        display_lookup=display_lookup,
    )
    total_double_paren_collapsed += post_merge_citation_normalizations
    if metrics:
        metrics.set(
            "writer_post_merge_citation_normalizations",
            post_merge_citation_normalizations,
        )

    final_structure = _audit_final_structure(
        full_text,
        closing_docs,
        enforce_citation_integrity=citation_integrity_enforced,
    )
    _deduped_check, residual_duplicate_citations = (
        _dedupe_grouped_citation_identities(
            full_text, display_lookup=display_lookup,
        )
    )
    final_structure = _apply_residual_duplicate_gate(
        final_structure,
        residual_duplicate_citations,
        citation_integrity_enforced,
    )
    citation_surface_faults = _find_citation_surface_faults(
        full_text,
        display_lookup,
    )
    blocking_surface_faults = [
        fault for fault in citation_surface_faults
        if fault.get("type") in _BLOCKING_CITATION_SURFACE_FAULTS
    ]
    final_structure["citation_surface_faults"] = citation_surface_faults
    final_structure["blocking_citation_surface_faults"] = blocking_surface_faults
    final_structure["cosmetic_citation_surface_fault_count"] = (
        len(citation_surface_faults) - len(blocking_surface_faults)
    )
    if citation_integrity_enforced and blocking_surface_faults:
        final_structure["ok"] = False
    final_structure_manifest = {
        key: value
        for key, value in final_structure.items()
        if key != "text_without_marker"
    }
    if metrics:
        metrics.set("writer_final_structure", final_structure_manifest)
        metrics.set("writer_structure_retries_total", total_structure_retries)
        metrics.set(
            "writer_closing_model_retries_total",
            total_closing_model_retries,
        )
        metrics.set(
            "writer_structure_failure_records",
            structure_failure_records[:20],
        )
    if not final_structure["ok"]:
        ensure_dir(str(runs_path()))
        failure_record = {
            "schema_version": "v15.8",
            "error": "final_writer_structure_validation_failed",
            "final_structure": final_structure_manifest,
            "generated_closing": closing_generated_audit,
            "structure_retries": total_structure_retries,
            "structure_failures": structure_failure_records,
            "duplicate_citation_identities_removed": duplicate_citations_removed,
            "author_year_repair": {
                **author_year_repair_stats,
                "records": author_year_repair_records[:20],
            },
            "unknown_evidence_ids": total_unknown_eids,
            "unknown_evidence_id_snippets": unknown_eid_snippets[:10],
            "attribution_mismatches": total_attribution_mismatches,
        }
        with open(
            runs_path("writer_structure_failure.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(failure_record, f, indent=2, ensure_ascii=False)
        raise RuntimeError(
            "final writer structure validation failed; refusing to write review"
        )

    full_text = final_structure["text_without_marker"]
    cited_docs = _collect_cited_docs(full_text, allowed_docs, author_year_to_docid)
    cited_docids = sorted(cited_docs)

    full_text, citations_manifest = _emit_citations_manifest(
        full_text, allowed_docs, display_lookup,
        pdf_paths_by_docid, pdf_page_offsets, dois_by_docid,
        linkify=(os.environ.get("RRR_LINKIFY", "0") == "1"),
    )
    if citations_manifest:
        with open(runs_path("citations.json"), "w", encoding="utf-8") as f:
            json.dump(citations_manifest, f, indent=2, ensure_ascii=False)
        print(f"[Writer] citations.json written ({len(citations_manifest['citations'])} cites, "
              f"{citations_manifest['distinct_docs']} distinct docs)")

    total_words = _count_words(full_text)

    ensure_dir(str(runs_path()))
    out_path = runs_path("review_composed.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(runs_path("review_cited_docs.json"), "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    quality_manifest = {
        "schema_version": "v15.8",
        "word_count": total_words,
        "distinct_docs_cited": len(cited_docids),
        "chunks_written": len(chunks),
        "chunks_expected": expected_chunk_count,
        "all_planned_sections_present": len(chunks) == expected_chunk_count,
        "attribution_mismatches": total_attribution_mismatches,
        "attribution_retries": total_attribution_retries,
        "attribution_mismatch_snippets": attribution_mismatch_snippets[:20],
        "unknown_evidence_ids": total_unknown_eids,
        "unknown_evidence_id_snippets": unknown_eid_snippets[:10],
        "unsupported_year_only_citations": (
            total_unsupported_year_only_citations
        ),
        "unsupported_year_only_citation_snippets": (
            unsupported_year_only_snippets[:10]
        ),
        "author_year_repair": {
            **author_year_repair_stats,
            "records": author_year_repair_records[:20],
        },
        "invalid_citations_removed": total_removed_citations,
        "double_paren_collapsed": total_double_paren_collapsed,
        "adjacent_paren_merges": adjacent_paren_merges,
        "duplicate_citation_identities_removed": duplicate_citations_removed,
        "residual_duplicate_citation_identities": residual_duplicate_citations,
        "structure_retries": total_structure_retries,
        "closing_model_retries": total_closing_model_retries,
        "structure_failures": structure_failure_records,
        "final_structure": final_structure_manifest,
        "closing": {
            "generated_words": closing_generated_audit["word_count"],
            "generated_cited_doc_count": closing_generated_audit["cited_doc_count"],
            "generated_cited_docs": closing_generated_audit["cited_docs"],
            "generated_paragraph_count": closing_generated_audit["paragraph_count"],
            "generated_complete": closing_generated_audit["structure_ok"],
            "generated_within_word_ceiling": closing_generated_audit[
                "closing_word_ceiling_ok"
            ],
            "words_before_redundancy": closing_words_before_redundancy,
            "shipped_words": final_structure_manifest["closing"]["word_count"],
            "shipped_cited_doc_count": final_structure_manifest["closing"]["cited_doc_count"],
            "shipped_cited_docs": final_structure_manifest["closing"]["cited_docs"],
            "shipped_paragraph_count": final_structure_manifest["closing"]["paragraph_count"],
            "shipped_complete": final_structure_manifest["closing"]["structure_ok"],
            "minimum_words": _MIN_CLOSING_WORDS,
            "maximum_words": None,
        },
        "zero_cite_paragraphs_dropped": len(zero_cite_dropped),
        "zero_cite_closing_kept": 1 if zc_kept_closing else 0,
        "redundancy_drops": len(redundancy_drops),
        "style_sentences_removed": total_style_removed,
        "style_enforcement": style_stats,
        "coverage_fallbacks": total_coverage_fallbacks,
        "coverage_patches_dropped_at_final": n_patches_dropped,
        "coverage_patches_shipped_to_user": n_patches_kept,
        "quote_verify": {
            "checked": quote_verify_stats.get("checked_quotes", 0),
            "verified_real": quote_verify_stats.get("verified_real", 0),
            "fabricated_stripped": quote_verify_stats.get("fabricated_stripped", 0),
            "strand_guard_kept": quote_verify_stats.get("strand_guard_kept", 0),
            "strand_guard_dequoted": quote_verify_stats.get("strand_guard_dequoted", 0),
        },
        "outline": {
            "topic_shape": (outline_plan or {}).get("topic_shape", ""),
            "topic_cause": (outline_plan or {}).get("topic_cause", ""),
            "topic_outcome": (outline_plan or {}).get("topic_outcome", ""),
            "clusters_count": len((outline_plan or {}).get("clusters", [])),
            "unassigned_share": (outline_plan or {}).get("unassigned_share"),
            "relation_distribution": (outline_plan or {}).get("relation_distribution", {}),
        },
    }
    with open(runs_path("quality_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(quality_manifest, f, indent=2)
    print(
        f"[Writer] quality_manifest.json written "
        f"(attribution_mismatches={total_attribution_mismatches}, "
        f"unknown_eids={total_unknown_eids}, "
        f"display_leaks≈{total_removed_citations}, "
        f"coverage_patches_shipped={n_patches_kept})"
    )

    print(f"[Writer] review_composed.md written ({total_words} words).")
    print(
        f"[Writer] stats: chunks={len(chunks)} distinct_docs={len(cited_docids)} "
        f"removed={total_removed_citations} style_removed={total_style_removed} "
        f"coverage_fallbacks={total_coverage_fallbacks} evidence_id_renders={total_evidence_id_renders} "
        f"author_year_repairs={author_year_repair_stats['replacements_accepted']} "
        f"double_paren_collapsed={total_double_paren_collapsed} author_led_openings={total_author_led_openings} "
        f"redundancy_drops={len(redundancy_drops)} bracket_id_rewrites={total_bracket_id_rewrites} "
        f"unknown_eids={total_unknown_eids} attribution_mismatches={total_attribution_mismatches} "
        f"unsupported_year_only={total_unsupported_year_only_citations} "
        f"attribution_retries={total_attribution_retries} "
        f"structure_retries={total_structure_retries} "
        f"closing_model_retries={total_closing_model_retries} "
        f"duplicate_citations_removed={duplicate_citations_removed}"
    )
    if metrics:
        metrics.set("writer_stats", {
            "chunks_written": len(chunks),
            "distinct_docs_cited": len(cited_docids),
            "word_count": total_words,
            "removed_citations": total_removed_citations,
            "style_sentences_removed": total_style_removed,
            "coverage_fallbacks": total_coverage_fallbacks,
            "evidence_id_renders": total_evidence_id_renders,
            "author_year_repair": {
                **author_year_repair_stats,
                "records": author_year_repair_records[:20],
            },
            "double_paren_collapsed": total_double_paren_collapsed,
            "author_led_openings": total_author_led_openings,
            "redundancy_drops": len(redundancy_drops),
            "bracket_id_rewrites": total_bracket_id_rewrites,
            "unknown_eids": total_unknown_eids,
            "attribution_mismatches": total_attribution_mismatches,
            "unsupported_year_only_citations": (
                total_unsupported_year_only_citations
            ),
            "attribution_retries": total_attribution_retries,
            "structure_retries": total_structure_retries,
            "closing_model_retries": total_closing_model_retries,
            "duplicate_citation_identities_removed": duplicate_citations_removed,
            "final_structure": final_structure_manifest,
            "section_claims": section_claims,
            "section_coverage": section_coverage,
        })
        metrics.set("writer_mode", "chunked")
        metrics.inc("writer_removed_citations", total_removed_citations)
        metrics.inc("writer_style_sentences_removed", total_style_removed)
        metrics.inc("writer_evidence_id_renders", total_evidence_id_renders)
        metrics.set(
            "writer_allowed_author_year_repairs",
            author_year_repair_stats["replacements_accepted"],
        )
        metrics.inc("writer_double_paren_collapsed", total_double_paren_collapsed)
        metrics.inc("writer_author_led_openings", total_author_led_openings)
        metrics.set("writer_unknown_evidence_id", total_unknown_eids)
        metrics.set(
            "writer_unsupported_year_only_citations",
            total_unsupported_year_only_citations,
        )
        metrics.set("writer_attribution_mismatches", total_attribution_mismatches)
        metrics.set("writer_attribution_retries_total", total_attribution_retries)
        if unknown_eid_snippets:
            metrics.set("writer_unknown_evidence_id_snippets", unknown_eid_snippets[:10])
        if attribution_mismatch_snippets:
            metrics.set(
                "writer_attribution_mismatch_snippets",
                attribution_mismatch_snippets[:20],
            )
        if total_bracket_id_rewrites:
            metrics.inc("writer_bracket_id_rewrites", total_bracket_id_rewrites)

    return str(out_path)


def compose_review(ledger_path=None, metrics=None):
    mode = os.environ.get("RRR_WRITER_MODE", "chunked").strip().lower()
    if mode != "chunked":
        raise SystemExit(
            f"[Writer] unsupported RRR_WRITER_MODE={mode!r}; the current "
            "architecture uses the chunked corpus-outline writer"
        )
    return compose_from_ledger(ledger_path=ledger_path, metrics=metrics)


if __name__ == "__main__":
    compose_review()
