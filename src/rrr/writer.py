import os
import json
import re
import time
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
    _build_author_year_lookup,
    _build_display_lookup,
    _collect_cited_docs as _shared_collect_cited_docs,
    _doc_id_to_author_label,
)
# v15.5: citation surface postproc (rewrite_canonical_to_display,
# rewrite_misplaced_narrative_citations, collapse_nested_narrative_multicite,
# fix_orphan_outer_paren_narrative, merge_redundant_inline_paren_citations)
# moved to rrr.legacy_citation_postproc. The writer now emits only [E####]
# evidence-ID cites and the renderer below produces clean display form
# directly — eliminating the surface variations those passes patched.

# v11.2 lever 2: per-stage model selection. The writer needs prose quality;
# the reasoner needs JSON-schema obedience. Splitting lets the writer stay on
# a prose-tier model while the reasoner runs on a smaller, cheaper model.
# Falls back to RRR_MODEL if unset (preserves v11.1 behaviour).
_MODEL = os.environ.get("RRR_WRITER_MODEL", os.environ.get("RRR_MODEL", "mistral"))
_KEEP_ALIVE = "30m"

# v13: RRR_WRITER_T/CTX/PRED/TOPP/TAIL_CHARS retired (lever pruning). Per-stage
# tuning was never overridden in the wild; v7-v8 values are frozen below.
# v7 raised temperature from 0.30 to 0.50; v8 R12 dropped num_ctx from 32768
# to 12288 (largest observed writer prompt is ~2900 tokens); v8 dropped
# num_predict to 1500 (largest observed output ~620 tokens, 2.4x slack).
_DEFAULT_CHAT_OPTIONS = {
    "temperature": 0.50,
    "num_ctx": 12288,
    "num_predict": 1500,
    "top_p": 0.9,
}

# v15.4.0 (Bug 4 half-fix): bump the previous-chunk tail from 250 to 600
# chars so the next writer call sees enough prior prose to recognise an
# author who was just introduced. The cross-paragraph "Ogilvie introduced
# twice" echo in the v15.2.3 INST review traced to this slice being too
# narrow to include the previous paragraph's citation surnames.
_TAIL_CHARS = 600

# v13: _PAGE_ONLY_RE / _AUTHOR_NAME_RE / _DOC_WITHOUT_PAGE_RE / _AUTHOR_YEAR_*
# / _MULTIPAGE_CITE_RE removed. These regexes fed only the loose-pattern arm
# of _remove_invalid_citations, which is now retired per architecture item
# A-030 (evidence-id rendering + display_lookup gating supersede the
# loose-pattern scrub). The v12 smoke had writer_removed_citations=12 — those
# came from the loose arm catching display-form variants outside the
# canonical-pair gating, but with the v13 display/canonical pipeline the model
# only emits forms reachable via the lookup, so the loose arm is dead.
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
    r"thereby contributing|"
    # v13: near-paraphrases confirmed by the post-v12 prose audit. Six surfaces
    # survived v12: "complex interactions between" (paraphrase of interplay),
    # "played/playing a significant role" and "plays a significant role" (the
    # pivotal-role family with 'significant' substituted), "significantly
    # influenced" (intensifier + impact-verb pattern), "a more nuanced
    # understanding" (review wishlist), "It is crucial to examine" (paired
    # filler intent), and "are pivotal in" (predicate-adjective form of plays a
    # pivotal role).
    r"complex interactions between|"
    r"play(?:s|ed)?\s+a\s+significant\s+role|"
    r"significantly\s+influenc(?:e|ed|es)|"
    r"a\s+more\s+nuanced\s+understanding|"
    r"it\s+is\s+crucial\s+to\s+examine|"
    r"are\s+pivotal\s+in|"
    # v13.1 FIX-C: widen the pivotal family. The v13 smoke had openers like
    # "stands as a pivotal question" (topic 1) and "is pivotal for understanding"
    # (topic 4) that "are pivotal in" did not catch. Two alternation arms:
    # (1) "pivotal <noun>" surfaces (question/role/issue/factor/moment/area/
    # importance/insight), (2) modal + optional article + "pivotal" predicate
    # forms ("is/are/stands as/remains/becomes a pivotal..."). Routed to the
    # existing batched LLM style-rewriter, same as the other entries.
    r"pivotal\s+(?:question|role|issue|factor|moment|area|importance|insight)|"
    r"(?:is|are|stands\s+as|remains|becomes)\s+(?:a\s+|the\s+)?pivotal"
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
# v15.7: display-form double-paren patterns the audit found 50 times in the
# v15.6 smoke shipped to users. Display surfaces:
#   ((Author Year, p.N))          — single paren-display cite wrapped
#   ((Author Year, p.N); (Author2 Year2, p.M)) — group of paren-display cites
#   ((Author Year, p.N) (Author2 Year2, p.M))  — same, space-joined
_DOUBLE_PAREN_DISPLAY_RE = re.compile(
    r"\(\(((?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"(?:\s+(?:and\s+(?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"|et\s+al\.))?"
    r"\s+\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+)\)\)"
)
# Outer paren wrapping a group of inner display-paren cites separated by
# whitespace or semicolons. Conservative: inner must be two or more cite
# units wrapped together with no other prose in the outer paren.
_DOUBLE_PAREN_DISPLAY_GROUP_RE = re.compile(
    r"\(((?:\([^()]+?\)[\s;,]*){2,})\)"
)
# v15.7.1: asymmetric outer-paren residue — the model writes '((cite).' (open
# wrap, single close, sentence terminator after). The symmetric collapser
# above misses these because it requires '))'. The v15.7 smoke showed two of
# these shipped in GD 04 ('((Broadberry and Gupta 2006, p.17).',
# '((North 1989, p.6) (Ogilvie 2007, p.2).'). Match '((' + non-paren content
# + ')' NOT followed by ')' — strip the leading '(' (capture the content +
# close as one group). When a SECOND cite follows ('((cite1) (cite2).'), the
# outer-wrap was over the FIRST cite only; the regex catches the leading
# half and the second cite stays as-is.
_ASYM_OUTER_PAREN_RE = re.compile(
    r"\(\(([^()]+\))(?!\))"
)

# v10: detect-then-LLM-rewrite style enforcement. We deliberately do NOT
# substitute forbidden words mechanically because the right replacement is
# sense-dependent (e.g. "sharp distinction" vs "sharply declined"). Sentences
# containing any of these patterns are routed to a single batched LLM rewrite
# call per writer section.
_FORBIDDEN_LEXEME_RE = re.compile(r"\b(?:legible|sharp(?:ly)?)\b", re.IGNORECASE)
_ADVERSATIVE_RE = re.compile(
    r"(?:\b("
    # v13: widen 'not X but Y' to span 0-15 tokens between not and but. The v12
    # smoke had 'not merely the identification of factors that drive economic
    # progress, but also' which is 11 tokens — the v10 6-token cap missed it.
    r"not\s+(?:\S+\s+){0,15}but(?:\s+(?:rather|also))?\s+"
    r"|rather than\b"
    r"|in (?:sharp\s+)?contrast(?:\s+to)?\b"
    r"|instead of\b"
    r"|unlike\b"
    # v13: sentence-initial Conversely/Yet/However acting as adversative pivots.
    # The (?<=^|[.!?]\s)|(?<=\n) lookbehind anchors to a sentence boundary so we
    # don't match mid-sentence 'however' (which is rare anyway). Comma after the
    # pivot is the model's tell — these always sit at sentence start with comma.
    r")\b)|(?:(?<=[.!?]\s)(?:Conversely|Yet|However),)|(?:^(?:Conversely|Yet|However),)",
    re.IGNORECASE | re.MULTILINE,
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
# v13: RRR_STYLE_CTX/PRED retired; tuning frozen.
_OPTIONS_STYLE_REWRITE = {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1500}

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
# v13: RRR_WRITER_ENFORCE_COVERAGE retired (always on). Section-level coverage
# enforcement is part of the corpus-grounding contract; disabling it is no
# longer a supported configuration.
_ENFORCE_COVERAGE = True


def _writer_system_prompt() -> str:
    """v15.11: compose the writer's system message with an optional
    language directive prepended so the model emits prose in the user's
    language. Empty prefix for English (byte-identical to the pre-v15.11
    system prompt)."""
    from rrr.language import language_directive
    directive = language_directive(os.environ.get("RRR_TOPIC_LANG", "en"))
    if directive:
        return f"{directive}\n\n{_SYSTEM_CITATION_INSTRUCTION}"
    return _SYSTEM_CITATION_INSTRUCTION


# v10: cite as 'Author (Year, p.N)'. Underscore-keyed canonical form retired.
_SYSTEM_CITATION_INSTRUCTION = (
    # v14.2.1 simplification: reorganised into BOUNDARY (architecturally
    # enforced) and FORMAT (style) buckets so the model sees the hard rules
    # first. Citation-format material moved here from _PROSE_DIRECTIVE
    # (evidence-ID syntax + multi-source preference) since both belong with
    # the format example, not with style guidance.
    # v15.5: ONE citation surface — the evidence-ID marker [E####]. The
    # renderer produces the appropriate narrative or parenthetical form
    # depending on whether the author is already named in the prose, so
    # the writer never decides citation surface. This eliminates the
    # whole class of citation-format bugs the older pipeline patched
    # (placeholder leaks, orphan parens, redundant inline+trailing,
    # nested parens, etc.).
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
    # v15.7: removed trailing render_citation. The display surface
    # 'Author (Year, p.N)' shown here was the model's training set for
    # copy-pasting the forbidden surface into its prose. Provenance is
    # carried by the [E####] marker plus the descriptive line ('from paper
    # X, page Y') so the writer can attribute correctly without ever
    # seeing the surface it is forbidden to emit.
    did = str(q.get("doc_id", "")).strip()
    pg = int(q.get("page", 0) or 0)
    tx = _clip(q.get("text", ""), n=260)
    eid = str(q.get("evidence_id", "")).strip()
    prefix = f"[{eid}] " if eid else ""
    suffix = f" (from {did}, p.{pg})" if did and pg else ""
    return f'{prefix}"{tx}"{suffix}'


def _format_doc_entry(d) -> str:
    """v15: per-doc evidence block for the writer. The header carries doc id
    + display citation surface; the body carries the paper's CLAIM (from
    Stage 0 claim extraction) plus 1-2 evidence quotes. Legacy v9-v14 fields
    (stance, mechanism, rationale, contested, mechanisms) are no longer
    set by the v15 reasoner — when present (from a stale ledger) they are
    rendered for diagnostic continuity, but the new path renders only
    `claim + quotes` per doc.
    """
    did = str(d.get("doc_id", "")).strip()
    author_label = _doc_id_to_author_label(did)
    lines = [f"[{did}] cite as {author_label!s}"]
    claim = str(d.get("claim", "") or "").strip()
    if claim:
        lines.append(f"  Paper's central claim: {_clip(claim, n=240)}")
    # Diagnostic continuity: render the legacy v14.4 anchors when present.
    lead_mechanism = str(d.get("mechanism", "") or "").strip()
    if lead_mechanism:
        lines.append(f"  Lead mechanism (legacy): {_clip(lead_mechanism, n=180)}")
    rationale = str(d.get("rationale", "") or "").strip()
    if rationale:
        lines.append(f"  Why this stance (legacy): {_clip(rationale, n=220)}")
    contested = str(d.get("contested", "") or "").strip()
    if contested:
        lines.append(f"  What this contests (legacy): {_clip(contested, n=220)}")
    mechanisms = [str(m).strip() for m in d.get("mechanisms", []) if str(m).strip()]
    if mechanisms:
        lines.append("  Mechanisms (legacy):")
        for m in mechanisms[:2]:
            lines.append(f"  - {_clip(m, n=180)}")
    quotes_per_doc = int(os.environ.get("RRR_WRITER_QUOTES_PER_DOC", "2"))
    qs = d.get("quotes") or []
    for q in qs[:max(1, quotes_per_doc)]:
        lines.append(f"  {_format_quote(q)}")
    return "\n".join(lines)


def _list_allowed_citations(docs, allowed_pages_by_doc) -> str:
    # v15.7: emit DESCRIPTIVE provenance only — never the rendered display
    # surface. Each line maps [E####] to (paper doc_id, page) plus the
    # author surname as plain text. The writer's system prompt forbids
    # display form '(Author Year, p.N)' yet the v15.5/v15.6 list shown
    # the surface 32+ times per prompt, which is the source of the 16
    # display-form leaks the v15.6 audit found. Closing the source.
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


def _author_surnames_only(label: str) -> str:
    """Strip the trailing '(Year)' from an author label so we get just the
    surname(s) — 'Acemoglu et al. (2001)' -> 'Acemoglu et al.'."""
    if not label:
        return ""
    return re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()


# v15.5: how many chars to look BACK from an [E####] marker to decide
# whether the author surname is already named in the preceding prose.
# Keeps the lookback short so we don't false-positive across sentence
# boundaries.
_EID_LOOKBACK_CHARS = 80


def _render_evidence_id_citations(text: str, evidence_map: dict) -> tuple:
    """v15.7: context-aware evidence-ID renderer + attribution gate.

    For each [E####] marker:
      - if the author label is IMMEDIATELY before the marker (the
        immediate prior text ends with the surname label), render as
        just '(Year, p.N)' so the result forms 'Author (Year, p.N)' —
        which DISPLAY_CITE_RE recognises as a citation.
      - otherwise render as the parenthetical '(Author Year, p.N)' —
        which DISPLAY_PAREN_CITE_RE recognises.

    v15.7 (attribution gate): if the lookback finds a DIFFERENT surname
    immediately before the marker than the doc_id's expected surnames,
    REFUSE the bare-year form (which would attach the wrong author to
    the cite). Emit '(CorrectAuthor Year, p.N)' instead so the prose's
    wrong surname is at least not reinforced by a year-only cite the
    reader will attribute to that surname. Record the snippet in
    stats['attribution_mismatches'] so the caller can decide whether to
    trigger a coverage retry.

    v15.5 used a substring-anywhere-in-last-60-chars check, which fired
    on 'Mokyr argues that institutions follow from technology [E0001]'
    and rendered the marker as a bare '(1989, p.5)' embedded in lowercase
    prose. v15.6 tightened to immediate-adjacency. v15.7 adds the
    attribution check at the same lookback so the renderer becomes the
    single integrity gate.

    Returns (text, stats) where stats is a dict with keys:
        replacements: int (count of [E####] markers expanded)
        unknown_eids: int (count of markers whose eid was not in evidence_map)
        unknown_eid_snippets: list[str] (up to 10 ~120-char snippets)
        attribution_mismatches: int (count of prose-surname/eid-author mismatches)
        attribution_mismatch_snippets: list[dict] with keys
            {snippet, prose_surname, expected_surnames, eid, doc_id, page}
    """
    stats = {
        "replacements": 0,
        "unknown_eids": 0,
        "unknown_eid_snippets": [],
        "attribution_mismatches": 0,
        "attribution_mismatch_snippets": [],
    }
    if not text:
        return text or "", stats
    # v15.7.2: widened to 1-5 digits and normalised to 4-digit zero-padded
    # form before lookup. The v15.7 smoke shipped '[E009]' (3-digit — model
    # dropped a leading zero from E0009); previous regex \d{4} skipped it
    # entirely so it wasn't rendered AND wasn't counted as unknown. Now:
    # any [E<digits>] shape (1-5 digits) is caught, normalised via
    # zfill(4), and either resolved or counted.
    pattern = re.compile(r"\[([Ee]\d{1,5})\]")
    out_parts: list = []
    last_end = 0
    # v15.7: pre-build a known-surname set across the evidence_map so the
    # attribution check can recognise "prose surname is a valid author
    # surname for SOME other doc" — distinguishing a genuine wrong-author
    # binding from a model artefact like "He argues" where the immediate
    # lookback token is a pronoun.
    known_surnames = set()
    for _ev in evidence_map.values():
        _did = _ev.get("doc_id", "")
        _lbl = _author_surnames_only(_doc_id_to_author_label(_did))
        if _lbl:
            known_surnames.add(_lbl.lower())
    for match in pattern.finditer(text):
        raw_eid = match.group(1).upper()
        # v15.7.2: normalise to canonical 'E####' 4-digit zero-padded form.
        # Strip leading zeros first so '[E00009]' → '9' → '0009' → 'E0009'
        # (matches whatever the reasoner minted). '[E0]' or '[E]' would
        # normalise to 'E0000' and fail lookup (bumps unknown counter),
        # which is the right behaviour for a marker with no numeric ID.
        digits = raw_eid[1:].lstrip("0") or "0"
        eid = "E" + digits.zfill(4)
        ev = evidence_map.get(eid)
        if not ev:
            stats["unknown_eids"] += 1
            if len(stats["unknown_eid_snippets"]) < 10:
                lo = max(0, match.start() - 60)
                hi = min(len(text), match.end() + 60)
                stats["unknown_eid_snippets"].append(text[lo:hi].replace("\n", " "))
            # Leave the bare [E####] marker in place — downstream
            # _drop_zero_citation_paragraphs treats it as a recognised
            # cite (v15.7) so the LLM's prose around an unknown marker
            # is not silently lost.
            continue
        doc_id = ev["doc_id"]
        page = int(ev["page"])
        full_label = _doc_id_to_author_label(doc_id)  # "Author et al. (Year)"
        surnames = _author_surnames_only(full_label)   # "Author et al."
        # Extract bare year from the label.
        ym = re.match(r"^.*?\((\d{4}[a-z]?)\)\s*$", full_label)
        year = ym.group(1) if ym else ""

        # v15.6: stricter adjacency — author label must END the immediate
        # prefix (no intervening prose). Case-insensitive because the
        # model may capitalise particle prefixes (Van/van).
        prefix = text[max(0, match.start() - _EID_LOOKBACK_CHARS):match.start()]
        immediate = prefix.rstrip()
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
            # v15.7 attribution gate: if THIS doc's surname is not
            # adjacent, check whether some OTHER known author surname is.
            # If yes, that is a prose-vs-cite mismatch — record and force
            # case-B (parenthetical with correct author) so the bare-year
            # would not be misread as belonging to the prose surname.
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
            # Force case-B with the CORRECT author so the wrong-named
            # prose surname is not reinforced by a bare-year cite.
            if surnames and year:
                rendered = f"({surnames} {year}, p.{page})"
            else:
                rendered = render_citation(doc_id, page)
        elif author_immediately_before and year:
            # Author label is right before the marker: emit bare
            # (Year, p.N). Surrounding prose forms 'Author (Year, p.N)'
            # which DISPLAY_CITE_RE matches.
            rendered = f"({year}, p.{page})"
        elif surnames and year:
            # Author not immediately adjacent: emit (Author Year, p.N).
            # DISPLAY_PAREN_CITE_RE matches.
            rendered = f"({surnames} {year}, p.{page})"
        else:
            # Fall back to canonical narrative renderer.
            rendered = render_citation(doc_id, page)

        out_parts.append(text[last_end:match.start()])
        out_parts.append(rendered)
        last_end = match.end()
        stats["replacements"] += 1
    out_parts.append(text[last_end:])
    return "".join(out_parts), stats


# v14 FIX-BRACKET: catch [Doc_Year] / [Doc&Doc_Year] form the model invents when
# it confuses bracket-evidence-id syntax (the [E####] form we prompt for) with a
# bracketed canonical doc_id. Observed in the v13.1 3x3 smoke run 04, which
# produced `([Nunn_2008]; [Sokoloff&Engerman_2000])` -- bare canonical doc_ids
# wrapped in square brackets, no pages. _render_evidence_id_citations only
# matches the 4-digit [E####] shape, so this form slipped through.
_BRACKETED_DOC_ID_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9&.\-]*_\d{4}[a-z]?)\]")


def _render_bracketed_doc_ids(text: str, allowed_docs, doc_to_eids) -> tuple:
    """v14 FIX-BRACKET: rewrite [Doc_Year] / [Doc&Doc_Year] bracketed doc_ids
    into the display surface 'Author (Year)'. When the bracketed doc_id is in
    allowed_docs we drop the brackets and keep the display label so the prose
    still mentions the source; no page is invented because the model gave none.
    Returns (text, count). Conservative: unknown doc_ids are left untouched so
    a later validator can flag them.
    """
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


# ---------------------------------------------------------------------------
# v15.9 (#6): citations.json provenance manifest + optional --linkify
# ---------------------------------------------------------------------------

def _emit_citations_manifest(
    text: str,
    allowed_docs: set,
    display_lookup: dict,
    pdf_paths_by_docid: dict,
    pdf_page_offsets: dict,
    dois_by_docid: dict,
    linkify: bool = False,
):
    """Walk every citation in the final review text; build a machine-readable
    manifest with (doc_id, page, pdf_path, pdf_page, doi_or_url) per hit.
    Optionally rewrite the text so each cite becomes a markdown link to the
    source PDF at the right page.

    Returns (maybe_rewritten_text, manifest_dict). manifest_dict has:
        {
          "citations": [{
             "index": int,          # order of appearance in the text
             "cite_text": str,      # exact matched substring
             "doc_id": str|None,
             "page": int,
             "pdf_path": str|None,
             "pdf_page": int|None,  # page + pdf_page_offset
             "doi_or_url": str|None,
             "start": int,          # char offset in the PRE-linkify text
             "end": int,
             "surface": str         # 'canonical'|'display_narrative'|'display_paren'
          }, ...],
          "distinct_docs": int,
          "linkify": bool
        }
    """
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
        # Rewrite from end to start so char offsets stay valid.
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
    """Build a file:// URL with #page=N. Works in Preview, Chrome, Edge,
    Adobe, Zotero, and Obsidian."""
    from urllib.parse import quote
    p = str(pdf_path).replace("\\", "/")
    # Absolute file URL: file:///abs/path.pdf on unix, file:///C:/... on win
    if len(p) > 1 and p[1] == ":":  # Windows drive letter
        url = "file:///" + quote(p, safe="/:")
    elif p.startswith("/"):
        url = "file://" + quote(p, safe="/")
    else:
        url = "file://" + quote(p, safe="/")
    return f"{url}#page={int(page)}"


def _merge_adjacent_paren_cites(text: str) -> tuple:
    """v15.7.2: merge whitespace-adjacent DISPLAY_PAREN cites into one
    semicolon-joined parenthetical.
        '(Ogilvie 2007, p.9) (North 1989, p.6)' →
        '(Ogilvie 2007, p.9; North 1989, p.6)'
    Chains of 3+ merge similarly. Only merges pairs separated by pure
    whitespace (space/tab/newline) — never across prose. Narrative-form
    cites ('Author (Year, p.N)') are NOT merged because 'Author' is
    sentence prose. Runs as the very last postproc step so upstream
    validators, _collect_cited_docs, and quality_manifest all see the
    unmerged per-cite form; the merged form is user-facing only.

    Returns (text, count) where count is the number of pairs merged.
    """
    if not text:
        return text or "", 0
    matches = list(DISPLAY_PAREN_CITE_RE.finditer(text))
    if len(matches) < 2:
        return text, 0
    groups: list = []
    current: list = [matches[0]]
    ws_re = re.compile(r"\A\s+\Z")
    for m in matches[1:]:
        between = text[current[-1].end():m.start()]
        if ws_re.match(between):
            current.append(m)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [m]
    if len(current) > 1:
        groups.append(current)
    if not groups:
        return text, 0
    out: list = []
    cursor = 0
    merged_pairs = 0
    for grp in groups:
        out.append(text[cursor:grp[0].start()])
        # Strip outer '(' ')' from each match; join with '; '.
        inners = [m.group(0)[1:-1] for m in grp]
        out.append("(" + "; ".join(inners) + ")")
        cursor = grp[-1].end()
        merged_pairs += len(grp) - 1
    out.append(text[cursor:])
    return "".join(out), merged_pairs


def _collapse_double_parens(text: str) -> tuple:
    # v8 (R5): collapse ((DocId: p.N)) -> (DocId: p.N).
    # v9.1: also collapse ((DocId: p.X): p.Y) -> (DocId: p.X).
    # v15.7: extended to the THREE display-form patterns the v15.6 audit
    # found 50 times in production output while writer_double_paren_collapsed
    # reported 0 (the helper had no call site in the chunked path AND the
    # regex was canonical-only). All three normalise (cite_inner) by stripping
    # the redundant outer parens.
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
        # Sanity: every char inside must be either a single-paren cite or
        # whitespace/punctuation. We've already constrained that via the
        # regex, but double-check no stray prose.
        if re.search(r"[A-Za-z][A-Za-z0-9]{2,}", re.sub(r"\([^()]*\)", "", inner)):
            return m.group(0)
        count += 1
        # Re-join the inner cites with '; ' for a clean group.
        cites = re.findall(r"\([^()]+?\)", inner)
        return "; ".join(cites)

    def repl_asym(m):
        # v15.7.1: '((cite).' with the OUTER paren unclosed. Strip ONE of
        # the leading '(' (the unmatched one) and keep the well-formed inner.
        nonlocal count
        count += 1
        return "(" + m.group(1)

    text, _ = (_DOUBLE_PAREN_CITE_RE.subn(repl_inner_only, text))
    text, _ = (_NESTED_PAGE_SUFFIX_RE.subn(repl_nested_suffix, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_RE.subn(repl_inner_only, text))
    text, _ = (_DOUBLE_PAREN_DISPLAY_GROUP_RE.subn(repl_group, text))
    text, _ = (_ASYM_OUTER_PAREN_RE.subn(repl_asym, text))
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


def _mechanical_dash_replace(text: str) -> tuple:
    # v13: mechanical fallback for em-dashes when the LLM style rewriter rejects
    # the rewrite because preserving every (Year, p.N) citation token would
    # require an unsupported edit. The post-v12 prose audit found one such
    # sentence in the opening of the v12 smoke ("institutions—defined as the
    # rules and norms governing social interactions—determine long-term
    # economic outcomes"): the model placed em-dashes around a parenthetical
    # gloss, and the rewriter couldn't replace them without rephrasing the
    # surrounding clauses, which would risk dropping citations. The mechanical
    # substitution is safe because em/en dashes flanking a parenthetical gloss
    # always reduce to commas, and dashes used as terminal pause always reduce
    # to commas or semicolons. We pick comma uniformly — it preserves the gloss
    # structure and never changes citation tokens.
    if not text:
        return text or "", 0
    count = len(_TYPOGRAPHIC_DASH_RE.findall(text))
    if count == 0:
        return text, 0
    cleaned = _TYPOGRAPHIC_DASH_RE.sub(",", text)
    # The dash often sat between two words with no surrounding space ("word—word")
    # or with single spaces ("word — word"). Either way, ", " is the target.
    # Collapse "  ," and ",  " and ", ," that the naive substitution can create.
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    # Don't double up on existing punctuation: ", ," -> ",", ". ," -> ".",
    cleaned = re.sub(r"([.,;:!?])\s*,", r"\1", cleaned)
    cleaned = re.sub(r",\s*([.,;:!?])", r"\1", cleaned)
    return cleaned, count


# v13: detect the "mid-sentence narrative citation without preceding
# punctuation" pattern that v12 produced twice in the same run (paras 1 and 3).
# The shape is a noun phrase followed by an author label directly attached, no
# comma or period between: "extractive economic controls Akyeampong and Fofack
# (2014, p.20)" or "divergent paths of development Sokoloff and Engerman (2000,
# p.8)". We match a lowercase content word, followed by 1+ spaces, followed by
# a capital-cased author label and a "(YYYY, p.N)" parenthetical. We exclude
# attribution verb subjects (where "X argues that" is the legitimate narrative
# shape) by requiring the prior word to be a *content* noun/adjective ending in
# a typical noun/adjective suffix (or being in a small content-word blacklist),
# not a verb stem. Conservative: we only fire when (a) the preceding char is a
# space (not punctuation) AND (b) the prior word's last letter is not a
# whitespace/punctuation/comma. The fix is to insert ", " before the author
# label, turning the citation into a proper narrative continuation.
_MID_SENTENCE_NARRATIVE_CITE_RE = re.compile(
    r"(?<=[a-z0-9])\s+"  # preceded by a lowercase content character + whitespace
    r"((?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)\s+[A-Z][A-Za-z&.\-]+)"
    r"(?:\s+and\s+(?:[A-Z][A-Za-z&.\-]+|(?:van|von|de|del|der)\s+[A-Z][A-Za-z&.\-]+))?"
    r"(?:\s+et\s+al\.?)?)"
    r"\s+\((\d{4})\s*,\s*p\.(\d+)\)"
)


def _fix_mid_sentence_narrative_cites(text: str) -> tuple:
    """v13: insert a comma before narrative citations that sit mid-sentence
    without preceding punctuation. Returns (text, count). Conservative — only
    fires when the prior character is a lowercase letter or digit; never edits
    a citation that already sits after a sentence boundary or comma.
    """
    if not text:
        return text or "", 0
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        author = m.group(1)
        year = m.group(2)
        page = m.group(3)
        return f", {author} ({year}, p.{page})"

    cleaned = _MID_SENTENCE_NARRATIVE_CITE_RE.sub(repl, text)
    return cleaned, count


# v13.1 FIX-A: stray-comma cleanup inside author labels. The v13 smoke produced
# "Sokoloff and, Engerman (2000, p.5)", "North and, Weingast (1989, p.2)",
# "van, Zanden (1999, p.18)" in every topic — a stray comma between a
# connective ("and"/"with"/"or") or a lowercase particle ("van"/"von"/"de"/
# "del"/"der") and the next capitalised surname. Two regexes, conservative:
# the comma must be immediately followed by whitespace + uppercase letter.
_AUTHOR_LABEL_CONNECTIVE_COMMA_RE = re.compile(
    r"\b(and|with|or),(\s+)([A-Z][A-Za-z\-]+)"
)
_AUTHOR_LABEL_PARTICLE_COMMA_RE = re.compile(
    r"\b(van|von|de|del|der),(\s+)([A-Z][A-Za-z\-]+)"
)
# v13.1.1 (FIX-A widening): the v13.1 3x3 smoke caught "Frankema and, van
# Waijenburg" in topic 3 runs 03 + 06 — the comma sits between a CONNECTIVE
# (and/with/or) and a LOWERCASE PARTICLE (van/von/de/del/der) that itself
# precedes a capitalised surname. The v13.1 CONNECTIVE regex required an
# uppercase letter directly after the comma and missed this case. Capture
# the particle + the surname in one group so the strip preserves both.
_AUTHOR_LABEL_CONNECTIVE_PARTICLE_COMMA_RE = re.compile(
    r"\b(and|with|or),(\s+)((?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+)"
)


def _strip_malformed_author_label_commas(text: str) -> tuple:
    """v13.1 FIX-A: strip a comma sitting between an author connective
    ("and"/"with"/"or") or a lowercase particle ("van"/"von"/"de"/"del"/"der")
    and the next capitalised surname. Conservative — only fires when the
    comma is immediately followed by whitespace + uppercase letter, so
    legitimate Oxford-comma constructions in surrounding prose are untouched.
    Returns (cleaned_text, count).
    """
    if not text:
        return text or "", 0
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # v13.1.1: run the connective+particle variant FIRST so it consumes its
    # longer match (and, van Waijenburg) before the plain connective regex
    # would try to match (and, van) and produce a wrong split.
    cleaned, n0 = _AUTHOR_LABEL_CONNECTIVE_PARTICLE_COMMA_RE.subn(repl, text)
    cleaned, n1 = _AUTHOR_LABEL_CONNECTIVE_COMMA_RE.subn(repl, cleaned)
    cleaned, n2 = _AUTHOR_LABEL_PARTICLE_COMMA_RE.subn(repl, cleaned)
    return cleaned, count


# v13.1 FIX-B: literal "(Author, Year)" placeholder strip. Topic 4 of the v13
# smoke had the literal string "(Author, Year)" — the model copied the writer
# prompt's NARRATIVE/PARENTHETICAL exemplar surface verbatim. Cover the three
# shapes observed in practice: "(Author, Year)", "(Author Year)", and
# "(Author, Year, p.N)". Run before _drop_zero_citation_paragraphs so a
# paragraph whose ONLY cite was a placeholder is correctly identified as
# zero-cite and dropped.
_AUTHOR_YEAR_PLACEHOLDER_RE = re.compile(
    # v14.2 widening: comma between Year and p. is now optional, so the no-comma
    # form (Author Year p.N) — observed in the v14.1 smoke run 1 para 4 — is also
    # stripped. The previous regex required either ",\s*p.\d+" or ",\s*p.N".
    # v15.4.0 widening: also catch the case where the model substitutes a real
    # 4-digit year but leaves the literal English word "Author" as the surname
    # placeholder — e.g. "(Author 2007)" instead of "(Ogilvie 2007)". Observed
    # in the v15.2.3 INST review on Ogilvie_2007. The previous regex required
    # the literal "Year" token; this version accepts either "Year" or a real
    # year. No legitimate paper in the test corpora has surname "Author".
    r"\s?\(Author(?:,)?\s+(?:Year|\d{4}[a-z]?)"
    r"(?:(?:,\s*|\s+)p\.\s*(?:\d+|N))?\)"
)


def _strip_author_year_placeholder(text: str) -> tuple:
    """v13.1 FIX-B (widened in v14.2): strip the literal "(Author, Year)" /
    "(Author Year)" / "(Author, Year, p.N)" / "(Author Year p.N)" placeholder
    surfaces that the model occasionally copies from the system prompt's
    generic exemplar. Also strips a leading space so the surrounding sentence
    does not gain a double space. Returns (cleaned_text, count).
    """
    if not text:
        return text or "", 0
    cleaned, count = _AUTHOR_YEAR_PLACEHOLDER_RE.subn("", text)
    return cleaned, count


# v13.1 FIX-D: collapse same-(author, year) pseudo-plural multi-cites inside
# one parenthetical. The v13 smoke produced "(Bryant (2006, p.11), Bryant
# (2006, p.3))" and "(Sokoloff and Engerman (2000, p.12); Sokoloff and
# Engerman (2000, p.3))" — multiple cites of the same paper at different
# pages dressed up as multi-source support. After
# collapse_nested_narrative_multicite has flattened the inner narratives the
# form looks like "(Author Year, p.A; Author Year, p.B)" or with comma
# separators. Group by (author_label, year); if any (author, year) appears
# more than once, collapse to "(Author Year, pp.A, B)" (sorted, deduped). If
# the parenthetical reduces to ONE distinct (author, year) overall, also
# rewrite to "(Author Year, p.A)" or pp form.
_FLAT_PAREN_CITE_UNIT_RE = re.compile(
    r"(" + r"(?:(?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"(?:\s+(?:and\s+(?:(?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|"
    r"[A-Z][A-Za-z\-]+)|et\s+al\.))?"
    r")\s+(\d{4})[a-z]?,\s*p\.\s*(\d+)"
)


def _collapse_same_author_year_paren_cites(text: str) -> tuple:
    """v13.1 FIX-D: collapse pseudo-plural multi-cites where the same
    (author, year) pair appears more than once inside one top-level
    parenthetical. Returns (cleaned_text, count). Conservative: skips any
    parenthetical whose contents still have nested '(' to avoid grabbing
    wrong structure (this runs AFTER collapse_nested_narrative_multicite so
    the well-formed shape is already flat).
    """
    if not text:
        return text or "", 0
    collapses = 0

    def process_paren(m):
        nonlocal collapses
        inner = m.group(1)
        # Skip anything that still has a nested paren — leave for other passes.
        if "(" in inner or ")" in inner:
            return m.group(0)
        # Strip and parse cite units. Accept ';' or ',' as the separator
        # between cites; the model produces both.
        units = list(_FLAT_PAREN_CITE_UNIT_RE.finditer(inner))
        if len(units) < 2:
            return m.group(0)
        # Sanity: the units, joined back with their separators, must cover
        # essentially the whole inner string (allow whitespace and the
        # ';' or ',' separators). If there is substantive non-cite text
        # between units, skip — don't risk losing prose.
        residue = _FLAT_PAREN_CITE_UNIT_RE.sub("", inner)
        residue_stripped = re.sub(r"[\s,;]", "", residue)
        if residue_stripped:
            return m.group(0)
        # Group by (author, year) key (case-insensitive on the label).
        groups = []
        index = {}
        for u in units:
            author = u.group(1).strip()
            year = u.group(2)
            page = int(u.group(3))
            key = (author.lower(), year)
            if key not in index:
                index[key] = len(groups)
                groups.append({"author": author, "year": year, "pages": []})
            if page not in groups[index[key]]["pages"]:
                groups[index[key]]["pages"].append(page)
        # If every (author, year) appears exactly once and pages are all
        # singletons, nothing to do.
        if all(len(g["pages"]) == 1 for g in groups) and len(groups) == len(units):
            return m.group(0)
        collapses += 1

        def render_group(g):
            pages = sorted(g["pages"])
            if len(pages) == 1:
                return f"{g['author']} {g['year']}, p.{pages[0]}"
            return f"{g['author']} {g['year']}, pp.{', '.join(str(p) for p in pages)}"

        return "(" + "; ".join(render_group(g) for g in groups) + ")"

    # Match top-level parentheticals. We use a non-greedy match against the
    # inner content; the nested-paren guard inside process_paren rejects any
    # candidate that still contains '('.
    cleaned = re.sub(r"\(([^()]+)\)", process_paren, text)
    return cleaned, collapses


def _drop_zero_citation_paragraphs(text: str, keep_closing: bool = False) -> tuple:
    """v13: hard enforcement of rule 9 (each paragraph must integrate at least
    one cited document; the prompt asks for >=2 but the safety net catches the
    worst case of zero). The v12 smoke had three zero-citation paragraphs
    (paras 4, 9, 12 — the closing) that the prose-grounding contract forbids.
    Drop them.

    Returns (cleaned_text, [removed_paragraph_snippets], kept_closing_snippet|None).
    Conservative: we only count CANONICAL `(Doc_Year: p.N)` cites or DISPLAY-form
    `Author (Year, p.N)` / `(Author Year, p.N)` cites. If neither shape is
    present the paragraph cites zero documents and gets dropped.

    v14: when keep_closing=True and the LAST non-empty paragraph would be
    dropped, keep it anyway (a weak closing reads better than a missing
    closing — the closing is structurally important to the essay). The
    returned third tuple element is the snippet of that kept closing (or None).
    """
    if not text:
        return text or "", [], None
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text or "", [], None
    last_idx = len(paragraphs) - 1
    kept = []
    removed = []
    kept_closing_snippet = None
    # v15.7: recognise unrendered [E####] markers as a citation form too.
    # When the renderer can't map a marker (unknown eid), it leaves the
    # bare marker in place AND bumps writer_unknown_evidence_id. Without
    # this check, the surrounding prose would be silently dropped as
    # zero-citation when the real failure is a single bad marker.
    # v15.7.2: also recognise 1-3 digit variants and non-numeric E-prefixed
    # sentinels (e.g. '[E9]', '[Evidence]') so a hallucinated-marker
    # paragraph isn't lost — the quality manifest surfaces them separately.
    _UNRENDERED_EID_RE = re.compile(r"\[[Ee]\w*\]")
    for i, para in enumerate(paragraphs):
        has_canonical = bool(CITE_RE.search(para))
        has_display_narrative = bool(DISPLAY_CITE_RE.search(para))
        has_display_paren = bool(DISPLAY_PAREN_CITE_RE.search(para))
        has_unrendered_eid = bool(_UNRENDERED_EID_RE.search(para))
        if has_canonical or has_display_narrative or has_display_paren or has_unrendered_eid:
            kept.append(para)
            continue
        snippet = re.sub(r"\s+", " ", para.strip())[:200]
        snippet = snippet + ("…" if len(para.strip()) > 200 else "")
        if keep_closing and i == last_idx:
            # Closing is structurally important; keep the weak closing rather
            # than truncate the essay. Emit the metric via the caller.
            kept.append(para)
            kept_closing_snippet = snippet
            continue
        removed.append(snippet)
    cleaned = "\n\n".join(kept)
    return cleaned, removed, kept_closing_snippet


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
      4. v13: any em-dashes that survived (because the rewriter rejected the
         rewrite to preserve citation tokens) get mechanically replaced with
         commas. This is the post-v12 fix for the opening-sentence em-dash that
         the LLM rewriter couldn't touch because the sentence had four cites
         and the rewriter's citation-preservation guard fired.

    Returns (text, stats_dict). Stats keys: 'trailing_stripped', 'violations',
    'rewrites_applied', 'fallback_reason', 'mechanical_dashes_replaced'.
    """
    # v13: RRR_STYLE_ENFORCE retired (always on); a no-style-enforcement
    # writer is no longer a supported configuration.
    if not text:
        return text, {"trailing_stripped": 0, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "empty",
                      "mechanical_dashes_replaced": 0}

    text, trailing_stripped = _strip_trailing_significance(text)

    sentences, violations = _collect_style_violations(text)
    if not violations:
        # Even with no other violations, a stray em-dash can sit in text that
        # didn't make the violation list (e.g. dashes in the input were already
        # there before the run). Run the mechanical pass anyway as a safety net.
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped, "violations": 0,
                      "rewrites_applied": 0, "fallback_reason": "no_violations",
                      "mechanical_dashes_replaced": mechanical_replaced}

    new_sentences, applied, reason = _rewrite_style_violations(
        sentences, violations, metrics=metrics,
    )
    if applied == 0:
        # LLM rewrite produced nothing usable. Still clean em-dashes
        # mechanically so a citation-preservation rejection doesn't leave the
        # dashes in the output.
        text, mechanical_replaced = _mechanical_dash_replace(text)
        return text, {"trailing_stripped": trailing_stripped,
                      "violations": len(violations),
                      "rewrites_applied": 0,
                      "fallback_reason": reason,
                      "mechanical_dashes_replaced": mechanical_replaced}
    rewritten_text = _splice_sentences_back(text, new_sentences)
    # v13: post-LLM-rewrite mechanical sweep. Sentences whose rewrites were
    # individually rejected (citation drift) still hold their em-dashes.
    rewritten_text, mechanical_replaced = _mechanical_dash_replace(rewritten_text)
    return rewritten_text, {"trailing_stripped": trailing_stripped,
                            "violations": len(violations),
                            "rewrites_applied": applied,
                            "fallback_reason": reason,
                            "mechanical_dashes_replaced": mechanical_replaced}


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


def _sentence_citation_pairs(sentence: str, display_lookup=None):
    # v15.7: was canonical-only; now resolves display surfaces too via
    # parse_citations + display_lookup. Drop-aware-of-surface so the
    # redundancy walker doesn't keep a sentence that "has no canonical
    # cite" simply because the cite is in display form.
    pairs = set()
    for c in parse_citations(sentence, display_lookup=display_lookup):
        if c["doc_id"] is None:
            continue
        pairs.add((c["doc_id"], c["page"]))
    return pairs


def _sentence_doc_ids(sentence: str, display_lookup=None) -> set:
    """v15.7: helper for the attribution-aware drop guard. Returns just the
    doc_id set (no pages) for a sentence."""
    return {p[0] for p in _sentence_citation_pairs(sentence, display_lookup=display_lookup)}


def _drop_would_strand_surname(prev_text: str, sent: str, allowed_docs,
                                 display_lookup=None) -> bool:
    """v15.7 drop-aware-of-surname guard. Returns True if dropping `sent`
    would leave the IMMEDIATELY-PRECEDING prose in prev_text with an
    author surname that has no surviving cite of theirs in the dropped-
    sentence's cite set — exactly the GD 04 Kuznets→Mokyr failure class.

    Heuristic: take the last 80 chars of prev_text, scan for any known
    author surname (from allowed_docs / display_lookup), and check if any
    of those surnames maps to a doc_id in sent's cite set. If yes, the
    drop strands the surname (because the cite that gave it semantic
    cover would be removed).
    """
    if not prev_text or not sent:
        return False
    tail = prev_text[-120:]
    sent_doc_ids = _sentence_doc_ids(sent, display_lookup=display_lookup)
    if not sent_doc_ids:
        return False
    # Build surname -> doc_ids index for the allowed corpus.
    surname_to_docids: dict = {}
    for did in (allowed_docs or set()):
        surnames = _author_surnames_only(_doc_id_to_author_label(did)).lower()
        if surnames:
            surname_to_docids.setdefault(surnames, set()).add(did)
    # Check tail for any surname whose doc set overlaps sent's cites.
    tail_lower = tail.lower()
    for surnames, dids in surname_to_docids.items():
        if surnames in tail_lower and (dids & sent_doc_ids):
            return True
    return False


def _drop_cross_section_redundancy(text: str, min_token_overlap: int = 4,
                                     display_lookup=None, allowed_docs=None):
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
                sent_pairs = _sentence_citation_pairs(sent, display_lookup=display_lookup)
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
                    # v15.7: drop-aware-of-surname guard. Refuse the drop
                    # when removing `sent` would leave a preceding-prose
                    # surname stranded (its only cite was in `sent`).
                    prev_text = " ".join(kept_sentences)
                    if _drop_would_strand_surname(
                        prev_text, sent, allowed_docs, display_lookup=display_lookup,
                    ):
                        kept_sentences.append(sent)
                        seen_tokens.append(this_tokens)
                        continue
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


def _split_sentences_for_cleanup(line: str):
    sentinel = "__RRR_DOT__"

    def protect(m):
        return m.group(0).replace(".", sentinel)

    protected = re.sub(r"\bet\s+al\.", protect, line, flags=re.IGNORECASE)
    protected = re.sub(r"\b(?:e\.g|i\.e|cf)\.", protect, protected, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace(sentinel, ".") for p in parts if p.strip()]


def _remove_invalid_citations(text: str, allowed_docs: set, allowed_pairs=None,
                                display_lookup=None) -> tuple:
    # v15.7: surface-agnostic via parse_citations(text, display_lookup).
    # Previously CITE_RE-only — counted writer_removed_citations=0 in all
    # v15.6 runs while the audit found 5 mis-attribution blockers and 16
    # display-form leaks. With display_lookup provided, display-form
    # cites resolve to a doc_id and the validator can detect:
    #   - unknown_doc: doc_id not in allowed_docs (cite to a paper outside
    #     this chunk's allowed set)
    #   - invalid_page: doc_id known but (doc, page) not in allowed_pairs
    #   - unresolved_display: display-form whose (label, year) does not
    #     resolve via display_lookup at all (probably an author label the
    #     model invented or a collision in the lookup)
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
                # Display-form that did not resolve via display_lookup.
                invalid.append((
                    c["start"], c["end"],
                    c.get("label") or "?",
                    page,
                    "unresolved_display",
                ))
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


# v12: detect meta-commentary about the review/paper itself. The closing
# section of v11.2 leaked sentences like "The literature reviewed here
# converges on...". Catch-and-strip rather than catch-and-warn because these
# are pure noise — the substantive claim sits in the next sentence.
_META_COMMENTARY_RE = re.compile(
    r"\b("
    r"the literature reviewed (?:here|above)?"
    r"|this (?:review|paper|essay|article|analysis|discussion|investigation)"
    r"|the (?:sources|works|studies|papers) (?:cited|reviewed|discussed|surveyed)"
    r"|as (?:discussed|noted|shown|argued|established) (?:above|earlier)"
    r"|in (?:this|the (?:foregoing|preceding)) (?:review|paper|essay|article|section|discussion)"
    r"|we (?:will|shall) (?:examine|explore|investigate|consider|argue|show|discuss|analyse|analyze)"
    r"|this (?:section|paragraph) (?:examines|explores|investigates|considers|argues|shows|discusses)"
    # v13: "the thesis" used as self-reference to the review's own argument
    # ("important qualification to the thesis", "applies the thesis to",
    # "illustrates how the thesis can be applied"). Two patterns:
    # - "the thesis" preceded by a self-referential preposition or noun
    # - "this is an important [noun] to the thesis" — the v12 audit's rule 6
    #   trailing-significance miss is really meta-commentary about the review.
    r"|(?:applies|illustrates how|qualification(?:s)? to|adds nuance to|in support of|against) the thesis"
    r"|this is (?:an? )?important (?:qualification|implication|consequence|insight|point) (?:to|for|of) the thesis"
    r")\b",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> tuple:
    """v12: drop sentences that talk about the review itself rather than the
    substantive question. Returns (cleaned_text, list_of_removed_snippets)."""
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
    # v12: default flipped from "1" to "0". Sequential writing with the v11.2
    # claims-so-far context produces tighter cross-section coherence (0 twin
    # openers vs 3 in parallel) at minimal runtime cost (+10-15s). Set
    # RRR_WRITER_PARALLEL=1 to opt back into parallel.
    if n_chunks <= 1 or os.environ.get("RRR_WRITER_PARALLEL", "0") == "0":
        return 1
    raw = os.environ.get("RRR_WRITER_PARALLELISM") or os.environ.get("RRR_CONCURRENCY") or "2"
    try:
        workers = int(raw)
    except Exception:
        workers = 1
    return max(1, min(workers, n_chunks))


def _strict_cited_doc_ids(text: str, allowed_pairs=None, display_lookup=None) -> set:
    # v15.7: parse_citations now iterates all three surfaces. Pass
    # display_lookup so display-form cites resolve to a doc_id (without
    # it they yield doc_id=None and are invisible to the coverage audit
    # — exactly the v15.5 silent-failure mode that produced the writer
    # collapse). When allowed_pairs is empty all resolved cites are
    # counted; otherwise only (doc, page) pairs in allowed_pairs.
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
        return 1
    return min(max(1, _MIN_SECTION_CITED_DOCS), n_docs)


def _audit_section_coverage(text: str, chunk_docs, section_kind: str):
    chunk_pairs, chunk_allowed_docs, _ = _build_allowed_citations(chunk_docs)
    # v15.7: pass a display_lookup built from chunk_allowed_docs so the
    # audit can resolve display-form cites natively (no canonical
    # conversion needed).
    chunk_display_lookup_local = _build_display_lookup(chunk_allowed_docs)
    cited = _strict_cited_doc_ids(
        text, allowed_pairs=chunk_pairs, display_lookup=chunk_display_lookup_local,
    )
    required = _coverage_requirement(chunk_docs, section_kind)
    return {
        "section": section_kind,
        "required_cited_docs": required,
        "cited_doc_count": len(cited),
        "cited_docs": sorted(cited),
        "provided_doc_count": len(chunk_allowed_docs),
        "ok": len(cited) >= required,
    }


# v15.7: sentinel prefix on coverage-fallback patch sentences. Final
# assembly drops these unless removing them would push the section
# below its coverage floor. v15.6 run 04 shipped a patch sentence
# verbatim as the first line of the review ("A further source records
# ... (Kuznets_1973: p.7).") — a UX failure mode this sentinel fixes.
_COVERAGE_PATCH_SENTINEL = "[[COV-PATCH]] "


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
        # v15.7: prepend sentinel; final assembly strips these unless
        # removing them would push the section below coverage.
        additions.append(
            f'{_COVERAGE_PATCH_SENTINEL}A further source records "{tx}" {render_citation(did, pg)}.'
        )
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


def _strip_coverage_patches_when_safe(text: str, allowed_docs, allowed_pairs,
                                        display_lookup=None) -> tuple:
    """v15.7: walk the assembled essay paragraph-by-paragraph; for each
    paragraph that contains a coverage-patch sentence (prefixed with the
    sentinel), test whether removing the patch sentence still leaves the
    paragraph above its coverage floor (>=1 cited doc — paragraphs already
    pass the section-level coverage at this point; this is a per-paragraph
    safety check). Drop patch sentences when safe, strip the sentinel when
    not. Returns (cleaned_text, n_dropped, n_kept).
    """
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
        if len(non_patch_cited) >= 1:
            # Paragraph stands on its own — drop the patch sentences.
            new_para = " ".join(non_patch_sents).strip()
            if new_para:
                out_paras.append(new_para)
                n_dropped += sum(
                    1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
                )
            else:
                # Empty after drop — keep with sentinel stripped.
                cleaned = para.replace(_COVERAGE_PATCH_SENTINEL, "")
                out_paras.append(cleaned)
                n_kept += sum(
                    1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
                )
        else:
            # Patch is load-bearing — strip sentinel, keep sentence.
            cleaned = para.replace(_COVERAGE_PATCH_SENTINEL, "")
            out_paras.append(cleaned)
            n_kept += sum(
                1 for s in sentences if _COVERAGE_PATCH_SENTINEL in s
            )
    cleaned_text = "\n\n".join(out_paras)
    return cleaned_text, n_dropped, n_kept


def _coverage_retry_prompt(original_prompt: str, prior_chunk: str, required_docs: int) -> str:
    # v15.7: retry prompt now demands [E####] markers consistent with
    # _SYSTEM_CITATION_INSTRUCTION. The previous version forbade
    # author-year citations and demanded canonical '(DocId: p.N)' — both
    # contradicting the system prompt's '[E####] only' rule. The new
    # version restates the marker contract and asks for at least N
    # distinct DOCS-worth of markers (the audit checks doc coverage, not
    # marker count).
    return f"""{original_prompt}

Coverage repair:
The previous draft failed the citation coverage rule. Write the section again using only [E####] markers drawn from the ALLOWED CITATIONS list at the top of this prompt.

Requirements:
- Cite at least {required_docs} DIFFERENT papers when that many are listed. (One paper can contribute multiple markers; the count is of distinct papers.)
- Every paragraph must include at least one [E####] marker.
- Use ONLY [E####] markers from the ALLOWED CITATIONS list. Do not invent IDs. Do not write '(Author Year, p.N)', 'Author (Year, p.N)', '(DocId: p.N)' or any other citation surface — the renderer expands every [E####] into the correct surface.
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


# =============================================================================
# v7: LEANER PROMPTS - Claims about phenomena, not scholars
# =============================================================================

_PROSE_DIRECTIVE = (
    # v14.2.1 simplification: trimmed from ~190 words to ~80. Citation-format
    # material moved to _SYSTEM_CITATION_INSTRUCTION.
    # v14.4: reframed as a "literature review organised around streams of
    # related scholarship" — each section now presents one stream of work
    # and synthesises across the sources within it. The per-section builders
    # match the framing ("present this stream", "this stream challenges...",
    # "this stream introduces..."). Streams are a more natural unit for
    # literature-review prose than the older "chain of inference" metaphor.
    "Write a scholarly literature review organised around streams of "
    "related scholarship. Each section presents ONE stream and synthesises "
    "across its sources. Argue in direct positive claims; avoid contrastive "
    "framings such as 'not X but Y', 'rather than', 'unlike', or 'in "
    "contrast'. Vary sentence length; introduce claims as full sentences "
    "rather than via colon setups. "
    # v12 multi-doc rule.
    "Each paragraph must integrate at least two DISTINCT documents; do not "
    "cite the same document for more than two consecutive citations. "
    # v12 anti-meta + topic restate.
    "Make claims directly about the substantive question. Do NOT write meta-"
    "statements about the review itself ('this review', 'the literature "
    "reviewed here', 'as discussed above', 'we will examine'). "
    "Do not restate the topic or the thesis."
)

# v8 (R4): single canonical set of prompt builders. The previous file had two
# definitions per builder — the first was dead code shadowed by the second.
# Each stance builder now has a structurally DISTINCT instruction so the
# resulting paragraphs read as substantively different arguments rather than
# three parallel templates with one verb swapped.

def _prev_tail_block(previous_tail: str) -> str:
    """v14.2 cond D: render the 'Previous ending: …' block with an optional
    strong directive (RRR_WRITER_PREV_TAIL_STRONG=1) that explicitly tells the
    writer NOT to restate the topic and instead continue from the tail. Returns
    the formatted block; safe to include even when previous_tail is empty (the
    opening section's first call has no prior chunk)."""
    if not previous_tail:
        return ""
    block = f"Previous ending:\n...{previous_tail}\n"
    # v14.2 default flipped 0 -> 1: the v14.2 cond D smoke showed the strong
    # directive cut mean seam count from 3.3 (cond A) to 2.3 with zero LLM
    # cost. Set RRR_WRITER_PREV_TAIL_STRONG=0 to reproduce v14.1 prompt shape.
    if os.environ.get("RRR_WRITER_PREV_TAIL_STRONG", "1") == "1":
        block += (
            "\nIMPORTANT: Open your first sentence as a natural continuation "
            "of the Previous ending above — pick up a thread, contrast a "
            "finding, follow a scope condition. Do NOT begin by restating the "
            "topic on its own; the reader has just read the previous section.\n"
        )
    return block


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


# v15.4.0 (Bug 4 half-fix): extract author surnames the writer has ALREADY
# named in prior chunk prose, so the next chunk's prompt can explicitly
# forbid re-introducing them with a fresh "X argues..." sentence.
# Pulls surnames from both display narrative cites "Surname (Year" and
# canonical "(Doc_Year: p.N)" tokens (via _doc_id_to_author_label).
_DISPLAY_NAMED_AUTHOR_RE = re.compile(
    r"\b((?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"(?:\s+and\s+(?:(?:van|von|de|del|der)\s+)?[A-Z][A-Za-z\-]+"
    r"|\s+et\s+al\.)?)\s+\(\d{4}[a-z]?(?:,|\s)"
)


def _extract_named_surnames(chunk: str) -> list:
    """Return the list of unique author labels named in prose (narrative
    cites and canonical cites resolved to author labels). Order-preserving.
    """
    if not chunk:
        return []
    seen = []
    for m in _DISPLAY_NAMED_AUTHOR_RE.finditer(chunk):
        label = m.group(1).strip()
        # heuristic guard: drop ones that are clearly sentence-initial
        # English words ("The (2020)" etc.). Common short labels are fine.
        if label.lower() in {"the", "this", "that", "these", "those", "many", "some",
                             "such", "their", "they", "his", "her", "our", "your"}:
            continue
        if label not in seen:
            seen.append(label)
    for m in CITE_RE.finditer(chunk):
        did = m.group(1)
        label = _doc_id_to_author_label(did)
        # strip the trailing "(Year)" if present in the label
        label = re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", label).strip()
        if label and label not in seen:
            seen.append(label)
    return seen


def _format_claims_so_far(section_claims) -> str:
    """v11.2 lever 3: render the running record of prior sections into a prompt
    block so the next writer call knows what claims have already been made.
    Returns empty string when there are no prior sections (the opening, the
    first stance section, or the parallel writer path where every section
    starts simultaneously).
    v15.4.0 (Bug 4 half-fix): also surface 'Already named in prose: ...' so
    the next chunk's prompt knows which authors have been introduced
    narratively and should not be re-introduced with a fresh 'X argues...'.
    """
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


def _build_supports_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                           previous_tail: str, synthesis_block: str = "",
                           claims_so_far: str = ""):
    # v8 (R4) — distinct shape: single chain of inference, ~260-300 words.
    # v11.2 lever 3: claims_so_far renders the running record of prior sections
    # when the writer runs sequentially (RRR_WRITER_PARALLEL=0).
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    return f"""Continue this literature review on: {topic}

{prev_block}
{prior_section}Theme: {cluster}. These sources SUPPORT the thesis through corroborating mechanisms, measurements, or historical conditions.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Present this stream of literature. Open by stating the shared mechanism this body of work converges on, citing the supporting sources together (per the CLUSTER SYNTHESIS above if present, else the strongest 2-3 in Evidence). Narrow to one source for a specific phrase that establishes the mechanism; show how a second source confirms it under different conditions (period, region, measurement); trace one downstream implication. 260-300 words. End mid-thought.

Continue:"""


def _build_critiques_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                            previous_tail: str, synthesis_block: str = "",
                            claims_so_far: str = ""):
    # v8 (R4) — distinct shape: force a choice. Shorter, sharper.
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    return f"""Continue this literature review on: {topic}

{prev_block}
{prior_section}Theme: {cluster}. These sources CHALLENGE the thesis by identifying weak assumptions, alternative causes, or contrary evidence.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Present the stream of literature that challenges the supporting case. Identify the specific assumption this stream attacks, then cite the evidence it brings to bear inside the same sentence. When the CLUSTER SYNTHESIS above identifies a shared critical move across multiple sources, name it once with a multi-citation parenthetical. Force the reader to weigh the assumption against the counter-evidence. 200-240 words. End mid-thought.

Continue:"""


def _build_complicates_prompt(topic: str, cluster: str, evidence: str, allowed_list: str,
                              previous_tail: str, synthesis_block: str = "",
                              claims_so_far: str = ""):
    # v8 (R4) — distinct shape: pick the binding scope condition and develop it.
    synthesis_section = (synthesis_block + "\n\n") if synthesis_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    return f"""Continue this literature review on: {topic}

{prev_block}
{prior_section}Theme: {cluster}. These sources COMPLICATE the thesis by setting scope conditions, contingencies, or measurement limits.

ALLOWED CITATIONS:
{allowed_list}

{synthesis_section}Evidence:
{evidence}

{_PROSE_DIRECTIVE}

Present the stream of literature that complicates the supporting claim. Pick the single scope condition this stream introduces that most narrows the claim, and use one extended example to show what the thesis can and cannot explain under it. If the CLUSTER SYNTHESIS above names a scope condition shared across multiple sources, cite them together when introducing it. Treat the qualification as reshaping the question, not as a rejection. 220-260 words. End mid-thought.

Continue:"""


# =============================================================================
# v15: single stream-section builder. Replaces the per-stance triplet
# (_build_supports_prompt / _build_critiques_prompt / _build_complicates_prompt).
# The cluster's posture (free-text elaboration + structural relation) is the
# section's substantive claim; the writer never sees "supports / critiques /
# complicates" vocabulary in its instructions.
# =============================================================================

# Human-readable, topic-agnostic glosses for each per-shape relation enum
# value. Used by the section prompt to tell the writer HOW this stream stands
# to the topic — structurally, never with bucket vocabulary.
_RELATION_OPENERS = {
    # causal shape (6 values)
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
    # comparative shape (5 values)
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
    # descriptive shape (4 values)
    "confirms_description":
        "this stream's account is CONSISTENT with the topic's description "
        "and adds corroborating evidence for the same pattern",
    "contradicts_description":
        "this stream's account is INCOMPATIBLE with the topic's "
        "description and reports a different pattern in the same subject",
    "adds_nuance":
        "this stream ACCEPTS the topic's description but refines its scope, "
        "magnitude, or mechanism",
}


def _format_outline_block(cluster: dict, doc_to_evidence_ids) -> str:
    """v15: render a cluster's outline posture into a prompt-ready block.

    Replaces v11-C's `_format_cluster_synthesis_block`. The new block carries
    (a) the free-text elaboration the writer should render verbatim as the
    section's posture, (b) a multi-citation list of the cluster's papers via
    evidence IDs (the writer is told to use these together when stating the
    shared claim), (c) internal disagreement if any.
    """
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
                         forbidden_opener_surnames: list = None):
    """v15: single section builder. Same arity as the legacy per-stance
    builders so the dispatcher loop in compose_from_ledger does not need to
    branch on shape. `relation` and `topic_shape` are passed through so the
    prompt can name the stream's structural relation to the topic — without
    any supports/critiques/complicates vocabulary.

    v15.7: `lead_surname` + `forbidden_opener_surnames` is the
    parallel-writer guard. Each section computes its lead author from
    outline_plan's lead_doc_id upfront; downstream sections are told
    explicitly which surnames they MUST NOT use as the section opener.
    Closes the INST 02 cloned-author-opener class ('Austin emphasizes
    ... patterns and paths' appearing in 3 different chunks with 3
    different cites) which the parallel writer's empty claims_so_far
    silenced in v15.6.
    """
    outline_section = (outline_block + "\n\n") if outline_block else ""
    prior_section = (claims_so_far + "\n\n") if claims_so_far else ""
    prev_block = _prev_tail_block(previous_tail)
    relation_gloss = _RELATION_OPENERS.get(relation or "", "")
    relation_line = (
        f"Structural relation of this stream to the topic ({relation or 'unspecified'}): "
        f"{relation_gloss}.\n"
        if relation_gloss else ""
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

Present this stream of literature. Open with a direct, substantive claim grounded in the STREAM POSTURE above — never name the stream's structural relation in the prose (no 'this stream supports/critiques/complicates'); express the relation by the shape of the argument and the sources you cite together. Cite the stream's lead and one supporting source in the same sentence when stating the shared claim. Develop the claim through one specific mechanism, condition, or pattern the sources jointly establish; trace one downstream implication that another source in the stream sharpens or qualifies. 220-300 words. End mid-thought; the next section continues the review.

Continue:"""


def _build_closing_prompt(topic: str, evidence: str, allowed_list: str, previous_tail: str):
    # v8 (R4/R6): closing must synthesise without restating prior section claims;
    # do not paraphrase the same two mechanisms that opened.
    # v13: the v12 smoke produced a methodology-stub closing ("This would
    # involve measuring..."), zero citations, no synthesis. The prompt now
    # forbids the future-research / methodology-stub register explicitly and
    # demands the closing cite at least two distinct documents from the
    # allowed list. The synthesis structure (agreement, disagreement,
    # resolution) is preserved but reframed as ACTIVE statements grounded in
    # the cited sources, not as hypothetical assessments of what would settle
    # the question.
    prev_block = _prev_tail_block(previous_tail)
    return f"""Close this literature review on: {topic}

{prev_block}
ALLOWED CITATIONS:
{allowed_list}

Section claims already made (do NOT restate the specific mechanisms below; synthesise across them):
{evidence}

{_PROSE_DIRECTIVE}

Write the closing as a synthesis grounded in the cited sources. Structure:
1. Name the underlying point of agreement that the literature converges on, citing AT LEAST ONE source by author and page.
2. Name the precise remaining disagreement that the cited sources leave open, citing AT LEAST ONE distinct source by author and page (so the closing cites at least 2 distinct documents in total).
3. Identify the specific historical evidence or empirical comparison that would settle the disagreement, drawing on a measurement type or case that the cited sources have already used — not a future-research wishlist.

STRICT RULES:
- Do NOT write methodology-stub prose: avoid "this would involve measuring", "future research should", "further work would assess", "would provide a nuanced understanding", "remains an open question", "remains a subject of debate".
- v13.1 FIX-E: also forbidden — "would be necessary", "would illuminate", "would clarify", "would resolve", "would shed light", "would inform", "would allow us to", "thorough comparison", "thorough examination". More generally: do NOT attach the modal "would" to a methodology verb such as comparison, examination, assessment, analysis, or investigation.
- Use present-tense indicative statements about what the cited sources SHOW, not conditional-tense statements about what future work WOULD show.
- Do NOT write "In conclusion" or "To summarize".
- Every claim must be tied to a cited source from the allowed list and must name an author AND a page. Closing paragraphs without citations are forbidden.
- 150-200 words.

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
    _system = _writer_system_prompt()
    _dump_writer_prompt(stage, _system, prompt)
    start = time.perf_counter()
    try:
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _system},
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


def _apply_full_essay_harmonisation(full_text: str, topic: str, metrics=None):
    """v14.1 (Option A): one LLM call that rewrites the FULL assembled chunked
    essay for prose continuity. v14.2 added two knobs to address mistral-small's
    citation-drift failures: RRR_HARMONISE_DRIFT_TOLERANCE (allow ≤N missing
    tuples while still forbidding inventions of new documents) and
    RRR_HARMONISE_MODEL (override the harmoniser model — useful for testing
    qwen3:32b against mistral-small:24b on the same draft).

    Background. The v14 single-pass writer experiment (pod_outputs/
    runpod_20260628_213140) confirmed that the mistral family wraps up around
    400 words regardless of the prompt's length target — so for the long-form
    case we must keep chunked generation, and address the "starts again" seam
    at section boundaries by editing the assembled essay rather than by
    expanding any individual writer call. The cross-section stitch (v11-B)
    already rewrites the FIRST SENTENCE of each interior section; this
    harmonisation pass goes broader, rewriting the whole essay's prose flow
    while a citation-set check protects every (Doc_Year: p.N) token.

    Operates on the canonical-form essay (citations already normalised to
    (Doc_Year: p.N) by the prior display->canonical step). On any verification
    failure the original text is returned unchanged.

    Returns (text, stats_dict). stats keys: applied (bool), fallback_reason,
    cite_count_in, cite_count_out, len_in, len_out, duration_s, model,
    drift_tolerance.

    Toggle: RRR_WRITER_HARMONISE (default 1; set 0 to disable for the v13.1.1
    chunked baseline comparison).
    """
    harmonise_model = os.environ.get("RRR_HARMONISE_MODEL", _MODEL)
    # v14.2 default flipped 0 -> 1: the v14.2 12-run shoot-out showed that with
    # tolerance=0 mistral-small:24b rejected 3/3 full-essay rewrites; with
    # tolerance=1 + invented-doc safety check, 2/3 accept (the 1 rejection was
    # the safety net catching a malformed `_2001` cite). See current_status.md
    # v14.2 section.
    try:
        drift_tol = max(0, int(os.environ.get("RRR_HARMONISE_DRIFT_TOLERANCE", "1")))
    except ValueError:
        drift_tol = 1
    stats = {
        "applied": False, "fallback_reason": "disabled",
        "cite_count_in": 0, "cite_count_out": 0,
        "len_in": len(full_text or ""), "len_out": len(full_text or ""),
        "duration_s": 0.0,
        "model": harmonise_model, "drift_tolerance": drift_tol,
    }
    if not full_text or os.environ.get("RRR_WRITER_HARMONISE", "1") != "1":
        return full_text, stats

    # Canonical cite tuples are the contract we must preserve EXACTLY (default)
    # or within drift_tol tuples (when RRR_HARMONISE_DRIFT_TOLERANCE is set).
    orig_cites = sorted([(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(full_text)])
    stats["cite_count_in"] = len(orig_cites)
    if not orig_cites:
        stats["fallback_reason"] = "no_canonical_cites_to_protect"
        return full_text, stats

    prompt = (
        "You are revising a literature review for prose continuity. The draft "
        "below was assembled from independently-written sections; the seams "
        "where sections meet currently read as 'topic restarts'. Rewrite the "
        "draft so that:\n\n"
        "- Each paragraph FLOWS from the previous one (pick up a thread, "
        "contrast a finding, follow a scope condition, build on a mechanism). "
        "Eliminate any paragraph opener that re-states the topic or re-"
        "introduces the framing.\n"
        "- Vary connective phrasing across paragraphs. Do NOT repeat patterns "
        "like 'X is another dimension of this debate', 'Another perspective "
        "is', 'This view is challenged by'. Each paragraph deserves a distinct "
        "opener shape.\n"
        "- Preserve EVERY (Doc_Year: p.N) citation token EXACTLY. Do not add, "
        "remove, alter, or reposition any citation. Keep the same documents "
        "on the same claims. (This is hard-checked; if any citation moves, "
        "your revision is rejected and the draft ships unchanged.)\n"
        "- Preserve the substantive content. You MAY reorder sentences within "
        "a paragraph or move a sentence to a different paragraph if it "
        "improves flow, but do not invent claims, do not drop claims, and do "
        "not merge claims that are sourced to different documents.\n"
        "- Keep total length within plus-or-minus 10% of the input.\n"
        "- No headings, no labels, no preamble. Output ONLY the rewritten "
        "essay prose.\n\n"
        "Topic: " + topic + "\n\n"
        "Draft to revise:\n" + full_text + "\n\n"
        "Begin the revision now:"
    )

    options = {
        "temperature": 0.3,        # below writer 0.50 — we want conservative edits
        "num_ctx": 16384,          # essay + draft can hit 8-10K tokens
        "num_predict": 3500,
        "top_p": 0.9,
    }

    start = time.perf_counter()
    try:
        import ollama
        res = ollama.chat(
            model=harmonise_model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
    except Exception as e:
        stats["fallback_reason"] = f"llm_call_failed:{type(e).__name__}"
        if metrics:
            metrics.record_llm("writer_harmonise", harmonise_model, options=options,
                               success=False, error=e)
        return full_text, stats
    duration = time.perf_counter() - start
    stats["duration_s"] = round(duration, 2)
    if metrics:
        metrics.record_llm(
            "writer_harmonise", harmonise_model, options=options,
            duration_s=duration,
            prompt_chars=len(prompt),
            response_chars=len(raw),
        )

    rewritten = _strip_wrapping(raw)
    stats["len_out"] = len(rewritten)
    if not rewritten:
        stats["fallback_reason"] = "empty_response"
        return full_text, stats

    # Citation-preservation check. Default (drift_tol=0): every (doc_id, page)
    # tuple must still be present with the same multiplicity. With drift_tol>0:
    # allow up to `drift_tol` unique tuples to be dropped, but reject any new
    # tuple whose doc_id was NOT cited anywhere in the original (prevents
    # invention of new corpus references; allows the model to re-cite an
    # existing doc at an existing page in a different position).
    new_cites = sorted([(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(rewritten)])
    stats["cite_count_out"] = len(new_cites)
    orig_set = set(orig_cites)
    new_set = set(new_cites)
    missing_tuples = orig_set - new_set
    added_tuples = new_set - orig_set
    orig_docs = {c[0] for c in orig_cites}
    invented_doc_tuples = [c for c in added_tuples if c[0] not in orig_docs]
    if drift_tol == 0:
        if new_cites != orig_cites:
            stats["fallback_reason"] = (
                f"citation_drift (missing={len(missing_tuples)} added={len(added_tuples)})"
            )
            return full_text, stats
    else:
        if invented_doc_tuples:
            stats["fallback_reason"] = (
                f"invented_doc (new_docs={sorted({c[0] for c in invented_doc_tuples})[:3]} tol={drift_tol})"
            )
            return full_text, stats
        if len(missing_tuples) > drift_tol:
            stats["fallback_reason"] = (
                f"too_many_missing (missing={len(missing_tuples)} tol={drift_tol})"
            )
            return full_text, stats
        # Track diagnostic drift even when accepted.
        stats["accepted_with_drift"] = {
            "missing": len(missing_tuples), "added": len(added_tuples),
        }

    # Length check: rewrite must be within [85%, 120%] of original character length.
    lo, hi = 0.85 * stats["len_in"], 1.20 * stats["len_in"]
    if not (lo <= stats["len_out"] <= hi):
        stats["fallback_reason"] = (
            f"length_out_of_bounds (out={stats['len_out']} target=[{int(lo)}, {int(hi)}])"
        )
        return full_text, stats

    stats["applied"] = True
    stats["fallback_reason"] = "ok"
    return rewritten, stats


def _apply_per_paragraph_harmonisation(full_text: str, topic: str, metrics=None):
    """v14.2 condition A: rewrite each paragraph independently for prose flow
    while preserving every canonical citation tuple exactly within that
    paragraph. Each paragraph with >=1 canonical citation is sent in isolation
    with the original previous paragraph as context. Per-paragraph rejection
    isolates cost — a 4/12 paragraph rejection still ships 8 improved
    paragraphs, vs the full-essay pass that loses everything on a single drift.

    Returns (text, stats_dict). stats keys mirror the full-essay function but
    add paragraphs_total / paragraphs_with_cites / paragraphs_rewritten /
    paragraphs_rejected / rejection_reasons dict.
    """
    import time
    harmonise_model = os.environ.get("RRR_HARMONISE_MODEL", _MODEL)
    stats = {
        "applied": False,
        "fallback_reason": "off",
        "mode": "per_para",
        "model": harmonise_model,
        "paragraphs_total": 0,
        "paragraphs_with_cites": 0,
        "paragraphs_rewritten": 0,
        "paragraphs_rejected": 0,
        "rejection_reasons": {},
        "cite_count_in": 0,
        "cite_count_out": 0,
        "len_in": len(full_text or ""),
        "len_out": 0,
        "duration_s": 0.0,
    }
    if not full_text or os.environ.get("RRR_WRITER_HARMONISE", "1") != "1":
        return full_text, stats

    paragraphs = full_text.split("\n\n")
    stats["paragraphs_total"] = len(paragraphs)
    orig_cites_total = [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(full_text)]
    stats["cite_count_in"] = len(orig_cites_total)
    if not orig_cites_total:
        stats["fallback_reason"] = "no_canonical_cites_to_protect"
        return full_text, stats

    options = {"temperature": 0.3, "num_ctx": 8192, "num_predict": 1200, "top_p": 0.9}

    start = time.perf_counter()
    out_paras = []
    prev_para_for_context = ""
    for para in paragraphs:
        cites_in_para = sorted(
            [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(para)]
        )
        if not cites_in_para:
            out_paras.append(para)
            prev_para_for_context = para
            continue
        stats["paragraphs_with_cites"] += 1

        prev_para_block = prev_para_for_context if prev_para_for_context else \
            "[this is the opening paragraph of the review]"
        prompt = (
            "You are revising one paragraph of a literature review for prose "
            "continuity with the previous paragraph. Rewrite the paragraph so:\n\n"
            "- It FLOWS naturally from the previous paragraph (pick up a thread, "
            "contrast a finding, follow a scope condition). Do NOT open by "
            "restating the topic framing on its own.\n"
            "- Every (Doc_Year: p.N) citation token appears EXACTLY as in the "
            "original — same docs, same pages, same multiplicity. (Hard-checked; "
            "on any drift the original paragraph ships unchanged.)\n"
            "- Preserve every substantive claim. You may reorder sentences within "
            "the paragraph if it improves flow.\n"
            "- Length within plus-or-minus 20% of the input.\n"
            "- Output ONLY the rewritten paragraph; no preamble, no labels.\n\n"
            f"Topic: {topic}\n\n"
            f"Previous paragraph:\n{prev_para_block}\n\n"
            f"Paragraph to revise:\n{para}\n\n"
            "Begin the revision now:"
        )

        try:
            import ollama
            res = ollama.chat(
                model=harmonise_model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive=_KEEP_ALIVE,
                stream=False,
            )
            raw = (res.get("message", {}).get("content") or "").strip()
        except Exception as e:
            stats["rejection_reasons"]["llm_error"] = (
                stats["rejection_reasons"].get("llm_error", 0) + 1
            )
            out_paras.append(para)
            prev_para_for_context = para
            stats["paragraphs_rejected"] += 1
            if metrics:
                metrics.record_llm("writer_harmonise_para", harmonise_model,
                                   options=options, success=False, error=e)
            continue

        rewritten_para = _strip_wrapping(raw)
        if not rewritten_para:
            stats["rejection_reasons"]["empty"] = (
                stats["rejection_reasons"].get("empty", 0) + 1
            )
            out_paras.append(para)
            prev_para_for_context = para
            stats["paragraphs_rejected"] += 1
            continue

        new_cites = sorted(
            [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(rewritten_para)]
        )
        if new_cites != cites_in_para:
            stats["rejection_reasons"]["citation_drift"] = (
                stats["rejection_reasons"].get("citation_drift", 0) + 1
            )
            out_paras.append(para)
            prev_para_for_context = para
            stats["paragraphs_rejected"] += 1
            continue

        lo, hi = 0.80 * len(para), 1.20 * len(para)
        if not (lo <= len(rewritten_para) <= hi):
            stats["rejection_reasons"]["length"] = (
                stats["rejection_reasons"].get("length", 0) + 1
            )
            out_paras.append(para)
            prev_para_for_context = para
            stats["paragraphs_rejected"] += 1
            continue

        out_paras.append(rewritten_para)
        prev_para_for_context = para  # use ORIGINAL prev for next context
        stats["paragraphs_rewritten"] += 1
        if metrics:
            metrics.record_llm("writer_harmonise_para", harmonise_model,
                               options=options,
                               prompt_chars=len(prompt),
                               response_chars=len(raw))

    duration = time.perf_counter() - start
    stats["duration_s"] = round(duration, 2)

    new_text = "\n\n".join(out_paras)
    stats["len_out"] = len(new_text)
    stats["cite_count_out"] = sum(1 for _ in CITE_RE.finditer(new_text))

    if stats["paragraphs_rewritten"] > 0:
        stats["applied"] = True
        stats["fallback_reason"] = "ok"
    elif stats["paragraphs_with_cites"] == 0:
        stats["fallback_reason"] = "no_paragraphs_with_cites"
    else:
        stats["fallback_reason"] = "all_paragraphs_rejected"
    return new_text, stats


_FABRICATED_QUOTE_RE = re.compile(r'"([^"\n]{20,})"')
_NEARBY_CANONICAL_CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")


def _is_period_inside_paren(text: str, idx: int) -> bool:
    """v15.7: True if the '.' at text[idx] sits inside an unclosed '(' on the
    same line. Used by the sentence-boundary walker in _strip_fabricated_quotes
    so a period inside '(Author 2005, p.49)' is NOT treated as a sentence
    terminator — that was the v15.6 audit's broken-cite-stub bug producing
    '(Ogilvie_2007: p.' floating in prose after a quote-strip.
    """
    if idx < 0 or idx >= len(text):
        return False
    # Walk back to the previous newline or start; count parens.
    depth = 0
    i = idx - 1
    while i >= 0 and text[i] != "\n":
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                # Unclosed open-paren before this period → period is inside it.
                return True
            depth -= 1
        i -= 1
    return False


def _normalise_for_quote_match(s: str) -> str:
    """Normalise for verbatim-quote substring matching.

    Lowercases, drops quote marks (straight + smart) and ellipsis markers,
    joins OCR hyphen-line-breaks (`deteriora-tion` from the PDF extractor
    becomes `deterioration` so a modern-prose quote of `deterioration` can
    match the original page text), and collapses whitespace.

    v14.2.2: added ellipsis-strip + hyphen-join after the v14.2.1 smoke
    showed two false-positive strips — Nunn p.4 had `deteriora-tion` in
    the page text vs `deterioration` in the model output; Acemoglu p.33
    had a real first-half quote that the model truncated with `...`. Both
    are now matched correctly.
    """
    if not s:
        return ""
    s = s.lower()
    # Drop quote marks (straight + curly/smart).
    s = re.sub(r"[\"'`“”‘’«»]", "", s)
    # Drop ellipsis markers (the model often inserts `...` or `…` to mark
    # truncation inside what is otherwise a verbatim quote; the cited
    # text after the ellipsis IS on the page, but the strict substring
    # check fails because the literal `...` is not).
    s = s.replace("…", " ")
    s = re.sub(r"\.{2,}", " ", s)
    # Join OCR hyphen-line-breaks: `word-\nword`, `word- word`, and the
    # in-text `word-word` artifact where pdfminer inserted a hyphen at a
    # line-end without preserving the newline. Conservative: only join
    # when the hyphen sits between two lowercase alphabetic chars (so
    # legit compounds like `long-run`, `cross-section` survive — they
    # have lowercase on both sides too, hmm, see note). The pdfminer
    # OCR pattern is `letter-letter` mid-word; the model paraphrasing
    # writes the dehyphenated form. We rejoin to make them match.
    s = re.sub(r"([a-z])-\s*([a-z])", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_fabricated_quotes(full_text: str, allowed_docs, metrics=None,
                              display_lookup=None):
    """v14.2 fix #1: scan the assembled essay for every '"..."' span >=20 chars,
    locate the nearest canonical (doc_id: p.N) citation within +/-200 chars,
    and verify the normalised quoted text is a substring of
    data/page_text/{doc_id}_page_{N}.txt. Sentences containing an
    UNVERIFIED quote are stripped.

    The check normalises both sides (lowercase, drop quote marks, collapse
    whitespace) so OCR-spaced page text and the model's tidy prose can still
    match. Runs AFTER _display_to_canonical (so the cite form is canonical and
    doc_id is directly resolvable) and BEFORE harmonisation (so the harmoniser
    operates on cleaned input). Pre-existing E1/E2 checks operate on the
    same canonical form so the verifier shares the lookup contract.

    Returns (text, stats_dict). stats keys:
      checked_quotes, verified_real, fabricated_stripped,
      fabricated_kept_no_citation, fabricated_kept_doc_not_in_corpus,
      fabricated_kept_page_not_readable, fabrications[], duration_s.

    The `fabrications` list records (quote, doc_id, page, action) for each
    strip so the metric is auditable.

    Disabled when RRR_QUOTE_VERIFY=0 (default 1). Defensive: any unexpected
    error returns the original text unchanged with fallback_reason set.
    """
    import time
    from rrr.paths import page_text_path

    stats = {
        "enabled": True,
        "checked_quotes": 0,
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
    for qm in _FABRICATED_QUOTE_RE.finditer(full_text):
        stats["checked_quotes"] += 1
        quote = qm.group(1)
        q_start, q_end = qm.span()

        win_start = max(0, q_start - 200)
        win_end = min(len(full_text), q_end + 200)
        window = full_text[win_start:win_end]
        # v15.7: find nearest cite across ALL surfaces using parse_citations.
        # Previously CITE_RE-only — missed display-form cites entirely,
        # which after the surface flip is all of them.
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

        # Fabricated quote. Strip the SENTENCE containing it.
        # v15.7: previous boundary walk halted on '.' inside '(Doc: p.N)'
        # producing stubs like '(Ogilvie_2007: p.' floating in the prose.
        # Use a smarter walk that treats a '.' inside an unclosed '('
        # as not a sentence terminator.
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
        # v15.7: drop-aware-of-surname guard. If stripping this sentence
        # would strand a surname in the prior 120 chars (the cite that
        # gave the surname semantic cover lives in `sent`), keep the
        # sentence — better to ship a fabricated-quote sentence than a
        # confidently-wrong attribution. Record the skip so the quality
        # manifest can flag it.
        sent_text = full_text[sent_start:sent_end]
        prev_text = full_text[max(0, sent_start - 200):sent_start]
        if _drop_would_strand_surname(
            prev_text, sent_text, allowed_doc_set, display_lookup=display_lookup,
        ):
            stats.setdefault("strand_guard_kept", 0)
            stats["strand_guard_kept"] += 1
            continue
        spans_to_strip.append((sent_start, sent_end, quote, doc_id, page))

    if spans_to_strip:
        # Strip in reverse order so earlier strips do not shift later spans.
        spans_to_strip.sort(key=lambda x: -x[0])
        out = full_text
        for sent_start, sent_end, quote, doc_id, page in spans_to_strip:
            out = out[:sent_start] + out[sent_end:]
            stats["fabricated_stripped"] += 1
            stats["fabrications"].append({
                "quote": quote[:240],
                "doc_id": doc_id,
                "page": page,
                "action": "sentence_stripped",
            })
        out = re.sub(r"  +", " ", out)
        out = re.sub(r"\s+\.", ".", out)
        out = re.sub(r"\n[ \t]+\n", "\n\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
    else:
        out = full_text

    stats["duration_s"] = round(time.perf_counter() - start, 3)
    return out, stats


def _apply_cross_section_stitch(chunks, topic: str, metrics=None):
    """v11-B: rewrite the first sentence(s) of each interior section so the
    review flows from section to section instead of each restating the topic
    framing. One batched LLM call. Falls back to the original chunks on any
    parse / verification failure.

    v14.2 cond D: RRR_WRITER_STITCH_SENTENCES (default 1) controls how many
    opening sentences are rewritten together as a block. With N=3 the model
    has more room to set up a transition that does not restate the topic,
    instead of being forced to do all the work in a single sentence.

    Verification: the rewritten opener block must contain the SAME set of
    canonical citations (Doc_Year: p.N) as the original opener block; any
    addition or removal causes that block to be discarded. Only the first N
    sentences are replaced; the rest of each section is untouched.
    """
    import time
    # v13: RRR_WRITER_STITCH retired (always on). The v11-B stitch + v12
    # tail-echo guard are core to the cross-section flow story.
    # Need at least opening + 2 interior sections + closing to make stitching
    # meaningful. (Opening alone doesn't need stitching; closing already
    # synthesises across sections.)
    if not chunks or len(chunks) < 4:
        return chunks

    try:
        stitch_n = max(1, int(os.environ.get("RRR_WRITER_STITCH_SENTENCES", "1")))
    except ValueError:
        stitch_n = 1

    interior_indices = list(range(1, len(chunks) - 1))
    extracts = []
    for i in interior_indices:
        sents = _split_sentences_for_cleanup(chunks[i])
        if not sents:
            continue
        opener_sents = sents[:stitch_n]
        opener = " ".join(s.strip() for s in opener_sents).strip()
        prev_sents = _split_sentences_for_cleanup(chunks[i - 1])
        prev_tail = prev_sents[-1].strip() if prev_sents else ""
        if not opener:
            continue
        extracts.append({
            "index": i,
            "opener": opener,
            "opener_sent_count": len(opener_sents),
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
        f"each OPENING ({opener_unit}) so it flows from the previous "
        "section's last sentence — pick up a thread, contrast a finding, "
        "follow a scope condition — rather than re-introducing the topic.",
        "",
        "STRICT RULES:",
        "- Keep every (Doc_Year: p.N) citation token UNCHANGED.",
        "- Do NOT add or remove any citation.",
        "- Do NOT exceed the original block length by more than ~20%.",
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
        # v13: RRR_WRITER_STITCH_T/CTX/PRED retired; tuning frozen.
        res = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 1500},
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
    skipped_topic_paraphrase = 0
    skipped_tail_echo = 0
    interior_set = set(interior_indices)
    prev_tail_by_index = {e["index"]: e["prev_tail"] for e in extracts}
    cite_token_re = re.compile(r"\([A-Za-z0-9_&.\-]+:\s*p\.\d+\)")

    # v11.1: topic-paraphrase guard. Compute the topic's content-token set once,
    # then reject any stitch rewrite that pulls back to a near-restatement of
    # the topic line. Closes the v11 failure on section 4 where the LLM
    # produced "The question of whether institutions are indeed the fundamental
    # cause of long-run economic growth remains a pivotal area of inquiry..."
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
    # Threshold: if the rewrite shares >= 2/3 of the topic's content tokens, it's
    # a paraphrase. Floor at 4 so a 5-word topic doesn't trip on coincidence.
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
        sents = _split_sentences_for_cleanup(new_chunks[idx])
        if not sents:
            continue
        n_open = opener_sent_count_by_index.get(idx, 1)
        n_open = min(n_open, len(sents))
        orig_opener_block = " ".join(s.strip() for s in sents[:n_open])
        orig_cites = sorted(cite_token_re.findall(orig_opener_block))
        new_cites = sorted(cite_token_re.findall(rewritten))
        if orig_cites != new_cites:
            skipped_citation += 1
            continue
        # v11.1: reject if the rewrite paraphrases the topic line.
        rewrite_tokens = _content_tokens(rewritten)
        overlap = len(rewrite_tokens & topic_tokens)
        if topic_tokens and overlap >= paraphrase_threshold:
            skipped_topic_paraphrase += 1
            continue
        # v12: reject if the rewrite is an echo of the previous section's tail.
        # The original failure: the LLM "flows from prev_tail" by literally
        # copying it as the new opener, producing visible duplicate sentences
        # at every section boundary. Same shape as the topic-paraphrase guard
        # but referenced against prev_tail.
        prev_tail = prev_tail_by_index.get(idx, "")
        if prev_tail:
            prev_tail_tokens = _content_tokens(prev_tail)
            tail_overlap = len(rewrite_tokens & prev_tail_tokens)
            # Threshold: shared >= 6 content tokens, OR rewrite tokens are a
            # strict subset of prev_tail tokens with at least 4 shared. The
            # subset check catches paraphrases that change a few words.
            if (tail_overlap >= 6) or (
                tail_overlap >= 4 and rewrite_tokens and rewrite_tokens.issubset(prev_tail_tokens)
            ):
                skipped_tail_echo += 1
                continue
        new_chunks[idx] = " ".join([rewritten] + [s.strip() for s in sents[n_open:]])
        applied += 1

    if metrics:
        metrics.set("writer_stitch_applied", applied)
        metrics.set("writer_stitch_skipped_citation_change", skipped_citation)
        metrics.set("writer_stitch_skipped_empty", skipped_empty)
        metrics.set("writer_stitch_skipped_topic_paraphrase", skipped_topic_paraphrase)
        metrics.set("writer_stitch_skipped_tail_echo", skipped_tail_echo)
    if applied or skipped_topic_paraphrase or skipped_tail_echo:
        print(f"[Writer] Cross-section stitch: rewrote {applied} section "
              f"opener(s) (skipped {skipped_citation} citation drift, "
              f"{skipped_topic_paraphrase} topic paraphrase, "
              f"{skipped_tail_echo} tail echo)")
    return new_chunks


# v13: _build_author_year_lookup promoted to render.py; we import it via the
# top-of-file import block. The writer-side _collect_cited_docs extends the
# shared one with display-form scans (DISPLAY_CITE_RE / DISPLAY_PAREN_CITE_RE)
# that the reasoner-side caller doesn't need.

def _collect_cited_docs(text: str, allowed_docs, author_year_to_docid):
    """Writer-side collector: shared canonical / bare-canonical / legacy
    author-year scans (via render._collect_cited_docs) PLUS the v10 display
    surfaces. The display branch lives here because reasoner's fallback path
    doesn't emit display forms.
    """
    cited_docs = _shared_collect_cited_docs(text, allowed_docs, author_year_to_docid)
    display_lookup = _build_display_lookup(allowed_docs)
    for m in DISPLAY_CITE_RE.finditer(text or ""):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)
    for m in DISPLAY_PAREN_CITE_RE.finditer(text or ""):
        key = (m.group(1).strip().lower(), m.group(2))
        did = display_lookup.get(key)
        if did:
            cited_docs.add(did)
    return cited_docs


def compose_from_ledger(ledger_path=None, metrics=None):
    ledger_path = ledger_path or str(runs_path("review_ledger.json"))
    if not os.path.isfile(ledger_path):
        raise SystemExit(f"Ledger not found: {ledger_path}")

    with open(ledger_path, encoding="utf-8") as f:
        data = json.load(f)

    topic_display = data.get("topic", "(no topic)")
    # v12: writer prompts use the reformulated topic question. Falls back to
    # raw topic when the planner didn't (or couldn't) reformulate.
    topic = (data.get("plan", {}) or {}).get("topic_question") or topic_display
    topic_dimensions = (data.get("plan", {}) or {}).get("topic_dimensions") or []
    if topic != topic_display:
        print(f"[Writer] using reformulated topic: '{topic}'")
        if topic_dimensions:
            print(f"[Writer] topic dimensions: {topic_dimensions}")
    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise SystemExit("Ledger empty or malformed (no docs).")

    # v15.9 (#6): provenance data flowed in through the ledger from the
    # reasoner (which read it from metadata.csv). Absent in older ledgers →
    # citations.json still emits (with pdf_path=null) so downstream tools
    # can detect the missing-provenance case.
    pdf_paths_by_docid = data.get("pdf_paths_by_docid", {}) or {}
    pdf_page_offsets = data.get("pdf_page_offsets", {}) or {}
    dois_by_docid = data.get("dois_by_docid", {}) or {}

    allowed_pairs, allowed_docs, allowed_pages_by_doc = _build_allowed_citations(docs)
    if not allowed_pairs:
        raise SystemExit("No allowed citations found in ledger.")

    author_year_to_docid = _build_author_year_lookup(allowed_docs)
    evidence_id_map = _build_evidence_id_map(docs)

    # v15: read the corpus-level OUTLINE PLAN instead of stance-bucketed
    # cluster_syntheses. Each cluster is top-level (no nested-under-stance
    # structure), has a relation tag + free-text elaboration, and a
    # pre-computed section order.
    outline_plan = data.get("outline_plan") or {}
    if not outline_plan or not outline_plan.get("clusters"):
        raise SystemExit("Ledger missing outline_plan or outline produced no clusters.")
    topic_shape = outline_plan.get("topic_shape") or "causal"
    clusters_by_id = {c["cluster_id"]: c for c in outline_plan.get("clusters", [])}
    ordered_cluster_ids = (
        outline_plan.get("ordered_cluster_ids")
        or list(clusters_by_id.keys())
    )

    # Reverse lookup: doc_id -> evidence_ids (used by outline blocks for
    # multi-citation parentheticals).
    doc_to_evidence_ids = defaultdict(list)
    for eid, ev in evidence_id_map.items():
        did = (ev.get("doc_id") or "").strip()
        if did:
            doc_to_evidence_ids[did].append(eid)
    for did in list(doc_to_evidence_ids.keys()):
        doc_to_evidence_ids[did] = sorted(doc_to_evidence_ids[did])

    docs_by_id = {d.get("doc_id"): d for d in docs}

    # Relation distribution summary for the opening prompt — describes the
    # corpus's structural shape without any supports/critiques vocabulary.
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
    print(f"[Writer] topic_shape={topic_shape} clusters={len(clusters_by_id)} "
          f"relations={relation_dist}")

    # Build chunk_plan in the cluster order Stage 3 picked. Each entry uses
    # the single v15 stream-section builder, partial-applied with the
    # cluster's relation + topic_shape.
    # v15.7: compute lead author surname per cluster from outline_plan's
    # lead_doc_id (Stage 2 field). Each cluster's prompt then knows which
    # surname to OPEN with and which surnames are reserved by OTHER
    # clusters' openers — closing the parallel-writer cloned-opener gap.
    from functools import partial
    cluster_lead_surnames: dict = {}
    for cid in ordered_cluster_ids:
        c = clusters_by_id.get(cid)
        if not c:
            continue
        lead_did = (c.get("lead_doc_id") or "").strip()
        if not lead_did:
            # Fall back to the first doc_id in the cluster.
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
        outline_block = _format_outline_block(c, doc_to_evidence_ids)
        this_lead = cluster_lead_surnames.get(cid, "")
        forbidden = sorted(s for s in all_lead_surnames if s and s != this_lead)
        builder = partial(_build_stream_prompt,
                          relation=c.get("relation", ""),
                          topic_shape=topic_shape,
                          lead_surname=this_lead,
                          forbidden_opener_surnames=forbidden)
        # Reuse the legacy 5-tuple shape so the dispatcher loop below works
        # unchanged. Position 0 ("stance") now carries the relation; position
        # 1 ("cluster") carries the cluster's shared_thread label.
        chunk_plan.append((
            c.get("relation", "stream"),
            c.get("shared_thread") or cid,
            cluster_docs,
            builder,
            outline_block,
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
    # v8 (R5): observability for new postproc surfaces
    total_double_paren_collapsed = 0
    total_author_led_openings = 0
    # v14 FIX-BRACKET: count [Doc_Year] bracket-id rewrites across all chunks
    # and the final assembly pass.
    total_bracket_id_rewrites = 0
    # v15.7: track renderer-side attribution gate + unknown-eid signal across
    # chunks so the quality manifest and finalize_covered_chunk can act on them.
    total_unknown_eids = 0
    total_attribution_mismatches = 0
    total_attribution_retries = 0
    unknown_eid_snippets: list = []
    attribution_mismatch_snippets: list = []
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
        nonlocal total_removed_citations, total_style_removed, total_evidence_id_renders
        nonlocal total_double_paren_collapsed, total_author_led_openings
        nonlocal total_bracket_id_rewrites
        nonlocal total_unknown_eids, total_attribution_mismatches

        chunk = _strip_wrapping(chunk)
        # v15.5: the writer now produces ONLY [E####] evidence-ID markers.
        # The context-aware renderer below converts each marker to either
        # the narrative '(Year, p.N)' (when the author is already named in
        # prose) or the parenthetical '(Author Year, p.N)' (otherwise).
        # This eliminates the display->canonical->display surface-cycling
        # the older pipeline needed.
        # v15.7: renderer now also acts as attribution gate — returns stats
        # dict with attribution_mismatches that finalize_covered_chunk uses
        # to decide whether to trigger a coverage retry.
        chunk, render_stats = _render_evidence_id_citations(chunk, evidence_id_map)
        total_evidence_id_renders += render_stats["replacements"]
        total_unknown_eids += render_stats["unknown_eids"]
        unknown_eid_snippets.extend(render_stats["unknown_eid_snippets"])
        total_attribution_mismatches += render_stats["attribution_mismatches"]
        attribution_mismatch_snippets.extend(render_stats["attribution_mismatch_snippets"])

        # v14 FIX-BRACKET: catch [Doc_Year] bracketed canonical doc_ids the
        # model invents when it confuses bracket-evidence-id syntax with the
        # canonical doc_id.
        chunk, bracket_rewrites = _render_bracketed_doc_ids(
            chunk, allowed_docs, doc_to_evidence_ids,
        )
        total_bracket_id_rewrites += bracket_rewrites

        # v15.7: _chunk_display_to_canonical retired. The validators
        # (_audit_section_coverage, _remove_invalid_citations,
        # _drop_cross_section_redundancy, _strict_cited_doc_ids) now
        # consume display-form citations natively via parse_citations +
        # display_lookup. The canonical surface is no longer used as an
        # internal validation form. The user-facing output is display
        # form '(Author Year, p.N)' / 'Author (Year, p.N)' — no more
        # display→canonical→display surface-cycling.
        # v15.7: wire _collapse_double_parens into the chunked path. v15.6
        # reported writer_double_paren_collapsed=0 in every run because
        # the only call site was in the dead single-pass code; the audit
        # found 50 ((...)) wrappings shipped to users.
        chunk, dp_collapsed = _collapse_double_parens(chunk)
        total_double_paren_collapsed += dp_collapsed
        # v8 (R5): observe author-led-opening violations and count them.
        # v15.7: was being called TWICE (the comment was duplicated and so was
        # the call); single invocation now matches every other observation
        # point in postprocess_chunk.
        total_author_led_openings += _count_author_led_openings(chunk)

        # v13: removed the legacy pre-cleanup arms (_strip_placeholder_citations,
        # _fix_ajr_abbreviation, _repair_year_only_citations, _extract_citation_dumps).
        # All four had 0-hit metrics on the v12 smoke and the v13 prompt surface
        # no longer produces any of the shapes they targeted.

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

        chunk, style_removed = _remove_style_violations(chunk)
        total_style_removed += len(style_removed)

        return chunk, 0, 0, 0, len(style_removed), render_stats

    def finalize_covered_chunk(raw, prompt, chunk_docs, section_kind, stage):
        nonlocal total_coverage_fallbacks, total_attribution_retries
        chunk, repairs, placeholders, ajr, style_removed, render_stats = postprocess_chunk(raw, chunk_docs)
        audit = _audit_section_coverage(chunk, chunk_docs, section_kind)

        # v15.7: renderer-as-attribution-gate triggers a coverage retry when
        # the renderer detected a prose-surname-vs-eid-author mismatch. The
        # gate ALREADY fixed the cite shape (emits '(CorrectAuthor Year, p.N)'
        # instead of bare year), but the prose still names the wrong author.
        # A retry gives the writer a fresh attempt with the original prompt;
        # if the retry still mismatches, we accept the self-healed render and
        # surface the mismatch in the quality manifest.
        attribution_retry_done = False
        if (
            _writer_enforcement_enabled()
            and render_stats["attribution_mismatches"] > 0
        ):
            if metrics:
                metrics.inc("writer_attribution_retries")
            total_attribution_retries += 1
            attribution_retry_done = True
            retry_prompt = _coverage_retry_prompt(
                prompt, chunk, audit["required_cited_docs"],
            )
            raw_retry = _ollama_chat(
                retry_prompt, metrics=metrics, stage=f"{stage}_attribution_retry",
            )
            chunk_retry, repairs_r, placeholders_r, ajr_r, style_removed_r, render_stats_retry = postprocess_chunk(
                raw_retry, chunk_docs,
            )
            audit_retry = _audit_section_coverage(chunk_retry, chunk_docs, section_kind)
            # Only accept the retry if it (a) clears the audit, AND (b) has
            # fewer attribution mismatches than the original. Otherwise keep
            # the original (already self-healed by the renderer).
            retry_better = (
                audit_retry["ok"]
                and render_stats_retry["attribution_mismatches"]
                < render_stats["attribution_mismatches"]
            )
            if retry_better:
                chunk = chunk_retry
                audit = audit_retry
                repairs += repairs_r
                placeholders += placeholders_r
                ajr += ajr_r
                style_removed += style_removed_r
                if metrics:
                    metrics.inc("writer_attribution_retry_accepted")
            else:
                if metrics:
                    metrics.inc("writer_attribution_retry_rejected")

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

        # Deterministic fallback couldn't reach the required doc count.
        # Escalate to LLM coverage retry as the last resort — but skip it if
        # we already did the attribution retry on this section, to avoid
        # paying two LLM retries.
        if not attribution_retry_done:
            if metrics:
                metrics.inc("writer_section_coverage_retries")
            retry_prompt = _coverage_retry_prompt(prompt, chunk, audit["required_cited_docs"])
            raw = _ollama_chat(retry_prompt, metrics=metrics, stage=f"{stage}_coverage_retry")
            chunk, repairs2, placeholders2, ajr2, style_removed2, _retry_stats = postprocess_chunk(raw, chunk_docs)
            audit = _audit_section_coverage(chunk, chunk_docs, section_kind)
        else:
            repairs2 = placeholders2 = ajr2 = 0
            style_removed2 = 0
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

    # v15: opening docs draw the top-2 by score from EACH cluster in the
    # chunk plan, then the top 6 overall. This keeps the opening's evidence
    # diversified across streams without referencing any stance bucket.
    opening_docs = []
    for relation, cluster_label, cluster_docs, _builder, _block in chunk_plan:
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
            # v15.4.0 (Bug 4 half-fix): extract author surnames the chunk
            # named in prose, so the next section's claims_so_far block can
            # forbid re-introducing them.
            "surnames_named": _extract_named_surnames(chunk),
        })
        chunks.append(chunk)

    def log_stance_chunk(job, word_count, repairs, placeholders, ajr, style_removed, audit):
        # v13: repairs / placeholders / ajr are vestigial (the helpers that
        # produced them were retired); keep the signature for caller stability
        # but only log the live signals (style sentences removed, citation
        # count). The unused parameters are accepted to keep the call sites
        # untouched.
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
        # v11.2 lever 3: when sequential, build the accumulated claims summary
        # right before each section's prompt so it sees what's already been
        # argued. Empty for the first stance section (only the opening exists),
        # then grows as each section completes.
        print(f"[Writer] Sequential stance chunks: claims-so-far context enabled")
        for i, (stance, cluster, cluster_docs, prompt_builder, synthesis_block) in enumerate(chunk_plan):
            cluster_docs_sorted = sorted(cluster_docs, key=_score_doc, reverse=True)[:6]
            allowed_list = _list_allowed_citations(cluster_docs_sorted, allowed_pages_by_doc)
            evidence = "\n\n".join(_format_doc_entry(d) for d in cluster_docs_sorted)
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
    # v15.7: final-assembly renderer is a safety net (per-chunk pass already
    # ran). Any new mismatches detected here are surfaced in the quality
    # manifest but cannot retry — at this point the writer LLM has finished.
    full_text, final_render_stats = _render_evidence_id_citations(full_text, evidence_id_map)
    total_evidence_id_renders += final_render_stats["replacements"]
    total_unknown_eids += final_render_stats["unknown_eids"]
    unknown_eid_snippets.extend(final_render_stats["unknown_eid_snippets"])
    total_attribution_mismatches += final_render_stats["attribution_mismatches"]
    attribution_mismatch_snippets.extend(final_render_stats["attribution_mismatch_snippets"])

    # v14 FIX-BRACKET: final-assembly bracket-id rewrite for [Doc_Year] residue
    # that survived per-chunk postprocess (rare but possible if a chunk join
    # introduced new bracketed-doc-id text in stitch fallback paths).
    full_text, final_bracket_rewrites = _render_bracketed_doc_ids(
        full_text, allowed_docs, doc_to_evidence_ids,
    )
    total_bracket_id_rewrites += final_bracket_rewrites

    # v15.7: _display_to_canonical retired at final assembly. The user-facing
    # surface is now display ('Author (Year, p.N)' / '(Author Year, p.N)')
    # end-to-end. Downstream validators consume display surfaces natively via
    # parse_citations + display_lookup. The display_lookup itself is still
    # built here because validators want it for doc_id resolution.
    display_lookup = _build_display_lookup(allowed_docs)

    # v14.2 fix #1: quote verification. Scan every '"..."' span >=20 chars,
    # locate the nearest canonical (doc_id: p.N) citation, and verify the
    # quoted text against the actual page text. Sentences containing an
    # UNVERIFIED quote are stripped. Runs AFTER display->canonical (so the
    # cite form is canonical) and BEFORE harmonisation (so the harmoniser
    # sees a clean draft). The v14.2 investigation found 2 fabrications in
    # the 12-run smoke; the architecture's pre-fix validators only checked
    # citation tuples and never inspected quoted prose. Toggle via
    # RRR_QUOTE_VERIFY=0 to disable (default 1).
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

    # v15.7: harmonisation retired. The v14.1/v14.2 full-essay and
    # per-paragraph rewriters were rejected in 4/4 v15.6 smoke runs as
    # 'too_many_missing' (drift_tolerance=1 against 8-15 actual missing
    # tuples). Cost: ~75s/run of LLM time for zero applied output. The
    # drift-check used CITE_RE only, so it was structurally blind to
    # display-form citations the harmoniser might drop. The v15.7 approach
    # is to fix attribution and surface coherence upstream (renderer-as-
    # attribution-gate + cleaner ALLOWED CITATIONS block + drop-aware
    # postproc) rather than try to repair the assembled essay post-hoc.
    # _apply_full_essay_harmonisation / _apply_per_paragraph_harmonisation
    # and the RRR_HARMONISE_* env knobs remain in the module as dead code
    # for diagnostic continuity; no call site in the production path.

    # v13: removed the legacy final-assembly arms (_repair_year_only_citations,
    # _fix_ajr_abbreviation, _strip_placeholder_citations, _extract_citation_dumps,
    # _normalize_citation_case). All five had 0-hit metrics on the v12 smoke
    # and target shapes the v13 prompt+display surface no longer produces.

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

    full_text, final_style_removed = _remove_style_violations(full_text)
    total_style_removed += len(final_style_removed)

    # v9 (R6): final-assembly cross-section redundancy drop. Conservative —
    # only fires when a sentence's citations are a strict subset of those
    # already cited AND it has substantial content overlap with an earlier
    # sentence. Configurable via env (set 0 to disable, or raise the overlap
    # threshold to be even more conservative).
    # v13: RRR_WRITER_DROP_REDUNDANCY and RRR_WRITER_REDUNDANCY_OVERLAP retired.
    # The v9 R6 safety-net dedupe is conservative (fires 0-6 times per smoke)
    # and disabling it has no production justification.
    full_text, redundancy_drops = _drop_cross_section_redundancy(
        full_text, min_token_overlap=4, display_lookup=display_lookup,
        allowed_docs=allowed_docs,
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

    # v12: drop sentences that talk about the review itself ("the literature
    # reviewed here converges...", "this review will examine..."). Runs after
    # the structural strips and before the citation surface rewrite so the
    # detector sees the canonical form, not the display form.
    full_text, meta_removed = _strip_meta_commentary(full_text)
    if meta_removed and metrics:
        metrics.inc("writer_meta_commentary_stripped", len(meta_removed))
    if meta_removed:
        print(f"[Writer] Stripped {len(meta_removed)} meta-commentary sentence(s):")
        for s in meta_removed[:3]:
            print(f"         - {s}")

    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # v15.5: deprecated citation-surface postproc chain removed from the
    # main pipeline. The writer now emits ONLY [E####] markers (rendered
    # context-aware in postprocess_chunk), so the assorted passes below
    # had nothing to fix. Functions preserved in
    # rrr.legacy_citation_postproc for cross-model fallback testing:
    #   - rewrite_canonical_to_display
    #   - rewrite_misplaced_narrative_citations
    #   - collapse_nested_narrative_multicite
    #   - fix_orphan_outer_paren_narrative
    #   - merge_redundant_inline_paren_citations
    #   - _strip_author_year_placeholder
    #   - _fix_mid_sentence_narrative_cites
    #   - _strip_malformed_author_label_commas
    #   - _collapse_same_author_year_paren_cites

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

    # v15.5: placeholder strip retired — the writer no longer copies the
    # "(Author, Year)" exemplar because the prompt no longer SHOWS that
    # exemplar (replaced with "[E####]" usage examples).

    # v13: hard rule-9 enforcement at final assembly. The v12 prose audit found
    # three zero-citation paragraphs (paras 4, 9, 12 — the closing) that the
    # prompt-level rule-9 directive failed to prevent. The prompt directive is
    # the right place to ASK for multi-source paragraphs; this is the safety
    # net that drops the worst case (zero citations) when the model still
    # produces filler / bridging / methodology paragraphs. Conservative: only
    # drops a paragraph when no citation surface of any kind is present.
    # v15.7: strip coverage-fallback patch sentences when they are no longer
    # load-bearing for paragraph coverage. v15.6 run 04 shipped the first
    # line of the review as a verbatim 'A further source records ...' patch.
    # Sentinel-tagged at fallback time; dropped here unless paragraph would
    # become uncited otherwise.
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

    # v15.7: pass keep_closing=True at final assembly. Without it, a closing
    # paragraph whose only cited sentence was upstream-stripped would be
    # silently deleted along with the rest of the structurally important
    # synthesis paragraph. The per-chunk variant already uses keep_closing;
    # this aligns final assembly with the same contract.
    full_text, zero_cite_dropped, zc_kept_closing = _drop_zero_citation_paragraphs(
        full_text, keep_closing=True
    )
    if zero_cite_dropped and metrics:
        metrics.inc("writer_zero_cite_paragraphs_dropped", len(zero_cite_dropped))
        metrics.set("writer_zero_cite_paragraphs", zero_cite_dropped[:5])
    if zc_kept_closing and metrics:
        metrics.inc("writer_zero_cite_closing_kept")
        metrics.set("writer_zero_cite_closing_kept_snippet", zc_kept_closing)
    if zero_cite_dropped:
        print(f"[Writer] Dropped {len(zero_cite_dropped)} zero-citation paragraph(s):")
        for snippet in zero_cite_dropped[:3]:
            print(f"         - {snippet}")
    if zc_kept_closing:
        print(f"[Writer] Kept weak closing (no recognised cite): {zc_kept_closing}")

    cited_docs = _collect_cited_docs(full_text, allowed_docs, author_year_to_docid)
    # v13: all_dump_citations was previously a side-channel from
    # _extract_citation_dumps. With that helper retired (citation_dump_docs=0
    # on every recent smoke), cited_docs is derived purely from the rendered
    # full_text — no side channel needed.
    cited_docids = sorted(cited_docs)

    # v15.7.2: merge whitespace-adjacent paren cites into one semicolon-
    # joined parenthetical. Runs AFTER _collect_cited_docs so ref-list
    # counting is unaffected — the merged form is the user-facing surface
    # only. '(A 2007, p.9) (B 1989, p.6)' → '(A 2007, p.9; B 1989, p.6)'.
    full_text, adjacent_paren_merges = _merge_adjacent_paren_cites(full_text)
    if metrics:
        metrics.set("writer_adjacent_paren_merges", adjacent_paren_merges)
    if adjacent_paren_merges:
        print(f"[Writer] Merged {adjacent_paren_merges} adjacent paren-cite pair(s).")

    # v15.9 (#6): build citations.json provenance manifest AND, if
    # RRR_LINKIFY=1, rewrite in-text citations as markdown links pointing at
    # the source PDF at the right page. Runs BEFORE the final write so the
    # markdown that lands on disk is the linkified version when opted in.
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

    # v15.7: quality_manifest.json — surfaces every observable quality
    # signal the writer produced, plus snippets where applicable, so an
    # operator (or the 9-battery harness) can grade a review without
    # re-running the audit pipeline. NOT a runtime gate — the precheck
    # decision stays one-shot at Stage 0 — but a transparent record of
    # what the writer detected and self-healed.
    quality_manifest = {
        "schema_version": "v15.7",
        "word_count": total_words,
        "distinct_docs_cited": len(cited_docids),
        "chunks_written": len(chunks),
        # Attribution gate (renderer-as-gate; coverage retry on mismatch)
        "attribution_mismatches": total_attribution_mismatches,
        "attribution_retries": total_attribution_retries,
        "attribution_mismatch_snippets": attribution_mismatch_snippets[:20],
        # Evidence-id integrity
        "unknown_evidence_ids": total_unknown_eids,
        "unknown_evidence_id_snippets": unknown_eid_snippets[:10],
        # Surface coherence (display-form leaks should now be 0 with the
        # ALLOWED CITATIONS prompt fix; tracked here to detect regressions)
        "invalid_citations_removed": total_removed_citations,
        "double_paren_collapsed": total_double_paren_collapsed,
        "adjacent_paren_merges": adjacent_paren_merges,
        # Drop validators
        "zero_cite_paragraphs_dropped": len(zero_cite_dropped),
        "zero_cite_closing_kept": 1 if zc_kept_closing else 0,
        "redundancy_drops": len(redundancy_drops),
        "style_sentences_removed": total_style_removed,
        # Coverage fallback
        "coverage_fallbacks": total_coverage_fallbacks,
        "coverage_patches_dropped_at_final": n_patches_dropped,
        "coverage_patches_shipped_to_user": n_patches_kept,
        # Quote verification
        "quote_verify": {
            "checked": quote_verify_stats.get("checked_quotes", 0),
            "verified_real": quote_verify_stats.get("verified_real", 0),
            "fabricated_stripped": quote_verify_stats.get("fabricated_stripped", 0),
            "strand_guard_kept": quote_verify_stats.get("strand_guard_kept", 0),
        },
        # Outline (carried up from Stage 0/1/2 for one-stop quality view)
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
        f"double_paren_collapsed={total_double_paren_collapsed} author_led_openings={total_author_led_openings} "
        f"redundancy_drops={len(redundancy_drops)} bracket_id_rewrites={total_bracket_id_rewrites} "
        f"unknown_eids={total_unknown_eids} attribution_mismatches={total_attribution_mismatches} "
        f"attribution_retries={total_attribution_retries}"
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
            "double_paren_collapsed": total_double_paren_collapsed,
            "author_led_openings": total_author_led_openings,
            "redundancy_drops": len(redundancy_drops),
            "bracket_id_rewrites": total_bracket_id_rewrites,
            "unknown_eids": total_unknown_eids,
            "attribution_mismatches": total_attribution_mismatches,
            "attribution_retries": total_attribution_retries,
            "section_claims": section_claims,
            "section_coverage": section_coverage,
        })
        metrics.set("writer_mode", "chunked")
        metrics.inc("writer_removed_citations", total_removed_citations)
        metrics.inc("writer_style_sentences_removed", total_style_removed)
        metrics.inc("writer_evidence_id_renders", total_evidence_id_renders)
        metrics.inc("writer_double_paren_collapsed", total_double_paren_collapsed)
        metrics.inc("writer_author_led_openings", total_author_led_openings)
        # v15.7: always emit attribution counters (even at 0) so the 9-battery
        # can distinguish "fired with zero hits" from "never fired".
        metrics.set("writer_unknown_evidence_id", total_unknown_eids)
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


# =============================================================================
# v14 FIX-C: Option C — single-pass writer.
#
# Empirically validated via the run_07 ledger smoke against mistral-small:24b:
# a single LLM call, given (a) the reformulated topic + dimensions, (b) the
# ALLOWED CITATIONS list, (c) the cluster syntheses, and (d) per-doc evidence
# capped at 2 quotes/doc, produces a coherent literature review. The reference
# probe lives at scratchpad/single_pass_writer_probe.py; build_single_pass_prompt
# below mirrors its build_single_pass_prompt() exactly, and postprocess mirrors
# its postprocess() with one v14 ordering tweak (keep_closing in
# _drop_zero_citation_paragraphs).
#
# The chunked writer (compose_from_ledger above) is preserved as the
# RRR_WRITER_MODE=chunked fallback for measurement and v13.1.1 comparison.
# =============================================================================


def _build_single_pass_user_prompt(data, allowed_pairs, allowed_docs,
                                   allowed_pages_by_doc, evidence_id_map,
                                   doc_to_evidence_ids, quotes_per_doc,
                                   target_words):
    """Assemble the user-message body for the single-pass writer call.

    Mirrors scratchpad/single_pass_writer_probe.py::build_single_pass_prompt.
    Returns the user prompt string only — the system message stays
    _SYSTEM_CITATION_INSTRUCTION exactly like the chunked path.
    """
    topic_display = data.get("topic", "(no topic)")
    plan = data.get("plan") or {}
    topic = (plan.get("topic_question") or topic_display).strip()
    topic_dimensions = plan.get("topic_dimensions") or []

    docs = [d for d in data.get("docs", []) if d.get("doc_id")]
    cluster_syntheses_raw = data.get("cluster_syntheses", {}) or {}

    # Stance distribution one-liner — same shape the chunked path prints.
    stance_counts = Counter()
    for d in docs:
        stance_counts[d.get("stance", "tangential")] += 1
    stance_summary = (
        f"Of {len(docs)} sources, "
        f"{stance_counts.get('supports', 0)} support the thesis, "
        f"{stance_counts.get('critiques', 0)} offer critiques, and "
        f"{stance_counts.get('complicates', 0)} add nuance or qualifications."
    )

    # Cluster syntheses block (ordered supports -> critiques -> complicates).
    synth_blocks = []
    for stance_order in ("supports", "critiques", "complicates"):
        for key, synth in cluster_syntheses_raw.items():
            if "::" not in key or not isinstance(synth, dict):
                continue
            stance_part, _, cluster_part = key.partition("::")
            if stance_part != stance_order:
                continue
            shared = (synth.get("shared_mechanism") or "").strip()
            contested = (synth.get("contested_dimension") or "").strip()
            if not shared:
                continue

            def _eids(doc_ids):
                out = []
                for did in doc_ids or []:
                    ids = doc_to_evidence_ids.get(did) or []
                    if ids:
                        out.append(ids[0])
                return out

            sup_eids = _eids(synth.get("supporting_doc_ids", []))
            qual_eids = _eids(synth.get("qualifying_doc_ids", []))
            lines = [f"CLUSTER [{stance_part.upper()} / {cluster_part}]",
                     f"  Shared mechanism: {shared}"]
            if sup_eids:
                cite_block = "; ".join(f"[{e}]" for e in sup_eids[:5])
                lines.append(
                    f"  Supporting sources (cite together): ({cite_block})"
                )
            if qual_eids:
                qblock = "; ".join(f"[{e}]" for e in qual_eids[:3])
                lines.append(
                    f"  Qualifying sources (cite as follow-up): ({qblock})"
                )
            if contested:
                lines.append(f"  Internal disagreement: {contested}")
            synth_blocks.append("\n".join(lines))

    # Per-doc evidence, grouped by stance for readability. Cap quotes via
    # RRR_WRITER_QUOTES_PER_DOC (default 2) — same env var as the chunked path.
    qpd = int(os.environ.get("RRR_WRITER_QUOTES_PER_DOC", str(quotes_per_doc)))
    # Snapshot env around _format_doc_entry which reads the env var directly.
    prev_qpd = os.environ.get("RRR_WRITER_QUOTES_PER_DOC")
    os.environ["RRR_WRITER_QUOTES_PER_DOC"] = str(max(1, qpd))
    try:
        doc_blocks = []
        for stance_order in ("supports", "critiques", "complicates", "tangential"):
            stance_docs = [d for d in docs if d.get("stance") == stance_order]
            for d in stance_docs:
                doc_blocks.append(_format_doc_entry(d))
    finally:
        if prev_qpd is None:
            os.environ.pop("RRR_WRITER_QUOTES_PER_DOC", None)
        else:
            os.environ["RRR_WRITER_QUOTES_PER_DOC"] = prev_qpd

    allowed_list = _list_allowed_citations(docs, allowed_pages_by_doc)

    parts = [f"Literature review topic: {topic}"]
    if topic_display != topic:
        parts.append(f"Display topic on the rendered output: {topic_display}")
    if topic_dimensions:
        parts.append(
            "Dimensions of disagreement to weave in: " + "; ".join(topic_dimensions)
        )

    parts.append("")
    parts.append(
        "ALLOWED CITATIONS (use these surfaces verbatim; every claim needs an author+page):"
    )
    parts.append(allowed_list)
    parts.append("")

    parts.append(stance_summary)
    parts.append("")

    if synth_blocks:
        parts.append(
            "CLUSTER SYNTHESES (these are the SHARED mechanisms across sources; "
            "when stating a shared claim, cite the listed evidence IDs together "
            "in ONE parenthetical):"
        )
        parts.append("\n\n".join(synth_blocks))
        parts.append("")

    parts.append(
        "EVIDENCE PER DOCUMENT (each block: doc_id, stance, lead mechanism, "
        "rationale, contested dimension, quotes):"
    )
    parts.append("\n\n".join(doc_blocks))
    parts.append("")

    parts.append(_PROSE_DIRECTIVE)
    parts.append("")

    low = max(400, target_words - 100)
    high = target_words + 200
    parts.append(
        f"TASK: write the complete literature review in ONE pass, {low}-{high} words. "
        "Structure as a CONTINUOUS essay (NOT as labelled sections). The shape is:\n"
        "  1. Open by stating the substantive question and what is at stake "
        "intellectually. Name the live disagreement: what positions exist, "
        "where they conflict, without listing them as a survey. Cite at least "
        "2 distinct sources here.\n"
        "  2. Develop the supporting argument through the SUPPORTS cluster "
        "mechanism(s). Lead with the shared mechanism using the cluster's "
        "multi-citation parenthetical; ground with a quoted phrase from one "
        "source; show how another source confirms it under different conditions.\n"
        "  3. Develop the critique through the CRITIQUES cluster mechanism(s). "
        "Force the choice: name the assumption the supporting case rests on, "
        "then break it with the cited evidence in the same sentence.\n"
        "  4. Develop the qualification through the COMPLICATES cluster "
        "mechanism(s). Pick the binding scope condition; use one extended "
        "example to show what the thesis can and cannot explain under that "
        "condition.\n"
        "  5. Close by naming the underlying point of agreement (cite >=1 "
        "source), the precise remaining disagreement (cite >=1 distinct "
        "source), and the specific historical evidence or empirical comparison "
        "that would settle it. NO methodology stubs: forbid 'would be necessary', "
        "'would illuminate', 'would clarify', 'would resolve', 'would shed "
        "light', 'would inform', 'would allow us to', 'thorough comparison', "
        "'thorough examination', 'remains an open question', 'remains a subject "
        "of debate', 'further investigation'.\n"
        "\n"
        "TRANSITIONS MATTER: do NOT open each paragraph with a topic-restate; "
        "weave each paragraph from the previous one (pick up a thread, contrast "
        "a finding, follow a scope condition). The essay should read as ONE "
        "continuous argument, not several mini-essays glued together.\n"
        "\n"
        "Begin now. Output ONLY the essay prose. No headings, no labels, no preamble."
    )
    return "\n".join(parts), topic, topic_display


def _ollama_chat_single_pass(user_prompt: str, metrics=None,
                             stage="writer_single_pass"):
    """One-shot chat call with the single-pass options profile. Mirrors
    _ollama_chat's instrumentation but bumps num_ctx to 16384 (the single-pass
    prompt is ~8500 tokens, well past the chunked 12288 ceiling) and
    num_predict to 3500 (probe smoke under-produced at 3000 vs target 1300, so
    we hold a comfortable budget)."""
    import ollama
    import time as _time
    options = {
        "temperature": 0.50,
        "num_ctx": 16384,
        "num_predict": 3500,
        "top_p": 0.9,
    }
    _system = _writer_system_prompt()
    _dump_writer_prompt(stage, _system, user_prompt)
    start = _time.perf_counter()
    try:
        res = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _system},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=False,
        )
    except Exception as e:
        dur = _time.perf_counter() - start
        if metrics:
            metrics.record_llm(stage, _MODEL, options=options, success=False,
                               duration_s=dur, prompt_chars=len(user_prompt),
                               error=e)
        raise
    dur = _time.perf_counter() - start
    out = (res.get("message", {}).get("content") or "").strip()
    if metrics:
        metrics.record_llm(stage, _MODEL, options=options,
                           duration_s=dur, prompt_chars=len(user_prompt),
                           response_chars=len(out))
    return out, dur


def compose_from_ledger_single_pass(ledger_path=None, metrics=None):
    """v14 FIX-C: one-shot writer. Reads the ledger, builds a single prompt
    with topic + dimensions + cluster syntheses + per-doc evidence + writing
    instructions, sends ONE chat call to the writer model, runs the full v13.1.1
    postprocess pipeline on the response, and writes review_composed.md plus
    review_cited_docs.json. Mirrors compose_from_ledger's signature and output
    contract so downstream readers don't need changes.

    v15: NOT YET PORTED to outline_plan. v14.x cluster_syntheses no longer
    appear in the ledger. Use RRR_WRITER_MODE=chunked (the v15 default).
    """
    raise SystemExit(
        "[Writer] single-pass mode is not v15-compatible — outline_plan "
        "replaced cluster_syntheses. Use RRR_WRITER_MODE=chunked (default)."
    )
    ledger_path = ledger_path or str(runs_path("review_ledger.json"))
    if not os.path.isfile(ledger_path):
        raise SystemExit(f"Ledger not found: {ledger_path}")

    with open(ledger_path, encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("docs", [])
    if not isinstance(docs, list) or not docs:
        raise SystemExit("Ledger empty or malformed (no docs).")

    allowed_pairs, allowed_docs, allowed_pages_by_doc = _build_allowed_citations(docs)
    if not allowed_pairs:
        raise SystemExit("No allowed citations found in ledger.")

    author_year_to_docid = _build_author_year_lookup(allowed_docs)
    evidence_id_map = _build_evidence_id_map(docs)

    doc_to_evidence_ids = defaultdict(list)
    for eid, ev in evidence_id_map.items():
        did = (ev.get("doc_id") or "").strip()
        if did:
            doc_to_evidence_ids[did].append(eid)
    for did in list(doc_to_evidence_ids.keys()):
        doc_to_evidence_ids[did] = sorted(doc_to_evidence_ids[did])

    quotes_per_doc = int(os.environ.get("RRR_WRITER_QUOTES_PER_DOC", "2"))
    target_words = int(os.environ.get("RRR_WRITER_SINGLE_TARGET_WORDS", "1300"))

    user_prompt, topic, topic_display = _build_single_pass_user_prompt(
        data, allowed_pairs, allowed_docs, allowed_pages_by_doc,
        evidence_id_map, doc_to_evidence_ids, quotes_per_doc, target_words,
    )
    if topic != topic_display:
        print(f"[Writer/single] using reformulated topic: '{topic}'")

    print(f"[Writer/single] prompt: system={len(_SYSTEM_CITATION_INSTRUCTION)} "
          f"chars + user={len(user_prompt)} chars "
          f"(~{(len(_SYSTEM_CITATION_INSTRUCTION) + len(user_prompt)) // 4} tokens)")
    print(f"[Writer/single] eligible cites: {len(allowed_pairs)} (doc,page) "
          f"pairs across {len(allowed_docs)} docs")

    raw, ollama_duration_s = _ollama_chat_single_pass(
        user_prompt, metrics=metrics, stage="writer_single_pass",
    )
    print(f"[Writer/single] ollama returned {len(raw)} chars in "
          f"{ollama_duration_s:.1f}s")

    # ----- Postprocess pipeline (mirrors compose_from_ledger, same order) -----
    full_text = _strip_wrapping(raw)

    full_text, _single_render_stats = _render_evidence_id_citations(
        full_text, evidence_id_map,
    )
    evidence_renders = _single_render_stats["replacements"]

    # v14 FIX-BRACKET: rewrite [Doc_Year] bracketed canonical doc_ids before
    # display->canonical so the resulting display label gets normalised by
    # downstream passes.
    full_text, bracket_rewrites = _render_bracketed_doc_ids(
        full_text, allowed_docs, doc_to_evidence_ids,
    )

    # Bring display surface to canonical for validation (same lookup the
    # chunked path uses).
    display_lookup = _build_display_lookup(allowed_docs)

    def _disp_to_canon(t):
        def repl(m):
            label = m.group(1).strip().lower()
            year = m.group(2)
            page = int(m.group(3))
            did = display_lookup.get((label, year))
            if did:
                return render_citation_canonical(did, page)
            return m.group(0)
        t = DISPLAY_CITE_RE.sub(repl, t)
        t = DISPLAY_PAREN_CITE_RE.sub(repl, t)
        return t

    full_text = _disp_to_canon(full_text)
    full_text, double_collapsed = _collapse_double_parens(full_text)

    if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        removed_citations = []
        if metrics:
            metrics.set("writer_bypass_validation", True)
    else:
        full_text, removed_citations = _remove_invalid_citations(
            full_text, allowed_docs, allowed_pairs=allowed_pairs,
        )

    full_text, style_removed = _remove_style_violations(full_text)
    full_text, redundancy_drops = _drop_cross_section_redundancy(
        full_text, min_token_overlap=4,
    )
    full_text = _strip_orphaned_citations(full_text)
    full_text = _strip_references_section(full_text)
    full_text = _strip_continuation_markers(full_text)
    full_text, meta_removed = _strip_meta_commentary(full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    full_text, canonical_to_display_count = rewrite_canonical_to_display(full_text)
    full_text, narrative_rewrites = rewrite_misplaced_narrative_citations(full_text)
    full_text, mid_cite_fixes = _fix_mid_sentence_narrative_cites(full_text)
    full_text, nested_collapses = collapse_nested_narrative_multicite(full_text)
    full_text, label_comma_fixes = _strip_malformed_author_label_commas(full_text)
    full_text, same_author_collapses = _collapse_same_author_year_paren_cites(full_text)
    full_text, style_stats = _apply_style_enforcement(full_text, metrics=metrics)
    full_text, placeholder_stripped = _strip_author_year_placeholder(full_text)

    # v14 keep_closing: the closing paragraph is structurally important to the
    # essay. If style enforcement strips the only citation from the last
    # paragraph (probe smoke showed this happening when the closing leans on a
    # methodology-stub sentence), keep the weak closing rather than truncate.
    full_text, zero_cite_dropped, zc_kept_closing = _drop_zero_citation_paragraphs(
        full_text, keep_closing=True,
    )
    if zero_cite_dropped:
        print(f"[Writer/single] Dropped {len(zero_cite_dropped)} "
              f"zero-citation paragraph(s):")
        for snippet in zero_cite_dropped[:3]:
            print(f"         - {snippet}")
    if zc_kept_closing:
        print(f"[Writer/single] Kept zero-citation closing paragraph (would "
              f"have been dropped): {zc_kept_closing[:120]}")
        if metrics:
            metrics.inc("writer_zero_cite_closing_kept", 1)
            metrics.set("writer_zero_cite_closing_kept_snippet", zc_kept_closing)

    cited_docs = _collect_cited_docs(full_text, allowed_docs, author_year_to_docid)
    cited_docids = sorted(cited_docs)
    total_words = _count_words(full_text)

    ensure_dir(str(runs_path()))
    out_path = runs_path("review_composed.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(runs_path("review_cited_docs.json"), "w", encoding="utf-8") as f:
        json.dump(cited_docids, f, indent=2)

    print(f"[Writer/single] review_composed.md written ({total_words} words).")
    print(
        f"[Writer/single] stats: distinct_docs={len(cited_docids)} "
        f"removed={len(removed_citations)} "
        f"style_removed={len(style_removed)} "
        f"evidence_id_renders={evidence_renders} "
        f"bracket_id_rewrites={bracket_rewrites} "
        f"double_paren_collapsed={double_collapsed} "
        f"redundancy_drops={len(redundancy_drops)} "
        f"meta_removed={len(meta_removed)} "
        f"narrative_rewrites={narrative_rewrites} "
        f"mid_cite_fixes={mid_cite_fixes} "
        f"nested_collapses={nested_collapses} "
        f"same_author_collapses={same_author_collapses} "
        f"placeholder_stripped={placeholder_stripped} "
        f"canonical_to_display={canonical_to_display_count} "
        f"label_comma_fixes={label_comma_fixes} "
        f"zero_cite_dropped={len(zero_cite_dropped)}"
    )

    if metrics:
        metrics.set("writer_stats", {
            "chunks_written": 1,
            "distinct_docs_cited": len(cited_docids),
            "word_count": total_words,
            "removed_citations": len(removed_citations),
            "style_sentences_removed": len(style_removed),
            "evidence_id_renders": evidence_renders,
            "bracket_id_rewrites": bracket_rewrites,
            "double_paren_collapsed": double_collapsed,
            "redundancy_drops": len(redundancy_drops),
            "meta_commentary_stripped": len(meta_removed),
            "narrative_to_paren": narrative_rewrites,
            "mid_sentence_cite_fixes": mid_cite_fixes,
            "nested_paren_collapsed": nested_collapses,
            "same_author_collapses": same_author_collapses,
            "placeholder_citation_stripped": placeholder_stripped,
            "canonical_to_display": canonical_to_display_count,
            "author_label_comma_fixes": label_comma_fixes,
            "zero_cite_paragraphs_dropped": len(zero_cite_dropped),
            "ollama_duration_s": round(ollama_duration_s, 2),
            "prompt_user_chars": len(user_prompt),
            "raw_chars": len(raw),
        })
        metrics.set("writer_mode", "single")
        metrics.set("writer_style_enforcement", style_stats)
        metrics.inc("writer_removed_citations", len(removed_citations))
        metrics.inc("writer_style_sentences_removed", len(style_removed))
        metrics.inc("writer_evidence_id_renders", evidence_renders)
        metrics.inc("writer_double_paren_collapsed", double_collapsed)
        if bracket_rewrites:
            metrics.inc("writer_bracket_id_rewrites", bracket_rewrites)
        if redundancy_drops:
            metrics.inc("writer_redundancy_drops", len(redundancy_drops))
        if meta_removed:
            metrics.inc("writer_meta_commentary_stripped", len(meta_removed))
        if narrative_rewrites:
            metrics.inc("writer_narrative_to_paren", narrative_rewrites)
        if mid_cite_fixes:
            metrics.inc("writer_mid_sentence_cite_fixes", mid_cite_fixes)
        if nested_collapses:
            metrics.inc("writer_nested_paren_collapsed", nested_collapses)
        if same_author_collapses:
            metrics.inc("writer_same_author_collapses", same_author_collapses)
        if placeholder_stripped:
            metrics.inc("writer_placeholder_citation_stripped", placeholder_stripped)
        if canonical_to_display_count:
            metrics.inc("writer_canonical_to_display", canonical_to_display_count)
        if label_comma_fixes:
            metrics.inc("writer_author_label_comma_fixes", label_comma_fixes)
        if zero_cite_dropped:
            metrics.inc("writer_zero_cite_paragraphs_dropped", len(zero_cite_dropped))
            metrics.set("writer_zero_cite_paragraphs", zero_cite_dropped[:5])
        metrics.inc("writer_sections_succeeded")  # the single pass counts as one

    return str(out_path)


def compose_review(ledger_path=None, metrics=None):
    """v14 dispatcher: pick the writer path via RRR_WRITER_MODE.
      chunked -> compose_from_ledger             (v13.1.1 multi-call writer; DEFAULT)
      single  -> compose_from_ledger_single_pass (one LLM call; experimental)

    v14 single-pass smoke (pod_outputs/runpod_20260628_213140) confirmed that
    mistral models — both 7B (v9) and small:24b (v14) — wrap up around
    400 words regardless of prompt length target. Single-pass mean was 547
    words vs chunked ~990. For long-form deliverables on the mistral family,
    chunked is the architecture. Single-pass is preserved here for: (a) future
    non-mistral models without the wrap-up urgency, (b) short-form modes.
    See revision_notes/current_status.md § "v14 Experiment" for details.
    """
    mode = os.environ.get("RRR_WRITER_MODE", "chunked").strip().lower()
    if mode == "single":
        return compose_from_ledger_single_pass(ledger_path=ledger_path, metrics=metrics)
    if mode != "chunked":
        print(f"[Writer] Unknown RRR_WRITER_MODE='{mode}', defaulting to 'chunked'.")
    return compose_from_ledger(ledger_path=ledger_path, metrics=metrics)


if __name__ == "__main__":
    compose_review()
