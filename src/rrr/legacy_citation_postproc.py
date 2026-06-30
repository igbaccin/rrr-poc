"""Legacy citation post-processing — preserved as fallback (v15.5).

These functions were the main post-processing pipeline through v15.4.0.
In v15.5 the writer was switched to evidence-ID-only citations
(`[E0001]` markers), and the renderer (`_render_evidence_id_citations`
in writer.py) became context-aware — detecting whether the author
label is already in the preceding prose and rendering as `(Year, p.N)`
or `(Author Year, p.N)` accordingly.

That redesign removes the root cause of the surface-variation bugs
these functions were patching:
  - `(Author 2007)` literal placeholder
  - `(Author (Year, p.N).` orphan outer paren
  - `Author (Year)... (Author Year, p.N)` redundant inline + trailing
  - `((Doc: p.N))` double paren
  - mid-sentence narrative cites without preceding punctuation
  - same-author-year pseudo-plural multi-cites
  - and similar.

These functions are PRESERVED HERE (not deleted) because:
  - smaller models (e.g. mistral 7B) may still produce display-form
    cites despite the `[E####]`-only prompt; the legacy passes are
    the regression net
  - the existing v15.x batteries on mistral-small:24b confirm the
    new path is clean, but cross-model testing (qwen3:32b, mistral 7B,
    etc.) may surface model-specific behaviour that still needs them
  - rather than re-derive these regexes if needed later, we keep them
    in source — to enable, import the function and call it in
    writer.py's postprocess pipeline.

Re-enabling: import from this module and add a call at the relevant
point in writer.py's `postprocess_chunk` or the final-assembly
pipeline. Watch the corresponding metric counter to confirm it fires.

Last active commit before move: c9d0de6 (v15.4.0).
"""

from __future__ import annotations

import re


# v10 display surface — two shapes the model produced interchangeably
# before v15.5 forced evidence-ID-only:
#   1. BARE  : "Author (Year, p.N)"          -- narrative form
#   2. PAREN : "(Author Year, p.N)"          -- parenthetical form
_DISPLAY_UNIT = (
    r"(?:(?i:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
)
_DISPLAY_LABEL = (
    r"(?:" + _DISPLAY_UNIT +
    r"(?:\s+(?:and\s+" + _DISPLAY_UNIT + r"|et\s+al\.))?)"
)
DISPLAY_CITE_RE = re.compile(
    r"(?<!\w)(" + _DISPLAY_LABEL + r")\s+"
    r"\((\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)
DISPLAY_PAREN_CITE_RE = re.compile(
    r"\((" + _DISPLAY_LABEL + r")(?:,?\s+)(\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)


# v13.1 FIX-B: literal "(Author, Year)" placeholder strip.
# v15.4.0 widening: also catch "(Author 2007)" with a real year.
_AUTHOR_YEAR_PLACEHOLDER_RE = re.compile(
    r"\s?\(Author(?:,)?\s+(?:Year|\d{4}[a-z]?)"
    r"(?:(?:,\s*|\s+)p\.\s*(?:\d+|N))?\)"
)


def _strip_author_year_placeholder(text: str) -> tuple:
    """v13.1 FIX-B (widened in v14.2 + v15.4.0): strip literal
    "(Author, Year)" / "(Author Year)" / "(Author 2007)" placeholders.
    """
    if not text:
        return text or "", 0
    cleaned, count = _AUTHOR_YEAR_PLACEHOLDER_RE.subn("", text)
    return cleaned, count


# v8 (R5) / v9.1: catch nested citation wraps that escape every other
# cleanup because CITE_RE matches only the single-paren form.
_DOUBLE_PAREN_CITE_RE = re.compile(
    r"\(\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)(?::\s*p\.(\d+))?\)"
)


def _collapse_double_parens(text: str) -> tuple:
    """v8 (R5): collapse ((Doc: p.N)) -> (Doc: p.N), and the doubled
    page-suffix shape ((Doc: p.X): p.Y) -> (Doc: p.Y).
    """
    if not text:
        return text or "", 0
    count = 0

    def _repl(m):
        nonlocal count
        count += 1
        doc = m.group(1)
        outer_page = m.group(3)
        inner_page = m.group(2)
        page = outer_page if outer_page else inner_page
        return f"({doc}: p.{int(page)})"

    return _DOUBLE_PAREN_CITE_RE.sub(_repl, text), count


# v13.1 FIX-A: stray commas inside author labels
# ("Sokoloff and, Engerman ...", "van, Zanden ...").
_AUTHOR_LABEL_CONNECTIVE_RE = re.compile(
    r"\b((?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"\s+and,\s+([A-Z][A-Za-z\-]+)"
)
_AUTHOR_LABEL_PARTICLE_COMMA_RE = re.compile(
    r"\b(van|von|de|del|der),\s+([A-Z][A-Za-z\-]+)"
)
_AUTHOR_LABEL_CONNECTIVE_COMMA_RE = re.compile(
    r"\b((?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"\s+and,\s+((?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
)
_AUTHOR_LABEL_CONNECTIVE_PARTICLE_COMMA_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]+)\s+and,\s+((?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+)"
)


def _strip_malformed_author_label_commas(text: str) -> tuple:
    if not text:
        return text or "", 0
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return f"{m.group(1)} and {m.group(2)}"

    cleaned, n0 = _AUTHOR_LABEL_CONNECTIVE_PARTICLE_COMMA_RE.subn(repl, text)
    cleaned, n1 = _AUTHOR_LABEL_CONNECTIVE_COMMA_RE.subn(repl, cleaned)
    cleaned, n2 = _AUTHOR_LABEL_PARTICLE_COMMA_RE.subn(repl, cleaned)
    return cleaned, count


# v13.1 FIX-D: collapse same-(author, year) pseudo-plural multi-cites
# inside one parenthetical.
_FLAT_PAREN_CITE_UNIT_RE = re.compile(
    r"(" + r"(?:(?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|[A-Z][A-Za-z\-]+)"
    r"(?:\s+(?:and\s+(?:(?:van|von|de|del|der)\s+[A-Z][A-Za-z\-]+|"
    r"[A-Z][A-Za-z\-]+)|et\s+al\.))?"
    r")\s+(\d{4})[a-z]?,\s*p\.\s*(\d+)"
)


# v15.4.0 (Bug 2): orphan outer paren "(Author (Year, p.N)<punct>".
_ORPHAN_OUTER_PAREN_NARRATIVE_RE = re.compile(
    r"\((" + _DISPLAY_LABEL + r")\s+\((\d{4}[a-z]?)(?:,\s*|\s+)p\.\s*(\d+)\)"
    r"(?=\s*[.!?;,])"
)


def fix_orphan_outer_paren_narrative(text: str) -> tuple:
    """v15.4.0 (Bug 2): rewrite '(Author (Year, p.N)<punct>' to
    '(Author Year, p.N)<punct>'."""
    if not text:
        return text, 0
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return f"({m.group(1).strip()} {m.group(2)}, p.{m.group(3)})"

    return _ORPHAN_OUTER_PAREN_NARRATIVE_RE.sub(repl, text), count


# v15.4.0 (Bug 3): redundant in-text + trailing parenthetical merge.
_INTEXT_NARRATIVE_FULL_RE = re.compile(
    r"(" + _DISPLAY_LABEL + r")\s+\((\d{4}[a-z]?)(?:(?:,\s*|\s+)p\.\s*(\d+))?\)"
)


def _norm_author_label(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+&\s+", " and ", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def merge_redundant_inline_paren_citations(text: str) -> tuple:
    """v15.4.0 (Bug 3): merge a sentence that names the author in-text
    AND repeats the surname+year in a trailing parenthetical for the
    same (label, year)."""
    if not text:
        return text, 0
    parts = re.split(r"(?<=[.!?])(\s+)(?=[A-Z(\[])", text)
    out_parts: list = []
    count = 0
    for i, segment in enumerate(parts):
        if i % 2 == 1:
            out_parts.append(segment)
            continue
        sentence = segment
        intext = _INTEXT_NARRATIVE_FULL_RE.search(sentence)
        if not intext:
            out_parts.append(sentence)
            continue
        in_label_norm = _norm_author_label(intext.group(1))
        in_year = intext.group(2)
        in_page = intext.group(3)
        tail_search_start = intext.end()
        tail = None
        for m in DISPLAY_PAREN_CITE_RE.finditer(sentence, tail_search_start):
            if (_norm_author_label(m.group(1)) == in_label_norm
                    and m.group(2) == in_year):
                tail = m
                break
        if tail is None:
            out_parts.append(sentence)
            continue
        try:
            tail_page = int(tail.group(3))
        except (TypeError, ValueError):
            out_parts.append(sentence)
            continue
        if in_page is None:
            new_intext = f"{intext.group(1).strip()} ({in_year}, p.{tail_page})"
        else:
            try:
                in_page_int = int(in_page)
            except ValueError:
                out_parts.append(sentence)
                continue
            if in_page_int == tail_page:
                new_intext = intext.group(0)
            else:
                pages = sorted({in_page_int, tail_page})
                new_intext = (
                    f"{intext.group(1).strip()} ({in_year}, "
                    f"pp.{', '.join(str(p) for p in pages)})"
                )
        rebuilt = sentence[:intext.start()] + new_intext + sentence[intext.end():tail.start()]
        if rebuilt.endswith(" "):
            rebuilt = rebuilt[:-1]
        rebuilt += sentence[tail.end():]
        out_parts.append(rebuilt)
        count += 1
    return "".join(out_parts), count


# v11.1: collapse the nested-narrative-inside-paren multi-cite form.
_NESTED_NARRATIVE_OUTER_RE = re.compile(
    r"\(\s*"
    r"(?:" + _DISPLAY_LABEL + r")\s+\(\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+\)"
    r"(?:\s*;\s*(?:" + _DISPLAY_LABEL + r")\s+\(\d{4}[a-z]?(?:,\s*|\s+)p\.\s*\d+\))*"
    r"\s*\)"
)
_INNER_NARRATIVE_CITE_RE = re.compile(
    r"(" + _DISPLAY_LABEL + r")\s+\((\d{4})[a-z]?(?:,\s*|\s+)p\.\s*(\d+)\)"
)


def collapse_nested_narrative_multicite(text: str) -> tuple:
    """v11.1: rewrite '(Author (Year, p.N); Author (Year, p.N))' to
    '(Author Year, p.N; Author Year, p.N)'."""
    if not text:
        return text, 0
    rewrites = 0

    def repl_outer(m):
        nonlocal rewrites
        inner = m.group(0)[1:-1]
        new_inner, n_inner = _INNER_NARRATIVE_CITE_RE.subn(
            lambda im: f"{im.group(1).strip()} {im.group(2)}, p.{im.group(3)}",
            inner,
        )
        if n_inner == 0:
            return m.group(0)
        rewrites += 1
        return f"({new_inner.strip()})"

    new_text = _NESTED_NARRATIVE_OUTER_RE.sub(repl_outer, text)
    return new_text, rewrites


def rewrite_canonical_to_display(text: str, display_renderer=None) -> tuple:
    """v10: rewrite '(Doc_Year: p.N)' canonical -> 'Author (Year, p.N)'
    display. display_renderer should be render_citation from render.py.
    """
    if display_renderer is None:
        from rrr.render import render_citation
        display_renderer = render_citation
    if not text:
        return text or "", 0
    canon = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")
    count = 0

    def repl(m):
        nonlocal count
        count += 1
        return display_renderer(m.group(1), int(m.group(2)))

    return canon.sub(repl, text), count


def rewrite_misplaced_narrative_citations(text: str) -> tuple:
    """v11-A: rewrite narrative citations 'Author (Year, p.N)' at the
    end of a sentence (where the subject is not the author) to the
    parenthetical form '(Author Year, p.N)'. Heuristic — checks the
    next non-whitespace char is sentence-ending punct."""
    if not text:
        return text, 0
    rewrites = 0
    out_parts: list = []
    last_end = 0
    for m in DISPLAY_CITE_RE.finditer(text):
        tail = text[m.end():m.end() + 3]
        if not tail or tail[0] not in ".!?;,":
            continue
        label = m.group(1)
        year = m.group(2)
        page = m.group(3)
        start = m.start()
        end = m.end()
        out_parts.append(text[last_end:start])
        out_parts.append(f"({label} {year}, p.{page})")
        last_end = end
        rewrites += 1
    if rewrites == 0:
        return text, 0
    out_parts.append(text[last_end:])
    return "".join(out_parts), rewrites
