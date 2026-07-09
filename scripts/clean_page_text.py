#!/usr/bin/env python3
"""
clean_page_text.py — v14.3 post-extraction cleanup pipeline for the RRR corpus.

Applies eight ordered cleanup steps to every `data/page_text/{doc_id}_page_{N}.txt`
file produced by `python -m rrr.preprocess`. Designed to fix the silent OCR /
layout artifacts the v14.3 extraction audit (workflow wf_b750f0f0-eeb)
quantified across 1240 corpus pages:

  - 3186 unresolved fi/fl ligatures across 488 pages (39% of corpus)
  - 1753 period-no-space joins across 678 pages (55%)
  - 1406 camelCase line-break joins across 583 pages (47%)
  - 1094 inline page-number leaks across 545 pages (44%)
  - 502 OCR-known-errors across 337 pages (`foumal`, `fiom`, `ofthe`, `tlie`)
  - 420 embedded running-header repetitions

Pipeline (ORDER IS LOAD-BEARING):

  1. unicode_normalize         NFKC + explicit ligature/dash/quote map; strip cid:NNNN
  2. dehyphenate_linebreaks    `productiv- ity` -> `productivity` (compound protect-list)
  3. ocr_known_errors          known one-shot OCR substitutions (`foumal` -> `Journal`)
  4. strip_running_headers     cross-page repeated short headers / footers (per-paper)
  5. strip_jstor_oup_boilerplate  publisher-footer regex strip
  6. fix_missing_spaces        digit->letter / lower->Upper / punct->letter spacers
  7. collapse_whitespace       `[ \\t]+` -> single space; preserve paragraph breaks
  8. strip_footnote_anchors    inline `[a-z]\\d{1,2}[ ][A-Z]` heuristic

The pipeline writes either IN PLACE (default — overwrites `data/page_text/`)
or to a separate `--output-dir` (for iteration without re-extracting).

CLI:
  python scripts/clean_page_text.py                     # in place; default input data/page_text/
  python scripts/clean_page_text.py --output-dir data/page_text_clean
  python scripts/clean_page_text.py --report             # show artifact counts before/after
  python scripts/clean_page_text.py --doc-ids Acemoglu*  # limit to matching doc_ids
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: unicode normalisation + explicit char-map (ligatures, dashes, quotes)
# ─────────────────────────────────────────────────────────────────────────────

# Ligature decompositions and control-char replacements that NFKC doesn't
# always catch (or that we want to force).
_UNICODE_MAP = {
    "ﬀ": "ff",   # ﬀ
    "ﬁ": "fi",   # ﬁ
    "ﬂ": "fl",   # ﬂ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
    "ﬅ": "st",   # ﬅ
    "ﬆ": "st",   # ﬆ
    " ": " ",    # nbsp
    "‐": "-", "‑": "-", "‒": "-",  # various hyphens
    "–": "-", "—": "-",                  # en/em dash
    "‘": "'", "’": "'",                  # smart single quotes
    "“": '"', "”": '"',                  # smart double quotes
    "…": "...",                               # ellipsis
}
_CID_RE = re.compile(r"\(cid:\d+\)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def unicode_normalize(text: str) -> str:
    """Step 1: ligature decomposition + dash/quote canonicalisation +
    cid-marker strip + control-char strip. NFKC is permissive (rewrites
    unicode math minus U+2212 to ASCII -); on table-dominant pages this
    could damage equations, but the BM25 index doesn't rely on equation
    glyphs so the loss is acceptable here.
    """
    if not text:
        return ""
    text = _CID_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    for src, dst in _UNICODE_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    # NFKC after explicit map so we catch the most common ligatures with
    # known intent first, then let unicodedata handle the long tail.
    text = unicodedata.normalize("NFKC", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: dehyphenate line-break splits with compound protect-list
# ─────────────────────────────────────────────────────────────────────────────

# Compounds that should NOT be dehyphenated. List is small but covers the
# common academic-prose compounds we've seen in this corpus. The pipeline
# checks before re-joining: if the proposed rejoin is in this list (or its
# hyphenated form is in the list), skip the join.
_COMPOUND_PROTECT = {
    "long-run", "long-term", "short-run", "short-term", "pre-modern",
    "post-war", "post-colonial", "post-1500", "cross-section", "cross-country",
    "non-trivial", "non-linear", "non-zero", "self-reinforcing",
    "factor-endowments", "rule-of-law", "winner-take-all", "free-rider",
    "rent-seeking", "well-being", "well-defined", "well-known",
    "above-mentioned", "decision-making", "double-entry", "year-on-year",
    "twentieth-century", "nineteenth-century", "eighteenth-century",
    "high-income", "low-income", "middle-income", "two-thirds", "one-half",
    "one-third", "three-quarters", "open-access", "open-source",
    "real-world", "in-depth", "data-driven", "evidence-based",
    "country-level", "region-level", "paper-and-pencil", "et-al",
    "u.s.a.", "u.s.",
}

# Hyphen-line-break pattern: word fragment, hyphen, optional whitespace
# (including newline), continuation word fragment.
_LINEBREAK_HYPHEN_RE = re.compile(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})")
_INLINE_HYPHEN_RE = re.compile(r"([a-z]{2,})-\s+([a-z]{2,})")


def dehyphenate_linebreaks(text: str) -> str:
    """Step 2: rejoin `productiv-\\nity` -> `productivity`, with compound-
    protect-list as safety. Two passes: first the explicit line-break form
    (always rejoin), then the inline `word- word` form (heuristic-rejoin
    skipped for protected compounds and when either fragment is too short
    to be a real word).
    """

    def _join(m):
        left, right = m.group(1), m.group(2)
        candidate = f"{left}{right}".lower()
        hyphenated = f"{left}-{right}".lower()
        if hyphenated in _COMPOUND_PROTECT:
            return m.group(0)
        return left + right

    text = _LINEBREAK_HYPHEN_RE.sub(_join, text)
    text = _INLINE_HYPHEN_RE.sub(_join, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: known OCR character errors
# ─────────────────────────────────────────────────────────────────────────────

# A curated dictionary of OCR substitutions we've directly observed in the
# corpus extraction audit. Conservative: only one-shot corrections of clear
# garbage tokens that would never appear as legit English. Keys are
# case-sensitive; we match whole-word with \b boundaries.
_OCR_ERROR_DICT = {
    # `f`/`J` confusion (the J's top stroke gets OCR'd as f)
    "foumal": "Journal",
    "fiom": "from",
    # `m`/`rn` confusion
    "modem": "modern",  # only when not in tech context — for an economic-history corpus, "modem" essentially never means the device
    "atrnosphere": "atmosphere",
    "univetsity": "university",
    # `cl`/`d` confusion
    "stuclies": "studies",
    "studie": "studied",
    # `tl`/`th` confusion
    "tlie": "the",
    "tlrey": "they",
    # missing-space joins commonly produced
    "ofthe": "of the",
    "tothe": "to the",
    "andthe": "and the",
    "ortho": "or the",
    "inthe": "in the",
    "forthe": "for the",
    "withthe": "with the",
    "havebeen": "have been",
    "hasbeen": "has been",
    "hadbeen": "had been",
    # known stealth Sokoloff h/b substitutions (still partial — true fix
    # is re-OCR; these are the most-frequent corrections)
    "hasis": "basis",
    "lahor": "labor",
    "puhlished": "published",
    "Lahor": "Labor",
}


def _build_ocr_error_regex():
    keys = sorted(_OCR_ERROR_DICT.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.compile(pattern)


_OCR_ERROR_RE = _build_ocr_error_regex()


def ocr_known_errors(text: str) -> str:
    """Step 3: replace known one-shot OCR substitutions. Whole-word match
    only — the substitutions never spuriously fire on legit tokens (e.g.
    `foumal` is not English; `ofthe` is not English). The set is small and
    explicit so the failure mode is bounded."""
    return _OCR_ERROR_RE.sub(lambda m: _OCR_ERROR_DICT[m.group(1)], text)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: strip running headers / footers (per-paper cross-page repetition)
# ─────────────────────────────────────────────────────────────────────────────

_PAGENUM_ONLY_RE = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)


def detect_repeated_headers(pages: list, min_repetition_ratio: float = 0.5) -> set:
    """Identify lines that appear at the top OR bottom of >=
    `min_repetition_ratio` of the paper's pages — these are the running
    headers / footers. Returns a set of exact-match strings to strip.
    """
    if len(pages) < 3:
        return set()
    head_candidates = Counter()
    tail_candidates = Counter()
    for p in pages:
        lines = [L.strip() for L in p.splitlines() if L.strip()]
        if not lines:
            continue
        head = lines[0]
        tail = lines[-1]
        if 4 <= len(head) <= 80:
            head_candidates[head] += 1
        if 4 <= len(tail) <= 80:
            tail_candidates[tail] += 1
    threshold = max(2, int(len(pages) * min_repetition_ratio))
    repeated = set()
    for line, count in head_candidates.items():
        if count >= threshold:
            repeated.add(line)
    for line, count in tail_candidates.items():
        if count >= threshold:
            repeated.add(line)
    return repeated


def strip_running_headers(text: str, headers_to_strip: set) -> str:
    """Step 4: strip exact-match repeated header/footer lines. Also strips
    bare page-number lines (a digit-only line is never main-text)."""
    if headers_to_strip:
        for header in headers_to_strip:
            text = re.sub(
                r"^[ \t]*" + re.escape(header) + r"[ \t]*\n?",
                "",
                text,
                flags=re.MULTILINE,
            )
    text = _PAGENUM_ONLY_RE.sub("", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: strip JSTOR / OUP boilerplate footers (universal regex)
# ─────────────────────────────────────────────────────────────────────────────

_JSTOR_FOOTER_RE = re.compile(
    # v14.3.1: actual PyMuPDF output truncates at "This content downloaded from"
    # without the jstor.org/terms tail (rest of line is rendered as an image).
    # Match the phrase + up to 300 chars of following metadata noise.
    r"This content downloaded from[^\n]{0,300}",
)
_JSTOR_BANNER_RE = re.compile(
    r"Your use of the JSTOR archive.*?Terms and Conditions of Use.*?(?=\n\n|\Z)",
    re.DOTALL,
)
_JSTOR_SPLASH_RE = re.compile(
    r"Your use of the JSTOR archive indicates your acceptance of JSTOR.*?(?=\n\n|\Z)",
    re.DOTALL,
)
_OUP_BANNER_RE = re.compile(
    r"©\s*The Author.*?Oxford University Press.*?(?=\n\n|\Z)",
    re.DOTALL,
)
# v14.3: catch the standalone "extend access to The Journal of X" line that
# JSTOR puts on every splash page. The truncated PyMuPDF output for these
# splash pages leaves these as the dominant content.
_JSTOR_EXTEND_ACCESS_RE = re.compile(
    r"(?:linked references are available.*?\n|extend\s*\n?\s*access to[^\n]+\n)",
    re.IGNORECASE,
)


def strip_publisher_boilerplate(text: str) -> str:
    """Step 5: strip JSTOR and Oxford-University-Press standard boilerplate
    that appears at the bottom of nearly every page of papers from those
    publishers. ~463 pages affected per the v14.3 audit."""
    text = _JSTOR_FOOTER_RE.sub("", text)
    text = _JSTOR_BANNER_RE.sub("", text)
    text = _JSTOR_SPLASH_RE.sub("", text)
    text = _OUP_BANNER_RE.sub("", text)
    text = _JSTOR_EXTEND_ACCESS_RE.sub("", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: fix missing spaces (digit->letter, lower->Upper, punct->letter)
# ─────────────────────────────────────────────────────────────────────────────

# Acronyms / surnames where lower->Upper is legitimate. Conservative — better
# to leave some unjoined than to break `iPhone`. This is small for academic prose.
_CAMELCASE_PROTECT = {
    "iPhone", "eBay", "iPad", "iPod", "iMac", "iOS", "macOS",
    # Initials commonly written as "J.M.Keynes" style — leave alone
}

_DIGIT_LETTER_RE = re.compile(r"(\d)([A-Z][a-z])")
_LOWER_UPPER_RE = re.compile(r"([a-z]{2,})([A-Z][a-z]{2,})")
_PUNCT_LETTER_RE = re.compile(r"([,.;:!?])([A-Z][a-z])")
# Stricter: comma/period immediately followed by lowercase = missing space too.
_PUNCT_LOWER_RE = re.compile(r"([,.;:])([a-z]{3,})")


def fix_missing_spaces(text: str) -> str:
    """Step 6: insert a space where the extractor dropped one. Three
    patterns:

      - digit followed by capitalised word: `726WORLD` -> `726 WORLD`
      - lowercase word followed by capitalised word: `byThe` -> `by The`
        (skipped for the acronym/brand protect-list)
      - punctuation followed by capitalised or lowercase word with no space

    The lower->Upper rule requires >=2 chars on each side to avoid splitting
    initials like `J.MKeynes`. The protect-list short-circuits `iPhone` etc.
    """

    text = _DIGIT_LETTER_RE.sub(r"\1 \2", text)

    def _split_lu(m):
        joined = m.group(0)
        if joined in _CAMELCASE_PROTECT:
            return joined
        return f"{m.group(1)} {m.group(2)}"

    text = _LOWER_UPPER_RE.sub(_split_lu, text)
    text = _PUNCT_LETTER_RE.sub(r"\1 \2", text)
    text = _PUNCT_LOWER_RE.sub(r"\1 \2", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: collapse whitespace (preserve paragraph breaks)
# ─────────────────────────────────────────────────────────────────────────────

_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def collapse_whitespace(text: str) -> str:
    """Step 7: `   ` / tabs -> single space; `\\n\\n\\n+` -> `\\n\\n`.
    Preserves paragraph boundaries (double-newline) so sentence-splitting
    downstream still sees structure."""
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: strip footnote anchors (heuristic, no PyMuPDF metadata yet)
# ─────────────────────────────────────────────────────────────────────────────

# Inline-footnote-anchor pattern: end of word + 1-2 digit + space + capital.
# E.g. "extractive in previously prosperous places.2 The main reason" -> the
# `.2 ` is the anchor for footnote 2. Conservative: also requires that the
# digit is small (<=99); larger numbers are usually page references.
_FOOTNOTE_ANCHOR_RE = re.compile(r"([a-z][.,!?])(\d{1,2})\s+([A-Z])")


def strip_footnote_anchors(text: str) -> str:
    """Step 8: heuristic strip of inline footnote-anchor digits. PyMuPDF
    exposes per-character superscript metadata that would make this much
    more accurate; this regex-only version is the ~40% fallback. Wired off
    by default — set RRR_STRIP_FOOTNOTE_ANCHORS=1 to enable, since the
    false-positive rate on cases like `lived in 14 countries` is non-trivial."""
    if os.environ.get("RRR_STRIP_FOOTNOTE_ANCHORS", "0") != "1":
        return text
    return _FOOTNOTE_ANCHOR_RE.sub(r"\1 \3", text)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ─────────────────────────────────────────────────────────────────────────────

def is_low_signal_page(text: str) -> bool:
    """v14.3 Step 9: after cleanup, drop pages that are pure JSTOR splash,
    front-matter, or otherwise too thin to be main text. Threshold: <200
    alpha-token chars OR alpha-token ratio <0.4 (page is mostly digits/
    punctuation, e.g. tables with little prose)."""
    if not text or len(text) < 200:
        return True
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if len(tokens) < 30:
        return True
    alpha_chars = sum(len(t) for t in tokens)
    return alpha_chars / max(1, len(text)) < 0.4


def clean_one_paper(pages: list) -> tuple:
    """Run the cleanup pipeline on all pages of one paper. Cross-page steps
    (running-header detection) operate over the page list; per-page steps
    are applied to each page in turn. Returns (cleaned_pages, dropped_indices).
    """
    pages = [unicode_normalize(p) for p in pages]
    pages = [dehyphenate_linebreaks(p) for p in pages]
    pages = [ocr_known_errors(p) for p in pages]
    headers_to_strip = detect_repeated_headers(pages)
    pages = [strip_running_headers(p, headers_to_strip) for p in pages]
    pages = [strip_publisher_boilerplate(p) for p in pages]
    pages = [fix_missing_spaces(p) for p in pages]
    pages = [collapse_whitespace(p) for p in pages]
    pages = [strip_footnote_anchors(p) for p in pages]
    # Drop low-signal pages AFTER all cleanup so the threshold judges the
    # final content, not the noisy original.
    dropped = []
    out = []
    for i, p in enumerate(pages):
        if is_low_signal_page(p):
            dropped.append(i)
        else:
            out.append(p)
    return out, dropped


def group_pages_by_doc(page_dir: Path) -> dict:
    """Return {doc_id: [(page_num, path), ...]} sorted by page_num."""
    groups = defaultdict(list)
    for path in page_dir.glob("*_page_*.txt"):
        m = re.match(r"^(.+)_page_(\d+)\.txt$", path.name)
        if not m:
            continue
        doc_id, page_num = m.group(1), int(m.group(2))
        groups[doc_id].append((page_num, path))
    for doc_id in groups:
        groups[doc_id].sort()
    return groups


def count_artifacts(text: str) -> dict:
    """Quick artifact count for before/after reporting."""
    return {
        "ligatures": sum(1 for c in text if c in "ﬀﬁﬂﬃﬄﬅﬆ"),
        "double_space": len(re.findall(r"  +", text)),
        "hyphen_linebreak": len(re.findall(r"[a-z]-\s*\n\s*[a-z]", text)),
        "camel_join": len(re.findall(r"[a-z]{2,}[A-Z][a-z]{2,}", text)),
        "punct_no_space": len(re.findall(r"[.,;:][a-z]{3,}", text)),
        "jstor_footer": len(re.findall(r"This content downloaded", text)),
        "known_ocr_errors": sum(1 for k in _OCR_ERROR_DICT if re.search(r"\b" + re.escape(k) + r"\b", text)),
        "chars": len(text),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="data/page_text", help="Source directory of page text files")
    ap.add_argument("--output-dir", default=None, help="Destination (default: in place on --input-dir)")
    ap.add_argument("--doc-ids", default=None, help="Comma-separated doc_id filter (substring match)")
    ap.add_argument("--report", action="store_true", help="Print before/after artifact counts per paper")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    in_place = output_dir == input_dir

    if not input_dir.is_dir():
        sys.exit(f"input-dir not found: {input_dir}")

    groups = group_pages_by_doc(input_dir)
    if args.doc_ids:
        wanted = [s.strip() for s in args.doc_ids.split(",") if s.strip()]
        groups = {d: pgs for d, pgs in groups.items() if any(w.lower() in d.lower() for w in wanted)}

    print(f"[clean] {len(groups)} papers, {sum(len(p) for p in groups.values())} pages, "
          f"input={input_dir}, output={output_dir}{' (IN PLACE)' if in_place else ''}")

    before_totals = Counter()
    after_totals = Counter()
    per_paper_report = []

    total_dropped = 0
    for doc_id, pgs in sorted(groups.items()):
        before_texts = []
        for _, path in pgs:
            before_texts.append(path.read_text(encoding="utf-8", errors="replace"))
        # v14.3: clean_one_paper now returns (cleaned_pages, dropped_indices).
        # cleaned_pages is the SHORTER list (low-signal pages removed); the
        # dropped_indices reference positions in the ORIGINAL pgs list so we
        # can delete the corresponding files (when in_place) and skip writing.
        after_texts, dropped_idx = clean_one_paper(before_texts)
        dropped_set = set(dropped_idx)
        total_dropped += len(dropped_set)

        b_tot = Counter()
        a_tot = Counter()
        for i, bt in enumerate(before_texts):
            for k, v in count_artifacts(bt).items():
                b_tot[k] += v
        for at in after_texts:
            for k, v in count_artifacts(at).items():
                a_tot[k] += v
        for k, v in b_tot.items():
            before_totals[k] += v
        for k, v in a_tot.items():
            after_totals[k] += v

        # Walk original pages list; write kept pages, delete (in place) or
        # skip (separate output) the dropped ones.
        cleaned_iter = iter(after_texts)
        for i, (page_num, src_path) in enumerate(pgs):
            dst_path = output_dir / src_path.name
            if i in dropped_set:
                if in_place and dst_path.is_file():
                    dst_path.unlink()
                continue
            cleaned = next(cleaned_iter)
            dst_path.write_text(cleaned, encoding="utf-8")
        per_paper_report.append((doc_id, len(pgs), b_tot, a_tot, len(dropped_set)))

    kept_files = sum(len(p) for p in groups.values()) - total_dropped
    print(f"\n[clean] DONE — wrote {kept_files} files to {output_dir} "
          f"(dropped {total_dropped} low-signal pages)")

    print(f"\n{'metric':<24} {'before':>12} {'after':>12} {'delta':>10}")
    print("-" * 64)
    for k in sorted(set(before_totals) | set(after_totals)):
        if k == "chars":
            continue
        b, a = before_totals[k], after_totals[k]
        delta = a - b
        pct = f"{(delta / b * 100):+.1f}%" if b else "n/a"
        print(f"{k:<24} {b:>12} {a:>12} {pct:>10}")
    b_chars = before_totals.get("chars", 0)
    a_chars = after_totals.get("chars", 0)
    print(f"{'chars':<24} {b_chars:>12} {a_chars:>12} {(a_chars - b_chars) / max(1, b_chars) * 100:+.2f}%")

    if args.report:
        print(f"\n--- per-paper artifact deltas (worst before -> after) ---")
        per_paper_report.sort(key=lambda r: sum(r[2].values()), reverse=True)
        print(f"{'doc_id':<35} {'pages':>5} {'drop':>5} {'before':>7} {'after':>7} {'pct':>7}")
        for doc_id, n_pages, b, a, n_dropped in per_paper_report[:20]:
            b_sum = sum(v for k, v in b.items() if k != "chars")
            a_sum = sum(v for k, v in a.items() if k != "chars")
            pct = f"{(a_sum - b_sum) / max(1, b_sum) * 100:+.1f}%" if b_sum else "n/a"
            print(f"{doc_id:<35} {n_pages:>5} {n_dropped:>5} {b_sum:>7} {a_sum:>7} {pct:>7}")


if __name__ == "__main__":
    main()
