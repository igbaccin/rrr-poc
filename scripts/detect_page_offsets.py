#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_page_offsets.py -- deterministic printed-page-offset detector for the RRR corpus.

For every document in the page-text directory (files <doc_id>_page_<N>.txt,
N = 1-based physical PDF page), extract printed-page-number candidates from
the first 3 / last 3 non-boilerplate lines of each page, vote a per-document
offset (printed = pdf_page_index + offset), and validate the implied printed
range against the 'pages' column of metadata.csv.

Design notes (from corpus recon):
  * PUA oldstyle-figure digits (U+F730..U+F739) are translated to ASCII first
    (Temin_2002, Koepke&Baten_2005 emit ALL digits that way).
  * Known boilerplate lines (JSTOR download footer trio, OUP/Cambridge
    watermarks, bare URLs, Elsevier eLocator headers) are dropped before the
    head/tail windows are taken, because they mask the real header/footer.
  * Candidate forms:
      - bare number on its own line (head or tail window), rejected when a
        directly adjacent line is also purely numeric (chart ticks / table
        cell runs) or when it is a plausible calendar year outside the
        metadata printed range;
      - number embedded at the start/end of a running-head line (head window
        only, first 3 lines): '32 Evsey D. Domar', 'Causes of Slavery 25',
        '404 Canadian Journal of Sociology'; guarded against footnotes,
        'VOL./NO./NUMBER' banners, year ranges, pp.-citations, DOIs/URLs.
  * JSTOR / Taylor&Francis cover pages (unpaginated boilerplate) are detected
    by signature, excluded from candidate extraction and from the predicted
    printed range. The JSTOR cover 'pp. A-B' range is captured as independent
    evidence (it is authoritative even when metadata.csv is wrong).
  * Offset = mode of per-page votes; confidence = share of voting pages that
    agree with the mode. Pages contributing no agreeing candidate are
    reported (covers, scan first pages, figure-only pages, ...).
  * Metadata match uses the range START as anchor (tolerance 1) and requires
    the predicted end not to overrun the declared end by more than 1; PDFs
    that end BEFORE the declared range end are legitimate (common) and only
    get a soft note.

Stdlib only, deterministic, UTF-8 everywhere.

Usage:
  python scripts/detect_page_offsets.py \
      --page-text data/page_text --metadata metadata.csv \
      --out revision_notes/page_offset_report.json
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- constants

# Adobe oldstyle-figure private-use digits -> ASCII digits.
PUA_DIGITS = {0xF730 + d: ord('0') + d for d in range(10)}

# Lines dropped before taking the head/tail windows (they hide real footers).
BOILER = [
    re.compile(r'^This content downloaded from', re.I),
    re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\s+on\s.*UTC', re.I),
    re.compile(r'^All use subject to ', re.I),
    re.compile(r'academic\.oup\.com'),                       # OUP watermark line
    re.compile(r'Published online by Cambridge University Press'),
    re.compile(r'^Downloaded from\b', re.I),
    re.compile(r'^https?://\S+$'),                           # bare URL / DOI line
    re.compile(r'^Explorations in Economic History\s+\d+\s+\(\d{4}\)\s+\d{5,}\s*$'),  # eLocator header
]

BARE = re.compile(r'^\d{1,4}$')
# Pure-number neighbour line (chart ticks, table cells, decimals). Kept tight:
# '(1995).' (citation year fragment) must NOT count, or real footers next to
# footnote text get rejected (Goldin_2006 p2).
NUMISH = re.compile(r'^[\d.,]+$')
LEAD = re.compile(r'^(\d{1,4})\s+(.+)$')
TRAIL = re.compile(r'^(.*?[^\s\d\-–—])\s+(\d{1,4})$')
# Lines never treated as running heads (citations, DOIs, prices, OCR ranges).
BADLINE = re.compile(r'(?i)\bpp?\.?\s*\d|https?://|\bdoi\b|©|\bissn\b|%|\$')
MONTHS = re.compile(r'(?i)^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s,]')
STOPTOK = {'NO', 'NUMBER', 'VOL', 'VOLUME', 'PART', 'TABLE', 'FIGURE', 'FIG',
           'CHAPTER', 'SECTION', 'APPENDIX', 'ISSUE', 'PAGE', 'PLATE'}

RANGE_PP = re.compile(r'(?i)pp\.?\s*(\d{1,4})\s*[-–—]{1,2}\s*(\d{1,4})')
RANGE_PAGES = re.compile(r'(?i)pages\s+(\d{1,4})\s*[-–—]\s*(\d{1,4})')
OUP_URL = re.compile(r'academic\.oup\.com/\w+/article(?:-abstract)?/(\d+)/(\d+)/(\d+)/')

YEAR_LO, YEAR_HI = 1500, 2099


# ---------------------------------------------------------------- helpers

def parse_pages_field(s):
    """Parse metadata 'pages' -> ((start, end), kind) with kind in
    {'range', 'single', None}. '8--38' -> ((8, 38), 'range');
    '101424' -> ((101424, 101424), 'single') (eLocator suspect)."""
    s = (s or '').strip()
    if not s:
        return None, None
    m = re.search(r'(\d{1,5})\s*[-–—]{1,2}\s*(\d{1,5})', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b >= a:
            return (a, b), 'range'
        return None, None
    if re.fullmatch(r'\d{1,7}', s):
        v = int(s)
        return (v, v), 'single'
    return None, None


def plausible_page_number(v, meta_rng):
    """Reject 0 and calendar years 1500-2099 unless the declared printed
    range actually spans such values."""
    if v <= 0:
        return False
    if YEAR_LO <= v <= YEAR_HI:
        return bool(meta_rng) and (meta_rng[0] - 3) <= v <= (meta_rng[1] + 3)
    return True


def is_boiler(line):
    return any(rx.search(line) for rx in BOILER)


def numericish(line):
    return bool(NUMISH.fullmatch(line)) and any(c.isdigit() for c in line)


def page_candidates(lines, meta_rng):
    """Extract printed-page-number candidates from the head/tail windows of
    the non-boilerplate lines. Bare numbers are taken from the first 4 / last
    4 lines (EHR-style footers stack number + AUTHORS + (c) + venue = 4
    lines); embedded running-head forms only from the first 3.
    Returns list of (value, kind, line_idx, line)."""
    n = len(lines)
    out = []
    idxs = sorted(set(list(range(min(4, n))) + list(range(max(0, n - 4), n))))
    for i in idxs:
        ln = lines[i]
        # --- bare number on its own line (head or tail) -------------------
        if BARE.fullmatch(ln):
            v = int(ln)
            if len(ln) > 1 and ln[0] == '0':
                continue                       # '07' etc: not a page number
            if not plausible_page_number(v, meta_rng):
                continue
            prev_num = i > 0 and numericish(lines[i - 1])
            next_num = i + 1 < n and numericish(lines[i + 1])
            if prev_num or next_num:
                continue                       # chart ticks / table cell run
            out.append((v, 'bare', i, ln))
            continue
        # --- embedded running-head forms: physical top of page only -------
        if i > 2:
            continue
        if BADLINE.search(ln):
            continue
        m = LEAD.match(ln)
        if m:
            v = int(m.group(1))
            rest = m.group(2).strip()
            if (plausible_page_number(v, meta_rng) and len(rest) >= 4
                    and rest[0].isalpha() and rest[0].isupper()
                    and not MONTHS.match(rest)):
                out.append((v, 'lead', i, ln))
                continue
        m = TRAIL.match(ln)
        if m:
            pre = m.group(1)
            v = int(m.group(2))
            lastch = pre[-1]
            lasttok = pre.split()[-1].rstrip('.,:;').upper()
            if (plausible_page_number(v, meta_rng) and len(pre) >= 4
                    and (lastch.isalpha() or lastch in ')?.’')
                    and lasttok not in STOPTOK):
                out.append((v, 'trail', i, ln))
    return out


def detect_front_matter(raw):
    """True if the page is an unpaginated publisher cover (JSTOR / T&F)."""
    if 'jstor.org/stable' in raw and 'JSTOR is a not-for-profit service' in raw:
        return 'jstor_cover'
    if 'tandfonline.com/action/journalInformation' in raw:
        return 'tf_cover'
    if 'To cite this article' in raw and 'tandfonline' in raw:
        return 'tf_cover'
    return None


# ---------------------------------------------------------------- per doc

def process_doc(doc, pages_raw, meta_row):
    meta_str = (meta_row or {}).get('pages', '') if meta_row else ''
    meta_rng, meta_kind = parse_pages_field(meta_str)

    page_nums = sorted(pages_raw)
    n_pages = len(page_nums)
    missing = [i for i in range(1, (page_nums[-1] if page_nums else 0) + 1)
               if i not in pages_raw]

    translated = {p: pages_raw[p].translate(PUA_DIGITS) for p in page_nums}

    front, front_kinds = [], {}
    jstor_footprint = 0
    page_votes = {}          # page -> {offset: (value, kind, idx, line)}
    for p in page_nums:
        raw = translated[p]
        fm = detect_front_matter(raw)
        if fm:
            front.append(p)
            front_kinds[p] = fm
        if 'about.jstor.org/terms' in raw or 'This content downloaded' in raw:
            jstor_footprint += 1
        lines = [l.strip() for l in raw.splitlines()]
        lines = [l for l in lines if l and not is_boiler(l)]
        cands = [] if fm else page_candidates(lines, meta_rng)
        offs = {}
        for (v, kind, i, ln) in cands:
            o = v - p
            if o not in offs:
                offs[o] = (v, kind, i, ln)
        page_votes[p] = offs

    votes = Counter()
    for p in page_nums:
        for o in page_votes[p]:
            votes[o] += 1

    offset = None
    confidence = 0.0
    n_voting = sum(1 for p in page_nums if page_votes[p])
    n_agree = 0
    if votes:
        # deterministic: most page-votes, then smallest |offset|, then value
        offset = sorted(votes.items(), key=lambda kv: (-kv[1], abs(kv[0]), kv[0]))[0][0]
        n_agree = sum(1 for p in page_nums if offset in page_votes[p])
        confidence = round(n_agree / n_voting, 3) if n_voting else 0.0

    # pages contributing no agreeing candidate (covers, scan first pages, ...)
    if offset is not None:
        no_agree = [p for p in page_nums if offset not in page_votes[p]]
    else:
        no_agree = sorted(front)

    # evidence samples (first 3 agreeing pages)
    evidence = []
    if offset is not None:
        for p in page_nums:
            if offset in page_votes[p]:
                v, kind, i, ln = page_votes[p][offset]
                evidence.append({'page': p, 'printed': v, 'kind': kind,
                                 'line': ln[:80]})
                if len(evidence) == 3:
                    break

    # predicted printed range over content pages (front matter excluded)
    content = [p for p in page_nums if p not in front] or page_nums
    pred = None
    if offset is not None and content:
        pred = [offset + content[0], offset + content[-1]]

    # independent anchors
    cover_rng = None
    if page_nums and page_nums[0] in front:
        joined = ' '.join(l.strip() for l in translated[page_nums[0]].splitlines()
                          if l.strip())
        m = RANGE_PP.search(joined)
        if m:
            cover_rng = [int(m.group(1)), int(m.group(2))]
    firstpage_rng = None
    if page_nums and page_nums[0] not in front:
        joined = ' '.join(l.strip() for l in translated[page_nums[0]].splitlines()
                          if l.strip())
        m = RANGE_PP.search(joined) or RANGE_PAGES.search(joined)
        if m:
            firstpage_rng = [int(m.group(1)), int(m.group(2))]
    oup_start = None
    for p in page_nums:
        m = OUP_URL.search(translated[p])
        if m:
            oup_start = int(m.group(3))
            break

    # ---------------- validation vs metadata ---------------------------
    notes = []
    matches = False
    if offset is None:
        notes.append('no printed-page candidates detected')
    elif confidence < 0.5:
        notes.append('low confidence ({:.2f})'.format(confidence))
    if missing:
        notes.append('missing page files: {}'.format(missing[:5]))

    if meta_kind == 'range' and offset is not None and pred:
        start_ok = abs(pred[0] - meta_rng[0]) <= 1
        end_ok = pred[1] <= meta_rng[1] + 1
        matches = bool(start_ok and end_ok)
        if not start_ok:
            notes.append('start mismatch: predicted {} vs metadata {}'.format(
                pred[0], meta_rng[0]))
        if not end_ok:
            notes.append('predicted end {} overruns metadata end {}'.format(
                pred[1], meta_rng[1]))
        if start_ok and end_ok and (meta_rng[1] - pred[1]) >= 2:
            notes.append('note: pdf ends early (last predicted printed {}, '
                         'metadata end {})'.format(pred[1], meta_rng[1]))
    elif meta_kind == 'single':
        notes.append("metadata 'pages' is a single value {!r} (eLocator?); "
                     'range match impossible'.format(meta_str))
    elif meta_kind is None:
        notes.append("metadata 'pages' missing/unparsable: {!r}".format(meta_str))

    if cover_rng and offset is not None and pred:
        if abs(cover_rng[0] - pred[0]) <= 1:
            if meta_kind == 'range' and not matches:
                notes.append('JSTOR cover pp. {}-{} AGREES with detection; '
                             'metadata row looks wrong'.format(*cover_rng))
        else:
            notes.append('JSTOR cover pp. {}-{} disagrees with predicted '
                         'start {}'.format(cover_rng[0], cover_rng[1], pred[0]))

    anomalous = (offset is None) or (not matches) or (confidence < 0.5)

    return {
        'doc': doc,
        'n_pages': n_pages,
        'offset': offset,
        'confidence': confidence,
        'n_voting_pages': n_voting,
        'n_agreeing_pages': n_agree,
        'cover_pages': no_agree,
        'front_matter_pages': sorted(front),
        'front_matter_kinds': front_kinds,
        'jstor_doc': bool(n_pages and jstor_footprint >= n_pages / 2.0),
        'printed_range_meta': meta_str,
        'printed_range_meta_parsed': list(meta_rng) if meta_rng else None,
        'printed_range_pred': pred,
        'matches_metadata': matches,
        'jstor_cover_range': cover_rng,
        'firstpage_citation_range': firstpage_rng,
        'oup_url_start_page': oup_start,
        'anomalous': anomalous,
        'anomaly': '; '.join(notes),
        'evidence': evidence,
    }


# ---------------------------------------------------------------- driver

def load_corpus(page_text_dir):
    docs = {}
    for f in sorted(Path(page_text_dir).glob('*_page_*.txt')):
        stem = f.stem
        doc, sep, num = stem.rpartition('_page_')
        if not sep or not num.isdigit():
            continue
        docs.setdefault(doc, {})[int(num)] = f.read_text(
            encoding='utf-8-sig', errors='replace')
    return docs


def main(argv=None):
    ap = argparse.ArgumentParser(description='Detect per-doc printed-page offsets.')
    ap.add_argument('--page-text', default='data/page_text')
    ap.add_argument('--metadata', default='metadata.csv')
    ap.add_argument('--out', default='revision_notes/page_offset_report.json')
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    meta = {}
    with open(args.metadata, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            doc_id = (row.get('doc_id') or '').strip()
            if doc_id:
                meta[doc_id] = row

    corpus = load_corpus(args.page_text)
    if not corpus:
        print('ERROR: no page files found under', args.page_text)
        return 2

    records = []
    for doc in sorted(corpus):
        records.append(process_doc(doc, corpus[doc], meta.get(doc)))

    n_pages_total = sum(r['n_pages'] for r in records)
    docs_no_meta = sorted(set(corpus) - set(meta))
    meta_no_docs = sorted(set(meta) - set(corpus))

    summary = {
        'n_docs': len(records),
        'n_pages_total': n_pages_total,
        'n_offset_detected': sum(1 for r in records if r['offset'] is not None),
        'n_matches_metadata': sum(1 for r in records if r['matches_metadata']),
        'n_anomalous': sum(1 for r in records if r['anomalous']),
        'anomalous_docs': [r['doc'] for r in records if r['anomalous']],
        'docs_without_metadata': docs_no_meta,
        'metadata_without_docs': meta_no_docs,
        'offset_definition': 'printed_page = pdf_page_index + offset',
        'confidence_definition': 'agreeing_pages / pages_with_any_candidate',
        'matches_metadata_definition': ('|pred_start - meta_start| <= 1 AND '
                                        'pred_end <= meta_end + 1 (pdfs may end '
                                        'before the published range end)'),
        'cover_pages_definition': ('pdf pages contributing no candidate that '
                                   'agrees with the modal offset (publisher '
                                   'covers, scan first pages, figure-only and '
                                   'unnumbered final pages)'),
    }
    report = {'generated_by': 'scripts/detect_page_offsets.py',
              'params': {'page_text': args.page_text, 'metadata': args.metadata},
              'summary': summary,
              'docs': records}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=True)
        fh.write('\n')

    # ---------------- compact stdout table -----------------------------
    hdr = '{:<28} {:>5} {:>7} {:>5} {:>12} {:>12} {:>5}  {}'.format(
        'doc', 'pages', 'offset', 'conf', 'pred', 'meta', 'match', 'anomaly')
    print(hdr)
    print('-' * len(hdr))
    for r in records:
        pred = ('{}-{}'.format(*r['printed_range_pred'])
                if r['printed_range_pred'] else '-')
        mrng = ('{}-{}'.format(*r['printed_range_meta_parsed'])
                if r['printed_range_meta_parsed'] else '-')
        print('{:<28} {:>5} {:>7} {:>5} {:>12} {:>12} {:>5}  {}'.format(
            r['doc'][:28], r['n_pages'],
            '-' if r['offset'] is None else '{:+d}'.format(r['offset']),
            '{:.2f}'.format(r['confidence']), pred, mrng,
            'yes' if r['matches_metadata'] else 'NO',
            (r['anomaly'][:70] if r['anomaly'] else '')))
    print('-' * len(hdr))
    print('docs={} pages={} offsets_detected={} match_meta={} anomalous={}'.format(
        summary['n_docs'], summary['n_pages_total'],
        summary['n_offset_detected'], summary['n_matches_metadata'],
        summary['n_anomalous']))
    print('report written to', out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
