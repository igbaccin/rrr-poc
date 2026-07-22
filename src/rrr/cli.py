import argparse, json, os, re, sys


# v15.10: deterministic sanity gate before any LLM call. Rejects topics that
# are obviously not scholarly questions — keyboard slams, all-punctuation
# strings, single-word fragments. Kills the descriptive+PROCEED loophole for
# word-salad topics that reach Stage 0 with real English tokens and no
# coherent meaning. The gate is intentionally lenient: real scholarly topics
# with unusual phrasings must not be rejected. If in doubt, let it through
# and rely on Stage 0.
#
# v15.11: Unicode-aware. Recognises non-Latin scholarly topics (CJK, Arabic,
# Cyrillic, Devanagari) as legitimate. The letter-count minimum uses
# Unicode isalpha() so each CJK character counts as one letter. Multi-token
# rule fires only when whitespace is present (Latin, Cyrillic, Arabic, etc.);
# scriptio-continua languages (CJK) get a special path that rejects only
# when the input mixes letters with digits or punctuation (the keyboard-slam
# signature: '09p<GYHKLGCH' is a single letter run alongside symbols, while
# '机构改革与经济增长' is a pure letter run).
_MIN_LETTER_CODEPOINTS = 4
_MIN_MULTI_LETTER_TOKENS = 2
_MULTI_LETTER_TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _reject_gibberish_topic(topic: str) -> str | None:
    """Return a rejection reason if the topic is obviously not scholarly,
    else None. Applied at CLI entry before any pipeline setup or LLM call.
    """
    if topic is None:
        return "topic is None"
    stripped = topic.strip()
    if not stripped:
        return "topic is empty"
    letter_count = sum(1 for c in stripped if c.isalpha())
    if letter_count < _MIN_LETTER_CODEPOINTS:
        return (f"topic has {letter_count} letter codepoint(s) "
                f"(<{_MIN_LETTER_CODEPOINTS}); not a scholarly research question")
    if re.search(r"\s", stripped):
        tokens = _MULTI_LETTER_TOKEN_RE.findall(stripped)
        if len(tokens) < _MIN_MULTI_LETTER_TOKENS:
            return (f"topic has {len(tokens)} multi-letter token(s) "
                    f"(<{_MIN_MULTI_LETTER_TOKENS}); not a scholarly "
                    "research question")
    else:
        non_letter = sum(1 for c in stripped
                          if not c.isalpha() and not c.isspace())
        if non_letter > 0:
            return ("single-run topic mixes letters with digits or "
                    "punctuation; looks like a keyboard slam")
    return None


def t2(args, meta_path):
    # v13: cli.t2 now only dispatches to the live multi-pass path. The legacy
    # single-pass T2 (`t2` without --multi) was the v6-era strict reasoner
    # entrypoint; the layered_t2 architecture (v8+) supersedes it end-to-end
    # and is what every battery script invokes. Same for the v6-era t1 (single
    # claim, no writer) and t3 (page extraction) entrypoints — t3 stays retired
    # (v13 removed). t1 REVIVED in v15.9 as a claim-evaluator mode: same
    # pipeline as T2 up to Stage 2 posture, but stops before the writer and
    # emits claim_verdict.md instead of review_composed.md.
    if not getattr(args, "multi", False):
        raise SystemExit(
            "cli.t2 now requires --multi. The single-pass path was retired in v13; "
            "use scripts/run_small_validation.py or scripts/run_battery.sh for the "
            "full layered pipeline."
        )
    # v15.11: deferred import so env vars set by main() (RRR_MODEL,
    # RRR_TOPIC_LANG) take effect before reasoner reads them at import time.
    from rrr.reasoner import layered_t2
    layered_t2(args, meta_path)


def t1(args, meta_path):
    """v15.9 revival: claim-eval mode. Same pipeline as T2 up to Stage 2
    posture, then stops. Emits runs/claim_verdict.md with per-cluster + aggregate
    breakdown of how the corpus stands on the claim. ~15-20% of T2's runtime
    because it skips the writer + validation chain."""
    # We reuse layered_t2's whole entry, gated by args.t1_only which the
    # reasoner short-circuits on after Stage 2 + ledger write.
    setattr(args, "multi", True)
    setattr(args, "t1_only", True)
    # narrative-only saves us the T2_review.md appendix step we don't need.
    setattr(args, "narrative_only", True)
    # v15.11: deferred import (see t2 comment).
    from rrr.reasoner import layered_t2
    layered_t2(args, meta_path)


def ingest_main(argv):
    """v16: `rrr ingest` — the confidence-gated front door for arbitrary-
    filename corpora (wires src/rrr/ingest.py into the main path; it was
    previously eval-only).

    Gate contract (per revision_notes design): every metadata field carries
    a recorded source; rows that are LOW confidence, FAILED, or purely
    LLM-inferred without external corroboration do NOT enter the canonical
    metadata.csv. They are written to <output>.pending.csv for human review
    and the command exits 3 until the user either hand-edits them in or
    reruns with --accept-low-confidence (an explicit, logged choice).
    """
    ap = argparse.ArgumentParser(
        prog="rrr ingest",
        description="Build metadata.csv from a folder of PDFs (arbitrary "
                    "filenames) via the confidence-gated ingest cascade.")
    ap.add_argument("--corpus", required=True, help="Folder of PDFs")
    ap.add_argument("--output", default="metadata.csv")
    ap.add_argument("--bib", default=None, help="Optional BibTeX sidecar")
    ap.add_argument("--no-llm", action="store_true",
                    help="Disable the LLM-extraction rung of the cascade")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--no-openalex", action="store_true")
    ap.add_argument("--accept-low-confidence", action="store_true",
                    help="Explicitly admit pending rows into metadata.csv "
                         "(recorded in the ingest report)")
    ap.add_argument("--report", default=None,
                    help="Ingest report path (default: <output dir>/ingest_report.json)")
    args = ap.parse_args(argv)

    from pathlib import Path
    from rrr.ingest import ingest_corpus, _write_metadata_csv

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        sys.stderr.write(f"[ingest] corpus folder not found: {corpus_dir}\n")
        raise SystemExit(2)
    out_path = Path(args.output)
    report_path = Path(args.report) if args.report else (
        out_path.parent / "ingest_report.json")

    results = ingest_corpus(
        corpus_dir,
        output_csv=None,  # gate BEFORE anything reaches the canonical file
        sidecar_bib=Path(args.bib) if args.bib else None,
        use_llm=not args.no_llm,
        use_crossref=not args.no_crossref,
        use_openalex=not args.no_openalex,
    )

    def _needs_review(m):
        if m.confidence in ("low", "failed"):
            return True
        # Purely LLM-inferred with no external corroboration requires human
        # approval regardless of the model's own confidence.
        return m.source == "llm_extraction"

    approved = [m for m in results if not _needs_review(m)]
    pending = [m for m in results if _needs_review(m)]
    unusable = [m for m in results if not m.doc_id or not m.pdf_path]

    def _row_status(m):
        if m in unusable:
            return "correction_required"
        if m in approved:
            return "approved"
        if args.accept_low_confidence:
            return "accepted_by_flag"
        return "pending_review"

    report = {
        "corpus": str(corpus_dir),
        "n_pdfs": len(results),
        "n_approved": len(approved),
        "n_pending": len(pending),
        "accept_low_confidence": bool(args.accept_low_confidence),
        "rows": [
            {"doc_id": m.doc_id, "pdf": m.pdf_path, "title": m.title,
             "authors": m.authors_short, "year": m.year,
             "confidence": m.confidence, "source": m.source,
             "lang": m.lang, "notes": m.notes,
             "status": _row_status(m)}
            for m in results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.accept_low_confidence and unusable:
        _write_metadata_csv(approved, out_path)
        pending_path = out_path.with_suffix(".pending.csv")
        _write_metadata_csv(pending, pending_path)
        print(
            f"[ingest] {len(unusable)} row(s) still lack the metadata needed "
            "for indexing. Approval alone cannot admit them."
        )
        for m in unusable:
            print(
                f"  CORRECT: {m.pdf_path}  doc_id={m.doc_id!r} "
                f"source={m.source}"
            )
        print(f"[ingest] Correct the rows in {pending_path} and rerun.")
        print(f"[ingest] report: {report_path}")
        raise SystemExit(3)

    if pending and not args.accept_low_confidence:
        _write_metadata_csv(approved, out_path)
        pending_path = out_path.with_suffix(".pending.csv")
        _write_metadata_csv(pending, pending_path)
        print(f"[ingest] {len(approved)} rows -> {out_path}")
        print(f"[ingest] {len(pending)} rows NEED REVIEW -> {pending_path}")
        for m in pending:
            print(f"  REVIEW: {m.pdf_path}  confidence={m.confidence} "
                  f"source={m.source}  ({m.notes or 'no notes'})")
        print("[ingest] Inspect the pending rows, then either hand-correct "
              "them into metadata.csv or rerun with --accept-low-confidence.")
        print(f"[ingest] report: {report_path}")
        raise SystemExit(3)

    _write_metadata_csv(results, out_path)
    print(f"[ingest] {len(results)} rows -> {out_path} "
          f"({len(pending)} admitted via --accept-low-confidence)" if pending
          else f"[ingest] {len(results)} rows -> {out_path}")
    print(f"[ingest] report: {report_path}")


def host_doctor_main(argv):
    """Verify that native host mode will use a product subscription."""
    ap = argparse.ArgumentParser(
        prog="rrr host-doctor",
        description="Verify Codex or Claude subscription-backed host mode.",
    )
    ap.add_argument("--host", choices=["codex", "claude"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run one subscription-backed inference call after the auth check.",
    )
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    if args.host:
        os.environ["RRR_HOST"] = args.host
    if args.model:
        os.environ["RRR_HOST_MODEL"] = args.model

    from rrr.host_backend import (
        HostBackendError,
        host_diagnostics,
        host_smoke_test,
    )

    try:
        status = host_diagnostics()
        if args.smoke:
            status.update(host_smoke_test())
    except HostBackendError as exc:
        sys.stderr.write(f"[RRR] host doctor failed: {exc}\n")
        raise SystemExit(2)

    if args.as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    auth = status["authentication"]
    subscription = auth.get("subscription_type")
    suffix = f" ({subscription})" if subscription else ""
    print(
        f"[RRR] authentication ready: {status['host']} {status['model']} via "
        f"{auth['method']} subscription{suffix}"
    )
    print(f"[RRR] executable: {status['executable']}")
    if status.get("inference_ready"):
        print("[RRR] inference smoke test: passed")
    else:
        print(
            "[RRR] inference was not tested. Add --smoke to spend one "
            "subscription call on a JSON contract check."
        )


def prepare_main(argv):
    """Prepare a managed RRR workspace from a folder containing PDFs."""
    ap = argparse.ArgumentParser(
        prog="rrr prepare",
        description=(
            "Find or select a folder of PDFs, then create a reusable RRR "
            "metadata catalog and page index."
        ),
    )
    ap.add_argument(
        "corpus",
        nargs="?",
        help="Folder containing PDFs. When omitted, inspect the current folder.",
    )
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--bib", default=None, help="Optional BibTeX sidecar")
    ap.add_argument("--with-llm-metadata", action="store_true")
    ap.add_argument("--accept-low-confidence", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    from pathlib import Path
    from rrr.product_workspace import (
        CorpusDiscoveryError,
        discover_corpus_dir,
        prepare_workspace,
    )

    try:
        corpus_dir = (
            Path(args.corpus).expanduser().resolve()
            if args.corpus
            else discover_corpus_dir(Path.cwd())
        )
        bibliography = Path(args.bib) if args.bib else None
        if bibliography is None:
            for candidate in (
                corpus_dir / "bibliography.bib",
                corpus_dir.parent / "bibliography.bib",
            ):
                if candidate.is_file():
                    bibliography = candidate
                    break
        prepared = prepare_workspace(
            corpus_dir=corpus_dir,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            bibliography=bibliography,
            use_llm=args.with_llm_metadata,
            accept_low_confidence=args.accept_low_confidence,
            force=args.force,
            quiet=args.as_json,
        )
    except CorpusDiscoveryError as exc:
        error = {
            "error": str(exc),
            "candidates": [str(path) for path in exc.candidates],
        }
        if args.as_json:
            print(json.dumps(error, ensure_ascii=False))
        else:
            sys.stderr.write(f"[RRR] {error['error']}\n")
            for candidate in error["candidates"]:
                sys.stderr.write(f"  {candidate}\n")
        raise SystemExit(2)

    result = prepared.as_dict()
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        action = "reused" if prepared.reused else "prepared"
        print(
            f"[RRR] {action} {prepared.pdf_count} PDFs in "
            f"{prepared.workspace_dir}"
        )
        print(f"[RRR] metadata: {prepared.metadata_path}")


def _select_topic_runtime(topic):
    """Detect the topic language, then validate the configured runtime."""
    from rrr.language import detect_topic_language, select_model

    try:
        topic_lang = detect_topic_language(topic)
    except Exception as lang_e:
        topic_lang = "en"
        sys.stderr.write(
            f"[RRR] language detector unavailable ({lang_e}); "
            "falling back to topic_lang=en\n"
        )
    return topic_lang, select_model(topic_lang)


def main():
    # v15.12: force UTF-8 on stdout/stderr so multilingual output (accented
    # Latin, CJK, Arabic, Cyrillic) prints correctly regardless of the host
    # locale. On a POSIX/C locale (common on cloud pods) Python may otherwise
    # pick an ASCII console encoding and mangle non-ASCII prints. File writes
    # already pass encoding="utf-8" explicitly; this covers the console path.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # v16: `rrr ingest` has its own argument contract; dispatch before the
    # t1/t2 parser (whose --metadata/--topic are required) sees the argv.
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "host-doctor":
        host_doctor_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare_main(sys.argv[2:])
        return

    ap = argparse.ArgumentParser(
        description="RRR CLI. `t2` runs the full literature-review pipeline; "
                    "`t1` runs the claim-evaluator (same up to Stage 2, no writer)."
    )
    ap.add_argument("task", choices=["t1", "t2"],
                    help="Which pipeline task to run.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--topic", required=True,
                    help="Topic (T2) or claim (T1) to evaluate.")
    ap.add_argument("--multi", action="store_true",
                    help="Run the layered T2 pipeline (default for both t1 and t2).")
    ap.add_argument("--narrative-only", action="store_true",
                    help="T2: skip the T2_review.md appendix. Ignored by t1.")
    ap.add_argument("--linkify", action="store_true",
                    help="T2: rewrite in-text citations as clickable markdown links "
                         "to the source PDF page. Sets RRR_LINKIFY=1.")
    args = ap.parse_args()

    reject = _reject_gibberish_topic(args.topic)
    if reject:
        sys.stderr.write(f"[RRR] refusing to run: {reject}\n")
        sys.stderr.write(f"[RRR] topic was: {args.topic!r}\n")
        raise SystemExit(2)

    if args.linkify:
        os.environ["RRR_LINKIFY"] = "1"

    # Resolve the model before importing the reasoner, whose modules bind the
    # selection at import time. Language-detection failures can use English as
    # a fallback. Runtime and provider configuration errors remain fatal.
    topic_lang, selected_model = _select_topic_runtime(args.topic)
    os.environ["RRR_TOPIC_LANG"] = topic_lang
    os.environ["RRR_MODEL"] = selected_model
    sys.stderr.write(
        f"[RRR] topic_lang={topic_lang} selected_model={selected_model}\n"
    )

    if args.task == "t1":
        t1(args, args.metadata)
    else:
        t2(args, args.metadata)


if __name__ == "__main__":
    main()
