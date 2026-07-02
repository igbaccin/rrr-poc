---
name: rrr-review
description: Produce a verifiable, page-cited literature review from a folder of PDFs using the RRR (Retrieval-Restricted Reasoning) pipeline. Use when the user asks for a literature review, evidence-grounded synthesis, or corpus-restricted summary of a set of papers/documents, and wants citations they can actually check. Requires python 3.10+ and either a local Ollama server or an Anthropic API key.
---

# RRR literature review

You are orchestrating the RRR pipeline — the field-owned scholarly-contract
tool — NOT writing the review yourself. Your job is setup, invocation, and
honest reporting. Never write or edit the review text; never bypass the
pipeline's validation (do not set RRR_BYPASS_VALIDATION).

## What RRR guarantees (why you must not improvise)

Every citation names a corpus document and a real page; quotes are verified
verbatim against the page text; claims outside the corpus's evidence are
refused rather than papered over; every run leaves an audit trail
(`review_ledger.json`, `run_manifest.json`, `quality_manifest.json`,
`citations.json`) a third party can check.

## Workflow

1. **Install (once per workspace).**
   - `pip install rrr-poc` if available on the index; otherwise
     `git clone https://github.com/igbaccin/rrr-poc && pip install -e rrr-poc`.
   - Verify: `rrr --help` (or `python -m rrr.cli --help` with
     `PYTHONPATH=<repo>/src`).

2. **Choose the LLM runtime.**
   - Local: Ollama running with `mistral-small:24b` pulled (16 GB VRAM), or
     a smaller tier via `RRR_MODEL_LATIN` on modest hardware.
   - API: `export RRR_RUNTIME=api` and ensure `ANTHROPIC_API_KEY` is set
     (ask the user; NEVER read or echo the key's value).

3. **Ingest the corpus (confidence-gated).**
   - `rrr ingest --corpus <pdf_folder> --output metadata.csv`
     (add `--bib <file.bib>` if the user has one — it upgrades confidence).
   - Exit 3 means rows need review: SHOW the user the pending table from
     `ingest_report.json` and ask them to confirm, correct, or drop each
     row. Only rerun with `--accept-low-confidence` if the user explicitly
     approves. Do not approve on their behalf.

4. **Preprocess + index (once per corpus).**
   - `python -m rrr.preprocess --metadata metadata.csv`
   - `python -m rrr.index --metadata metadata.csv`

5. **Run the review.**
   - `rrr t2 --multi --metadata metadata.csv --topic "<the user's topic>"`
   - Reuse existing metadata/indices on repeat topics over the same corpus;
     re-run ingest only when the PDF set changes.

6. **Report honestly.**
   - Deliver `runs/<run_id>/review_composed.md` verbatim, plus one short
     paragraph pointing at the audit artifacts in the same directory.
   - If the pipeline REFUSES (insufficient evidence, off-topic corpus),
     report the refusal and its recorded reason as the correct outcome —
     do not write a substitute review, do not retry with weakened
     thresholds.
   - Report `topic_fit` warnings to the user when present.

## Hard rules

- Never edit `review_composed.md` beyond delivering it.
- Never set `RRR_BYPASS_VALIDATION`, lower refusal thresholds, or delete
  runs to hide failures.
- Never cite anything yourself; only the pipeline cites.
- If any step fails, show the actual error and the pipeline's logs; do not
  improvise a fallback review from your own knowledge.
