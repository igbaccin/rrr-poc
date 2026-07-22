---
name: rrr
description: Produce a verifiable, page-cited literature review from a user-supplied PDF corpus with Retrieval-Restricted Reasoning. Use when the user requests corpus-bounded synthesis, an evidence-grounded literature review, or a claim evaluation with citations that can be checked. Requires Python 3.10+ and either local Ollama or a configured Anthropic or OpenAI API account.
---

# RRR literature review

Use the RRR pipeline to process the corpus and generate the review. Your role is to prepare the workspace, run the commands, inspect the recorded outcome, and deliver the resulting artifacts. The pipeline owns the review text and citations.

## Required workflow

1. Verify the local checkout.

   - Resolve the repository root containing `pyproject.toml` and `src/rrr/`.
   - Run `rrr --help` or `python -m rrr.cli --help`.
   - When the command is unavailable, install the current local checkout with `python -m pip install -e <repo-root>`. Do not substitute a remote package for the checkout supplied by the user.

2. Choose the model runtime.

   - For local inference, verify that Ollama is reachable and that the selected model is installed.
   - For API inference, set `RRR_RUNTIME=api`, `RRR_API_PROVIDER`, and `RRR_API_MODEL` as needed. Confirm that the provider credential exists without reading, printing, or storing its value.

3. Establish a corpus workspace.

   - Keep the user's PDFs under `<workspace>/corpus/`.
   - Set `RRR_PROJECT_ROOT=<workspace>` so derived artifacts stay with the corpus.
   - Keep PDFs, metadata, indices, caches, and run outputs outside version control.

4. Build the metadata catalog.

   - Run `rrr ingest --corpus <workspace>/corpus --output <workspace>/metadata.csv`.
   - Add `--bib <file.bib>` when the user supplies a BibTeX sidecar.
   - Exit status `3` means that one or more records require review. Show the pending records to the user and request a decision. Use `--accept-low-confidence` only after explicit approval.

5. Preprocess and index the corpus.

   - Run `python -m rrr.preprocess --metadata <workspace>/metadata.csv`.
   - Run `python -m rrr.index --metadata <workspace>/metadata.csv`.
   - Stop and report any reference-section invariant failure. Do not continue with an incomplete index.

6. Run the requested task.

   - Claim evaluation: `rrr t1 --metadata <workspace>/metadata.csv --topic "<claim>"`.
   - Literature review: `rrr t2 --multi --metadata <workspace>/metadata.csv --topic "<topic>"`.
   - Reuse the metadata and index for later questions over an unchanged corpus.

7. Report the recorded outcome.

   - Deliver `review_composed.md` verbatim for a successful `t2` run.
   - Deliver `claim_verdict.md` verbatim for a successful `t1` run.
   - Identify the associated `review_ledger.json`, `citations.json`, `quality_manifest.json`, and `run_manifest.json` when present.
   - Report corpus-fit warnings and refusal reasons in plain language.

## Integrity rules

- Never edit generated review or verdict text before delivery.
- Never set `RRR_BYPASS_VALIDATION` during ordinary use.
- Never lower evidence or refusal thresholds to force an answer.
- Never add citations from personal knowledge or web search.
- Never approve uncertain metadata on the user's behalf.
- Never expose provider credentials or include them in logs.
- Never replace a recorded refusal with an improvised review.
- Preserve failed-run artifacts when diagnosing an error.
