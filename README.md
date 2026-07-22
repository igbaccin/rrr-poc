# Retrieval-Restricted Reasoning (RRR)

**About RRR** · [Installation modes](docs/product_installation.md) · [Before running](docs/before_running.md)

RRR produces page-cited literature reviews and claim evaluations from a
collection of PDFs supplied by the user. It is designed to preserve a
checkable path from the released prose to the source page.

## What RRR does

RRR supports two tasks.

- **T1 evaluates a claim.** It compares a statement with the evidence found in
  the declared PDF collection.
- **T2 writes a literature review.** It retrieves relevant pages, composes a
  review, and renders author, year, and page citations.

The compact commands are:

```text
$rrr t1 <claim>       Codex
$rrr t2 <topic>       Codex
/rrr t1 <claim>       Claude Code
/rrr t2 <topic>       Claude Code
```

RRR can also be run directly from a terminal.

## One pipeline with several ways to run it

The user chooses how to start RRR and which model performs its language-model
calls. The retrieval, source checks, citation validation, and run records use
the same RRR code in every mode.

| Setup | Model used | Where to begin |
| --- | --- | --- |
| Codex with `$rrr` | GPT through the user's Codex subscription | [Install the Codex plugin](docs/product_installation.md#option-1-codex-with-the-codex-subscription) |
| Claude Code with `/rrr` | Claude through the user's Claude subscription | [Install the Claude skill](docs/product_installation.md#option-2-claude-code-with-the-claude-subscription) |
| Terminal | Mistral or Qwen through Ollama | [Install the local command-line setup](docs/product_installation.md#option-3-local-command-line-use-with-ollama) |
| Codex or Claude with local inference | Mistral or Qwen through Ollama | [Combine a skill with Ollama](docs/product_installation.md#option-4-codex-or-claude-with-local-ollama-inference) |
| Provider API | OpenAI or Anthropic API model | [Configure API mode](docs/product_installation.md#option-5-provider-api-mode) |

The [installation guide](docs/product_installation.md) explains the software,
account, and hardware requirements for each route.

## From PDFs to a released review

Every run follows the same broad path.

1. The user declares a folder of PDFs.
2. RRR checks that the files contain usable text and builds a bibliographic
   catalogue called `metadata.csv`.
3. RRR extracts and indexes the corpus page by page.
4. The selected model works only with passages admitted from that corpus.
5. RRR validates the citations and releases the review with its audit records.

The metadata stage provides the connection between a PDF, its bibliographic
identity, and every page later cited in the review. The [before-running
guide](docs/before_running.md) explains the corpus requirements, automatic
metadata sources, optional `bibliography.bib` file, and human review process.

## What remains local

| Component | Local Ollama | Native Codex or Claude | Provider API |
| --- | --- | --- | --- |
| Original PDFs | Local | Local | Local |
| Extracted page text and search index | Local | Local | Local |
| Retrieval and source admission | Local | Local | Local |
| Citation validation and run records | Local | Local | Local |
| Language-model calls | Local Ollama | Codex or Claude subscription | OpenAI or Anthropic API |
| Evidence passages sent to a provider | No | Yes | Yes |

Native mode uses the model supplied through the signed-in Codex or Claude
product. The RRR engine, corpus index, validation stages, and audit records
continue to run on the user's computer.

## Evidence

In the reported evaluation, full RRR produced 100 clean reviews in 100 runs.
The released reviews contained no E1 through E5 citation failures under the
automated checker.

| Code | Failure checked |
| --- | --- |
| E1 | A citation points outside the declared corpus. |
| E2 | A citation points to a page that does not exist. |
| E3 | A citation does not follow the required format. |
| E4 | A quotation cannot be verified on the cited page. |
| E5 | A quotation is attributed to the wrong source. |

The accompanying paper describes the architecture, comparisons, evaluation,
and limitations in detail. The deposited records and reproduction procedure
are available in the separate
[`rrr-replication`](https://github.com/igbaccin/rrr-replication) repository.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/rrr/` | RRR source code |
| `plugins/rrr/` | Installable Codex plugin |
| `skills/rrr/` | Portable Codex and Claude skill bundle |
| `dist/` | Installable Python wheel and SHA-256 checksum |
| `docs/` | Installation and corpus-preparation guides |
| `tests/unit/` | Automated tests for the pipeline and runtime modes |

Evaluation records and generated exhibits remain in the
[`rrr-replication`](https://github.com/igbaccin/rrr-replication) repository.

## Citation and licence

The source code is distributed under the terms in `LICENSE.txt`. Please cite
the accompanying paper when using RRR in published research.

Questions about RRR can be directed to Igor Martins at
`igor.martins@ekh.lu.se`.
