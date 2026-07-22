# Before running RRR

[About RRR](../README.md) · [Installation modes](product_installation.md) · **Before running**

RRR works over a collection declared by the user. Before the first review, it
checks the PDFs, establishes their bibliographic identities, extracts their
pages, and builds the search index. This preparation is reused while the
collection remains unchanged.

## Prepare the PDF collection

Place the PDFs to be analysed together in one folder. The folder can have any
name. RRR accepts arbitrary PDF filenames and leaves the original files in
place.

Every PDF needs usable embedded text or OCR. A scanned image without OCR cannot
yet enter RRR's reliable extraction path. OCR quality also affects retrieval
and quotation verification, so the extracted text should be checked when the
source is difficult to scan.

RRR searches only the folder selected by the user. It does not expand the
collection with material from the web or from the model's general knowledge.

## What metadata RRR needs

RRR creates a catalogue called `metadata.csv`. Each row connects one PDF to a
stable document identifier and its bibliographic information.

The catalogue records:

- the document identifier used inside RRR;
- title, authors, and year;
- publication information when available;
- the PDF's original file path and content hash;
- the detected language; and
- the confidence level and source of the metadata.

This catalogue supplies the bibliographic half of every final citation. Page
extraction supplies the page number. Together they allow RRR to trace a
sentence in the released review back to the identified document and source
page.

## How the metadata catalogue is created

The user does not normally write `metadata.csv` by hand. RRR builds it during
the first preparation.

| Stage | What RRR does |
| --- | --- |
| Text check | Reads the opening pages and confirms that the PDF contains enough usable text. A scan without OCR stops here. |
| Bibliography match | Uses a nearby `bibliography.bib` file when one is available. |
| Filename hint | Uses an author-and-year filename as a matching hint. A filename alone is insufficient for accepted metadata. |
| DOI lookup | Looks for a DOI in the PDF and queries Crossref for the corresponding bibliographic record. |
| Optional model extraction | When the user permits it, asks the selected model to read bibliographic details from the opening pages. Author and year must also appear in the PDF text. A result supported only by model extraction still requires human approval. |
| Human gate | Places uncertain records in `metadata.pending.csv`, explains them in `ingest_report.json`, and pauses before indexing. |

The default preparation does not ask a language model to infer metadata. It
uses the bibliography sidecar and DOI lookup, then stops for unresolved
documents. Model-assisted extraction is an explicit optional step.

## Supplying a bibliography file

A BibTeX file gives RRR a trusted catalogue to match against the PDFs. Name it
`bibliography.bib` and place it inside the PDF folder or in the folder directly
above it. RRR detects either location automatically.

A bibliography file is optional. It is especially useful when PDFs lack a DOI,
have uninformative filenames, or come from collections whose first pages do not
follow a modern journal format.

A normal BibTeX entry is sufficient:

```bibtex
@article{NorthWeingast1989,
  author  = {North, Douglass C. and Weingast, Barry R.},
  title   = {Constitutions and Commitment},
  journal = {The Journal of Economic History},
  year    = {1989},
  volume  = {49},
  number  = {4},
  pages   = {803--832}
}
```

RRR can match a PDF to the entry through the BibTeX key, an author-and-year
filename, or bibliographic information found in the PDF text.

## Preparation through Codex or Claude Code

Open the folder containing the PDFs, then invoke `$rrr` in Codex or `/rrr` in
Claude Code. The skill runs the preparation automatically before the first
review.

RRR checks the open folder first. It also recognizes a conventional `corpus/`
subfolder or one nearby folder containing PDFs. When no collection is found,
the skill asks the user to select the correct folder. When several collections
are possible, it lists them and waits for the user's choice.

The folder name and layout do not otherwise matter.

## Prepare the corpus from a terminal

After installing the command-line package, run:

```bash
rrr prepare "PATH_TO_PDF_FOLDER"
```

RRR prints the managed workspace path and the location of `metadata.csv`. Set
the printed workspace as `RRR_PROJECT_ROOT` before running T1 or T2.

On macOS or Linux:

```bash
export RRR_PROJECT_ROOT="PATH_PRINTED_BY_RRR_PREPARE"
```

On Windows PowerShell:

```powershell
$env:RRR_PROJECT_ROOT = "PATH_PRINTED_BY_RRR_PREPARE"
```

Run a literature review with:

```bash
rrr t2 --multi --metadata "PATH_TO_METADATA.CSV" --topic "How did institutions shape long-run economic growth?"
```

Run a claim evaluation with:

```bash
rrr t1 --metadata "PATH_TO_METADATA.CSV" --topic "Institutions are the fundamental cause of long-run economic growth."
```

## When metadata needs human review

RRR accepts externally supported, high-confidence metadata automatically.
Uncertain rows are written to `metadata.pending.csv`. The corresponding
`ingest_report.json` records why each row was held back.

The user can verify the PDF and correct the bibliographic fields before
continuing. A usable low-confidence row can also be admitted through an
explicit approval:

```bash
rrr prepare "PATH_TO_PDF_FOLDER" --accept-low-confidence
```

Approval cannot admit a PDF that lacks the minimum identity fields or usable
text. Those records need correction or OCR before RRR can index them.

When model assistance is appropriate, rerun preparation with:

```bash
rrr prepare "PATH_TO_PDF_FOLDER" --with-llm-metadata
```

This adds model calls for unresolved OCR-readable PDFs. Pure model extraction
still requires human approval before the row enters the accepted catalogue.

## Files created by preparation

RRR stores derived material in a managed workspace under the user's RRR data
directory. The original PDFs remain where the user placed them.

| Path inside the managed workspace | Purpose |
| --- | --- |
| `metadata.csv` | Accepted bibliographic catalogue |
| `metadata.pending.csv` | Records waiting for human review |
| `ingest_report.json` | Source, confidence, and status of every metadata row |
| `data/page_text/` | Page-level extracted text |
| `indices/` | Search indices and document map |
| `runs/` | Released outputs, manifests, citations, and audit records |
| `workspace.json` | Corpus location and file snapshot used for reuse checks |

The default data locations are:

| Platform | RRR data directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\RRR` |
| macOS | `~/Library/Application Support/RRR` |
| Linux | `$XDG_DATA_HOME/rrr` or `~/.local/share/rrr` |

RRR reuses the prepared text and indices when the PDF collection is unchanged.
Adding, removing, or replacing a PDF causes the collection to be prepared
again.

## Final checklist

Before starting T1 or T2, confirm that:

- the intended PDFs are together in one selected folder;
- each PDF contains usable text or OCR;
- any available `bibliography.bib` file is beside the collection;
- pending metadata has been checked; and
- the selected runtime has the required subscription, local model, or API
  credential.

The collection is then ready for a claim evaluation or literature review.
