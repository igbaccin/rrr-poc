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
- the label under which a review cites the document, such as
  `World Bank (1995)` or `Plange (1979a)`;
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
| Bibliography match | Matches the PDF to an entry of a nearby `bibliography.bib`, in the order given in the next section. |
| DOI lookup | For a PDF the bibliography did not settle, looks for a DOI in the PDF and queries Crossref for the corresponding bibliographic record. |
| Optional model extraction | When the user permits it, asks the selected model to read bibliographic details from the opening pages. Author and year must also appear in the PDF text. A result supported only by model extraction still requires human approval. |
| Year letters | Gives every work of a same-author, same-year group its own letter (`1979a`, `1979b`). |
| Label check | Renders a citation for every identified document and reads it back, so that a label RRR could not resolve is found before any model call. |
| Human gate | Places uncertain records in `metadata.pending.csv`, explains each of them in `ingest_report.json`, and pauses before indexing. |

The default preparation does not ask a language model to infer metadata. It
uses the bibliography sidecar and DOI lookup, then stops for unresolved
documents. Model-assisted extraction is an explicit optional step.

## How a PDF is matched to the bibliography

RRR tries four routes for each PDF and takes the first that succeeds. A matched
row records its route in the `notes` column (`bib_match:file_field`,
`bib_match:key`, `bib_match:file_name`, `bib_match:title`).

1. **The `file` field.** An entry that names the PDF in its `file` field is
   that PDF's entry, whatever the file is called. RRR reads a plain file name,
   the `description:path:type` form that JabRef writes, the form that Zotero
   exports, and several files separated by semicolons. Only the file name
   counts, without regard to letter case, so the folder recorded by the
   reference manager may differ from the folder the PDFs are in today. This is
   the most reliable route, and an export from Zotero or JabRef with file
   links provides it at no cost.
2. **The BibTeX key.** A PDF named `NorthWeingast1989.pdf` belongs to the
   entry with the key `NorthWeingast1989`.
3. **Author and year in the file name.** RRR reads the forms listed below and
   looks for the entry with that first author and year. When several entries
   fit, a second author named in the file, the year letter, and then the title
   on the first pages decide.
4. **The title.** The title of an entry stands on the first pages as one
   unbroken phrase. The earliest such title is the work's own, since a paper
   prints its title before it cites anything. A title found within the first
   150 words is accepted. A title that first appears further in may belong to
   a work the PDF cites, so RRR offers that match for confirmation and sends
   the row to `metadata.pending.csv`.

RRR never settles a doubt by taking the first candidate. A PDF that fits two
entries, an entry claimed by two PDFs, and a title found late all go to the
pending file with a note that names the entries involved and says what would
settle the matter, which is usually a `file` field or a year letter.

### File names RRR reads

| Form | Examples |
| --- | --- |
| Author and year in brackets | `Plange (1979).pdf`, `Aryeetey and Kanbur (2008).pdf`, `Frankema & van Waijenburg (2012).pdf`, `Appiah-Kubi et al. (2008).pdf`, `World Bank (1995).pdf` |
| Author, year, title (Zotero, Mendeley) | `Moradi - 2012 - Climate, height and development.pdf`, `Moradi et al. - 2013 - Heights and Development.pdf`, `Plange (1979) Underdevelopment in Northern Ghana.pdf` |
| Author then year | `Plange 1979.pdf`, `Plange-1979.pdf`, `Plange_1979.pdf`, `Plange1979.pdf` |
| Year first | `2012 - Moradi - Climate.pdf`, `2012_Moradi_Climate.pdf`, `2012 Moradi.pdf` |
| The older RRR convention | `AcemogluEtAl_2001.pdf`, `North&Weingast_1989.pdf` |

Each form accepts a year letter (`Plange (1979a).pdf`). Surnames may have
several words, hyphens, accents and particles (`de Haas`, `van der Geest`,
`Pörtner`), and corporate authors are read as one name. A file named after its
first author alone says nothing about co-authors, so `Aboagye (2021).pdf` fits
both a single-author work and a co-authored one of that year, and the title or
a `file` field then decides.

A name with no author and year in it, such as a DOI or a publisher's file
code, is matched by its title alone.

### Two works by one author in one year

RRR cites a document as `Surname (Year, p.N)`, so two works by one author in
one year need letters. Every member of such a group receives one, in the
catalogue year, in the document identifier and in the citation label. A letter
the user has already given is kept, and RRR looks for it in the bibliography
year (`year = {1979a}`), in the BibTeX key (`Plange1979a`) and in the file name
(`Plange (1979a).pdf`), in that order. The remaining works take the free
letters in file-name order, so the result is the same on every run.

`rrr prepare` lists each group in its output and in `ingest_report.json` under
`same_author_year`, with the letter of each work and where the letter came
from. Letters written into the bibliography are kept as they are, so the
user's own record decides which work is `a` and which is `b`.

### How authors are read from BibTeX

A name keeps its particle (`de Haas`, `van der Geest`, `Meier zu Selhausen`)
and its hyphen (`Aryeetey-Attoh`). Protective braces around a corporate author
(`{World Bank}`) and LaTeX accent commands (`P{\"o}rtner`) are resolved before
the name enters the catalogue. A review cites the work under these names as
the bibliography gives them. The document identifier is an ASCII form of the
same names (`Canagarajah&Portner_2003`), used for file names inside the
workspace.

The `display_label` column of `metadata.csv` holds the citation label. Editing
it changes how a work is cited.

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

RRR can match a PDF to the entry through a `file` field, the BibTeX key, an
author-and-year file name, or the title on the first pages. The section on
matching above gives the order and the accepted file names. An entry with a
`file` field is the surest way to bind a PDF to its record:

```bibtex
@article{Plange1979a,
  author  = {Plange, Nii-K.},
  title   = {Underdevelopment in Northern Ghana: Natural Causes or Colonial Capitalism?},
  journal = {Review of African Political Economy},
  year    = {1979a},
  file    = {Plange (1979a).pdf}
}
```

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

Every command also runs as `python -m rrr`, for example
`python -m rrr prepare "PATH_TO_PDF_FOLDER"`. Use that form where the `rrr`
launcher is unavailable or blocked. `rrr --help` lists the commands.

RRR prints the managed workspace path and the location of `metadata.csv`. Set
the printed workspace as `RRR_PROJECT_ROOT` before running T1 or T2.

### Exit codes of `rrr prepare`

| Code | Meaning | What to do |
| --- | --- | --- |
| `0` | The collection is prepared, or an unchanged collection was reused. Warnings may accompany the result. | Read any warning, then run T1 or T2. |
| `1` | A step failed. With `--strict`, a retained page that still holds a reference list ends preparation with this code. | Read the last lines of the output. |
| `2` | RRR could not tell which folder to use: no PDF collection was found, or several were. | Name the folder in the command. |
| `3` | Some metadata needs the user. `metadata.pending.csv` holds the rows and `ingest_report.json` gives the reason for each. | Correct the bibliography or the file names and run the command again, or approve usable rows with `--accept-low-confidence`. |

### Warnings

With `--json`, `rrr prepare` returns a `warnings` list beside the workspace
paths. Without it, the same warnings are printed on the error stream.

- `reference_list_removed` names a document and a page that still held part of
  a reference list after extraction. RRR cut the page at the list, or set the
  page aside when no text was left, and the entry quotes the first lines it
  removed. Page numbers are never shifted. `rrr prepare --strict`, or
  `RRR_STRICT_PREPARE=1`, stops on such a page.
- `shared_citation_label` names documents that are cited under one label and
  year. The review is still released, because RRR resolves each such citation
  from the evidence it was written for, but a reader cannot tell the works
  apart. Year letters in the bibliography remove the warning.

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
`ingest_report.json` records why each row was held back, and the `notes`
column of the pending file says the same in words.

| `source` of a pending row | Why it waits | What settles it |
| --- | --- | --- |
| `none` | No route identified the PDF. The note lists what was tried: how the file name was read, whether a title was found, whether the PDF has a DOI. | Add an entry or a `file` field to the bibliography, or rename the file. |
| `bib_ambiguous` | The PDF fits several bibliography entries, which the note names. | Add the year letter to the file name, or a `file` field to the right entry. |
| `bib_title_late` | The title of an entry appears in the PDF, but well into the text. The PDF is that work or a work that cites it. | Confirm with a `file` field, or approve the row. |
| `bib_repeated` | Another PDF already holds this bibliography entry. | Remove the second copy, or give the PDF its own entry. |
| `needs_ocr` | The first pages carry too little text. | Run OCR on the PDF. |
| `llm_extraction` | Only a model read the metadata. | Check the row, then approve it. |

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
