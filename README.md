# Retrieval-Restricted Reasoning (RRR)

RRR is a research pipeline for producing corpus-bounded, page-cited literature reviews with language models. It retrieves evidence from a user-supplied PDF collection, verifies quotations and page references, records the evidence admitted to generation, and refuses topics that the corpus cannot support.

The repository contains the reusable RRR implementation. The frozen materials used to reproduce the accompanying paper belong in the separate [`rrr-replication`](https://github.com/igbaccin/rrr-replication) repository.

## What RRR contributes

Retrieval-augmented generation usually retrieves passages and places them in a model's context. RRR adds a controlled evidence process suited to scholarly synthesis:

- a closed corpus defines the admissible source boundary;
- retrieval operates at page level and preserves document identity;
- evidence is filtered, deduplicated, and checked before generation;
- generated citations must resolve to admitted document-page pairs;
- corpus-fit and evidence-coverage gates can refuse unsupported requests;
- each run records manifests, evidence ledgers, citation data, and quality checks.

RRR serves a narrower purpose than general question-answering systems. Its value lies in traceability, bounded claims, and inspectable failure. Conventional RAG remains useful when broad coverage, conversational retrieval, or open-ended assistance is the primary objective.

## Pipeline

```text
PDF corpus
    -> confidence-gated metadata
    -> page extraction and reference-section exclusion
    -> BM25 page index
    -> query planning and retrieval
    -> evidence filtering and validation
    -> corpus-level outline and document posture
    -> evidence-constrained composition
    -> citation and provenance artifacts
```

RRR exposes two tasks:

- `t1` evaluates how the corpus bears on a claim and stops before review composition.
- `t2` produces a literature review from admitted evidence.

## Requirements

- Python 3.10 or newer
- a folder of text-readable PDF files
- one of the following model runtimes:
  - [Ollama](https://ollama.com/) for local inference;
  - an Anthropic or OpenAI API account for API inference.

The default local model is `mistral-small:24b` for Latin-script topics. You can select another installed Ollama model through `RRR_MODEL_LATIN`.

## Installation

```bash
git clone https://github.com/igbaccin/rrr-poc.git
cd rrr-poc
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the local package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install API support when needed:

```bash
python -m pip install -e ".[api]"
```

## Quick start with a local model

Pull a model and place your PDFs under `corpus/`:

```bash
ollama pull mistral-small:24b
mkdir corpus
```

Build the metadata catalog. Supplying a BibTeX file improves metadata confidence when one is available.

```bash
rrr ingest --corpus corpus --output metadata.csv
# With a BibTeX sidecar:
# rrr ingest --corpus corpus --output metadata.csv --bib bibliography.bib
```

The ingest gate writes uncertain records to `metadata.pending.csv` and exits with status `3`. Review those records before admitting them. The `--accept-low-confidence` option records an explicit decision to retain them.

Extract the PDFs and build the page index:

```bash
python -m rrr.preprocess --metadata metadata.csv
python -m rrr.index --metadata metadata.csv
```

Run a literature review:

```bash
rrr t2 --multi --metadata metadata.csv \
  --topic "How did institutions shape long-run economic development?"
```

Run the claim-evaluation mode:

```bash
rrr t1 --metadata metadata.csv \
  --topic "Colonial institutions reduced later economic growth."
```

## API runtime

Install the `api` extra, set the runtime and provider, and expose the provider credential through its standard environment variable.

Linux or macOS:

```bash
export RRR_RUNTIME=api
export RRR_API_PROVIDER=openai
export RRR_API_MODEL=<model-id>
export OPENAI_API_KEY=<key>
```

Windows PowerShell:

```powershell
$env:RRR_RUNTIME = "api"
$env:RRR_API_PROVIDER = "openai"
$env:RRR_API_MODEL = "<model-id>"
$env:OPENAI_API_KEY = "<key>"
```

Use `RRR_API_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` for Anthropic models. RRR does not write credential values to run manifests or metrics.

## Outputs

Each invocation receives a run directory under `runs/`. A completed `t2` run normally includes:

| Artifact | Purpose |
|---|---|
| `review_composed.md` | Final page-cited review |
| `review_ledger.json` | Evidence admitted to the writer |
| `citations.json` | Parsed citations and source provenance |
| `topic_fit.json` | Corpus-fit decision and warnings |
| `quality_manifest.json` | Deterministic output checks |
| `run_manifest.json` | Code, model, corpus, and environment provenance |
| `run_metrics.json` | Stage timings and model-call statistics |

The optional `--linkify` flag adds local links from citations to PDF pages. Generated corpus data, indices, run outputs, caches, PDFs, metadata, and credentials are excluded from Git by default.

## Configuration

The main user-facing settings are environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `RRR_MODEL_LATIN` | `mistral-small:24b` | Local model for Latin-script topics |
| `RRR_MODEL_NONLATIN` | `qwen3:14b` | Local model for non-Latin-script topics |
| `RRR_PROJECT_ROOT` | repository root | Workspace containing corpus artifacts |
| `RRR_CONCURRENCY` | `4` | Concurrent local model calls |
| `RRR_DOC_BUDGET` | `24` | Maximum admitted documents passed to later stages |
| `RRR_OLLAMA_TIMEOUT` | `600` | Per-request timeout in seconds |
| `RRR_RUNTIME` | local Ollama | Set to `api` for the API backend |
| `RRR_API_PROVIDER` | `anthropic` | API provider, `anthropic` or `openai` |
| `RRR_API_MODEL` | provider default | API model identifier |

Additional research controls remain available in the source. Validation bypasses are intended for controlled evaluation and should stay disabled during ordinary use.

## Agent skill

[`skills/rrr/SKILL.md`](skills/rrr/SKILL.md) provides an agent workflow for installing the current checkout, processing a user corpus, running RRR, and returning the audited output. The skill is available in Claude Code as `/rrr` and preserves the same refusal and validation contract as the command-line pipeline.

## Tests

```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

The selected public tests cover citation parsing, citation verification, provenance generation, and the writer's evidence contract. The paper's complete battery and reference outputs remain in the replication repository.

## Project status

RRR is research software and a proof of concept. Its audit artifacts make claims and failures inspectable, while output quality still depends on corpus composition, PDF extraction quality, model behavior, and the research question.

## License and citation

The code is released under the MIT License. See [`LICENSE.txt`](LICENSE.txt).

If you use RRR in scholarly work, please cite:

> Martins, Igor B. "Retrieval-Restricted Reasoning: A Proof of Concept for Adapting Language Models to Economic History." *Historical Methods* (forthcoming).
