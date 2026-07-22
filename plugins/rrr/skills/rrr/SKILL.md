---
name: rrr
description: Run Retrieval-Restricted Reasoning over a user-declared PDF corpus to produce a page-cited literature review or evaluate a claim. Use for corpus-bounded synthesis that must preserve source locations, refusals, and audit artifacts. Supports local Ollama, Anthropic or OpenAI provider APIs, and subscription-backed Codex or Claude Code inference.
---

# RRR literature review

Run the RRR pipeline over the user's corpus. Resolve the task and workspace,
select the requested or inferred model runtime, execute the pipeline, and
deliver its recorded output. Keep generated review text and citations
unchanged.

## Interpret the invocation

Accept natural-language requests and the compact forms used by Codex and
Claude Code.

- `$rrr t2 <topic>` in Codex or `/rrr t2 <topic>` in Claude Code runs a
  literature review.
- `$rrr t1 <claim>` in Codex or `/rrr t1 <claim>` in Claude Code evaluates a
  claim.
- `$rrr <topic>` in Codex or `/rrr <topic>` in Claude Code defaults to T2.

Treat the text after `t1` or `t2` as the complete claim or topic. These are
prompt forms, so the user does not need to add shell quotation marks. Add the
required quotation marks when translating the request into an RRR command.

## Select the runtime

An explicit `$rrr` invocation in Codex uses native Codex mode by default. Set
`RRR_RUNTIME=host` and `RRR_HOST=codex`. An explicit `/rrr` invocation in
Claude Code uses native Claude mode by default. Set `RRR_RUNTIME=host` and
`RRR_HOST=claude`. In both cases, clear `RRR_API_PROVIDER` and
`RRR_API_MODEL`.

The invocation itself selects the native route, so do not ask for a second
runtime confirmation. Before execution, briefly state which host will run the
model calls and that an already indexed T2 review usually requires about 35 to
40 calls against the product allowance. If the execution surface cannot be
identified, ask the user to choose Codex, Claude, local Ollama, or a provider
API.

Honor an explicit runtime request. `local` or `ollama` selects local mode.
`api anthropic` and `api openai` select the corresponding provider API.
`codex` or `claude` selects that native host.

- Local Ollama keeps model inference under the user's control. Clear inherited
  `RRR_RUNTIME`, `RRR_HOST`, `RRR_API_PROVIDER`, and `RRR_API_MODEL` values.
  Confirm that Ollama is reachable and that the routed Mistral or Qwen model is
  installed.
- Provider API mode uses a separate API account. Set `RRR_RUNTIME=api`, then set
  `RRR_API_PROVIDER=anthropic` or `openai`. Set `RRR_API_MODEL` only when the
  user requests an override. OpenAI overrides must use a standard GPT-5.5 or
  GPT-5.6 model. Confirm that the relevant credential exists without reading
  or displaying its value. Tell the user that stage instructions and admitted
  evidence passages go to the provider. Provider retention policies apply.
- Native Codex mode uses the Codex product login and its subscription allowance.
  Set `RRR_RUNTIME=host` and `RRR_HOST=codex`. Clear `RRR_API_PROVIDER` and
  `RRR_API_MODEL`. Run `rrr host-doctor --host codex`; it checks the executable
  and refuses API-key authentication. Use `--smoke` when an end-to-end host
  diagnostic is needed. The default host model is `gpt-5.6-sol`.
- Native Claude mode uses the Claude Code product login and its subscription
  allowance. Set `RRR_RUNTIME=host` and `RRR_HOST=claude`. Clear
  `RRR_API_PROVIDER` and `RRR_API_MODEL`. Run
  `rrr host-doctor --host claude`; it requires Claude Code 2.1.169 or newer and
  a Claude.ai Pro, Max, Team, or Enterprise subscription. Use `--smoke` when an
  end-to-end host diagnostic is needed. The default host model is `opus`.

Host mode starts a fresh tool-restricted session for each model call. An
already indexed T2 review can require roughly 35 to 40 calls. RRR verifies
subscription authentication and gives the child process a narrow environment
allowlist. The PDFs and local index remain in the workspace. Each admitted
evidence passage needed for a model call is sent to OpenAI through Codex or to
Anthropic through Claude. Host audit records are required when auditing is
enabled, so an audit write failure stops delivery.

The native default applies to the skill interface. A direct Python or `rrr`
CLI invocation with `RRR_RUNTIME` unset continues to use local Ollama.

## Resolve the RRR command

1. Use an existing `rrr` command when `rrr prepare --help` succeeds.
2. When the current project is an RRR checkout containing `pyproject.toml` and
   `src/rrr/`, install that checkout if needed. Use `python -m pip install -e
   <repo-root>` for local or host mode and add `[api]` for provider API mode.
3. Otherwise, look for `runtime/runtime.json` in this skill directory and in
   the installed plugin root two levels above it. When a bundle is present,
   tell the user that RRR is completing its first-time installation. Run the
   adjacent `scripts/bootstrap_rrr.py --json` and use the `rrr` executable
   returned in its JSON output. Add `--api` when the user selected a provider
   API.

The plugin bootstrap installs its verified bundled wheel and dependencies in
RRR's private user-data directory. Do not ask the user to install the Python
package manually when this bundle is available. Do not replace a checkout
supplied by the user with a remote package.

## Find and prepare the PDFs

Do not require the user to know RRR's directory layout or create a folder named
`corpus`.

1. Inspect the current project first. When it already contains `metadata.csv`,
   `data/page_text/`, and `indices/`, use that project as the workspace and set
   `RRR_PROJECT_ROOT` to its root.
2. When the user names or selects a folder of PDFs, use that folder directly.
3. Otherwise, run `rrr prepare --json` from the current folder. It recognizes
   PDFs in the open folder, a conventional `corpus/` subfolder, or one nearby
   folder containing PDFs.
4. If no collection is found, ask the user to select or open the folder
   containing the PDFs to analyse. If several collections are found, show the
   candidate folders and ask which one to use.
5. Run `rrr prepare "<selected-folder>" --json`. Use the returned
   `workspace_dir` as `RRR_PROJECT_ROOT` and the returned `metadata_path` for
   the task. RRR stores derived text, indices, and run artifacts in a managed
   user-data directory. It leaves the PDFs in their original folder.

The prepare command reuses a complete index while the PDF collection remains
unchanged. It also detects a nearby `bibliography.bib`. Use `--bib <file>` when
the user provides a different sidecar.

Exit status `3` means that some metadata needs human review. Show those records
and wait for the user's decision. Add `--accept-low-confidence` only after
approval. The optional `--with-llm-metadata` route can add one model call for
each unresolved OCR-usable PDF, so explain that additional usage before using
it. Stop on an extraction or reference-section invariant failure. PDFs without
usable OCR text remain outside RRR's reliable extraction path.

## Run the task

- Translate `$rrr t1 <claim>` or `/rrr t1 <claim>` to
  `rrr t1 --metadata <workspace>/metadata.csv --topic "<claim>"`.
- Translate `$rrr t2 <topic>`, `/rrr t2 <topic>`, or the form without an
  explicit task token to
  `rrr t2 --multi --metadata <workspace>/metadata.csv --topic "<topic>"`.

Reuse the metadata and index for later questions over an unchanged corpus.

## Deliver the recorded outcome

- Deliver `claim_verdict.md` verbatim after a successful T1 run.
- Deliver `review_composed.md` verbatim after a successful T2 run.
- Identify `review_ledger.json`, `citations.json`, `quality_manifest.json`, and
  `run_manifest.json` when present.
- In host mode, identify the `host_calls/` audit directory.
- Report corpus-fit warnings, refusals, and failed stages in plain language.

## Preserve integrity

- Do not edit generated review or verdict text before delivery.
- Do not set `RRR_BYPASS_VALIDATION` during ordinary use.
- Do not lower evidence or refusal thresholds to force an answer.
- Do not add citations from personal knowledge or web search.
- Do not approve uncertain metadata on the user's behalf.
- Do not expose provider credentials or include them in logs.
- Do not replace a recorded refusal with an improvised answer.
- Preserve failed-run artifacts when diagnosing an error.
