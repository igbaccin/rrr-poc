# Installation modes

[About RRR](../README.md) · **Installation modes** · [Before running](before_running.md)

RRR has one pipeline and several ways to run it. This guide explains what each
route installs, which model it uses, and what the user needs before starting.

## Choose a mode

| What the user wants | How RRR is started | Model used | Main requirements |
| --- | --- | --- | --- |
| The simplest Codex setup | `$rrr` in Codex | GPT through the Codex subscription | Codex, Python 3.10 or later, and OCR-readable PDFs |
| The simplest Claude setup | `/rrr` in Claude Code | Claude through the Claude subscription | Claude Code, Python 3.10 or later, and OCR-readable PDFs |
| A fully local run | `rrr` in a terminal | Mistral or Qwen through Ollama | Python, Ollama, model weights, suitable hardware, and OCR-readable PDFs |
| Codex or Claude controlling a local run | `$rrr` or `/rrr`, with a request to use Ollama | Mistral or Qwen through Ollama | The relevant skill, Python, Ollama, model weights, and OCR-readable PDFs |
| A provider API run | Terminal, Codex, or Claude | OpenAI or Anthropic API model | Python, a provider API key, paid API access, and OCR-readable PDFs |

The Codex plugin and Claude skill are ways to install and start RRR. The model
runtime determines where the language-model calls are performed.

## What the plugin or skill installs

The distributed Codex plugin and Claude skill contain:

- the RRR workflow instructions;
- a verified universal Python wheel;
- a first-use installation script; and
- the metadata needed for Codex or Claude to expose `$rrr` or `/rrr`.

On first use, the installer creates a private Python environment under the
user's RRR data directory. It installs the bundled wheel and its dependencies
there. The system Python installation is left unchanged.

The bundle includes RRR. It does not include the Python interpreter, Ollama, or
the Mistral and Qwen model weights. Every route currently requires Python 3.10
or later. The local routes also require Ollama and the selected model.

## Option 1. Codex with the Codex subscription

This is the easiest route for a Codex user. RRR uses the existing Codex login
and subscription allowance. Ollama and an OpenAI API key are unnecessary.

### Requirements

- Codex installed and signed in with a ChatGPT account
- Python 3.10 or later
- Internet access during the first installation
- PDFs containing usable text or OCR

Check the two programs from a terminal:

```bash
codex --version
python --version
```

On Windows, use `py --version` when the `python` command is unavailable.

### Install once

```bash
codex plugin marketplace add igbaccin/rrr-poc --ref main
codex plugin add rrr@rrr-poc
```

The first command registers the public RRR repository as a Codex marketplace.
The second installs its RRR plugin. Start a new Codex task after installation
so that Codex loads the `$rrr` skill.

### Start a review

Open the folder containing the PDFs in Codex and enter:

```text
$rrr t2 How did institutions shape long-run economic growth?
```

Use `t1` to evaluate a claim:

```text
$rrr t1 Institutions are the fundamental cause of long-run economic growth.
```

Codex starts the local RRR pipeline. GPT performs RRR's language-model calls
through the user's Codex subscription. An already indexed T2 review usually
requires about 35 to 40 model calls, which count against the Codex allowance.

Continue with [Before running](before_running.md) to see how RRR finds the PDF
folder and prepares its metadata catalogue.

## Option 2. Claude Code with the Claude subscription

This route uses the Claude Code login and Claude subscription allowance.
Ollama and an Anthropic API key are unnecessary.

### Requirements

- Claude Code 2.1.169 or later, signed in with a Claude subscription
- Python 3.10 or later
- Git or another way to download this repository
- Internet access during the first installation
- PDFs containing usable text or OCR

### Ask Claude Code to install RRR

Paste this message into Claude Code:

```text
Install RRR as a personal Claude Code skill from
https://github.com/igbaccin/rrr-poc.

Copy the complete skills/rrr directory to ~/.claude/skills/rrr. Preserve
SKILL.md, runtime, scripts, and agents. Run scripts/bootstrap_rrr.py --json
from the installed skill, then use the returned rrr executable to run
rrr host-doctor --host claude --json. Use my Claude subscription. Do not
configure an Anthropic API key or Ollama. Stop after confirming that /rrr is
installed and tell me whether Claude Code needs to be restarted.
```

The complete `skills/rrr/` directory is required. `SKILL.md` contains the
workflow, `runtime/` contains the verified wheel, and `scripts/` contains the
first-use installer.

The personal skill is installed under:

```text
~/.claude/skills/rrr/
```

Restart Claude Code when the top-level personal skills directory was created
during the current session.

### Start a review

Open the folder containing the PDFs in Claude Code and enter:

```text
/rrr t2 How did institutions shape long-run economic growth?
```

Claude Code starts the local RRR pipeline. Claude performs RRR's
language-model calls through the user's Claude subscription.

Continue with [Before running](before_running.md) to understand corpus and
metadata preparation.

## Option 3. Local command-line use with Ollama

This route gives the user direct control over model inference and the operating
environment.

### Requirements

- Python 3.10 or later
- Ollama
- Mistral or Qwen model weights
- Suitable local hardware or a user-controlled Ollama server
- PDFs containing usable text or OCR

Prompts written in a supported Latin-script language use
`mistral-small:24b`. Prompts written in other scripts use `qwen3:14b` by
default.

### Install RRR

```bash
python -m pip install "git+https://github.com/igbaccin/rrr-poc.git"
```

Install and start Ollama using its platform-specific installer. Download the
Mistral model with:

```bash
ollama pull mistral-small:24b
```

Download Qwen when prompts will be written in other scripts:

```bash
ollama pull qwen3:14b
```

The next step is to prepare the PDF collection. Follow the terminal procedure
under [Before running](before_running.md#prepare-the-corpus-from-a-terminal).

## Option 4. Codex or Claude with local Ollama inference

This route combines the `$rrr` or `/rrr` interface with local model inference.

1. Install the Codex plugin or Claude skill using Option 1 or Option 2.
2. Install Ollama and the required model weights using Option 3.
3. Make the local runtime explicit in the request.

In Codex:

```text
$rrr t2 How did institutions shape long-run economic growth? Use local Ollama.
```

In Claude Code:

```text
/rrr t2 How did institutions shape long-run economic growth? Use local Ollama.
```

Codex or Claude coordinates the task. RRR's internal language-model calls go
to Ollama. The surrounding Codex or Claude conversation continues to use the
normal subscription allowance for that product.

## Option 5. Provider API mode

This route is intended for automation and controlled provider comparisons. RRR
sends its language-model calls directly to the OpenAI or Anthropic API. It
requires a separate provider account, an API key, and paid API usage.

The Codex and Claude skills install the optional API dependencies when the user
explicitly requests `api openai` or `api anthropic`.

A terminal user can clone the repository and install the API extras:

```bash
git clone https://github.com/igbaccin/rrr-poc.git
cd rrr-poc
python -m pip install ".[api]"
```

The credential must be available through `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`. RRR does not accept credentials as command arguments and
does not print them in its logs.

In Codex, request the OpenAI API explicitly:

```text
$rrr t2 <topic>. Use the OpenAI API.
```

In Claude Code, request the Anthropic API explicitly:

```text
/rrr t2 <topic>. Use the Anthropic API.
```

The [before-running guide](before_running.md) applies to every runtime mode.
