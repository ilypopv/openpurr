# OpenPullRequest

<img src="https://raw.githubusercontent.com/ilypopv/openpurr/feat/logo/imgs/openpurr.png" alt="logo" width="180" align="left">

High-performance local developer CLI utility for automated git PR workflows.
Built against **Ollama** (local LLM engine) with `gemma4:26b` as the default
model. All source code, CLI output, prompts, and generated content are
strictly English-only.

<br clear="left" />

## Installation

```bash
uv tool install .
```

This exposes the `openpurr` command globally.

## Requirements

* Python 3.11+
* [Ollama](https://ollama.com) running locally (default: `http://localhost:11434`)
* `git` on PATH (for `devtool pr`)

## Commands

```bash
opo [--base main] [--unload]
opo review [-c / --commits N] [--unload]
```

Generates a Conventional Commits PR title + structured Markdown description
from the diff against a base branch, or a "Changes since last review"
summary from the last N commits.

## VRAM Lifecycle

Every LLM request passes a `keep_alive` value. Use `--unload` on any command
to force Ollama to immediately flush the model from VRAM after the request
completes (equivalent to `keep_alive: "0"`).
