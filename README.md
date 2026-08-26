# OpenPullRequest

<img src="https://raw.githubusercontent.com/ilypopv/openpurr/main/imgs/openpurr.png" alt="logo" width="180" align="left">

CLI tool that generates PR titles, descriptions, and post-review change summaries using a local or cloud LLM. Supports Ollama, OpenAI, Anthropic, Google Gemini, OpenRouter, DeepSeek, llama.cpp, and MLX.

<br clear="left" />

<br clear="left" />

## Installation

```bash
uv tool install .
```

On first run, the setup wizard creates `~/.openpurr` automatically. You can also run it any time:

```bash
opo setup
```

## Requirements

- Python 3.11+
- `git` on PATH
- One of: [Ollama](https://ollama.com) running locally, or an API key for a cloud provider

## Usage

```bash
# Generate a PR title + description from the diff against main
opo

# Diff against a different base branch
opo --base develop

# Summarise changes made since the last N commits (for reviewers)
opo review
opo review --commits 3
```

## Configuration

Settings are stored in `~/.openpurr` as flat `OPO_KEY=value` lines — no sections, no quoting:

```text
OPO_PROVIDER=ollama
OPO_MODEL=gemma4:26b-mlx
OPO_API_KEY=
OPO_HOST=http://localhost:11434
OPO_TEMPERATURE=0.0
OPO_KEEP_ALIVE=5m
OPO_BASE=main
```

Use the `config` subcommand to inspect and update values without editing the file directly.

```bash
# Show all keys with descriptions and current values
opo config describe

# Read a single value
opo config get model

# Update a value
opo config set provider openai
opo config set model gpt-4o-mini
opo config set api_key sk-...
opo config set keep_alive 0s
```

### Configuration keys

| Key | Env var | Default | Description |
| --- | ------- | ------- | ----------- |
| `provider` | `OPO_PROVIDER` | `ollama` | `ollama` · `openai` · `anthropic` · `gemini` · `openrouter` · `deepseek` · `llamacpp` · `mlx` |
| `model` | `OPO_MODEL` | _(empty)_ | Model name — set via `opo setup` or `opo config set model <name>` |
| `api_key` | `OPO_API_KEY` | _(empty)_ | API key for cloud providers |
| `host` | `OPO_HOST` | `http://localhost:11434` | Base URL — Ollama default; set to a custom endpoint when needed |
| `temperature` | `OPO_TEMPERATURE` | `0.0` | Sampling temperature (`0.0` = deterministic) |
| `keep_alive` | `OPO_KEEP_ALIVE` | `5m` | Ollama VRAM keep-alive (`0s` = unload immediately, `5m` = keep warm) |
| `base` | `OPO_BASE` | `main` | Default base branch to diff against |

There's no hardcoded default model: `opo setup` always has you pick one (from a live-fetched list, or typed manually), and `opo`/`opo review` will error out with a pointer back to `opo setup` if `model` is ever left blank (e.g. after hand-editing the file).

### Example: switch to OpenAI

```bash
opo config set provider openai
opo config set model gpt-4o-mini
opo config set api_key sk-...
```

### Example: unload Ollama from VRAM after each request

```bash
opo config set keep_alive 0s
```

## Models

List available models for the current or a specified provider:

```bash
opo models
opo models --provider anthropic
```

Model lists are always fetched live from the provider — Ollama's `/api/tags`, OpenAI/Anthropic/Gemini/OpenRouter/DeepSeek/llama.cpp/MLX's `models.list()` (or public models endpoint) — never a hardcoded/curated list baked into the tool. If the fetch fails (offline, bad key, local server not running) the command prints an empty result instead of erroring; check connectivity/credentials, or pull a model directly with `ollama pull <model>`.
