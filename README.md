# OpenPullRequest

<img src="https://raw.githubusercontent.com/ilypopv/openpurr/main/imgs/openpurr.png" alt="logo" width="180" align="left">

CLI tool that generates PR titles, descriptions, and post-review change summaries using a local or cloud LLM. Supports Ollama, OpenAI, Anthropic, OpenRouter, DeepSeek, llama.cpp, and MLX.

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

Settings are stored in `~/.openpurr` (TOML). Use the `config` subcommand to inspect and update them without editing the file directly.

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

| Key | Default | Description |
| --- | ------- | ----------- |
| `provider` | `ollama` | `ollama` · `openai` · `anthropic` · `openrouter` · `deepseek` · `llamacpp` · `mlx` |
| `model` | `gemma4:26b` | Model name (e.g. `gpt-4o-mini`, `claude-opus-4-8`) |
| `api_key` | _(empty)_ | API key for cloud providers |
| `host` | `http://localhost:11434` | Base URL — Ollama default; set to a custom endpoint when needed |
| `temperature` | `0.0` | Sampling temperature (`0.0` = deterministic) |
| `keep_alive` | `5m` | Ollama VRAM keep-alive (`0s` = unload immediately, `5m` = keep warm) |
| `base` | `main` | Default base branch to diff against |

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

For Ollama, this queries the live `/api/tags` endpoint. For other providers it shows a curated list; pull models directly with `ollama pull <model>`.
