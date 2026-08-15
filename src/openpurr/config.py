"""Configuration loader and writer for openpurr (~/.openpurr)."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".openpurr"

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "gemma4:26b",
        "api_key": "",
        "host": "http://localhost:11434",
        "temperature": 0.0,
        "keep_alive": "5m",
    },
    "pr": {
        "default_base": "main",
    },
}

# Short CLI keys → internal dotted keys (section.field)
SHORT_KEY_MAP: dict[str, str] = {
    "provider": "llm.provider",
    "model": "llm.model",
    "api_key": "llm.api_key",
    "host": "llm.host",
    "temperature": "llm.temperature",
    "keep_alive": "llm.keep_alive",
    "base": "pr.default_base",
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "provider": "LLM provider — ollama | openai | anthropic | openrouter | deepseek | llamacpp | mlx",
    "model": "Model name (e.g. gemma4:26b, gpt-4o-mini, claude-opus-4-8)",
    "api_key": "API key for cloud providers (not required for ollama / llamacpp / mlx)",
    "host": "Base URL — Ollama default is http://localhost:11434; custom endpoint for other providers",
    "temperature": "Sampling temperature: 0.0 = deterministic",
    "keep_alive": "Ollama VRAM keep-alive duration — 0s = unload immediately, 5m = keep 5 min",
    "base": "Default base branch to diff against (main, master, …)",
}

KNOWN_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openrouter": [
        "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat:free",
        "google/gemini-2.5-flash-preview",
    ],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "llamacpp": [],
    "mlx": [],
    "ollama": [],
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "llamacpp": "http://localhost:8080/v1",
    "mlx": "http://localhost:8080/v1",
}


def is_first_run() -> bool:
    return not CONFIG_PATH.exists()


def _render_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, val in values.items():
            if isinstance(val, bool):
                lines.append(f"{key} = {'true' if val else 'false'}")
            elif isinstance(val, str):
                escaped = val.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key} = "{escaped}"')
            elif isinstance(val, float) or isinstance(val, int):
                lines.append(f"{key} = {val}")
        lines.append("")
    return "\n".join(lines)


def write_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(_render_toml(data))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with CONFIG_PATH.open("rb") as f:
        return _deep_merge(DEFAULT_CONFIG, tomllib.load(f))


def _resolve_key(key: str) -> str:
    """Resolve a short key (e.g. 'model') to its dotted form ('llm.model')."""
    return SHORT_KEY_MAP.get(key, key)


def set_config_value(key: str, value: str) -> None:
    dotted_key = _resolve_key(key)
    parts = dotted_key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    section, key = parts
    data = load_config()
    if section not in data:
        data[section] = {}
    default_val = DEFAULT_CONFIG.get(section, {}).get(key)
    if isinstance(default_val, float):
        data[section][key] = float(value)
    elif isinstance(default_val, int):
        data[section][key] = int(value)
    elif isinstance(default_val, bool):
        data[section][key] = value.lower() in ("true", "1", "yes")
    else:
        data[section][key] = value
    write_config(data)


def get_config_value(key: str) -> Any:
    dotted_key = _resolve_key(key)
    parts = dotted_key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    section, field = parts
    data = load_config()
    if section not in data or field not in data[section]:
        raise KeyError(f"Config key not found: {key!r}")
    return data[section][field]


class Config:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data if data is not None else load_config()

    @property
    def llm_provider(self) -> str:
        return self._data["llm"]["provider"]

    @property
    def llm_model(self) -> str:
        return self._data["llm"]["model"]

    @property
    def llm_api_key(self) -> str:
        return self._data["llm"].get("api_key", "")

    @property
    def llm_host(self) -> str:
        return self._data["llm"].get("host", "http://localhost:11434")

    @property
    def llm_temperature(self) -> float:
        return self._data["llm"]["temperature"]

    @property
    def llm_keep_alive(self) -> str:
        return self._data["llm"]["keep_alive"]

    @property
    def pr_default_base(self) -> str:
        return self._data["pr"]["default_base"]
