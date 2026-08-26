"""Configuration loader and writer for openpurr (~/.openpurr).

Storage format is flat `OPO_KEY=value` lines (no sections, no quoting) —
mirrors opencommit's `~/.opencommit` rather than TOML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".openpurr"

DEFAULT_CONFIG: dict[str, str] = {
    "OPO_PROVIDER": "ollama",
    "OPO_MODEL": "",
    "OPO_API_KEY": "",
    "OPO_HOST": "http://localhost:11434",
    "OPO_TEMPERATURE": "0.0",
    "OPO_KEEP_ALIVE": "5m",
    "OPO_BASE": "main",
}

# Short CLI keys → OPO_* env keys
SHORT_KEY_MAP: dict[str, str] = {
    "provider": "OPO_PROVIDER",
    "model": "OPO_MODEL",
    "api_key": "OPO_API_KEY",
    "host": "OPO_HOST",
    "temperature": "OPO_TEMPERATURE",
    "keep_alive": "OPO_KEEP_ALIVE",
    "base": "OPO_BASE",
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "provider": "LLM provider — ollama | openai | anthropic | gemini | openrouter | deepseek | llamacpp | mlx",
    "model": "Model name (fetched live per-provider via `opo models`)",
    "api_key": "API key for cloud providers (not required for ollama / llamacpp / mlx)",
    "host": "Base URL — Ollama default is http://localhost:11434; custom endpoint for other providers",
    "temperature": "Sampling temperature: 0.0 = deterministic",
    "keep_alive": "Ollama VRAM keep-alive duration — 0s = unload immediately, 5m = keep 5 min",
    "base": "Default base branch to diff against (main, master, …)",
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "llamacpp": "http://localhost:8080/v1",
    "mlx": "http://localhost:8080/v1",
}


def resolve_base_url(provider: str, host: str | None) -> str | None:
    """Explicit non-default host wins, otherwise fall back to the provider's default."""
    if host and host != DEFAULT_CONFIG["OPO_HOST"]:
        return host
    return PROVIDER_BASE_URLS.get(provider)


def is_first_run() -> bool:
    return not CONFIG_PATH.exists()


def _parse_env(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return data


def _render_env(data: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in data.items()) + "\n"


def write_config(data: dict[str, str]) -> None:
    CONFIG_PATH.write_text(_render_env(data))


def load_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    return {**DEFAULT_CONFIG, **_parse_env(CONFIG_PATH.read_text())}


def _resolve_key(key: str) -> str:
    """Resolve a short key (e.g. 'model') to its OPO_* form ('OPO_MODEL')."""
    return SHORT_KEY_MAP.get(key, key)


def set_config_value(key: str, value: str) -> None:
    env_key = _resolve_key(key)
    if env_key not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    data = load_config()
    data[env_key] = value
    write_config(data)


def get_config_value(key: str) -> Any:
    env_key = _resolve_key(key)
    if env_key not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    return load_config()[env_key]


class Config:
    def __init__(self, data: dict[str, str] | None = None) -> None:
        self._data = data if data is not None else load_config()

    @property
    def llm_provider(self) -> str:
        return self._data.get("OPO_PROVIDER", DEFAULT_CONFIG["OPO_PROVIDER"])

    @property
    def llm_model(self) -> str:
        return self._data.get("OPO_MODEL", DEFAULT_CONFIG["OPO_MODEL"])

    @property
    def llm_api_key(self) -> str:
        return self._data.get("OPO_API_KEY", "")

    @property
    def llm_host(self) -> str:
        return self._data.get("OPO_HOST", DEFAULT_CONFIG["OPO_HOST"])

    @property
    def llm_temperature(self) -> float:
        return float(
            self._data.get("OPO_TEMPERATURE", DEFAULT_CONFIG["OPO_TEMPERATURE"])
        )

    @property
    def llm_keep_alive(self) -> str:
        return self._data.get("OPO_KEEP_ALIVE", DEFAULT_CONFIG["OPO_KEEP_ALIVE"])

    @property
    def pr_default_base(self) -> str:
        return self._data.get("OPO_BASE", DEFAULT_CONFIG["OPO_BASE"])
