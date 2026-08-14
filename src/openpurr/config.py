"""Configuration loader for openpurr.

Reads ~/.openpurr (plain TOML, no extension). Creates it with defaults on first run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".openpurr"

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "gemma4:26b",
        "temperature": 0.0,
        "keep_alive": "0s",
        "host": "http://localhost:11434",
    },
    "pr": {
        "default_base": "main",
    },
}

_DEFAULT_TOML = """\
[llm]
provider = "ollama"
model = "gemma4:26b"
temperature = 0.0
keep_alive = "0s"
host = "http://localhost:11434"

[pr]
default_base = "main"
"""


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
        CONFIG_PATH.write_text(_DEFAULT_TOML)
    with CONFIG_PATH.open("rb") as f:
        return _deep_merge(DEFAULT_CONFIG, tomllib.load(f))


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
    def llm_temperature(self) -> float:
        return self._data["llm"]["temperature"]

    @property
    def llm_keep_alive(self) -> str:
        return self._data["llm"]["keep_alive"]

    @property
    def llm_host(self) -> str:
        return self._data["llm"]["host"]

    @property
    def pr_default_base(self) -> str:
        return self._data["pr"]["default_base"]
