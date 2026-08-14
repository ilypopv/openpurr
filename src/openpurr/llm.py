"""LLM provider factory."""

from __future__ import annotations

from openpurr.config import Config
from openpurr.providers.base import BaseLLMProvider
from openpurr.providers.ollama import OllamaProvider


def build_provider(config: Config) -> BaseLLMProvider:
    if config.llm_provider == "ollama":
        return OllamaProvider(host=config.llm_host, model=config.llm_model)
    raise ValueError(
        f"Unsupported LLM provider: '{config.llm_provider}'. Only 'ollama' is supported."
    )
